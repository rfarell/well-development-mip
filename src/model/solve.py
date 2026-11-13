"""Certified solver for the planning model (Section S5.6 of the Supplementary Materials of the paper):
outer approximation in factor space with Loewner-safe surrogates and explicit sparse exposure rows,
held-unit binaries for fringe terminal value, the producing base as node constants, a committed-plan mode,
fixed-first-stage solves (exact recourse for any first stage), size-matched and tilted risk-neutral MIPs,
and a perfect-information bound.
"""
import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp
import gurobipy as gp
from gurobipy import GRB

from instance import T, NB, single_node
from model import Model, PlanBuilder, MODEL_VERSION

ENV = gp.Env(params={"OutputFlag": 0, "Threads": int(os.environ.get("GRB_THREADS", 2)),
                     "MemLimit": float(os.environ.get("GRB_MEMLIMIT", 1.2)),
                     # branch-and-bound nodes spill to disk beyond this many GB, so a hard MILP hits the time limit instead of
                     # the memory limit (added 2026-10-04 after a deck-grid solve ran out of memory at 4 threads)
                     "NodefileStart": float(os.environ.get("GRB_NODEFILESTART", 0.2)),
                     # the masters mix coefficients from 1e-10 to 1e3 (tangent cuts of the variance); with the default
                     # numeric focus Gurobi once returned a dual bound that a feasible plan beat (node study, 25-node tree,
                     # 2026-10-04). NumericFocus 2 removed it; solve_ce also checks every round's bound against its incumbent.
                     "NumericFocus": int(os.environ.get("GRB_NUMERICFOCUS", 0))})
# NumericFocus 2 made the risk-neutral MILPs run out of memory at 40 units (22:10), so the default is 0 again and the defense
# against a wrong dual bound is the per-round validation in solve_ce (re-solve at NumericFocus 3, refuted bounds excluded).

# Solver revision, stamped in every run file. s1 (runs before 2026-10-05 12:00): Gurobi's default presolve and tolerances; a
# cross-solve audit then refuted 5 upper bounds, and a replay showed the default presolve returning as optimal a value that a
# feasible point of the same master beats (Section S3.4 of the Supplementary Materials). s2: masters with exposure rows are
# solved with SAFE_PARAMS, and solve_ce checks every round's bound against the master's own value at each known plan.
SOLVER_REV = "s2"
SAFE_PARAMS = dict(DualReductions=0, FeasibilityTol=1e-9, OptimalityTol=1e-9, IntFeasTol=1e-9)


def loewner_factor(Sigma, rank=None, tol=1e-10):
    """F with F F^T = Sigma_r <= Sigma (eigen-truncation keeps the top eigenpairs: Loewner-safe)."""
    w, V = np.linalg.eigh(Sigma)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    keep = w > tol * w[0]
    if rank is not None:
        keep[rank:] = False
    return V[:, keep] * np.sqrt(w[keep])


class Master:
    """Gurobi model of the constraints of section 5, with per-node exposure rows (quad in {False, 'soc', 'cuts'})."""
    def __init__(self, M: Model, quad="cuts", rank=None, time_limit=600, mip_gap=2e-4, committed=False, fixed=None,
                 max_expected_wells=None):
        self.M = M
        I, U, K, G = M.I, M.U, M.K, M.G
        m = gp.Model(env=ENV)
        m.Params.MIPGap, m.Params.TimeLimit = mip_gap, time_limit
        if quad and os.environ.get("GRB_SAFE", "1") != "0":                  # exposure rows: coefficients from 1e-11 to 1e2
            for name_, val_ in SAFE_PARAMS.items():
                m.setParam(name_, val_)
        z0 = m.addMVar(M.n0, vtype=GRB.BINARY)
        zk = m.addMVar((K, M.n1), vtype=GRB.BINARY)
        y = m.addMVar(G, vtype=GRB.BINARY)
        w0 = m.addMVar(U, vtype=GRB.BINARY)
        wk = m.addMVar((K, U, T - 1), vtype=GRB.BINARY)
        h = m.addMVar((K, U), vtype=GRB.BINARY)                             # fringe unit held by a well by its deadline
        iota = m.addMVar((K, M.nub), lb=0.0, ub=1.0)                         # credited fraction of terminal inventory
        if committed:
            for k in range(1, K):
                m.addConstr(zk[k] == zk[0]); m.addConstr(wk[k] == wk[0])
        if fixed is not None:                                                # fix the first stage of a given plan
            z0.LB = fixed["z0"]; z0.UB = fixed["z0"]; y.LB = fixed["y"]; y.UB = fixed["y"]; w0.LB = fixed["w0"]; w0.UB = fixed["w0"]
        A0 = {}; A1 = {}
        for name, Mat in (("ub", M.Mub), ("lanes", M.Lanes), ("ubt", M.Mubt), ("holdu", M.HoldU)):
            A0[name], A1[name] = Mat[:, M.idx0], Mat[:, M.idx1]
        nonown = ~M.ub_owner_ok
        Pk = sp.csr_matrix((np.ones(nonown.sum()), (np.flatnonzero(nonown), M.pkg[nonown])), shape=(M.n, G))
        own0, own1 = nonown[M.idx0], nonown[M.idx1]
        if own0.any():
            m.addConstr(z0[np.flatnonzero(own0)] <= Pk[M.idx0][own0] @ y)
        nb_sel = M.nbr_pkg >= 0
        Pn = sp.csr_matrix((np.ones(nb_sel.sum()), (np.flatnonzero(nb_sel), M.nbr_pkg[nb_sel])), shape=(M.n, G))
        nb0, nb1 = nb_sel[M.idx0], nb_sel[M.idx1]
        if nb0.any():
            m.addConstr(z0[np.flatnonzero(nb0)] <= Pn[M.idx0][nb0] @ y)
        # pad precedence per unit-bench-year: cumulative pads of the unit
        r_idx, c_idx = [], []
        for u in range(U):
            for t in range(T):
                for tt in range(t + 1):
                    r_idx.append(u * T + t); c_idx.append(u * T + tt)
        Cum = sp.csr_matrix((np.ones(len(r_idx)), (r_idx, c_idx)), shape=(U * T, U * T))
        Cum0, Cum1 = Cum[:, np.arange(U) * T], Cum[:, np.setdiff1d(np.arange(U * T), np.arange(U) * T)]
        rows_ubt = np.arange(M.nub * T)
        Rep = sp.csr_matrix((np.ones(M.nub * T), (rows_ubt, (rows_ubt // (NB * T)) * T + rows_ubt % T)), shape=(M.nub * T, U * T))
        Cum0, Cum1 = Rep @ Cum0, Rep @ Cum1
        # year-0 capacity rows (shared)
        m.addConstr(M.rigdays[M.idx0] @ z0 <= I["cap"]["rig"][0])
        m.addConstr(M.lat_miles[M.idx0] @ z0 <= I["cap"]["crew"][0])
        m.addConstr(M.C[M.idx0] @ z0 + M.pad_cost[:, 0] @ w0 <= I["cap"]["capital"][0])
        if G:
            m.addConstr(I["pkg_cost"] @ y <= I["cap"]["acq"])
        if max_expected_wells is not None:                                    # size-matched risk-neutral baseline
            p = I["prob"]
            m.addConstr(M.wells[M.idx0] @ z0 + gp.quicksum(float(p[k]) * (M.wells[M.idx1] @ zk[k]) for k in range(K)) <= max_expected_wells)
        self.s = self.g = self.wv = None
        if quad:
            F = loewner_factor(M.Sigma, rank)
            self.F, r = F, F.shape[1]
            s = m.addMVar(K, lb=0.0)
            g = m.addMVar((K, r), lb=-GRB.INFINITY)
            wv = m.addMVar((K, M.nub), lb=-GRB.INFINITY)
        own_const = np.repeat((I["owned"] & ~M.fringe_u).astype(float), NB)            # kind 0: held beyond the horizon
        kind1_u = np.flatnonzero(I["owned"] & M.fringe_u)                               # fringe units: own = h[k,u]
        pk_ub = np.repeat(I["pkg_of"], NB); sel_ub = pk_ub >= 0
        self.Ty = sp.csr_matrix((np.ones(sel_ub.sum()), (np.flatnonzero(sel_ub), pk_ub[sel_ub])), shape=(M.nub, G))
        self.Th = sp.csr_matrix((np.ones(len(kind1_u) * NB), (np.repeat(kind1_u, NB) * NB + np.tile(np.arange(NB), len(kind1_u)),
                                                              np.repeat(kind1_u, NB))), shape=(M.nub, U))
        self.own_const = own_const
        hu_r, hu_c = M.HoldU.nonzero()                                          # (unit, hold option) pairs of fringe units
        if len(hu_r):
            nnz = len(hu_r)
            Sh = sp.csr_matrix((np.ones(nnz), (np.arange(nnz), hu_r)), shape=(nnz, U))
            Eo = sp.csr_matrix((np.ones(nnz), (np.arange(nnz), hu_c)), shape=(nnz, M.n))
            E0, E1 = Eo[:, M.idx0], Eo[:, M.idx1]
        else:
            Sh = None
        for k in range(K):
            zz1 = zk[k]
            m.addConstr(A0["ub"] @ z0 + A1["ub"] @ zz1 <= 1)
            m.addConstr(A0["lanes"] @ z0 + A1["lanes"] @ zz1 <= M.lane_cap)
            if own1.any():
                m.addConstr(zz1[np.flatnonzero(own1)] <= Pk[M.idx1][own1] @ y)
            if nb1.any():
                m.addConstr(zz1[np.flatnonzero(nb1)] <= Pn[M.idx1][nb1] @ y)
            wkf = wk[k].reshape(-1)
            m.addConstr(A0["ubt"] @ z0 + A1["ubt"] @ zz1 <= Cum0 @ w0 + Cum1 @ wkf)
            m.addConstr(w0 + wk[k].sum(axis=1) <= 1)
            if len(M.post_rows):
                m.addConstr(M.Hold[:, M.idx0] @ z0 + M.Hold[:, M.idx1] @ zz1 <= 0)
            if I.get("codevelop", False):                                        # v3.3 variant: bench 0 in year t and bench 1 in year t' != t excluded
                for u in range(U):
                    for t in range(T):
                        for tt in range(T):
                            if t == tt:
                                continue
                            a = np.flatnonzero((M.o_u == u) & (M.o_b == 0) & (M.o_t == t)); b = np.flatnonzero((M.o_u == u) & (M.o_b == 1) & (M.o_t == tt))
                            if len(a) == 0 or len(b) == 0:
                                continue
                            ex = gp.LinExpr()
                            for o in np.r_[a, b]:
                                ex += z0[M.pos0[o]] if M.o_t[o] == 0 else zz1[M.pos1[o]]
                            m.addConstr(ex <= 1)
            m.addConstr(h[k] <= A0["holdu"] @ z0 + A1["holdu"] @ zz1)        # held only if a well by the deadline
            # inventory credit: iota <= own - dev and iota <= 1 - lanes consumed by neighbors' long laterals
            own_expr = (self.Ty @ y if G else 0) + self.Th @ h[k] + own_const
            dev_expr = A0["ub"] @ z0 + A1["ub"] @ zz1
            cons_expr = M.Cons[:, M.idx0] @ z0 + M.Cons[:, M.idx1] @ zz1
            m.addConstr(iota[k] <= own_expr - dev_expr)
            m.addConstr(iota[k] <= 1 - cons_expr)
            # v3.2: McCormick lower bound makes the credit fraction exact, iota = min(own - dev, 1 - consumed),
            # because own - dev is binary (dev <= own) and 1 - consumed lies in [0, 1]
            m.addConstr(iota[k] >= own_expr - dev_expr - cons_expr)
            if Sh is not None:                                               # ... and held whenever such a well exists (h = max z_o)
                m.addConstr(Sh @ h[k] - E0 @ z0 - E1 @ zz1 >= 0)
            for t in range(1, T):
                sel = M.o_t[M.idx1] == t
                m.addConstr((M.rigdays[M.idx1] * sel) @ zz1 <= I["cap"]["rig"][t])
                m.addConstr((M.lat_miles[M.idx1] * sel) @ zz1 <= I["cap"]["crew"][t])
                m.addConstr((M.C[M.idx1] * sel) @ zz1 + M.pad_cost[:, t] @ wk[k][:, t - 1] <= I["cap"]["capital"][t])
            if quad:
                Ex = M.Wexp @ sp.diags(M.scale[:, k])                            # option exposures in node k
                Tk = sp.csr_matrix(M.Texp[k])
                m.addConstr(wv[k] == Ex[:, M.idx0] @ z0 + Ex[:, M.idx1] @ zz1 + Tk @ iota[k] + M.b_k[:, k])
                m.addConstr(g[k] == F.T @ wv[k])
                if quad == "soc":
                    m.addConstr(g[k] @ g[k] <= s[k])
        m.update()
        self.m, self.z0, self.zk, self.y, self.w0, self.wk, self.h, self.iota = m, z0, zk, y, w0, wk, h, iota
        if quad:
            self.s, self.g, self.wv = s, g, wv
        self.quad = quad

    def D_expr(self, k):
        """Linear expression of D_k (deterministic NPV in node k, incl. the producing base) in the master's variables."""
        M, I = self.M, self.M.I
        c_opt = M.revk[:, k] - M.C
        return (c_opt[M.idx0] @ self.z0 + c_opt[M.idx1] @ self.zk[k] - (I["pkg_cost"] @ self.y if M.G else 0) + M.tau[:, k] @ self.iota[k]
                - M.pad_cost[:, 0] @ self.w0 - M.pad_cost[:, 1:].reshape(-1) @ self.wk[k].reshape(-1) + float(M.B_k[k]))

    def D_value(self, plan, k):
        """The same linear function evaluated at a plan (self-tests), with the evaluator's inventory credit."""
        M, I = self.M, self.M.I
        z = M.full_z(plan, k)
        inv = M.inventory(plan, z)
        return ((M.revk[:, k] - M.C) @ z - I["pkg_cost"] @ plan["y"] + M.tau[:, k] @ inv
                - M.pad_cost[:, 0] @ plan["w0"] - (M.pad_cost[:, 1:] * plan["wk"][k]).sum() + float(M.B_k[k]))

    def fix_inventory(self, plan):
        """Pin the master's credit fractions to the evaluator's values (self-tests only)."""
        for k in range(self.M.K):
            inv = self.M.inventory(plan, self.M.full_z(plan, k))
            self.iota[k].LB = inv; self.iota[k].UB = inv

    def add_quad_cuts(self, plan):
        """Tangent cuts s_k >= 2 g^.g_k - g^.g^ at the plan's exposures (under-estimators of ||g_k||^2)."""
        M = self.M
        for k in range(M.K):
            _, w, _, _ = M.node_terms(plan, k)
            gh = self.F.T @ w
            gh = np.where(np.abs(gh) < 1e-9 * max(1.0, np.abs(gh).max()), 0.0, gh)   # drop negligible coefficients: the cut
            if np.any(gh != 0):                                                        # is a valid tangent at any point
                self.m.addConstr(self.s[k] >= 2 * (gh @ self.g[k]) - float(gh @ gh))

    def extract(self):
        return dict(z0=(np.array(self.z0.X) > .5).astype(float), zk=(np.array(self.zk.X) > .5).astype(float),
                    y=(np.array(self.y.X) > .5).astype(float), w0=(np.array(self.w0.X) > .5).astype(float),
                    wk=(np.array(self.wk.X) > .5).astype(float))

    def value_at(self, plan):
        """The master's objective with every binary fixed at the plan, or None if the master has no such point. A valid dual
        bound of the master (a minimization) is at most this value, whatever cuts the master holds."""
        m = self.m
        groups = [(v, np.asarray(plan[n], float)) for v, n in ((self.z0, "z0"), (self.zk, "zk"), (self.y, "y"), (self.w0, "w0"), (self.wk, "wk"))
                  if getattr(v, "size", 0)]
        saved = [(v.LB.copy(), v.UB.copy()) for v, _ in groups]
        try:
            for v, x in groups:
                x = x.reshape(v.shape)
                v.LB = x; v.UB = x
            m.optimize()
            return float(m.ObjVal) if m.Status == GRB.OPTIMAL else None
        except gp.GurobiError:
            return None
        finally:
            for (v, _), (lb_, ub_) in zip(groups, saved):
                v.LB = lb_; v.UB = ub_
            m.update()

    def set_start(self, plan):
        self.z0.Start, self.zk.Start, self.y.Start, self.w0.Start, self.wk.Start = plan["z0"], plan["zk"], plan["y"], plan["w0"], plan["wk"]


def solve_risk_neutral(M: Model, time_limit=600, mip_gap=1e-4, relax=False, committed=False, fixed=None, weights=None,
                       max_expected_wells=None):
    """Two-stage expected-NPV MILP; `weights` replaces the node probabilities (tilted risk-neutral MIP);
    `max_expected_wells` adds the size-matching row; `fixed` fixes the first stage of a plan."""
    mm = Master(M, quad=False, time_limit=time_limit, mip_gap=mip_gap, committed=committed, fixed=fixed, max_expected_wells=max_expected_wells)
    p = M.I["prob"] if weights is None else np.asarray(weights)
    mm.m.setObjective(gp.quicksum(float(p[k]) * mm.D_expr(k) for k in range(M.K)), GRB.MAXIMIZE)
    t0 = time.time()
    if relax:
        mm.m.update()
        rel = mm.m.relax(); rel.optimize()
        return dict(obj=rel.ObjVal, x=np.array([v.X for v in rel.getVars()]), time=time.time() - t0)
    mm.m.optimize()
    return dict(plan=mm.extract(), obj=mm.m.ObjVal, bound=mm.m.ObjBound, gap=mm.m.MIPGap, time=time.time() - t0)


def solve_ce(M: Model, seeds=(), tol=1e-3, max_iter=30, rank=None, time_limit=600, mip_gap=2e-4, log=print, master="cuts",
             committed=False, fixed=None):
    """Certified CE maximization: OA on log-sum-exp over nodes, exposure rows with a Loewner-safe factor.
    Returns the incumbent, the history of [CE_LB, CE_UB] and the final certificate (running-minimum upper bound)."""
    lam, K, p = M.lam, M.K, M.I["prob"]
    mm = Master(M, quad=master, rank=rank, time_limit=time_limit, mip_gap=mip_gap, committed=committed, fixed=fixed)
    m = mm.m
    th = m.addVar(lb=-GRB.INFINITY, name="theta")
    m.setObjective(th, GRB.MINIMIZE)
    Dex = [mm.D_expr(k) for k in range(K)]
    cg = [M.cgf_k[M.idx0, k] @ mm.z0 + M.cgf_k[M.idx1, k] @ mm.zk[k] for k in range(K)]

    def add_cut(qhat):
        phi = logsumexp(qhat, b=p)
        pi = p * np.exp(qhat - phi)
        m.addConstr(th >= float(phi - pi @ qhat) + gp.quicksum(float(pi[k]) * (-lam * Dex[k] + 0.5 * lam**2 * mm.s[k] + cg[k]) for k in range(K)))

    add_cut(np.zeros(K))
    best, best_plan, hist, best_ub, seen = np.inf, None, [], np.inf, set()
    known, known_keys = [], set()                                        # feasible plans of this problem: witnesses for the bound check

    def remember(pl_):
        key_ = (pl_["z0"].tobytes(), pl_["zk"].tobytes(), pl_["y"].tobytes(), pl_["w0"].tobytes(), pl_["wk"].tobytes())
        if key_ not in known_keys:
            known_keys.add(key_); known.append(pl_)

    def master_floor():
        """The largest value (CE units) the current master gives a known plan: no valid upper bound can lie below it."""
        vals = [v for v in (mm.value_at(pl_) for pl_ in known) if v is not None]
        return max(-v / lam for v in vals) if vals else -np.inf

    for pl in seeds:
        if pl is None or not M.feasible(pl):
            continue
        if fixed is not None and not all(np.array_equal(pl[k_], fixed[k_]) for k_ in ("z0", "y", "w0")):
            continue                                                      # a seed of a fixed-first-stage solve must share that first stage
        if committed and not (all(np.array_equal(pl["zk"][k_], pl["zk"][0]) for k_ in range(K)) and all(np.array_equal(pl["wk"][k_], pl["wk"][0]) for k_ in range(K))):
            continue                                                      # a seed of a committed solve must be a committed plan
        remember(pl)
        q = M.evaluate(pl)["q"]; add_cut(q)
        if master == "cuts":
            mm.add_quad_cuts(pl)
        phi = logsumexp(q, b=p)
        if phi < best:
            best, best_plan = phi, pl
    t_start = time.time()
    numeric_resolves, bound_valid, disproved = 0, True, []
    witness_checks, witness_refuted = 0, 0
    round_bounds = []                                                     # every round's dual bound (CE units) that is still unrefuted
    tol_abs = lambda lb_: 1e-7 * max(1.0, abs(lb_))                       # noqa: E731
    for it in range(max_iter):
        if best_plan is not None:
            mm.set_start(best_plan)
        t0 = time.time(); m.optimize()
        if m.SolCount == 0:
            log(f"    OA it{it}: no solution (status {m.Status})"); break
        pl = mm.extract()
        assert M.feasible(pl, True), "master incumbent violates Model.feasible: certificate would be invalid"
        q = M.evaluate(pl)["q"]; phi = logsumexp(q, b=p)
        if phi < best:                                                    # keep every incumbent before any re-solve
            best, best_plan = phi, pl
        lb = -best / lam
        round_ub = -m.ObjBound / lam
        # a dual bound below an exact incumbent is numerically wrong: re-solve at the strictest numeric focus; if it is still
        # below, the bound is left out of the certificate and the solve is marked
        if round_ub < lb - tol_abs(lb):
            log(f"    OA it{it}: bound {round_ub:.4f} below incumbent {lb:.4f}; re-solving with NumericFocus 3")
            m.Params.NumericFocus = 3; numeric_resolves += 1
            m.optimize()
            if m.SolCount:
                pl = mm.extract(); assert M.feasible(pl, True)
                q = M.evaluate(pl)["q"]; phi = logsumexp(q, b=p)
                if phi < best:
                    best, best_plan = phi, pl
                lb = -best / lam
            round_ub = -m.ObjBound / lam
            if round_ub < lb - tol_abs(lb):
                disproved.append(round_ub); bound_valid = False; round_ub = np.inf
        gap_attr = m.MIPGap
        remember(pl)
        # the bound must also cover the master's own value at every known plan (seeds and incumbents): a presolve that removes
        # feasible points returns a bound that agrees with its incumbent and fails here. Re-solve without presolve; if the bound
        # still fails it is left out of the certificate and the solve is marked
        if np.isfinite(round_ub):
            floor = master_floor(); witness_checks += 1
            if round_ub < floor - tol_abs(floor):
                log(f"    OA it{it}: bound {round_ub:.4f} below the master's value {floor:.4f} at a known plan; re-solving without presolve")
                m.Params.Presolve = 0; numeric_resolves += 1
                mm.set_start(best_plan); m.optimize()
                if m.SolCount:
                    pl = mm.extract(); assert M.feasible(pl, True)
                    q = M.evaluate(pl)["q"]; phi = logsumexp(q, b=p)
                    if phi < best:
                        best, best_plan = phi, pl
                    lb = -best / lam
                    remember(pl)
                round_ub = -m.ObjBound / lam; gap_attr = m.MIPGap
                floor = master_floor()
                if round_ub < floor - tol_abs(floor) or round_ub < lb - tol_abs(lb):
                    disproved.append(round_ub); bound_valid = False; witness_refuted += 1; round_ub = np.inf
        round_bounds.append(round_ub)
        # a new incumbent also refutes every earlier bound below it: drop those and recompute the running minimum
        kept = [b for b in round_bounds if b >= lb - tol_abs(lb)]
        if len(kept) < len(round_bounds):
            disproved += [b for b in round_bounds if b < lb - tol_abs(lb)]; bound_valid = False
            round_bounds = kept
        best_ub = min(round_bounds) if round_bounds else np.inf
        hist.append(dict(it=it, ce_lb=lb, ce_ub=best_ub, time=time.time() - t0, master_gap=gap_attr, round_bound=round_ub,
                         numeric_focus=int(m.Params.NumericFocus)))
        log(f"    OA it{it}: CE in [{lb:.2f}, {best_ub:.2f}]  wells(yr0)={int(pl['z0'] @ M.wells[M.idx0])}  t={time.time()-t0:.1f}s  master gap {gap_attr:.1e}")
        key = (pl["z0"].tobytes(), pl["zk"].tobytes(), pl["y"].tobytes())
        if (np.isfinite(best_ub) and best_ub - lb <= tol * max(1.0, abs(lb))) or key in seen:
            break
        seen.add(key)
        add_cut(q)
        if master == "cuts":
            mm.add_quad_cuts(pl)
    return dict(plan=best_plan, hist=hist, ce_lb=-best / lam, ce_ub=best_ub, time=time.time() - t_start,
                rounds=len(hist), rank=mm.F.shape[1], master=master, numeric_resolves=numeric_resolves, bound_valid=bound_valid,
                disproved_bounds=disproved, model_version=MODEL_VERSION, solver_rev=SOLVER_REV, witness_checks=witness_checks,
                witness_refuted=witness_refuted)


def solve_recourse_exact(M: Model, plan, extra_seeds=(), **kw):
    """Certified CE-optimal recourse for a given first stage (z0, y, w0 fixed): the fair way to score any
    method's committed decisions, and the basis of the fine-tree out-of-sample test. extra_seeds: further plans with the
    same first stage (others are ignored), for example the result of an earlier run."""
    return solve_ce(M, seeds=(plan,) + tuple(extra_seeds), fixed=plan, log=kw.pop("log", lambda *a: None), **kw)


def perfect_information(M: Model, **kw):
    """CE under perfect information about the factors: solve each node alone (first stage free), then
    CE_PI = -(1/lam) log sum_k p_k exp(-lam CE_k). An upper bound on any adaptive policy's CE for this tree."""
    lam, p = M.lam, M.I["prob"]
    ces, ubs, valid = [], [], []
    for k in range(M.K):
        Mk = Model(single_node(M.I, k), lam, deck=M.deck)
        # use the tree model's node-k constants (inventory credit, terminal exposure, producing base) so the
        # single-node problem is exactly node k of the tree problem with the first stage free
        Mk.tau, Mk.tau_best, Mk.Texp = M.tau[:, [k]].copy(), M.tau_best.copy(), M.Texp[[k]].copy()
        Mk.B_k, Mk.b_k = M.B_k[[k]].copy(), M.b_k[:, [k]].copy()
        Mk.Cons = M.Cons.copy()
        r = solve_ce(Mk, log=lambda *a: None, **kw)
        ces.append(r["ce_lb"]); ubs.append(r["ce_ub"]); valid.append(r["bound_valid"])
    ces, ubs = np.array(ces), np.array(ubs)
    return dict(ce_pi_lb=float(-logsumexp(-lam * ces, b=p) / lam), ce_pi_ub=float(-logsumexp(-lam * ubs, b=p) / lam), node_ce=ces.tolist(),
                node_bound_valid=valid, bound_valid=all(valid))


def relax_and_round(M: Model, lp=None):
    """Round the risk-neutral LP relaxation greedily (by fractional value) under the real constraints."""
    if lp is None:
        lp = solve_risk_neutral(M, relax=True)
    x = lp["x"]
    n0, K, n1 = M.n0, M.K, M.n1
    x0 = x[:n0]; xk = x[n0:n0 + K * n1].reshape(K, n1); y = x[n0 + K * n1:n0 + K * n1 + M.G]
    B = PlanBuilder(M)
    for g in np.argsort(-y):
        if y[g] > 0.5 and B.can_buy(g):
            B.buy(g)
    for o in M.idx0[np.argsort(-x0)]:
        if x0[M.pos0[o]] > 1e-3 and B.can_add(o):
            B.add(o)
    for k in range(K):
        for o in M.idx1[np.argsort(-xk[k])]:
            if xk[k, M.pos1[o]] > 1e-3 and B.can_add(o, k):
                B.add(o, k)
    return B.plan
