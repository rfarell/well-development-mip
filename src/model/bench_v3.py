"""Full benchmark harness for the planning model, v3.2.

It includes a fair metaheuristic and seeding protocol, the deck family and the tilted risk-neutral MIP, reporting in
planner units, a size-matched risk-neutral plan and the first-stage/recourse split, the information ladder, the
fine-tree out-of-sample test, perturbed-truth paired scoring, rig utilization, per-stage robustness with saved plans,
re-planning comparators with a clean VSS, and first-stage robustness under perturbed truths. The exact
credit-fraction row is in solve.py and the conventions study in exp_conventions.py.

Every method is run on the same instances and scored on the same exact objective (Model.evaluate), with its
wall-clock. Rows per (instance, rho):

  heur_ev, heur_ce        rank-and-fill with recourse, settings grid-tuned by E[NPV] / by CE (the CE one is oracle-tuned)
  rn                      risk-neutral two-stage MIP at the strip (proven to the MIP gap); UB_N is its dual bound
  rn_lp_round             greedy rounding of the risk-neutral LP relaxation ("RN-LP rounding"; not the CE master's LP)
  rn_deck                 risk-neutral MIP at the oracle-selected planning deck (best CE under the true model) on the grid
                          0.50-1.10 by 0.025 (1.0 is the rn plan itself, so rn_deck >= rn by construction); the whole deck
                          curve is stored
  rn_tilt1, rn_tilt2      pi-tilted risk-neutral MIP after one and two fixed-point iterations from the rn plan (no oracle)
  rn_tilt                 pi-tilted risk-neutral MIP at the CE plan's softmax node weights (the linear part of the final OA
                          cut, without its exposure and CGF terms; an oracle baseline)
  rn_size                 risk-neutral MIP with expected wells <= the CE plan's (size-matched: composition against size;
                          the size is taken from the CE plan, an oracle)
  pso, ga                 binary PSO / GA on CE with the tuned recourse rule (fill = heur_ce's config), best of --pso_seeds
                          seeds at max(--budget, ce_cold's wall-clock) seconds each (equal or larger wall-clock than
                          the seed-free certified solve); every run's best-vs-time curve is stored
  pso_long                one PSO at --long_mult (default 5) times that budget
  pso_ev, ga_ev           PSO / GA on E[NPV] with heur_ev's recourse rule (the fair rows for the E[NPV] comparison)
  ce_cold                 certified CE solver from scratch: its wall-clock is the fair solver time
  ce                      certified CE solver seeded with rn, rn_lp_round and heur_ce; its time includes the seeds' time
  ce_1round               one OA round from the rn seed with a 30 s master (time includes rn's time)
  ce_gauss                certified plan under the seed paper's Gaussian per-well treatment, scored under the exact model
  X+exact_recourse        first stage (z0, y, w0) of X fixed, recourse re-optimized to a certificate, for X in heur_ce,
                          pso, rn_deck and rn
  mv+exact_recourse       the deterministic mean-value first stage (theta = P = 1, solved risk-neutrally) with certified
                          CE recourse: the EEV rung of v3.1, which mixes the stochastic model with the risk model
  mv_rn                   (v3.2) the same mean-value first stage with risk-neutral recourse on the tree,
                          solve_risk_neutral(M, fixed=mv): the rolling deterministic re-solve (each node re-plans years 1-3
                          at its own factors), scored on CE and E[NPV]
  mv_ce+exact_recourse    (v3.2) the CE mean-value first stage (solve_ce on the mean-value model at the same lambda,
                          seeded with the risk-neutral mean-value plan) with certified CE recourse on the tree
  ce_fs+rn_recourse       the CE plan's first stage with risk-neutral recourse
  ce_committed            certified CE plan with one recourse shared by all nodes (the committed rung of the ladder)

The mean-value problem is one node at theta = P = 1, built from a copy of the instance (mean_value_model), so its
credited set, tau, Texp, Cons and base are those of theta = P = 1, as a deterministic planner would build them.

Per run it also records the screen UB_N - CE(x_N) (an upper bound on any CE gain over the rn plan), Beck's statistic
(E[NPV] cost beside the CE gain, in $ and % of E[NPV](rn)), the information ladder (committed <= two-stage <= perfect
information, with the value of recourse, EVPI and VSS; the perfect-information rung is solved per node on a model that
carries the full model's terminal credit for that node and is seeded with the two-stage plan, so EVPI >= 0 exactly;
v3.2 adds every rung difference as a certificate interval [LB_a - UB_b, UB_a - LB_b] (keys *_lb, *_ub) and the two
clean VSS: RN-VSS = E[NPV](rn) - E[NPV](mv_rn), bracketed by the two MIPs' dual bounds, and CE-VSS = CE(ce) -
CE(mv_ce+exact_recourse), bracketed by the two certificates; the v3.1 keys eev and vss keep the mixed number, which is
also stored as vss_mixed), the fine-tree out-of-sample test (every method's first stage with certified recourse on a
--fine x --fine node tree, the headline first-stage comparison; v3.2 adds risk-neutral recourse for the first stages
of rn, mv_rn and ce_fs+rn_recourse and the fine-tree RN-VSS, CE-VSS and mixed VSS in fine['vss']), the perturbed-truth
table (every plan re-scored under ten perturbations of the generator's truth, recourse as planned) and (v3.2,
--perturb_reopt) the re-optimized perturbed-truth table perturbed_reopt: the first stages of --perturb_methods fixed
and the recourse re-optimized under each truth, risk-neutral (scored on E[NPV]) and certified (scored on CE, with
OA_REOPT), with z_A (the ce row) against the best member of the deck family {rn_deck, rn_tilt1, rn_tilt2} on both
scores (see perturbed_reopt).

Robustness and bookkeeping (v3.2): every stage runs in its own try/except. A failure is recorded in run['errors']
(stage, exception, traceback tail) and the run continues; a stage whose inputs are missing is skipped. After every
stage the JSON is rewritten atomically (temporary file, fsync, rename) with the stage's record in run['stages'] (ok,
wall-clock, the process's peak RSS so far, the rise of the peak during the stage, current RSS), and the memory guard
runs. The fine-tree and re-optimized perturbed-truth stages also save after every solve (after every truth, and a
failing first stage under one truth drops only that first stage's rows). A run is in res['runs'] from its first stage,
with run['status']: running, done, done with N stage error(s), "stopping: peak RSS ... (after <stage>)" (the guard), or
failed (only when the model cannot be built; also listed in res['errors'] as in v3.1). Every method's plan is saved in
run['plans'] as sparse index lists (sparse_plan; dense_plan inverts it; run['plan_format'] records the dimensions and a
SHA-1 of the option table).

Usage: bench_v3.py nx ny seed_from seed_to rho_list out.json [--budget 60] [--fine 5] [--pso_seeds 2] [--ev_seeds 1]
       [--long_mult 5] [--perturb_reopt | --no-perturb_reopt] [--perturb_methods ce,rn,rn_deck,rn_tilt2,heur_ce,ga]
Environment: GRB_THREADS (Gurobi threads, default 2), GRB_MEMLIMIT (GB, default 1.2), OMP_NUM_THREADS (numpy),
BENCH_RSS_CAP_GB (memory guard, default 1.9); all are recorded in the output. The memory guard stops the process when
the peak RSS exceeds the cap, after saving the run with its status. It reads the process's peak, so a single-value
rho_list (one process per (seed, rho)) makes it a per-run guard.
"""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
import re
import resource
import sys
import time
import traceback

import numpy as np
from scipy.special import logsumexp

from instance import make_instance, perturb, single_node, NB, T
from model import Model, MODEL_VERSION
from solve import solve_ce, solve_risk_neutral, solve_recourse_exact, relax_and_round, SOLVER_REV
import baselines as BL

try:                                                                   # current RSS per stage (optional)
    import psutil
    _PROC = psutil.Process()
except Exception:
    _PROC = None

VERSION = "v3.3"
RSS_CAP_GB = float(os.environ.get("BENCH_RSS_CAP_GB", 1.9))
MC_DRAWS = 2000
OA = dict(tol=1e-3, max_iter=25, time_limit=600)                     # certified solves (ce, hybrids, ladder, fine tree)
OA_REOPT = dict(OA, max_iter=10, time_limit=120)                     # certified recourse under perturbed truths
DECKS = [round(0.5 + 0.025 * i, 3) for i in range(25)]              # 0.500 ... 1.100 by 0.025, 1.0 and 1.1 included
OWN, DEAD, PARENT = (0.5, 0.65, 0.8), (0.1, 0.25, 0.4), (0.2, 0.35, 0.5)
RIG, RANGE = (2.0, 2.5, 3.0), (2.0, 3.0, 4.0)
PERTURBATIONS = [("mu_bias_0.85", dict(mu_bias=0.85)), ("mu_bias_1.15", dict(mu_bias=1.15)),
                 ("h_slope_0.5", dict(h_slope=0.5)), ("h_slope_1.5", dict(h_slope=1.5)),
                 ("range_0.5", dict(range_mult=0.5)), ("range_2.0", dict(range_mult=2.0)),
                 ("eur4_1.4", dict(eur4=1.4)), ("eur4_1.7", dict(eur4=1.7)),
                 ("sig_p_0.15", dict(sig_p=0.15)), ("sig_p_0.35", dict(sig_p=0.35))]
FAMILY = ("rn_deck", "rn_tilt1", "rn_tilt2")      # the registered deck family; rn_tilt uses z_A's weights: not a member
Z_A = "ce"                                         # the certified plan the family is compared with
PERTURB_METHODS = ("ce", "rn", "rn_deck", "rn_tilt2", "heur_ce", "ga")   # default (--perturb_methods)
FINE_RN = ("rn", "mv_rn", "ce_fs+rn_recourse")     # rows whose recourse is risk-neutral: the fine tree gives them RN recourse too
NAN = float("nan")
_T_RE = re.compile(r"t=([0-9.]+)s")


def quiet(*a, **k):
    pass


# ================================================================================================
# instance knobs, memory guard, saving, small helpers
# ================================================================================================
def knobs_for_seed(s):
    """Generator knobs varied by seed through five different modular functions (all ten combinations differ for s < 10)."""
    return dict(own_frac=OWN[s % 3], deadline_frac=DEAD[(s // 3) % 3], parent_frac=PARENT[(s + s // 3) % 3],
                rig_mult=RIG[(s + 2 * (s // 3)) % 3], range_mi=RANGE[(s // 2) % 3])


def rss_gb():
    """Peak RSS of this process so far, GB."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6             # bytes on macOS, kilobytes on Linux


def rss_now_gb():
    """Current RSS of this process, GB (None without psutil)."""
    try:
        return _PROC.memory_info().rss / 1e9 if _PROC is not None else None
    except Exception:
        return None


def guard(tag):
    """Stop the process when the peak RSS exceeds RSS_CAP_GB (save first: completed work must already be on disk)."""
    g = rss_gb()
    print(f"  [{tag}] peak RSS {g:.2f} GB", flush=True)
    if g > RSS_CAP_GB:
        sys.exit(f"stopping: peak RSS {g:.2f} GB exceeds {RSS_CAP_GB} GB")


class Clock:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.t = time.time() - self.t0


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def dump(res, path):
    """Atomic JSON write: a temporary file next to the target, flushed and fsynced, then renamed over it."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(jsonable(res), f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def error_entry(stage, e, t=None):
    """A failure record for run['errors']; call it inside the except block (the traceback is the current exception's)."""
    return dict(stage=stage, error=repr(e), traceback=traceback.format_exc()[-4000:], time=t)


def expected_wells(M, plan):
    p = M.I["prob"]
    return float(plan["z0"] @ M.wells[M.idx0]) + float(sum(p[k] * (plan["zk"][k] @ M.wells[M.idx1]) for k in range(M.K)))


def first_stage_key(plan):
    return plan["z0"].tobytes() + plan["y"].tobytes() + plan["w0"].tobytes()


def first_stage_into(M_to, plan):
    """The plan's first stage in an empty plan of another Model on the same geometry (z0, y, w0 do not depend on the tree)."""
    q = M_to.empty_plan()
    assert all(np.shape(plan[k]) == q[k].shape for k in ("z0", "y", "w0")), "first stage of a different option table"
    q["z0"], q["y"], q["w0"] = plan["z0"].copy(), plan["y"].copy(), plan["w0"].copy()
    return q


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def sparse_plan(plan):
    """A plan as sparse index lists: z0, the positions in M.idx0 of the year-0 options taken (option M.idx0[i]); zk,
    per node, the positions in M.idx1; y, the packages bought; w0, the units padded in year 0; wk, per node, the flat
    indices u * (T - 1) + (t - 1) of the pads built in year t >= 1. dense_plan(M, .) inverts it."""
    nz = lambda a: np.flatnonzero(np.asarray(a).ravel() > 0.5).tolist()  # noqa: E731
    return dict(z0=nz(plan["z0"]), zk=[nz(r) for r in plan["zk"]], y=nz(plan["y"]), w0=nz(plan["w0"]),
                wk=[nz(w) for w in plan["wk"]])


def dense_plan(M, sp_plan):
    """The plan dict of a sparse_plan on Model M (same instance and tree)."""
    ix = lambda v: np.asarray(v, dtype=np.intp)  # noqa: E731
    q = M.empty_plan()
    q["z0"][ix(sp_plan["z0"])] = 1.0
    q["y"][ix(sp_plan["y"])] = 1.0
    q["w0"][ix(sp_plan["w0"])] = 1.0
    for k in range(M.K):
        q["zk"][k, ix(sp_plan["zk"][k])] = 1.0
        w = np.zeros(M.U * (T - 1))
        w[ix(sp_plan["wk"][k])] = 1.0
        q["wk"][k] = w.reshape(M.U, T - 1)
    return q


def plan_format(M):
    """The dimensions behind run['plans'] and a SHA-1 of the option table (o_u, o_b, o_len, o_k, o_t as int64, in Model's
    enumeration order), so saved plans can be checked against a rebuilt Model before dense_plan."""
    tab = np.ascontiguousarray(np.c_[M.o_u, M.o_b, M.o_len, M.o_k, M.o_t].astype(np.int64))
    return dict(n0=M.n0, n1=M.n1, K=M.K, G=M.G, U=M.U, T=T, options_sha1=hashlib.sha1(tab.tobytes()).hexdigest(),
                encoding="z0/zk: positions in idx0/idx1; y: packages; w0: units; wk: per node, u * (T - 1) + (t - 1)")


def mean_value_model(I, lam, deck=1.0):
    """The deterministic mean-value problem: one node at theta = P = 1 with probability 1, built from a copy of the
    instance as a deterministic planner would build it (its credited set, tau, Texp, Cons and base are those of
    theta = P = 1). The field and the per-well noise stay, so its CE version is risk-averse to them."""
    J = copy.deepcopy(I)
    J["theta"], J["price"], J["prob"], J["n_nodes"] = np.array([1.0]), np.array([1.0]), np.array([1.0]), 1
    return Model(J, lam, deck=deck)


# ================================================================================================
# scoring: exact CE, E[NPV], sigma, lognormal MC with SE, variance decomposition, implied tilt,
# placement, per-node contingency, rig/crew/capital utilization
# ================================================================================================
def placement(M, plan):
    """Where the wells go: unit-benches developed in year 0 and in any node; wells by year, bench and length
    (expected over nodes); packages; a per-node contingency table; the first stage as option tuples."""
    I, p, z0 = M.I, M.I["prob"], plan["z0"]
    o0 = M.idx0[z0 > 0.5]
    by_bench, by_len, by_year = np.zeros(NB), np.zeros(3), np.zeros(T)
    for o in o0:
        by_bench[M.o_b[o]] += M.o_k[o]; by_len[M.o_len[o] - 2] += M.o_k[o]; by_year[0] += M.o_k[o]
    by_node, dev_any = [], set()
    for k in range(M.K):
        ok = M.idx1[plan["zk"][k] > 0.5]
        for o in ok:
            by_bench[M.o_b[o]] += p[k] * M.o_k[o]; by_len[M.o_len[o] - 2] += p[k] * M.o_k[o]; by_year[M.o_t[o]] += p[k] * M.o_k[o]
        z = M.full_z(plan, k)
        dev_k = set(M.o_ub[z > 0.5].tolist()); dev_any |= dev_k
        capex = float(M.C[M.idx1] @ plan["zk"][k] + (M.pad_cost[:, 1:] * plan["wk"][k]).sum())
        by_node.append(dict(theta=float(I["theta"][k]), price=float(I["price"][k]), prob=float(p[k]),
                            wells_recourse=float(plan["zk"][k] @ M.wells[M.idx1]), capex_recourse=capex, dev_ub=len(dev_k)))
    return dict(wells_expected=expected_wells(M, plan), wells_year0=float(z0 @ M.wells[M.idx0]),
                units_year0=len(set(M.o_u[o0].tolist())), dev_ub_year0=sorted(set(M.o_ub[o0].tolist())), dev_ub_any=sorted(dev_any),
                by_bench=by_bench.tolist(), by_len={str(l): float(by_len[l - 2]) for l in (2, 3, 4)}, by_year=by_year.tolist(),
                packages=int(plan["y"].sum()), by_node=by_node,
                first_stage=dict(options=[(int(M.o_u[o]), int(M.o_b[o]), int(M.o_len[o]), int(M.o_k[o])) for o in o0],
                                 packages=np.flatnonzero(plan["y"] > 0.5).tolist()))


def utilization(M, plan, k):
    """Rig, crew and capital utilization per year (share of capacity) in node k; year 0 is the same in every node."""
    I, z = M.I, M.full_z(plan, k)
    pads = np.c_[plan["w0"], plan["wk"][k]]
    rig, crew, cap = [], [], []
    for t in range(T):
        sel = z * (M.o_t == t)
        rig.append(float(M.rigdays @ sel / I["cap"]["rig"][t]))
        crew.append(float(M.lat_miles @ sel / I["cap"]["crew"][t]))
        cap.append(float((M.C @ sel + M.pad_cost[:, t] @ pads[:, t]) / I["cap"]["capital"][t]))
    return dict(rig=rig, crew=crew, capital=cap)


def score(M, name, plan, extra=None, label=None):
    ev = M.evaluate(plan)
    mc, se = M.ce_mc(plan, MC_DRAWS, return_se=True)
    tilt = M.implied_tilt(plan)
    p = M.I["prob"]
    util_exp = {q: list(np.zeros(T)) for q in ("rig", "crew", "capital")}
    for k in range(M.K):
        uk = utilization(M, plan, k)
        for q in util_exp:
            util_exp[q] = [a + p[k] * b for a, b in zip(util_exp[q], uk[q])]
    row = dict(label=label or name, feasible=bool(M.feasible(plan)), ce=ev["ce"], enpv=ev["enpv"], sigma=ev["sigma"],
               lam_sigma=ev["lam_sigma"], risk_premium=ev["enpv"] - ev["ce"], ce_mc=mc, ce_mc_se=se,
               sigma_decomp=M.sigma_decomposition(plan),
               tilt=dict(pi=tilt["pi"].tolist(), implied_price=tilt["implied_price"], implied_theta=tilt["implied_theta"]),
               placement=placement(M, plan), util_mid=utilization(M, plan, M.K // 2), util_exp=util_exp)
    if extra:
        row.update(extra)
    cert = f"  cert [{row['ce_lb']:.2f}, {row['ce_ub']:.2f}] {row['rounds']}r" if "ce_lb" in row else ""
    print(f"  {name:22s} CE {row['ce']:9.2f}  E[NPV] {row['enpv']:9.2f}  lam*sig {row['lam_sigma']:5.2f}  "
          f"MC {mc:9.2f}+-{se:5.2f}  wells {row['placement']['wells_expected']:5.1f}  pkgs {row['placement']['packages']}  "
          f"{'ok' if row['feasible'] else 'INFEASIBLE'}  {row.get('time', 0.0):6.1f}s{cert}", flush=True)
    return row


def cert_extra(r, wall, seed_time=0.0):
    """Certificate fields of a solve_ce result: bracket, gap, rounds, rank and the history with cumulative wall-clock
    (solve_ce's per-round times exclude Master construction and the seed cuts; that overhead, wall minus the sum of the
    round times, is charged before the first round so cum_time is time since the solve started)."""
    hist, cum = [], max(0.0, wall - sum(h["time"] for h in r["hist"]))
    for h in r["hist"]:
        cum += h["time"]; hist.append(dict(h, cum_time=cum))
    return dict(time=wall + seed_time, solver_time=wall, seed_time=seed_time, ce_lb=r["ce_lb"], ce_ub=r["ce_ub"],
                gap_pct=100 * (r["ce_ub"] - r["ce_lb"]) / max(1e-9, abs(r["ce_lb"])), rounds=r["rounds"], rank=r["rank"],
                master=r["master"], hist=hist, bound_valid=r.get("bound_valid", True), numeric_resolves=r.get("numeric_resolves", 0),
                disproved_bounds=r.get("disproved_bounds", []), solver_rev=r.get("solver_rev"), witness_checks=r.get("witness_checks", 0),
                witness_refuted=r.get("witness_refuted", 0))


# ================================================================================================
# method runners and the run-level tests
# ================================================================================================
def deck_family(I, M, lam, rn_plan):
    """Risk-neutral MIP at every deck of the grid, scored by CE under the true model; deck 1.0 is the rn plan itself."""
    curve, best = [], None
    for d in DECKS:
        if abs(d - 1.0) < 1e-9:
            pd, tt = rn_plan, 0.0
        else:
            with Clock() as c:
                pd = solve_risk_neutral(Model(I, lam, deck=d))["plan"]
            tt = c.t
        e = M.evaluate(pd)
        curve.append(dict(deck=d, ce=e["ce"], enpv=e["enpv"], time=tt))
        if best is None or e["ce"] > best[0]:
            best = (e["ce"], pd, d)
    return best, curve


def run_meta(fn, M, budget, seed, objective, fill, n_particles=40):
    """One PSO/GA run with its best-vs-time curve: iteration times are parsed from the verbose log (the library keeps
    best_history per iteration without times); entry 0 is the initial population (its time is estimated from the mean
    evaluation time), which contains the heuristic seed."""
    buf = io.StringIO()
    with Clock() as c, contextlib.redirect_stdout(buf):
        plan, info = fn(M, time_budget=budget, seed=seed, objective=objective, fill=fill, verbose=True)
    times = [float(v) for v in _T_RE.findall(buf.getvalue())]
    hist = [float(v) for v in info["best_history"]]
    t_init = min(times[0] if times else c.t, n_particles * c.t / max(1, info["evals"]))
    tt = [t_init] + times if len(times) == len(hist) - 1 else list(np.linspace(t_init, info["time"], len(hist)))
    return plan, dict(seed=seed, iters=info["iters"], evals=info["evals"], time=info["time"], wall=c.t,
                      seed_score=info["seed_score"], best_score=info["best_score"],
                      best_history=[(round(a, 2), b) for a, b in zip(tt, hist)])


def fine_vss(F):
    """The VSS on the fine tree from fine_tree_test's rows: RN-VSS (rn's against mv_rn's first stage, both with
    risk-neutral recourse; interval from the MIPs' dual bounds), CE-VSS (ce's against mv_ce+exact_recourse's first stage)
    and the mixed VSS (ce's against mv+exact_recourse's), both with certified recourse; point values and certificate
    intervals."""
    out = {}
    a, b = F.get("rn", {}).get("rn_recourse"), F.get("mv_rn", {}).get("rn_recourse")
    if a and b:
        out.update(rn_vss=a["enpv"] - b["enpv"], rn_vss_lb=a["enpv"] - b["bound"], rn_vss_ub=a["bound"] - b["enpv"])
    for key, other in (("ce_vss", "mv_ce+exact_recourse"), ("vss_mixed", "mv+exact_recourse")):
        a, b = F.get(Z_A), F.get(other)
        if a and b:
            out.update({key: a["ce_lb"] - b["ce_lb"], key + "_lb": a["ce_lb"] - b["ce_ub"], key + "_ub": a["ce_ub"] - b["ce_lb"]})
    return out


def fine_tree_test(I, M, lam, plans, n_fine, out=None, checkpoint=None, errors=None):
    """Every method's first stage with certified recourse on a finer n_fine x n_fine node tree (one solve per distinct
    first stage); the fairest first-stage comparison. v3.2: the first stages of the FINE_RN rows (whose recourse is
    risk-neutral by definition) also get risk-neutral recourse on the fine tree (row['rn_recourse']: E[NPV] with the
    MIP's dual bound, and the CE of that plan), and out['vss'] holds the fine-tree RN-VSS, CE-VSS and mixed VSS
    (fine_vss). Results go into `out` as they are computed, with checkpoint() after every new solve; a failing first
    stage is recorded in out['errors'] (and in `errors`) and the test goes on."""
    out = {} if out is None else out
    t0 = time.time()
    MJ = Model(perturb(I, n_nodes=n_fine), lam)
    assert MJ.n0 == M.n0 and MJ.n == M.n, "options depend on the tree: cannot map first stages"
    out.update(n_nodes=n_fine, K=MJ.K)
    F = out.setdefault("methods", {})
    cache, rn_cache, failed = {}, {}, {}
    for name, plan in plans.items():
        key, new = first_stage_key(plan), False
        if key in failed:
            out.setdefault("errors", {})[name] = f"same first stage as {failed[key]}, whose certified recourse failed"
            continue
        try:
            if key not in cache:
                with Clock() as c:
                    r = solve_recourse_exact(MJ, first_stage_into(MJ, plan), **OA)
                e = MJ.evaluate(r["plan"])
                cache[key] = dict(ce_lb=r["ce_lb"], ce_ub=r["ce_ub"], gap_pct=100 * (r["ce_ub"] - r["ce_lb"]) / max(1e-9, abs(r["ce_lb"])),
                                  rounds=r["rounds"], time=c.t, enpv=e["enpv"], lam_sigma=e["lam_sigma"], bound_valid=r.get("bound_valid", True),
                                  wells_expected=expected_wells(MJ, r["plan"]))
                new = True
                print(f"  fine-tree {name:22s} certified CE [{r['ce_lb']:.2f}, {r['ce_ub']:.2f}]  E[NPV] {e['enpv']:.2f}  {c.t:.1f}s", flush=True)
            else:
                print(f"  fine-tree {name:22s} same first stage as an earlier row: CE {cache[key]['ce_lb']:.2f}", flush=True)
            F[name] = dict(cache[key])
        except Exception as e:
            traceback.print_exc()
            failed[key] = name
            out.setdefault("errors", {})[name] = repr(e)
            if errors is not None:
                errors.append(error_entry(f"fine:{name}", e))
            new = True
        if name in FINE_RN and name in F:
            try:
                if key not in rn_cache:
                    with Clock() as c:
                        rr = solve_risk_neutral(MJ, fixed=first_stage_into(MJ, plan))
                    er = MJ.evaluate(rr["plan"])
                    rn_cache[key] = dict(enpv=er["enpv"], bound=rr["bound"], mip_gap=rr["gap"], ce=er["ce"], lam_sigma=er["lam_sigma"],
                                         time=c.t, wells_expected=expected_wells(MJ, rr["plan"]))
                    new = True
                    print(f"  fine-tree {name:22s} risk-neutral recourse E[NPV] {er['enpv']:.2f} (bound {rr['bound']:.2f})  "
                          f"CE {er['ce']:.2f}  {c.t:.1f}s", flush=True)
                F[name]["rn_recourse"] = dict(rn_cache[key])
            except Exception as e:
                traceback.print_exc()
                out.setdefault("errors", {})[name + " (rn recourse)"] = repr(e)
                if errors is not None:
                    errors.append(error_entry(f"fine:{name}:rn_recourse", e))
                new = True
        out["time"] = time.time() - t0
        if new and checkpoint is not None:
            checkpoint(f"fine:{name}")
    out["vss"] = fine_vss(F)
    out["time"] = time.time() - t0
    return out


def perfect_information_bound(M, ref_plan, **kw):
    """CE under perfect information, CE_PI = -(1/lam) log sum_k p_k exp(-lam CE_k), with CE_k the certified optimum
    of node k's single-node problem (first stage free). Unlike solve.perfect_information, the node model carries the
    full model's terminal credit and exposure for node k (Model rebuilds the credited set with prob = [1], which values
    a different model), and the node solve is seeded with the two-stage plan restricted to node k; so each CE_k is at
    least that plan's node-k CE and CE_PI >= CE(ref_plan) holds exactly, not only up to the solver tolerance."""
    lam, p = M.lam, M.I["prob"]
    q_ref = M.evaluate(ref_plan)["q"]
    ces, ubs, valid = [], [], []
    for k in range(M.K):
        Mk = Model(single_node(M.I, k), lam, deck=M.deck)
        Mk.tau, Mk.tau_best, Mk.Texp = M.tau[:, [k]].copy(), M.tau_best.copy(), M.Texp[[k]].copy()
        Mk.Cons = M.Cons.copy()                                      # v3.1: lane-consumption netting follows the credited set
        Mk.B_k, Mk.b_k = M.B_k[[k]].copy(), M.b_k[:, [k]].copy()
        seed = dict(z0=ref_plan["z0"].copy(), zk=ref_plan["zk"][[k]].copy(), y=ref_plan["y"].copy(),
                    w0=ref_plan["w0"].copy(), wk=ref_plan["wk"][[k]].copy())
        qk = Mk.evaluate(seed)["q"][0]
        assert abs(qk - q_ref[k]) <= 1e-7 * max(1.0, abs(q_ref[k])), f"node model {k} does not reproduce the full model's node term"
        r = solve_ce(Mk, seeds=(seed,), log=quiet, **kw)
        ces.append(r["ce_lb"]); ubs.append(r["ce_ub"]); valid.append(bool(r.get("bound_valid", True)))
    ces, ubs = np.array(ces), np.array(ubs)
    return dict(ce_pi_lb=float(-logsumexp(-lam * ces, b=p) / lam), ce_pi_ub=float(-logsumexp(-lam * ubs, b=p) / lam),
                node_ce=ces.tolist(), node_ub=ubs.tolist(), node_bound_valid=valid, bound_valid=all(valid))


def ladder_block(two, com=None, pi=None, eev=None, rn=None, mv_rn=None, mv_ce=None, two_best=None, time_pi=NAN):
    """The information ladder and the VSS, as stored in run['ladder'].

    two, com, eev, mv_ce: (LB, UB) certificates of the two-stage CE plan, the committed plan, the mean-value first stage
    (theta = P = 1, risk-neutral) with certified CE recourse, and the CE mean-value first stage with certified CE
    recourse; pi: perfect_information_bound's dict; rn, mv_rn: (E[NPV] of the plan, the MIP's dual bound) of the
    risk-neutral two-stage MIP and of the mean-value first stage with risk-neutral recourse; two_best: (exact CE, UB,
    method) of the best certified incumbent and the best upper bound over the certified solves (z_A). Missing
    pieces give NaN. The v3.1 keys keep their meaning (value_of_recourse, evpi and vss are lower-end differences; eev and
    vss are the mixed EEV and VSS). v3.2 adds each difference's certificate interval [LB_a - UB_b, UB_a - LB_b] as *_lb
    and *_ub, vss_mixed (= vss), the RN-VSS E[NPV](rn) - E[NPV](mv_rn) and the CE-VSS CE(ce) - CE(mv_ce+exact_recourse)."""
    pair = lambda x: (float(x[0]), float(x[1])) if x is not None else (NAN, NAN)  # noqa: E731
    two, com, eev, mvce, rnb, mvrn = (pair(x) for x in (two, com, eev, mv_ce, rn, mv_rn))
    pib = (float(pi["ce_pi_lb"]), float(pi["ce_pi_ub"])) if pi else (NAN, NAN)

    def diff(name, a, b):                     # (optimum a) - (optimum b), each given by its (low end, high end)
        return {name: a[0] - b[0], name + "_lb": a[0] - b[1], name + "_ub": a[1] - b[0]}

    L = dict(committed_lb=com[0], committed_ub=com[1], two_stage_lb=two[0], two_stage_ub=two[1], pi_lb=pib[0], pi_ub=pib[1],
             node_ce=list(pi["node_ce"]) if pi else [], node_ub=list(pi["node_ub"]) if pi else [], eev=eev[0], eev_ub=eev[1],
             time_pi=time_pi)
    L.update(diff("value_of_recourse", two, com)); L.update(diff("evpi", pib, two)); L.update(diff("vss", two, eev))
    L.update(diff("vss_mixed", two, eev))
    L.update(enpv_rn=rnb[0], enpv_rn_bound=rnb[1], enpv_mv_rn=mvrn[0], enpv_mv_rn_bound=mvrn[1], eev_ce_lb=mvce[0], eev_ce_ub=mvce[1])
    L.update(diff("rn_vss", rnb, mvrn)); L.update(diff("ce_vss", two, mvce))
    L.update(two_stage_best_lb=float(two_best[0]) if two_best else NAN, two_stage_best_ub=float(two_best[1]) if two_best else NAN,
             two_stage_best_method=two_best[2] if two_best else None)      # always present: a stable key set
    return L


def perturbed_truth(I, M, lam, plans, errors=None):
    """Every plan re-scored (paired) under perturbations of the generator's truth, recourse as planned; constraints
    are untouched. A failing truth is left out of the table and recorded in `errors`."""
    table = {}
    for label, kw in PERTURBATIONS:
        try:
            MP = Model(perturb(I, **kw), lam)
            assert MP.n == M.n and MP.K == M.K, "perturbation changed the option set or the tree"
            t = {}
            for name, plan in plans.items():
                e = MP.evaluate(plan)
                t[name] = dict(ce=e["ce"], enpv=e["enpv"], lam_sigma=e["lam_sigma"])
            table[label] = t
            print(f"  perturbed {label:13s} CE: " + "  ".join(f"{n} {t[n]['ce']:.1f}" for n in ("rn", "rn_deck", "rn_tilt2", "heur_ce", "pso", "ce") if n in t), flush=True)
        except Exception as e:
            traceback.print_exc()
            if errors is not None:
                errors.append(error_entry(f"perturbed:{label}", e))
    return table


def zA_vs_family(rows, enpv_zN):
    """z_A's gain over the best deck-family member under one truth, from perturbed_reopt's rows (None without z_A or a
    member). Each member's optimum lies in its [low, high] bracket, so the best member's lies in [max low, max high] and
    the gain in [low_A - max high, high_A - max low]; the point value compares the incumbents. CE: certified recourse,
    brackets [ce_lb, ce_ub]; E[NPV]: risk-neutral recourse, brackets [enpv, enpv_bound]. beats_* means the interval lies
    above 0, loses_* below 0 (neither: a tie within the certificates); *_pct is % of enpv_zN."""
    fam = [m for m in FAMILY if m in rows]
    if Z_A not in rows or not fam:
        return None
    A, out = rows[Z_A], dict(members=fam, enpv_zN=enpv_zN)
    for score_, lo, hi in (("ce", "ce_lb", "ce_ub"), ("enpv", "enpv", "enpv_bound")):
        best = max(fam, key=lambda m: rows[m][lo])
        lo_f, hi_f = max(rows[m][lo] for m in fam), max(rows[m][hi] for m in fam)
        g, g_lo, g_hi = A[lo] - lo_f, A[lo] - hi_f, A[hi] - lo_f
        out.update({f"best_{score_}": best, f"gain_{score_}": g, f"gain_{score_}_lb": g_lo, f"gain_{score_}_ub": g_hi,
                    f"gain_{score_}_pct": 100.0 * g / abs(enpv_zN) if np.isfinite(enpv_zN) and enpv_zN else NAN,
                    f"beats_{score_}": bool(g_lo > 0), f"loses_{score_}": bool(g_hi < 0)})
    return out


def perturbed_reopt(I, M, lam, plans, methods, out=None, checkpoint=None, errors=None, enpv_zN=NAN, oa=None):
    """First-stage robustness under perturbed truths. For every truth of PERTURBATIONS and every listed method
    (missing ones are skipped; when z_A is listed, the deck family {rn_deck, rn_tilt1, rn_tilt2} is completed so the
    comparison is the registered one), the first stage (z0, y, w0) is fixed and the recourse re-optimized under the truth
    twice: risk-neutral, solve_risk_neutral(MP, fixed=.), scored on E[NPV] with the MIP's dual bound; and certified,
    solve_ce(MP, fixed=., seeds=every listed plan with that first stage) with OA_REOPT, scored on CE with its bracket.
    One pair of solves per distinct first stage and truth (out['groups'] lists the methods that share one).

    out['table'][truth][method]: ce (= ce_lb, the certified incumbent), ce_lb, ce_ub, rounds, enpv (risk-neutral
    recourse), enpv_bound, mip_gap, the other score of each re-optimized plan (enpv_ce_recourse, ce_rn_recourse), the plan
    as planned (ce_asis, enpv_asis) and the solve times. out['zA_vs_family'][truth]: zA_vs_family with E(z_N) under that
    truth (z_N's first stage with risk-neutral recourse; enpv_zN, the model truth's, without rn). out['summary']: counts
    over truths. A failing truth, or a failing first stage under one truth (its methods are then missing from that
    truth's rows), is recorded in out['errors'] (and in `errors`); checkpoint() runs after every truth."""
    oa = OA_REOPT if oa is None else oa
    out = {} if out is None else out
    t0 = time.time()
    want = [m for m in methods if m in plans]
    added = [m for m in FAMILY if Z_A in want and m in plans and m not in want]
    want += added
    groups = {}
    for m in want:
        groups.setdefault(first_stage_key(plans[m]), []).append(m)
    out.update(requested=list(methods), methods=want, skipped=[m for m in methods if m not in plans], added_for_family=added,
               family=list(FAMILY), z_A=Z_A, oa=dict(oa), groups=list(groups.values()), table={}, zA_vs_family={})
    print(f"  perturbed truths, recourse re-optimized: {len(want)} methods in {len(groups)} distinct first stages"
          + (f"; skipped {', '.join(out['skipped'])}" if out["skipped"] else "") + (f"; added for the family {', '.join(added)}" if added else ""), flush=True)

    def failed(tag, e):
        traceback.print_exc()
        out.setdefault("errors", {})[tag] = repr(e)
        if errors is not None:
            errors.append(error_entry(f"perturbed_reopt:{tag}", e))

    for label, kw in PERTURBATIONS:
        try:
            MP = Model(perturb(I, **kw), lam)
            assert MP.n == M.n and MP.K == M.K, "perturbation changed the option set or the tree"
        except Exception as e:
            failed(label, e)
            MP = None
        if MP is not None:
            rows = {}
            for names in groups.values():                             # a failing first stage drops only its own rows
                try:
                    seeds = tuple(plans[m] for m in names)
                    with Clock() as c1:
                        rr = solve_risk_neutral(MP, fixed=seeds[0])
                    er = MP.evaluate(rr["plan"])
                    with Clock() as c2:
                        rc = solve_ce(MP, seeds=seeds, fixed=seeds[0], log=quiet, **oa)
                    ec = MP.evaluate(rc["plan"])
                    common = dict(ce=rc["ce_lb"], ce_lb=rc["ce_lb"], ce_ub=rc["ce_ub"], rounds=rc["rounds"], bound_valid=rc.get("bound_valid", True), enpv_ce_recourse=ec["enpv"],
                                  enpv=er["enpv"], enpv_bound=rr["bound"], mip_gap=rr["gap"], ce_rn_recourse=er["ce"],
                                  time_rn=c1.t, time_ce=c2.t)
                    for m in names:
                        e0 = MP.evaluate(plans[m])
                        rows[m] = dict(common, ce_asis=e0["ce"], enpv_asis=e0["enpv"])
                except Exception as e:
                    failed(f"{label}:{'/'.join(names)}", e)
            out["table"][label] = rows
            try:
                den = rows["rn"]["enpv"] if "rn" in rows else enpv_zN
                v = out["zA_vs_family"][label] = zA_vs_family(rows, den)
                vs = (f"; z_A - best family: CE {v['gain_ce']:+.2f} [{v['gain_ce_lb']:+.2f}, {v['gain_ce_ub']:+.2f}] vs {v['best_ce']}, "
                      f"E[NPV] {v['gain_enpv']:+.2f} [{v['gain_enpv_lb']:+.2f}, {v['gain_enpv_ub']:+.2f}] vs {v['best_enpv']}") if v else ""
                print(f"  reopt {label:13s} CE|E[NPV]: " + "  ".join(f"{m} {rows[m]['ce']:.1f}|{rows[m]['enpv']:.1f}" for m in want if m in rows)
                      + vs, flush=True)
            except Exception as e:
                failed(f"{label}:comparison", e)
        V = [v for v in out["zA_vs_family"].values() if v]              # running summary: an interrupted stage still carries one
        out["summary"] = dict(n_truths=len(V), **{f"{w}_{s}": int(sum(v[f"{w}_{s}"] for v in V)) for w in ("beats", "loses") for s in ("ce", "enpv")},
                              min_gain_ce_pct=float(min((v["gain_ce_pct"] for v in V), default=NAN)),
                              max_gain_ce_pct=float(max((v["gain_ce_pct"] for v in V), default=NAN)))
        out["time"] = time.time() - t0
        if checkpoint is not None:
            checkpoint(f"perturbed_reopt:{label}")
    return out


# ================================================================================================
# one (instance, rho), stage by stage
# ================================================================================================
class Stages:
    """Each stage of a run in its own try/except. A stage whose inputs (rows of `plans` or keys of `ctx`) are
    missing is skipped; a failing stage appends error_entry(...) to run['errors'] and the run goes on. After every stage
    its record (ok, wall-clock, the process's peak RSS so far, the rise of the peak during the stage, current RSS) goes
    to run['stages'][name] and checkpoint() saves the JSON and applies the memory guard."""

    def __init__(self, run, plans, ctx, save):
        self.run, self.plans, self.ctx, self.save, self.t0 = run, plans, ctx, save, time.time()
        run.setdefault("errors", []); run.setdefault("stages", {})

    def _save(self, tag):
        try:
            self.save()
        except Exception:                                               # retried at the next checkpoint; main's final dump raises
            traceback.print_exc()
            print(f"  [{tag}] could not save the JSON; retrying at the next checkpoint", flush=True)

    def checkpoint(self, tag):
        """Save the run; when the peak RSS exceeds RSS_CAP_GB, record the stop in run['status'], save and exit."""
        self.run["time_total"] = time.time() - self.t0
        self._save(tag)
        g = rss_gb()
        if g > RSS_CAP_GB:
            msg = f"stopping: peak RSS {g:.2f} GB exceeds {RSS_CAP_GB} GB"
            self.run["status"] = f"{msg} (after {tag})"
            self._save(tag)
            sys.exit(msg)

    def __call__(self, name, fn, needs=()):
        missing = [n for n in needs if n not in self.plans and n not in self.ctx]
        if missing:
            self.run["stages"][name] = dict(ok=False, skipped=True, missing=missing)
            print(f"  [{name}] skipped: missing {', '.join(missing)}", flush=True)
            self.checkpoint(name)
            return None
        t0, peak0 = time.time(), rss_gb()
        ok, out = True, None
        try:
            out = fn()
        except Exception as e:
            ok = False
            traceback.print_exc()
            self.run["errors"].append(error_entry(name, e, time.time() - t0))
        peak = rss_gb()
        self.run["stages"][name] = dict(ok=ok, time=time.time() - t0, peak_rss_gb=peak, rss_rise_gb=peak - peak0, rss_now_gb=rss_now_gb())
        print(f"  [{name}] {'ok' if ok else 'error (recorded; the run goes on)'}  {time.time() - t0:.1f}s  peak RSS {peak:.2f} GB", flush=True)
        self.checkpoint(name)
        return out


def bench_one(I, s, knobs, rho, args, run, save):
    """One (instance, rho), filling `run` (already in the output) stage by stage; save() rewrites the output."""
    lam = 1.0 / rho
    M = Model(I, lam)
    cap = I["cap"]
    print(f"=== seed {s}  U={M.U} nub={M.nub} options={M.n} n0={M.n0} K={M.K} G={M.G} binaries~{M.n0 + M.K * M.n1}  rho ${rho:,.0f}M  "
          f"knobs " + " ".join(f"{k}={v}" for k, v in knobs.items()), flush=True)
    run.update(dict(seed=s, knobs=knobs, model_version=MODEL_VERSION, solver_rev=SOLVER_REV, U=M.U, nub=M.nub, options=M.n, n0=M.n0, n1=M.n1, K=M.K, G=M.G, binaries=M.n0 + M.K * M.n1,
                    rho=rho, lam=lam, cap=dict(rig=float(cap["rig"][0]), crew=float(cap["crew"][0]), capital=float(cap["capital"][0]), acq=float(cap["acq"])),
                    fringe_units=int(M.fringe_u.sum()), credited_terminal_ub=int((M.tau_best >= 0).sum()), nu_term=M.nu_term,
                    methods={}, plans={}, plan_format=plan_format(M)))
    R, P, ctx = run["methods"], {}, {}
    stage = Stages(run, P, ctx, save)

    def add(name, plan, extra=None, label=None):
        row = score(M, name, plan, extra, label)
        P[name], R[name], run["plans"][name] = plan, row, sparse_plan(plan)

    # ---- 1. heuristics ---------------------------------------------------------------------------------------------
    def heur(name, by, label):
        with Clock() as c:
            h, cfg, _ = BL.tune_heuristic(M, by=by)
        add(name, h, dict(time=c.t, cfg=cfg), label)

    stage("heur_ev", lambda: heur("heur_ev", "enpv", "rank-and-fill with recourse, settings tuned by E[NPV]"))
    stage("heur_ce", lambda: heur("heur_ce", "ce", "rank-and-fill with recourse, settings tuned by CE (oracle)"))

    # ---- 2. risk-neutral MIP, Beck's screen, RN-LP rounding, the deck family -----------------------------------------
    def rn_mip():
        with Clock() as c:
            rn = solve_risk_neutral(M)
        add("rn", rn["plan"], dict(time=c.t, mip_gap=rn["gap"], bound=rn["bound"], obj=rn["obj"]), "risk-neutral two-stage MIP at the strip")
        screen = rn["bound"] - R["rn"]["ce"]
        run["screen"] = dict(ub_n=rn["bound"], ce_rn=R["rn"]["ce"], value=screen, pct_enpv_rn=100 * screen / max(1e-9, abs(R["rn"]["enpv"])))
        print(f"  screen: UB_N - CE(x_N) = {rn['bound']:.2f} - {R['rn']['ce']:.2f} = {screen:.2f} "
              f"({run['screen']['pct_enpv_rn']:.1f}% of E[NPV](rn)); no plan can gain more CE over rn", flush=True)

    def lp_round():
        with Clock() as c:
            rr = relax_and_round(M)
        add("rn_lp_round", rr, dict(time=c.t), "greedy rounding of the risk-neutral LP relaxation (RN-LP rounding)")

    def deck():
        with Clock() as c:
            (_, deck_plan, d), curve = deck_family(I, M, lam, P["rn"])
        add("rn_deck", deck_plan, dict(time=c.t + R["rn"]["time"], deck=d, deck_curve=curve),
            f"risk-neutral MIP at the oracle-selected planning deck (best CE under the true model; grid {DECKS[0]:.3g}-{DECKS[-1]:.3g} by "
            f"{DECKS[1] - DECKS[0]:.3g}; best {d}); time includes rn")

    stage("rn", rn_mip)
    stage("rn_lp_round", lp_round)
    stage("rn_deck", deck, needs=("rn",))

    # ---- 3. certified solves: cold (its wall-clock is the metaheuristics' budget floor), warm (time includes the seeds), one round
    def ce_cold():
        with Clock() as c:
            cold = solve_ce(M, seeds=(), log=quiet, **OA)
        ctx["ub_ce_cold"] = cold["ce_ub"]
        add("ce_cold", cold["plan"], cert_extra(cold, c.t), "certified CE solver, cold start (wall-clock = fair solver time)")

    def ce_warm():
        names = [n for n in ("rn", "rn_lp_round", "heur_ce") if n in P]
        with Clock() as c:
            ce = solve_ce(M, seeds=tuple(P[n] for n in names), log=quiet, **OA)
        ctx["ub_ce"] = ce["ce_ub"]
        add("ce", ce["plan"], dict(cert_extra(ce, c.t, seed_time=sum(R[n]["time"] for n in names)), seeds=names),
            f"certified CE solver seeded with {', '.join(names) or 'nothing'} (time includes the seeds)")

    def ce_1round():
        with Clock() as c:
            q1 = solve_ce(M, seeds=(P["rn"],), max_iter=1, time_limit=30, log=quiet)
        ctx["ub_ce_1round"] = q1["ce_ub"]
        add("ce_1round", q1["plan"], cert_extra(q1, c.t, seed_time=R["rn"]["time"]), "one OA round from the rn seed, 30 s master (time includes rn)")

    stage("ce_cold", ce_cold)
    stage("ce", ce_warm)
    stage("ce_1round", ce_1round, needs=("rn",))
    ubs = [ctx[k] for k in ("ub_ce_cold", "ub_ce", "ub_ce_1round") if k in ctx]
    if ubs:
        run["ce_ub_best"] = float(min(ubs))                              # every certified solve bounds the same problem

    # ---- 4. metaheuristics (tuned recourse rule, several seeds, curves stored, both objectives; the per-seed budget
    #         is at least the seed-free certified solve's wall-clock, so PSO/GA never get less time than ce_cold) -------
    budget = max(args.budget, R["ce_cold"]["solver_time"]) if "ce_cold" in R else args.budget
    run["meta_budget"] = budget

    def meta(name, fn, obj, src, bud, nseeds):
        fill = R[src]["cfg"]
        best, infos, failed = None, [], []
        for j in range(nseeds):
            try:
                plan, info = run_meta(fn, M, bud, 1000 * s + j, obj, fill)
            except Exception as e:                                       # one failing seed does not discard the others
                traceback.print_exc()
                run["errors"].append(error_entry(f"{name}[seed {j}]", e))
                failed.append(j)
                continue
            infos.append(info)
            v = float(M.evaluate(plan)[obj])
            if best is None or v > best[0]:
                best = (v, plan, j)
        if best is None:
            raise RuntimeError(f"all {nseeds} seed(s) of {name} failed")
        add(name, best[1], dict(time=sum(i["wall"] for i in infos), budget=bud, n_seeds=nseeds, objective=obj, fill=fill,
                                evals=[i["evals"] for i in infos], best_seed=best[2], runs=infos, failed_seeds=failed),
            f"{'binary PSO' if fn is BL.pso_plan else 'GA'} on {obj}, {nseeds} seed(s) x {bud:.0f} s, recourse rule = cfg of {src}")

    specs = [("pso", BL.pso_plan, "ce", "heur_ce", budget, args.pso_seeds),
             ("ga", BL.ga_plan, "ce", "heur_ce", budget, args.pso_seeds),
             ("pso_long", BL.pso_plan, "ce", "heur_ce", args.long_mult * budget, 1),
             ("pso_ev", BL.pso_plan, "enpv", "heur_ev", budget, args.ev_seeds),
             ("ga_ev", BL.ga_plan, "enpv", "heur_ev", budget, args.ev_seeds)]
    for spec in specs:
        if spec[5] > 0:
            stage(spec[0], lambda spec=spec: meta(*spec), needs=(spec[3],))

    # ---- 5. risk-neutral variants that need the CE plan: size-matched, pi-tilted (fixed point and oracle); ce_gauss --
    def rn_size():
        we_ce = expected_wells(M, P["ce"])
        with Clock() as c:
            sz = solve_risk_neutral(M, max_expected_wells=we_ce)
        add("rn_size", sz["plan"], dict(time=c.t + R["ce"]["time"], max_expected_wells=we_ce, mip_gap=sz["gap"]),
            "risk-neutral MIP with expected wells <= the CE plan's (size-matched; the size is an oracle input, time includes ce)")

    def tilted(name, src, label):
        with Clock() as c:
            tl = M.implied_tilt(P[src]); r = solve_risk_neutral(M, weights=tl["pi"])
        add(name, r["plan"], dict(time=c.t + R[src]["time"], implied_price=tl["implied_price"], implied_theta=tl["implied_theta"],
                                  pi=tl["pi"].tolist()), label)

    def ce_gauss():                                    # the seed paper's per-well treatment: second-cumulant truncation of the exact CGF
        names = [n for n in ("rn", "rn_lp_round", "heur_ce") if n in P]
        with Clock() as c:
            Mg = Model(I, lam, gaussian_wells=True)
            gz = solve_ce(Mg, seeds=tuple(P[n] for n in names), **OA)
        add("ce_gauss", gz["plan"], dict(time=c.t + (R["rn"]["time"] if "rn" in R else 0.0), ce_lb_gauss=gz["ce_lb"], ce_ub_gauss=gz["ce_ub"],
                                         rounds=gz["rounds"]),
            "certified plan under the seed's Gaussian (second-cumulant) per-well treatment, scored under the exact model (time includes rn)")

    stage("rn_size", rn_size, needs=("ce",))
    stage("rn_tilt1", lambda: tilted("rn_tilt1", "rn", "pi-tilted risk-neutral MIP, one fixed-point iteration from the rn plan (no oracle; time includes rn)"),
          needs=("rn",))
    stage("rn_tilt2", lambda: tilted("rn_tilt2", "rn_tilt1", "pi-tilted risk-neutral MIP, two fixed-point iterations from the rn plan "
                                     "(no oracle; time includes rn and rn_tilt1)"), needs=("rn_tilt1",))
    stage("rn_tilt", lambda: tilted("rn_tilt", "ce", "pi-tilted risk-neutral MIP at the CE plan's softmax weights (oracle: the linear part "
                                    "of the final OA cut, without its exposure and CGF terms; time includes ce)"), needs=("ce",))
    stage("ce_gauss", ce_gauss)

    # ---- 6. hybrids: first stage fixed, recourse exact; the mean-value first stages;
    #         the CE first stage with RN recourse -------------------------------------------------------------------
    def exact_recourse(src):
        with Clock() as c:
            r = solve_recourse_exact(M, P[src], **OA)
        add(f"{src}+exact_recourse", r["plan"], cert_extra(r, c.t, seed_time=R[src]["time"]), f"first stage of {src}, recourse re-optimized to a certificate")

    for src in ("heur_ce", "pso", "rn_deck", "rn"):
        if src == "pso" and args.pso_seeds <= 0:                         # pso is absent by design with --pso_seeds 0
            continue
        stage(f"{src}+exact_recourse", lambda src=src: exact_recourse(src), needs=(src,))

    def mv():
        with Clock() as c:
            Mm = mean_value_model(I, lam)
            r = solve_risk_neutral(Mm)
        ctx.update(Mm=Mm, mv=r, mv_fs=first_stage_into(M, r["plan"]), t_mv=c.t)
        run.setdefault("mean_value", {})["rn"] = dict(obj=r["obj"], bound=r["bound"], mip_gap=r["gap"], time=c.t)
        print(f"  mean-value plan (theta = P = 1, risk-neutral): objective {r['obj']:.2f}  wells(yr0) {int(r['plan']['z0'] @ M.wells[M.idx0])}  "
              f"pkgs {int(r['plan']['y'].sum())}  {c.t:.1f}s", flush=True)

    def mv_exact():
        with Clock() as c:
            r = solve_recourse_exact(M, ctx["mv_fs"], **OA)
        add("mv+exact_recourse", r["plan"], cert_extra(r, c.t, seed_time=ctx["t_mv"]),
            "deterministic mean-value first stage (theta = P = 1, risk-neutral), certified CE recourse (the v3.1 EEV rung; mixes the "
            "stochastic model with the risk model)")

    def mv_rn():
        with Clock() as c:
            r = solve_risk_neutral(M, fixed=ctx["mv_fs"])
        add("mv_rn", r["plan"], dict(time=c.t + ctx["t_mv"], solver_time=c.t, seed_time=ctx["t_mv"], mip_gap=r["gap"], bound=r["bound"], obj=r["obj"]),
            "mean-value first stage (theta = P = 1, risk-neutral) with risk-neutral recourse on the tree: the rolling deterministic "
            "re-solve (time includes the mean-value solve)")

    def mv_ce():
        Mm = ctx["Mm"] if "Mm" in ctx else mean_value_model(I, lam)
        with Clock() as c:
            r = solve_ce(Mm, seeds=(ctx["mv"]["plan"],) if "mv" in ctx else (), log=quiet, **OA)
        ctx.update(mvce_fs=first_stage_into(M, r["plan"]), t_mvce=c.t + ctx.get("t_mv", 0.0))
        run.setdefault("mean_value", {})["ce"] = dict(ce_lb=r["ce_lb"], ce_ub=r["ce_ub"], rounds=r["rounds"], time=c.t)
        print(f"  mean-value plan (theta = P = 1, CE at lambda): certified [{r['ce_lb']:.2f}, {r['ce_ub']:.2f}]  "
              f"wells(yr0) {int(r['plan']['z0'] @ M.wells[M.idx0])}  pkgs {int(r['plan']['y'].sum())}  {c.t:.1f}s", flush=True)

    def mv_ce_exact():
        with Clock() as c:
            r = solve_recourse_exact(M, ctx["mvce_fs"], **OA)
        add("mv_ce+exact_recourse", r["plan"], cert_extra(r, c.t, seed_time=ctx["t_mvce"]),
            "CE mean-value first stage (solve_ce at theta = P = 1 and the same lambda), certified CE recourse on the tree "
            "(time includes the mean-value solves)")

    def ce_fs_rn():
        with Clock() as c:
            r = solve_risk_neutral(M, fixed=P["ce"])
        add("ce_fs+rn_recourse", r["plan"], dict(time=c.t + R["ce"]["time"], mip_gap=r["gap"], bound=r["bound"], obj=r["obj"]),
            "the CE plan's first stage with risk-neutral recourse")

    stage("mv", mv)
    stage("mv+exact_recourse", mv_exact, needs=("mv_fs",))
    stage("mv_rn", mv_rn, needs=("mv_fs",))
    stage("mv_ce", mv_ce)
    stage("mv_ce+exact_recourse", mv_ce_exact, needs=("mvce_fs",))
    stage("ce_fs+rn_recourse", ce_fs_rn, needs=("ce",))

    # ---- 7. information ladder with brackets and the clean VSS ---------------------------------------------
    def committed():
        with Clock() as c:
            cc = solve_ce(M, seeds=(solve_risk_neutral(M, committed=True)["plan"],), committed=True, log=quiet, **OA)
        add("ce_committed", cc["plan"], cert_extra(cc, c.t), "certified CE plan with one recourse shared by all nodes (committed)")

    def pi_rung():
        with Clock() as c:
            ctx["pi"] = perfect_information_bound(M, P["ce"], **OA)
        ctx["t_pi"] = c.t

    def ladder():
        br = lambda n: (R[n]["ce_lb"], R[n]["ce_ub"]) if n in R else None  # noqa: E731
        rnv = lambda n: (R[n]["enpv"], R[n]["bound"]) if n in R else None  # noqa: E731
        inc = [(R[n]["ce"], n) for n in ("ce", "ce_cold", "ce_1round", "ce_gauss") if n in R]
        best = max(inc) if inc else None
        L = run["ladder"] = ladder_block(br("ce"), br("ce_committed"), ctx.get("pi"), br("mv+exact_recourse"), rnv("rn"), rnv("mv_rn"),
                                         br("mv_ce+exact_recourse"), (best[0], run.get("ce_ub_best", NAN), best[1]) if best else None,
                                         ctx.get("t_pi", NAN))
        print(f"  ladder: committed [{L['committed_lb']:.2f}, {L['committed_ub']:.2f}] <= two-stage [{L['two_stage_lb']:.2f}, {L['two_stage_ub']:.2f}] <= "
              f"perfect information [{L['pi_lb']:.2f}, {L['pi_ub']:.2f}]; value of recourse {L['value_of_recourse']:.2f}, "
              f"EVPI {L['evpi']:.2f}, EEV {L['eev']:.2f}, VSS {L['vss']:.2f}", flush=True)
        print(f"  VSS: RN-VSS {L['rn_vss']:.2f} [{L['rn_vss_lb']:.2f}, {L['rn_vss_ub']:.2f}] (E[NPV] rn - mv_rn); "
              f"CE-VSS {L['ce_vss']:.2f} [{L['ce_vss_lb']:.2f}, {L['ce_vss_ub']:.2f}] (CE ce - mv_ce+exact_recourse); "
              f"mixed {L['vss_mixed']:.2f} [{L['vss_mixed_lb']:.2f}, {L['vss_mixed_ub']:.2f}] (CE ce - mv+exact_recourse)", flush=True)

    stage("ce_committed", committed)
    stage("pi", pi_rung, needs=("ce",))
    stage("ladder", ladder)                     # no needs: missing rungs give NaN, so the RN-VSS survives a failed ce stage

    # ---- 8. Beck's statistic and the overlap with the CE plan ----------------------------------------------------
    def beck():
        e_rn, e_ce, e_dk, e_t2 = R["rn"], R["ce"], R["rn_deck"], R["rn_tilt2"]
        den = max(1e-9, abs(e_rn["enpv"]))
        run["beck"] = dict(lam_sigma_rn=e_rn["lam_sigma"], lam_sigma_ce=e_ce["lam_sigma"], enpv_rn=e_rn["enpv"], ce_rn=e_rn["ce"],
                           enpv_ce=e_ce["enpv"], ce_ce=e_ce["ce"], enpv_cost=e_rn["enpv"] - e_ce["enpv"], enpv_cost_pct=100 * (e_rn["enpv"] - e_ce["enpv"]) / den,
                           ce_gain=e_ce["ce"] - e_rn["ce"], ce_gain_pct=100 * (e_ce["ce"] - e_rn["ce"]) / den,
                           ce_gain_pct_ce_rn=100 * (e_ce["ce"] - e_rn["ce"]) / max(1e-9, abs(e_rn["ce"])),
                           gain_vs_deck=e_ce["ce"] - e_dk["ce"], gain_vs_deck_pct=100 * (e_ce["ce"] - e_dk["ce"]) / den,
                           gain_vs_tilt2=e_ce["ce"] - e_t2["ce"], gain_vs_tilt2_pct=100 * (e_ce["ce"] - e_t2["ce"]) / den,
                           screen=run["screen"]["value"], screen_pct=run["screen"]["pct_enpv_rn"], mip_gap_pct=e_ce["gap_pct"],
                           lognormal_shift_ce=e_ce["ce_mc"] - e_ce["ce"], lognormal_shift_rn=e_rn["ce_mc"] - e_rn["ce"], lognormal_shift_deck=e_dk["ce_mc"] - e_dk["ce"])
        b = run["beck"]
        print(f"  Beck: E[NPV] cost {b['enpv_cost']:.2f} ({b['enpv_cost_pct']:.1f}% of E[NPV](rn)), CE gain {b['ce_gain']:.2f} ({b['ce_gain_pct']:.1f}%), "
              f"vs best deck {b['gain_vs_deck']:.2f} ({b['gain_vs_deck_pct']:.1f}%), vs tilt2 {b['gain_vs_tilt2']:.2f}; lam*sigma_W(rn) {b['lam_sigma_rn']:.2f}; "
              f"lognormal shift ce {b['lognormal_shift_ce']:+.2f} rn {b['lognormal_shift_rn']:+.2f}", flush=True)

    def overlap():
        ref0, refa = R["ce"]["placement"]["dev_ub_year0"], R["ce"]["placement"]["dev_ub_any"]
        for row in R.values():
            row["placement"]["jaccard_vs_ce_year0"] = jaccard(row["placement"]["dev_ub_year0"], ref0)
            row["placement"]["jaccard_vs_ce_any"] = jaccard(row["placement"]["dev_ub_any"], refa)

    stage("beck", beck, needs=("rn", "ce", "rn_deck", "rn_tilt2"))
    stage("jaccard", overlap, needs=("ce",))

    # ---- 9. fine-tree out-of-sample test, perturbed-truth paired scoring, re-optimized perturbed truths
    if args.fine > 0:
        def fine():
            F = run["fine"] = dict(n_nodes=args.fine, methods={})
            try:
                fine_tree_test(I, M, lam, P, args.fine, F, stage.checkpoint, run["errors"])
            except Exception as e:
                F["error"] = repr(e)
                raise
            f, v = F["methods"], F["vss"]
            print(f"  fine-tree ({F['K']} nodes) certified CE: " + "  ".join(f"{n} {f[n]['ce_lb']:.2f}" for n in ("rn", "rn_deck", "rn_tilt2", "heur_ce", "pso", "ce_cold", "ce") if n in f), flush=True)
            print("  fine-tree VSS: " + "; ".join(f"{k} {v[k]:.2f} [{v[k + '_lb']:.2f}, {v[k + '_ub']:.2f}]" for k in ("rn_vss", "ce_vss", "vss_mixed") if k in v), flush=True)

        stage("fine", fine)

    def perturbed():
        try:
            with Clock() as c:
                run["perturbed"] = perturbed_truth(I, M, lam, P, run["errors"])
            run["perturbed_time"] = c.t
        except Exception as e:
            run["perturbed"] = dict(error=repr(e))
            raise

    stage("perturbed", perturbed)

    if args.perturb_reopt:
        def reopt():
            O = run["perturbed_reopt"] = {}
            try:
                perturbed_reopt(I, M, lam, P, args.perturb_methods, O, stage.checkpoint, run["errors"],
                                enpv_zN=R["rn"]["enpv"] if "rn" in R else NAN)
            except Exception as e:
                O["error"] = repr(e)
                raise
            sm = O["summary"]
            print(f"  re-optimized truths: z_A beats the best family member on CE in {sm['beats_ce']}/{sm['n_truths']} truths (loses in "
                  f"{sm['loses_ce']}), on E[NPV] in {sm['beats_enpv']}/{sm['n_truths']} (loses in {sm['loses_enpv']}); "
                  f"gain on CE {sm['min_gain_ce_pct']:+.2f}% to {sm['max_gain_ce_pct']:+.2f}% of E(z_N)", flush=True)

        stage("perturbed_reopt", reopt)
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nx", type=int); ap.add_argument("ny", type=int)
    ap.add_argument("seed_from", type=int); ap.add_argument("seed_to", type=int)
    ap.add_argument("rho_list"); ap.add_argument("out")
    ap.add_argument("--budget", type=float, default=60.0, help="PSO/GA time budget per seed, s; floored at ce_cold's wall-clock per instance")
    ap.add_argument("--long_mult", type=float, default=5.0, help="pso_long's budget as a multiple of the per-seed budget")
    ap.add_argument("--fine", type=int, default=5, help="fine tree: n x n nodes for the out-of-sample recourse test (0 skips)")
    ap.add_argument("--pso_seeds", type=int, default=2, help="PSO/GA seeds per instance on the CE objective")
    ap.add_argument("--ev_seeds", type=int, default=1, help="PSO/GA seeds per instance on the E[NPV] objective (0 skips)")
    ap.add_argument("--perturb_reopt", action=argparse.BooleanOptionalAction, default=True,
                    help="re-optimize the recourse of the --perturb_methods first stages under every perturbed truth "
                         "(on by default; --no-perturb_reopt skips it)")
    ap.add_argument("--perturb_methods", type=lambda v: [x.strip() for x in v.split(",") if x.strip()], default=list(PERTURB_METHODS),
                    help="comma-separated methods for --perturb_reopt (default " + ",".join(PERTURB_METHODS) + "); missing ones are "
                         "skipped, and with ce listed the deck family rn_deck, rn_tilt1, rn_tilt2 is completed")
    args = ap.parse_args()
    rhos = [float(v) for v in args.rho_list.split(",")]
    res = dict(version=VERSION, nx=args.nx, ny=args.ny, args=vars(args), rhos=rhos, started=time.strftime("%Y-%m-%d %H:%M:%S"),
               threads=dict(gurobi=int(os.environ.get("GRB_THREADS", 2)), omp=os.environ.get("OMP_NUM_THREADS")),
               gurobi_memlimit_gb=float(os.environ.get("GRB_MEMLIMIT", 1.2)), rss_cap_gb=RSS_CAP_GB, decks=DECKS, oa=OA, oa_reopt=OA_REOPT,
               mc_draws=MC_DRAWS, family=list(FAMILY), z_A=Z_A, fine_rn=list(FINE_RN),
               perturbations=[dict(label=l, **kw) for l, kw in PERTURBATIONS], runs=[], errors=[])
    dump(res, args.out)                                                  # fail fast on an unwritable output path
    for s in range(args.seed_from, args.seed_to):
        knobs = knobs_for_seed(s)
        try:
            I = make_instance(args.nx, args.ny, seed=s, **knobs)
        except Exception as e:
            traceback.print_exc()
            res["errors"] += [dict(seed=s, rho=rho, error=repr(e), time=0.0) for rho in rhos]
            dump(res, args.out)
            continue
        for rho in rhos:
            t0 = time.time()
            run = dict(seed=s, rho=rho, version=VERSION, status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"), errors=[], stages={})
            res["runs"].append(run)
            try:
                bench_one(I, s, knobs, rho, args, run, lambda: dump(res, args.out))
                n_err = len(run["errors"])
                run["status"] = "done" if n_err == 0 else f"done with {n_err} stage error(s)"
            except Exception as e:                                      # only what escapes the stages; the guard's SystemExit passes through
                traceback.print_exc()
                run["status"] = "failed"
                run["errors"].append(error_entry("run", e, time.time() - t0))
                res["errors"].append(dict(seed=s, rho=rho, error=repr(e), time=time.time() - t0))
            run["time_total"] = time.time() - t0
            print(f"  seed {s} rho {rho:.0f} done in {run['time_total']:.0f}s ({run['status']})", flush=True)
            dump(res, args.out)
    print("done", flush=True)
