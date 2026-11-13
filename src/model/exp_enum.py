"""Independent enumeration check of the certified solver on a tiny instance (added 2026-10-04).

The instance has one unit (two unit-benches, no neighbor, so only two-mile options), the 3x3 tree and capacities set so
that several options fit each year. Every complete plan is enumerated without the solver: a first stage is a choice per
unit-bench of a year-0 option or none plus the pad decision (built in year 0 even with no well, which the master permits,
or not), and, because the objective -(1/lambda) log sum_k p_k exp(-lambda CE_k) increases in every node's own term, the best
recourse of each node given the first stage is found by enumerating that node's recourse alone (per unit-bench a later-year
option or none, pad year = the first well's year). Feasibility of every candidate is Model.feasible (lanes, capacities,
drill-to-hold, pads), the objective is Model.evaluate (exact), and nothing from solve.py is used in the enumeration. The
enumerated optimum is then compared with solve_ce's bracket and, for expected NPV, with the risk-neutral MILP.

Usage: exp_enum.py out.json [--rho 20] [--seed 100]
"""
import argparse
import itertools
import json
import time

import numpy as np

from instance import make_instance, NB, T
from model import Model, MODEL_VERSION
from solve import solve_ce, solve_risk_neutral
import baselines as BL
import bench_v3 as B

BASE = dict(own_frac=0.65, deadline_frac=0.25, parent_frac=0.35, rig_mult=2.5, range_mi=3.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out"); ap.add_argument("--rho", type=float, default=20.0); ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()
    I = make_instance(1, 2, seed=args.seed, **BASE)
    # capacities that admit two or three two-mile options per year (the generator's capacities for one unit admit none)
    I["cap"] = dict(rig=np.full(T, 160.0), crew=np.full(T, 14.0), capital=np.full(T, 60.0), acq=0.0)
    I["owned"] = np.array([True]); I["pkg_of"] = np.array([-1]); I["pkg_cost"] = np.zeros(0)   # the one unit is owned, no package
    I["deadline"][:] = 1                                                    # the unit is fringe: a well by year 1 holds it
    lam = 1.0 / args.rho
    M = Model(I, lam)
    assert M.U == 1 and M.G == 0, "the check expects one owned unit and no package"
    print(f"enum instance: options {M.n} (year 0: {M.n0}), K {M.K}, credited unit-benches {int((M.tau_best >= 0).sum())}, fringe {int(M.fringe_u.sum())}", flush=True)
    opts0 = {ub: M.idx0[M.o_ub[M.idx0] == ub].tolist() + [None] for ub in range(M.nub)}
    opts1 = {ub: M.idx1[M.o_ub[M.idx1] == ub].tolist() + [None] for ub in range(M.nub)}
    t0 = time.time()
    n_first, n_full, best, best_rn = 0, 0, (-np.inf, None), (-np.inf, None)
    for w0 in (0, 1):
        for choice in itertools.product(*[opts0[ub] for ub in range(M.nub)]):
            plan = M.empty_plan()
            for o in choice:
                if o is not None:
                    plan["z0"][M.pos0[o]] = 1.0
            if any(o is not None for o in choice) and w0 == 0:              # a year-0 well needs the pad in year 0
                continue
            plan["w0"][0] = float(w0)
            if not M.feasible(plan):
                continue
            n_first += 1
            # per node: enumerate this node's recourse (the first stage fixed), keep the node's best exact term
            node_best = []
            for k in range(M.K):
                free = [ub for ub in range(M.nub) if choice[ub] is None]
                cands = []
                for rec in itertools.product(*[opts1[ub] for ub in free]):
                    pk = dict(z0=plan["z0"], y=plan["y"], w0=plan["w0"], zk=np.zeros((M.K, M.n1)), wk=np.zeros((M.K, M.U, T - 1)))
                    years = [M.o_t[o] for o in rec if o is not None]
                    if years and w0 == 0:
                        pk["wk"][k, 0, min(years) - 1] = 1.0
                    for o in rec:
                        if o is not None:
                            pk["zk"][k, M.pos1[o]] = 1.0
                    # feasibility of node k alone: evaluate a plan whose other nodes copy node k (feasible() checks every node)
                    pkk = dict(pk); pkk["zk"] = np.repeat(pk["zk"][[k]], M.K, 0); pkk["wk"] = np.repeat(pk["wk"][[k]], M.K, 0)
                    if not M.feasible(pkk):
                        continue
                    D, w, cg, vi = M.node_terms(pkk, k)
                    q = -lam * D + 0.5 * lam**2 * float(w @ M.Sigma @ w) + cg
                    cands.append((q, D, pk["zk"][k].copy(), pk["wk"][k].copy()))
                    n_full += 1
                node_best.append((min(cands, key=lambda c: c[0]), max(cands, key=lambda c: c[1])))
            # assemble the CE-optimal and the E[NPV]-optimal complete plans for this first stage
            pce, prn = dict(plan), dict(plan)
            pce["zk"] = np.stack([nb[0][2] for nb in node_best]); pce["wk"] = np.stack([nb[0][3] for nb in node_best])
            prn["zk"] = np.stack([nb[1][2] for nb in node_best]); prn["wk"] = np.stack([nb[1][3] for nb in node_best])
            assert M.feasible(pce) and M.feasible(prn)
            ce = M.evaluate(pce)["ce"]; en = M.evaluate(prn)["enpv"]
            if ce > best[0]:
                best = (ce, B.sparse_plan(pce))
            if en > best_rn[0]:
                best_rn = (en, B.sparse_plan(prn))
    t_enum = time.time() - t0
    print(f"enumerated {n_first} first stages, {n_full} node recourses, {t_enum:.0f}s: best CE {best[0]:.6f}, best E[NPV] {best_rn[0]:.6f}", flush=True)
    rn = solve_risk_neutral(M, mip_gap=1e-7)
    h, _, _ = BL.tune_heuristic(M, by="ce")
    t1 = time.time()
    ce = solve_ce(M, seeds=(rn["plan"], h), tol=1e-6, max_iter=60, time_limit=300, mip_gap=1e-7, log=B.quiet)
    import solve as _SV
    res = dict(model_version=MODEL_VERSION, solver_rev=_SV.SOLVER_REV, rho=args.rho, seed=args.seed, n_first_stages=n_first, n_node_recourses=n_full, time_enum=t_enum,
               enum_ce=best[0], enum_enpv=best_rn[0], enum_plan=best[1], milp=dict(obj=rn["obj"], bound=rn["bound"], gap=rn["gap"]),
               solve_ce=dict(ce_lb=ce["ce_lb"], ce_ub=ce["ce_ub"], rounds=ce["rounds"], time=time.time() - t1, bound_valid=ce["bound_valid"]),
               checks=dict(solver_lb_matches_enum=bool(abs(ce["ce_lb"] - best[0]) <= 1e-5 * max(1.0, abs(best[0]))),
                           enum_within_bracket=bool(best[0] <= ce["ce_ub"] + 1e-6 * max(1.0, abs(best[0]))),
                           milp_matches_enum=bool(abs(rn["obj"] - best_rn[0]) <= 1e-5 * max(1.0, abs(best_rn[0])))))
    print(f"solver: CE [{ce['ce_lb']:.6f}, {ce['ce_ub']:.6f}] in {ce['rounds']} rounds; MILP E[NPV] {rn['obj']:.6f} (bound {rn['bound']:.6f}); checks {res['checks']}", flush=True)
    B.dump(res, args.out)


if __name__ == "__main__":
    main()
