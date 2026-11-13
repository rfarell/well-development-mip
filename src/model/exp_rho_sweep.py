"""Risk-tolerance sweep on one instance: placement across rho (no-regret / never-drill unit-benches),
the preference regime (lambda*sigma_W), Beck's statistic (expected-NPV cost) beside the CE gain, and the
value of recourse (two-stage plan vs a committed plan under the same scenario tree).

Usage: exp_rho_sweep.py nx ny seed out.json [rho list]
"""
import json
import resource
import sys
import time

import numpy as np

from instance import make_instance, NB, T
from model import Model
from solve import solve_ce, solve_risk_neutral, relax_and_round
import baselines as BL

if __name__ == "__main__":
    nx, ny, seed, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    rhos = [float(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 else [100, 200, 300, 500, 1000, 3000, 10000]
    I = make_instance(nx, ny, seed=seed)
    M0 = Model(I, 1.0 / rhos[-1])
    rn = solve_risk_neutral(M0)["plan"]                      # the risk-neutral plan is rho-independent
    from model import MODEL_VERSION
    import solve as _SV
    res = dict(nx=nx, ny=ny, seed=seed, U=int(I["U"]), model_version=MODEL_VERSION, solver_rev=_SV.SOLVER_REV, rows=[])
    dev_sets = {}
    for rho in rhos:
        lam = 1.0 / rho
        M = Model(I, lam)
        t0 = time.time()
        h, cfg, _ = BL.tune_heuristic(M, by="ce")
        rr = relax_and_round(M)
        ce = solve_ce(M, seeds=(rn, rr, h), tol=1e-3, max_iter=25, log=lambda *a: None)
        cc = solve_ce(M, seeds=(solve_risk_neutral(M, committed=True)["plan"],), tol=1e-3, max_iter=25, log=lambda *a: None, committed=True)
        pl = ce["plan"]
        e, ern, ecc = M.evaluate(pl), M.evaluate(rn), M.evaluate(cc["plan"])
        p = I["prob"]
        dev0 = set(M.o_ub[M.idx0][pl["z0"] > 0.5].tolist())
        dev_any = set()
        for k in range(M.K):
            dev_any |= set(M.o_ub[M.full_z(pl, k) > 0.5].tolist())
        dev_sets[rho] = (dev0, dev_any)
        by_year = np.zeros(T); by_bench = np.zeros(NB); by_len = np.zeros(3)
        for o in M.idx0[pl["z0"] > 0.5]:
            by_year[0] += M.o_k[o]; by_bench[M.o_b[o]] += M.o_k[o]; by_len[M.o_len[o] - 2] += M.o_k[o]
        for k in range(M.K):
            for o in M.idx1[pl["zk"][k] > 0.5]:
                by_year[M.o_t[o]] += p[k] * M.o_k[o]; by_bench[M.o_b[o]] += p[k] * M.o_k[o]; by_len[M.o_len[o] - 2] += p[k] * M.o_k[o]
        row = dict(rho=rho, ce=e["ce"], ce_ub=ce["ce_ub"], enpv=e["enpv"], lam_sigma=e["lam_sigma"], sigma=e["sigma"],
                   rn_ce=ern["ce"], rn_enpv=ern["enpv"], rn_lam_sigma=ern["lam_sigma"],
                   ce_gain=e["ce"] - ern["ce"], enpv_cost=ern["enpv"] - e["enpv"],
                   committed_ce=ecc["ce"], committed_ub=cc["ce_ub"], value_of_recourse=e["ce"] - ecc["ce"],
                   heuristic_ce=M.evaluate(h)["ce"], wells_by_year=by_year.tolist(), wells_by_bench=by_bench.tolist(),
                   wells_by_len=by_len.tolist(), packages=int(pl["y"].sum()), dev_year0=sorted(dev0), dev_any=sorted(dev_any),
                   rounds=ce["rounds"], time=time.time() - t0)
        print(f"rho {rho:6.0f}: CE {e['ce']:7.2f} (ub {ce['ce_ub']:7.2f})  E[NPV] {e['enpv']:7.2f}  lam*sig {e['lam_sigma']:.2f}  "
              f"gain vs rn {row['ce_gain']:6.2f}  enpv cost {row['enpv_cost']:6.2f}  recourse value {row['value_of_recourse']:6.2f}  "
              f"wells/yr {np.round(by_year, 1).tolist()}  TF share {by_bench[1] / max(1e-9, by_bench.sum()):.2f}  3-4mi {by_len[1:].sum() / max(1e-9, by_len.sum()):.2f}  "
              f"pkgs {row['packages']}  {row['time']:.0f}s  RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB", flush=True)
        res["rows"].append(row)
        json.dump(res, open(out, "w"), indent=1, default=float)
    # no-regret and never-drill unit-benches across the sweep (year-0 placement)
    sets0 = [dev_sets[r][0] for r in rhos]
    core = set.intersection(*sets0); ever = set.union(*sets0)
    res["no_regret_year0"] = sorted(core); res["never_drill_year0"] = sorted(set(range(M0.nub)) - ever)
    res["undecided_year0"] = sorted(ever - core)
    print(f"year-0 placement across rho: no-regret {len(core)}, undecided {len(ever - core)}, never-drill {M0.nub - len(ever)} of {M0.nub} unit-benches", flush=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
    print("done", flush=True)
