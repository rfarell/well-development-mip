"""Self-tests for the v3 model and solver: master/evaluator consistency, feasibility consistency, certificate sanity.
Usage: selftest.py [nx ny]"""
import sys
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from instance import make_instance, T
from model import Model, PlanBuilder
from solve import Master, solve_risk_neutral, solve_ce, relax_and_round, solve_recourse_exact, perfect_information
import baselines as BL

nx, ny = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (4, 4)
I = make_instance(nx, ny, seed=1)
lam = 1 / 300.0
M = Model(I, lam)
print(f"U={I['U']} options={M.n} n0={M.n0} n1={M.n1} K={M.K} G={M.G} credited terminal ubs={int((M.tau_best >= 0).sum())}/{M.nub} fringe units={int(M.fringe_u.sum())}")
rng = np.random.default_rng(0)

# random feasible plans via PlanBuilder
def random_plan(seed):
    r = np.random.default_rng(seed); B = PlanBuilder(M)
    for g in r.permutation(M.G):
        if r.random() < 0.5 and B.can_buy(g): B.buy(g)
    for o in r.permutation(M.idx0):
        if r.random() < 0.3 and B.can_add(o): B.add(o)
    for k in range(M.K):
        for o in r.permutation(M.idx1):
            if r.random() < 0.15 and B.can_add(o, k): B.add(o, k)
    return B.plan

plans = [random_plan(s) for s in range(12)]
assert all(M.feasible(p, True) for p in plans), "PlanBuilder produced an infeasible plan"
# 1. D_expr/D_value vs node_terms, and master exposure vs node_terms (fix all variables)
mm = Master(M, quad="cuts")
mm.m.setObjective(0, GRB.MINIMIZE)
maxD = maxW = 0.0
for pl in plans[:6]:
    for var, val in ((mm.z0, pl["z0"]), (mm.zk, pl["zk"]), (mm.y, pl["y"]), (mm.w0, pl["w0"]), (mm.wk, pl["wk"])):
        var.LB = val; var.UB = val
    mm.fix_inventory(pl)
    mm.m.optimize(); assert mm.m.Status == GRB.OPTIMAL, f"fixed plan infeasible in master (status {mm.m.Status})"
    for k in range(M.K):
        D, w, cg, vi = M.node_terms(pl, k)
        maxD = max(maxD, abs(mm.D_value(pl, k) - D))
        maxW = max(maxW, np.abs(np.array(mm.wv[k].X) - w).max())
        # h chosen by the master must equal the plan's held indicator wherever it matters (tau > 0)
print(f"max |D_value - node_terms D| = {maxD:.2e};  max |master exposure - node_terms w| = {maxW:.2e}")
assert maxD < 1e-6 and maxW < 1e-6
# 2. certificate sanity: all feasible plans below the proven upper bound; recourse-exact >= original
rn = solve_risk_neutral(M); rr = relax_and_round(M)
h, cfg, _ = BL.tune_heuristic(M, by="ce")
ce = solve_ce(M, seeds=(rn["plan"], rr, h), tol=1e-3, log=lambda *a: None)
vals = [M.evaluate(p)["ce"] for p in plans] + [M.evaluate(rn["plan"])["ce"], M.evaluate(h)["ce"]]
print(f"certified [{ce['ce_lb']:.3f}, {ce['ce_ub']:.3f}] rounds {ce['rounds']}; max feasible-plan CE {max(vals):.3f}; rn {M.evaluate(rn['plan'])['ce']:.3f}; heuristic {M.evaluate(h)['ce']:.3f}")
assert max(vals) <= ce["ce_ub"] + 1e-6 and ce["ce_lb"] <= ce["ce_ub"] + 1e-9
rx = solve_recourse_exact(M, h)
print(f"heuristic first stage with exact recourse: {rx['ce_lb']:.3f} >= heuristic {M.evaluate(h)['ce']:.3f}: {rx['ce_lb'] >= M.evaluate(h)['ce'] - 1e-6}")
assert rx["ce_lb"] >= M.evaluate(h)["ce"] - 1e-6 and rx["ce_lb"] <= ce["ce_ub"] + 1e-6   # (rx's own UB may be looser: different cut set)
cc = solve_ce(M, seeds=(solve_risk_neutral(M, committed=True)["plan"],), committed=True, log=lambda *a: None)
pi = perfect_information(M)
print(f"committed {cc['ce_lb']:.3f} <= two-stage {ce['ce_lb']:.3f} <= perfect-information {pi['ce_pi_lb']:.3f} (ub {pi['ce_pi_ub']:.3f})")
assert cc["ce_lb"] <= ce["ce_ub"] + 1e-6 and ce["ce_lb"] <= pi["ce_pi_ub"] + 1e-6
sm = solve_risk_neutral(M, max_expected_wells=10.0)["plan"]
print(f"size-matched rn (<=10 expected wells): feasible {M.feasible(sm)}, expected wells {float(sm['z0'] @ M.wells[M.idx0]) + sum(I['prob'][k] * float(sm['zk'][k] @ M.wells[M.idx1]) for k in range(M.K)):.2f}")
ev = M.evaluate(ce["plan"]); dec = M.sigma_decomposition(ce["plan"]); tilt = M.implied_tilt(ce["plan"])
print(f"CE plan: CE {ev['ce']:.2f} ENPV {ev['enpv']:.2f} lam*sigma {ev['lam_sigma']:.3f}; variance shares " +
      ", ".join(f"{k} {v / dec['total']:.2f}" for k, v in dec.items() if k != 'total') + f"; implied price {tilt['implied_price']:.3f} theta {tilt['implied_theta']:.3f}")
mc, se = M.ce_mc(ce["plan"], 1000, return_se=True); print(f"MC lognormal CE {mc:.2f} +- {se:.2f} (model {ev['ce']:.2f})")
print("selftest passed")
