"""Scale, rank, node-count, master-type and enumeration studies of the certified solver.

  scale   U in {40, 100, 160} units (8x10, 10x20, 16x20 sections), base knobs, seed 100: tuned rule, risk-neutral MIP
          (time, gap), certified CE solve seeded with both (rounds, time per round, bracket, peak RSS). rho 100 and 300
          at 40 and 100 units, rho 100 at 160 units. A run that exceeds Gurobi's MemLimit is recorded as such.
  rank    40 units, rho 100: solve_ce with the Loewner-safe factor truncated to rank r in {4, 8, 16, 32, full}; the
          bracket stays valid at every rank (the master relaxes the risk; the incumbent is scored exactly).
  nodes   40 units, rho 100: trees of 3x3, 4x4 and 5x5 nodes; certificate time and bracket; the 3x3 first stage's
          overlap with the finer trees' first stages (Jaccard, year-0 unit-benches), and every first stage scored on the
          5x5 tree with certified recourse (a discretization test of the first stage).
  master  40 units, rho 100: linear tangent cuts ('cuts') against the second-order-cone master ('soc').
  tol     the four v3.2 instances (seeds 0-3, joint knobs) at rho 100: time to a 1e-4 certificate from the rn seed.
  enum    2x2 sections (2 units, no neighbor), 3x3 tree, rho 100: every first stage (at most one year-0 option per
          unit-bench, pads where year-0 wells are, every package choice) with certified recourse at tol 1e-5; the best is
          the exact optimum up to that tolerance; compared with solve_ce's bracket at tol 1e-5. The same for the
          risk-neutral MILP against an enumeration with risk-neutral recourse.

Usage: exp_scale.py out.json [--parts scale,rank,nodes,master,tol,enum] [--big 16,20]
Environment as bench_v3.py. Every part runs in its own try/except and the JSON is rewritten after each item.
"""
import argparse
import itertools
import os
import resource
import sys
import time
import traceback

import numpy as np
import gurobipy as gp

from instance import make_instance, perturb, NB, T
from model import Model, PlanBuilder
from solve import solve_ce, solve_risk_neutral, solve_recourse_exact
import baselines as BL
import bench_v3 as B

BASE = dict(own_frac=0.65, deadline_frac=0.25, parent_frac=0.35, rig_mult=2.5, range_mi=3.0)
OUT = {}
PATH = None


def save():
    B.dump(OUT, PATH)


def peak():
    return B.rss_gb()


def dev0(M, plan):
    return sorted(set(M.o_ub[M.idx0][plan["z0"] > 0.5].tolist()))


def guarded(fn, *a, **k):
    """Run fn; return (result, None) or (None, error string). Gurobi MemLimit and the RSS cap are recorded, not fatal."""
    try:
        return fn(*a, **k), None
    except (gp.GurobiError, MemoryError) as e:
        traceback.print_exc(); return None, f"{type(e).__name__}: {e}"
    except Exception as e:
        traceback.print_exc(); return None, repr(e)


# ------------------------------------------------------------------------------------------------------------------
def part_scale(big):
    rows = OUT.setdefault("scale", [])
    cases = [(8, 10, 100.0), (8, 10, 300.0), (10, 20, 100.0), (10, 20, 300.0), (big[0], big[1], 100.0)]
    for nx, ny, rho in cases:
        I = make_instance(nx, ny, seed=100, **BASE)
        lam = 1.0 / rho
        row = dict(nx=nx, ny=ny, U=int(I["U"]), rho=rho, started=time.strftime("%H:%M:%S"))
        rows.append(row); save()
        try:
            M = Model(I, lam)
            row.update(options=M.n, n0=M.n0, n1=M.n1, K=M.K, G=M.G, binaries=M.n0 + M.K * M.n1, nub=M.nub)
            with B.Clock() as c:
                h, cfg, _ = BL.tune_heuristic(M, by="ce")
            row["heur"] = dict(time=c.t, ce=M.evaluate(h)["ce"], cfg=cfg)
            with B.Clock() as c:
                rn, err = guarded(solve_risk_neutral, M, time_limit=900)
            if err:
                row["rn"] = dict(error=err, time=c.t); save(); continue
            e = M.evaluate(rn["plan"])
            row["rn"] = dict(time=c.t, gap=rn["gap"], bound=rn["bound"], enpv=e["enpv"], ce=e["ce"], lam_sigma=e["lam_sigma"], peak_rss_gb=peak())
            with B.Clock() as c:
                ce, err = guarded(solve_ce, M, seeds=(rn["plan"], h), log=B.quiet, tol=1e-3, max_iter=25, time_limit=900)
            if err:
                row["ce"] = dict(error=err, time=c.t, peak_rss_gb=peak()); save(); continue
            e = M.evaluate(ce["plan"])
            row["ce"] = dict(time=c.t, rounds=ce["rounds"], ce_lb=ce["ce_lb"], ce_ub=ce["ce_ub"], rank=ce["rank"],
                             gap_pct=100 * (ce["ce_ub"] - ce["ce_lb"]) / max(1e-9, abs(ce["ce_lb"])), enpv=e["enpv"], lam_sigma=e["lam_sigma"],
                             hist=ce["hist"], wells_expected=B.expected_wells(M, ce["plan"]), peak_rss_gb=peak())
            print(f"scale U={I['U']} rho {rho:.0f}: binaries {row['binaries']}, rn {row['rn']['time']:.0f}s gap {rn['gap']:.1e}, "
                  f"ce [{ce['ce_lb']:.2f}, {ce['ce_ub']:.2f}] {ce['rounds']}r {c.t:.0f}s, peak {peak():.2f} GB", flush=True)
        except Exception as e:
            traceback.print_exc(); row["error"] = repr(e)
        save()
        if peak() > B.RSS_CAP_GB:
            row["stopped"] = f"peak RSS {peak():.2f} GB exceeds {B.RSS_CAP_GB}"; save()
            print("memory cap reached in the scale study; later cases skipped", flush=True)
            break


def part_rank():
    rows = OUT.setdefault("rank", [])
    I = make_instance(8, 10, seed=100, **BASE)
    M = Model(I, 1 / 100.0)
    rn = solve_risk_neutral(M)["plan"]; h, _, _ = BL.tune_heuristic(M, by="ce")
    ev = np.linalg.eigvalsh(M.Sigma)[::-1]
    OUT["rank_spectrum"] = dict(eigenvalues=ev.tolist(), cum_share=(np.cumsum(ev) / ev.sum()).tolist())
    for r in (4, 8, 16, 32, None):
        with B.Clock() as c:
            ce, err = guarded(solve_ce, M, seeds=(rn, h), rank=r, log=B.quiet, tol=1e-3, max_iter=25, time_limit=600)
        row = dict(rank=r if r is not None else int(M.nub), time=c.t)
        if err:
            row["error"] = err
        else:
            row.update(ce_lb=ce["ce_lb"], ce_ub=ce["ce_ub"], rounds=ce["rounds"], rank_used=ce["rank"],
                       gap_pct=100 * (ce["ce_ub"] - ce["ce_lb"]) / max(1e-9, abs(ce["ce_lb"])), hist=ce["hist"],
                       dev0=dev0(M, ce["plan"]), share_of_variance=float(ev[:ce["rank"]].sum() / ev.sum()))
            print(f"rank {row['rank']}: [{ce['ce_lb']:.2f}, {ce['ce_ub']:.2f}] {ce['rounds']}r {c.t:.0f}s  variance share {row['share_of_variance']:.3f}", flush=True)
        rows.append(row); save()


def part_nodes():
    for seed in (100, 101, 102):
        part_nodes_seed(seed)


def part_nodes_seed(seed):
    rows = OUT.setdefault("nodes", [])
    I = make_instance(8, 10, seed=seed, **BASE)
    lam = 1 / 100.0
    plans, models = {}, {}
    for n in (3, 4, 5):
        J = perturb(I, n_nodes=n)
        M = Model(J, lam)
        models[n] = M
        row = dict(seed=seed, n=n, K=M.K, binaries=M.n0 + M.K * M.n1)
        try:
            with B.Clock() as c:
                rn = solve_risk_neutral(M, time_limit=900)
            h, _, _ = BL.tune_heuristic(M, by="ce")
            row["rn"] = dict(time=c.t, gap=rn["gap"], enpv=M.evaluate(rn["plan"])["enpv"])
            with B.Clock() as c:
                ce, err = guarded(solve_ce, M, seeds=(rn["plan"], h), log=B.quiet, tol=1e-3, max_iter=25, time_limit=900)
            if err:
                row["ce"] = dict(error=err, time=c.t, peak_rss_gb=peak()); rows.append(row); save(); continue
            plans[n] = ce["plan"]
            row["ce"] = dict(time=c.t, rounds=ce["rounds"], ce_lb=ce["ce_lb"], ce_ub=ce["ce_ub"], hist=ce["hist"],
                             gap_pct=100 * (ce["ce_ub"] - ce["ce_lb"]) / max(1e-9, abs(ce["ce_lb"])), dev0=dev0(M, ce["plan"]),
                             wells_year0=float(ce["plan"]["z0"] @ M.wells[M.idx0]), wells_expected=B.expected_wells(M, ce["plan"]), peak_rss_gb=peak())
            if 3 in plans:
                row["jaccard_vs_3x3"] = B.jaccard(dev0(M, ce["plan"]), dev0(models[3], plans[3]))
            print(f"nodes {n}x{n}: binaries {row['binaries']}, ce [{ce['ce_lb']:.2f}, {ce['ce_ub']:.2f}] {ce['rounds']}r {c.t:.0f}s, "
                  f"peak {peak():.2f} GB", flush=True)
        except Exception as e:
            traceback.print_exc(); row["error"] = repr(e)
        rows.append(row); save()
        if peak() > B.RSS_CAP_GB:
            row["stopped"] = f"peak RSS {peak():.2f} GB"; save(); return
    # every first stage with certified recourse on the finest tree that solved: the regret of planning on a coarser tree
    fine = max(plans)
    MF = models[fine]
    E = solve_risk_neutral(MF, time_limit=600)["obj"] if plans else float("nan")
    FT = OUT.setdefault("nodes_fine_test", {})
    FT[str(seed)] = dict(seed=seed, fine=fine, K=MF.K, enpv_rn_fine=E, rows={})
    for n, pl in plans.items():
        try:
            with B.Clock() as c:
                r = solve_recourse_exact(MF, B.first_stage_into(MF, pl), tol=1e-3, max_iter=25, time_limit=900)
            FT[str(seed)]["rows"][str(n)] = dict(ce_lb=r["ce_lb"], ce_ub=r["ce_ub"], rounds=r["rounds"], time=c.t, bound_valid=r["bound_valid"],
                                               wells_year0=float(pl["z0"] @ MF.wells[MF.idx0]))
            print(f"  seed {seed}: first stage of {n}x{n} on the {fine}x{fine} tree: [{r['ce_lb']:.2f}, {r['ce_ub']:.2f}] {c.t:.0f}s", flush=True)
        except Exception as e:
            traceback.print_exc(); FT[str(seed)]["rows"][str(n)] = dict(error=repr(e))
        save()
    rr = FT[str(seed)]["rows"]
    if str(fine) in rr and "ce_lb" in rr[str(fine)]:
        for n in rr:
            if "ce_lb" in rr[n]:
                # regret interval of the n x n first stage on the fine tree, % of E(z_N) on the fine tree
                rr[n]["regret_lb"] = 100 * (rr[str(fine)]["ce_lb"] - rr[n]["ce_ub"]) / abs(E); rr[n]["regret_ub"] = 100 * (rr[str(fine)]["ce_ub"] - rr[n]["ce_lb"]) / abs(E)
        save()


def part_master():
    rows = OUT.setdefault("master", [])
    I = make_instance(8, 10, seed=100, **BASE)
    M = Model(I, 1 / 100.0)
    rn = solve_risk_neutral(M)["plan"]; h, _, _ = BL.tune_heuristic(M, by="ce")
    for master in ("cuts", "soc"):
        with B.Clock() as c:
            ce, err = guarded(solve_ce, M, seeds=(rn, h), master=master, log=B.quiet, tol=1e-3, max_iter=25, time_limit=900)
        row = dict(master=master, time=c.t)
        if err:
            row["error"] = err
        else:
            row.update(ce_lb=ce["ce_lb"], ce_ub=ce["ce_ub"], rounds=ce["rounds"], hist=ce["hist"], peak_rss_gb=peak())
            print(f"master {master}: [{ce['ce_lb']:.2f}, {ce['ce_ub']:.2f}] {ce['rounds']}r {c.t:.0f}s", flush=True)
        rows.append(row); save()


def part_tol():
    rows = OUT.setdefault("tol", [])
    for s in range(4):
        I = make_instance(8, 10, seed=s, **B.knobs_for_seed(s))
        M = Model(I, 1 / 100.0)
        rn = solve_risk_neutral(M)["plan"]; h, _, _ = BL.tune_heuristic(M, by="ce")
        row = dict(seed=s, rho=100.0)
        for tol in (1e-3, 1e-4):
            with B.Clock() as c:
                ce, err = guarded(solve_ce, M, seeds=(rn, h), log=B.quiet, tol=tol, max_iter=40, time_limit=900, mip_gap=2e-5 if tol < 1e-3 else 2e-4)
            row[f"tol_{tol:g}"] = dict(error=err, time=c.t) if err else dict(time=c.t, rounds=ce["rounds"], ce_lb=ce["ce_lb"], ce_ub=ce["ce_ub"],
                                                                             gap_pct=100 * (ce["ce_ub"] - ce["ce_lb"]) / max(1e-9, abs(ce["ce_lb"])), hist=ce["hist"])
            if not err:
                print(f"tol seed {s} tol {tol:g}: [{ce['ce_lb']:.3f}, {ce['ce_ub']:.3f}] {ce['rounds']}r {c.t:.0f}s", flush=True)
        rows.append(row); save()


# ------------------------------------------------------------------------------------------------------------------
def enumerate_first_stages(M):
    """Every first stage: per unit-bench one year-0 option or none, packages any subset; pads in year 0 where wells are.
    Yields plans with empty recourse that PlanBuilder accepts (lanes, ownership, capacities checked by PlanBuilder)."""
    by_ub = [M.idx0[M.o_ub[M.idx0] == ub].tolist() + [None] for ub in range(M.nub)]
    for ybits in itertools.product((0, 1), repeat=M.G):
        for choice in itertools.product(*by_ub):
            Bd = PlanBuilder(M)
            ok = True
            for g, yb in enumerate(ybits):
                if yb:
                    if not Bd.can_buy(g):
                        ok = False; break
                    Bd.buy(g)
            if not ok:
                continue
            for o in choice:
                if o is None:
                    continue
                if not Bd.can_add(o):
                    ok = False; break
                Bd.add(o)
            if ok:
                yield Bd.plan


def part_enum():
    I = make_instance(2, 2, seed=100, **BASE)
    lam = 1 / 100.0
    M = Model(I, lam)
    res = OUT["enum"] = dict(U=int(I["U"]), K=M.K, G=M.G, n0=M.n0, n1=M.n1, owned=I["owned"].tolist())
    t0 = time.time()
    plans = list(enumerate_first_stages(M))
    res["n_first_stages"] = len(plans)
    print(f"enum: {len(plans)} feasible first stages (U={I['U']}, G={M.G}, n0={M.n0})", flush=True)
    save()
    best_ce, best_rn, n_done = (-np.inf, None, None), (-np.inf, None), 0
    seen_ce, seen_rn = {}, {}
    for pl in plans:
        key = B.first_stage_key(pl)
        if key in seen_ce:
            continue
        rc = solve_recourse_exact(M, pl, tol=1e-5, max_iter=30, time_limit=60, mip_gap=1e-6)
        seen_ce[key] = (rc["ce_lb"], rc["ce_ub"])
        if rc["ce_lb"] > best_ce[0]:
            best_ce = (rc["ce_lb"], rc["ce_ub"], B.sparse_plan(rc["plan"]))
        rr = solve_risk_neutral(M, fixed=pl, mip_gap=1e-6)
        seen_rn[key] = (rr["obj"], rr["bound"])
        if rr["obj"] > best_rn[0]:
            best_rn = (rr["obj"], rr["bound"])
        n_done += 1
        if n_done % 100 == 0:
            res.update(n_done=n_done, time=time.time() - t0, best_ce_so_far=best_ce[:2], best_rn_so_far=best_rn)
            save()
            print(f"  enum {n_done}/{len(plans)}: best CE {best_ce[0]:.4f}, best E[NPV] {best_rn[0]:.4f}, {time.time() - t0:.0f}s", flush=True)
    res.update(n_done=n_done, time_enum=time.time() - t0, enum_ce_lb=best_ce[0], enum_ce_ub_of_best=best_ce[1],
               enum_ce_max_ub=max(v[1] for v in seen_ce.values()), enum_rn_obj=best_rn[0], enum_rn_max_bound=max(v[1] for v in seen_rn.values()),
               enum_best_plan=best_ce[2])
    with B.Clock() as c:
        rn = solve_risk_neutral(M, mip_gap=1e-6)
    res["milp"] = dict(obj=rn["obj"], bound=rn["bound"], gap=rn["gap"], time=c.t)
    h, _, _ = BL.tune_heuristic(M, by="ce")
    with B.Clock() as c:
        ce = solve_ce(M, seeds=(rn["plan"], h), tol=1e-5, max_iter=40, time_limit=300, mip_gap=1e-6, log=B.quiet)
    res["solve_ce"] = dict(ce_lb=ce["ce_lb"], ce_ub=ce["ce_ub"], rounds=ce["rounds"], time=c.t)
    res["checks"] = dict(ce_lb_within_tol_of_enum=bool(abs(ce["ce_lb"] - best_ce[0]) <= 2e-5 * max(1, abs(best_ce[0])) or ce["ce_lb"] >= best_ce[0] - 1e-9),
                         enum_optimum_below_ce_ub=bool(res["enum_ce_max_ub"] <= ce["ce_ub"] + 1e-6 * max(1, abs(ce["ce_ub"]))),
                         milp_matches_enum=bool(abs(rn["obj"] - best_rn[0]) <= 2e-5 * max(1, abs(best_rn[0]))))
    print(f"enum: optimum CE in [{best_ce[0]:.5f}, {res['enum_ce_max_ub']:.5f}] vs solve_ce [{ce['ce_lb']:.5f}, {ce['ce_ub']:.5f}] ({ce['rounds']} rounds); "
          f"E[NPV] enum {best_rn[0]:.5f} vs MILP {rn['obj']:.5f}; checks {res['checks']}", flush=True)
    save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--parts", default="enum,rank,master,nodes,tol,scale")
    ap.add_argument("--big", default="16,20", help="nx,ny of the largest scale case")
    args = ap.parse_args()
    PATH = args.out
    big = tuple(int(v) for v in args.big.split(","))
    OUT.update(started=time.strftime("%Y-%m-%d %H:%M:%S"), model_version=B.MODEL_VERSION, solver_rev=B.SOLVER_REV, parts=args.parts.split(","), base_knobs=BASE, oa=B.OA,
               gurobi_memlimit_gb=float(os.environ.get("GRB_MEMLIMIT", 1.2)), rss_cap_gb=B.RSS_CAP_GB, errors={})
    save()
    for part in args.parts.split(","):
        fn = dict(scale=lambda: part_scale(big), rank=part_rank, nodes=part_nodes, master=part_master, tol=part_tol, enum=part_enum)[part]
        t0 = time.time()
        try:
            fn()
        except SystemExit:
            raise
        except Exception as e:
            traceback.print_exc(); OUT["errors"][part] = repr(e)
        OUT.setdefault("times", {})[part] = time.time() - t0
        OUT["peak_rss_gb"] = peak()
        save()
        print(f"[{part}] done in {time.time() - t0:.0f}s, peak RSS {peak():.2f} GB", flush=True)
        if peak() > B.RSS_CAP_GB:
            print(f"stopping: peak RSS {peak():.2f} GB exceeds {B.RSS_CAP_GB} GB", flush=True)
            sys.exit(0)
    print("done", flush=True)
