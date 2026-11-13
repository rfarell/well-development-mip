"""Terminal-credit and hedge conventions.

On one instance (bench_v3's knobs for the seed by default) and a rho list, re-solve the certified CE plan z_A (seeded
with z_N, RN-LP rounding and the CE-tuned heuristic, as bench_v3's ce row) and the risk-neutral plan z_N under
alternative conventions, one at a time around the base case (the instance's own: nu = instance.NU_TERM = 0.5,
f0 = instance.F0_HEDGE = 0.6, the producing base's in-tenor share pdp = I['pdp_hedge'] = 0.5):

  nu   share of the terminal inventory's NPV credited at the horizon, in {0, 0.25, 0.5, 0.75, 1} (I['nu_term'], read by
       model.Model since v3.2; the credited set is the same for every nu > 0, and nu = 0 credits nothing)
  f0   the hedge convention: the hedgeable share of a committed (year-0) well's revenue, I['f_hedge'] = (f0, 0, 0, 0),
       given as f0, or as f0:pdp to set the producing base's in-tenor share I['pdp_hedge'] with it (the tenor fixes
       both). Default 0.6, 0.73:0.68 and 0:
         0.6        the 36-month tenor (the base case; the paper's 0.60 is the share of a new well's discounted volume
                    inside a 36-month tenor);
         0.73:0.68  the 60-month tenor. Arps declines calibrated to 0.60 inside 36 months (b 0.8-1.2, terminal decline
                    6-8%, discounting at 5%, first sales nine months after spud or at spud) place 0.71-0.75 of a new
                    well's discounted volume inside 60 months, and 0.46-0.52 inside 24 months (so an f0 of 0.45 would
                    be a tenor shorter than 24 months, not a longer one). The base's 0.5 inside 36 months becomes
                    1 - 0.5^(60/36) = 0.685 inside 60 months for an exponential tail, a little less for a hyperbolic
                    one, so 0.68;
         0          no hedge: the value of flexibility with f0 = 0 on both rungs (with f0 > 0 only year-0 wells are hedged,
                    although the committed rung also commits its later-year wells now, which favors the two-stage
                    rung).
       Pass --f0s 0.6,0.73,0 to move only the new wells' share.

(--cross runs the full nu x hedge grid instead.) Per (rho, convention) it reports for z_A and z_N: CE (z_A with its
certificate), E[NPV], lambda*sigma_W, the expected terminal credit inside E[NPV], expected wells by year, length and
bench, packages (count and ids), the developed unit-benches in year 0 and in any node with their Jaccard overlap with
the base case's (jaccard_year0_vs_base, jaccard_any_vs_base); and bench_v3's ladder (ladder_block: committed, two-stage
and perfect-information rungs, value of recourse, EVPI, the mixed VSS, the RN-VSS and the CE-VSS, each with its
certificate interval; --no-ladder skips it). Every plan is saved as sparse index lists (bench_v3.sparse_plan; the
option table, the same for every convention, is in res['plan_format']). The JSON is rewritten atomically after every
convention; a failing step is recorded in the row's errors and the row goes on. res['status'] is running, done, or
"stopping: peak RSS ..." when the memory guard stops the process.

Usage: exp_conventions.py nx ny seed out.json [rho_list] [--nus 0,0.25,0.5,0.75,1] [--f0s 0.6,0.73:0.68,0] [--cross]
       [--knobs bench|base] [--no-ladder]
Environment as bench_v3.py: GRB_THREADS, GRB_MEMLIMIT, OMP_NUM_THREADS, BENCH_RSS_CAP_GB (the memory guard runs after
every convention, once the JSON is saved).
"""
import argparse
import copy
import os
import time
import traceback

import bench_v3 as B
import baselines as BL
from instance import make_instance, NU_TERM
from model import Model
from solve import solve_ce, solve_risk_neutral, solve_recourse_exact, relax_and_round

F0S_DEFAULT = "0.6,0.73:0.68,0"      # 36-month tenor (base), 60-month tenor (new wells and base), no hedge (see above)


def floats(v):
    return [float(x) for x in str(v).split(",") if x.strip()]


def hedges(v, pdp0):
    """--f0s entries, f0 or f0:pdp, as (f0, pdp) pairs; pdp defaults to the instance's own base share pdp0."""
    out = []
    for x in str(v).split(","):
        f0, _, pdp = x.strip().partition(":")
        if f0.strip():
            out.append((float(f0), float(pdp) if pdp.strip() else float(pdp0)))
    return out


def with_conventions(I0, nu, f0, pdp):
    """A copy of the instance with terminal credit share nu, committed-well hedge share f0 (later years unchanged) and
    the producing base's in-tenor share pdp."""
    I = copy.deepcopy(I0)
    I["nu_term"] = float(nu)
    fh = list(I0["f_hedge"]); fh[0] = float(f0)
    I["f_hedge"] = tuple(fh)
    I["pdp_hedge"] = float(pdp)
    return I


def conventions(nus, hedge_list, base, cross):
    """The (nu, f0, pdp) triples to solve, the base case first: one convention at a time around the base, or the full
    grid of nus x hedges."""
    grid = ([(nu, f0, pdp) for f0, pdp in hedge_list for nu in nus] if cross
            else [(nu, base[1], base[2]) for nu in nus] + [(base[0], f0, pdp) for f0, pdp in hedge_list])
    out = [base]
    for c in grid:
        if all(any(abs(a - b) > 1e-9 for a, b in zip(c, x)) for x in out):
            out.append(c)
    return out


def label_of(nu, f0, pdp, base):
    parts = ([f"nu={nu:g}"] if abs(nu - base[0]) > 1e-9 else []) + ([f"f0={f0:g}"] if abs(f0 - base[1]) > 1e-9 else []) \
        + ([f"pdp={pdp:g}"] if abs(pdp - base[2]) > 1e-9 else [])
    return ",".join(parts) or "base"


def terminal_credit(M, plan):
    """Expected terminal inventory credit inside the plan's E[NPV], $M: sum_k p_k tau_k . iota_k."""
    p = M.I["prob"]
    return float(sum(p[k] * (M.tau[:, k] @ M.inventory(plan, M.full_z(plan, k))) for k in range(M.K)))


def plan_summary(M, plan, **extra):
    e, pl = M.evaluate(plan), B.placement(M, plan)
    return dict(ce=e["ce"], enpv=e["enpv"], lam_sigma=e["lam_sigma"], terminal_credit=terminal_credit(M, plan),
                wells_expected=pl["wells_expected"], wells_year0=pl["wells_year0"], wells_by_year=pl["by_year"], by_len=pl["by_len"],
                by_bench=pl["by_bench"], packages=pl["packages"], package_ids=pl["first_stage"]["packages"],
                dev_ub_year0=pl["dev_ub_year0"], dev_ub_any=pl["dev_ub_any"], plan=B.sparse_plan(plan), **extra)


def step(row, name, fn):
    """One step of a convention row: its result, or None with the failure recorded in row['errors']."""
    t0 = time.time()
    try:
        return fn()
    except Exception as e:
        traceback.print_exc()
        row["errors"].append(B.error_entry(name, e, time.time() - t0))
        return None


def solve_convention(I0, lam, nu, f0, pdp, ladder=True):
    """z_N, z_A and (optionally) the ladder of one convention; see the module docstring."""
    I = with_conventions(I0, nu, f0, pdp)
    M = Model(I, lam)
    row = dict(nu=nu, f0=f0, pdp_hedge=pdp, nu_term=M.nu_term, f_hedge=list(I["f_hedge"]),
               credited_terminal_ub=int((M.tau_best >= 0).sum()), errors=[], times={})

    def timed(name, fn):
        t0 = time.time()
        out = step(row, name, fn)
        row["times"][name] = time.time() - t0
        return out

    rn = timed("rn", lambda: solve_risk_neutral(M))
    if rn:
        row["z_N"] = plan_summary(M, rn["plan"], mip_gap=rn["gap"], bound=rn["bound"])
    h = timed("heur_ce", lambda: BL.tune_heuristic(M, by="ce")[0])
    rr = timed("rn_lp_round", lambda: relax_and_round(M))
    seeds = tuple(x for x in (rn["plan"] if rn else None, rr, h) if x is not None)
    ce = timed("ce", lambda: solve_ce(M, seeds=seeds, log=B.quiet, **B.OA))
    if ce:
        row["z_A"] = plan_summary(M, ce["plan"], ce_lb=ce["ce_lb"], ce_ub=ce["ce_ub"], rounds=ce["rounds"])
    if not (ladder and ce):
        return row
    cc = timed("ce_committed", lambda: solve_ce(M, seeds=(solve_risk_neutral(M, committed=True)["plan"],), committed=True, log=B.quiet, **B.OA))
    pi = timed("pi", lambda: B.perfect_information_bound(M, ce["plan"], **B.OA))
    Mm = timed("mean_value_model", lambda: B.mean_value_model(I, lam))
    mv = timed("mv", lambda: solve_risk_neutral(Mm)) if Mm else None
    mv_fs = B.first_stage_into(M, mv["plan"]) if mv else None
    eev = timed("mv+exact_recourse", lambda: solve_recourse_exact(M, mv_fs, **B.OA)) if mv else None
    mvrn = timed("mv_rn", lambda: solve_risk_neutral(M, fixed=mv_fs)) if mv else None
    mvce = timed("mv_ce", lambda: solve_ce(Mm, seeds=(mv["plan"],) if mv else (), log=B.quiet, **B.OA)) if Mm else None
    mvce_ex = timed("mv_ce+exact_recourse", lambda: solve_recourse_exact(M, B.first_stage_into(M, mvce["plan"]), **B.OA)) if mvce else None
    br = lambda r: (r["ce_lb"], r["ce_ub"]) if r else None  # noqa: E731
    rnv = lambda r: (M.evaluate(r["plan"])["enpv"], r["bound"]) if r else None  # noqa: E731
    row["ladder"] = B.ladder_block(br(ce), br(cc), pi, br(eev), rnv(rn), rnv(mvrn), br(mvce_ex), time_pi=row["times"].get("pi", float("nan")))
    row["ladder_plans"] = {n: B.sparse_plan(r["plan"]) for n, r in (("ce_committed", cc), ("mv+exact_recourse", eev), ("mv_rn", mvrn),
                                                                     ("mv_ce+exact_recourse", mvce_ex)) if r}
    return row


def overlap_with_base(row, base_row):
    """Jaccard overlap of the developed unit-benches (year 0, any node) with the base case's, for z_A and z_N."""
    for who in ("z_A", "z_N"):
        if who in row and base_row is not None and who in base_row:
            a, b = row[who], base_row[who]
            a["jaccard_year0_vs_base"] = B.jaccard(a["dev_ub_year0"], b["dev_ub_year0"])
            a["jaccard_any_vs_base"] = B.jaccard(a["dev_ub_any"], b["dev_ub_any"])


def line(row):
    def plan(who):
        a = row.get(who)
        if not a:
            return f"{who} --"
        cert = f" [{a['ce_lb']:.2f}, {a['ce_ub']:.2f}]" if "ce_lb" in a else ""
        return (f"{who} CE {a['ce']:.2f}{cert} E[NPV] {a['enpv']:.2f} credit {a['terminal_credit']:.2f} wells/yr "
                f"{[round(x, 1) for x in a['wells_by_year']]} pkgs {a['packages']} J0 {a.get('jaccard_year0_vs_base', float('nan')):.2f}")
    L = row.get("ladder")
    lad = (f" | ladder com {L['committed_lb']:.2f} two {L['two_stage_lb']:.2f} PI {L['pi_lb']:.2f}; VoR {L['value_of_recourse']:.2f} "
           f"EVPI {L['evpi']:.2f} VSS mixed {L['vss_mixed']:.2f} RN {L['rn_vss']:.2f} CE {L['ce_vss']:.2f}") if L else ""
    return (f"rho {row['rho']:6.0f}  {row['label']:17s} credited {row.get('credited_terminal_ub', '-')}  {plan('z_A')} | {plan('z_N')}{lad}  "
            f"{row['time']:.0f}s  peak RSS {row['peak_rss_gb']:.2f} GB" + (f"  errors {len(row['errors'])}" if row.get("errors") else ""))


def guard(res, path, tag):
    """bench_v3's memory guard, with the stop recorded in res['status'] and saved before the exit."""
    g = B.rss_gb()
    if g > B.RSS_CAP_GB:
        res["status"] = f"stopping: peak RSS {g:.2f} GB exceeds {B.RSS_CAP_GB} GB (after {tag})"
        B.dump(res, path)
    B.guard(tag)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nx", type=int); ap.add_argument("ny", type=int); ap.add_argument("seed", type=int); ap.add_argument("out")
    ap.add_argument("rho_list", nargs="?", default="100,300,1000", help="comma-separated risk tolerances, $M (default 100,300,1000)")
    ap.add_argument("--nus", default="0,0.25,0.5,0.75,1.0", help="terminal credit shares (default 0,0.25,0.5,0.75,1.0)")
    ap.add_argument("--f0s", default=F0S_DEFAULT,
                    help=f"hedge conventions, f0 or f0:pdp (default {F0S_DEFAULT}: the 36-month tenor, the 60-month tenor and no hedge)")
    ap.add_argument("--cross", action="store_true", help="the full nu x hedge grid instead of one convention at a time")
    ap.add_argument("--knobs", choices=("bench", "base"), default="bench",
                    help="bench: bench_v3.knobs_for_seed(seed) (default); base: make_instance's defaults (the rho sweeps' setting)")
    ap.add_argument("--no-ladder", dest="ladder", action="store_false", help="skip the ladder (z_A and z_N only)")
    args = ap.parse_args()
    rhos, nus = floats(args.rho_list), floats(args.nus)
    knobs = B.knobs_for_seed(args.seed) if args.knobs == "bench" else {}
    I0 = make_instance(args.nx, args.ny, seed=args.seed, **knobs)
    base = (float(I0.get("nu_term", NU_TERM)), float(I0["f_hedge"][0]), float(I0["pdp_hedge"]))
    configs = conventions(nus, hedges(args.f0s, base[2]), base, args.cross)
    res = dict(script="exp_conventions.py", version=B.VERSION, model_version=B.MODEL_VERSION, solver_rev=B.SOLVER_REV, status="running", nx=args.nx, ny=args.ny, seed=args.seed, knobs=knobs,
               args=vars(args), rhos=rhos, base=dict(nu=base[0], f0=base[1], pdp_hedge=base[2]),
               configs=[dict(nu=c[0], f0=c[1], pdp_hedge=c[2], label=label_of(*c, base)) for c in configs],
               plan_format=B.plan_format(Model(I0, 1.0 / rhos[0])), started=time.strftime("%Y-%m-%d %H:%M:%S"),
               threads=dict(gurobi=int(os.environ.get("GRB_THREADS", 2)), omp=os.environ.get("OMP_NUM_THREADS")),
               gurobi_memlimit_gb=float(os.environ.get("GRB_MEMLIMIT", 1.2)), rss_cap_gb=B.RSS_CAP_GB, oa=B.OA, rows=[], errors=[])
    B.dump(res, args.out)                                               # fail fast on an unwritable output path
    print(f"conventions on seed {args.seed} ({args.nx}x{args.ny}, knobs {args.knobs}): base nu {base[0]:g}, f0 {base[1]:g}, "
          f"pdp {base[2]:g}; {len(configs)} conventions x {len(rhos)} rho: {', '.join(c['label'] for c in res['configs'])}", flush=True)
    for rho in rhos:
        base_row = None
        for nu, f0, pdp in configs:
            t0 = time.time()
            try:
                row = solve_convention(I0, 1.0 / rho, nu, f0, pdp, args.ladder)
            except Exception as e:                                      # building the model; the steps record their own failures
                traceback.print_exc()
                row = dict(nu=nu, f0=f0, pdp_hedge=pdp, errors=[B.error_entry("convention", e, time.time() - t0)])
                res["errors"].append(dict(rho=rho, nu=nu, f0=f0, pdp_hedge=pdp, error=repr(e)))
            row.update(rho=rho, label=label_of(nu, f0, pdp, base), is_base=(nu, f0, pdp) == base, time=time.time() - t0,
                       peak_rss_gb=B.rss_gb(), rss_now_gb=B.rss_now_gb())
            if row["is_base"]:
                base_row = row
            overlap_with_base(row, base_row)
            res["rows"].append(row)
            print(line(row), flush=True)
            B.dump(res, args.out)
            guard(res, args.out, f"rho {rho:g} {row['label']}")
    res["status"] = "done"
    B.dump(res, args.out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
