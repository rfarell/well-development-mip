"""Lean replicate harness: the main comparisons on many instances, about four minutes per (instance, rho) at
40 units, written in bench_v3's JSON layout so summarize_replicates.py can read them.

Rows per (instance, rho):
  heur_ev, heur_ce              rank-and-fill with recourse, grid-tuned by E[NPV] / by CE
  rn                            risk-neutral two-stage MIP at the strip (+ the screen UB_N - CE(x_N))
  rn_tilt1, rn_tilt2            pi-tilted risk-neutral MIP, one and two fixed-point iterations from rn
  rn_deck                       risk-neutral MIP at the oracle-selected deck on --decks (default 0.70:1.05:0.05); curve stored
  ce                            certified CE solver seeded with rn, heur_ce, rn_tilt2 (time includes the seeds)
  heur_ev+rn_recourse           the rule's committed decisions with risk-neutral recourse (robust value of optimization)
  X+exact_recourse              first stage of X fixed, certified CE recourse, for X in heur_ev, heur_ce, rn, rn_deck, rn_tilt2
  mv_rn                         mean-value first stage (theta = P = 1) with risk-neutral recourse on the tree (RN-VSS)
  ce_committed                  certified CE plan with one recourse shared by all nodes (--committed, default on)
plus the perfect-information rung (--pi, default on), the ladder, Beck's statistic, Jaccard overlaps and the as-planned
perturbed-truth table. Every stage runs in its own try/except (bench_v3.Stages) and the JSON is rewritten after each.

Usage: exp_replicates.py nx ny seed_from seed_to rho_list out.json [--knobs own_frac=0.5,range_mi=4] [--decks 0.70:1.05:0.05]
       [--no-pi] [--no-committed] [--tag base]
Environment as bench_v3.py (GRB_THREADS, GRB_MEMLIMIT, OMP_NUM_THREADS, BENCH_RSS_CAP_GB).
"""
import argparse
import json
import os
import sys
import time
import traceback

import pickle

import numpy as np

from instance import make_instance, build_nodes, DEFAULTS
from model import Model
from solve import solve_ce, solve_risk_neutral, solve_recourse_exact
import baselines as BL
import bench_v3 as B

VERSION = "replicates-1.2"
OA_PI = dict(B.OA, max_iter=6, time_limit=45)   # the perfect-information rung: 9 single-node solves; capped (bracket valid at any cap)
BASE_KNOBS = dict(own_frac=0.65, deadline_frac=0.25, parent_frac=0.35, rig_mult=2.5, range_mi=3.0)
FAMILY = B.FAMILY


def parse_knobs(s):
    out = {}
    for kv in (s or "").split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split("=")
        out[k.strip()] = float(v)
    return out


def parse_decks(s):
    a, b, h = (float(v) for v in s.split(":"))
    n = int(round((b - a) / h)) + 1
    return [round(a + i * h, 3) for i in range(n)]


def bench_lean(I, s, knobs, rho, args, run, save, reuse=None):
    lam = 1.0 / rho
    M = Model(I, lam)
    cap = I["cap"]
    print(f"=== seed {s}  U={M.U} nub={M.nub} options={M.n} n0={M.n0} K={M.K} G={M.G} binaries~{M.n0 + M.K * M.n1}  rho ${rho:,.0f}M  "
          f"knobs " + " ".join(f"{k}={v}" for k, v in knobs.items()), flush=True)
    run.update(dict(seed=s, knobs=knobs, tag=args.tag, model_version=B.MODEL_VERSION, solver_rev=B.SOLVER_REV, U=M.U, nub=M.nub, options=M.n, n0=M.n0, n1=M.n1, K=M.K, G=M.G,
                    binaries=M.n0 + M.K * M.n1, rho=rho, lam=lam,
                    cap=dict(rig=float(cap["rig"][0]), crew=float(cap["crew"][0]), capital=float(cap["capital"][0]), acq=float(cap["acq"])),
                    fringe_units=int(M.fringe_u.sum()), credited_terminal_ub=int((M.tau_best >= 0).sum()), nu_term=M.nu_term,
                    owned_units=int(I["owned"].sum()), parents_total=int(I["parents"].sum()), packages=int(M.G),
                    methods={}, plans={}, plan_format=B.plan_format(M)))
    R, P, ctx = run["methods"], {}, {}
    if reuse is not None:
        ctx["reuse_src"] = reuse
    stage = B.Stages(run, P, ctx, save)

    def add(name, plan, extra=None, label=None):
        row = B.score(M, name, plan, extra, label)
        P[name], R[name], run["plans"][name] = plan, row, B.sparse_plan(plan)

    # ---- 0. the empty plan: E(0), CE(0), lambda*sigma_W(0) (the producing base and credited inventory alone) -------------
    def base_value():
        e = M.evaluate(M.empty_plan())
        run["base_value"] = dict(enpv0=e["enpv"], ce0=e["ce"], sigma0=e["sigma"], lam_sigma0=e["lam_sigma"], risk_premium0=e["enpv"] - e["ce"])
        print(f"  empty plan: E(0) {e['enpv']:.2f}  CE(0) {e['ce']:.2f}  lam*sigma_W(0) {e['lam_sigma']:.2f}", flush=True)

    stage("base_value", base_value)

    # ---- 1. heuristics ----------------------------------------------------------------------------------------------
    def heur(name, by, label):
        with B.Clock() as c:
            h, cfg, _ = BL.tune_heuristic(M, by=by)
        add(name, h, dict(time=c.t, cfg=cfg), label)

    stage("heur_ev", lambda: heur("heur_ev", "enpv", "rank-and-fill with recourse, settings tuned by E[NPV]"))
    stage("heur_ce", lambda: heur("heur_ce", "ce", "rank-and-fill with recourse, settings tuned by CE (oracle)"))

    # ---- 2. risk-neutral MIP, screen, tilts, the deck family ---------------------------------------------------------
    def reuse_source():
        """The finished run of the same instance at another rho (--reuse-decks), if its option table matches; else None."""
        src = ctx.get("reuse_src")
        if src is None:
            return None
        if src.get("plan_format", {}).get("options_sha1") != run["plan_format"]["options_sha1"] or src.get("seed") != s:
            print("  reuse: option table or seed differs; solving afresh", flush=True)
            return None
        return src

    def rn_mip():
        src = reuse_source()
        if src and "rn" in src.get("plans", {}) and "rn" in src.get("methods", {}):
            m0 = src["methods"]["rn"]
            rn, t = dict(plan=B.dense_plan(M, src["plans"]["rn"]), gap=m0["mip_gap"], bound=m0["bound"], obj=m0["obj"]), 0.0
            extra = dict(time=t, mip_gap=rn["gap"], bound=rn["bound"], obj=rn["obj"], reused_from=src.get("_file"), time_original=m0.get("time"))
        else:
            with B.Clock() as c:
                rn = solve_risk_neutral(M)
            extra = dict(time=c.t, mip_gap=rn["gap"], bound=rn["bound"], obj=rn["obj"])
        add("rn", rn["plan"], extra, "risk-neutral two-stage MIP at the strip" + (" (plan reused: it does not depend on rho)" if "reused_from" in extra else ""))
        screen = rn["bound"] - R["rn"]["ce"]
        run["screen"] = dict(ub_n=rn["bound"], ce_rn=R["rn"]["ce"], value=screen, pct_enpv_rn=100 * screen / max(1e-9, abs(R["rn"]["enpv"])))
        print(f"  screen: UB_N - CE(x_N) = {screen:.2f} ({run['screen']['pct_enpv_rn']:.1f}% of E[NPV](rn))", flush=True)

    def tilted(name, src, label):
        with B.Clock() as c:
            tl = M.implied_tilt(P[src]); r = solve_risk_neutral(M, weights=tl["pi"])
        add(name, r["plan"], dict(time=c.t + R[src]["time"], implied_price=tl["implied_price"], implied_theta=tl["implied_theta"],
                                  pi=tl["pi"].tolist()), label)

    def deck():
        curve, best, plans, reused = [], None, {}, 0
        src = reuse_source()
        src_plans = src.get("deck_plans", {}) if src else {}
        with B.Clock() as c:
            for d in args.decks:
                if abs(d - 1.0) < 1e-9:
                    pd, tt = P["rn"], 0.0
                elif str(d) in src_plans:
                    pd, tt, reused = B.dense_plan(M, src_plans[str(d)]), 0.0, reused + 1
                else:
                    with B.Clock() as c1:
                        pd = solve_risk_neutral(Model(I, lam, deck=d))["plan"]
                    tt = c1.t
                e = M.evaluate(pd)
                curve.append(dict(deck=d, ce=e["ce"], enpv=e["enpv"], time=tt))
                plans[d] = pd
                if best is None or e["ce"] > best[0]:
                    best = (e["ce"], d)
        d = best[1]
        run["deck_plans"] = {str(dd): B.sparse_plan(plans[dd]) for dd in args.decks}
        add("rn_deck", plans[d], dict(time=c.t + R["rn"]["time"], deck=d, deck_curve=curve, reused_plans=reused),
            f"risk-neutral MIP at the oracle-selected planning deck (grid {args.decks[0]:.3g}-{args.decks[-1]:.3g} by "
            f"{args.decks[1] - args.decks[0]:.3g}; best {d}); time includes rn")
        if 0.9 in plans and abs(d - 0.9) > 1e-9:
            ctx["deck09"] = plans[0.9]

    stage("rn", rn_mip)
    stage("rn_tilt1", lambda: tilted("rn_tilt1", "rn", "pi-tilted risk-neutral MIP, one fixed-point iteration from rn (time includes rn)"), needs=("rn",))
    stage("rn_tilt2", lambda: tilted("rn_tilt2", "rn_tilt1", "pi-tilted risk-neutral MIP, two fixed-point iterations from rn (time includes rn, rn_tilt1)"),
          needs=("rn_tilt1",))
    stage("rn_deck", deck, needs=("rn",))

    # ---- 3. certified solve -----------------------------------------------------------------------------------------
    def prior_plan(name):
        """The plan an earlier run of this instance at this risk tolerance stored under the name (--reuse-decks), as one more
        seed: any feasible plan is a legitimate seed, and it keeps the lower end from falling below the earlier run's."""
        src = reuse_source()
        if src is None or abs(float(src.get("rho", -1.0)) - rho) > 1e-9 or name not in src.get("plans", {}):
            return None
        try:
            pl = B.dense_plan(M, src["plans"][name])
        except Exception:
            return None
        return pl if M.feasible(pl) else None

    def ce_warm():
        names = [n for n in ("rn", "heur_ce", "rn_tilt2") if n in P]
        prior = prior_plan("ce")
        run["prior_seeds"] = prior is not None
        with B.Clock() as c:
            ce = solve_ce(M, seeds=tuple(P[n] for n in names) + ((prior,) if prior is not None else ()), log=B.quiet, **B.OA)
        ctx["ub_ce"] = ce["ce_ub"]
        run["ce_ub_best"] = float(ce["ce_ub"])
        add("ce", ce["plan"], dict(B.cert_extra(ce, c.t, seed_time=sum(R[n]["time"] for n in names)), seeds=names),
            f"certified CE solver seeded with {', '.join(names)} (time includes the seeds)")

    stage("ce", ce_warm)

    # ---- 4. first stages under optimal recourse ----------------------------------------------------------------------
    def rn_recourse(src):
        with B.Clock() as c:
            r = solve_risk_neutral(M, fixed=P[src])
        add(f"{src}+rn_recourse", r["plan"], dict(time=c.t + R[src]["time"], mip_gap=r["gap"], bound=r["bound"], obj=r["obj"]),
            f"first stage of {src}, risk-neutral recourse")

    def exact_recourse(src):
        with B.Clock() as c:
            prior = prior_plan(f"{src}+exact_recourse")
            r = solve_recourse_exact(M, P[src], extra_seeds=(prior,) if prior is not None else (), **B.OA)
        add(f"{src}+exact_recourse", r["plan"], B.cert_extra(r, c.t, seed_time=R[src]["time"]), f"first stage of {src}, certified CE recourse")

    stage("heur_ev+rn_recourse", lambda: rn_recourse("heur_ev"), needs=("heur_ev",))
    for src in ("heur_ev", "heur_ce", "rn", "rn_deck", "rn_tilt2"):
        stage(f"{src}+exact_recourse", lambda src=src: exact_recourse(src), needs=(src,))

    # ---- 5. mean-value first stage with risk-neutral recourse (RN-VSS) -----------------------------------------------
    def mv_rn():
        with B.Clock() as c:
            Mm = B.mean_value_model(I, lam)
            r0 = solve_risk_neutral(Mm)
            fs = B.first_stage_into(M, r0["plan"])
            r = solve_risk_neutral(M, fixed=fs)
        run["mean_value"] = dict(rn=dict(obj=r0["obj"], bound=r0["bound"], mip_gap=r0["gap"]))
        add("mv_rn", r["plan"], dict(time=c.t, mip_gap=r["gap"], bound=r["bound"], obj=r["obj"]),
            "mean-value first stage (theta = P = 1, risk-neutral) with risk-neutral recourse on the tree")

    stage("mv_rn", mv_rn)

    # ---- 6. committed plan, perfect information, ladder ---------------------------------------------------------------
    def committed():
        with B.Clock() as c:
            prior = prior_plan("ce_committed")
            cc = solve_ce(M, seeds=(solve_risk_neutral(M, committed=True)["plan"],) + ((prior,) if prior is not None else ()), committed=True, log=B.quiet, **B.OA)
        add("ce_committed", cc["plan"], B.cert_extra(cc, c.t), "certified CE plan with one recourse shared by all nodes (committed)")

    def pi_rung():
        with B.Clock() as c:
            ctx["pi"] = B.perfect_information_bound(M, P["ce"], **OA_PI)
        ctx["pi"]["oa"] = dict(OA_PI)
        ctx["t_pi"] = c.t

    def ladder():
        br = lambda n: (R[n]["ce_lb"], R[n]["ce_ub"]) if n in R else None  # noqa: E731
        rnv = lambda n: (R[n]["enpv"], R[n]["bound"]) if n in R else None  # noqa: E731
        two_best = (R["ce"]["ce"], run.get("ce_ub_best", B.NAN), "ce") if "ce" in R else None
        L = run["ladder"] = B.ladder_block(br("ce"), br("ce_committed"), ctx.get("pi"), None, rnv("rn"), rnv("mv_rn"), None, two_best, ctx.get("t_pi", B.NAN))
        print(f"  ladder: committed [{L['committed_lb']:.2f}, {L['committed_ub']:.2f}] <= two-stage [{L['two_stage_lb']:.2f}, {L['two_stage_ub']:.2f}] <= "
              f"PI [{L['pi_lb']:.2f}, {L['pi_ub']:.2f}]; flexibility {L['value_of_recourse']:.2f}, EVPI {L['evpi']:.2f}; "
              f"RN-VSS {L['rn_vss']:.2f} [{L['rn_vss_lb']:.2f}, {L['rn_vss_ub']:.2f}]", flush=True)

    if args.committed:
        stage("ce_committed", committed)
    if args.pi:
        stage("pi", pi_rung, needs=("ce",))
    stage("ladder", ladder)

    # ---- 7. Beck's statistic, overlaps, the family comparison, as-planned perturbed truths ----------------------------
    def beck():
        e_rn, e_ce, e_dk, e_t2 = R["rn"], R["ce"], R["rn_deck"], R["rn_tilt2"]
        den = max(1e-9, abs(e_rn["enpv"]))
        fam = [m for m in FAMILY if m in R]
        best_fam = max(fam, key=lambda m: R[m]["ce"])
        gain_fam = e_ce["ce"] - R[best_fam]["ce"]
        fam_x = [m + "+exact_recourse" for m in ("rn_deck", "rn_tilt2") if m + "+exact_recourse" in R]
        best_fam_x = max(fam_x, key=lambda m: R[m]["ce_lb"]) if fam_x else None
        run["beck"] = dict(lam_sigma_rn=e_rn["lam_sigma"], lam_sigma_ce=e_ce["lam_sigma"], enpv_rn=e_rn["enpv"], ce_rn=e_rn["ce"],
                           enpv_ce=e_ce["enpv"], ce_ce=e_ce["ce"], enpv_cost=e_rn["enpv"] - e_ce["enpv"],
                           enpv_cost_pct=100 * (e_rn["enpv"] - e_ce["enpv"]) / den,
                           ce_gain=e_ce["ce"] - e_rn["ce"], ce_gain_pct=100 * (e_ce["ce"] - e_rn["ce"]) / den,
                           ce_gain_pct_ce_rn=100 * (e_ce["ce"] - e_rn["ce"]) / max(1e-9, abs(e_rn["ce"])),
                           gain_vs_deck=e_ce["ce"] - e_dk["ce"], gain_vs_deck_pct=100 * (e_ce["ce"] - e_dk["ce"]) / den,
                           gain_vs_tilt2=e_ce["ce"] - e_t2["ce"], gain_vs_tilt2_pct=100 * (e_ce["ce"] - e_t2["ce"]) / den,
                           best_family=best_fam, gain_vs_family=gain_fam, gain_vs_family_pct=100 * gain_fam / den,
                           best_family_exact=best_fam_x,
                           # the lower end takes the largest upper end over the members (the two maxima are separate); files written
                           # before 2026-10-05 hold the old value, which the reporting scripts do not read
                           gain_vs_family_exact_lb=(e_ce["ce_lb"] - max(R[m]["ce_ub"] for m in fam_x)) if best_fam_x else B.NAN,
                           gain_vs_family_exact_ub=(e_ce["ce_ub"] - R[best_fam_x]["ce_lb"]) if best_fam_x else B.NAN,
                           screen=run["screen"]["value"], screen_pct=run["screen"]["pct_enpv_rn"], mip_gap_pct=e_ce["gap_pct"],
                           lognormal_shift_ce=e_ce["ce_mc"] - e_ce["ce"], lognormal_shift_rn=e_rn["ce_mc"] - e_rn["ce"],
                           lognormal_shift_deck=e_dk["ce_mc"] - e_dk["ce"])
        b = run["beck"]
        print(f"  Beck: E[NPV] cost {b['enpv_cost']:.2f} ({b['enpv_cost_pct']:.1f}%), CE gain {b['ce_gain']:.2f} ({b['ce_gain_pct']:.1f}%), "
              f"vs best family ({best_fam}) {gain_fam:.2f} ({b['gain_vs_family_pct']:.2f}%); lam*sigma_W(rn) {b['lam_sigma_rn']:.2f}", flush=True)

    def vopt():
        """Value of optimization at model truth, its deck-matched split and its robust forms, in $M and % of E[NPV](rn)."""
        den = max(1e-9, abs(R["rn"]["enpv"]))
        curve = {round(c["deck"], 3): c for c in R["rn_deck"]["deck_curve"]}
        out = dict(enpv_rn=R["rn"]["enpv"], enpv_heur_ev=R["heur_ev"]["enpv"], vopt=R["rn"]["enpv"] - R["heur_ev"]["enpv"],
                   heur_ev_deck=R["heur_ev"]["cfg"]["deck"], heur_ce_deck=R["heur_ce"]["cfg"]["deck"])
        d = round(float(R["heur_ev"]["cfg"]["deck"]), 3)
        if d in curve:
            out.update(enpv_rn_at_rule_deck=curve[d]["enpv"], vopt_at_rule_deck=curve[d]["enpv"] - R["heur_ev"]["enpv"])
        if "heur_ev+rn_recourse" in R:
            out.update(enpv_heur_ev_rn_recourse=R["heur_ev+rn_recourse"]["enpv"], vopt_robust=R["rn"]["enpv"] - R["heur_ev+rn_recourse"]["enpv"],
                       vopt_robust_lb=R["rn"]["enpv"] - R["heur_ev+rn_recourse"]["bound"], vopt_robust_ub=R["rn"]["bound"] - R["heur_ev+rn_recourse"]["enpv"])
        if "heur_ce+exact_recourse" in R and "ce" in R:
            a, b = R["ce"], R["heur_ce+exact_recourse"]
            out.update(fs_gap_ce=a["ce_lb"] - b["ce_lb"], fs_gap_ce_lb=a["ce_lb"] - b["ce_ub"], fs_gap_ce_ub=a["ce_ub"] - b["ce_lb"])
        for k in list(out):
            if k.startswith(("vopt", "fs_gap")):
                out[k + "_pct"] = 100 * out[k] / den
        run["vopt"] = out
        print(f"  value of optimization: {out['vopt']:.2f} ({out['vopt_pct']:.2f}%), at the rule's deck {out.get('vopt_at_rule_deck', B.NAN):.2f}, "
              f"robust (RN recourse) {out.get('vopt_robust', B.NAN):.2f} ({out.get('vopt_robust_pct', B.NAN):.2f}%), "
              f"first-stage CE gap {out.get('fs_gap_ce', B.NAN):.2f} ({out.get('fs_gap_ce_pct', B.NAN):.2f}%)", flush=True)

    def overlap():
        ref0, refa = R["ce"]["placement"]["dev_ub_year0"], R["ce"]["placement"]["dev_ub_any"]
        for row in R.values():
            row["placement"]["jaccard_vs_ce_year0"] = B.jaccard(row["placement"]["dev_ub_year0"], ref0)
            row["placement"]["jaccard_vs_ce_any"] = B.jaccard(row["placement"]["dev_ub_any"], refa)

    def perturbed():
        with B.Clock() as c:
            run["perturbed"] = B.perturbed_truth(I, M, lam, P, run["errors"])
        run["perturbed_time"] = c.t

    stage("beck", beck, needs=("rn", "ce", "rn_deck", "rn_tilt2"))
    stage("vopt", vopt, needs=("rn", "heur_ev", "heur_ce", "rn_deck"))
    stage("jaccard", overlap, needs=("ce",))
    stage("perturbed", perturbed)
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nx", type=int); ap.add_argument("ny", type=int)
    ap.add_argument("seed_from", type=int); ap.add_argument("seed_to", type=int)
    ap.add_argument("rho_list"); ap.add_argument("out")
    ap.add_argument("--knobs", default="", help="generator knobs as k=v,k=v (own_frac, deadline_frac, parent_frac, rig_mult, range_mi, core_mix, ...)")
    ap.add_argument("--decks", default="0.70:1.05:0.05", help="deck grid a:b:step")
    ap.add_argument("--pi", action=argparse.BooleanOptionalAction, default=True, help="perfect-information rung")
    ap.add_argument("--committed", action=argparse.BooleanOptionalAction, default=True, help="committed-plan rung")
    ap.add_argument("--tag", default="base", help="design label stored in every run (base, oat:<knob>=<value>, ...)")
    ap.add_argument("--price-market", action="store_true", help="value the long-dated price at the strip in every year and the whole producing "
                    "base inside the hedge (f_hedge = 1, pdp_hedge = 1): only the type curve, the field and per-well noise are charged for risk")
    ap.add_argument("--reuse-decks", default=None, help="a finished run file of the same instance at another rho: its strip plan and deck-grid plans "
                    "(risk-neutral MIPs, independent of rho) are reused instead of re-solved when the option table matches")
    ap.add_argument("--codevelop", action="store_true", help="co-development variant: a unit's two benches are developed in the same year or one bench only")
    ap.add_argument("--instance", default=None, help="pickled instance dict (e.g. the NDIC-calibrated instance) used instead of the generator; "
                    "seed_from/seed_to are ignored and the run's seed is the pickle's 'seed' (default 0)")
    args = ap.parse_args()
    args.decks = parse_decks(args.decks)
    over = parse_knobs(args.knobs)
    knobs = dict(BASE_KNOBS); knobs.update({k: v for k, v in over.items() if k in BASE_KNOBS})
    extra = {k: v for k, v in over.items() if k not in BASE_KNOBS}
    for k in extra:
        assert k in DEFAULTS, f"unknown knob {k}"
    if args.price_market:
        extra.update(f_hedge=(1.0, 1.0, 1.0, 1.0), pdp_hedge=1.0)
    if args.codevelop:
        extra.update(codevelop=True)
    rhos = [float(v) for v in args.rho_list.split(",")]
    res = dict(version=VERSION, nx=args.nx, ny=args.ny, args=dict(vars(args), decks=args.decks), rhos=rhos, knobs=dict(knobs, **extra),
               started=time.strftime("%Y-%m-%d %H:%M:%S"),
               threads=dict(gurobi=int(os.environ.get("GRB_THREADS", 2)), omp=os.environ.get("OMP_NUM_THREADS")),
               gurobi_memlimit_gb=float(os.environ.get("GRB_MEMLIMIT", 1.2)), rss_cap_gb=B.RSS_CAP_GB, decks=args.decks, oa=B.OA,
               mc_draws=B.MC_DRAWS, family=list(FAMILY), z_A=B.Z_A, oa_pi=OA_PI,
               perturbations=[dict(label=l, **kw) for l, kw in B.PERTURBATIONS], runs=[], errors=[])
    B.dump(res, args.out)
    seeds = range(args.seed_from, args.seed_to) if args.instance is None else [None]
    for s in seeds:
        try:
            if args.instance is None:
                I = make_instance(args.nx, args.ny, seed=s, **knobs, **extra)
            else:
                with open(args.instance, "rb") as fh:
                    I = pickle.load(fh)
                if "prob" not in I:
                    build_nodes(I, 3)
                s = int(I.get("seed", 0))
                knobs = {k: I[k] for k in BASE_KNOBS if k in I}
                res["instance_file"] = args.instance
                res["provenance"] = I.get("provenance", None)
        except Exception as e:
            traceback.print_exc()
            res["errors"] += [dict(seed=s, rho=rho, error=repr(e), time=0.0) for rho in rhos]
            B.dump(res, args.out)
            continue
        for rho in rhos:
            t0 = time.time()
            run = dict(seed=s, rho=rho, version=VERSION, status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"), errors=[], stages={})
            res["runs"].append(run)
            try:
                reuse = None
                if args.reuse_decks and os.path.exists(args.reuse_decks):
                    try:
                        rj = json.load(open(args.reuse_decks))
                        reuse = next((rr for rr in rj.get("runs", []) if rr.get("seed") == s and str(rr.get("status", "")).startswith("done")), None)
                        if reuse is not None:
                            reuse["_file"] = os.path.basename(args.reuse_decks)
                    except Exception as e:
                        print(f"  reuse: cannot read {args.reuse_decks}: {e!r}", flush=True)
                run["reuse_decks"] = args.reuse_decks if reuse is not None else None
                bench_lean(I, s, dict(knobs, **extra), rho, args, run, lambda: B.dump(res, args.out), reuse)
                n_err = len(run["errors"])
                run["status"] = "done" if n_err == 0 else f"done with {n_err} stage error(s)"
            except Exception as e:
                traceback.print_exc()
                run["status"] = "failed"
                run["errors"].append(B.error_entry("run", e, time.time() - t0))
                res["errors"].append(dict(seed=s, rho=rho, error=repr(e), time=time.time() - t0))
            run["time_total"] = time.time() - t0
            print(f"  seed {s} rho {rho:.0f} done in {run['time_total']:.0f}s ({run['status']})  peak RSS {B.rss_gb():.2f} GB", flush=True)
            B.dump(res, args.out)
    print("done", flush=True)
