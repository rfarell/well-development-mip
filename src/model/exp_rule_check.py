"""Rule comparator check (post hoc addition, 2026-11-02).

Tests whether the CE-tuned rule's first-year gap survives (a) a rule that prices lateral-mile completion capacity, (b) the
best of every distinct rule first year under certified continuation, (c) a fixed acreage footprint (NDIC case only:
acquisition budget zero, so no method buys a package) and (d) a decoupled selection that scores every first year with
the same continuation rule instead of the setting's own later drilling.

For each instance variant and risk tolerance:
  grid            every setting of the grid is built and scored: the registered 80 settings (decks 0.7-1.0, rig charges
                  0-0.4 $M/day, PI or NPV ranking, terminal credit deducted or not), crossed with a lateral-mile charge
                  of 0, 0.5, 1, 2 or 4 $M/mile when --grid extended (400 settings); distinct first years (z0, y, w0) are
                  collected; the NPV-tuned rule (highest E[NPV] of the complete plan; "practice" in the keys) and the CE-tuned rule (highest CE; "oracle")
                  are selected in the registered grid and in the extended grid
  decoupled       every distinct first year is given later years filled by the rule at its default setting (deck 1.0,
                  no charges, profitability index, no terminal deduction) and scored exactly; the decoupled selection
                  takes the first year with the highest such CE (no optimizer anywhere)
  rn              risk-neutral MIP, solved once per variant (it does not depend on rho) and re-scored at each rho, or
                  taken from the stored replicate run (--reuse-dir) when the option table matches
  ce              certified risk-averse plan, seeded with rn and the CE-tuned rules' plans, or taken from the stored run
  fs<i>           certified CE recourse for distinct first year i (first year fixed, continuation optimized)
  <sel>+rn_recourse  risk-neutral recourse for the practice-selected first years (the robust expected-NPV deficit)
  summary         first-year gaps CE(z_A) - CE(x + exact recourse) with the interval from the two certificates, in $M
                  and in % of E[NPV](rn), as in exp_replicates.vopt; ranks of the selected first years by certified CE;
                  expected-NPV deficits of the NPV-tuned rules

Usage (from src/model):
  NDIC case:  python exp_rule_check.py --instance ../../data/derived/ndic_mckenzie_40u.pkl --out results/ndic_rule_check.json
                                       [--rhos 100,300,1000] [--variants packages,fixed] [--grid extended]
  synthetic:  python exp_rule_check.py --seeds 100 --rhos 100 --grid registered --reuse-dir results/replicates
                                       --out results/rule_check/synth_s100_r100.json
  --resume     continue a run file: finished stages are restored from it, new stages run (adds `decoupled` to a finished file)
  --grid-only  enumerate, select and decouple only (no solver; for a dry run)
  --summarize  numpy only: the NDIC summary of --out (markdown next to it, macros to --macros), or with --glob the
               synthetic summary across run files (--md, --macros)
"""
import argparse
import glob
import itertools
import json
import os
import pickle
import time

import numpy as np

from instance import build_nodes
from model import Model, PlanBuilder
import baselines as BL
import bench_v3 as B

VERSION = "rule-check-1.1"
DECKS = (0.7, 0.8, 0.9, 1.0)
MUS_RIG = (0.0, 0.05, 0.1, 0.2, 0.4)
METRICS = ("pi", "npv")
TERMINALS = (False, True)
MUS_CREW = dict(registered=(0.0,), extended=(0.0, 0.5, 1.0, 2.0, 4.0))
DEFAULT_RULE = dict(deck=1.0, mu_rig=0.0, metric="pi", concept=True, terminal=False, mu_crew=0.0)   # baselines.recourse_fill defaults
RHO_WORD = {50: "Fifty", 100: "Hundred", 300: "ThreeHundred", 1000: "Thousand"}
VAR_WORD = {"packages": "", "fixed": "Fixed"}
SELECTIONS = ("reg_oracle", "ext_oracle", "reg_best", "ext_best", "reg_decoupled", "ext_decoupled")
PRACTICE = ("reg_practice", "ext_practice")
SEL_WORD = dict(reg_oracle="RegOracle", ext_oracle="ExtOracle", reg_best="RegBest", ext_best="ExtBest",
                reg_decoupled="RegDecoupled", ext_decoupled="ExtDecoupled", reg_practice="RegPractice", ext_practice="ExtPractice")
CERT_KEYS = ("ce", "enpv", "ce_lb", "ce_ub", "gap_pct", "rounds", "time", "solver_time", "bound_valid", "witness_refuted", "numeric_resolves", "feasible")


def settings(mus_crew):
    for deck, mu, metric, term, muc in itertools.product(DECKS, MUS_RIG, METRICS, TERMINALS, mus_crew):
        yield dict(deck=deck, mu_rig=mu, metric=metric, concept=True, terminal=term, mu_crew=muc)


def variant_instance(I, variant):
    if variant == "packages":
        return I
    if variant == "fixed":
        return dict(I, cap=dict(I["cap"], acq=0.0))
    raise ValueError(variant)


def year0(M, plan):
    """First-year use of the three capacities and the packages bought (year 0 is shared by every node)."""
    z0 = plan["z0"]
    return dict(wells0=float(z0 @ M.wells[M.idx0]), lat_miles0=float(z0 @ M.lat_miles[M.idx0]), rig_days0=float(z0 @ M.rigdays[M.idx0]),
                packages=int(round(float(plan["y"].sum()))), pkg_cost=float(M.I["pkg_cost"] @ plan["y"]))


def cfg_label(cfg):
    return f"deck {cfg['deck']}, rig {cfg['mu_rig']}, mile {cfg.get('mu_crew', 0.0)}, {cfg['metric']}, terminal {cfg['terminal']}"


def same_setting(a, b):
    return all(a.get(k) == b.get(k) for k in ("deck", "mu_rig", "metric", "terminal")) and a.get("mu_crew", 0.0) == b.get("mu_crew", 0.0)


def load_reuse(path, M):
    """The stored replicate run (one run per file) whose option table matches M, else None."""
    if not path or not os.path.exists(path):
        return None
    rj = json.load(open(path))
    runs = [r for r in rj.get("runs", []) if str(r.get("status", "")).startswith("done")]
    if not runs:
        return None
    r = runs[0]
    fmt = B.plan_format(M)
    if r.get("plan_format", {}).get("options_sha1") != fmt["options_sha1"]:
        print(f"  reuse: option table of {path} differs; solving afresh", flush=True)
        return None
    return dict(file=os.path.basename(path), run=r)


# ---------------------------------------------------------------------------------------------------------------------
def run_one(I, variant, rho, run, save, ctx):
    lam = 1.0 / rho
    M = Model(I, lam)
    resume = bool(ctx.get("resume")) and bool(run.get("stages"))
    run.update(dict(variant=variant, rho=rho, lam=lam, U=M.U, options=M.n, n0=M.n0, K=M.K, G=M.G, acq=float(I["cap"]["acq"]),
                    cap=dict(rig=float(I["cap"]["rig"][0]), crew=float(I["cap"]["crew"][0]), capital=float(I["cap"]["capital"][0])),
                    model_version=B.MODEL_VERSION if hasattr(B, "MODEL_VERSION") else None, solver_rev=B.SOLVER_REV,
                    methods=run.get("methods", {}) if resume else {}, plans=run.get("plans", {}) if resume else {}, plan_format=B.plan_format(M),
                    grid=ctx["grid"], mus_crew=list(MUS_CREW[ctx["grid"]])))
    R, P = run["methods"], {}
    stage = B.Stages(run, P, ctx, save)
    reuse = load_reuse(ctx.get("reuse_file"), M)
    run["reuse_file"] = reuse["file"] if reuse else None
    print(f"=== {variant} rho ${rho:,.0f}M  U={M.U} options={M.n} n0={M.n0} K={M.K} G={M.G} acq {I['cap']['acq']}  grid {ctx['grid']}"
          + (f"  reuse {reuse['file']}" if reuse else "") + ("  resume" if resume else ""), flush=True)

    def add(name, plan, extra=None, label=None):
        row = B.score(M, name, plan, dict(extra or {}, **year0(M, plan)), label)
        P[name], R[name], run["plans"][name] = plan, row, B.sparse_plan(plan)
        return row

    def st(name, fn, needs=(), restore=None):
        """Run a stage, or restore its outputs from the file when resuming and the stage finished."""
        rec = run["stages"].get(name)
        if resume and rec and rec.get("ok") and restore is not None and restore():
            print(f"  [{name}] restored", flush=True)
            return
        stage(name, fn, needs=needs)

    def row_restore(name):
        def f():
            if name in R and name in run["plans"]:
                P[name] = B.dense_plan(M, run["plans"][name])
                return True
            return False
        return f

    # ---- grid: every setting, distinct first years, selections ------------------------------------------------------
    def grid():
        rows, fs, index = [], [], {}
        with B.Clock() as c:
            for i, cfg in enumerate(settings(MUS_CREW[ctx["grid"]])):
                plan = BL.heuristic_plan(M, **cfg)
                ev = M.evaluate(plan)
                key = B.first_stage_key(plan)
                if key not in index:
                    index[key] = len(fs)
                    fs.append(dict(id=len(fs), settings=[], registered=False, ce_complete=-np.inf, enpv_complete=-np.inf, best_setting=None, **year0(M, plan)))
                e = fs[index[key]]
                e["settings"].append(i)
                e["registered"] = bool(e["registered"] or cfg["mu_crew"] == 0.0)
                if ev["ce"] > e["ce_complete"]:
                    e["ce_complete"], e["best_setting"] = float(ev["ce"]), i
                    ctx[f"fsplan{e['id']}"] = plan                       # the complete plan of the first year's best setting (its seed)
                e["enpv_complete"] = max(e["enpv_complete"], float(ev["enpv"]))
                rows.append(dict(i=i, **cfg, ce=float(ev["ce"]), enpv=float(ev["enpv"]), fs=index[key]))
        for e in fs:
            run["plans"][f"fs{e['id']}"] = B.sparse_plan(ctx[f"fsplan{e['id']}"])
        reg = [r for r in rows if r["mu_crew"] == 0.0]
        pick = lambda sub, by: max(sub, key=lambda r: r[by])
        sel = dict(reg_oracle=pick(reg, "ce"), reg_practice=pick(reg, "enpv"), ext_oracle=pick(rows, "ce"), ext_practice=pick(rows, "enpv"))
        run["settings"], run["first_years"] = rows, fs
        run["selected"] = {k: dict(setting=v["i"], fs=v["fs"], ce=v["ce"], enpv=v["enpv"]) for k, v in sel.items()}
        run["counts"] = dict(settings=len(rows), settings_registered=len(reg), first_years_registered=int(sum(e["registered"] for e in fs)),
                             first_years_extended=len(fs), grid_time=c.t)
        for k, v in sel.items():
            P[k] = ctx[f"fsplan{v['fs']}"]
        print(f"  grid: {len(rows)} settings in {c.t:.1f}s; distinct first years {run['counts']['first_years_registered']} registered, {len(fs)} extended", flush=True)
        for k, v in sel.items():
            print(f"  {k:13s} setting {v['i']:3d} ({cfg_label(v)}) fs {v['fs']:3d}  CE {v['ce']:.2f}  E[NPV] {v['enpv']:.2f}", flush=True)
        if reuse:                                                       # reproduction of the stored oracle rule (registered grid)
            s = reuse["run"]["methods"].get("heur_ce", {})
            our = run["settings"][sel["reg_oracle"]["i"]]
            key_stored = B.first_stage_key(B.dense_plan(M, reuse["run"]["plans"]["heur_ce"])) if "heur_ce" in reuse["run"]["plans"] else None
            run["stored"] = dict(file=reuse["file"], heur_ce_cfg=s.get("cfg"), heur_ce_ce=s.get("ce"), heur_ce_enpv=s.get("enpv"),
                                 setting_match=bool(s.get("cfg") and same_setting(our, s["cfg"])),
                                 first_year_match=bool(key_stored is not None and key_stored == B.first_stage_key(P["reg_oracle"])),
                                 ce_match=bool(s.get("ce") is not None and abs(float(s["ce"]) - sel["reg_oracle"]["ce"]) < 1e-4),
                                 vopt={k: v for k, v in reuse["run"].get("vopt", {}).items() if k.startswith("fs_gap") or k.startswith("vopt")},
                                 exact_recourse={k: reuse["run"]["methods"].get("heur_ce+exact_recourse", {}).get(k) for k in ("ce_lb", "ce_ub", "gap_pct", "time")},
                                 ce={k: reuse["run"]["methods"].get("ce", {}).get(k) for k in ("ce_lb", "ce_ub", "gap_pct", "time")},
                                 enpv_rn=reuse["run"]["methods"].get("rn", {}).get("enpv"))
            print(f"  stored oracle rule: {cfg_label(s.get('cfg', {}))} CE {s.get('ce', float('nan')):.2f}; setting match {run['stored']['setting_match']}, "
                  f"first year match {run['stored']['first_year_match']}, CE match {run['stored']['ce_match']}", flush=True)

    def grid_restore():
        if not all(k in run for k in ("settings", "first_years", "selected", "counts")):
            return False
        for e in run["first_years"]:
            ctx[f"fsplan{e['id']}"] = B.dense_plan(M, run["plans"][f"fs{e['id']}"])
        for k, v in run["selected"].items():
            P[k] = ctx[f"fsplan{v['fs']}"]
        return True

    st("grid", grid, restore=grid_restore)
    if "first_years" not in run:
        return run

    # ---- decoupled selection: every first year with the default rule's continuation (no optimizer) ------------------
    def decoupled():
        fs = run["first_years"]
        with B.Clock() as c:
            for e in fs:
                src = ctx[f"fsplan{e['id']}"]
                bld = PlanBuilder(M)
                for g in np.flatnonzero(src["y"] > 0.5):
                    bld.buy(int(g))
                for i in np.flatnonzero(src["z0"] > 0.5):
                    o = int(M.idx0[i])
                    assert bld.can_add(o, None), f"first year of fs{e['id']} cannot be rebuilt (option {o})"
                    bld.add(o, None)
                for k in range(M.K):
                    BL.recourse_fill(M, bld, k, **DEFAULT_RULE)
                plan = bld.plan
                ev = M.evaluate(plan)
                e["decoupled"] = dict(ce=float(ev["ce"]), enpv=float(ev["enpv"]), same_first_year=bool(B.first_stage_key(plan) == B.first_stage_key(src)),
                                      feasible=bool(M.feasible(plan)))
        bad = [e["id"] for e in fs if not (e["decoupled"]["same_first_year"] and e["decoupled"]["feasible"])]
        if bad:
            print(f"  decoupled: first year not reproduced or infeasible for fs {bad}; those are excluded from the selection", flush=True)
        ok = [e for e in fs if e["decoupled"]["same_first_year"] and e["decoupled"]["feasible"]]
        pick = lambda sub: max(sub, key=lambda e: e["decoupled"]["ce"])
        for name, sub in (("reg_decoupled", [e for e in ok if e["registered"]]), ("ext_decoupled", ok)):
            if sub:
                e = pick(sub)
                run["selected"][name] = dict(setting=e["best_setting"], fs=e["id"], ce=e["decoupled"]["ce"], enpv=e["decoupled"]["enpv"], decoupled=True)
                print(f"  {name:13s} fs {e['id']:3d} (first year of setting {e['best_setting']})  decoupled CE {e['decoupled']['ce']:.2f}  E[NPV] {e['decoupled']['enpv']:.2f}", flush=True)
        run["counts"]["decoupled_time"] = c.t
        run["counts"]["decoupled_excluded"] = len(bad)

    stage("decoupled", decoupled)
    if ctx.get("grid_only"):
        return run

    from solve import solve_ce, solve_risk_neutral, solve_recourse_exact

    # ---- risk-neutral MIP, once per variant (or the stored plan) ----------------------------------------------------
    def rn():
        if reuse and "rn" in reuse["run"]["plans"]:
            s = reuse["run"]["methods"]["rn"]
            row = add("rn", B.dense_plan(M, reuse["run"]["plans"]["rn"]), dict(time=0.0, bound=s.get("bound"), obj=s.get("obj"), mip_gap=s.get("mip_gap"),
                                                                              reused_from=reuse["file"], time_original=s.get("time")),
                      f"risk-neutral two-stage MIP at the strip (plan and bound from {reuse['file']})")
            assert abs(row["enpv"] - float(s["enpv"])) < 1e-3, f"stored rn plan scores {row['enpv']:.4f} here against {s['enpv']:.4f} stored"
            return
        cached = ctx.get(("rn", variant))
        if cached is None:
            with B.Clock() as c:
                r = solve_risk_neutral(M)
            cached = dict(plan=r["plan"], obj=r["obj"], bound=r["bound"], gap=r["gap"], time=c.t)
            ctx[("rn", variant)] = cached
            extra = dict(time=c.t, mip_gap=r["gap"], bound=r["bound"], obj=r["obj"])
        else:
            extra = dict(time=0.0, mip_gap=cached["gap"], bound=cached["bound"], obj=cached["obj"], reused=True, time_original=cached["time"])
        add("rn", cached["plan"], extra, "risk-neutral two-stage MIP at the strip" + (" (plan reused: it does not depend on rho)" if "reused" in extra else ""))

    st("rn", rn, restore=row_restore("rn"))

    # ---- certified risk-averse plan (or the stored one) -------------------------------------------------------------
    def ce():
        if reuse and "ce" in reuse["run"]["plans"]:
            s = reuse["run"]["methods"]["ce"]
            extra = {k: s.get(k) for k in ("ce_lb", "ce_ub", "gap_pct", "rounds", "bound_valid", "witness_refuted", "numeric_resolves", "solver_rev")}
            row = add("ce", B.dense_plan(M, reuse["run"]["plans"]["ce"]), dict(extra, time=0.0, reused_from=reuse["file"], time_original=s.get("time")),
                      f"certified risk-averse plan from {reuse['file']}")
            assert abs(row["ce"] - float(s["ce_lb"])) < 1e-3, f"stored z_A scores {row['ce']:.4f} here against {s['ce_lb']:.4f} stored"
            return
        seeds = [P["rn"], P["reg_oracle"]] + ([P["ext_oracle"]] if run["selected"]["ext_oracle"]["fs"] != run["selected"]["reg_oracle"]["fs"] else [])
        with B.Clock() as c:
            r = solve_ce(M, seeds=tuple(seeds), **B.OA)
        add("ce", r["plan"], B.cert_extra(r, c.t), "certified CE solver seeded with rn and the oracle rules")

    st("ce", ce, needs=("rn", "reg_oracle"), restore=row_restore("ce"))

    # ---- certified recourse for every distinct first year -----------------------------------------------------------
    def exact(e):
        def fn():
            plan = ctx[f"fsplan{e['id']}"]
            with B.Clock() as c:
                r = solve_recourse_exact(M, plan, **B.OA)
            cfg = run["settings"][e["best_setting"]]
            row = add(f"fs{e['id']}", r["plan"], B.cert_extra(r, c.t), f"first year of rule setting {e['best_setting']} ({cfg_label(cfg)}), certified CE recourse")
            e["cert"] = {k: row[k] for k in CERT_KEYS}
        return fn

    for e in run["first_years"]:
        st(f"fs{e['id']}", exact(e), restore=lambda e=e: "cert" in e and row_restore(f"fs{e['id']}")())

    # ---- risk-neutral recourse for the practice-selected first years ------------------------------------------------
    def rn_rec(name):
        def fn():
            with B.Clock() as c:
                r = solve_risk_neutral(M, fixed=P[name])
            add(f"{name}+rn_recourse", r["plan"], dict(time=c.t, mip_gap=r["gap"], bound=r["bound"], obj=r["obj"]), f"first year of {name}, risk-neutral recourse")
        return fn

    st("reg_practice+rn_recourse", rn_rec("reg_practice"), needs=("reg_practice",), restore=row_restore("reg_practice+rn_recourse"))
    if run["selected"]["ext_practice"]["fs"] != run["selected"]["reg_practice"]["fs"]:
        st("ext_practice+rn_recourse", rn_rec("ext_practice"), needs=("ext_practice",), restore=row_restore("ext_practice+rn_recourse"))
    elif "reg_practice+rn_recourse" in R:
        R["ext_practice+rn_recourse"] = dict(R["reg_practice+rn_recourse"], label="same first year as reg_practice")

    stage("summary", lambda: summarize_run(run), needs=("rn", "ce"))
    return run


# ---------------------------------------------------------------------------------------------------------------------
def summarize_run(run):
    """Gaps, ranks and deficits of a finished run, in $M and % of E[NPV](rn); stored in run['summary']."""
    from scipy.stats import spearmanr
    R, fs, sel = run["methods"], run["first_years"], run["selected"]
    zA, rn = R["ce"], R["rn"]
    den = max(1e-9, abs(rn["enpv"]))
    pct = lambda x: 100.0 * x / den
    out = dict(enpv_rn=rn["enpv"], ce_rn=rn["ce"], bound_rn=rn.get("bound"), ce_zA=zA["ce_lb"], ce_zA_ub=zA["ce_ub"], enpv_zA=zA["enpv"], gap_pct_zA=zA["gap_pct"],
               zA_year0={k: zA[k] for k in ("wells0", "lat_miles0", "rig_days0", "packages", "pkg_cost")},
               rn_year0={k: rn[k] for k in ("wells0", "lat_miles0", "rig_days0", "packages", "pkg_cost")},
               ce_gain=zA["ce_lb"] - rn["ce"], ce_gain_pct=pct(zA["ce_lb"] - rn["ce"]), counts=run["counts"])
    certified = [e for e in fs if "cert" in e and e["cert"].get("bound_valid", True) and e["cert"].get("feasible", True)]
    regc = [e for e in certified if e["registered"]]

    def rank_of(e, pool):
        return 1 + int(sum(x["cert"]["ce_lb"] > e["cert"]["ce_lb"] + 1e-6 for x in pool))

    def gap(e, pool):
        c, cfg = e["cert"], run["settings"][e["best_setting"]]
        g = zA["ce_lb"] - c["ce_lb"]
        d = dict(fs=e["id"], setting=cfg, n_settings=len(e["settings"]), registered=e["registered"], ce_complete=e["ce_complete"], enpv_complete=e["enpv_complete"],
                 ce_cert=c["ce_lb"], ce_cert_ub=c["ce_ub"], enpv_cert=c["enpv"], cert_gap_pct=c["gap_pct"],
                 fs_gap=g, fs_gap_lb=zA["ce_lb"] - c["ce_ub"], fs_gap_ub=zA["ce_ub"] - c["ce_lb"],
                 fs_gap_pct=pct(g), fs_gap_lb_pct=pct(zA["ce_lb"] - c["ce_ub"]), fs_gap_ub_pct=pct(zA["ce_ub"] - c["ce_lb"]),
                 rank=rank_of(e, pool), n_pool=len(pool), year0={k: e[k] for k in ("wells0", "lat_miles0", "rig_days0", "packages", "pkg_cost")})
        if "decoupled" in e:
            d.update(plan_ce=e["decoupled"]["ce"], plan_enpv=e["decoupled"]["enpv"], plan_gap=zA["ce_lb"] - e["decoupled"]["ce"], plan_gap_pct=pct(zA["ce_lb"] - e["decoupled"]["ce"]))
        return d

    for name, pool in (("reg_oracle", regc), ("ext_oracle", certified), ("reg_decoupled", regc), ("ext_decoupled", certified)):
        if name in sel:
            e = fs[sel[name]["fs"]]
            if "cert" in e:
                out[name] = gap(e, pool)
    if regc:
        out["reg_best"] = gap(max(regc, key=lambda e: e["cert"]["ce_lb"]), regc)
    if certified:
        out["ext_best"] = gap(max(certified, key=lambda e: e["cert"]["ce_lb"]), certified)
    if "reg_oracle" in out:
        ref = out["reg_oracle"]["ce_cert"]
        out["n_reg_better"] = int(sum(e["cert"]["ce_lb"] > ref + 1e-6 for e in regc))
        out["n_ext_better"] = int(sum(e["cert"]["ce_lb"] > ref + 1e-6 for e in certified))
        out["n_certified"] = len(certified)
        out["n_certified_registered"] = len(regc)
    sp = {}
    for lab, pool in (("reg", regc), ("ext", certified)):
        if len(pool) >= 3:
            cert_ce = [e["cert"]["ce_lb"] for e in pool]
            sp[f"complete_{lab}"] = float(spearmanr([e["ce_complete"] for e in pool], cert_ce)[0])
            if all("decoupled" in e for e in pool):
                sp[f"decoupled_{lab}"] = float(spearmanr([e["decoupled"]["ce"] for e in pool], cert_ce)[0])
    out["spearman"] = sp
    for name in PRACTICE:
        s = sel[name]
        d = dict(setting=run["settings"][s["setting"]], fs=s["fs"], enpv=s["enpv"], ce=s["ce"], vopt=rn["enpv"] - s["enpv"], vopt_pct=pct(rn["enpv"] - s["enpv"]))
        row = R.get(name + "+rn_recourse")
        if row:
            d.update(enpv_rn_recourse=row["enpv"], vopt_robust=rn["enpv"] - row["enpv"], vopt_robust_pct=pct(rn["enpv"] - row["enpv"]),
                     vopt_robust_lb=rn["enpv"] - row["bound"], vopt_robust_ub=rn["bound"] - row["enpv"],
                     vopt_robust_lb_pct=pct(rn["enpv"] - row["bound"]), vopt_robust_ub_pct=pct(rn["bound"] - row["enpv"]))
        e = fs[s["fs"]]
        if "cert" in e:
            d.update(ce_cert=e["cert"]["ce_lb"], fs_gap=zA["ce_lb"] - e["cert"]["ce_lb"], fs_gap_pct=pct(zA["ce_lb"] - e["cert"]["ce_lb"]))
        out[name] = d
    if run.get("stored"):
        s = run["stored"]
        out["stored"] = dict(s, fs_gap_diff_pct=(out["reg_oracle"]["fs_gap_pct"] - s["vopt"]["fs_gap_ce_pct"]) if "reg_oracle" in out and "fs_gap_ce_pct" in s.get("vopt", {}) else None)
    run["summary"] = out
    print("  summary: " + ", ".join(f"{k} gap {out[k]['fs_gap_pct']:.2f}% (rank {out[k]['rank']} of {out[k]['n_pool']})" for k in SELECTIONS if k in out)
          + f"; CE(z_A) {out['ce_zA']:.2f}, E[NPV](rn) {out['enpv_rn']:.1f}; first years better than the registered oracle's: "
          f"{out.get('n_reg_better', '--')} of {out.get('n_certified_registered', '--')} registered, {out.get('n_ext_better', '--')} of {out.get('n_certified', '--')} extended"
          + (f"; stored gap {out['stored']['vopt'].get('fs_gap_ce_pct', float('nan')):.2f}% (diff {out['stored']['fs_gap_diff_pct']:+.3f})" if out.get("stored") and out["stored"].get("fs_gap_diff_pct") is not None else ""), flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------------------
def fnum(x, d=2):
    try:
        return "--" if x is None or not np.isfinite(float(x)) else f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "--"


def write_summary(res, md_path, tex_path):
    """The NDIC summary: one markdown table per variant and the \\ndicrule macros."""
    runs = [r for r in res["runs"] if r.get("summary")]
    n_reg, n_ext = len(list(settings(MUS_CREW["registered"]))), len(list(settings(MUS_CREW["extended"])))
    mile_charges = ", ".join(f"{m:g}" for m in MUS_CREW["extended"][1:])
    mile_charges = " and ".join(mile_charges.rsplit(", ", 1))
    L = [f"# NDIC rule check ({res['version']}, started {res['started']}; {len(runs)} finished runs of {len(res['runs'])})", "",
         "Post hoc comparator check on the NDIC-calibrated case. Gaps are "
         "CE(z_A) minus the certified CE of the first year with optimized continuation, in % of E[NPV](rn) [certificate interval]. "
         "`reg` = the registered 80-setting grid; `ext` = the same grid crossed with a lateral-mile charge of 0, 0.5, 1, 2 or 4 $M/mile (400 settings). "
         "`oracle` = the CE-tuned rule, the setting with the highest complete-plan CE; `best` = the smallest gap over every distinct first year of the grid under certified continuation; "
         "`decoupled` = the first year with the highest CE when every first year's later years are filled by the rule at its default setting (no optimizer; "
         "`plan gap` is the gap of that complete plan itself); `rank` = position of the selected first year among the grid's distinct first years by certified CE; "
         "`practice` = the NPV-tuned rule, the setting with the highest complete-plan E[NPV], with its expected-NPV deficit and the robust form (risk-neutral continuation).", ""]
    T = ["% generated by exp_rule_check.py --summarize (NDIC rule check, post hoc 2026-11-02); do not edit",
         f"\\newcommand{{\\ndicruleNSettingsReg}}{{{n_reg}}}", f"\\newcommand{{\\ndicruleNSettingsExt}}{{{n_ext}}}",
         f"\\newcommand{{\\ndicruleMileCharges}}{{{mile_charges}}}"]
    for variant in ("packages", "fixed"):
        Q = sorted([r for r in runs if r["variant"] == variant], key=lambda r: r["rho"])
        if not Q:
            continue
        L += [f"## Variant `{variant}` (acquisition budget {fnum(Q[0]['acq'], 0)} $M)", "", "| quantity | " + " | ".join(f"rho {r['rho']:.0f}" for r in Q) + " |", "|---|" + "---|" * len(Q)]

        def row(label, cells):
            L.append(f"| {label} | " + " | ".join(cells) + " |")

        for name in SELECTIONS:
            row(f"{name} first-year gap, % [interval]", [f"{fnum(r['summary'][name]['fs_gap_pct'])} [{fnum(r['summary'][name]['fs_gap_lb_pct'])}, {fnum(r['summary'][name]['fs_gap_ub_pct'])}]" if name in r["summary"] else "--" for r in Q])
            row(f"{name} first-year gap, $M", [fnum(r["summary"][name]["fs_gap"]) if name in r["summary"] else "--" for r in Q])
            row(f"{name} rank by certified CE", [f"{r['summary'][name]['rank']} of {r['summary'][name]['n_pool']}" if name in r["summary"] else "--" for r in Q])
            if name.endswith("decoupled"):
                row(f"{name} plan gap (no optimizer), %", [fnum(r["summary"][name].get("plan_gap_pct")) if name in r["summary"] else "--" for r in Q])
            row(f"{name} setting", [cfg_label(r["summary"][name]["setting"]) if name in r["summary"] else "--" for r in Q])
            row(f"{name} year 0: wells / lateral miles / rig days / packages", [" / ".join(fnum(r["summary"][name]["year0"][k], 0) for k in ("wells0", "lat_miles0", "rig_days0", "packages")) if name in r["summary"] else "--" for r in Q])
        for name in PRACTICE:
            row(f"{name} expected-NPV deficit, % (robust [interval])", [f"{fnum(r['summary'][name]['vopt_pct'])} ({fnum(r['summary'][name].get('vopt_robust_pct'))} [{fnum(r['summary'][name].get('vopt_robust_lb_pct'))}, {fnum(r['summary'][name].get('vopt_robust_ub_pct'))}])" for r in Q])
            row(f"{name} setting", [cfg_label(r["summary"][name]["setting"]) for r in Q])
        row("z_A year 0: wells / lateral miles / rig days / packages", [" / ".join(fnum(r["summary"]["zA_year0"][k], 0) for k in ("wells0", "lat_miles0", "rig_days0", "packages")) for r in Q])
        row("rn year 0: wells / lateral miles / rig days / packages", [" / ".join(fnum(r["summary"]["rn_year0"][k], 0) for k in ("wells0", "lat_miles0", "rig_days0", "packages")) for r in Q])
        row("capacity year 0: lateral miles / rig days", [f"{fnum(r['cap']['crew'], 1)} / {fnum(r['cap']['rig'], 0)}" for r in Q])
        row("E[NPV](rn), CE(z_A) [interval], CE gain %", [f"{fnum(r['summary']['enpv_rn'], 1)}, {fnum(r['summary']['ce_zA'])} [{fnum(r['summary']['ce_zA'])}, {fnum(r['summary']['ce_zA_ub'])}], {fnum(r['summary']['ce_gain_pct'])}" for r in Q])
        row("distinct first years certified: registered / extended", [f"{r['summary'].get('n_certified_registered', '--')} / {r['summary'].get('n_certified', '--')}" for r in Q])
        row("first years with a certified CE above the registered oracle's: registered / extended", [f"{r['summary'].get('n_reg_better', '--')} / {r['summary'].get('n_ext_better', '--')}" for r in Q])
        row("Spearman(complete-plan CE, certified CE), registered / extended", [f"{fnum(r['summary'].get('spearman', {}).get('complete_reg'))} / {fnum(r['summary'].get('spearman', {}).get('complete_ext'))}" for r in Q])
        row("Spearman(decoupled CE, certified CE), registered / extended", [f"{fnum(r['summary'].get('spearman', {}).get('decoupled_reg'))} / {fnum(r['summary'].get('spearman', {}).get('decoupled_ext'))}" for r in Q])
        row("errors", [str(len(r.get("errors", []))) for r in Q])
        L.append("")
        for r in Q:
            S, w = r["summary"], VAR_WORD[variant] + RHO_WORD.get(int(r["rho"]), str(int(r["rho"])))
            for name in SELECTIONS:
                if name in S:
                    s = SEL_WORD[name]
                    T += [f"\\newcommand{{\\ndicrule{s}Gap{w}}}{{{fnum(S[name]['fs_gap_pct'])}}}", f"\\newcommand{{\\ndicrule{s}GapDollar{w}}}{{{fnum(S[name]['fs_gap'])}}}",
                          f"\\newcommand{{\\ndicrule{s}GapLo{w}}}{{{fnum(S[name]['fs_gap_lb_pct'])}}}", f"\\newcommand{{\\ndicrule{s}GapHi{w}}}{{{fnum(S[name]['fs_gap_ub_pct'])}}}",
                          f"\\newcommand{{\\ndicrule{s}Rank{w}}}{{{S[name]['rank']}}}",
                          f"\\newcommand{{\\ndicrule{s}MileCharge{w}}}{{{fnum(S[name]['setting']['mu_crew'], 1)}}}", f"\\newcommand{{\\ndicrule{s}RigCharge{w}}}{{{fnum(S[name]['setting']['mu_rig'], 2)}}}",
                          f"\\newcommand{{\\ndicrule{s}Deck{w}}}{{{fnum(S[name]['setting']['deck'], 1)}}}",
                          f"\\newcommand{{\\ndicrule{s}Wells{w}}}{{{fnum(S[name]['year0']['wells0'], 0)}}}", f"\\newcommand{{\\ndicrule{s}LatMiles{w}}}{{{fnum(S[name]['year0']['lat_miles0'], 0)}}}",
                          f"\\newcommand{{\\ndicrule{s}RigDays{w}}}{{{fnum(S[name]['year0']['rig_days0'], 0)}}}", f"\\newcommand{{\\ndicrule{s}Packages{w}}}{{{fnum(S[name]['year0']['packages'], 0)}}}"]
                    if "plan_gap_pct" in S[name]:
                        T += [f"\\newcommand{{\\ndicrule{s}PlanGap{w}}}{{{fnum(S[name]['plan_gap_pct'])}}}", f"\\newcommand{{\\ndicrule{s}PlanGapDollar{w}}}{{{fnum(S[name]['plan_gap'])}}}"]
            for name in PRACTICE:
                s = SEL_WORD[name]
                T += [f"\\newcommand{{\\ndicrule{s}Vopt{w}}}{{{fnum(S[name]['vopt_pct'])}}}", f"\\newcommand{{\\ndicrule{s}VoptDollar{w}}}{{{fnum(S[name]['vopt'])}}}",
                      f"\\newcommand{{\\ndicrule{s}VoptRobust{w}}}{{{fnum(S[name].get('vopt_robust_pct'))}}}", f"\\newcommand{{\\ndicrule{s}VoptRobustDollar{w}}}{{{fnum(S[name].get('vopt_robust'))}}}",
                      f"\\newcommand{{\\ndicrule{s}MileCharge{w}}}{{{fnum(S[name]['setting']['mu_crew'], 1)}}}"]
            T += [f"\\newcommand{{\\ndicruleEnpvRn{w}}}{{{fnum(S['enpv_rn'], 1)}}}", f"\\newcommand{{\\ndicruleCeZa{w}}}{{{fnum(S['ce_zA'])}}}", f"\\newcommand{{\\ndicruleCeGain{w}}}{{{fnum(S['ce_gain_pct'])}}}",
                  f"\\newcommand{{\\ndicruleZaWells{w}}}{{{fnum(S['zA_year0']['wells0'], 0)}}}", f"\\newcommand{{\\ndicruleZaLatMiles{w}}}{{{fnum(S['zA_year0']['lat_miles0'], 0)}}}",
                  f"\\newcommand{{\\ndicruleZaRigDays{w}}}{{{fnum(S['zA_year0']['rig_days0'], 0)}}}", f"\\newcommand{{\\ndicruleZaPackages{w}}}{{{fnum(S['zA_year0']['packages'], 0)}}}",
                  f"\\newcommand{{\\ndicruleCapLatMiles{w}}}{{{fnum(r['cap']['crew'], 1)}}}", f"\\newcommand{{\\ndicruleCapRigDays{w}}}{{{fnum(r['cap']['rig'], 0)}}}",
                  f"\\newcommand{{\\ndicruleNFsReg{w}}}{{{S.get('n_certified_registered', '--')}}}", f"\\newcommand{{\\ndicruleNFsExt{w}}}{{{S.get('n_certified', '--')}}}",
                  f"\\newcommand{{\\ndicruleNRegBetter{w}}}{{{S.get('n_reg_better', '--')}}}", f"\\newcommand{{\\ndicruleNExtBetter{w}}}{{{S.get('n_ext_better', '--')}}}",
                  f"\\newcommand{{\\ndicruleSpearmanComplete{w}}}{{{fnum(S.get('spearman', {}).get('complete_reg'))}}}",
                  f"\\newcommand{{\\ndicruleSpearmanDecoupled{w}}}{{{fnum(S.get('spearman', {}).get('decoupled_reg'))}}}"]
    if res.get("errors"):
        L += ["## Run-level errors", ""] + [f"- {e}" for e in res["errors"]] + [""]
    open(md_path, "w").write("\n".join(L) + "\n")
    if tex_path:
        open(tex_path, "w").write("\n".join(T) + "\n")
    print(f"wrote {md_path}" + (f" and {tex_path}" if tex_path else ""), flush=True)


def boot_mean(x, n=4000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan"),) * 3
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def program_value(run, rep_dir):
    """E(z_N) - E(0) of the area in $M, as in summarize_replicates.quantities: E(0) is base_value.enpv0 of the stored
    replicate run. NaN when that run is not found."""
    name = run.get("reuse_file") or run.get("summary", {}).get("stored", {}).get("file")
    path = os.path.join(rep_dir, os.path.basename(name)) if name else None
    if not path or not os.path.exists(path):
        return float("nan")
    for r in json.load(open(path)).get("runs", []):
        if float(r.get("rho", -1)) == float(run["rho"]) and "enpv0" in r.get("base_value", {}):
            return float(run["summary"]["enpv_rn"]) - float(r["base_value"]["enpv0"])
    return float("nan")


def fixed_setting_gaps(Q):
    """First-year gap, in % of E[NPV](rn), of every registered setting used unchanged on all the areas of Q: an array
    with one row per setting and one column per area, defined as fs_gap_pct in summarize_run. None if a run's grid
    differs from the registered grid."""
    grid = list(settings(MUS_CREW["registered"]))
    gaps = [[] for _ in grid]
    for r in Q:
        S, fy = r["summary"], {e["id"]: e for e in r["first_years"]}
        rows = [x for x in r["settings"] if x["mu_crew"] == 0.0]
        if [{k: x[k] for k in grid[0]} for x in rows] != grid:
            return grid, None
        for j, x in enumerate(rows):
            c = fy[x["fs"]].get("cert", {})
            ok = "ce_lb" in c and c.get("bound_valid", True) and c.get("feasible", True)
            gaps[j].append(100.0 * (S["ce_zA"] - c["ce_lb"]) / max(1e-9, abs(S["enpv_rn"])) if ok else float("nan"))
    return grid, np.array(gaps, dtype=float)


def write_summary_synth(files, md_path, tex_path, rep_dir="results/replicates"):
    """The synthetic summary across one-run files: per tolerance, the registered oracle's gap (and its reproduction of the
    stored replicate run), the best first year's gap, the decoupled selection's gap and ranks; the \\synthrule macros.
    Each first-year gap is also given in % of program value, E(z_N) - E(0), with E(0) from the replicate runs in rep_dir."""
    runs = []
    for f in sorted(files):
        rj = json.load(open(f))
        for r in rj.get("runs", []):
            if r.get("summary"):
                runs.append(dict(r, _file=os.path.basename(f), _seed=rj.get("seed")))
    rhos = sorted({int(r["rho"]) for r in runs})
    L = [f"# Synthetic rule check ({VERSION}; {len(runs)} finished runs from {len(files)} files)", "",
         "Post hoc comparator check on the synthetic base areas (registered 80-setting grid; rn and the certified risk-averse plan taken from the stored "
         "replicate runs when the option table matches). Gaps are CE(z_A) minus the certified CE of the first year with optimized continuation, in % of "
         "E[NPV](rn). `oracle` = the setting with the highest complete-plan CE (the paper's CE-tuned rule; `stored` is its first-year gap in the replicate "
         "run file); `best` = the smallest gap over every distinct first year of the grid; `decoupled` = the first year with the highest CE when every "
         "first year's later years are filled by the rule at its default setting (no optimizer); `plan gap` is the gap of that decoupled plan itself; "
         "ranks are by certified CE among the grid's distinct first years. Means carry a 4,000-draw bootstrap 95% interval.", ""]
    T = ["% generated by exp_rule_check.py --summarize --glob (synthetic rule check, post hoc 2026-11-02); do not edit"]
    for rho in rhos:
        Q = sorted([r for r in runs if int(r["rho"]) == rho], key=lambda r: r["_seed"])
        w = RHO_WORD.get(rho, str(rho))
        g = lambda name, key="fs_gap_pct": np.array([r["summary"][name][key] for r in Q if name in r["summary"]], dtype=float)
        n = len(Q)
        stats = {}
        for name in ("reg_oracle", "reg_best", "reg_decoupled"):
            x = g(name)
            if len(x):
                mean, lo, hi = boot_mean(x)
                stats[name] = dict(n=len(x), mean=mean, lo=lo, hi=hi, median=float(np.median(x)), min=float(x.min()), max=float(x.max()),
                                   dollar_mean=float(g(name, "fs_gap").mean()), resolved=int((g(name, "fs_gap_lb_pct") > 0).sum()))
        pv = np.array([program_value(r, rep_dir) for r in Q], dtype=float)
        if len(pv) and np.all(np.isfinite(pv)) and np.all(pv > 0):
            for name in ("reg_oracle", "reg_best", "reg_decoupled"):
                if name in stats and all(name in r["summary"] for r in Q):
                    stats[name]["pv_mean"] = float((100.0 * g(name, "fs_gap") / pv).mean())
        else:
            print(f"WARNING: rho {rho}: no program value for {int((~(np.isfinite(pv) & (pv > 0))).sum())} of {len(pv)} areas "
                  f"(replicate runs not found in {rep_dir}); the GapPvMean macros are NOT written", flush=True)
        plan = g("reg_decoupled", "plan_gap_pct")
        if len(plan):
            mean, lo, hi = boot_mean(plan)
            stats["reg_decoupled_plan"] = dict(n=len(plan), mean=mean, lo=lo, hi=hi, median=float(np.median(plan)), min=float(plan.min()), max=float(plan.max()))
        stored = np.array([r["summary"]["stored"]["vopt"]["fs_gap_ce_pct"] for r in Q if r["summary"].get("stored") and "fs_gap_ce_pct" in r["summary"]["stored"].get("vopt", {})], dtype=float)
        diff = np.array([r["summary"]["stored"]["fs_gap_diff_pct"] for r in Q if r["summary"].get("stored") and r["summary"]["stored"].get("fs_gap_diff_pct") is not None], dtype=float)
        paired = g("reg_oracle") - g("reg_best") if len(g("reg_oracle")) == len(g("reg_best")) else np.array([])
        pmean, plo, phi = boot_mean(paired) if len(paired) else (float("nan"),) * 3
        rank_o = np.array([r["summary"]["reg_oracle"]["rank"] for r in Q if "reg_oracle" in r["summary"]])
        rank_d = np.array([r["summary"]["reg_decoupled"]["rank"] for r in Q if "reg_decoupled" in r["summary"]])
        nfs = np.array([r["summary"].get("n_certified_registered", 0) for r in Q])
        nbetter = np.array([r["summary"].get("n_reg_better", 0) for r in Q])
        sp_c = np.array([r["summary"].get("spearman", {}).get("complete_reg", np.nan) for r in Q], dtype=float)
        sp_d = np.array([r["summary"].get("spearman", {}).get("decoupled_reg", np.nan) for r in Q], dtype=float)
        grid, fixed = fixed_setting_gaps(Q)
        fx = None
        if fixed is not None and len(g("reg_oracle")) == n and np.all(np.isfinite(fixed)):
            tuned, fmean = g("reg_oracle"), fixed.mean(axis=1)
            jb, jd = int(np.argmin(fmean)), grid.index(DEFAULT_RULE)
            fx = dict(better=int((fmean < tuned.mean()).sum()), best=boot_mean(fixed[jb]), best_cfg=grid[jb],
                      best_areas=int((fixed[jb] < tuned).sum()), default=boot_mean(fixed[jd]))
        matches = sum(bool(r["summary"].get("stored", {}).get("first_year_match")) for r in Q)
        errors = sum(len(r.get("errors", [])) for r in Q)
        L += [f"## Risk tolerance ${rho}M: {n} areas (seeds {Q[0]['_seed']} to {Q[-1]['_seed']}), {errors} stage errors", "",
              "| selection | mean % [95% boot] | median | min | max | mean $M | areas with z_A certified above it |", "|---|---|---|---|---|---|---|"]
        for name, lab in (("reg_oracle", "registered CE-tuned rule (the paper's)"), ("reg_best", "best registered first year"), ("reg_decoupled", "decoupled selection"), ("reg_decoupled_plan", "decoupled plan itself (no optimizer)")):
            s = stats.get(name)
            if s:
                L.append(f"| {lab} | {fnum(s['mean'])} [{fnum(s['lo'])}, {fnum(s['hi'])}] | {fnum(s['median'])} | {fnum(s['min'])} | {fnum(s['max'])} | {fnum(s.get('dollar_mean'))} | {s.get('resolved', '--')} of {s['n']} |")
        L += ["", f"- reproduction of the stored oracle rule: first year matches on {matches} of {n} areas; stored first-year gap mean {fnum(stored.mean()) if len(stored) else '--'}%, "
              f"largest absolute difference from the re-solved gap {fnum(np.abs(diff).max(), 3) if len(diff) else '--'} points",
              f"- paired reduction, oracle minus best first year: mean {fnum(pmean)} [{fnum(plo)}, {fnum(phi)}] points",
              f"- rank of the oracle's first year among the registered first years: mean {fnum(rank_o.mean(), 1) if len(rank_o) else '--'}, best on {int((rank_o == 1).sum()) if len(rank_o) else '--'} of {n} areas; "
              f"distinct registered first years per area: mean {fnum(nfs.mean(), 1)}; first years certifying above the oracle's: mean {fnum(nbetter.mean(), 1)}",
              f"- rank of the decoupled selection's first year: mean {fnum(rank_d.mean(), 1) if len(rank_d) else '--'}, best on {int((rank_d == 1).sum()) if len(rank_d) else '--'} of {n} areas",
              f"- Spearman(complete-plan CE, certified CE): mean {fnum(np.nanmean(sp_c))}; Spearman(decoupled CE, certified CE): mean {fnum(np.nanmean(sp_d))}"]
        if all("pv_mean" in stats.get(name, {}) for name in ("reg_oracle", "reg_best", "reg_decoupled")):
            L.append(f"- first-year gaps in % of program value, E(z_N) - E(0): oracle {fnum(stats['reg_oracle']['pv_mean'])}, best first year {fnum(stats['reg_best']['pv_mean'])}, "
                     f"decoupled selection {fnum(stats['reg_decoupled']['pv_mean'])} (means)")
        if fx:
            L.append(f"- one setting used unchanged on every area: {fx['better']} of {len(grid)} settings have a smaller mean first-year gap than the rule tuned area by area; "
                     f"best ({cfg_label(fx['best_cfg'])}) mean {fnum(fx['best'][0])} [{fnum(fx['best'][1])}, {fnum(fx['best'][2])}]%, smaller on {fx['best_areas']} of {n} areas; "
                     f"default setting mean {fnum(fx['default'][0])} [{fnum(fx['default'][1])}, {fnum(fx['default'][2])}]%")
        L.append("")
        L += ["| seed | oracle gap % | stored % | best % | decoupled % | decoupled plan % | oracle rank | decoupled rank | n first years | errors |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in Q:
            S = r["summary"]
            L.append(f"| {r['_seed']} | {fnum(S.get('reg_oracle', {}).get('fs_gap_pct'))} | {fnum(S.get('stored', {}).get('vopt', {}).get('fs_gap_ce_pct'))} | {fnum(S.get('reg_best', {}).get('fs_gap_pct'))} | "
                     f"{fnum(S.get('reg_decoupled', {}).get('fs_gap_pct'))} | {fnum(S.get('reg_decoupled', {}).get('plan_gap_pct'))} | {S.get('reg_oracle', {}).get('rank', '--')} of {S.get('n_certified_registered', '--')} | "
                     f"{S.get('reg_decoupled', {}).get('rank', '--')} | {S.get('n_certified_registered', '--')} | {len(r.get('errors', []))} |")
        L.append("")
        for name, word in (("reg_oracle", "RegOracle"), ("reg_best", "RegBest"), ("reg_decoupled", "RegDecoupled"), ("reg_decoupled_plan", "RegDecoupledPlan")):
            s = stats.get(name)
            if s:
                T += [f"\\newcommand{{\\synthrule{word}GapMean{w}}}{{{fnum(s['mean'])}}}", f"\\newcommand{{\\synthrule{word}GapMeanLo{w}}}{{{fnum(s['lo'])}}}",
                      f"\\newcommand{{\\synthrule{word}GapMeanHi{w}}}{{{fnum(s['hi'])}}}", f"\\newcommand{{\\synthrule{word}GapMedian{w}}}{{{fnum(s['median'])}}}",
                      f"\\newcommand{{\\synthrule{word}GapMin{w}}}{{{fnum(s['min'])}}}", f"\\newcommand{{\\synthrule{word}GapMax{w}}}{{{fnum(s['max'])}}}"]
                if "dollar_mean" in s:
                    T += [f"\\newcommand{{\\synthrule{word}GapDollarMean{w}}}{{{fnum(s['dollar_mean'])}}}", f"\\newcommand{{\\synthrule{word}Resolved{w}}}{{{s['resolved']}}}"]
                if "pv_mean" in s:
                    T.append(f"\\newcommand{{\\synthrule{word}GapPvMean{w}}}{{{fnum(s['pv_mean'])}}}")
        T += [f"\\newcommand{{\\synthruleNAreas{w}}}{{{n}}}", f"\\newcommand{{\\synthruleStoredGapMean{w}}}{{{fnum(stored.mean()) if len(stored) else '--'}}}",
              f"\\newcommand{{\\synthruleReproMaxDiff{w}}}{{{fnum(np.abs(diff).max(), 3) if len(diff) else '--'}}}", f"\\newcommand{{\\synthruleFirstYearMatches{w}}}{{{matches}}}",
              f"\\newcommand{{\\synthrulePairedReductionMean{w}}}{{{fnum(pmean)}}}", f"\\newcommand{{\\synthrulePairedReductionLo{w}}}{{{fnum(plo)}}}", f"\\newcommand{{\\synthrulePairedReductionHi{w}}}{{{fnum(phi)}}}",
              f"\\newcommand{{\\synthruleOracleRankMean{w}}}{{{fnum(rank_o.mean(), 1) if len(rank_o) else '--'}}}", f"\\newcommand{{\\synthruleOracleIsBest{w}}}{{{int((rank_o == 1).sum()) if len(rank_o) else '--'}}}",
              f"\\newcommand{{\\synthruleDecoupledRankMean{w}}}{{{fnum(rank_d.mean(), 1) if len(rank_d) else '--'}}}", f"\\newcommand{{\\synthruleDecoupledIsBest{w}}}{{{int((rank_d == 1).sum()) if len(rank_d) else '--'}}}",
              f"\\newcommand{{\\synthruleNFsMean{w}}}{{{fnum(nfs.mean(), 1)}}}", f"\\newcommand{{\\synthruleNBetterMean{w}}}{{{fnum(nbetter.mean(), 1)}}}",
              f"\\newcommand{{\\synthruleSpearmanCompleteMean{w}}}{{{fnum(np.nanmean(sp_c))}}}", f"\\newcommand{{\\synthruleSpearmanDecoupledMean{w}}}{{{fnum(np.nanmean(sp_d))}}}",
              f"\\newcommand{{\\synthruleErrors{w}}}{{{errors}}}"]
        if fx:
            T += [f"\\newcommand{{\\synthruleFixedBetter{w}}}{{{fx['better']}}}", f"\\newcommand{{\\synthruleFixedBestGapMean{w}}}{{{fnum(fx['best'][0])}}}",
                  f"\\newcommand{{\\synthruleFixedBestGapMeanLo{w}}}{{{fnum(fx['best'][1])}}}", f"\\newcommand{{\\synthruleFixedBestGapMeanHi{w}}}{{{fnum(fx['best'][2])}}}",
                  f"\\newcommand{{\\synthruleFixedBestAreas{w}}}{{{fx['best_areas']}}}", f"\\newcommand{{\\synthruleFixedBestDeck{w}}}{{{fx['best_cfg']['deck']:g}}}",
                  f"\\newcommand{{\\synthruleFixedDefaultGapMean{w}}}{{{fnum(fx['default'][0])}}}", f"\\newcommand{{\\synthruleFixedDefaultGapMeanLo{w}}}{{{fnum(fx['default'][1])}}}",
                  f"\\newcommand{{\\synthruleFixedDefaultGapMeanHi{w}}}{{{fnum(fx['default'][2])}}}"]
    open(md_path, "w").write("\n".join(L) + "\n")
    if tex_path:
        open(tex_path, "w").write("\n".join(T) + "\n")
    print(f"wrote {md_path}" + (f" and {tex_path}" if tex_path else ""), flush=True)


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default=None, help="pickled instance (the NDIC case); default ../../data/derived/ndic_mckenzie_40u.pkl unless --seeds is given")
    ap.add_argument("--seeds", default=None, help="synthetic base areas by generator seed, e.g. 100 or 100-129 (exp_replicates.BASE_KNOBS, nx x ny)")
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--ny", type=int, default=10)
    ap.add_argument("--out", default="results/ndic_rule_check.json")
    ap.add_argument("--rhos", default=None, help="default 100,300,1000 for the NDIC case, 100 for synthetic areas")
    ap.add_argument("--variants", default=None, help="default packages,fixed for the NDIC case, packages for synthetic areas")
    ap.add_argument("--grid", default=None, choices=("registered", "extended"), help="default extended for the NDIC case, registered for synthetic areas")
    ap.add_argument("--reuse-dir", default=None, help="folder of stored replicate runs rep_s<seed>_r<rho>.json whose rn and ce are reused when the option table matches")
    ap.add_argument("--resume", action="store_true", help="continue the run file at --out: finished stages are restored, the rest run")
    ap.add_argument("--grid-only", action="store_true", help="enumerate, select and decouple only (no solver; for a dry run)")
    ap.add_argument("--summarize", action="store_true", help="numpy only: write the markdown summary and the macros from finished run files")
    ap.add_argument("--glob", default=None, help="with --summarize: run files of the synthetic check, e.g. 'results/rule_check/synth_s*_r*.json'")
    ap.add_argument("--md", default=None, help="with --summarize --glob: the markdown path (default results/rule_check/summary_synth.md)")
    ap.add_argument("--macros", default="", help="macro file written by --summarize; '' skips it")
    args = ap.parse_args()
    synthetic = args.seeds is not None
    if args.summarize:
        if args.glob:
            files = sorted(glob.glob(args.glob))
            write_summary_synth(files, args.md or "results/rule_check/summary_synth.md", args.macros or None, args.reuse_dir or "results/replicates")
        else:
            res = json.load(open(args.out))
            for run in res["runs"]:
                if "summary" not in run and all(k in run.get("methods", {}) for k in ("rn", "ce")):
                    summarize_run(run)
            write_summary(res, os.path.splitext(args.out)[0] + ".md", args.macros or None)
        raise SystemExit
    rhos = [float(v) for v in (args.rhos or ("100" if synthetic else "100,300,1000")).split(",")]
    variants = (args.variants or ("packages" if synthetic else "packages,fixed")).split(",")
    grid_name = args.grid or ("registered" if synthetic else "extended")
    if synthetic:
        from instance import make_instance
        from exp_replicates import BASE_KNOBS
        seeds = parse_seeds(args.seeds)
        assert len(seeds) == 1, "one seed per run file (one queue job per area)"
        seed = seeds[0]
        I = make_instance(args.nx, args.ny, seed=seed, **BASE_KNOBS)
        head = dict(mode="synthetic", seed=seed, nx=args.nx, ny=args.ny, knobs=dict(BASE_KNOBS))
    else:
        path = args.instance or "../../data/derived/ndic_mckenzie_40u.pkl"
        with open(path, "rb") as fh:
            I = pickle.load(fh)
        if "prob" not in I:
            build_nodes(I, 3)
        seed = None
        head = dict(mode="ndic", instance_file=path, provenance=I.get("provenance"))
    if args.resume and os.path.exists(args.out):
        res = json.load(open(args.out))
        res.setdefault("resumed", []).append(dict(at=time.strftime("%Y-%m-%d %H:%M:%S"), version_before=res.get("version")))
        res["version"] = VERSION
        res["args_resume"] = vars(args)
    else:
        res = dict(version=VERSION, args=vars(args), **head, grid=grid_name, mus_crew=list(MUS_CREW[grid_name]), rhos=rhos, variants=variants,
                   started=time.strftime("%Y-%m-%d %H:%M:%S"), oa=B.OA, mc_draws=B.MC_DRAWS, solver_rev=B.SOLVER_REV,
                   threads=dict(gurobi=int(os.environ.get("GRB_THREADS", 2)), omp=os.environ.get("OMP_NUM_THREADS")),
                   gurobi_memlimit_gb=float(os.environ.get("GRB_MEMLIMIT", 1.2)), rss_cap_gb=B.RSS_CAP_GB, runs=[], errors=[])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    B.dump(res, args.out)
    for variant in variants:
        Iv = variant_instance(I, variant)
        for rho in rhos:
            t0 = time.time()
            ctx = dict(grid_only=args.grid_only, resume=args.resume, grid=grid_name,
                       reuse_file=os.path.join(args.reuse_dir, f"rep_s{seed}_r{rho:.0f}.json") if (args.reuse_dir and synthetic) else None)
            run = next((r for r in res["runs"] if r.get("variant") == variant and abs(float(r.get("rho", -1)) - rho) < 1e-9), None) if args.resume else None
            if run is None:
                run = dict(variant=variant, rho=rho, version=VERSION, status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"), errors=[], stages={})
                res["runs"].append(run)
            else:
                run["status"], run["errors"] = "running (resumed)", []
            try:
                run_one(Iv, variant, rho, run, lambda: B.dump(res, args.out), ctx)
                n_err = len(run["errors"])
                run["status"] = "done" if n_err == 0 else f"done with {n_err} stage error(s)"
            except Exception as e:
                import traceback
                traceback.print_exc()
                run["status"] = "failed"
                run["errors"].append(B.error_entry("run", e, time.time() - t0))
                res["errors"].append(dict(variant=variant, rho=rho, error=repr(e), time=time.time() - t0))
            run["time_total"] = time.time() - t0
            print(f"  {variant} rho {rho:.0f} done in {run['time_total']:.0f}s ({run['status']})  peak RSS {B.rss_gb():.2f} GB", flush=True)
            B.dump(res, args.out)
    print("done", flush=True)
