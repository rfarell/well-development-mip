"""Tables, macros and figures of the search-method study (run_meta.py, scored by score_meta.py).

Usage: summarize_meta.py results/meta/scored_out.json [--runs results/meta/out] [--paper ../../paper] [--md results/summary_meta.md]

Primary metric: the first-stage gap to z_A under certified recourse, the interval [gap_lo, gap_hi] = [LB_A - UB_x, UB_A - LB_x]
in % of E[NPV](rn) of the instance; reported by method and budget multiple as the median and the worst (largest gap_hi)
over all runs with the heuristic seed, pooled over instances at rho = $100M (and separately at $300M), with the share of
runs whose first stage ties z_A's (gap interval contains 0), the share that beat the rule's certified first stage, and the
share that returned the heuristic seed's plan unchanged. Secondary: the in-sample gap to UB_A, CE_insample vs UB_A.
Outputs (with --paper): sections/results_meta.tex, sections/results_numbers_meta.tex (\\meta...), and with --runs
figures/bound_scale.pdf (search trajectories and matched post-search continuation intervals) and
figures/meta_curves_supplement.pdf (search curves and single long runs).
Use --figures-only to regenerate bound_scale and meta_curves_supplement without changing tables, macros, or summaries.
Table labels: only rho = $100M with the heuristic seed is tab:meta (the main-text table); the variants are
tab:meta-noseed, tab:meta-threehundred and tab:meta-noseed-threehundred (other rho: tab:meta[-noseed]-<rho word>),
which the SI sets.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from report_currency import format_table_currency

METHODS = [("bpso_v", "V-shaped binary PSO"), ("rk_pso", "random-key PSO"), ("ga2", "GA (1/n mutation, swap)"), ("bpso_sigmoid", "sigmoid binary PSO (original)"), ("ga", "GA (original)")]
RHO_WORD = {25: "TwentyFive", 50: "Fifty", 100: "Hundred", 300: "ThreeHundred", 1000: "Thousand"}


def f(x, d=2):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


DIGIT_WORDS = dict(zip("0123456789", ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")))
MULT_WORDS = {1: "One", 5: "Five", 10: "Ten", 50: "Fifty"}


def macro_name(s):
    """A LaTeX-safe macro stem: title case, digits spelled out (ga2 -> GaTwo)."""
    t = "".join(ch for ch in str(s).replace("_", " ").title() if ch.isalnum())
    return "".join(DIGIT_WORDS.get(ch, ch) for ch in t)


def load_rows(path):
    d = json.load(open(path))
    rows = [r for r in d["rows"] if "ce_lb" in r and r.get("run_model_version", "v3.3") == "v3.3" and r.get("plan_feasible_now", True)]
    refs = d.get("refs", {})
    return rows, refs, d


def enrich(rows, refs):
    out = []
    for r in rows:
        ref = refs.get(f"{r['instance']}_r{float(r['rho']):g}") or {}
        E = ref.get("enpv_rn") or r.get("enpv_rn")
        zlb, zub = ref.get("zA_ce_lb", r.get("zA_ce_lb")), ref.get("zA_ce_ub", r.get("zA_ce_ub"))
        if E is None or zlb is None:
            if "gap_lo_pct" in r:
                q = dict(r); out.append(q)
            continue
        q = dict(r)
        q.setdefault("gap_lo", zlb - r["ce_ub"]); q.setdefault("gap_hi", zub - r["ce_lb"])
        q["gap_lo_pct"] = 100 * q["gap_lo"] / abs(E); q["gap_hi_pct"] = 100 * q["gap_hi"] / abs(E)
        q["gap_mid_pct"] = 0.5 * (q["gap_lo_pct"] + q["gap_hi_pct"])
        q["insample_gap_pct"] = 100 * (zub - r["ce"]) / abs(E)
        q["beats_rule"] = (r["ce_lb"] > ref["heur_ce_exact_ub"]) if "heur_ce_exact_ub" in ref else None
        q["ties_zA"] = q["gap_lo"] <= 0 <= q["gap_hi"] or q["gap_hi"] < 0
        out.append(q)
    return out


def cell_stats(R):
    g = np.array([r["gap_mid_pct"] for r in R], float); hi = np.array([r["gap_hi_pct"] for r in R], float); ins = np.array([r["insample_gap_pct"] for r in R], float)
    br = [r["beats_rule"] for r in R if r.get("beats_rule") is not None]
    return dict(n=len(R), med=float(np.median(g)), worst=float(hi.max()), best=float(g.min()), ins_med=float(np.median(ins)),
                tie=100 * np.mean([r["ties_zA"] for r in R]), seed_return=100 * np.mean([bool(r.get("returned_seed")) for r in R]),
                beats_rule=100 * np.mean(br) if br else np.nan, improve_med=float(np.median([r["ce"] - r["seed_ce"] for r in R])))


def meta_label(rho, with_seed):
    """tab:meta for rho = $100M with the heuristic seed (the main-text table); every variant carries a suffix."""
    w = RHO_WORD.get(int(round(float(rho))), str(int(round(float(rho)))))
    return "tab:meta" + ("" if with_seed else "-noseed") + ("" if float(rho) == 100.0 else "-" + w.lower())


def table(rows, rho, with_seed=True):
    R = [r for r in rows if r["rho"] == rho and bool(r.get("with_seed", True)) == with_seed and not r.get("long")]
    mults = sorted({r["mult"] for r in R})
    out, macros = [], []
    w = RHO_WORD.get(int(rho), str(int(rho)))
    hd = lambda a, b: r"\begin{tabular}[b]{@{}c@{}}" + a + r"\\ " + b + r"\end{tabular}"   # noqa: E731  two-line header
    out += [r"\begin{table}[tbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
            r"\caption{Rebuilt metaheuristics at $\rho=\$" + f"{rho:,.0f}" + r"$M" + ("" if with_seed else ", started without the heuristic seed") +
            r": first-stage gap to $z_A$ under certified recourse, in \% of $E[\mathrm{NPV}](z_N)$, pooled over instances and seeds (median of the interval midpoint; worst = largest upper end), the share of runs whose first stage ties $z_A$'s within the certificates, the share that beat the oracle rule's certified first stage, the share that returned the oracle rule's plan (the seed), and the in-sample gap to $\mathrm{UB}_A$ at the shared recourse rule. Budgets are multiples of the evaluations one thread makes in the certified solver's cold start, measured before the model correction.}",
            r"\label{" + meta_label(rho, with_seed) + "}",
            r"\begin{tabular}{>{\raggedright\arraybackslash}p{2.4cm}lrrrrrrr}", r"\toprule",
            "method & budget & runs & " + " & ".join([hd("median", "gap"), hd("worst", "gap"), hd("ties $z_A$", r"(\%)"), hd("beats rule", r"(\%)"), hd("returns", r"seed (\%)"), hd("in-sample", "gap")]) + r" \\", r"\midrule"]
    for m, lab in METHODS:
        first = True
        for mult in mults:
            S = [r for r in R if r["method"] == m and r["mult"] == mult]
            if not S:
                continue
            c = cell_stats(S)
            out.append((lab if first else "") + f" & {mult}$\\times$ & {c['n']} & {f(c['med'])} & {f(c['worst'])} & {f(c['tie'], 0)} & {f(c['beats_rule'], 0)} & {f(c['seed_return'], 0)} & {f(c['ins_med'])}" + r" \\")
            first = False
            tag = f"{macro_name(m)}X{MULT_WORDS.get(mult, macro_name(mult))}{w}" + ("" if with_seed else "NoSeed")
            macros += [(f"GapMed{tag}", f(c["med"])), (f"GapWorst{tag}", f(c["worst"])), (f"Tie{tag}", f(c["tie"], 0)), (f"BeatsRule{tag}", f(c["beats_rule"], 0)),
                       (f"SeedReturn{tag}", f(c["seed_return"], 0)), (f"InsampleGapMed{tag}", f(c["ins_med"])), (f"N{tag}", str(c["n"]))]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out), macros


def md_summary(rows):
    lines = ["# Rebuilt metaheuristics (scored by certified recourse)"]
    for rho in sorted({r["rho"] for r in rows}):
        for ws in (True, False):
            R = [r for r in rows if r["rho"] == rho and bool(r.get("with_seed", True)) == ws and not r.get("long")]
            if not R:
                continue
            lines.append(f"\n## rho {rho:.0f} {'with' if ws else 'without'} the heuristic seed ({len(R)} runs, instances {sorted({r['instance'] for r in R})})\n")
            lines.append("| method | mult | n | median gap % | worst gap % | ties z_A % | beats rule % | returns seed % | in-sample gap % | median improvement over seed $M |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for m, _ in METHODS:
                for mult in sorted({r["mult"] for r in R}):
                    S = [r for r in R if r["method"] == m and r["mult"] == mult]
                    if S:
                        c = cell_stats(S)
                        lines.append(f"| {m} | {mult} | {c['n']} | {f(c['med'])} | {f(c['worst'])} | {f(c['tie'], 0)} | {f(c['beats_rule'], 0)} | {f(c['seed_return'], 0)} | {f(c['ins_med'])} | {f(c['improve_med'], 3)} |")
    return "\n".join(lines) + "\n"



def _median_curve(runs, grid):
    """Median of one fixed cohort, without extrapolation past observed stops."""
    from solver_figure_data import cohort_median
    return cohort_median(runs, grid)


def _curve_style(ax):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 50]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", lw=0.5)
    ax.set_ylabel("Certainty equivalent (USD million)")


def _draw_curves(ax, run_dir, refs, bench_json, inst=0, rho=100, study=None):
    """Only the ten 10x runs per rebuilt method, ending at their observed stops."""
    from solver_figure_data import MAIN_METHODS, curve_points, load_study
    if study is None:
        study = load_study(run_dir, refs, bench_json, inst=inst, rho=rho)
    ref = study["reference"]
    ax.axhspan(ref["zA_ce_lb"], ref["zA_ce_ub"], facecolor="#CC79A7", alpha=0.3, lw=0, zorder=1)
    handles = []
    for method, label, color, marker, ls in MAIN_METHODS:
        cohort = study["main"][method]
        x, y = curve_points(cohort)
        line, = ax.step(x, y, where="post", color=color, ls=ls, lw=1.35, label=label, zorder=4)
        for budget in (1, 5, 10):
            value = _median_curve(cohort, [budget])[0]
            ax.plot(budget, value, marker=marker, ms=4, color=color, ls="none", zorder=5)
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], color=color, ls=ls, marker=marker, ms=4, lw=1.35, label=label))
    cold = study["cold"]
    times = np.array([h["cum_time"] for h in cold["hist"]]) / cold["solver_time"]
    ax.step(times, [h["ce_lb"] for h in cold["hist"]], where="post", color="0.12", lw=1.3, zorder=3)
    ax.step(times, [h["ce_ub"] for h in cold["hist"]], where="post", color="0.12", lw=1.3, ls="--", zorder=3)
    ax.plot(times[-1], cold["ce_lb"], "o", color="0.12", ms=2.6, zorder=5)
    ax.plot(times[-1], cold["ce_ub"], "|", color="0.12", ms=5, zorder=5)
    ax.axhline(ref["heur_ce_ce"], color="0.5", lw=0.8, ls=":", zorder=1)
    ax.text(1.8, ref["heur_ce_ce"] + 0.11, "Oracle rule seed", fontsize=8, color="0.35")
    ax.annotate("Solver upper bound", xy=(0.72, cold["hist"][1]["ce_ub"]), xytext=(1.8, 145.72),
                fontsize=8, ha="left", va="center", arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "0.25"})
    ax.annotate("Solver incumbent", xy=(0.68, cold["hist"][1]["ce_lb"]), xytext=(1.8, 145.18),
                fontsize=8, ha="left", va="center", arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "0.25"})
    ax.text(0.013, 145.68, f"Initial upper bound: {cold['hist'][0]['ce_ub']:.2f} (off scale)", fontsize=8, color="0.3")
    ax.text(0.013, 144.94, "Reference interval*", fontsize=8, color="#85466A")
    _curve_style(ax)
    ax.set_xlim(0.009, 12)
    ax.set_ylim(139.55, 146.12)
    ax.set_yticks([140, 142, 144, 146])
    ax.set_xlabel(f"Budget: search evaluations / {study['evals_1x']:,}; solver wall seconds / {cold['solver_time']:.1f}")
    return handles


def _write_figure(fig, path, data):
    stem = os.path.splitext(path)[0]
    fig.savefig(path)
    fig.savefig(stem + ".png", dpi=200)
    with open(stem + ".data.json", "w") as fh:
        json.dump(data, fh, indent=2)


def fig_bound_scale(run_dir, refs, bench_json, scale_json, path, inst=0, rho=100, scored_rows=None, scored_path=None):
    """Search and matched post-search continuation, replacing the duplicate scale panel.

    ``scale_json`` remains in the call signature for existing callers, but scale
    data are no longer used. Matching scored rows are mandatory for the lower
    panel. This routine only writes figures and an audit sidecar.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from solver_figure_data import FIGURE_STYLE, MAIN_METHODS, load_study, sidecar_data, source_record
    if scored_rows is None:
        raise ValueError("the search/continuation figure requires matching scored_rows")
    study = load_study(run_dir, refs, bench_json, scored_rows, inst, rho)
    ref = study["reference"]
    with plt.rc_context(FIGURE_STYLE):
        fig = plt.figure(figsize=(6.5, 6.65))
        top = fig.add_axes([0.115, 0.635, 0.855, 0.28])
        bottom = fig.add_axes([0.17, 0.115, 0.80, 0.345])
        handles = _draw_curves(top, run_dir, refs, bench_json, inst, rho, study=study)
        top.set_title("(a) Search with rule-filled later years", loc="left", pad=25)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.955), ncol=3,
                   frameon=False, handlelength=2.4, columnspacing=1.8)
        fig.text(0.115, 0.553, "Medians of ten seeds.", fontsize=8)
        fig.text(0.115, 0.524, "(b) The same first years after optimizing later drilling", fontsize=9)
        fig.text(0.17, 0.493, "Additional computation after search; one row per seed", fontsize=8)
        bottom.axvspan(ref["zA_ce_lb"], ref["zA_ce_ub"], facecolor="#CC79A7", alpha=0.3, lw=0, zorder=1)
        groups = []
        for group, (method, label, color, marker, _) in enumerate(MAIN_METHODS):
            rows = study["continuation"][method]
            ys = np.arange(33 - 11 * group, 23 - 11 * group, -1)
            groups.append((float(ys.mean()), label))
            for y, row in zip(ys, rows):
                bottom.plot([row["search_ce"], row["ce_lb"]], [y, y], color="0.80", lw=0.45, zorder=1)
                bottom.plot(row["search_ce"], y, marker=marker, color=color, ms=3, ls="none", zorder=3)
                bottom.plot([row["ce_lb"], row["ce_ub"]], [y, y], color=color, lw=1.3, zorder=4)
                bottom.plot([row["ce_lb"], row["ce_ub"]], [y, y], color=color, marker="|", ms=3.2, ls="none", zorder=4)
        yseed = -1.5
        bottom.plot([ref["heur_ce_ce"], ref["heur_ce_exact_lb"]], [yseed, yseed], color="0.75", lw=0.6)
        bottom.plot(ref["heur_ce_ce"], yseed, marker="D", color="0.4", ms=3.5, ls="none")
        bottom.plot([ref["heur_ce_exact_lb"], ref["heur_ce_exact_ub"]], [yseed, yseed], color="0.35", lw=1.3)
        bottom.plot([ref["heur_ce_exact_lb"], ref["heur_ce_exact_ub"]], [yseed, yseed], color="0.35", marker="|", ms=4, ls="none")
        groups.append((yseed, "Oracle rule seed"))
        bottom.set_yticks([g[0] for g in groups], [g[1] for g in groups])
        bottom.tick_params(axis="y", length=0, pad=5)
        bottom.set_ylim(-3.5, 34.5)
        bottom.set_xlim(139.55, 145.15)
        bottom.set_xticks([140, 141, 142, 143, 144, 145])
        bottom.set_xlabel("Certainty equivalent (USD million)")
        bottom.grid(axis="x", color="0.90", lw=0.5)
        bottom.spines[["top", "right", "left"]].set_visible(False)
        legend = [Line2D([], [], marker="o", color="0.25", ms=3.5, ls="none", label="Search score"),
                  Line2D([], [], color="0.25", marker="|", ms=4, lw=1.2, label="Continuation interval")]
        bottom.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.02, 1.005), ncol=2,
                      frameon=False, borderaxespad=0, handlelength=1.8)
        bottom.text(ref["zA_ce_lb"] - 0.04, 33.8, "Reference*", color="#85466A", fontsize=8, ha="right", va="bottom")
        fig.text(0.115, 0.029, "*Reference interval and all bounds are conditional on valid solver dual bounds.", fontsize=8)
        data = sidecar_data(study)
        if scored_path:
            data["scored_source"] = source_record(scored_path)
        data["lower_panel"] = {"one_row_per_seed": True, "additional_computation": True,
                               "intervals_are_confidence_intervals": False,
                               "oracle_seed": {k: ref[k] for k in ("heur_ce_ce", "heur_ce_exact_lb", "heur_ce_exact_ub")}}
        _write_figure(fig, path, data)
        plt.close(fig)
    return True


def fig_meta_supplement(run_dir, refs, bench_json, path, inst=0, rho=100):
    """Original 1x methods and single 50x swarm runs, with true stopping extents."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from solver_figure_data import FIGURE_STYLE, curve_points, load_study, sidecar_data
    study = load_study(run_dir, refs, bench_json, inst=inst, rho=rho)
    ref = study["reference"]
    styles = [(study["original"]["bpso_sigmoid"], "Original sigmoid PSO: ten 1× runs", "#666666", "D", ":"),
              (study["original"]["ga"], "Original GA: ten 1× runs", "#882255", "^", "-."),
              (study["long"]["bpso_v"], "V-shaped PSO: one 50× run", "#0072B2", "o", "-"),
              (study["long"]["rk_pso"], "Random-key PSO: one 50× run", "#D55E00", "s", "--")]
    extra = []
    with plt.rc_context(FIGURE_STYLE):
        fig = plt.figure(figsize=(6.5, 3.9))
        ax = fig.add_axes([0.115, 0.25, 0.855, 0.58])
        ax.axhspan(ref["zA_ce_lb"], ref["zA_ce_ub"], facecolor="#CC79A7", alpha=0.3, lw=0)
        handles = []
        for rr, label, color, marker, ls in styles:
            x, y = curve_points(rr)
            line, = ax.step(x, y, where="post", color=color, ls=ls, lw=1.2, label=label)
            ax.plot(x[-1], y[-1], marker=marker, ms=4.5, color=color, ls="none")
            handles.append(line)
            extra.append({"label": label, "n": len(rr), "files": [r["_file"] for r in rr],
                          "budget": x.tolist(), "median_ce": y.tolist(), "observed_stop": float(x[-1])})
        _curve_style(ax)
        ax.set_xlim(0.009, 65)
        ax.set_ylim(139.55, 145.18)
        ax.set_yticks([140, 141, 142, 143, 144, 145])
        ax.set_xlabel("Search evaluation budget / 3,509 (log scale)")
        ax.text(0.013, 144.87, "Reference interval*", fontsize=8, color="#85466A")
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.98), ncol=2,
                   frameon=False, handlelength=2.5, columnspacing=1.8)
        fig.text(0.115, 0.130, "Rule-filled later years; original-method curves show medians. Markers denote observed stops.", fontsize=8)
        fig.text(0.115, 0.075, "One instance at USD 100 million risk tolerance; the 50× runs each use one search seed.", fontsize=8)
        fig.text(0.115, 0.022, "*Reference interval assumes valid solver dual bounds.", fontsize=8)
        data = sidecar_data(study)
        data.pop("main_curves")
        data["supplement_curves"] = extra
        _write_figure(fig, path, data)
        plt.close(fig)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scored")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--paper", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--figures-only", action="store_true", help="write only figures and data sidecars; leave tables and macros untouched")
    ap.add_argument("--bench", default="results/b40v33_s0_r100.json")
    ap.add_argument("--scale", default="results/scale/scale_study.json", help="retained for CLI compatibility; no longer plotted")
    args = ap.parse_args()
    rows, refs, d = load_rows(args.scored)
    if args.figures_only:
        if not args.paper or not args.runs:
            ap.error("--figures-only requires --paper and --runs")
        fig_bound_scale(args.runs, refs, args.bench, None, os.path.join(args.paper, "figures", "bound_scale.pdf"),
                        scored_rows=rows, scored_path=args.scored)
        fig_meta_supplement(args.runs, refs, args.bench, os.path.join(args.paper, "figures", "meta_curves_supplement.pdf"))
        print("wrote bound_scale and meta_curves_supplement figures and data sidecars")
        sys.exit(0)
    rows = enrich(rows, refs)
    rows = [r for r in rows if "gap_mid_pct" in r]
    text = md_summary(rows)
    print(text)
    if args.md:
        open(args.md, "w").write(text)
    if args.paper:
        tex, macros = "% generated by summarize_meta.py; do not edit\n", []
        for rho in sorted({r["rho"] for r in rows}):
            for ws in (True, False):
                if any(r["rho"] == rho and bool(r.get("with_seed", True)) == ws for r in rows):
                    t, m = table(rows, rho, ws); tex += t; macros += m
        macros += [("Runs", str(len(rows))), ("Instances", str(len({r["instance"] for r in rows}))), ("Seeds", str(len({r["seed"] for r in rows if r.get("with_seed", True)})))]
        open(os.path.join(args.paper, "sections", "results_meta.tex"), "w").write(format_table_currency(tex))
        with open(os.path.join(args.paper, "sections", "results_numbers_meta.tex"), "w") as fh:
            fh.write("% generated by summarize_meta.py; do not edit\n")
            for k, v in macros:
                fh.write(f"\\newcommand{{\\meta{k}}}{{{v}}}\n")
        if args.runs:
            ok = fig_meta_supplement(args.runs, refs, args.bench, os.path.join(args.paper, "figures", "meta_curves_supplement.pdf"))
            print("figure meta_curves_supplement.pdf:", "written" if ok else "not written", file=sys.stderr)
            ok = fig_bound_scale(args.runs, refs, args.bench, args.scale, os.path.join(args.paper, "figures", "bound_scale.pdf"),
                                 scored_rows=rows, scored_path=args.scored)
            print("figure bound_scale.pdf:", "written" if ok is True else f"not written ({ok})", file=sys.stderr)
        print(f"wrote results_meta.tex and {len(macros)} macros", file=sys.stderr)
