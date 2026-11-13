"""Render the headline figure of the paper from the stored replicate runs.

Run from any working directory with the repository's dev Python:
    python /path/to/repo/src/model/fig_headline.py

Panel (a) also reads the rule-check runs: the first-year gap when the first year is
chosen by scoring every candidate with one later-drilling rule.

Writes only headline_results.pdf, .png and .json in paper/figures by default.
--check-only validates the inputs and statistics without writing output. No
solver is imported or run, and no generated tables or macros are regenerated.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
import scipy

import summarize_replicates as summary


ROOT = Path(__file__).resolve().parents[2]
RHOS = (50, 100, 300)
BASE_SEEDS = tuple(range(100, 130))
PAIRED_SEEDS = tuple(range(100, 110))
METRICS = ("fs_gap_ce", "ce_gain", "gain_tilt1", "rn_vss")
MACRO_PREFIX = {"fs_gap_ce": "repFsGapCe", "ce_gain": "repCeGain",
                "gain_tilt1": "repGainTiltOne", "rn_vss": "repRnVss"}
RHO_SUFFIX = {50: "Fifty", 100: "Hundred", 300: "ThreeHundred"}
RULE_SELECTION = "reg_decoupled"    # first year chosen by scoring every candidate with one later-drilling rule
RULE_MACRO = "synthruleRegDecoupledGapMean"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def file_record(path):
    return {"path": source_path(path), "sha256": sha256(path)}


def finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value)


def load_checked(results_dir):
    """Reject incomplete/mismatched populations before the reporter can skip rows."""
    expected = {f"rep_s{s}_r{rho}.json": (s, rho, "base")
                for s in BASE_SEEDS for rho in RHOS}
    expected.update({f"net_s{s}_r100.json": (s, 100, "net") for s in PAIRED_SEEDS})
    actual = {p.name for pattern in ("rep_s*_r*.json", "net_s*_r*.json")
              for p in results_dir.glob(pattern)}
    require(actual == set(expected),
            f"Unexpected study files: missing={sorted(set(expected)-actual)}, "
            f"extra={sorted(actual-set(expected))}")
    files = sorted(results_dir / name for name in expected)
    for path in files:
        doc = json.loads(path.read_text())
        require(not doc.get("errors"), f"File-level errors: {path}")
        runs = doc.get("runs", [])
        require(len(runs) == 1, f"Expected exactly one run: {path}")
        run = runs[0]
        identity = (run.get("seed"), run.get("rho"), run.get("tag"))
        require(identity == expected[path.name], f"Wrong run identity: {path}: {identity}")
        require(run.get("model_version") == "v3.3" and run.get("solver_rev") == "s2",
                f"Expected model v3.3 / solver s2: {path}")
        require(str(run.get("status", "")).startswith("done") and not run.get("errors"),
                f"Incomplete run or stage errors: {path}")
        require(not summary.refuted_in_run(run), f"Refuted certificate: {path}")
        for name in ("rn", "ce", "rn_tilt1", "heur_ce+exact_recourse", "mv_rn"):
            method = run.get("methods", {}).get(name, {})
            require(method.get("feasible") is True, f"Non-feasible/missing {name}: {path}")
            for key in ("ce", "enpv"):
                require(finite_number(method.get(key)), f"Non-finite {name}.{key}: {path}")
        for name in ("ce", "heur_ce+exact_recourse"):
            method = run["methods"][name]
            require(all(finite_number(method.get(k)) for k in ("ce_lb", "ce_ub")),
                    f"Non-finite CE certificate for {name}: {path}")
        ladder = run.get("ladder", {})
        require(all(finite_number(ladder.get(k)) for k in ("rn_vss", "rn_vss_lb", "rn_vss_ub")),
                f"Non-finite RN-VSS value/certificate: {path}")
        require(run["methods"]["rn"]["enpv"] > 0, f"Non-positive strip denominator: {path}")

    # Use the same sorted-file order and normalization as the published reporter.
    runs = summary.load([str(p) for p in files])
    require(len(runs) == 100, "The reporter unexpectedly omitted an input run")
    quantities = [summary.quantities(run) for run in runs]
    for q in quantities:
        require(q is not None and not q["invalid_certificates"], "Reporter rejected a run/certificate")
        require(all(finite_number(q.get(key + "_pct")) for key in METRICS),
                f"Non-finite plotted quantity: {q['file']}")
    return quantities, files


def load_rule_check(rule_dir):
    """Per-area first-year gap of the first year selected with one later-drilling rule (rule-check runs)."""
    expected = {f"synth_s{s}_r{rho}.json": (s, rho) for s in BASE_SEEDS for rho in RHOS}
    actual = {p.name for p in rule_dir.glob("synth_s*_r*.json")}
    require(actual == set(expected),
            f"Unexpected rule-check files: missing={sorted(set(expected)-actual)}, "
            f"extra={sorted(actual-set(expected))}")
    files = sorted(rule_dir / name for name in expected)
    values = {rho: {} for rho in RHOS}
    for path in files:
        seed, rho = expected[path.name]
        doc = json.loads(path.read_text())
        require(not doc.get("errors"), f"File-level errors: {path}")
        require(doc.get("seed") == seed, f"Wrong seed: {path}")
        runs = [run for run in doc.get("runs", []) if run.get("summary")]
        require(len(runs) == 1, f"Expected exactly one finished run: {path}")
        run = runs[0]
        require(float(run.get("rho", -1)) == float(rho), f"Wrong tolerance: {path}")
        require(run.get("model_version") == "v3.3" and run.get("solver_rev") == "s2",
                f"Expected model v3.3 / solver s2: {path}")
        require(str(run.get("status", "")).startswith("done") and not run.get("errors"),
                f"Incomplete run or stage errors: {path}")
        gap = run["summary"].get(RULE_SELECTION, {}).get("fs_gap_pct")
        require(finite_number(gap), f"Non-finite {RULE_SELECTION}.fs_gap_pct: {path}")
        values[rho][seed] = float(gap)
    return values, files


def rule_statistics(values):
    """Mean and percentile-bootstrap interval per tolerance, as the rule-check summary computes them
    (4,000 resamples of the areas, numpy seed 0)."""
    rows = []
    for rho in RHOS:
        require(tuple(sorted(values[rho])) == BASE_SEEDS, f"Rule-check seed set changed at {rho}")
        x = np.array([values[rho][seed] for seed in BASE_SEEDS], dtype=float)
        rng = np.random.default_rng(0)
        means = x[rng.integers(0, len(x), size=(4000, len(x)))].mean(axis=1)
        row = {"rho_usd_million": rho, "n": len(x), "mean": float(x.mean()),
               "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5))}
        require(row["lo"] <= row["mean"] <= row["hi"], f"Invalid mean interval for the rule-check series at {rho}")
        rows.append(row)
    return rows


def checked_stats(rows, metric):
    result = summary.stat(rows, metric)
    require(result is not None and result["n"] == len(rows), f"Dropped values in {metric}")
    require(all(finite_number(result[key]) for key in ("mean", "lo", "hi")),
            f"Non-finite statistics for {metric}")
    require(result["lo"] <= result["mean"] <= result["hi"], f"Invalid mean interval for {metric}")
    return result


def statistics(quantities):
    require(summary.N_BOOT == 4000, "Reporter bootstrap configuration changed; review before plotting")
    base = {}
    for rho in RHOS:
        rows = [q for q in quantities if q["tag"] == "base" and q["rho"] == rho]
        require(tuple(q["seed"] for q in rows) == BASE_SEEDS, f"Base seed order/set changed at {rho}")
        base[rho] = rows
    series = {metric: [{"rho_usd_million": rho, **checked_stats(base[rho], metric)}
                       for rho in RHOS] for metric in METRICS}
    paired_base = [q for q in base[100] if q["seed"] in PAIRED_SEEDS]
    net = [q for q in quantities if q["tag"] == "net" and q["rho"] == 100]
    require(tuple(q["seed"] for q in paired_base) == tuple(q["seed"] for q in net) == PAIRED_SEEDS,
            "The control and base must have the same ten seeds in the same order")
    changes = np.array([n["rn_vss_pct"] - b["rn_vss_pct"] for b, n in zip(paired_base, net)])
    lo, hi = summary.boot_ci(changes, n_boot=4000, seed=0)
    return {"base": {"seeds": list(BASE_SEEDS), "series": series},
            "paired_margin_control": {
                "rho_usd_million": 100, "seeds": list(PAIRED_SEEDS),
                "base": checked_stats(paired_base, "rn_vss"),
                "lower_margin": checked_stats(net, "rn_vss"),
                "paired_change_percentage_points": {
                    "n": len(changes), "mean": float(changes.mean()), "lo": lo, "hi": hi},
                "denominator": "each setting's own risk-neutral plan expected NPV",
                "rn_vss_is_risk_neutral": True,
                "lower_margin_mean_revenue_multiplier": 0.75}}


def verify_statistics(data, macro_path, reference_path, rule_macro_path):
    macros = {k: float(v) for k, v in re.findall(
        r"\\newcommand\{\\([^}]+)\}\{(-?[0-9.]+)\}", macro_path.read_text())}
    rule_macros = {k: float(v) for k, v in re.findall(
        r"\\newcommand\{\\([^}]+)\}\{(-?[0-9.]+)\}", rule_macro_path.read_text())}
    require(not set(macros) & set(rule_macros), "Macro names repeat across the two macro files")
    macros.update(rule_macros)
    reference = json.loads(reference_path.read_text())
    require(reference["base"]["seeds"] == list(BASE_SEEDS), "Independent base seed set changed")
    macro_checks = []

    def macro_check(name, value):
        require(name in macros and float(f"{value:.2f}") == macros[name],
                f"Rounded manuscript macro mismatch: {name}: {value}")
        macro_checks.append(name)

    def reference_check(actual, expected, location):
        for key in ("n", "mean", "lo", "hi"):
            require(key in expected and np.isclose(actual[key], expected[key], rtol=0, atol=1e-12),
                    f"Independent extract mismatch at {location}.{key}")

    for metric, series in data["base"]["series"].items():
        require(len(reference["base"]["series"][metric]) == len(RHOS),
                f"Independent extract tolerance count changed for {metric}")
        for row, prior in zip(series, reference["base"]["series"][metric]):
            rho = row["rho_usd_million"]
            require(rho == prior["rho_usd_million"], "Independent extract tolerance order changed")
            reference_check(row, prior, f"{metric}.{rho}")
            for field in ("mean", "lo", "hi"):
                macro_check(MACRO_PREFIX[metric] + field.title() + RHO_SUFFIX[rho], row[field])
            macro_check("repN" + RHO_SUFFIX[rho], row["n"])
    paired = data["paired_margin_control"]
    old_pair = reference["paired_margin_control"]
    require(paired["seeds"] == old_pair["seeds"], "Independent paired seed set changed")
    require(paired["rho_usd_million"] == old_pair["rho_usd_million"],
            "Independent control tolerance changed")
    for key in ("base", "lower_margin", "paired_change_percentage_points"):
        reference_check(paired[key], old_pair[key], f"paired_margin_control.{key}")
    macro_check("repNetRnVssBaseMean", paired["base"]["mean"])
    for field in ("mean", "lo", "hi"):
        macro_check("repNetRnVss" + field.title(), paired["lower_margin"][field])
        macro_check("repNetRnVssDiff" + ("" if field == "mean" else field.title()),
                    paired["paired_change_percentage_points"][field])
    macro_check("repNetN", paired["base"]["n"])
    for row in data["base"]["rule_selection"]:
        suffix = RHO_SUFFIX[row["rho_usd_million"]]
        require(row["n"] == len(BASE_SEEDS), "Rule-check series does not cover every base area")
        macro_check(RULE_MACRO + suffix, row["mean"])
        macro_check(RULE_MACRO + "Lo" + suffix, row["lo"])
        macro_check(RULE_MACRO + "Hi" + suffix, row["hi"])
    return {"manuscript_macros": sorted(set(macro_checks)),
            "independent_extract": "replicate-based means/CIs/counts and the paired change match within 1e-12; "
                                   "the rule-check series is checked against its macros only"}


def render(data, output_prefix):
    # Fail rather than silently substitute another font on an unconfigured host.
    font_path = font_manager.findfont("Arial", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#7b8389", "axes.linewidth": 0.6,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "text.color": "#202a33", "axes.labelcolor": "#202a33",
    })
    ink, residual_color = "#263d50", "#b14e16"
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.0), sharey=True,
                             gridspec_kw={"width_ratios": [1.05, 1.3, 0.95]})
    # Remove the caption's duplicate footer while preserving the physical plot height.
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.1835, top=0.6895, wspace=0.16)
    for i, ax in enumerate(axes):
        ax.set_ylim(-0.12, 7.5)
        ax.set_yticks([0, 2, 4, 6])
        ax.grid(axis="y", color="#dde1e4", linewidth=0.6, zorder=0)
        ax.axhline(0, color="#7b8389", lw=0.6, zorder=1)
        if i:
            ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    def dots(ax, x, rows, color=ink, marker="o"):
        mean, lo, hi = np.array([[row[k] for k in ("mean", "lo", "hi")] for row in rows]).T
        ax.errorbar(x, mean, yerr=[mean-lo, hi-mean], fmt=marker, linestyle="none",
                    markersize=3, markerfacecolor="white" if marker == "s" else color,
                    markeredgewidth=0.8, color=color, ecolor=color, elinewidth=0.9,
                    capsize=2.5, capthick=0.9, zorder=3, clip_on=False)
        for xi, yi, top in zip(x, mean, hi):
            ax.annotate(f"{yi:.2f}", (xi, top), xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=color)

    series = data["base"]["series"]
    dots(axes[0], np.arange(3)-0.2, series["fs_gap_ce"])
    dots(axes[0], np.arange(3)+0.2, data["base"]["rule_selection"], color=residual_color, marker="s")
    dots(axes[1], np.arange(3)-0.14, series["ce_gain"])
    dots(axes[1], np.arange(3)+0.14, series["gain_tilt1"], color=residual_color, marker="s")
    paired = data["paired_margin_control"]
    dots(axes[2], np.arange(2), [paired["base"], paired["lower_margin"]])
    for ax in axes[:2]:
        ax.set_xlim(-0.5, 2.5)
        ax.set_xticks([0, 1, 2], [str(rho) for rho in RHOS])
        ax.set_xlabel("Risk tolerance (USD million)", labelpad=8)
    axes[2].set_xlim(-0.45, 1.45)
    axes[2].set_xticks([0, 1], ["Base", "Revenue\n× 0.75"])
    axes[2].set_xlabel("Margin setting", labelpad=7)
    axes[0].set_ylabel("Difference (% of risk-neutral\nplan expected NPV)", labelpad=8)
    titles = ["(a) Optimization", "(b) Risk modeling", "(c) Anticipating revision"]
    subtitles = ["Rule's first-year CE gap", "Risk-averse plan's CE gain",
                 "Expected-NPV gain"]
    for ax, title, subtitle in zip(axes, titles, subtitles):
        left = ax.get_position().x0
        fig.text(left, 0.9425, title, weight="bold", fontsize=9, va="top")
        fig.text(left, 0.8735, subtitle, fontsize=8, va="top")
    handles = [Line2D([], [], color=ink, marker="o", linestyle="none", markersize=4),
               Line2D([], [], color=residual_color, marker="s", linestyle="none",
                      markerfacecolor="white", markersize=4)]
    axes[1].legend(handles, ["Over risk-neutral plan", "Over one re-solve"], frameon=False,
                   loc="upper right", bbox_to_anchor=(1.1, 1.10),
                   handletextpad=0.4, labelspacing=0.5, borderpad=0)
    axes[0].legend(handles, ["CE-tuned rule's first year", "One later-drilling rule"], frameon=False,
                   loc="upper left", bbox_to_anchor=(0.0, 1.10),
                   handletextpad=0.4, labelspacing=0.5, borderpad=0)
    fig.canvas.draw()
    visible_text = [text for text in fig.findobj(Text) if text.get_visible() and text.get_text()]
    require(all(text.get_fontsize() >= 8 for text in visible_text), "Text below the 8-point minimum")
    renderer = fig.canvas.get_renderer()
    for text in visible_text:
        bounds = text.get_window_extent(renderer)
        require(bounds.x0 >= 0 and bounds.y0 >= 0 and bounds.x1 <= fig.bbox.x1 and bounds.y1 <= fig.bbox.y1,
                f"Text would be clipped: {text.get_text()}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf = output_prefix.with_suffix(".pdf")
    png = output_prefix.with_suffix(".png")
    fig.savefig(pdf, facecolor="white", metadata={
        "Title": "Paired values of optimization, risk modeling and anticipating later revision",
        "Creator": "src/model/fig_headline.py", "CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    return {"width_inches": 6.5, "height_inches": 3.0, "minimum_font_points": 8,
            "font": "Arial", "font_path": font_path, "pdf_fonttype": 42,
            "pdf": file_record(pdf), "png": file_record(png)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "src/model/results/replicates")
    parser.add_argument("--macros", type=Path, default=ROOT / "paper/sections/results_numbers_rep.tex")
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "reference/headline-data.json")
    parser.add_argument("--rule-dir", type=Path, default=ROOT / "src/model/results/rule_check")
    parser.add_argument("--rule-macros", type=Path,
                        default=ROOT / "paper/sections/results_numbers_synthrule.tex")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "paper/figures/headline_results")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    quantities, files = load_checked(args.results_dir)
    rule_values, rule_files = load_rule_check(args.rule_dir)
    data = statistics(quantities)
    data["base"]["rule_selection"] = rule_statistics(rule_values)
    checks = verify_statistics(data, args.macros, args.reference, args.rule_macros)
    if args.check_only:
        print("PASS: 90 base, 10 paired-control and 90 rule-check runs; all statistics match the macros, "
              "and the replicate statistics match the independent extract.")
        return
    provenance = {
        "schema": "spe-headline-results-v2", "statistics": data,
        "bootstrap": {"resamples": 4000, "numpy_seed": 0, "percentiles": [2.5, 97.5],
                      "resampling_unit": "instance, preserving method pairing",
                      "method": "percentile interval of resampled means of per-instance normalized differences"},
        "normalization": "100 * raw difference / abs(methods.rn.enpv), separately for each instance and setting",
        "metric_sources": {
            "fs_gap_ce": "methods.ce.ce_lb - methods[heur_ce+exact_recourse].ce_lb (CE)",
            "ce_gain": "methods.ce.ce - methods.rn.ce (CE)",
            "gain_tilt1": "methods.ce.ce - methods.rn_tilt1.ce (full-plan CE)",
            "rn_vss": "ladder.rn_vss (expected NPV)",
            "rule_selection": "summary.reg_decoupled.fs_gap_pct of the rule-check runs (CE): the first year chosen by "
                              "scoring every distinct first year with the default later-drilling rule, with "
                              "continuation optimized afterward"},
        "validation": {"model_version": "v3.3", "solver_rev": "s2", "completed_runs": 100, "rule_check_runs": 90,
                       "excluded_runs": [], "refuted_certificates": [], "checks": checks,
                       "caveat": "Finite certificate checks can refute bounds, not establish their validity."},
        "negative_and_zero_values": "Retained without truncation or selection; sign-test ties use the existing reporter threshold.",
        "source_order": "sorted filenames, giving ascending seeds within each plotted group",
        "sources": [file_record(path) for path in files],
        "rule_check_sources": [file_record(path) for path in rule_files],
        "rule_selection_per_instance": [{"seed": seed, "rho_usd_million": rho, "fs_gap_pct": rule_values[rho][seed]}
                                        for rho in RHOS for seed in BASE_SEEDS],
        "code": [file_record(Path(__file__)), file_record(Path(summary.__file__)),
                 file_record(ROOT / "src/model/audit_certificates.py")],
        "cross_checks": {"macros": file_record(args.macros), "rule_macros": file_record(args.rule_macros),
                         "independent_extract": file_record(args.reference)},
        "software": {"numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "per_instance": [{"file": q["file"], "seed": q["seed"], "rho_usd_million": q["rho"],
                          "tag": q["tag"], "strip_expected_npv_usd_million": q["enpv_rn"],
                          **{key + "_pct": q[key + "_pct"] for key in METRICS}}
                         for q in quantities],
    }
    provenance["render"] = render(data, args.output_prefix)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output_prefix}.pdf, .png and .json; all source checks passed.")


if __name__ == "__main__":
    main()
