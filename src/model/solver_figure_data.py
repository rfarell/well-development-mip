"""Read-only data preparation for the SPE search and continuation figures.

No model construction or solver imports. Cohorts are fixed by method, stopping
budget and seed; intervals remain individual continuation certificates.
"""
import glob
import hashlib
import json
import os

import numpy as np


MAIN_METHODS = (
    ("bpso_v", "V-shaped PSO", "#0072B2", "o", "-"),
    ("rk_pso", "Random-key PSO", "#D55E00", "s", "--"),
    ("ga2", "Genetic algorithm", "#009E73", "^", "-."),
)
FIGURE_STYLE = {"font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
                "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42}


def source_record(path):
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return {"path": os.path.abspath(path), "sha256": digest}


def observed_history(run):
    """Return logged evaluation ratios and CE, rejecting an inconsistent history."""
    history = np.asarray(run["info"]["best_history"], float)
    if history.ndim != 2 or history.shape[1] != 3 or len(history) == 0 or not np.isfinite(history).all():
        raise ValueError("empty or nonfinite search history")
    if run["evals_1x"] <= 0 or np.any(np.diff(history[:, 0]) < 0) or np.any(np.diff(history[:, 2]) < -1e-8):
        raise ValueError("nonmonotone search history or invalid evaluation unit")
    if history[-1, 0] != run["info"]["evals"]:
        raise ValueError("search history does not reach the observed stopping count")
    if not np.isclose(history[-1, 2], run["ce"], rtol=0, atol=1e-8):
        raise ValueError("history endpoint differs from returned plan score")
    return history[:, 0] / run["evals_1x"], history[:, 2]


def cohort_median(runs, grid):
    """Right-continuous median of a fixed cohort, only while every run is observed.

    A completed 1x run cannot contribute a carried value at 5x or 10x. Missing
    initial observations are likewise not filled. NaN outside the common
    observed support prevents silently changing the cohort behind a curve.
    """
    if not runs:
        raise ValueError("empty search cohort")
    grid = np.asarray(grid, float)
    curves = []
    for run in runs:
        ev, best = observed_history(run)
        indices = np.searchsorted(ev, grid, side="right") - 1
        observed = (indices >= 0) & (grid <= ev[-1])
        curve = np.full(grid.shape, np.nan)
        curve[observed] = best[indices[observed]]
        curves.append(curve)
    matrix = np.vstack(curves)
    complete = np.isfinite(matrix).all(axis=0)
    result = np.full(grid.shape, np.nan)
    result[complete] = np.median(matrix[:, complete], axis=0)
    return result


def curve_points(runs):
    grid = np.unique(np.concatenate([observed_history(r)[0] for r in runs]))
    values = cohort_median(runs, grid)
    keep = np.isfinite(values)
    return grid[keep], values[keep]


def select_cohort(runs, method, multiple, long=False):
    selected = [d for d in runs if d["method"] == method and d["mult"] == multiple
                and d.get("with_seed") is True and bool(d.get("long")) == long]
    expected_seeds = {0} if long else set(range(10))
    seeds = [d["seed"] for d in selected]
    if len(seeds) != len(set(seeds)) or set(seeds) != expected_seeds:
        raise ValueError(f"{method} {multiple}x: missing or duplicate seeds")
    for run in selected:
        if run.get("model_version") != "v3.3" or not run.get("feasible") or run.get("objective") != "ce":
            raise ValueError(f"{method}: invalid model, feasibility or objective")
        ev, _ = observed_history(run)
        if ev[-1] != multiple or run["info"].get("stop") != "evals":
            raise ValueError(f"{method}: unexpected observed stopping budget")
        if run["max_evals"] != run["info"]["evals"]:
            raise ValueError(f"{method}: incomplete prescribed budget")
    return sorted(selected, key=lambda d: d["seed"])


def match_continuations(runs, scored_rows, reference):
    """Join exact run filenames; retain all ten seed-specific certificate pairs."""
    by_file = {}
    for row in scored_rows:
        if row["file"] in by_file:
            raise ValueError(f"duplicate scored filename {row['file']}")
        by_file[row["file"]] = row
    matched = []
    for run in runs:
        row = by_file.get(run["_file"])
        if row is None:
            raise ValueError(f"missing continuation certificate for {run['_file']}")
        for key in ("instance", "rho", "method", "mult", "seed", "with_seed", "long"):
            if row.get(key) != run.get(key):
                raise ValueError(f"mismatched {key} for {run['_file']}")
        if row.get("solver_rev") != "s2" or row.get("run_model_version") != "v3.3" or not row.get("plan_feasible_now"):
            raise ValueError("continuation certificate is not a feasible s2 result")
        lo, hi = float(row["ce_lb"]), float(row["ce_ub"])
        if not np.isfinite([lo, hi]).all() or lo > hi or lo < run["ce"] - 1e-7:
            raise ValueError("invalid continuation interval")
        if not np.isclose(row["ce"], run["ce"], rtol=0, atol=1e-8):
            raise ValueError("scored search value differs from the exact plotted run")
        gap_lo = reference["zA_ce_lb"] - hi
        gap_hi = reference["zA_ce_ub"] - lo
        if not np.allclose([row["gap_lo"], row["gap_hi"]], [gap_lo, gap_hi], rtol=0, atol=1e-8):
            raise ValueError("continuation row uses a different unrestricted reference")
        matched.append({"file": run["_file"], "seed": run["seed"], "search_ce": run["ce"],
                        "ce_lb": lo, "ce_ub": hi, "cert_time": row["cert_time"],
                        "gap_lo_pct": row["gap_lo_pct"], "gap_hi_pct": row["gap_hi_pct"]})
    return matched


def load_study(run_dir, refs, bench_json, scored_rows=None, inst=0, rho=100):
    paths = sorted(glob.glob(os.path.join(run_dir, f"i{inst}_r{rho:g}_*_seed*.json")))
    runs = []
    for path in paths:
        with open(path) as fh:
            run = json.load(fh)
        if run["instance"] != inst or float(run["rho"]) != float(rho):
            raise ValueError("run filename and instance disagree")
        run["_file"] = os.path.basename(path)
        run["_source"] = source_record(path)
        runs.append(run)
    ref = refs[f"{inst}_r{rho:g}"]
    if ref["zA_ce_lb"] > ref["zA_ce_ub"]:
        raise ValueError("invalid unrestricted reference interval")
    main = {m: select_cohort(runs, m, 10) for m, *_ in MAIN_METHODS}
    original = {m: select_cohort(runs, m, 1) for m in ("bpso_sigmoid", "ga")}
    long = {m: select_cohort(runs, m, 50, long=True) for m in ("bpso_v", "rk_pso")}
    used = [r for cohorts in (main, original, long) for rr in cohorts.values() for r in rr]
    if {r["evals_1x"] for r in used} != {3509}:
        raise ValueError("unexpected evaluation calibration: this figure is for instance 0 at rho 100")
    if any(r["options_sha1"] != ref["options_sha1"] for r in used):
        raise ValueError("search and reference option tables differ")
    if not np.allclose([r["seed_ce"] for r in used], ref["heur_ce_ce"], rtol=0, atol=1e-8):
        raise ValueError("different CE-tuned-rule seed plans")
    with open(bench_json) as fh:
        bench = json.load(fh)
    bench_rows = [r for r in bench["runs"] if r["seed"] == used[0]["instance_seed"] and r["rho"] == rho]
    if len(bench_rows) != 1 or bench_rows[0].get("solver_rev") != "s2":
        raise ValueError("missing solver trajectory")
    cold = bench_rows[0]["methods"]["ce_cold"]
    if cold.get("solver_rev") != "s2" or not cold.get("hist"):
        raise ValueError("missing s2 cold-start history")
    if not np.isclose(cold["hist"][-1]["cum_time"], cold["solver_time"], rtol=0, atol=1e-8):
        raise ValueError("cold-start normalization and history disagree")
    continuation = None
    if scored_rows is not None:
        continuation = {m: match_continuations(rr, scored_rows, ref) for m, rr in main.items()}
    return {"main": main, "original": original, "long": long, "reference": ref,
            "cold": cold, "continuation": continuation, "benchmark_source": source_record(bench_json),
            "instance": inst, "rho": rho, "evals_1x": 3509, "sources": [r["_source"] for r in used]}


def sidecar_data(study):
    data = {"instance": study["instance"], "rho_usd_million": study["rho"],
            "ce_unit": "USD million", "reference": study["reference"],
            "certificate_caveat": "conditional on valid solver dual bounds; not confidence intervals",
            "budgets": {"search_divisor_evaluations": study["evals_1x"],
                        "search_calibration": "v3.2 cold start, 36.05066394805908 s, 97.3248639073987 evaluations/CPU-s",
                        "solver_divisor_wall_seconds_s2": study["cold"]["solver_time"],
                        "equal_time_comparison": False},
            "cold_history": study["cold"]["hist"], "benchmark_source": study["benchmark_source"],
            "sources": study["sources"], "main_curves": {}, "continuations": study["continuation"]}
    for method, rr in study["main"].items():
        x, y = curve_points(rr)
        data["main_curves"][method] = {"n": len(rr), "seeds": [r["seed"] for r in rr],
                                       "files": [r["_file"] for r in rr], "budget": x.tolist(), "median_ce": y.tolist()}
    return data
