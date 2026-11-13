"""Figures of the SPE version that combine the replicate study, the stress cases and the sweeps.

Usage: fig_spe.py [--paper ../../paper] [--replicates 'results/replicates/rep_s*_r*.json'] [--stress 'results/b40v33_s*_r*.json']
                  [--sweeps 'results/rho_sweep40_v33_s*.json'] [--implied-only]

Writes figures/regime_map_spe.pdf: (a) CE gain of z_A over the risk-neutral plan, (b) Beck's expected-NPV cost, (c) residual over the
best non-oracle deck-family member (practice deck 0.9, tilt after one and after two iterations); x =
lambda sigma_W(z_N); replicates as small dots, stress cases as open diamonds, sweeps as lines (sweeps lack the deck family,
so panel (c) has no sweep line).
Writes figures/implied_deck_spe.pdf/.png/.data.json: the implied price and type-curve multipliers of the risk-neutral plan's tilted
weights and the oracle uniform revenue multiplier on one scale, with observed replicate ranges and stress cases.
"""
import argparse
import glob
import hashlib
import json
import math
import os

import numpy as np

BAND_PCT = 5.0


def load(pattern):
    out = []
    for fn in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        d["_source_path"] = os.path.abspath(fn)
        out.append(d)
    return out


def runs_of(datas, tag=None):
    R = []
    for d in datas:
        for r in d.get("runs", []):
            if not (str(r.get("status", "")).startswith("done") and not r.get("errors")) or r.get("model_version") != "v3.3":
                continue
            if tag is not None and r.get("tag", "base") != tag:
                continue
            R.append(dict(r, _source_path=d.get("_source_path")))
    return R


def residual_nonoracle(r):
    """CE(z_A) minus the best of {deck 0.9 from the curve, rn_tilt1, rn_tilt2}, in % of E(z_N)."""
    m = r["methods"]
    cands = [m[n]["ce"] for n in ("rn_tilt1", "rn_tilt2") if n in m]
    curve = m.get("rn_deck", {}).get("deck_curve", [])
    for c in curve:
        if abs(c["deck"] - 0.9) < 1e-9:
            cands.append(c["ce"])
    if not cands or "ce" not in m:
        return np.nan
    return 100 * (m["ce"]["ce"] - max(cands)) / abs(m["rn"]["enpv"])


def fig_regime(plt, reps, stress, sweeps, path):
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.9), sharex=True)
    axes[1].sharey(axes[0])
    for i, d in enumerate(sweeps):
        rows = sorted((x for x in d["rows"] if x.get("rn_lam_sigma", 0) > 0 and x.get("rn_enpv")), key=lambda x: x["rn_lam_sigma"])
        if not rows:
            continue
        xs = [x["rn_lam_sigma"] for x in rows]
        axes[0].plot(xs, [100 * x["ce_gain"] / abs(x["rn_enpv"]) for x in rows], "-", color=f"C{i}", lw=1.0, label=f"sweep, seed {d.get('seed')}")
        axes[1].plot(xs, [100 * x["enpv_cost"] / abs(x["rn_enpv"]) for x in rows], "-", color=f"C{i}", lw=1.0)
    if stress:
        kw = dict(marker="D", s=16, facecolors="none", edgecolors="k", linewidths=0.7, zorder=3, label=f"stress cases ({len(stress)} runs)")
        xs = [r["beck"]["lam_sigma_rn"] for r in stress]
        axes[0].scatter(xs, [r["beck"]["ce_gain_pct"] for r in stress], **kw)
        axes[1].scatter(xs, [r["beck"]["enpv_cost_pct"] for r in stress], **kw)
        axes[2].scatter(xs, [residual_nonoracle(r) for r in stress], **kw)
    if reps:
        kw = dict(marker="o", s=7, color="C3", alpha=0.65, linewidths=0, zorder=4, label=f"replicates ({len(reps)} runs)")
        xs = [r["beck"]["lam_sigma_rn"] for r in reps]
        axes[0].scatter(xs, [r["beck"]["ce_gain_pct"] for r in reps], **kw)
        axes[1].scatter(xs, [r["beck"]["enpv_cost_pct"] for r in reps], **kw)
        axes[2].scatter(xs, [residual_nonoracle(r) for r in reps], **kw)
    vals = [residual_nonoracle(r) for r in reps + stress]
    vals = [v for v in vals if np.isfinite(v)]
    if vals:
        top = max(max(vals) * 1.25, 1.0)
        axes[2].set_ylim(min(0, min(vals)) - 0.2, max(top, BAND_PCT + 0.5) if max(vals) > 0.6 * BAND_PCT else top)
    for ax, t in zip(axes, ("(a) CE gain of $z_A$ over the risk-neutral plan", "(b) expected-NPV cost (Beck)", "(c) residual over the best\nnon-oracle deck-family member")):
        ax.set_xscale("log"); ax.axhline(0, color="0.75", lw=0.6, zorder=0); ax.set_title(t, fontsize=8)
        ax.set_xlabel(r"$\lambda\sigma_W(z_N)$"); ax.grid(True, which="major", lw=0.3, alpha=0.5); ax.tick_params(labelsize=7)
    axes[0].set_ylabel(r"% of $E(z_N)$")
    hl = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l, h)
    fig.legend(list(hl.values()), list(hl), loc="lower center", ncol=min(4, len(hl)), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path); plt.close(fig)


def implied_data(reps, stress):
    """Validate the complete study before returning plot-ready values.

    Ranges are observed minima/maxima, not confidence intervals. A missing or
    duplicate run must fail rather than quietly change the scientific cohort.
    """
    expected = {
        "replicates": {(s, r) for s in range(100, 130) for r in (50.0, 100.0, 300.0)},
        "stress": {(s, r) for s in range(4) for r in (100.0, 300.0)},
    }
    result = {"model_version": "v3.3", "solver_rev": "s2", "range_kind": "observed minimum and maximum",
              "series": {}, "sources": []}
    for name, runs in (("replicates", reps), ("stress", stress)):
        keys = [(int(r["seed"]), float(r["rho"])) for r in runs]
        if len(keys) != len(set(keys)) or set(keys) != expected[name]:
            raise ValueError(f"implied deck: incomplete or duplicate {name} cohort")
        for r in runs:
            if (r.get("model_version") != "v3.3" or r.get("solver_rev") != "s2"
                    or not str(r.get("status", "")).startswith("done") or r.get("errors")):
                raise ValueError(f"implied deck: invalid run {(r['seed'], r['rho'])}")
            if name == "replicates" and r.get("tag", "base") != "base":
                raise ValueError("implied deck: non-base run in replicate cohort")
            pi = np.asarray(r["methods"]["rn"]["tilt"]["pi"], float)
            if pi.shape != (9,) or not np.isfinite(pi).all() or np.any(pi < 0) or not np.isclose(pi.sum(), 1):
                raise ValueError("implied deck: invalid tilted probabilities")
            source = r.get("_source_path")
            if source:
                with open(source, "rb") as fh:
                    digest = hashlib.sha256(fh.read()).hexdigest()
                result["sources"].append({"cohort": name, "seed": r["seed"], "rho": r["rho"],
                                          "path": source, "sha256": digest})
    getters = {
        "price": lambda r: r["methods"]["rn"]["tilt"]["implied_price"],
        "productivity": lambda r: r["methods"]["rn"]["tilt"]["implied_theta"],
        "oracle_deck": lambda r: r["methods"]["rn_deck"]["deck"],
    }
    for key, getter in getters.items():
        series = {"replicates": [], "stress": []}
        for rho in (50.0, 100.0, 300.0):
            values = np.asarray([getter(r) for r in reps if r["rho"] == rho], float)
            if values.shape != (30,) or not np.isfinite(values).all():
                raise ValueError(f"implied deck: missing {key} values at rho {rho:g}")
            series["replicates"].append(dict(rho=rho, n=30, mean=float(values.mean()),
                                               minimum=float(values.min()), maximum=float(values.max())))
        for r in sorted(stress, key=lambda r: (r["rho"], r["seed"])):
            value = float(getter(r))
            if not np.isfinite(value):
                raise ValueError(f"implied deck: missing stress {key}")
            series["stress"].append(dict(seed=r["seed"], rho=r["rho"], value=value))
        result["series"][key] = series
    return result


def fig_implied(plt, reps, stress, path):
    """Single common-axis diagnostic, at the manuscript's actual 6.5-inch width."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    data = implied_data(reps, stress)
    styles = (
        ("price", "Long-dated price / strip", "#0072B2", "o", "-"),
        ("productivity", "Productivity / mean", "#009E73", "s", "--"),
        ("oracle_deck", "Oracle revenue multiplier", "#D55E00", "^", "-.")
    )
    with plt.rc_context({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig = plt.figure(figsize=(6.5, 3.1))
        # Preserve plot dimensions while removing the caption's duplicate footer.
        ax = fig.add_axes([0.105, 0.382 / 3.1, 0.86, 2.286 / 3.1])
        handles = []
        for key, label, color, marker, ls in styles:
            series = data["series"][key]
            points = series["replicates"]
            x = [p["rho"] for p in points]
            ax.fill_between(x, [p["minimum"] for p in points], [p["maximum"] for p in points],
                            color=color, alpha=0.10, lw=0)
            handle, = ax.plot(x, [p["mean"] for p in points], color=color, marker=marker,
                              ls=ls, ms=4.5, lw=1.3, label=label, zorder=4)
            handles.append(handle)
            ax.scatter([p["rho"] for p in series["stress"]], [p["value"] for p in series["stress"]],
                       marker="D", s=23, facecolors="none", edgecolors=color, linewidths=0.8, zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(45, 335)
        ax.set_ylim(0.78, 1.014)
        ax.xaxis.set_major_locator(FixedLocator([50, 100, 300]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
        ax.axhline(1, color="0.5", ls=":", lw=0.8)
        ax.set_xlabel("Risk tolerance (USD million; log scale)")
        ax.set_ylabel("Multiplier")
        ax.grid(axis="y", color="0.88", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.515, 3.01 / 3.1), ncol=3,
                   frameon=False, handlelength=2.6, columnspacing=1.3, handletextpad=0.5)
        stem = os.path.splitext(path)[0]
        fig.savefig(path)
        fig.savefig(stem + ".png", dpi=200)
        plt.close(fig)
    with open(stem + ".data.json", "w") as fh:
        json.dump(data, fh, indent=2)
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper"))
    ap.add_argument("--replicates", default="results/replicates/rep_s*_r*.json")
    ap.add_argument("--stress", default="results/b40v33_s*_r*.json")
    ap.add_argument("--sweeps", default="results/rho_sweep40_v33_s*.json")
    ap.add_argument("--implied-only", action="store_true", help="write only the implied-deck figure and its data sidecar")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reps = [r for r in runs_of(load(args.replicates), "base") if isinstance(r.get("beck"), dict)]
    stress = [r for r in runs_of(load(args.stress)) if isinstance(r.get("beck"), dict)]
    sweeps = load(args.sweeps)
    figd = os.path.join(args.paper, "figures")
    if not args.implied_only:
        fig_regime(plt, reps, stress, sweeps, os.path.join(figd, "regime_map_spe.pdf"))
    fig_implied(plt, reps, stress, os.path.join(figd, "implied_deck_spe.pdf"))
    print(f"implied_deck_spe.pdf: {len(reps)} replicate runs, {len(stress)} stress runs")
