"""NDIC figures from the saved rho=100 plans; no optimization.

The main figure separates common year-0 home-unit commitments from ordinary-
probability-weighted occurrence of later home-unit options (years 1--3). A unit
may have both marks. Incoming laterals are not represented by this probability.
The supplement retains calibrated inputs and earliest home-unit years at the
central quadrature node, whose factors are not their distribution means.

All panels use a regular schematic section lattice. Geographic centroids remain
unchanged in the instance and are recorded separately in the JSON sidecar.
Both PDFs have exact 6.5-inch width; matching PNGs and a main JSON sidecar are
also written. A stale, ambiguous, or inconsistent run fails before plotting.
"""

import argparse
import glob
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

from instance import T
from metaheuristics import dense_plan, options_sha1
from model import MODEL_VERSION, Model


EXPECTED_SOLVER_REV = "s2"
EXPECTED_RHO = 100.0
FIGURE_WIDTH = 6.5
PKG_COLOR = "#D55E00"
GRID_COLOR = "#858C91"
NONE_COLOR = "#F3F4F5"
YEAR_COLORS = ("#175C86", "#559BB8", "#A0CED4", "#E9CC93")
PLAN_NAMES = {"rn": "Risk-neutral plan", "ce": "Risk-averse plan"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(path):
    """Portable repository-relative provenance where possible."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(__file__).resolve().parents[2]))
    except ValueError:
        return str(path)


def load_runs(pattern):
    """Read candidates without silently accepting incomplete records."""
    runs = []
    for filename in sorted(glob.glob(pattern)):
        with open(filename) as stream:
            document = json.load(stream)
        for run in document.get("runs", []):
            run = dict(run)
            run["_file"] = filename
            run["_file_errors"] = document.get("errors", [])
            runs.append(run)
    return runs


def select_run(pattern, rho):
    require(rho == EXPECTED_RHO, "This paper figure is pinned to rho=100 million USD.")
    runs = [r for r in load_runs(pattern) if float(r.get("rho", -1)) == rho]
    require(len(runs) == 1, f"Expected exactly one rho={rho:g} NDIC run, found {len(runs)} in {pattern}.")
    run = runs[0]
    require(run.get("tag") == "ndic" and run.get("seed") == 0, "Expected the NDIC case, seed 0.")
    require(run.get("status") == "done" and not run.get("errors") and not run["_file_errors"],
            "The selected run is incomplete or contains errors.")
    require(run.get("model_version") == MODEL_VERSION == "v3.3", "Expected model v3.3.")
    require(run.get("solver_rev") == EXPECTED_SOLVER_REV, "Expected solver revision s2.")
    for name in PLAN_NAMES:
        require(name in run.get("plans", {}) and name in run.get("methods", {}), f"Missing {name} plan or score.")
        require(run["methods"][name].get("feasible") is True, f"The stored {name} plan is not marked feasible.")
    require(run["methods"]["ce"].get("solver_rev") == EXPECTED_SOLVER_REV, "The CE score is from a stale solver revision.")
    return run


def validate_sparse_plan(M, plan):
    def indices(values, size, label):
        require(isinstance(values, list), f"{label}: expected sparse index list.")
        require(all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v < size for v in values),
                f"{label}: invalid sparse index.")
        require(len(values) == len(set(values)), f"{label}: duplicate sparse indices.")

    for field, size in (("z0", M.n0), ("y", M.G), ("w0", M.U)):
        indices(plan.get(field), size, field)
    for field, size in (("zk", M.n1), ("wk", M.U * (T - 1))):
        require(isinstance(plan.get(field), list) and len(plan[field]) == M.K, f"{field}: wrong node count.")
        for k, values in enumerate(plan[field]):
            indices(values, size, f"{field}[{k}]")


def unit_years(M, plan, k):
    """Earliest selected home-unit year in node k; -1 means no home-unit option."""
    selected = np.r_[M.idx0[plan["z0"] > 0.5], M.idx1[plan["zk"][k] > 0.5]]
    years = np.full(M.U, -1, dtype=int)
    for option in selected:
        u, year = int(M.o_u[option]), int(M.o_t[option])
        years[u] = year if years[u] < 0 else min(years[u], year)
    return years


def plan_summary(M, plan, probabilities, central):
    committed = M.idx0[plan["z0"] > 0.5]
    host = np.zeros(M.U, dtype=bool)
    host[M.o_u[committed]] = True
    later = np.zeros((M.K, M.U), dtype=bool)
    home = np.zeros_like(later)
    touched = np.zeros_like(later)
    wells = np.zeros(M.K, dtype=int)
    long_wells = np.zeros(M.K, dtype=int)
    for k in range(M.K):
        recourse = M.idx1[plan["zk"][k] > 0.5]
        require(np.all((M.o_t[recourse] >= 1) & (M.o_t[recourse] < T)), "Invalid later-option years.")
        later[k, M.o_u[recourse]] = True
        selected = np.r_[committed, recourse]
        home[k] = host | later[k]
        touched[k] = home[k]
        incoming = M.I["nbr"][M.o_u[selected[M.o_len[selected] >= 3]]]
        require(np.all(incoming >= 0), "Selected long option lacks a neighbor.")
        touched[k, incoming] = True
        wells[k] = M.o_k[selected].sum()
        long_wells[k] = M.o_k[selected[M.o_len[selected] >= 3]].sum()
    probability = probabilities @ later
    require(np.all((probability >= -1e-12) & (probability <= 1 + 1e-12)), "Invalid development probabilities.")
    explicit = np.array([sum(float(probabilities[k]) for k in range(M.K) if later[k, u]) for u in range(M.U)])
    require(np.allclose(probability, explicit, atol=1e-14, rtol=0), "Probability weighting mismatch.")
    bought = np.flatnonzero(plan["y"] > 0.5)
    acquired = np.isin(M.I["pkg_of"], bought)
    expected_wells = float(probabilities @ wells)
    return dict(
        hosts_year0=host, later_by_node=later, later_probability=np.clip(probability, 0, 1),
        any_home_probability=probabilities @ home, any_touched_probability=probabilities @ touched,
        acquired_units=acquired, bought_packages=bought, central_years=unit_years(M, plan, central),
        year0_wells=int(M.o_k[committed].sum()), year0_host_count=int(host.sum()),
        wells_by_node=wells, central_wells=int(wells[central]), expected_wells=expected_wells,
        expected_long_share=float((probabilities @ long_wells) / expected_wells),
    )


def load_data(args):
    run = select_run(args.runs, args.rho)
    with open(args.pkl, "rb") as stream:
        I = pickle.load(stream)
    require(all(key in I for key in ("prob", "theta", "price", "base_parents", "pdp_remain")),
            "Expected the NDIC instance with its saved tree and own producing base.")
    M = Model(I, 1.0 / args.rho)  # numpy/scipy reconstruction only; no solver import
    require((M.U, M.K, T, I["nx"], I["ny"]) == (40, 9, 4, 8, 10), "Unexpected case dimensions.")
    fmt = run.get("plan_format", {})
    require(fmt.get("options_sha1") == options_sha1(M), "The saved plan option-table SHA-1 does not match the instance.")
    for field, expected in (("n0", M.n0), ("n1", M.n1), ("K", M.K), ("G", M.G), ("U", M.U), ("T", T)):
        require(fmt.get(field) == expected, f"Plan format {field} does not match the reconstructed model.")
    prob = np.asarray(I["prob"], dtype=float)
    require(prob.shape == (M.K,) and np.all(np.isfinite(prob)) and np.all(prob > 0), "Invalid ordinary node probabilities.")
    require(np.isclose(prob.sum(), 1, atol=1e-12, rtol=0), "Ordinary probabilities do not sum to one.")
    marginal = np.array([1 / 6, 2 / 3, 1 / 6])
    require(np.allclose(prob, np.outer(marginal, marginal).ravel(), atol=1e-12, rtol=0), "Unexpected quadrature weights.")
    central = M.K // 2
    require(np.allclose([I["theta"][central], I["price"][central], prob[central]],
                        [np.exp(-I["sig_theta"]**2 / 2), np.exp(-I["sig_p"]**2 / 2), 4 / 9], atol=1e-12, rtol=0),
            "Unexpected central quadrature node.")
    col, pair = np.divmod(np.arange(M.U), I["ny"] // 2)
    expected_secs = np.c_[col * I["ny"] + 2 * pair, col * I["ny"] + 2 * pair + 1]
    require(np.array_equal(I["secs"], expected_secs), "Section indexing does not match the schematic lattice.")
    require(np.array_equal(I["nbr"], np.where(pair + 1 < I["ny"] // 2, np.arange(M.U) + 1, -1)),
            "Neighbor relationships do not match the schematic lattice.")
    data = {}
    expected = {"rn": (26, 7, 4, 107, 99.7777777777778), "ce": (26, 8, 2, 105, 96.3888888888889)}
    for name in PLAN_NAMES:
        validate_sparse_plan(M, run["plans"][name])
        plan = dense_plan(M, run["plans"][name])
        require(M.feasible(plan), f"Reconstructed {name} plan fails model feasibility checks.")
        d = plan_summary(M, plan, prob, central)
        stored = run["methods"][name]["placement"]
        require(len(stored["by_node"]) == M.K, f"{name}: wrong stored node count.")
        for k, node in enumerate(stored["by_node"]):
            require(np.allclose([node["theta"], node["price"], node["prob"]],
                                [I["theta"][k], I["price"][k], prob[k]], atol=1e-12, rtol=0),
                    f"{name}: stored node {k} differs from the instance tree.")
            require(node["wells_recourse"] + d["year0_wells"] == d["wells_by_node"][k],
                    f"{name}: reconstructed wells differ from stored node {k}.")
        require(np.isclose(d["expected_wells"], stored["wells_expected"], atol=1e-10, rtol=0), f"{name}: expected wells mismatch.")
        checks = (d["year0_wells"], d["year0_host_count"], len(d["bought_packages"]), d["central_wells"], d["expected_wells"])
        require(np.allclose(checks, expected[name], atol=1e-10, rtol=0), f"{name}: the paper's plan checks have changed: {checks}.")
        require(d["year0_wells"] == stored["wells_year0"] and d["year0_host_count"] == stored["units_year0"]
                and len(d["bought_packages"]) == stored["packages"], f"{name}: stored commitment counts mismatch.")
        data[name] = d
    return I, M, run, data, central, np.c_[col + 0.5, 2 * pair + 1.0]


def configure_plotting():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    font_manager.findfont("Arial", fallback_to_default=False)
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8.5,
        "axes.titlesize": 9.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white", "text.color": "#20272C",
    })


def map_axes(fig, x, y, width, title):
    height = width * FIGURE_WIDTH * 10 / 8 / fig.get_figheight()
    ax = fig.add_axes([x, y, width, height])
    ax.set(xlim=(0, 8), ylim=(0, 10), aspect="equal")
    ax.set_axis_off()
    ax.set_title(title, loc="left", pad=8, fontweight="bold")
    return ax


def cells(ax, coords, colors):
    from matplotlib.patches import Rectangle
    for (x, y), color in zip(coords, colors):
        ax.add_patch(Rectangle((x - 0.5, y - 1), 1, 2, facecolor=color, edgecolor=GRID_COLOR,
                               linewidth=0.45, clip_on=False, zorder=1))


def group_boundaries(ax, coords, groups, color, linewidth=1.6, linestyle="solid"):
    """Outline each nonnegative group, retaining boundaries between packages."""
    from matplotlib.collections import LineCollection
    segments = []
    lookup = {(int(x - 0.5), int((y - 1) / 2)): int(g) for (x, y), g in zip(coords, groups) if g >= 0}
    for (col, row), group in lookup.items():
        x0, y0 = col, 2 * row
        for neighbor, edge in (
            ((col - 1, row), [(x0, y0), (x0, y0 + 2)]),
            ((col + 1, row), [(x0 + 1, y0), (x0 + 1, y0 + 2)]),
            ((col, row - 1), [(x0, y0), (x0 + 1, y0)]),
            ((col, row + 1), [(x0, y0 + 2), (x0 + 1, y0 + 2)]),
        ):
            if lookup.get(neighbor) != group:
                segments.append(edge)
    if segments:
        ax.add_collection(LineCollection(segments, colors=color, linewidths=linewidth, linestyles=linestyle,
                                         capstyle="butt", clip_on=False, zorder=3))


def acquired_boundaries(ax, I, d, coords):
    group_boundaries(ax, coords, np.where(d["acquired_units"], I["pkg_of"], -1), PKG_COLOR)


def commitment_marks(ax, coords, mask):
    selected = coords[mask]
    # White face with a dark edge: the same symbol on light and dark cells (a white edge vanished on light ones).
    ax.plot(selected[:, 0], selected[:, 1], linestyle="none", marker="s", markersize=6.6,
            markerfacecolor="white", markeredgecolor="#111111", markeredgewidth=1.3, zorder=5)


def north_arrow(fig, x, y, height=0.072):
    from matplotlib.patches import FancyArrowPatch
    fig.add_artist(FancyArrowPatch((x, y), (x, y + height), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color="#20272C"))
    fig.text(x, y + height + 0.012, "N", ha="center", va="bottom", fontsize=8.5, fontweight="bold")


def save_figure(fig, output):
    from matplotlib import pyplot as plt
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve physical width and font sizes instead of cropping a tight bbox.
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main_figure(I, data, coords, output):
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(FIGURE_WIDTH, 5.2))
    cmap = LinearSegmentedColormap.from_list("later_drilling", [NONE_COLOR, "#BFDBE8", "#70ADC7", "#276789"])
    norm = Normalize(0, 1)
    for x, name, letter in ((0.055, "rn", "a"), (0.535, "ce", "b")):
        ax = map_axes(fig, x, 0.245, 0.410, f"({letter}) {PLAN_NAMES[name]}")
        cells(ax, coords, cmap(norm(data[name]["later_probability"])))
        acquired_boundaries(ax, I, data[name], coords)
        commitment_marks(ax, coords, data[name]["hosts_year0"])
    north_arrow(fig, 0.50, 0.74)
    fig.text(0.5, 0.215, "Schematic section lattice; each cell is one two-section unit", ha="center", fontsize=8)
    fig.text(0.5, 0.176, "Probability of later drilling (years 1–3)", ha="center", fontsize=8.5)
    cax = fig.add_axes([0.275, 0.132, 0.45, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=2, pad=2, labelsize=8)
    handles = [
        Line2D([], [], marker="s", linestyle="none", markersize=6.6, markerfacecolor="white",
               markeredgecolor="#111111", markeredgewidth=1.3, label="Hosts year-0 option"),
        Patch(facecolor="none", edgecolor=PKG_COLOR, linewidth=1.6, label="Acreage bought now"),
    ]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.062), ncol=2, frameon=False,
               handlelength=1.6, columnspacing=2.8, handletextpad=0.6)
    fig.text(0.5, 0.020, "Fill counts later options originating in each unit; incoming laterals are not shown.",
             ha="center", fontsize=8)
    save_figure(fig, output)


def context_figure(I, data, coords, central, output):
    from matplotlib import pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(FIGURE_WIDTH, 4.8))
    ax = map_axes(fig, 0.025, 0.43, 0.285, "(a) Calibrated inputs")
    mu = I["mu"][:, 0]
    norm, cmap = Normalize(float(mu.min()), float(mu.max())), plt.get_cmap("YlOrRd")
    cells(ax, coords, cmap(norm(mu)))
    group_boundaries(ax, coords, np.where(I["owned"], 0, -1), "#354B5B", linewidth=1.3, linestyle=(0, (3, 2)))
    for u, (x, y) in enumerate(coords):
        mb, tf = I["parents"][u]
        ax.text(x, y, f"{int(mb)}/{int(tf)}", ha="center", va="center", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.35, alpha=0.91), zorder=5)
    north_arrow(fig, 0.334, 0.74, height=0.06)
    cax = fig.add_axes([0.025, 0.393, 0.285, 0.014])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([18.5, 20, 22, 24])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=2, pad=2, labelsize=8)
    fig.text(0.1675, 0.322, "Mean revenue per 2-mile MB well\n(USD million)", ha="center", va="top", fontsize=8)
    for x, name, letter in ((0.358, "rn", "b"), (0.691, "ce", "c")):
        ax = map_axes(fig, x, 0.43, 0.285, f"({letter}) {PLAN_NAMES[name]}")
        years = data[name]["central_years"]
        cells(ax, coords, [YEAR_COLORS[y] if y >= 0 else NONE_COLOR for y in years])
        acquired_boundaries(ax, I, data[name], coords)
    fig.text(0.667, 0.399, "Earliest home-unit option year", ha="center", fontsize=8.5)
    year_handles = [Patch(facecolor=color, edgecolor=GRID_COLOR, linewidth=0.5, label=f"Year {year}")
                    for year, color in enumerate(YEAR_COLORS)]
    fig.legend(handles=year_handles, loc="center", bbox_to_anchor=(0.667, 0.362), ncol=4,
               frameon=False, fontsize=8, handlelength=1.1, columnspacing=1.2, handletextpad=0.5)
    fig.legend(handles=[Patch(facecolor=NONE_COLOR, edgecolor=GRID_COLOR, linewidth=0.5, label="No home-unit option"),
                        Patch(facecolor="none", edgecolor=PKG_COLOR, linewidth=1.6, label="Acreage bought now")],
               loc="center", bbox_to_anchor=(0.667, 0.315), ncol=2, frameon=False, fontsize=8,
               handlelength=1.3, columnspacing=1.2, handletextpad=0.5)
    fig.text(0.5, 0.240, f"Central quadrature node: type curve {I['theta'][central]:.3f}; "
             f"price {I['price'][central]:.3f} of strip; probability 4/9", ha="center", fontsize=8)
    fig.legend(handles=[Patch(facecolor="none", edgecolor="#354B5B", linewidth=1.3, linestyle=(0, (3, 2)),
                              label="Initial modeled acreage (panel a)")], loc="center", bbox_to_anchor=(0.5, 0.198),
               frameon=False, fontsize=8.5, handlelength=2.0)
    fig.text(0.5, 0.148, "Panel (a) labels: modeled Middle Bakken / Three Forks parents; capped at 6 per bench.",
             ha="center", fontsize=8)
    fig.text(0.5, 0.096, "Schematic section lattice; nominal unit: 1 × 2 miles. Year 0 is committed.",
             ha="center", fontsize=8)
    fig.text(0.5, 0.044, "Plans at USD 100 million risk tolerance. Incoming laterals are not shown.", ha="center", fontsize=8)
    save_figure(fig, output)


def sidecar(I, M, run, data, central, coords, args):
    check_counts = {}
    for name, d in data.items():
        check_counts[name] = dict(year0_wells=d["year0_wells"], year0_host_units=d["year0_host_count"],
                                 packages_bought=len(d["bought_packages"]), central_node_wells=d["central_wells"],
                                 expected_wells=d["expected_wells"], expected_long_lateral_share=d["expected_long_share"],
                                 wells_by_node=d["wells_by_node"].tolist(),
                                 central_units_by_first_year=np.bincount(d["central_years"][d["central_years"] >= 0], minlength=T).tolist(),
                                 central_units_without_home_option=int((d["central_years"] < 0).sum()))
    units = []
    for u in range(M.U):
        record = dict(unit=u, schematic_center_nominal_miles=coords[u].tolist(),
                      reference_centroid_miles_east_north=np.asarray(I["X"][u]).tolist(),
                      sections=[I["provenance"]["sections"][s] for s in I["secs"][u]],
                      north_neighbor=int(I["nbr"][u]), initially_in_modeled_acreage=bool(I["owned"][u]),
                      package=int(I["pkg_of"][u]), modeled_parents_mb_tf=np.asarray(I["parents"][u]).tolist(),
                      calibrated_mean_revenue_mb_usd_million=float(I["mu"][u, 0]), plans={})
        for name, d in data.items():
            record["plans"][name] = dict(hosts_year0_option=bool(d["hosts_year0"][u]),
                                          in_package_bought_now=bool(d["acquired_units"][u]),
                                          later_home_option_probability=float(d["later_probability"][u]),
                                          later_home_option_by_node=d["later_by_node"][:, u].tolist(),
                                          any_home_option_probability=float(d["any_home_probability"][u]),
                                          any_new_lateral_touches_unit_probability=float(d["any_touched_probability"][u]),
                                          central_first_home_option_year=int(d["central_years"][u]))
        units.append(record)
    return dict(
        schema="ndic-placement-figure-v2",
        sources=dict(instance=source_path(args.pkl), instance_sha256=sha256(args.pkl),
                     run=source_path(run["_file"]), run_sha256=sha256(run["_file"]),
                     script=source_path(__file__), script_sha256=sha256(__file__),
                     options_sha1=options_sha1(M), model_version=run["model_version"],
                     solver_rev=run["solver_rev"], status=run["status"], seed=run["seed"], rho_usd_million=run["rho"]),
        definitions=dict(
            fill="sum_k ordinary_probability[k] * indicator(any selected option originating in this unit in years 1--3 at node k)",
            filled_square="unit hosts at least one common year-0 option; it can also have later drilling",
            orange_outline="perimeter of acreage packages bought in the first stage",
            zero="no later home-unit option in any node; does not imply no parents, committed drilling, or incoming laterals",
            coordinates="regular schematic lattice; geographic reference centroids are preserved separately",
            supplementary_year="earliest year of a selected option originating in the unit at the central quadrature node; -1 means none",
            parent_counts="modeled per-bench counts, capped at six, not raw observed well counts",
            initial_acreage="modeled access footprint derived from operated parents and permits, not exclusive legal title",
            long_share="expected count of 3- and 4-mile wells divided by expected total wells",
        ),
        tree=dict(ordinary_probabilities=I["prob"].tolist(), type_curve_multipliers=I["theta"].tolist(),
                  price_multipliers_relative_to_strip=I["price"].tolist(), central_node_index=central,
                  central_type_curve_multiplier=float(I["theta"][central]), central_price_multiplier=float(I["price"][central]),
                  central_probability=float(I["prob"][central])),
        validation=dict(option_sha1_matches=True, probabilities_sum_to_one=True, quadrature_weights_match=True,
                        node_values_match_saved_scores=True, plans_feasible=True, fixed_reference_counts_match=True,
                        no_solver_called=True, check_counts=check_counts),
        presentation=dict(width_inches=FIGURE_WIDTH, ordinary_text_minimum_points=8, font="Arial", pdf_fonttype=42,
                          main_pdf=source_path(args.out), context_pdf=source_path(args.context_out)),
        units=units,
    )


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkl", type=Path, default=here / "../../data/derived/ndic_mckenzie_40u.pkl")
    parser.add_argument("--runs", default=str(here / "results/replicates/ndic_r*.json"))
    parser.add_argument("--rho", type=float, default=EXPECTED_RHO, choices=[EXPECTED_RHO], help="paper risk tolerance, USD million")
    parser.add_argument("--out", type=Path, default=here / "../../paper/figures/ndic_placement.pdf")
    parser.add_argument("--context-out", type=Path, default=here / "../../paper/figures/ndic_context_supplement.pdf")
    args = parser.parse_args()
    require(args.out.suffix == ".pdf" and args.context_out.suffix == ".pdf", "Both figure outputs must be PDFs.")
    I, M, run, data, central, coords = load_data(args)
    configure_plotting()
    main_figure(I, data, coords, args.out)
    context_figure(I, data, coords, central, args.context_out)
    args.out.with_suffix(".json").write_text(json.dumps(sidecar(I, M, run, data, central, coords, args), indent=2) + "\n")
    for name, d in data.items():
        print(f"{name}: year-0 wells {d['year0_wells']}; hosts {d['year0_host_count']}; "
              f"packages {len(d['bought_packages'])}; central wells {d['central_wells']}; expected wells {d['expected_wells']:.8f}")
    print(f"Wrote {args.out.resolve()} and {args.context_out.resolve()}, matching PNGs, and the main JSON sidecar.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError) as error:
        sys.exit(f"fig_ndic_placement: {error}")
