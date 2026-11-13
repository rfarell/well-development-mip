"""Plotting functions for three supplement figures (the deck curve, the regime map and the seed-0 placement map) and
the summary numbers of the two risk-tolerance sweeps. Used by fig_si.py and paper/make_sweep_macros.py."""

import math

BAND_PCT = 5.0                                   # materiality band, % of E[NPV](rn)
FAMILY = ("rn_deck", "rn_tilt1", "rn_tilt2")     # the deck family

WARNINGS = []


def warn(msg):
    WARNINGS.append(msg)
    print(f"warning: {msg}", flush=True)


def finite(x):
    try:
        return x is not None and not isinstance(x, bool) and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _fixed(x, d):
    """Fixed decimals; a negative value that rounds to zero keeps its sign (-0.00: the comparator was ahead), an exact
    zero does not."""
    return f"{abs(float(x)) if float(x) == 0.0 else float(x):.{d}f}"


def mval(x, d=2):
    if not finite(x):
        return None
    s = f"{round(float(x)):,}".replace(",", "{,}") if d == 0 else _fixed(x, d)
    return f"\\ensuremath{{{s}}}"


def sweep_macros(sweeps):
    rows = [(d, r) for d in sweeps for r in d["rows"] if finite(r.get("rn_enpv")) and r["rn_enpv"]]
    out = [("SweepSeeds", str(len(sweeps)), "rho sweeps (instances) read")]
    if not rows:
        return out
    pc = lambda r, k: 100.0 * r[k] / abs(r["rn_enpv"])  # noqa: E731

    def vof_iv(r):
        """Value of flexibility of a sweep row as a certificate interval [ce - committed_ub, ce_ub - committed_ce], % of
        E[NPV](rn); the point value_of_recourse when a bracket end is missing."""
        if all(finite(r.get(k)) for k in ("ce", "ce_ub", "committed_ce", "committed_ub")):
            return pc(dict(x=r["ce"] - r["committed_ub"], rn_enpv=r["rn_enpv"]), "x"), pc(dict(x=r["ce_ub"] - r["committed_ce"], rn_enpv=r["rn_enpv"]), "x")
        return pc(r, "value_of_recourse"), pc(r, "value_of_recourse")

    out += [("SweepRhoMin", mval(min(r["rho"] for _, r in rows), 0), "smallest rho in the sweeps, $M"),
            ("SweepRhoMax", mval(max(r["rho"] for _, r in rows), 0), "largest rho in the sweeps, $M"),
            ("SweepLamSigmaMin", mval(min(r["rn_lam_sigma"] for _, r in rows)), "smallest lambda*sigma_W(rn) in the sweeps"),
            ("SweepLamSigmaMax", mval(max(r["rn_lam_sigma"] for _, r in rows)), "largest lambda*sigma_W(rn) in the sweeps"),
            ("SweepGainPctMax", mval(max(pc(r, "ce_gain") for _, r in rows), 1), "largest CE gain over rn in the sweeps, % of E[NPV](rn)"),
            ("SweepCostPctMax", mval(max(pc(r, "enpv_cost") for _, r in rows), 1), "largest E[NPV] cost in the sweeps, % of E[NPV](rn)"),
            ("SweepVoRPctMin", mval(min(pc(r, "value_of_recourse") for _, r in rows), 1),
             "smallest value of flexibility (vs a committed schedule; legacy name VoR) in the sweeps, point CE difference, % of E[NPV](rn)"),
            ("SweepVoRPctMax", mval(max(pc(r, "value_of_recourse") for _, r in rows), 1),
             "largest value of flexibility (legacy name VoR) in the sweeps, point CE difference, % of E[NPV](rn)"),
            ("SweepVoFPctMin", mval(min(vof_iv(r)[0] for _, r in rows), 1),
             "value of flexibility (vs a committed schedule) in the sweeps: smallest lower end of the certificate interval, % of E[NPV](rn)"),
            ("SweepVoFPctMax", mval(max(vof_iv(r)[1] for _, r in rows), 1),
             "value of flexibility in the sweeps: largest upper end of the certificate interval, % of E[NPV](rn)")]
    return out


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def fig_regime_map(plt, sweeps, runs, path):
    panels = (("ce_gain", "CE gain over the risk-neutral plan"), ("enpv_cost", "Expected-NPV cost (Beck)"),
              ("value_of_recourse", "Value of flexibility\n(vs a committed schedule)"))
    B = [r for r in runs if isinstance(r.get("beck"), dict) and finite(r["beck"].get("lam_sigma_rn")) and r["beck"]["lam_sigma_rn"] > 0]
    if not sweeps and not B:
        warn("regime map: no sweep rows and no bench rows, figure skipped")
        return False
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.0), sharex=True)               # drawn at the text width
    axes[1].sharey(axes[0])                                          # gain and cost on one scale: Beck's comparison
    multiU = len({d.get("U") for d in sweeps}) > 1
    for i, d in enumerate(sweeps):
        rows = sorted((r for r in d["rows"] if finite(r.get("rn_lam_sigma")) and r["rn_lam_sigma"] > 0 and r.get("rn_enpv")),
                      key=lambda r: r["rn_lam_sigma"])
        if not rows:
            continue
        lab = f"sweep, seed {d.get('seed')}" + (f", U={d.get('U')}" if multiU else "") + f" ({len(rows)} values of $\\rho$)"
        for ax, (k, _) in zip(axes, panels):
            ax.plot([r["rn_lam_sigma"] for r in rows], [100.0 * r[k] / abs(r["rn_enpv"]) for r in rows], "-o", color=f"C{i % 10}",
                    lw=1.2, ms=3.5, label=lab)
    if B:
        kw = dict(marker="D", s=18, facecolors="none", edgecolors="k", linewidths=0.8, zorder=3, label=r"stress cases (area, $\rho$)")
        xb = [r["beck"]["lam_sigma_rn"] for r in B]
        axes[0].scatter(xb, [r["beck"]["ce_gain_pct"] for r in B], **kw)
        axes[1].scatter(xb, [r["beck"]["enpv_cost_pct"] for r in B], **kw)
        L = [r for r in B if isinstance(r.get("ladder"), dict) and r["beck"].get("enpv_rn")]
        if L:
            axes[2].scatter([r["beck"]["lam_sigma_rn"] for r in L],
                            [100.0 * r["ladder"]["value_of_recourse"] / abs(r["beck"]["enpv_rn"]) for r in L], **kw)
    axes[0].axhline(BAND_PCT, color="0.45", ls="--", lw=0.8, label=f"threshold, {BAND_PCT:g}% of $E(\\mathbf{{z}}_N)$")
    for ax, (_, title) in zip(axes, panels):
        ax.set_xscale("log")
        ax.axhline(0.0, color="0.75", lw=0.6, zorder=0)
        ax.set_title(title)
        ax.set_xlabel(r"$\lambda\sigma_W(\mathbf{z}_N)$")
        ax.grid(True, which="major", lw=0.3, alpha=0.5)
    axes[0].set_ylabel(r"% of $E(\mathbf{z}_N)$")
    hl = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l, h)
    ncol = min(len(hl), 3)
    fig.legend(list(hl.values()), list(hl), loc="lower center", ncol=ncol, frameon=False)
    fig.tight_layout(rect=(0, (0.1 + 0.17 * math.ceil(len(hl) / ncol)) / fig.get_figheight(), 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_placement(plt, sweep, path):
    """Panel (a): the mean revenue map, ownership and parent wells. Panel (b): placement across the sweep. No-regret
    means first year at every rho; never means no node at any rho; the rest are split into first year at some rho and
    later years only. The sets are recomputed from the rows' dev_year0 and dev_any."""
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    from instance import make_instance, NB
    I = make_instance(sweep["nx"], sweep["ny"], seed=sweep["seed"])
    X, U, nx, ny = I["X"], int(I["U"]), sweep["nx"], sweep["ny"]
    if int(sweep.get("U", U)) != U:
        warn(f"placement: sweep U={sweep.get('U')} but the regenerated instance has U={U}; figure skipped")
        return False

    def unit_rect(ax, u, color, alpha=1.0, hatch=None, lw=0.5):
        x, y = X[u, 0] - 0.5, X[u, 1] - 1.0
        ax.add_patch(plt.Rectangle((x, y), 1, 2, facecolor=color, alpha=alpha, hatch=hatch, edgecolor="k", linewidth=lw))

    def basin_axes(ax, title):
        ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.3))
    mu = I["mu"][:, 0]
    norm, cmap = Normalize(mu.min(), mu.max()), plt.get_cmap("YlOrRd")
    for u in range(U):
        unit_rect(axes[0], u, cmap(norm(mu[u])), hatch=None if I["owned"][u] else "//", lw=0.4)
        if I["parents"][u].sum() > 0:
            axes[0].plot(X[u, 0], X[u, 1], "k.", ms=4)
    basin_axes(axes[0], "(a) Mean revenue per well, Middle Bakken\nhatched: not owned; dots: parent wells")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0], fraction=0.06, pad=0.03, label="USD million per well")
    rows = sorted(sweep["rows"], key=lambda r: r["rho"])
    y0 = [set(r["dev_year0"]) for r in rows]
    anyd = [set(r.get("dev_any", r["dev_year0"])) | s for r, s in zip(rows, y0)]
    core, first, ever = set.intersection(*y0), set.union(*y0), set.union(*anyd)
    if "no_regret_year0" in sweep and set(sweep["no_regret_year0"]) != core:
        warn(f"placement: recomputed no-regret set differs from the sweep's stored no_regret_year0 ({sweep['_file']})")
    cats = (("#1b7837", r"no-regret: first year at every $\rho$", core), ("#fdae61", r"first year at some $\rho$", first - core),
            ("#9ecae1", r"later years only (some node, some $\rho$)", ever - first), ("#d9d9d9", "never developed", None))
    nub = U * NB
    for u in range(U):
        ubs = [u * NB + b for b in range(NB)]
        col = next((c for c, _, s in cats if s is not None and any(x in s for x in ubs)), cats[-1][0])
        unit_rect(axes[1], u, col, lw=0.4)
    rr = [r["rho"] for r in rows]
    span = (f"$\\rho$ = USD {rr[0]:,.0f} million only (sweep incomplete)" if len(rr) == 1
            else f"{len(rr)} values of $\\rho$: USD {rr[0]:,.0f} to {rr[-1]:,.0f} million")
    basin_axes(axes[1], f"(b) Placement across $\\rho$, sweep seed {sweep['seed']}\n{span}")
    counts = [len(core), len(first - core), len(ever - first), nub - len(ever | first)]
    axes[1].legend(handles=[Patch(facecolor=c, edgecolor="k", lw=0.4,
                                label=f"{l} ({k} {'unit-bench' if k == 1 else 'unit-benches'})")
                           for (c, l, _), k in zip(cats, counts)],
                   loc="upper center", bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _inst_colors(runs):
    keys = list(dict.fromkeys((r.get("U"), r.get("seed")) for r in runs))
    multiU = len({k[0] for k in keys}) > 1
    return {k: (f"C{i % 10}", f"seed {k[1]}" + (f", U={k[0]}" if multiU else "")) for i, k in enumerate(keys)}


def fig_deck_curve(plt, runs, path):
    """The stored deck curve: the risk-neutral MIP at every uniform planning deck, scored under the true model. Top: its
    CE relative to z_A, with the best deck-family member (dashed), the best deck (star) and the registered primary band
    (dotted, -5%); bottom: its E[NPV] relative to z_N, with the NPV-tuned rule at its own deck (cross). Both as % of
    E[NPV](rn), one column per rho, one color per instance."""
    R = [r for r in runs if r.get("methods", {}).get("rn_deck", {}).get("deck_curve") and "ce" in r["methods"] and "rn" in r["methods"]]
    if not R:
        warn("deck curve: no bench run stores a deck curve; figure skipped")
        return False
    from matplotlib.lines import Line2D
    rhos = sorted({float(r["rho"]) for r in R})
    nc = len(rhos)
    fig, axes = plt.subplots(2, nc, figsize=(6.5, 4.9), sharex=True, squeeze=False)
    colors = _inst_colors(R)
    rule_decks = set()
    for j, rho in enumerate(rhos):
        ax, axe = axes[0, j], axes[1, j]
        for r in [r for r in R if float(r["rho"]) == rho]:
            m = r["methods"]; E = abs(m["rn"]["enpv"]); c, _ = colors[(r.get("U"), r.get("seed"))]
            cur = sorted(m["rn_deck"]["deck_curve"], key=lambda e: e["deck"])
            x = [e["deck"] for e in cur]
            ax.plot(x, [100 * (e["ce"] - m["ce"]["ce"]) / E for e in cur], "-o", color=c, lw=1.0, ms=2.2)
            _, vb = best_family(m)
            if finite(vb):
                ax.axhline(100 * (vb - m["ce"]["ce"]) / E, color=c, ls="--", lw=0.8)
            if finite(m["rn_deck"].get("deck")) and finite(m["rn_deck"].get("ce")):
                ax.plot(m["rn_deck"]["deck"], 100 * (m["rn_deck"]["ce"] - m["ce"]["ce"]) / E, "*", color=c, ms=7, mec="k", mew=0.4, zorder=4)
            axe.plot(x, [100 * (e["enpv"] - m["rn"]["enpv"]) / E for e in cur], "-o", color=c, lw=1.0, ms=2.2)
            dk = m.get("heur_ev", {}).get("cfg", {}).get("deck")
            if finite(dk) and "enpv" in m.get("heur_ev", {}):
                rule_decks.add(dk)
                axe.plot(dk, 100 * (m["heur_ev"]["enpv"] - m["rn"]["enpv"]) / E, "x", color=c, ms=6, mew=1.3, zorder=4)
        ax.axhline(0.0, color="k", lw=0.9)
        ax.axhline(-BAND_PCT, color="0.4", ls=":", lw=0.9)
        axe.axhline(0.0, color="k", lw=0.9)
        for a in (ax, axe):
            a.axvline(1.0, color="0.55", ls="-.", lw=0.7)
            for dk in rule_decks:
                a.axvline(dk, color="0.75", ls="-", lw=0.6, zorder=0)
            a.grid(True, lw=0.3, alpha=0.5)
        ax.set_title(f"$\\rho$ = USD {rho:,.0f} million")
        axe.set_xlabel("planning deck (multiple of strip revenue)")
    axes[0, 0].set_ylabel("CE(deck plan) $-$ CE($\\mathbf{z}_A$)\n% of $E(\\mathbf{z}_N)$")
    axes[1, 0].set_ylabel("$E$(deck plan) $-$ $E(\\mathbf{z}_N)$\n% of $E(\\mathbf{z}_N)$")
    handles = [Line2D([], [], color=c, marker="o", ms=3, lw=1.0, label=lab) for c, lab in colors.values()]
    handles += [Line2D([], [], color="0.3", ls="--", lw=0.8, label="best deck-family member"),
                Line2D([], [], color="0.3", marker="*", ls="", ms=7, label="best deck (CE-tuned)"),
                Line2D([], [], color="0.3", marker="x", ls="", ms=6, mew=1.3, label="rank-and-fill rule at its deck"),
                Line2D([], [], color="0.4", ls=":", lw=0.9, label=f"threshold ($-${BAND_PCT:g}% of $E(\\mathbf{{z}}_N)$)"),
                Line2D([], [], color="0.55", ls="-.", lw=0.7, label="strip (deck 1)")]
    ncol = 3
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False)
    H = fig.get_figheight()
    fig.tight_layout(rect=(0, (0.1 + 0.17 * math.ceil(len(handles) / ncol)) / H, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return True


def best_family(m, key="ce"):
    """(name, value) of the best deck-family member by `key` (in-sample CE by default), (None, nan) without one."""
    fam = [x for x in FAMILY if x in m and finite(m[x].get(key))]
    if not fam:
        return None, float("nan")
    b = max(fam, key=lambda x: m[x][key])
    return b, float(m[b][key])
