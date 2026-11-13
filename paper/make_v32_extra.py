"""Write paper/sections/results_v32_extra.tex and paper/sections/results_numbers_extra.tex.

Reads the stress-case and conventions run files and writes four booktabs tables with their headline macros:

  tab:vss-clean       the value of the stochastic solution reported apart, per instance and rho: the RN-VSS
                      E[NPV](rn) - E[NPV](mv_rn) (rn's and the rolling deterministic re-solve's dual bounds give its
                      interval) and the CE-VSS CE(ce) - CE(mv_ce+exact_recourse) (the two certificates give its
                      interval), beside the mixed VSS of the v3.1 ladder (run['ladder'] keys rn_vss*, ce_vss*,
                      vss_mixed*, enpv_rn);
  tab:perturb-reopt   (main text, compact) z_A's first stage against the best deck-family member under each perturbed
                      truth with the recourse of every first stage re-optimized under that truth
                      (run['perturbed_reopt']['zA_vs_family'][truth]): one row per truth, B/T/L and the interval range in %
                      of E(z_N) at each rho, and a row over all truths;
  tab:perturb-reopt-detail  (SI) the same per instance in $M, with the expected-NPV shortfall of the oracle rule's first
                      stage against z_N's, both with risk-neutral recourse re-optimized under the truth
                      (run['perturbed_reopt']['table'][truth][method]);
  tab:conventions     seed 0's re-solves under other terminal-credit shares nu and hedge conventions f0
                      (the conventions run file, schema in src/model/exp_conventions.py).

Read-only on the result files; no solver and no model code. Run from anywhere:
    OMP_NUM_THREADS=2 python paper/make_v32_extra.py
"""
import datetime
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "model"))
from report_currency import format_table_currency   # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "src", "model", "results"))
# The files are selected by BENCH_PREFIX (default b40v33) and CONVENTIONS_FILE (default conventions40_v33_s100.json).
# Runs of another model version are skipped with a warning.
PREFIX = os.environ.get("BENCH_PREFIX", "b40v33")
CONV_FILE = os.environ.get("CONVENTIONS_FILE", "conventions40_v33_s100.json")
REQUIRED_MODEL_VERSION = "v3.3"

OUT_TAB = os.path.join(HERE, "sections", "results_v32_extra.tex")
OUT_NUM = os.path.join(HERE, "sections", "results_numbers_extra.tex")

SEEDS = (0, 1, 2, 3)
RHOS = (100, 300)
SUFFIX = {100: "RhoOneHundred", 300: ""}          # unsuffixed = USD 300 million
TIE = 1e-4                                         # tie tolerance, share of E[NPV](rn)
TRUTHS = [("mu_bias_0.85", r"Type-curve level $\times$0.85"), ("mu_bias_1.15", r"Type-curve level $\times$1.15"),
          ("h_slope_0.5", r"Interference slope $\times$0.5"), ("h_slope_1.5", r"Interference slope $\times$1.5"),
          ("range_0.5", r"Field range $\times$0.5"), ("range_2.0", r"Field range $\times$2.0"),
          ("eur4_1.4", r"Four-mile EUR multiplier 1.4"), ("eur4_1.7", r"Four-mile EUR multiplier 1.7"),
          ("sig_p_0.15", r"Price volatility 0.15"), ("sig_p_0.35", r"Price volatility 0.35")]
CONV_LABEL = {"base": r"Base case ($\nu=0.5$, $f_0=0.6$)", "nu=0": r"$\nu=0$", "nu=0.25": r"$\nu=0.25$",
              "nu=0.75": r"$\nu=0.75$", "nu=1": r"$\nu=1$", "f0=0.73,pdp=0.68": r"$f_0=0.73$, base share 0.68",
              "f0=0": r"$f_0=0$, new wells unhedged"}


def run(s, rho):
    path = os.path.join(RES, f"{PREFIX}_s{s}_r{rho}.json")
    d = json.load(open(path))
    assert d.get("version") in ("v3.2", "v3.3") and len(d["runs"]) == 1 and not d["errors"], path
    r = d["runs"][0]
    assert r["seed"] == s and float(r["rho"]) == float(rho) and not r["errors"], path
    assert r.get("model_version") == REQUIRED_MODEL_VERSION, f"{path}: model_version {r.get('model_version')!r} != {REQUIRED_MODEL_VERSION}"
    return r


def f2(x, signed=False):
    """Two decimals, LaTeX minus; -0.00 prints as 0.00."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    v = round(x, 2)
    if v == 0:
        v = 0.0
    s = f"{abs(v):.2f}"
    if v < 0:
        return "$-$" + s
    return ("+" if signed and v > 0 else "") + s


def f1(x):
    v = round(x, 1)
    v = 0.0 if v == 0 else v
    return ("$-$" if v < 0 else "") + f"{abs(v):.1f}"


def iv(lo, hi):
    return f"[{f2(lo)}, {f2(hi)}]"


def macro(name, val, comment, out):
    out.append(f"% {comment}")
    out.append(rf"\newcommand{{\{name}}}{{{val}}}")


def num(x, d=2):
    v = round(x, d)
    v = 0.0 if v == 0 else v
    s = f"{abs(v):.{d}f}"
    return r"\ensuremath{" + ("-" if v < 0 else "") + s + "}"


RUNS = {(s, rho): run(s, rho) for s in SEEDS for rho in RHOS}


def table_open(caption, label, colspec, header):
    return [r"\begin{table}[tbp]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering\footnotesize",
            r"\setlength{\tabcolsep}{3pt}", r"\smallskip",
            r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%", rf"\begin{{tabular}}{{{colspec}}}",
            r"\toprule"] + header + [r"\midrule"]


def table_close():
    return [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]


tab, mac = [], []

# ---------------------------------------------------------------- tab:vss-clean
cap = (r"The value of the stochastic solution reported apart. RN-VSS: $E(\mathbf z_N)$ less the expected NPV of "
       r"the mean-value first stage ($\theta=P=1$, solved risk-neutrally) given its risk-neutral optimal recourse on the "
       r"tree, which on this tree is the rolling deterministic re-solve; its interval comes from the dual bounds of the two "
       r"MIPs. CE-VSS: $\mathrm{CE}(\mathbf z_A)$ less the certainty equivalent of the first stage of the certainty-equivalent "
       r"problem at the factor means (same $\lambda$) given its certified CE recourse on the tree, with the certificate "
       r"interval $[\mathrm{LB}_a-\mathrm{UB}_b,\ \mathrm{UB}_a-\mathrm{LB}_b]$. Mixed: the VSS of Table~\ref{tab:ladder}, "
       r"whose mean-value first stage is solved risk-neutrally and given CE-optimal recourse, so it mixes the value of the "
       r"tree with that of risk aversion; it is kept for comparison with the v3.1 runs. Entries are the point value (the "
       r"incumbents' difference) with the interval in \$M, and the interval as a share of $E(\mathbf z_N)$. Neither "
       r"$\mathbf z_N$ nor the risk-neutral mean-value plan depends on $\rho$, so the RN-VSS repeats in the two blocks. "
       r"Where the CE mean-value first stage is the risk-neutral one (seed~0 at both $\rho$, seed~1 at \$300M), the CE-VSS "
       r"equals the mixed VSS. Range: smallest and largest point value over the instances, in \$M and as a share of $E(\mathbf z_N)$.")
tab += table_open(cap, "tab:vss-clean", "lrrrrrrr",
                  [r"Instance & $E(\mathbf z_N)$ & RN-VSS, \$M & \% & CE-VSS, \$M & \% & Mixed VSS, \$M & \% \\"])
agg = {}
for rho in RHOS:
    tab.append(rf"\multicolumn{{8}}{{l}}{{\emph{{$\rho$ = \${rho}M, 4 instances (s0, s1, s2, s3), $U=40$}}}} \\")
    for s in SEEDS:
        L = RUNS[(s, rho)]["ladder"]
        den = L["enpv_rn"]
        cells = [f"s{s}", f"{den:.2f}"]
        for k in ("rn_vss", "ce_vss", "vss_mixed"):
            assert L[k + "_lb"] - 1e-9 <= L[k] <= L[k + "_ub"] + 1e-9
            cells += [f"{f2(L[k])} {iv(L[k + '_lb'], L[k + '_ub'])}", f"[{100 * L[k + '_lb'] / den:.2f}, {100 * L[k + '_ub'] / den:.2f}]"]
            a = agg.setdefault((rho, k), {"v": [], "p": [], "w": []})
            a["v"].append(L[k]); a["p"].append(100 * L[k] / den); a["w"].append(L[k + "_ub"] - L[k + "_lb"])
        tab.append(" & ".join(cells) + rf" \\  % s{s} rho {rho}")
    cells = ["Range", ""]
    for k in ("rn_vss", "ce_vss", "vss_mixed"):
        a = agg[(rho, k)]
        cells += [f"{f2(min(a['v']))} to {f2(max(a['v']))}", f"{min(a['p']):.2f} to {max(a['p']):.2f}"]
    tab.append(r"\cmidrule(l){1-8}")
    tab.append(" & ".join(cells) + r" \\  % smallest and largest point value over instances")
    if rho != RHOS[-1]:
        tab.append(r"\midrule")
tab += table_close()

mac.append("% ---- tab:vss-clean (run['ladder']: rn_vss, ce_vss, vss_mixed and their _lb/_ub; enpv_rn) ----")
a = agg[(100, "rn_vss")]
for rho in RHOS:   # RN-VSS does not depend on rho; check
    assert all(abs(x - y) < 1e-9 for x, y in zip(agg[(rho, "rn_vss")]["v"], a["v"]))
macro("RnVssMin", num(min(a["v"])), "RN-VSS E[NPV](rn) - E[NPV](mv_rn): smallest point value over the instances, $M (does not depend on rho)", mac)
macro("RnVssMax", num(max(a["v"])), "RN-VSS: largest point value over the instances, $M", mac)
macro("RnVssPctMin", num(min(a["p"])), "RN-VSS: smallest, % of E[NPV](rn)", mac)
macro("RnVssPctMax", num(max(a["p"])), "RN-VSS: largest, % of E[NPV](rn)", mac)
for rho in RHOS:
    sfx = SUFFIX[rho]
    for k, nm, lab in (("ce_vss", "CeVss", "CE-VSS CE(ce) - CE(mv_ce+exact_recourse)"), ("vss_mixed", "MixedVss", "mixed VSS CE(ce) - CE(mv+exact_recourse)")):
        b = agg[(rho, k)]
        macro(f"{nm}Min{sfx}", num(min(b["v"])), f"{lab}: smallest point value over the instances, $M [rho = ${rho}M]", mac)
        macro(f"{nm}Max{sfx}", num(max(b["v"])), f"{lab}: largest point value, $M [rho = ${rho}M]", mac)
        macro(f"{nm}PctMin{sfx}", num(min(b["p"])), f"{lab}: smallest, % of E[NPV](rn) [rho = ${rho}M]", mac)
        macro(f"{nm}PctMax{sfx}", num(max(b["p"])), f"{lab}: largest, % of E[NPV](rn) [rho = ${rho}M]", mac)
wmax = max(max(agg[(rho, k)]["w"]) for rho in RHOS for k in ("rn_vss", "ce_vss"))
macro("VssCleanWidthMax", num(wmax), "largest width of an RN-VSS or CE-VSS interval over the eight runs, $M", mac)

# ---------------------------------------------------------------- tab:perturb-reopt (main) and tab:perturb-reopt-detail (SI)
# One pass over the runs fills PR[(rho, truth)] (the per-instance values of one row) and RE[rho] (the totals over the ten
# truths, which the macros below read); the two tables are then written from PR and RE.
PR, RE = {}, {}
for rho in RHOS:
    tot = dict(b=0, t=0, l=0, be=0, te=0, le=0, lo=1e9, hi=-1e9, plo=1e9, phi=-1e9, gp=[], rule=[], rulep=[], n=0, notbehind=0, shared=set())
    for key, lab in TRUTHS:
        g, lo, hi, plo, phi, bt, rule, rulep = [], [], [], [], [], [0, 0, 0], [], []
        for s in SEEDS:
            P = RUNS[(s, rho)]["perturbed_reopt"]
            assert not P.get("errors"), (s, rho)
            v = P["zA_vs_family"][key]
            rows = P["table"][key]
            den = rows["rn"]["enpv"]
            assert abs(v["enpv_zN"] - den) < 1e-9
            g.append(v["gain_ce"]); lo.append(v["gain_ce_lb"]); hi.append(v["gain_ce_ub"])
            plo.append(100 * v["gain_ce_lb"] / den); phi.append(100 * v["gain_ce_ub"] / den)
            tot["gp"].append(100 * v["gain_ce"] / den)
            if v["beats_ce"]:
                bt[0] += 1
            elif v["loses_ce"]:
                bt[2] += 1
            else:
                bt[1] += 1
            tot["be"] += v["beats_enpv"]; tot["le"] += v["loses_enpv"]; tot["te"] += (not v["beats_enpv"] and not v["loses_enpv"])
            if v["gain_ce"] == 0.0:
                tot["shared"].add(s)
            sh = rows["rn"]["enpv"] - rows["heur_ce"]["enpv"]
            rule.append(sh); rulep.append(100 * sh / den)
            tot["notbehind"] += (sh <= TIE * abs(den))
            tot["n"] += 1
        tot["b"] += bt[0]; tot["t"] += bt[1]; tot["l"] += bt[2]
        tot["lo"] = min(tot["lo"], min(lo)); tot["hi"] = max(tot["hi"], max(hi))
        tot["plo"] = min(tot["plo"], min(plo)); tot["phi"] = max(tot["phi"], max(phi))
        tot["rule"] += rule; tot["rulep"] += rulep
        PR[(rho, key)] = dict(g=g, lo=lo, hi=hi, plo=plo, phi=phi, bt=bt, rule=rule, rulep=rulep)
    RE[rho] = tot


def btl(bt):
    return f"{bt[0]}/{bt[1]}/{bt[2]}"


# main-text table: one row per truth, wins/ties/losses and the interval range in % at each rho
cap = (r"Perturbed truths with the recourse re-optimized (four instances, $U=40$). Under each truth, one "
       r"perturbation at a time, the first stage of each plan is held fixed and its recourse is re-optimized under that "
       r"truth, certified for the certainty equivalent. The first stage of $\mathbf z_A$ is compared with the best member "
       r"of the deck family (the uniform-deck MIP and the tilted MIPs after one and two iterations), the best chosen under "
       r"each truth by the lower end of its bracket, which favors the family. B/T/L: instances whose certificate interval "
       r"of the gain, $[\mathrm{LB}_A-\max\mathrm{UB},\ \mathrm{UB}_A-\max\mathrm{LB}]$, lies above zero, contains it, or "
       r"lies below it. Interval: the smallest lower end and the largest upper end of these intervals over the four "
       r"instances, in \% of $E(\mathbf z_N)$ under the same truth ($\mathbf z_N$'s first stage with risk-neutral recourse "
       r"re-optimized). All truths: the counts and the range over the 40 instance--truth pairs. The per-instance gains in "
       r"\$M and the comparison with the oracle rule are in the SI.")
tab += table_open(cap, "tab:perturb-reopt", "lcccc",
                  [" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$\rho=\${rho}$M}}" for rho in RHOS) + r" \\",
                   "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(RHOS))),
                   "Truth" + r" & B/T/L & Interval, \%" * len(RHOS) + r" \\"])
for key, lab in TRUTHS:
    cells = [lab]
    for rho in RHOS:
        q = PR[(rho, key)]
        cells += [btl(q["bt"]), f"[{f2(min(q['plo']))}, {f2(max(q['phi']))}]"]
    tab.append(" & ".join(cells) + rf" \\  % {key}")
tab.append(r"\midrule")
cells = ["All truths"]
for rho in RHOS:
    t = RE[rho]
    cells += [f"{t['b']}/{t['t']}/{t['l']}", f"[{f2(t['plo'])}, {f2(t['phi'])}]"]
tab.append(" & ".join(cells) + r" \\  % totals over the 40 instance-truth pairs at each rho")
tab += table_close()

# SI table: the per-instance gains in $M, the intervals in $M and %, and the oracle rule's columns
cap = (r"Perturbed truths with the recourse re-optimized, per instance (the detail of "
       r"Table~\ref{tab:perturb-reopt}). Under each truth, one perturbation at a time as in "
       r"Table~\ref{tab:perturbation}, the first stage of each plan is held fixed and its recourse is re-optimized under "
       r"that truth, certified for the certainty equivalent (at most 10 rounds of Algorithm~\ref{alg:oa}, 120~s per master) "
       r"and risk-neutral for expected NPV. Gain: the certainty equivalent of $\mathbf z_A$'s first stage less that of the "
       r"best member of the deck family under that truth (the uniform-deck MIP and the tilted MIPs after one and two "
       r"iterations, the best chosen by the lower end of its bracket), in \$M per instance in the order s0, s1, s2, s3. "
       r"Interval: the smallest lower end and the largest upper end over the instances of the certificate intervals "
       r"$[\mathrm{LB}_A-\max\mathrm{UB},\ \mathrm{UB}_A-\max\mathrm{LB}]$, in \$M and as shares of $E(\mathbf z_N)$ under the "
       r"same truth ($\mathbf z_N$'s first stage with risk-neutral recourse re-optimized). B/T/L: instances whose interval "
       r"lies above zero, contains it, or lies below it; a gain of 0.00 means that $\mathbf z_A$ and the best member commit "
       r"the same first stage. Rule: expected NPV of $\mathbf z_N$'s first stage less that of the oracle rank-and-fill "
       r"rule's (tuned by in-sample CE at that $\rho$; on seeds 0 and 1 the same configuration as the practice rule), both "
       r"with risk-neutral recourse re-optimized under the truth, in \$M per instance, and the smallest as a share of "
       r"$E(\mathbf z_N)$; a negative entry means that the rule's first stage earns more. Totals count instance--truth pairs.")
tab += table_open(cap, "tab:perturb-reopt-detail", "lrrrcrr",
                  [r"Truth & Gain over best member, \$M & Interval, \$M & Interval, \% & B/T/L & Rule: $E(\mathbf z_N)-E$, \$M & Rule min, \% \\"])
for rho in RHOS:
    tab.append(rf"\multicolumn{{7}}{{l}}{{\emph{{$\rho$ = \${rho}M, 4 instances (s0, s1, s2, s3), $U=40$; first stages of $\mathbf z_A$, $\mathbf z_N$, the deck family and the oracle rule}}}} \\")
    for key, lab in TRUTHS:
        q = PR[(rho, key)]
        tab.append(" & ".join([lab, "; ".join(f2(x) for x in q["g"]), iv(min(q["lo"]), max(q["hi"])), f"[{f2(min(q['plo']))}, {f2(max(q['phi']))}]",
                               btl(q["bt"]), "; ".join(f2(x) for x in q["rule"]), f1(min(q["rulep"]))]) + rf" \\  % {key}")
    tot = RE[rho]
    tab.append(r"\cmidrule(l){1-7}")
    tab.append(" & ".join(["All ten truths", "", iv(tot["lo"], tot["hi"]), f"[{f2(tot['plo'])}, {f2(tot['phi'])}]",
                           f"{tot['b']}/{tot['t']}/{tot['l']}", f"{f2(min(tot['rule']))} to {f2(max(tot['rule']))}", f1(min(tot["rulep"]))])
               + r" \\  % totals over instance-truth pairs")
    if rho != RHOS[-1]:
        tab.append(r"\midrule")
tab += table_close()

mac.append("% ---- tab:perturb-reopt (run['perturbed_reopt']: zA_vs_family[truth], table[truth][method]) ----")
for rho in RHOS:
    sfx, t = SUFFIX[rho], RE[rho]
    macro(f"ReoptCases{sfx}", str(t["n"]), f"instance-truth pairs with re-optimized recourse [rho = ${rho}M]", mac)
    macro(f"ReoptBeats{sfx}", str(t["b"]), f"pairs where z_A's CE interval over the best family member lies above 0 [rho = ${rho}M]", mac)
    macro(f"ReoptTies{sfx}", str(t["t"]), f"pairs where the interval contains 0 [rho = ${rho}M]", mac)
    macro(f"ReoptLoses{sfx}", str(t["l"]), f"pairs where the interval lies below 0 [rho = ${rho}M]", mac)
    macro(f"ReoptGainPctMin{sfx}", num(min(t["gp"])), f"z_A - best family member, CE, re-optimized recourse: smallest point value, % of E(z_N) under the truth [rho = ${rho}M]", mac)
    macro(f"ReoptGainPctMax{sfx}", num(max(t["gp"])), f"the same: largest point value, % [rho = ${rho}M]", mac)
    macro(f"ReoptSharedInst{sfx}", f"{len(t['shared'])}/4", f"instances where z_A's first stage is that of the best family member under some truth (a gain of exactly 0) [rho = ${rho}M]", mac)
    macro(f"ReoptRuleMin{sfx}", num(min(t["rule"])), f"E(z_N first stage) - E(oracle rule first stage), both with RN recourse re-optimized under the truth: smallest, $M [rho = ${rho}M]", mac)
    macro(f"ReoptRuleMax{sfx}", num(max(t["rule"])), f"the same: largest, $M [rho = ${rho}M]", mac)
    macro(f"ReoptRuleNotBehind{sfx}", str(t["notbehind"]), f"pairs where the rule's first stage is within the tie tolerance of z_N's or ahead [rho = ${rho}M]", mac)
A = {k: RE[100][k] + RE[300][k] for k in ("n", "b", "t", "l", "be", "te", "le", "notbehind")}
macro("ReoptCasesAll", str(A["n"]), "instance-truth pairs, both rho", mac)
macro("ReoptBeatsAll", str(A["b"]), "pairs where z_A beats the best family member (CE intervals separate above 0), both rho", mac)
macro("ReoptTiesAll", str(A["t"]), "pairs with a tie (interval contains 0), both rho", mac)
macro("ReoptLosesAll", str(A["l"]), "pairs where z_A loses (interval below 0), both rho", mac)
macro("ReoptGainPctMinAll", num(min(RE[100]["gp"] + RE[300]["gp"])), "z_A - best family member, re-optimized recourse: smallest point value over both rho, % of E(z_N) under the truth", mac)
macro("ReoptGainPctMaxAll", num(max(RE[100]["gp"] + RE[300]["gp"])), "the same: largest, %", mac)
macro("ReoptIntPctMinAll", num(min(RE[100]["plo"], RE[300]["plo"])), "smallest lower end of the gain's certificate interval over both rho, % of E(z_N) under the truth", mac)
macro("ReoptIntPctMaxAll", num(max(RE[100]["phi"], RE[300]["phi"])), "largest upper end of the interval over both rho, %", mac)
macro("ReoptEnpvBeatsAll", str(A["be"]), "pairs where z_A's first stage beats the best member on E[NPV] with RN recourse (bounds separate), both rho", mac)
macro("ReoptEnpvTiesAll", str(A["te"]), "pairs with an E[NPV] tie, both rho", mac)
macro("ReoptEnpvLosesAll", str(A["le"]), "pairs where z_A's first stage loses on E[NPV] with RN recourse, both rho", mac)
macro("ReoptRuleNotBehindAll", str(A["notbehind"]), "pairs where the oracle rule's first stage matches (tie tolerance) or beats z_N's on E[NPV], RN recourse re-optimized, both rho", mac)

# ---------------------------------------------------------------- tab:conventions
C = json.load(open(os.path.join(RES, CONV_FILE)))
CONV_SEED = int(C["seed"])
assert C["status"] == "done" and not C["errors"] and C.get("model_version", "v3.3") == REQUIRED_MODEL_VERSION, "conventions run"
order = ["base", "nu=0", "nu=0.25", "nu=0.75", "nu=1", "f0=0.73,pdp=0.68", "f0=0"]
cap = (rf"Terminal-credit and hedge conventions on seed~{CONV_SEED}, re-solved one at a time around the base case: "
       r"the share $\nu$ of the credited inventory's NPV counted at the horizon, and the hedge, either the 60-month "
       r"tenor ($f_0=0.73$ for a committed well's revenue, with the producing base's in-tenor share 0.68 in place of 0.5) "
       r"or no hedge on the new wells ($f_0=0$, the producing base's share staying 0.5). Under each convention $\mathbf z_A$ (certified, seeded as the warm start; every bracket closed "
       r"below 0.1\%) and $\mathbf z_N$ are re-solved and scored. CE and $E$ of $\mathbf z_A$ in \$M; gain: "
       r"$\mathrm{CE}(\mathbf z_A)-\mathrm{CE}(\mathbf z_N)$; cost: $E(\mathbf z_N)-E(\mathbf z_A)$; first-year wells and "
       r"packages of $\mathbf z_A$ / $\mathbf z_N$; Jaccard: the overlap of each plan's first-year unit-benches with the "
       r"same plan's in the base case at the same $\rho$.")
tab += table_open(cap, "tab:conventions", "lrrrrccc",
                  [r"Convention & $\mathrm{CE}(\mathbf z_A)$ & $E(\mathbf z_A)$ & Gain & Cost & Wells yr 0, $\mathbf z_A$ / $\mathbf z_N$ & Jaccard yr 0, $\mathbf z_A$ / $\mathbf z_N$ & Packages, $\mathbf z_A$ / $\mathbf z_N$ \\"])
CV = {}
for rho in RHOS:
    rows = {r["label"]: r for r in C["rows"] if float(r["rho"]) == float(rho)}
    assert set(rows) == set(order), rows.keys()
    tab.append(rf"\multicolumn{{8}}{{l}}{{\emph{{$\rho$ = \${rho}M, seed~{CONV_SEED}, $U=40$}}}} \\")
    for lab in order:
        r = rows[lab]
        assert not r["errors"], (rho, lab)
        A_, N_ = r["z_A"], r["z_N"]
        gain, cost = A_["ce"] - N_["ce"], N_["enpv"] - A_["enpv"]
        CV.setdefault(rho, []).append(dict(lab=lab, gain=gain, cost=cost, jA=A_["jaccard_year0_vs_base"], jN=N_["jaccard_year0_vs_base"],
                                           wA=A_["wells_year0"], wN=N_["wells_year0"], pA=A_["packages"], pN=N_["packages"],
                                           gap=100 * (A_["ce_ub"] - A_["ce_lb"]) / abs(A_["ce_lb"])))
        tab.append(" & ".join([CONV_LABEL[lab], f"{A_['ce']:.2f}", f"{A_['enpv']:.2f}", f2(gain), f2(cost),
                               f"{A_['wells_year0']:.0f} / {N_['wells_year0']:.0f}",
                               f"{A_['jaccard_year0_vs_base']:.2f} / {N_['jaccard_year0_vs_base']:.2f}",
                               f"{A_['packages']} / {N_['packages']}"]) + rf" \\  % {lab}, rho {rho}")
    if rho != RHOS[-1]:
        tab.append(r"\midrule")
tab += table_close()
assert max(c["gap"] for rho in RHOS for c in CV[rho]) < 0.1

mac.append("% ---- tab:conventions (results/conventions40_s0.json; seed 0) ----")
for rho in RHOS:
    sfx, rows = SUFFIX[rho], CV[rho]
    nus = [c for c in rows if c["lab"].startswith("nu=") or c["lab"] == "base"]
    macro(f"ConvGainMin{sfx}", num(min(c["gain"] for c in rows)), f"CE(z_A) - CE(z_N) over the seven conventions: smallest, $M [rho = ${rho}M]", mac)
    macro(f"ConvGainMax{sfx}", num(max(c["gain"] for c in rows)), f"the same: largest, $M [rho = ${rho}M]", mac)
    macro(f"ConvJaccardNuMin{sfx}", num(min(c["jA"] for c in nus)), f"z_A's first-year Jaccard against the base case over nu in 0..1: smallest [rho = ${rho}M]", mac)
    macro(f"ConvJaccardNuMinZN{sfx}", num(min(c["jN"] for c in nus)), f"z_N's first-year Jaccard against its base case over nu: smallest [rho = ${rho}M]", mac)
    macro(f"ConvWellsNuMin{sfx}", f"{min(c['wA'] for c in nus):.0f}", f"z_A's first-year wells over nu: fewest [rho = ${rho}M]", mac)
    macro(f"ConvWellsNuMax{sfx}", f"{max(c['wA'] for c in nus):.0f}", f"z_A's first-year wells over nu: most [rho = ${rho}M]", mac)
    pa = sorted({c["pA"] for c in rows}); pn = sorted({c["pN"] for c in rows})
    macro(f"ConvPackagesZA{sfx}", "/".join(str(x) for x in pa), f"packages z_A buys, over all seven conventions (one value: the same under every convention) [rho = ${rho}M]", mac)
    macro(f"ConvPackagesZN{sfx}", "/".join(str(x) for x in pn), f"packages z_N buys, over all seven conventions [rho = ${rho}M]", mac)
    macro(f"ConvCostAbove{sfx}", f"{sum(c['cost'] > c['gain'] for c in rows)}/{len(rows)}", f"conventions under which the E[NPV] cost exceeds the CE gain [rho = ${rho}M]", mac)

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
src = ", ".join(f"{PREFIX}_s{s}_r{rho}.json" for s in SEEDS for rho in RHOS) + f", {CONV_FILE}"
head = [f"% Generated by paper/make_v32_extra.py on {stamp}; do not edit by hand, rerun the script.",
        f"% Inputs (src/model/results/): {src}.",
        "% Read-only on the run files."]
with open(OUT_TAB, "w") as fh:
    fh.write("\n".join(head + ["% Tables need booktabs and graphicx (main.tex). They are collected with \\collectresultstables",
                               "% (main.tex) and placed by label in Section 7.", ""]) + format_table_currency("\n" + "\n".join(tab)))
with open(OUT_NUM, "w") as fh:
    fh.write("\n".join(head + ["% Unsuffixed macros refer to rho = $300M and ...RhoOneHundred to $100M, as in results_numbers.tex;",
                               "% ...All pools both rho. Numbers are bare; units and % signs belong to the prose.", ""] + mac) + "\n")
print("wrote", OUT_TAB, "and", OUT_NUM)
for rho in RHOS:
    t = RE[rho]
    print(rho, "B/T/L", t["b"], t["t"], t["l"], "E B/T/L", t["be"], t["te"], t["le"], "gain% range", round(min(t["gp"]), 2), round(max(t["gp"]), 2),
          "int% range", round(t["plo"], 2), round(t["phi"], 2), "rule", round(min(t["rule"]), 2), round(max(t["rule"]), 2), "notbehind", t["notbehind"], "shared", sorted(t["shared"]))
