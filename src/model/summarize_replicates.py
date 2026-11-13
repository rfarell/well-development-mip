"""Paired statistics of the replicate study (exp_replicates.py), the
one-at-a-time knob study, the price-market variant and the NDIC-calibrated case: markdown, LaTeX tables, macros and
figures for the paper.

Usage: summarize_replicates.py results/replicates/*.json [--stress 'results/b40v32_s*_r*.json'] [--paper ../../paper]
       [--boot 4000] [--md results/summary_replicates.md]

Per rho (base design): every quantity is a paired difference on one instance, in % of E[NPV](rn) of that instance
(suffix _pct) and in % of program value E(z_N) - E(0) (suffix _pv); over instances: the mean with a 95% percentile
bootstrap interval of the mean, median, min, max, wins/ties/losses with the exact sign test (ties: |d| <= TIE_REL x
E[NPV](rn)); for bracketed quantities also the number of instances whose certificate interval [LB_a - UB_b, UB_a - LB_b]
excludes zero ("resolved"). The registered band test: lower end of the bootstrap interval of the mean gain over the best
deck-family member against BAND_PCT. The sixteen v3.2 runs (--stress) give a min-max column.

Outputs (with --paper): sections/results_replicates.tex (tab:rep-main, tab:rep-truth-<rho>, tab:rep-oat,
tab:rep-pricemarket), sections/results_replicates_si.tex (per-instance tables), sections/results_numbers_rep.tex (\\rep...),
sections/results_ndic.tex + results_numbers_ndic.tex (the NDIC case), figures/replicates_main.pdf, figures/replicates_oat.pdf.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

from audit_certificates import refuted_in_run
from report_currency import format_table_currency

N_BOOT = 4000
REQUIRED_MODEL_VERSION = "v3.3"     # runs of another model version are skipped
TIE_REL = 1e-4
BAND_PCT = 5.0
FAMILY = ("rn_deck", "rn_tilt1", "rn_tilt2")
RHO_WORD = {25: "TwentyFive", 50: "Fifty", 100: "Hundred", 300: "ThreeHundred", 1000: "Thousand"}
KNOB_TEX = dict(own_frac="owned share", deadline_frac="fringe-lease share", parent_frac="parented share", rig_mult="rig capacity",
                range_mi="field range (mi)", core_mix="core weight")
KNOB_BASE = dict(own_frac=0.65, deadline_frac=0.25, parent_frac=0.35, rig_mult=2.5, range_mi=3.0, core_mix=0.5)
TRUTH_LOW, TRUTH_HIGH = "mu_bias_0.85", "mu_bias_1.15"
# macro stems (digits are not allowed in LaTeX macro names)
NAMES = dict(gain_tilt1="GainTiltOne", gain_tilt2="GainTilt", gain_deck_practice="GainDeckPractice", enpv0="EnpvZero", ce0="CeZero",
             lam_sigma0="LamSigmaZero", vopt_truth_low="VoptTruthLow", vopt_truth_high="VoptTruthHigh", ce_gain_truth_low="CeGainTruthLow",
             ce_gain_truth_high="CeGainTruthHigh", gain_family_truth_low="GainFamilyTruthLow", gain_family_truth_high="GainFamilyTruthHigh",
             wells0_rn="WellsYearZeroRn", wells0_ce="WellsYearZeroCe", wells0_heur="WellsYearZeroHeur")
# (label, key, certified?)  certified rows carry a bracket (key_lb, key_ub) and a resolved count
ROWS_MAIN = [
    ("risk-neutral plan over the NPV-tuned rule, at expected NPV", "vopt", False),
    (r"\quad the same two plans on the certainty equivalent", "vopt_ce", False),
    (r"\quad risk-neutral plan at the rule's own deck over the rule", "vopt_rule_deck", False),
    (r"\quad rule's committed first year with risk-neutral later years", "vopt_robust", True),
    (r"\quad with the type curve 15\% low (plans as planned)", "vopt_truth_low", False),
    (r"\quad with the type curve 15\% high", "vopt_truth_high", False),
    ("first-year gap of the CE-tuned rule", "fs_gap_ce", True),
    (r"\quad risk-averse plan over the CE-tuned rule's complete plan (added after the planned comparisons)", "plan_gap_ce", False),
    (r"risk-averse plan over the risk-neutral plan, $\mathrm{CE}(z_A)-\mathrm{CE}(z_N)$", "ce_gain", False),
    (r"\quad expected NPV given up (Beck's cost)", "enpv_cost", False),
    (r"\quad residual over the practice deck 0.9", "gain_deck_practice", False),
    (r"\quad residual over the tilted MIP, one iteration", "gain_tilt1", False),
    (r"\quad residual over the tilted MIP, two iterations", "gain_tilt2", False),
    (r"\quad residual over the best uniform deck (CE-tuned)", "gain_deck", False),
    (r"\quad residual over the best family member", "gain_family", False),
    (r"\quad first-year residual over the better of tilt 2 and the best deck", "gain_family_exact", True),
    (r"\quad gain over the risk-neutral plan, type curve 15\% low", "ce_gain_truth_low", False),
    (r"\quad gain over the risk-neutral plan, type curve 15\% high", "ce_gain_truth_high", False),
    ("RN-VSS: risk-neutral plan over the mean-value first year re-planned once", "rn_vss", True),
    ("value of flexibility: two-stage over committed", "flex", True),
    ("EVPI", "evpi", True),
]


# ---------------------------------------------------------------------------------------------------------------------
def load(files):
    runs = []
    for fn in files:
        try:
            d = json.load(open(fn))
        except Exception as e:
            print(f"skip {fn}: {e!r}", file=sys.stderr); continue
        for r in d.get("runs", []):
            r["_file"] = os.path.basename(fn)
            if r.get("model_version") != REQUIRED_MODEL_VERSION:
                print(f"skip {fn} seed {r.get('seed')} rho {r.get('rho')}: model_version {r.get('model_version')!r} != {REQUIRED_MODEL_VERSION}", file=sys.stderr)
                continue
            runs.append(r)
    return runs


def get(r, *path, default=np.nan):
    x = r
    for p in path:
        if isinstance(x, dict) and p in x:
            x = x[p]
        else:
            return default
    return x


def quantities(r):
    """Per-run quantities: raw ($M), _pct (% of E(z_N)), _pv (% of program value), brackets, counts, placement, regime."""
    m = r.get("methods", {})
    E = get(r, "methods", "rn", "enpv")
    if not np.isfinite(E) or E == 0:
        return None
    q = dict(seed=r["seed"], rho=float(r["rho"]), tag=r.get("tag", "base"), enpv_rn=E, status=r.get("status", ""), file=r.get("_file", ""))
    ce = lambda n: get(r, "methods", n, "ce")       # noqa: E731
    lb = lambda n: get(r, "methods", n, "ce_lb")    # noqa: E731
    ub = lambda n: get(r, "methods", n, "ce_ub")    # noqa: E731
    enpv = lambda n: get(r, "methods", n, "enpv")   # noqa: E731
    bv = r.get("base_value", {})
    q["enpv0"] = bv.get("enpv0", np.nan); q["ce0"] = bv.get("ce0", np.nan); q["lam_sigma0"] = bv.get("lam_sigma0", np.nan)
    q["program_value"] = E - q["enpv0"]
    q["program_value_share"] = 100.0 * q["program_value"] / abs(E)
    q["ce_program"] = ce("ce") - q["ce0"]                                   # CE the program adds
    q["premium_ratio"] = ((E - ce("rn")) - (q["enpv0"] - q["ce0"])) / q["program_value"] if np.isfinite(q["program_value"]) and q["program_value"] > 0 else np.nan
    q["strip_below_empty"] = bool(np.isfinite(q["ce0"]) and ce("rn") < q["ce0"])
    q["ce_rn_share"] = 100.0 * ce("rn") / abs(E)                           # the firm's certainty equivalent of z_N, % of E(z_N)
    q["rp_rn_share"] = 100.0 * (E - ce("rn")) / abs(E)                     # risk premiums of z_N and z_A, % of E(z_N)
    q["rp_ce_share"] = 100.0 * (enpv("ce") - ce("ce")) / abs(E)
    q["capital"] = get(r, "cap", "capital")
    # value of optimization
    q["vopt"] = enpv("rn") - enpv("heur_ev")
    q["vopt_ce"] = ce("rn") - ce("heur_ev")                                # the same two plans on the firm's certainty equivalent
    q["vopt_rule_deck"] = get(r, "vopt", "vopt_at_rule_deck")
    if not np.isfinite(q["vopt_rule_deck"]):
        d9 = {round(c["deck"], 3): c for c in get(r, "methods", "rn_deck", "deck_curve", default=[])}
        rdv = get(r, "methods", "heur_ev", "cfg", "deck", default=np.nan)
        rd = round(float(rdv), 3) if np.isfinite(rdv) else None
        if rd in d9:
            q["vopt_rule_deck"] = d9[rd]["enpv"] - enpv("heur_ev")
    if "heur_ev+rn_recourse" in m:
        q["vopt_robust"] = enpv("rn") - enpv("heur_ev+rn_recourse")
        q["vopt_robust_lb"] = enpv("rn") - get(r, "methods", "heur_ev+rn_recourse", "bound")
        q["vopt_robust_ub"] = get(r, "methods", "rn", "bound") - enpv("heur_ev+rn_recourse")
    else:
        q["vopt_robust"] = q["vopt_robust_lb"] = q["vopt_robust_ub"] = np.nan
    # refuted within its own solve (flagged, inverted) or by another solve of the same problem in this run (a feasible value
    # above the upper end): audit_certificates.refuted_in_run
    inv = [n for n in refuted_in_run(r) if n in m and "ce_lb" in m[n]]
    q["invalid_certificates"] = inv
    for n in inv:                                                           # a refuted bound certifies nothing: drop the bracket
        m[n] = dict(m[n], ce_ub=np.nan)
    q["fs_gap_ce"] = lb("ce") - lb("heur_ce+exact_recourse")
    q["fs_gap_ce_lb"] = lb("ce") - ub("heur_ce+exact_recourse")
    q["fs_gap_ce_ub"] = ub("ce") - lb("heur_ce+exact_recourse")
    q["plan_gap_ce"] = lb("ce") - ce("heur_ce")                            # the CE-tuned rule's complete plan, later years as the rule schedules them
    # risk modeling
    q["ce_gain"] = ce("ce") - ce("rn")
    q["enpv_cost"] = enpv("rn") - enpv("ce")
    q["cost_minus_gain"] = q["enpv_cost"] - q["ce_gain"]                   # Beck's cost less the CE gain
    fam = [n for n in FAMILY if n in m]
    best = max(fam, key=lambda n: ce(n)) if fam else None
    q["best_family"] = best
    q["gain_family"] = ce("ce") - ce(best) if best else np.nan
    q["gain_deck"] = ce("ce") - ce("rn_deck")
    q["gain_tilt1"] = ce("ce") - ce("rn_tilt1")
    q["gain_tilt2"] = ce("ce") - ce("rn_tilt2")
    q["enpv_cost_tilt1"] = enpv("rn") - enpv("rn_tilt1")                   # expected NPV given up by each tilted plan
    q["enpv_cost_tilt2"] = enpv("rn") - enpv("rn_tilt2")
    q["implied_price_tilt1"] = get(r, "methods", "rn_tilt1", "tilt", "implied_price")   # weights recomputed at the one-iteration plan
    curve = {round(c["deck"], 3): c for c in get(r, "methods", "rn_deck", "deck_curve", default=[])}
    q["gain_deck_practice"] = ce("ce") - curve[0.9]["ce"] if 0.9 in curve else np.nan
    famx = [n + "+exact_recourse" for n in ("rn_deck", "rn_tilt2") if n + "+exact_recourse" in m]
    if famx:
        # V_A - max_j V_j with V in [lb, ub]: the two maxima are taken separately, since the member with the largest
        # lower end need not have the largest upper end
        q["gain_family_exact"] = lb("ce") - max(lb(n) for n in famx)
        ubs = [ub(n) for n in famx]                                          # a member without a valid upper end leaves the lower end open
        q["gain_family_exact_lb"] = lb("ce") - max(ubs) if all(np.isfinite(ubs)) else np.nan
        q["gain_family_exact_ub"] = ub("ce") - max(lb(n) for n in famx)
    else:
        q["gain_family_exact"] = q["gain_family_exact_lb"] = q["gain_family_exact_ub"] = np.nan
    for key, src in (("rn_vss", "rn_vss"), ("flex", "value_of_recourse"), ("evpi", "evpi")):
        q[key] = get(r, "ladder", src); q[key + "_lb"] = get(r, "ladder", src + "_lb"); q[key + "_ub"] = get(r, "ladder", src + "_ub")
    # regime, certificates, plans
    q["lam_sigma_rn"] = get(r, "methods", "rn", "lam_sigma"); q["lam_sigma_ce"] = get(r, "methods", "ce", "lam_sigma")
    q["gap_pct"] = get(r, "methods", "ce", "gap_pct"); q["rounds"] = get(r, "methods", "ce", "rounds")
    q["t_ce"] = get(r, "methods", "ce", "solver_time"); q["t_rn"] = get(r, "methods", "rn", "time"); q["t_total"] = r.get("time_total", np.nan)
    q["deck"] = get(r, "methods", "rn_deck", "deck"); q["rule_deck"] = get(r, "methods", "heur_ev", "cfg", "deck")
    for n, tag in (("rn", "rn"), ("ce", "ce"), ("heur_ev", "heur")):
        q[f"wells_{tag}"] = get(r, "methods", n, "placement", "wells_expected")
        q[f"wells0_{tag}"] = get(r, "methods", n, "placement", "wells_year0")
        q[f"pkgs_{tag}"] = get(r, "methods", n, "placement", "packages")
        bl = get(r, "methods", n, "placement", "by_len", default={})
        tot = sum(float(v) for v in bl.values()) if bl else 0.0
        q[f"long_share_{tag}"] = 100.0 * (float(bl.get("3", 0)) + float(bl.get("4", 0))) / tot if tot > 0 else np.nan
        q[f"four_share_{tag}"] = 100.0 * float(bl.get("4", 0)) / tot if tot > 0 else np.nan
    q["jac_rn"] = get(r, "methods", "rn", "placement", "jaccard_vs_ce_year0")
    q["implied_price"] = get(r, "methods", "rn", "tilt", "implied_price"); q["implied_theta"] = get(r, "methods", "rn", "tilt", "implied_theta")
    q["implied_price_zA"] = get(r, "methods", "ce", "tilt", "implied_price"); q["implied_theta_zA"] = get(r, "methods", "ce", "tilt", "implied_theta")
    sd_rn, sd_ce = get(r, "methods", "rn", "sigma_decomp", default={}), get(r, "methods", "ce", "sigma_decomp", default={})
    if sd_rn and sd_ce:
        fac = lambda d: d["theta"] + d["price"] + d["interaction"]  # noqa: E731
        rem = sd_rn["total"] - sd_ce["total"]
        q["factor_share_rn"] = 100 * fac(sd_rn) / sd_rn["total"]; q["price_share_rn"] = 100 * sd_rn["price"] / sd_rn["total"]
        q["factor_share_removed"] = 100 * (fac(sd_rn) - fac(sd_ce)) / rem if rem > 0 else np.nan
        q["price_share_removed"] = 100 * (sd_rn["price"] - sd_ce["price"]) / rem if rem > 0 else np.nan
    else:
        q["factor_share_rn"] = q["price_share_rn"] = q["factor_share_removed"] = q["price_share_removed"] = np.nan
    q["rho_over_enpv"] = q["rho"] / abs(E); q["rho_over_program"] = q["rho"] / q["program_value"] if q["program_value"] > 0 else np.nan
    q["rho_over_capital"] = q["rho"] / q["capital"] if np.isfinite(q["capital"]) and q["capital"] > 0 else np.nan
    q["lognormal_shift_ce"] = get(r, "beck", "lognormal_shift_ce")
    q["errors"] = len(r.get("errors", []))
    # perturbed truths, as planned
    P = r.get("perturbed", {})
    q["truth_vopt_pct"], q["truth_gain_family_pct"], q["truth_gain_family_fixed_pct"], q["truth_ce_gain_pct"] = {}, {}, {}, {}
    for label, t in P.items():
        if not isinstance(t, dict) or "rn" not in t:
            continue
        Et = t["rn"]["enpv"]
        q["truth_vopt_pct"][label] = 100 * (t["rn"]["enpv"] - t["heur_ev"]["enpv"]) / abs(Et) if "heur_ev" in t else np.nan
        famt = [n for n in FAMILY if n in t]
        if "ce" in t and famt:
            q["truth_gain_family_pct"][label] = 100 * (t["ce"]["ce"] - max(t[n]["ce"] for n in famt)) / abs(Et)
            q["truth_gain_family_fixed_pct"][label] = 100 * (t["ce"]["ce"] - t[best]["ce"]) / abs(Et) if best in t else np.nan
            q["truth_ce_gain_pct"][label] = 100 * (t["ce"]["ce"] - t["rn"]["ce"]) / abs(Et)
    def truth_dollars(label, kind):
        t = P.get(label)
        if not isinstance(t, dict) or "rn" not in t:
            return np.nan
        if kind == "vopt":
            return t["rn"]["enpv"] - t["heur_ev"]["enpv"] if "heur_ev" in t else np.nan
        if kind == "ce_gain":
            return t["ce"]["ce"] - t["rn"]["ce"] if "ce" in t else np.nan
        return t["ce"]["ce"] - t[best]["ce"] if ("ce" in t and best in t) else np.nan
    for key, src, kind in (("vopt_truth_low", "truth_vopt_pct", "vopt"), ("vopt_truth_high", "truth_vopt_pct", "vopt"), ("ce_gain_truth_low", "truth_ce_gain_pct", "ce_gain"),
                           ("ce_gain_truth_high", "truth_ce_gain_pct", "ce_gain"), ("gain_family_truth_low", "truth_gain_family_fixed_pct", "fam"), ("gain_family_truth_high", "truth_gain_family_fixed_pct", "fam")):
        lab = TRUTH_LOW if key.endswith("low") else TRUTH_HIGH
        q[key + "_pct"] = q[src].get(lab, np.nan)                            # % of the perturbed E(z_N)
        q[key] = truth_dollars(lab, kind)                                   # $M under the perturbed truth
        q[key + "_pv"] = 100.0 * q[key] / q["program_value"] if np.isfinite(q[key]) and q["program_value"] > 0 else np.nan   # share of the base program value
    for k in list(q):
        if k.startswith(("vopt", "fs_gap", "plan_gap", "ce_gain", "enpv_cost", "cost_minus_gain", "gain_", "rn_vss", "flex", "evpi")) and isinstance(q[k], float) and not k.endswith(("_pct", "_pv")):
            if k + "_pct" not in q:
                q[k + "_pct"] = 100.0 * q[k] / abs(E)
            if k + "_pv" not in q:
                q[k + "_pv"] = 100.0 * q[k] / q["program_value"] if np.isfinite(q["program_value"]) and q["program_value"] > 0 else np.nan
    for key in ("vopt_robust", "fs_gap_ce", "gain_family_exact", "rn_vss", "flex", "evpi"):
        lo, hi = q.get(key + "_lb", np.nan), q.get(key + "_ub", np.nan)
        q[key + "_resolved"] = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
        q[key + "_cert_win"] = bool(np.isfinite(lo) and lo > 0); q[key + "_cert_loss"] = bool(np.isfinite(hi) and hi < 0)
    return q


# ---------------------------------------------------------------------------------------------------------------------
def boot_ci(x, n_boot=None, seed=0):
    n_boot = N_BOOT if n_boot is None else n_boot
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()), float(x.mean())) if len(x) else (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def sign_test(d, tol):
    d = np.asarray(d, float); tol = np.asarray(tol, float)
    w, l = int((d > tol).sum()), int((d < -tol).sum()); t = len(d) - w - l
    p = float(stats.binomtest(w, w + l, 0.5).pvalue) if w + l > 0 else 1.0
    return w, t, l, p


def stat(Q, key, pct=True):
    """Summary of a per-instance quantity: mean, CI, median, min, max, sign test, resolved counts, n.
    pct: True = % of E(z_N); 'pv' = % of program value; False = raw units."""
    k = key + ("_pv" if pct == "pv" else "_pct" if pct else "")
    vals = np.array([q.get(k, np.nan) for q in Q], float)
    ok = np.isfinite(vals)
    x = vals[ok]
    if not len(x):
        return None
    den = np.array([abs(q["enpv_rn"]) for q in Q], float)[ok]
    if pct is True:
        tol = np.full(len(x), TIE_REL * 100.0)
    elif pct == "pv":
        pv = np.array([q["program_value"] for q in Q], float)[ok]
        tol = TIE_REL * 100.0 * den / np.where(pv > 0, pv, np.nan)
        tol = np.where(np.isfinite(tol), tol, TIE_REL * 100.0)
    else:
        tol = TIE_REL * den
    lo, hi = boot_ci(x)
    w, t, l, p = sign_test(x, tol)
    out = dict(n=int(len(x)), mean=float(x.mean()), lo=lo, hi=hi, median=float(np.median(x)), min=float(x.min()), max=float(x.max()),
               wins=w, ties=t, losses=l, p=p, sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0)
    res = [q.get(key + "_resolved") for q in Q]
    if any(v is not None for v in res):
        out["resolved"] = int(sum(bool(v) for v in res)); out["cert_wins"] = int(sum(bool(q.get(key + "_cert_win")) for q in Q))
        out["cert_losses"] = int(sum(bool(q.get(key + "_cert_loss")) for q in Q))
    return out


def f(x, d=2):
    try:
        if x is None or not np.isfinite(float(x)):
            return "--"
        s = f"{float(x):.{d}f}"
        return s.lstrip("-") if float(s) == 0 else s                        # a value that rounds to zero carries no sign
    except (TypeError, ValueError):
        return "--"


def fsigned(x, d=2):
    """Signed change; a value that rounds to zero prints without a sign."""
    s = f"{float(x):+.{d}f}"
    return s[1:] if float(s) == 0 else s


def cell(s, d=2):
    return "--" if s is None else f"{f(s['mean'], d)} [{f(s['lo'], d)}, {f(s['hi'], d)}]"


def rng_cell(s, d=2):
    return "--" if s is None else f"{f(s['median'], d)} ({f(s['min'], d)}, {f(s['max'], d)})"


BY_CONSTRUCTION = {"vopt", "ce_gain", "gain_tilt2"}   # nonnegative by construction (z_N maximizes E[NPV] over a set holding the rule's plan; z_A is seeded with z_N and tilt 2)


def count_cell(s, certified, key=None, tex=False):
    if s is None:
        return ""
    if certified and "resolved" in s:
        return f"{s['resolved']}/{s['n']}"
    if key in BY_CONSTRUCTION:
        return "$\\geq 0$ by construction" if tex else ">= 0 by construction"
    return f"{s['wins']}/{s['ties']}/{s['losses']} ({ptxt(s['p'], tex)})"


def ptxt(p, tex=False):
    return ("$<$0.001" if tex else "<0.001") if p < 0.001 else f"{p:.3f}"


def macro_name(key):
    return NAMES.get(key, "".join(ch for ch in key.replace("_", " ").title() if ch.isalpha()))


# ---------------------------------------------------------------------------------------------------------------------
def md_block(rho, Q, stress):
    lines = [f"\n## rho = ${rho:,.0f}M: {len(Q)} instances (seeds {min(q['seed'] for q in Q)}-{max(q['seed'] for q in Q)})\n",
             "| quantity (% of E[NPV](z_N)) | mean [95% CI] | median (min, max) | resolved or w/t/l (p) | % of program value, mean [CI] | stress 16 runs min-max |", "|---|---|---|---|---|---|"]
    for lab, key, cert in ROWS_MAIN:
        s, sp, ss = stat(Q, key), stat(Q, key, pct="pv"), stat(stress, key) if stress else None
        lines.append(f"| {lab.replace(chr(92) + 'quad', '  ').replace(chr(92), '')} | {cell(s)} | {rng_cell(s)} | {count_cell(s, cert)} | {cell(sp)} | "
                     f"{f(ss['min']) + ' to ' + f(ss['max']) if ss else '--'} |")
    pv = stat(Q, "program_value_share", pct=False)
    if pv:
        lines.append(f"\nProgram value E(z_N) - E(0): mean {f(pv['mean'], 1)}% of E(z_N) (range {f(pv['min'], 1)}-{f(pv['max'], 1)}%); E(z_N) mean {f(stat(Q, 'enpv_rn', pct=False)['mean'], 1)} $M; "
                     f"capital {f(stat(Q, 'capital', pct=False)['mean'], 1)} $M/yr; rho/E {f(stat(Q, 'rho_over_enpv', pct=False)['mean'])}, rho/program {f(stat(Q, 'rho_over_program', pct=False)['mean'])}, rho/capital {f(stat(Q, 'rho_over_capital', pct=False)['mean'])}.")
    cg, cp = stat(Q, "ce_gain", pct=False), stat(Q, "ce_program", pct=False)
    if cg and cp and cp["mean"]:
        shares = [100 * q["ce_gain"] / q["ce_program"] for q in Q if np.isfinite(q.get("ce_program", np.nan)) and q["ce_program"] > 0]
        lines.append(f"CE scale: CE(z_A) - CE(0) mean {f(cp['mean'], 1)} $M; risk modeling's share of it {f(100 * cg['mean'] / cp['mean'], 1)}% (ratio of means; median {f(np.median(shares), 1) if shares else '--'}%); "
                     f"risk-neutral plan below the empty plan on {sum(q['strip_below_empty'] for q in Q)}/{len(Q)} instances; premium ratio mean {f(stat(Q, 'premium_ratio', pct=False)['mean'], 3)}.")
    s = stat(Q, "gain_family")
    if s:
        lines.append(f"\nRegistered band test: lower CI end of the mean gain over the best family member = {s['lo']:.2f}% vs {BAND_PCT}%: **{'CLEARS' if s['lo'] > BAND_PCT else 'fails'}**; "
                     f"largest single-instance critical band {s['max']:.2f}%.")
    g1, gf, cgs = stat(Q, "gain_tilt1"), stat(Q, "gain_family"), stat(Q, "ce_gain")
    if g1 and gf and cgs and cgs["mean"]:
        lines.append(f"Recovered share of the gain: tilt one iteration {100 * (1 - g1['mean'] / cgs['mean']):.0f}%, best family member {100 * (1 - gf['mean'] / cgs['mean']):.0f}%.")
    lam, gp, tc, rd = stat(Q, "lam_sigma_rn", pct=False), stat(Q, "gap_pct", pct=False), stat(Q, "t_ce", pct=False), stat(Q, "rounds", pct=False)
    lines.append(f"\nRegime: lambda*sigma_W(z_N) mean {f(lam['mean'])} (range {f(lam['min'])}-{f(lam['max'])}); lambda*sigma_W(0) mean {f(stat(Q, 'lam_sigma0', pct=False)['mean'])}; "
                 f"certificates: gap {f(gp['mean'], 3)}% (max {f(gp['max'], 3)}%), rounds median {f(rd['median'], 0)} (max {f(rd['max'], 0)}), solver time median {f(tc['median'], 0)} s (max {f(tc['max'], 0)} s).")
    lines.append(f"Plans: expected wells rule {f(stat(Q, 'wells_heur', pct=False)['mean'], 1)}, z_N {f(stat(Q, 'wells_rn', pct=False)['mean'], 1)}, z_A {f(stat(Q, 'wells_ce', pct=False)['mean'], 1)}; "
                 f"packages z_N {f(stat(Q, 'pkgs_rn', pct=False)['mean'], 2)}, z_A {f(stat(Q, 'pkgs_ce', pct=False)['mean'], 2)}; long-lateral share z_N {f(stat(Q, 'long_share_rn', pct=False)['mean'], 0)}%, z_A {f(stat(Q, 'long_share_ce', pct=False)['mean'], 0)}%; "
                 f"CE-tuned deck median {f(stat(Q, 'deck', pct=False)['median'], 3)}; implied deck of z_N's weights: price {f(stat(Q, 'implied_price', pct=False)['mean'], 3)}, type curve {f(stat(Q, 'implied_theta', pct=False)['mean'], 3)}; "
                 f"variance: common-factor share of z_N {f(stat(Q, 'factor_share_rn', pct=False)['mean'], 0)}%, of the variance removed {f(stat(Q, 'factor_share_removed', pct=False)['mean'], 0)}%, price share removed {f(stat(Q, 'price_share_removed', pct=False)['mean'], 0)}%.")
    truths = sorted({t for q in Q for t in q["truth_vopt_pct"]})
    if truths:
        lines += ["\n| truth | value of optimization, mean [CI] | min | gain over family (member fixed at truth), mean [CI] | gain over family (re-selected) | ce - rn, mean |", "|---|---|---|---|---|---|"]
        for t in truths:
            v = np.array([q["truth_vopt_pct"].get(t, np.nan) for q in Q]); g = np.array([q["truth_gain_family_fixed_pct"].get(t, np.nan) for q in Q])
            g2 = np.array([q["truth_gain_family_pct"].get(t, np.nan) for q in Q]); c = np.array([q["truth_ce_gain_pct"].get(t, np.nan) for q in Q])
            lo, hi = boot_ci(v); glo, ghi = boot_ci(g)
            lines.append(f"| {t} | {f(np.nanmean(v))} [{f(lo)}, {f(hi)}] | {f(np.nanmin(v))} | {f(np.nanmean(g))} [{f(glo)}, {f(ghi)}] | {f(np.nanmean(g2))} | {f(np.nanmean(c))} |")
    return "\n".join(lines)


def tex_main_table(by_rho, stress):
    rhos = sorted(by_rho)
    ncol = len(rhos) + (1 if stress else 0)
    head = "quantity & " + " & ".join(f"$\\rho=\\${r:,.0f}$M ($n={len(by_rho[r])}$)" for r in rhos) + (r" & stress cases, 16 runs" if stress else "") + r" \\"
    out = [r"\begin{table}[tbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
           r"\caption{Replicate study (base generator settings, seeds 100--129): paired differences in \% of $E(z_N)$ of the same instance, mean with the 95\% percentile-bootstrap interval of the mean over instances; after the semicolon, for bracketed rows the number of instances whose certificate interval $[\mathrm{LB}_a-\mathrm{UB}_b,\ \mathrm{UB}_a-\mathrm{LB}_b]$ excludes zero, and for the others wins/ties/losses with the exact sign test $p$-value. Bracketed rows are differences of the two brackets' lower ends (the best values found). The last column is the range over the sixteen joint-knob stress cases at $\rho=\$25$M to \$300M; RN-VSS does not depend on $\rho$, and the low end of its range comes from a run whose mean-value MIP returned a different optimal first year.}",
           r"\label{tab:rep-main}", r"\begin{tabular}{>{\raggedright\arraybackslash}p{3.4cm}" + "c" * ncol + "}", r"\toprule", head, r"\midrule"]
    for lab, key, cert in ROWS_MAIN:
        cells = []
        for r in rhos:
            s = stat(by_rho[r], key)
            cells.append("--" if s is None else r"\begin{tabular}[t]{@{}c@{}}" + f"{f(s['mean'])} [{f(s['lo'])}, {f(s['hi'])}]" + r"\\ " + count_cell(s, cert, key, tex=True) + r"\end{tabular}")
        if stress:
            ss = stat(stress, key)
            cells.append("--" if ss is None else f"{f(ss['min'])} to {f(ss['max'])}")
        out.append(lab + " & " + " & ".join(cells) + r" \\")
    # the shares of the program value and the descriptive rows go to a companion table (SI) so the main table fits one page
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
            r"\begin{table}[tbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
            r"\caption{Base study, companion to Table~\ref{tab:rep-main}: the main differences as shares of the program value $E(z_N)-E(\mathbf 0)$ (mean with the 95\% bootstrap interval; last column the range over the sixteen stress-case runs), and the scale, regime and certificate statistics of the instances.}",
            r"\label{tab:rep-main-extra}", r"\begin{tabular}{>{\raggedright\arraybackslash}p{3.4cm}" + "c" * ncol + "}", r"\toprule", head, r"\midrule"]
    for lab, key in (("risk-neutral plan over the NPV-tuned rule, \\% of program value", "vopt"), ("first-year gap of the CE-tuned rule, \\% of program value", "fs_gap_ce"),
                     (r"$\mathrm{CE}(z_A)-\mathrm{CE}(z_N)$, \% of program value", "ce_gain"), ("residual over the tilted MIP (one iteration), \\% of program value", "gain_tilt1"),
                     ("residual over the best family member, \\% of program value", "gain_family"), ("RN-VSS, \\% of program value", "rn_vss")):
        cells = []
        for r in rhos:
            s = stat(by_rho[r], key, pct="pv")
            cells.append("--" if s is None else f"{f(s['mean'])} [{f(s['lo'])}, {f(s['hi'])}]")
        if stress:
            ss = stat(stress, key, pct="pv"); cells.append("--" if ss is None else f"{f(ss['min'])} to {f(ss['max'])}")
        out.append(lab + " & " + " & ".join(cells) + r" \\")
    out.append(r"\midrule")
    for lab, key, d, how in (("$E(z_N)$, \\$M, mean (min--max)", "enpv_rn", 1, "range"), ("program value $E(z_N)-E(0)$, \\$M, mean (min--max)", "program_value", 1, "range"),
                             ("program value, \\% of $E(z_N)$, mean (min--max)", "program_value_share", 1, "range"), ("annual capital, \\$M", "capital", 1, "mean"),
                             ("first-year gap, \\$M, mean", "fs_gap_ce", 2, "mean"),
                             ("$\\lambda\\sigma_W(z_N)$, mean (min--max)", "lam_sigma_rn", 2, "range"), ("$\\lambda\\sigma_W(\\mathbf 0)$, mean", "lam_sigma0", 2, "mean"),
                             ("risk premium of $z_N$, $z_A$, \\% of $E(z_N)$", None, 1, "rp"),
                             ("$\\rho/E(z_N)$, $\\rho$/program value, $\\rho$/capital, means", None, 2, "rho"),
                             ("certificate gap, \\%, mean (max)", "gap_pct", 3, "max"), ("solver time, s, median (max)", "t_ce", 0, "medmax"), ("OA rounds, median (max)", "rounds", 0, "medmax")):
        cells = []
        for r in rhos:
            Q = by_rho[r]
            if how == "rho":
                a, b, c = stat(Q, "rho_over_enpv", pct=False), stat(Q, "rho_over_program", pct=False), stat(Q, "rho_over_capital", pct=False)
                cells.append(f"{f(a['mean'] if a else np.nan)}, {f(b['mean'] if b else np.nan)}, {f(c['mean'] if c else np.nan)}"); continue
            if how == "rp":
                a, b = stat(Q, "rp_rn_share", pct=False), stat(Q, "rp_ce_share", pct=False)
                cells.append(f"{f(a['mean'] if a else np.nan, d)}, {f(b['mean'] if b else np.nan, d)}"); continue
            s = stat(Q, key, pct=False)
            if s is None:
                cells.append("--")
            elif how == "range":
                cells.append(f"{f(s['mean'], d)} ({f(s['min'], d)}--{f(s['max'], d)})")
            elif how == "max":
                cells.append(f"{f(s['mean'], d)} ({f(s['max'], d)})")
            elif how == "medmax":
                cells.append(f"{f(s['median'], d)} ({f(s['max'], d)})")
            else:
                cells.append(f(s["mean"], d))
        if stress:
            cells.append("")
        out.append(lab + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def tex_truth_table(by_rho):
    rhos = sorted(by_rho)
    out = []
    for r in rhos:
        Q = by_rho[r]
        truths = sorted({t for q in Q for t in q["truth_vopt_pct"]})
        if not truths:
            continue
        w = RHO_WORD.get(int(r), str(int(r))).lower()
        out += [r"\begin{table}[tbp]", r"\centering", r"\scriptsize",
                r"\caption{Replicate study at $\rho=\$" + f"{r:,.0f}" + r"$M: the same plans re-scored under perturbed truths (recourse as planned), mean over " + f"{len(Q)}"
                + r" instances with the bootstrap interval, in \% of the perturbed $E(z_N)$. The family member is the one chosen at model truth; the re-selected column picks the best member under each truth (an in-sample choice).}",
                r"\label{tab:rep-truth-" + w + "}", r"\begin{tabular}{lcccc}", r"\toprule",
                r"truth & value of optimization & residual over the family member & residual, re-selected & $\mathrm{CE}(z_A)-\mathrm{CE}(z_N)$ \\", r"\midrule"]
        for t in truths:
            v = np.array([q["truth_vopt_pct"].get(t, np.nan) for q in Q]); g = np.array([q["truth_gain_family_fixed_pct"].get(t, np.nan) for q in Q])
            g2 = np.array([q["truth_gain_family_pct"].get(t, np.nan) for q in Q]); c = np.array([q["truth_ce_gain_pct"].get(t, np.nan) for q in Q])
            lo, hi = boot_ci(v); glo, ghi = boot_ci(g); clo, chi = boot_ci(c)
            out.append(t.replace("_", r"\_") + f" & {f(np.nanmean(v))} [{f(lo)}, {f(hi)}] & {f(np.nanmean(g))} [{f(glo)}, {f(ghi)}] & {f(np.nanmean(g2))} & {f(np.nanmean(c))} [{f(clo)}, {f(chi)}]" + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def tex_si_per_instance(by_rho):
    out = []
    keys = [("vopt", "value of opt."), ("fs_gap_ce", "first-year gap"), ("ce_gain", "CE gain"), ("gain_tilt1", "residual, tilt 1"),
            ("gain_family", "residual, family"), ("rn_vss", "RN-VSS"), ("flex", "flexibility"), ("evpi", "EVPI")]
    for rho in sorted(by_rho):
        Q = sorted(by_rho[rho], key=lambda q: q["seed"])
        w = RHO_WORD.get(int(rho), str(int(rho))).lower()
        out += [r"{\scriptsize\setlength{\tabcolsep}{2.8pt}", r"\begin{longtable}{rrrr" + "r" * len(keys) + "}",
                r"\caption{Replicate study at $\rho=\$" + f"{rho:,.0f}" + r"$M: per-instance values in \% of $E(z_N)$; $E(z_N)$ and $E(0)$ in \$M.}\label{tab:si-rep-" + w + r"} \\",
                r"\toprule seed & $E(z_N)$ & $E(0)$ & $\lambda\sigma_W$ & " + " & ".join(k[1] for k in keys) + r" \\ \midrule \endfirsthead",
                r"\toprule seed & $E(z_N)$ & $E(0)$ & $\lambda\sigma_W$ & " + " & ".join(k[1] for k in keys) + r" \\ \midrule \endhead", r"\bottomrule \endfoot"]
        for q in Q:
            out.append(f"{q['seed']} & {f(q['enpv_rn'], 1)} & {f(q['enpv0'], 1)} & {f(q['lam_sigma_rn'])} & " + " & ".join(f(q.get(k + "_pct", np.nan)) for k, _ in keys) + r" \\")
        out += [r"\end{longtable}", "}", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------------
def knob_level(knob, value):
    base = KNOB_BASE.get(knob)
    return "Low" if base is not None and value < base else "High"


def oat_rows(base_Q, oat_runs):
    base = {q["seed"]: q for q in base_Q}
    groups = {}
    for q in oat_runs:
        groups.setdefault(q["tag"], []).append(q)
    rows = []
    for tag in sorted(groups, key=lambda t: (t.split(":")[1].split("=")[0], float(t.split("=")[1]))):
        knob, val = tag.split(":")[1].split("=")
        Q = groups[tag]
        paired = [q for q in Q if q["seed"] in base]
        row = dict(knob=knob, value=float(val), level=knob_level(knob, float(val)), n=len(Q), n_paired=len(paired), stats={}, diff={})
        for key in ("vopt", "fs_gap_ce", "ce_gain", "gain_family", "gain_tilt1", "rn_vss", "flex", "evpi"):
            row["stats"][key] = stat(Q, key)
            d = np.array([q[key + "_pct"] - base[q["seed"]][key + "_pct"] for q in paired], float)
            d = d[np.isfinite(d)]
            if len(d):
                lo, hi = boot_ci(d)
                row["diff"][key] = dict(mean=float(d.mean()), lo=lo, hi=hi, n=int(len(d)))
        row["lam_sigma"] = stat(Q, "lam_sigma_rn", pct=False)
        rows.append(row)
    return rows


def tex_oat_table(base_Q, rows, rho):
    keys = [("vopt", "value of optimization"), ("fs_gap_ce", "first-year gap"), ("ce_gain", r"$\mathrm{CE}(z_A)-\mathrm{CE}(z_N)$"),
            ("gain_family", "residual over family"), ("rn_vss", "RN-VSS"), ("flex", "flexibility"), ("evpi", "EVPI")]
    out = [r"\begin{table}[tbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
           r"\caption{One-at-a-time knob study at $\rho=\$" + f"{rho:,.0f}" + r"$M (seeds 100--109; base row from the same seeds of the replicate study): mean of each quantity in \% of $E(z_N)$; below it, the mean paired change against the base run of the same seed and its 95\% bootstrap interval.}",
           r"\label{tab:rep-oat}", r"\begin{tabular}{>{\raggedright\arraybackslash}p{1.9cm}cc" + r">{\centering\arraybackslash}p{1.55cm}" * len(keys) + "}", r"\toprule",
           r"knob & level & $\lambda\sigma_W$ & " + " & ".join(k[1] for k in keys) + r" \\", r"\midrule"]
    bQ = [q for q in base_Q if 100 <= q["seed"] <= 109]
    ls = stat(bQ, "lam_sigma_rn", pct=False)
    out.append(r"base & -- & " + f(ls["mean"] if ls else np.nan) + " & " + " & ".join(f(stat(bQ, k)["mean"]) if stat(bQ, k) else "--" for k, _ in keys) + r" \\")
    out.append(r"\midrule")
    for row in rows:
        cells = []
        for k, _ in keys:
            s, d = row["stats"].get(k), row["diff"].get(k)
            cells.append("--" if s is None else f"{f(s['mean'])}" + (r"\newline " + f"{fsigned(d['mean'])}" + r"\newline " + f"({f(d['lo'])}, {f(d['hi'])})" if d else ""))
        out.append(KNOB_TEX.get(row["knob"], row["knob"]) + f" & {row['value']:g} & " + f(row["lam_sigma"]["mean"] if row["lam_sigma"] else np.nan) + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------------
PM_KEYS = (("vopt", "risk-neutral plan over the NPV-tuned rule, at expected NPV"), ("fs_gap_ce", "first-year gap of the CE-tuned rule"), ("ce_gain", r"$\mathrm{CE}(z_A)-\mathrm{CE}(z_N)$"),
           ("enpv_cost", r"expected NPV given up"), ("gain_family", "residual over best family member"), ("gain_tilt1", "residual over tilted MIP (one iteration)"),
           ("rn_vss", "RN-VSS"), ("implied_price", "implied price of $z_N$'s weights"), ("lam_sigma_rn", r"$\lambda\sigma_W(z_N)$"))
VARIANTS = {"pricemarket": ("Pm", r"price-market control: every hedge share set to 1, so the price enters the plan at the strip in every year and the producing base is fully hedged; it removes the price charge and the price recourse together"),
            "codevelop": ("Codev", r"co-development control: a unit's two benches are developed in the same year or one bench only, imposed on every method"),
            "net": ("Net", r"net-economics control: the mean revenue map scaled by 0.75 with the prior field covariance unchanged (the posterior covariance falls, because the noise of the parents' production scales with the mean), a lower-margin sensitivity and not a calibration of operating costs or production taxes")}


def pm_rows(base_Q, pm_Q):
    base = {(q["seed"], q["rho"]): q for q in base_Q}
    out = {}
    for key, _ in PM_KEYS:
        pct = key not in ("lam_sigma_rn", "implied_price")
        out[key] = dict(pm=stat(pm_Q, key, pct=pct), base=stat([base[(q["seed"], q["rho"])] for q in pm_Q if (q["seed"], q["rho"]) in base], key, pct=pct))
        k = key + ("_pct" if pct else "")
        d = np.array([q[k] - base[(q["seed"], q["rho"])][k] for q in pm_Q if (q["seed"], q["rho"]) in base], float)
        d = d[np.isfinite(d)]
        if len(d):
            lo, hi = boot_ci(d)
            out[key]["diff"] = dict(mean=float(d.mean()), lo=lo, hi=hi, n=int(len(d)))
    return out


def tex_pm_table(rows, rho, n, tag="pricemarket"):
    stem, desc = VARIANTS[tag]
    out = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize",
           r"\caption{" + desc[0].upper() + desc[1:] + r", at $\rho=\$" + f"{rho:,.0f}" + r"$M (seeds 100--109). Means over " + f"{n}" + r" instances in \% of $E(z_N)$ (the last two rows in their own units), the base model on the same seeds, and the paired change with its 95\% bootstrap interval.}",
           r"\label{tab:rep-" + tag + "}", r"\begin{tabular}{lccc}", r"\toprule", r"quantity & control & base model & change [95\% CI] \\", r"\midrule"]
    for key, lab in PM_KEYS:
        r = rows.get(key, {})
        pm, b, d = r.get("pm"), r.get("base"), r.get("diff")
        out.append(f"{lab} & {cell(pm)} & {cell(b)} & " + (f"{fsigned(d['mean'])} [{f(d['lo'])}, {f(d['hi'])}]" if d else "--") + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


NDIC_ROWS = [("vopt", "risk-neutral plan over the NPV-tuned rule, at expected NPV"), ("vopt_robust", r"\quad rule's committed first year with risk-neutral later years"), ("fs_gap_ce", "first-year gap of the CE-tuned rule"),
             ("ce_gain", r"$\mathrm{CE}(z_A)-\mathrm{CE}(z_N)$"), ("enpv_cost", r"\quad expected NPV given up"), ("gain_deck_practice", r"\quad residual over the practice deck 0.9"),
             ("gain_tilt1", r"\quad residual over the tilted MIP, one iteration"), ("gain_tilt2", r"\quad residual over the tilted MIP, two iterations"),
             ("gain_deck", r"\quad residual over the best uniform deck (CE-tuned)"), ("gain_family", r"\quad residual over the best family member"), ("rn_vss", "RN-VSS"), ("flex", "value of flexibility"), ("evpi", "EVPI")]


def tex_ndic_table(Q):
    Q = sorted(Q, key=lambda q: q["rho"])
    cols = "l" + "c" * len(Q)
    out = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize",
           r"\caption{The NDIC-calibrated case (T151--152N, R97--98W, McKenzie County; 40 units; the planning operator's own producing base): the replicate study's comparisons at three risk tolerances, in \% of $E(z_N)$ (bracketed rows: certificate interval), with the \$M scale below. One calibrated case, not pooled with the synthetic results.}",
           r"\label{tab:ndic}", r"\begin{tabular}{" + cols + "}", r"\toprule", "quantity & " + " & ".join(f"$\\rho=\\${q['rho']:,.0f}$M" for q in Q) + r" \\", r"\midrule"]
    for key, lab in NDIC_ROWS:
        cells = []
        for q in Q:
            if key in ("fs_gap_ce", "flex", "evpi", "rn_vss", "vopt_robust") and np.isfinite(q.get(key + "_lb", np.nan)):
                cells.append(f"[{f(100 * q[key + '_lb'] / abs(q['enpv_rn']))}, {f(100 * q[key + '_ub'] / abs(q['enpv_rn']))}]")
            else:
                cells.append(f(q.get(key + "_pct", np.nan)))
        out.append(lab + " & " + " & ".join(cells) + r" \\")
    out.append(r"\midrule")
    for lab, key, d in ((r"$\lambda\sigma_W(z_N)$", "lam_sigma_rn", 2), (r"$E(z_N)$, \$M", "enpv_rn", 1), (r"$E(\mathbf 0)$, \$M", "enpv0", 1), ("program value, \\$M", "program_value", 1),
                        (r"$\mathrm{CE}(z_A)-\mathrm{CE}(z_N)$, \$M", "ce_gain", 2), ("first-year gap, \\$M", "fs_gap_ce", 2),
                        ("expected wells, rule / $z_N$ / $z_A$", None, 1), ("packages, $z_N$ / $z_A$", "pk", 0), ("long-lateral share, $z_N$ / $z_A$ (\\%)", "ls", 0),
                        ("CE-tuned deck", "deck", 3), ("implied deck of $z_N$'s weights, price / type curve", "imp", 3), ("certificate gap, \\%", "gap_pct", 3), ("solver time, s", "t_ce", 0)):
        if key is None:
            cells = [f"{f(q['wells_heur'], 0)} / {f(q['wells_rn'], 0)} / {f(q['wells_ce'], 0)}" for q in Q]
        elif key == "pk":
            cells = [f"{f(q['pkgs_rn'], 0)} / {f(q['pkgs_ce'], 0)}" for q in Q]
        elif key == "ls":
            cells = [f"{f(q['long_share_rn'], 0)} / {f(q['long_share_ce'], 0)}" for q in Q]
        elif key == "imp":
            cells = [f"{f(q['implied_price'], 3)} / {f(q['implied_theta'], 3)}" for q in Q]
        else:
            cells = [f(q.get(key, np.nan), d) for q in Q]
        out.append(lab + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def ndic_macros(Q):
    L = ["% generated by summarize_replicates.py (NDIC-calibrated instance); do not edit"]
    for q in Q:
        w = RHO_WORD.get(int(q["rho"]), str(int(q["rho"])))
        for key, _ in NDIC_ROWS:
            L.append(f"\\newcommand{{\\ndic{macro_name(key)}{w}}}{{{f(q.get(key + '_pct', np.nan))}}}")
            L.append(f"\\newcommand{{\\ndic{macro_name(key)}Dollar{w}}}{{{f(q.get(key, np.nan))}}}")
            if key in ("fs_gap_ce", "flex", "evpi", "rn_vss", "vopt_robust") and np.isfinite(q.get(key + "_lb", np.nan)):
                L.append(f"\\newcommand{{\\ndic{macro_name(key)}Lo{w}}}{{{f(100 * q[key + '_lb'] / abs(q['enpv_rn']))}}}")
                L.append(f"\\newcommand{{\\ndic{macro_name(key)}Hi{w}}}{{{f(100 * q[key + '_ub'] / abs(q['enpv_rn']))}}}")
        L.append(f"\\newcommand{{\\ndicPlanGapCe{w}}}{{{f(q.get('plan_gap_ce_pct', np.nan))}}}")
        for key, d in (("lam_sigma_rn", 2), ("lam_sigma0", 2), ("enpv_rn", 1), ("enpv0", 1), ("ce0", 1), ("program_value", 1), ("program_value_share", 1), ("wells_heur", 0), ("wells_rn", 0), ("wells_ce", 0),
                       ("wells0_rn", 0), ("wells0_ce", 0), ("deck", 3), ("gap_pct", 3), ("t_ce", 0), ("rounds", 0), ("pkgs_rn", 0), ("pkgs_ce", 0), ("long_share_rn", 0), ("long_share_ce", 0),
                       ("implied_price", 3), ("implied_theta", 3), ("factor_share_rn", 0), ("factor_share_removed", 0), ("price_share_removed", 0), ("capital", 1), ("rho_over_capital", 2), ("rho_over_program", 2)):
            L.append(f"\\newcommand{{\\ndic{macro_name(key)}{w}}}{{{f(q.get(key, np.nan), d)}}}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
def write_macros(path, by_rho, oat, base_Q, pm_rows_out=None, stress=None):
    L = ["% generated by summarize_replicates.py; do not edit"]

    def put(name, val, d=2):
        L.append(f"\\newcommand{{\\rep{name}}}{{{f(val, d)}}}")

    for rho, Q in sorted(by_rho.items()):
        w = RHO_WORD.get(int(rho), str(int(rho)))
        put(f"N{w}", len(Q), 0)
        for lab, key, cert in ROWS_MAIN:
            s = stat(Q, key)
            if s is None:
                continue
            nm = macro_name(key)
            put(f"{nm}Mean{w}", s["mean"]); put(f"{nm}Lo{w}", s["lo"]); put(f"{nm}Hi{w}", s["hi"]); put(f"{nm}Med{w}", s["median"])
            put(f"{nm}Min{w}", s["min"]); put(f"{nm}Max{w}", s["max"]); put(f"{nm}Wins{w}", s["wins"], 0); put(f"{nm}Ties{w}", s["ties"], 0); put(f"{nm}Losses{w}", s["losses"], 0)
            L.append(f"\\newcommand{{\\rep{nm}P{w}}}{{{ptxt(s['p'], True)}}}")
            if "resolved" in s:
                put(f"{nm}Resolved{w}", s["resolved"], 0); put(f"{nm}CertWins{w}", s["cert_wins"], 0); put(f"{nm}CertLosses{w}", s["cert_losses"], 0)
            sp = stat(Q, key, pct="pv")
            if sp:
                put(f"{nm}PvMean{w}", sp["mean"]); put(f"{nm}PvLo{w}", sp["lo"]); put(f"{nm}PvHi{w}", sp["hi"]); put(f"{nm}PvMax{w}", sp["max"]); put(f"{nm}PvMin{w}", sp["min"])
            sr = stat(Q, key, pct=False)
            if sr:
                put(f"{nm}DollarMean{w}", sr["mean"]); put(f"{nm}DollarMax{w}", sr["max"])
            if key == "vopt_truth_low":
                put(f"VoptTruthLowNeg{w}", sum(1 for q in Q if np.isfinite(q.get("vopt_truth_low_pct", np.nan)) and q["vopt_truth_low_pct"] < 0), 0)
            if key == "ce_gain_truth_high":
                put(f"CeGainTruthHighNeg{w}", sum(1 for q in Q if np.isfinite(q.get("ce_gain_truth_high_pct", np.nan)) and q["ce_gain_truth_high_pct"] < 0), 0)
        cg = stat(Q, "ce_gain")
        if cg and cg["mean"]:
            for key, nm in (("gain_tilt1", "TiltOne"), ("gain_tilt2", "TiltTwo"), ("gain_family", "Family"), ("gain_deck_practice", "DeckPractice"), ("gain_deck", "DeckOracle")):
                g = stat(Q, key)
                if g:
                    put(f"{nm}Recovered{w}", 100 * (1 - g["mean"] / cg["mean"]), 0)
        cp, cgr = stat(Q, "ce_program", pct=False), stat(Q, "ce_gain", pct=False)
        if cgr and cp and cp["mean"]:
            shares = [100 * q["ce_gain"] / q["ce_program"] for q in Q if np.isfinite(q.get("ce_program", np.nan)) and q["ce_program"] > 0]
            put(f"CeProgramShare{w}", 100 * cgr["mean"] / cp["mean"], 0); put(f"CeProgramShareMed{w}", np.median(shares) if shares else np.nan, 0)
        put(f"StripBelowEmpty{w}", sum(q["strip_below_empty"] for q in Q), 0)
        for key, nm in (("enpv_cost_tilt1", "EnpvCostTiltOne"), ("enpv_cost_tilt2", "EnpvCostTiltTwo")):
            g = stat(Q, key)
            if g:
                put(f"{nm}Mean{w}", g["mean"])
        put(f"TiltTwoWorse{w}", sum(1 for q in Q if np.isfinite(q.get("gain_tilt1", np.nan)) and np.isfinite(q.get("gain_tilt2", np.nan))
                                    and q["gain_tilt2"] > q["gain_tilt1"] + 1e-9), 0)
        cmg = stat(Q, "cost_minus_gain")
        if cmg:
            put(f"CostAboveGainMean{w}", cmg["mean"]); put(f"CostAboveGainLo{w}", cmg["lo"]); put(f"CostAboveGainHi{w}", cmg["hi"])
            put(f"CostAboveGainCount{w}", sum(1 for q in Q if np.isfinite(q.get("cost_minus_gain", np.nan)) and q["cost_minus_gain"] > 0), 0)
            put(f"GainAboveCostCount{w}", sum(1 for q in Q if np.isfinite(q.get("cost_minus_gain", np.nan)) and q["cost_minus_gain"] < 0), 0)
        vce = [q.get("vopt_ce_pct", np.nan) for q in Q]
        put(f"VoptCeRuleAhead{w}", sum(1 for v in vce if np.isfinite(v) and v < 0), 0)
        for key, d in (("premium_ratio", 3), ("program_value_share", 1), ("enpv0", 1), ("ce0", 1), ("lam_sigma0", 2), ("program_value", 1), ("enpv_rn", 1), ("capital", 1), ("ce_program", 1),
                       ("lam_sigma_rn", 2), ("gap_pct", 3), ("t_ce", 0), ("rounds", 0), ("t_total", 0), ("wells_rn", 1), ("wells_ce", 1), ("wells_heur", 1), ("wells0_rn", 1), ("wells0_ce", 1),
                       ("pkgs_rn", 2), ("pkgs_ce", 2), ("long_share_rn", 0), ("long_share_ce", 0), ("four_share_rn", 1), ("four_share_ce", 1), ("four_share_heur", 1), ("long_share_heur", 0), ("deck", 3), ("rule_deck", 2), ("implied_price", 3), ("implied_theta", 3),
                       ("implied_price_zA", 3), ("implied_theta_zA", 3), ("factor_share_rn", 0), ("price_share_rn", 0), ("factor_share_removed", 0), ("price_share_removed", 0),
                       ("rho_over_enpv", 2), ("rho_over_program", 2), ("rho_over_capital", 2), ("jac_rn", 2),
                       ("ce_rn_share", 1), ("rp_rn_share", 1), ("rp_ce_share", 1), ("implied_price_tilt1", 3)):
            s = stat(Q, key, pct=False)
            if s is None:
                continue
            nm = macro_name(key)
            put(f"{nm}Mean{w}", s["mean"], d); put(f"{nm}Med{w}", s["median"], d); put(f"{nm}Min{w}", s["min"], d); put(f"{nm}Max{w}", s["max"], d)
        s = stat(Q, "gain_family")
        if s:
            L.append(f"\\newcommand{{\\repBandVerdict{w}}}{{{'clears' if s['lo'] > BAND_PCT else 'fails'}}}")
        truths = sorted({t for q in Q for t in q["truth_vopt_pct"]})
        if truths:
            means = {t: float(np.nanmean([q["truth_vopt_pct"].get(t, np.nan) for q in Q])) for t in truths}
            worst = min(means, key=means.get)
            L.append(f"\\newcommand{{\\repVoptWorstTruth{w}}}{{{worst.replace('_', chr(92) + '_')}}}")
            put(f"VoptWorstTruthMean{w}", means[worst])
            TRUTH_WORD = {"mu_bias_0.85": "TypeLow", "mu_bias_1.15": "TypeHigh", "h_slope_0.5": "SlopeLow", "h_slope_1.5": "SlopeHigh", "range_0.5": "RangeLow",
                          "range_2.0": "RangeHigh", "eur4_1.4": "EurFourLow", "eur4_1.7": "EurFourHigh", "sig_p_0.15": "SigPLow", "sig_p_0.35": "SigPHigh"}
            for t in truths:                                                # every stored truth, plans as planned, % of the perturbed E(z_N)
                if t not in TRUTH_WORD:
                    continue
                v = np.array([q["truth_vopt_pct"].get(t, np.nan) for q in Q], float); v = v[np.isfinite(v)]
                if len(v):
                    lo, hi = boot_ci(v)
                    put(f"VoptTruth{TRUTH_WORD[t]}Mean{w}", v.mean()); put(f"VoptTruth{TRUTH_WORD[t]}Lo{w}", lo); put(f"VoptTruth{TRUTH_WORD[t]}Hi{w}", hi)
            gm = {t: float(np.nanmean([q["truth_gain_family_fixed_pct"].get(t, np.nan) for q in Q])) for t in truths}
            put(f"GainFamilyTruthMax{w}", max(gm.values())); put(f"GainFamilyTruthMin{w}", min(gm.values()))
    if stress:
        for lab, key, cert in ROWS_MAIN:
            s = stat(stress, key)
            if s:
                nm = macro_name(key)
                put(f"Stress{nm}Min", s["min"]); put(f"Stress{nm}Max", s["max"])
        for rho in sorted({q["rho"] for q in stress}):
            w = RHO_WORD.get(int(rho), str(int(rho)))
            Qs = [q for q in stress if q["rho"] == rho]
            for lab, key, cert in ROWS_MAIN:
                s = stat(Qs, key)
                if s:
                    nm = macro_name(key)
                    put(f"Stress{nm}Min{w}", s["min"]); put(f"Stress{nm}Max{w}", s["max"])
    for tag, rows_v in (pm_rows_out or {}).items():
        stem = VARIANTS[tag][0]
        for key, _ in PM_KEYS:
            r = rows_v.get(key, {})
            if r.get("pm"):
                put(f"{stem}{macro_name(key)}Mean", r["pm"]["mean"]); put(f"{stem}{macro_name(key)}Lo", r["pm"]["lo"]); put(f"{stem}{macro_name(key)}Hi", r["pm"]["hi"])
            if r.get("base"):
                put(f"{stem}{macro_name(key)}BaseMean", r["base"]["mean"])
            if r.get("diff"):
                put(f"{stem}{macro_name(key)}Diff", r["diff"]["mean"]); put(f"{stem}{macro_name(key)}DiffLo", r["diff"]["lo"]); put(f"{stem}{macro_name(key)}DiffHi", r["diff"]["hi"])
        ns = [r["diff"]["n"] for r in rows_v.values() if r.get("diff")]
        if ns:
            put(f"{stem}N", max(ns), 0)
    if oat:
        put("OatN", sum(r["n"] for r in oat), 0)
        put("OatSettings", len(oat), 0)
        for r in oat:
            tag = macro_name(r["knob"]) + r["level"]
            for k in ("vopt", "ce_gain", "gain_family", "gain_tilt1", "rn_vss", "flex", "evpi", "fs_gap_ce"):
                s, d = r["stats"].get(k), r["diff"].get(k)
                if s:
                    put(f"Oat{macro_name(k)}Mean{tag}", s["mean"])
                if d:
                    put(f"Oat{macro_name(k)}Diff{tag}", d["mean"]); put(f"Oat{macro_name(k)}DiffLo{tag}", d["lo"]); put(f"Oat{macro_name(k)}DiffHi{tag}", d["hi"])
            if r["lam_sigma"]:
                put(f"OatLamSigmaMean{tag}", r["lam_sigma"]["mean"])
        for k in ("vopt", "ce_gain", "gain_family", "gain_tilt1", "rn_vss", "flex", "evpi", "fs_gap_ce"):
            vals = [r["stats"][k]["mean"] for r in oat if r["stats"].get(k)]
            if vals:
                put(f"Oat{macro_name(k)}MeanMin", min(vals)); put(f"Oat{macro_name(k)}MeanMax", max(vals))
            dif = [r["diff"][k] for r in oat if r["diff"].get(k)]
            if dif:
                put(f"Oat{macro_name(k)}DiffAbsMax", max(abs(x["mean"]) for x in dif))
                put(f"Oat{macro_name(k)}DiffExcludesZero", sum(1 for x in dif if x["lo"] > 0 or x["hi"] < 0), 0)
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return len(L) - 1


# ---------------------------------------------------------------------------------------------------------------------
def fig_main(by_rho, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rhos = sorted(by_rho)
    panels = [("vopt", "value of optimization\n(risk-neutral plan over rule)"), ("fs_gap_ce", "first-year gap\nof the CE-tuned rule"), ("ce_gain", "CE gain over\nthe risk-neutral plan"),
              ("gain_tilt1", "residual over the\ntilted MIP\n(one iteration)"), ("gain_family", "residual over best\nfamily member"), ("rn_vss", "RN-VSS"), ("flex", "value of flexibility"), ("evpi", "EVPI")]
    # Draw at manuscript width so labels retain their intended point size.
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 5.2), sharey=False)
    axes = axes.ravel()
    rng = np.random.default_rng(1)
    for ax, (key, title) in zip(axes, panels):
        for i, r in enumerate(rhos):
            x = np.array([q.get(key + "_pct", np.nan) for q in by_rho[r]], float); x = x[np.isfinite(x)]
            if not len(x):
                continue
            jit = rng.uniform(-0.18, 0.18, len(x))
            ax.scatter(i + jit, x, s=9, color="0.55", alpha=0.7, linewidths=0)
            lo, hi = boot_ci(x)
            ax.errorbar([i], [x.mean()], yerr=[[x.mean() - lo], [hi - x.mean()]], fmt="o", color="k", ms=4, capsize=3, lw=1.2)
        if key == "gain_family":
            ax.axhline(BAND_PCT, color="C3", lw=0.8, ls="--")
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_xticks(range(len(rhos))); ax.set_xticklabels([f"{r:,.0f}" for r in rhos], fontsize=7)
        ax.set_title(title, fontsize=8); ax.tick_params(axis="y", labelsize=7)
    fig.supylabel("% of E[NPV] of the risk-neutral plan", fontsize=8, x=0.01)
    fig.supxlabel("Risk tolerance (USD million)", fontsize=8, y=0.01)
    fig.tight_layout(rect=(0.025, 0.035, 1, 1), h_pad=2.0, w_pad=1.1)
    fig.savefig(path); plt.close(fig)


def fig_oat(base_Q, rows, path):
    """Forest plot: one row per knob level, the paired change against the base run of the same seed, per quantity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("vopt", "value of optimization"), ("fs_gap_ce", "first-year gap"), ("ce_gain", "CE gain over\nthe risk-neutral plan"), ("gain_family", "residual over family"), ("rn_vss", "RN-VSS")]
    labels = [f"{KNOB_TEX.get(r['knob'], r['knob'])} {r['value']:g}" for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 6.0), sharey=True)
    axes = axes.ravel()
    axes[-1].set_visible(False)
    for ax, (k, title) in zip(axes, keys):
        for j, r in enumerate(rows):
            d = r["diff"].get(k)
            if d is None:
                continue
            col = "C3" if (d["lo"] > 0 or d["hi"] < 0) else "0.35"
            ax.errorbar([d["mean"]], [j], xerr=[[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]], fmt="o", color=col, ms=3.5, capsize=2, lw=1)
        ax.axvline(0, color="0.5", lw=0.6)
        ax.set_title(title, fontsize=8); ax.tick_params(labelsize=7.5)
        ax.set_xlabel("paired change\n(% of $E(z_N)$)", fontsize=7.5)
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels(labels, fontsize=7.5); axes[0].invert_yaxis()
    fig.tight_layout(h_pad=2.0, w_pad=1.0); fig.savefig(path); plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--stress", nargs="*", default=None, help="globs of the stress-case run files (full harness b40v33_* and lean replicates/stress_*)")
    ap.add_argument("--paper", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    N_BOOT = args.boot
    files = sorted({fn for pat in args.files for fn in glob.glob(pat)})
    runs = load(files)
    incomplete = [r for r in runs if not (str(r.get("status", "")).startswith("done") and not r.get("errors"))]
    Q = [q for q in (quantities(r) for r in runs if (str(r.get("status", "")).startswith("done") and not r.get("errors"))) if q]
    stress = []
    if args.stress:
        sfiles = sorted({fn for pat in args.stress for fn in glob.glob(pat)})
        stress = [q for q in (quantities(r) for r in load(sfiles) if (str(r.get("status", "")).startswith("done") and not r.get("errors"))) if q]
    base = [q for q in Q if q["tag"] == "base"]
    oat = [q for q in Q if q["tag"].startswith("oat:")]
    variants = {tag: [q for q in Q if q["tag"] == tag] for tag in VARIANTS}
    variants = {tag: v for tag, v in variants.items() if v}
    pm = variants.get("pricemarket", [])
    ndic = [q for q in Q if q["tag"] == "ndic"]
    by_rho = {}
    for q in base:
        by_rho.setdefault(q["rho"], []).append(q)
    md = [f"# Replicate study ({len(runs)} runs in {len(files)} files; {len(base)} base, {len(oat)} one-at-a-time, {len(pm)} price-market, {len(ndic)} NDIC; {len(stress)} stress runs)",
          "Failed or incomplete (excluded): " + (", ".join(f"s{r['seed']} r{r['rho']:.0f} {r['status']}" for r in incomplete) or "none")
          + "; stage errors: " + (", ".join(f"s{q['seed']} r{q['rho']:.0f} ({q['errors']})" for q in Q if q["errors"]) or "none")
          + "; refuted or inverted certificates (bracket dropped): " + (", ".join(f"s{q['seed']} r{q['rho']:.0f} {q['invalid_certificates']}" for q in Q if q["invalid_certificates"]) or "none")]
    for rho in sorted(by_rho):
        md.append(md_block(rho, by_rho[rho], stress))
    rows = []
    if oat:
        rho_oat = oat[0]["rho"]
        rows = oat_rows(by_rho.get(rho_oat, []), oat)
        md.append(f"\n## One-at-a-time knobs at rho = ${rho_oat:,.0f}M\n")
        md.append("| knob | level | n | lambda*sigma | vopt | diff vs base [CI] | fs gap | diff | ce gain | diff | gain family | diff | RN-VSS | diff | flex | diff | EVPI | diff |")
        md.append("|---|---|---|---|" + "---|" * 14)
        for r in rows:
            cells = []
            for k in ("vopt", "fs_gap_ce", "ce_gain", "gain_family", "rn_vss", "flex", "evpi"):
                s, d = r["stats"].get(k), r["diff"].get(k)
                cells += [f(s["mean"]) if s else "--", f"{fsigned(d['mean'])} [{f(d['lo'])}, {f(d['hi'])}]" if d else "--"]
            md.append(f"| {r['knob']} | {r['value']:g} ({r['level']}) | {r['n']} | {f(r['lam_sigma']['mean']) if r['lam_sigma'] else '--'} | " + " | ".join(cells) + " |")
    pmr = {}
    for tag, V in variants.items():
        pmr[tag] = pm_rows(by_rho.get(V[0]["rho"], []), V)
        md.append(f"\n## Variant {tag} at rho = ${V[0]['rho']:,.0f}M ({len(V)} runs)\n")
        md.append("| quantity | variant | base | paired change [CI] |"); md.append("|---|---|---|---|")
        for key, lab in PM_KEYS:
            r = pmr[tag][key]; d = r.get("diff")
            md.append(f"| {key} | {cell(r['pm'])} | {cell(r['base'])} | " + (f"{fsigned(d['mean'])} [{f(d['lo'])}, {f(d['hi'])}]" if d else "--") + " |")
    if ndic:
        md.append("\n## NDIC-calibrated instance\n")
        for q in sorted(ndic, key=lambda q: q["rho"]):
            md.append(f"rho {q['rho']:.0f}: " + ", ".join(f"{k} {f(q.get(k + '_pct', np.nan))}%" for k, _ in NDIC_ROWS) + f"; lam*sigma {f(q['lam_sigma_rn'])}, E(z_N) {f(q['enpv_rn'], 1)}, E(0) {f(q['enpv0'], 1)}, "
                      f"wells rule/rn/ce {f(q['wells_heur'], 0)}/{f(q['wells_rn'], 0)}/{f(q['wells_ce'], 0)}, packages {f(q['pkgs_rn'], 0)}/{f(q['pkgs_ce'], 0)}, long share {f(q['long_share_rn'], 0)}/{f(q['long_share_ce'], 0)}")
    text = "\n".join(md) + "\n"
    print(text)
    if args.md:
        open(args.md, "w").write(text)
    if args.paper:
        sec = os.path.join(args.paper, "sections"); figd = os.path.join(args.paper, "figures")
        tex = tex_main_table(by_rho, stress) + tex_truth_table(by_rho)
        if rows:
            tex += tex_oat_table(by_rho.get(oat[0]["rho"], []), rows, oat[0]["rho"])
        for tag, V in variants.items():
            tex += tex_pm_table(pmr[tag], V[0]["rho"], len(V), tag)
        open(os.path.join(sec, "results_replicates.tex"), "w").write("% generated by summarize_replicates.py; do not edit\n" + format_table_currency(tex))
        open(os.path.join(sec, "results_replicates_si.tex"), "w").write("% generated by summarize_replicates.py (SI per-instance tables); do not edit\n" + format_table_currency(tex_si_per_instance(by_rho)))
        if ndic:
            open(os.path.join(sec, "results_ndic.tex"), "w").write("% generated by summarize_replicates.py; do not edit\n" + format_table_currency(tex_ndic_table(ndic)))
            open(os.path.join(sec, "results_numbers_ndic.tex"), "w").write(ndic_macros(ndic))
        n = write_macros(os.path.join(sec, "results_numbers_rep.tex"), by_rho, rows, base, pmr, stress)
        fig_main(by_rho, os.path.join(figd, "replicates_main.pdf"))
        if rows:
            fig_oat(by_rho.get(oat[0]["rho"], []), rows, os.path.join(figd, "replicates_oat.pdf"))
        print(f"wrote results_replicates.tex, results_replicates_si.tex, results_numbers_rep.tex ({n} macros), figures", file=sys.stderr)
