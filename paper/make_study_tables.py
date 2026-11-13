"""Tables and macros for the scale, rank, node, master, tolerance and enumeration studies (src/model/exp_scale.py,
exp_enum.py) and for the audit of stored bounds.

Usage: make_study_tables.py [--scale ../src/model/results/scale/scale_study.json]
Writes sections/results_studies.tex (tables tab:scale, tab:rank, tab:nodes, tab:master) and
sections/results_numbers_study.tex (macros \\study...). Missing inputs or parts are skipped with a note.
"""
import argparse
import json
import os
import glob
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "model"))
from audit_certificates import cross_validate_nodes   # noqa: E402  (json only, no solver)
from report_currency import format_table_currency   # noqa: E402


def f(x, d=2):
    try:
        return "--" if x is None or not np.isfinite(float(x)) else f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "--"


DIGITS = dict(zip("0123456789", ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")))
WORDS = {"4": "Four", "8": "Eight", "9": "Nine", "16": "Sixteen", "25": "TwentyFive", "32": "ThirtyTwo", "40": "Forty", "80": "Eighty", "100": "Hundred",
         "160": "OneSixty", "300": "ThreeHundred", "1000": "Thousand"}


def W(x):
    """A number as a word for a LaTeX macro name (digits are not allowed): known values by name, others digit by digit."""
    t = str(int(round(float(x))))
    return WORDS.get(t, "".join(DIGITS[c] for c in t))


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def tab(caption, label, cols, head, rows, size=r"\footnotesize"):
    out = [r"\begin{table}[tbp]", r"\centering", size, r"\caption{" + caption + "}", r"\label{" + label + "}", r"\begin{tabular}{" + cols + "}", r"\toprule",
           head + r" \\", r"\midrule"] + [r + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def scale_tables(S, macros):
    tex = ""
    rows = []
    for r in S.get("scale", []):
        ce, rn = r.get("ce", {}), r.get("rn", {})
        if "error" in r or "error" in ce or "error" in rn:
            err = r.get("error") or ce.get("error") or rn.get("error") or ""
            rows.append(f"{r['U']} & {f(r['rho'], 0)} & {r.get('binaries', '--')} & {f(rn.get('time'), 0)} & " + r"\multicolumn{4}{l}{" + esc(err[:60]) + "}")
            continue
        rows.append(f"{r['U']} & {f(r['rho'], 0)} & {r['binaries']:,} & {f(rn['time'], 0)} & {f(ce['time'], 0)} & {ce['rounds']} & {f(ce['gap_pct'], 3)}")
        tag = f"U{W(r['U'])}R{W(r['rho'])}"
        macros += [(f"ScaleTime{tag}", f(ce["time"], 0)), (f"ScaleRounds{tag}", str(ce["rounds"])), (f"ScaleGap{tag}", f(ce["gap_pct"], 3)),
                   (f"ScaleBin{tag}", f"{r['binaries']:,}"), (f"ScaleRnTime{tag}", f(rn["time"], 0))]
    if rows:
        tex += tab(r"Scale study (base generator settings, seed 100; one TACC Vista node, 32 Gurobi threads, solver memory limit 60 GB): size of the master, the risk-neutral MILP's time, and the certified solve's time, rounds and final gap. The 160-unit instance was solved at $\rho=\$100$M only.",
                   "tab:scale", "rrrrrrr", r"units $U$ & $\rho$ (\$M) & binaries & RN MILP (s) & certified (s) & rounds & gap (\%)", rows)
    rows = []
    for r in S.get("rank", []):
        if "error" in r:
            rows.append(f"{r['rank']} & " + r"\multicolumn{5}{l}{" + esc(r["error"][:60]) + "}"); continue
        rows.append(f"{r['rank_used']} & {f(r['share_of_variance'], 3)} & {f(r['ce_lb'])} & {f(r['ce_ub'])} & {f(r['gap_pct'], 3)} & {r['rounds']} & {f(r['time'], 0)}")
        macros += [(f"RankGapR{W(r['rank_used'])}", f(r["gap_pct"], 3)), (f"RankTimeR{W(r['rank_used'])}", f(r["time"], 0)), (f"RankShareR{W(r['rank_used'])}", f(r["share_of_variance"], 3))]
    if rows:
        tex += tab(r"Rank study (40 units, $\rho=\$100$M): the certified solve with the Loewner-safe factor truncated to rank $r$. The bracket is valid at every rank; the share is the fraction of the field variance the factor keeps.",
                   "tab:rank", "rrrrrrr", r"rank $r$ & variance share & CE lower & CE upper & gap (\%) & rounds & time (s)", rows)
    rows = []
    FTall = S.get("nodes_fine_test", {})
    regrets = {}
    XV = cross_validate_nodes(S)                       # an upper end below a fixed-first-stage value on the same tree is refuted
    for r in S.get("nodes", []):
        ce = dict(r.get("ce", {})); seed = r.get("seed", 100)
        xv = XV.get((seed, r.get("n")))
        mark = ""
        if xv and xv["refuted"]:
            ce.update(ce_lb=xv["ce_lb"], ce_ub=xv["ce_ub"], gap_pct=xv["gap_pct"]); mark = r"$^{\dagger}$"
        if "error" in r or "error" in ce:
            rows.append(f"{seed} & {r['n']}$\\times${r['n']} & {r['K']} & " + r"\multicolumn{6}{l}{" + esc((r.get("error") or ce.get("error", ""))[:60]) + "}"); continue
        ft = FTall.get(str(seed), {}).get("rows", {}).get(str(r["n"]), {})
        reg = f"[{f(ft.get('regret_lb'), 3)}, {f(ft.get('regret_ub'), 3)}]" if "regret_lb" in ft else "--"
        rows.append(f"{seed} & {r['n']}$\\times${r['n']} & {r['K']} & {f(ce['ce_lb'])} & {f(ce['ce_ub'])}{mark} & {ce['rounds']} & {f(ce['time'], 0)} & "
                    f"{f(r.get('jaccard_vs_3x3'), 2)} & {reg}")
        sw = W(seed)
        macros += [(f"NodesTimeK{W(r['K'])}S{sw}", f(ce["time"], 0)), (f"NodesGapK{W(r['K'])}S{sw}", f(ce["gap_pct"], 3)), (f"NodesJacK{W(r['K'])}S{sw}", f(r.get("jaccard_vs_3x3"), 2)),
                   (f"NodesRegretLbK{W(r['K'])}S{sw}", f(ft.get("regret_lb"), 3)), (f"NodesRegretUbK{W(r['K'])}S{sw}", f(ft.get("regret_ub"), 3)), (f"NodesWellsYearZeroK{W(r['K'])}S{sw}", f(ce.get("wells_year0"), 0))]
        if "regret_ub" in ft:
            regrets.setdefault(r["K"], []).append(ft["regret_ub"])
    ref = [c for c in XV.values() if c["refuted"]]
    macros += [("NodesRefuted", str(len(ref))), ("NodesSolved", str(len(XV))), ("NodesRefutedGapMin", f(min([c["gap_pct"] for c in ref], default=0.0), 2)),
               ("NodesRefutedGapMax", f(max([c["gap_pct"] for c in ref], default=0.0), 2)), ("NodesRefutedExcessMax", f(max([c["excess"] for c in ref], default=0.0), 3)),
               ("NodesRefutedExcessMin", f(min([c["excess"] for c in ref], default=0.0), 3))]
    for K, v in regrets.items():
        macros += [(f"NodesRegretUbMaxK{W(K)}", f(max(v), 3)), (f"NodesJacMinK{W(K)}", f(min(r.get("jaccard_vs_3x3", np.nan) for r in S.get("nodes", []) if r.get("K") == K and "jaccard_vs_3x3" in r), 2))]
    if rows:
        tex += tab(r"Node study (40 units, $\rho=\$100$M, three base instances): trees of $n\times n$ Gauss--Hermite nodes; the bracket and time on each tree, the year-0 overlap of the first stage with the $3\times3$ plan (Jaccard over unit-benches), and the regret of each first stage when the later years are re-optimized on the $5\times5$ tree, as a certificate interval in \% of that tree's $E(z_N)$.",
                   "tab:nodes", "rlrrrrrrr",
                   r"seed & tree & $K$ & CE lower & CE upper & rounds & time (s) & Jaccard vs $3\times3$ & regret on $5\times5$ (\%)", rows, size=r"\scriptsize")
    rows = []
    for r in S.get("master", []):
        if "error" in r:
            rows.append(f"{r['master']} & " + r"\multicolumn{4}{l}{" + esc(r["error"][:60]) + "}"); continue
        rows.append(f"{r['master']} & {f(r['ce_lb'])} & {f(r['ce_ub'])} & {r['rounds']} & {f(r['time'], 0)}")
        macros += [(f"Master{r['master'].title()}Time", f(r["time"], 0)), (f"Master{r['master'].title()}Rounds", str(r["rounds"]))]
    for r in S.get("tol", []):
        for tol in ("0.001", "0.0001"):
            x = r.get(f"tol_{tol}", {})
            if x and "error" not in x:
                rows.append(f"seed {r['seed']}, tol {tol} & {f(x['ce_lb'], 3)} & {f(x['ce_ub'], 3)} & {x['rounds']} & {f(x['time'], 0)}")
                tt = "Three" if tol == "0.001" else "Four"
                macros += [(f"TolTimeS{W(r['seed'])}Tol{tt}", f(x["time"], 0)), (f"TolRoundsS{W(r['seed'])}Tol{tt}", str(x["rounds"]))]
    if rows:
        tex += tab(r"Master type and certificate tolerance (40 units, $\rho=\$100$M): the linear tangent-cut master against the second-order-cone master on seed 100, and the four joint-knob instances certified to $10^{-3}$ and $10^{-4}$ of CE.",
                   "tab:master", "lrrrr", r"case & CE lower & CE upper & rounds & time (s)", rows)
    return tex


def enum_macros(E, macros):
    macros += [("EnumFirstStages", f"{E['n_first_stages']:,}"), ("EnumNodeRecourses", f"{E['n_node_recourses']:,}"), ("EnumCe", f(E["enum_ce"], 4)),
               ("EnumSolveLb", f(E["solve_ce"]["ce_lb"], 4)), ("EnumSolveUb", f(E["solve_ce"]["ce_ub"], 4)), ("EnumSolveRounds", str(E["solve_ce"]["rounds"])),
               ("EnumEnpv", f(E["enum_enpv"], 4)), ("EnumMilp", f(E["milp"]["obj"], 4)), ("EnumTime", f(E["time_enum"], 0)), ("EnumRho", f(E["rho"], 0)),
               ("EnumCheck", "passes" if all(E["checks"].values()) else "FAILS")]


def studies_summary_table(S, macros, E=None):
    """One main-text table: scale (U, rho), rank and node study rows side by side, each with binaries, bracket width, rounds, time."""
    rows = []
    for r in S.get("scale", []):
        ce = r.get("ce", {})
        if "error" in r or "error" in ce:
            rows.append(f"scale & $U={r['U']}$, $\\rho=\\${f(r['rho'], 0)}$M & {r.get('binaries', '--')} & \\multicolumn{{3}}{{l}}{{{esc((r.get('error') or ce.get('error', ''))[:45])}}}")
        else:
            rows.append(f"scale & $U={r['U']}$, $\\rho=\\${f(r['rho'], 0)}$M & {r['binaries']:,} & {f(ce['gap_pct'], 3)} & {ce['rounds']} & {f(ce['time'], 0)}")
    for r in S.get("rank", []):
        if "error" not in r:
            rows.append(f"rank & $r={r['rank_used']}$ ({f(100 * r['share_of_variance'], 1)}\\% of variance) & -- & {f(r['gap_pct'], 3)} & {r['rounds']} & {f(r['time'], 0)}")
    XV = cross_validate_nodes(S)
    for r in S.get("nodes", []):
        ce = dict(r.get("ce", {}))
        if "error" not in r and "error" not in ce:
            xv = XV.get((r.get("seed", 100), r["n"])); mark = ""
            if xv and xv["refuted"]:
                ce["gap_pct"] = xv["gap_pct"]; mark = r"$^{\dagger}$"
            rows.append(f"nodes & seed {r.get('seed', 100)}, ${r['n']}\\times{r['n']}$, Jaccard vs $3\\times3$ {f(r.get('jaccard_vs_3x3'), 2)} & {r['binaries']:,} & {f(ce['gap_pct'], 3)}{mark} & {ce['rounds']} & {f(ce['time'], 0)}")
    for r in S.get("master", []):
        if "error" not in r:
            rows.append(f"master & {r['master']} & -- & {f(100 * (r['ce_ub'] - r['ce_lb']) / max(1e-9, abs(r['ce_lb'])), 3)} & {r['rounds']} & {f(r['time'], 0)}")
    if E and "solve_ce" in E:
        rows.append(f"enumeration & one unit, {E['n_first_stages']:,} first stages, {E['n_node_recourses']:,} node recourses, $\\rho=\\${E['rho']:.0f}$M & -- & " + ("matches" if all(E["checks"].values()) else "FAILS") + f" & {E['solve_ce']['rounds']} & {f(E['solve_ce']['time'], 0)}")
    if not rows:
        return ""
    caption = r"Certificates at scale, rank, node count and master type (base generator settings, seed 100 unless stated; $\rho=\$100$M unless stated; TACC Vista nodes with 32 Gurobi threads; solver memory limit 16 GB for the master, rank and enumeration rows and 60 GB for the node and scale rows): size of the master, the final certificate gap, outer-approximation rounds and wall-clock. The enumeration row is the independent check on a one-unit instance."
    if any(r"$^{\dagger}$" in row for row in rows):
        caption += r" $^{\dagger}$The solver's final bound was refuted by a later solve on the same model; the gap is that of the bracket with the refuted bound dropped (Section~\ref{sec:res-cert})."
    caption += " Detailed tables are in the SI."
    return tab(caption,
               "tab:studies", "llrrrr", r"study & case & binaries & gap (\%) & rounds & time (s)", rows, size=r"\scriptsize")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", default=os.path.join(HERE, "..", "src", "model", "results", "scale", "scale_study.json"))
    ap.add_argument("--enum", default=os.path.join(HERE, "..", "src", "model", "results", "scale", "enum_check.json"))
    args = ap.parse_args()
    macros, tex = [], "% generated by make_study_tables.py; do not edit\n"
    if os.path.exists(args.scale):
        S = json.load(open(args.scale))
        extra = args.scale.replace(".json", "_ns.json")                  # the node and scale parts re-run with a larger memory guard
        if os.path.exists(extra):
            S2 = json.load(open(extra))
            for part in ("nodes", "nodes_fine_test", "scale"):
                if S2.get(part):
                    S[part] = S2[part]
        E = json.load(open(args.enum)) if os.path.exists(args.enum) else None
        tex += studies_summary_table(S, macros, E) + scale_tables(S, macros)
    else:
        print("no scale study file")
    def audit_macros(path, stem):
        """Audit macros for one audit summary: \\study<stem>Brackets, ...Refuted, ...; returns the cross-solve refutations."""
        A = json.load(open(path)); st = A["stats"]
        cross = [b for b in A["refuted"] if b["kind"] == "cross-solve"]
        main = [b for b in cross if "scale/" not in b["file"]]
        macros.extend([(f"{stem}Brackets", f"{st['brackets']:,}"), (f"{stem}CrossChecked", f"{st['rows_with_a_cross_check']:,}"), (f"{stem}Comparisons", f"{st['cross_pairs']:,}"),
                       (f"{stem}Refuted", str(st["refuted"])), (f"{stem}RefutedWithin", str(st["refuted_within_solve"])), (f"{stem}RefutedRuns", str(len(main))),
                       (f"{stem}RefutedRunFiles", str(len({b["file"] for b in main}))), (f"{stem}Runs", str(st["runs"])), (f"{stem}LikeForLikeRuns", str(st["like_for_like_runs"])),
                       (f"{stem}RefutedMaxPct", f(st["max_rel_pct"], 3)), (f"{stem}RefutedMaxExcess", f(st["max_excess"], 3)),
                       (f"{stem}RefutedRunsMaxPct", f(max([b["rel_pct"] for b in main], default=0.0), 3)), (f"{stem}RefutedRunsMinPct", f(min([b["rel_pct"] for b in main], default=0.0), 3))])
        return cross

    res_dir = os.path.join(HERE, "..", "src", "model", "results")
    audit = os.path.join(res_dir, "audit_summary.json")           # the audit of the files of record (src/model/audit_certificates.py --json)
    audit_s1 = os.path.join(res_dir, "audit_summary_s1.json")     # the audit of the runs under the first solver settings (kept for the record)
    if os.path.exists(audit):
        audit_macros(audit, "Audit")
    else:
        print("no audit summary (run src/model/audit_certificates.py ... --json results/audit_summary.json)")
    if os.path.exists(audit_s1):
        cross = audit_macros(audit_s1, "AuditOne")
        rows = []
        for b in cross:
            where = b["where"].replace("methods:", "").replace("+exact_recourse", ", fixed first year").replace("rn_tilt2", "tilt 2").replace("rn_deck", "best deck")
            where = where.replace("rn,", "strip plan,").replace("unrestricted", "full solve")
            where = re.sub(r" rho (\d+)\.0 ", lambda mo: r", $\rho=\$" + mo.group(1) + "$M: ", where)
            where = re.sub(r"(\d)x(\d)", lambda mo: mo.group(1) + r"$\times$" + mo.group(2), where)
            wit = b["witness"].replace("+exact_recourse", ", fixed first year").replace("rn_tilt2", "tilt 2").replace("rn_deck", "best deck").replace("ce", r"$\mathbf z_A$")
            rows.append(f"{where} & {b['ub']:.3f} & {b['value']:.3f} & {wit} & {b['excess']:.3f} & {b['rel_pct']:.3f}")
        if rows:
            tex += tab(r"Upper ends refuted by the cross-solve audit in the runs under the first solver settings (\$M): the solver's bound for the named solve, and a feasible value of the same problem found by another solve. The excess is a lower bound on the error of the bound. These runs were replaced by the rerun under the second settings, in which the audit refutes nothing.",
                       "tab:refuted", r">{\raggedright\arraybackslash}p{5.2cm}rr>{\raggedright\arraybackslash}p{3.6cm}rr", r"solve & upper end & feasible value & found by & excess & \% of CE", rows, size=r"\scriptsize")
    wit = dict(checks=0, refuted=0, resolves=0, brackets=0)                      # in-solve bound checks recorded by solver revision s2
    for fn in glob.glob(os.path.join(res_dir, "replicates", "*.json")) + glob.glob(os.path.join(res_dir, "b40v33_*.json")):
        for r_ in json.load(open(fn)).get("runs", []):
            for row_ in r_.get("methods", {}).values():
                if isinstance(row_, dict) and "ce_lb" in row_:
                    wit["brackets"] += 1; wit["checks"] += row_.get("witness_checks", 0) or 0; wit["refuted"] += row_.get("witness_refuted", 0) or 0
                    wit["resolves"] += row_.get("numeric_resolves", 0) or 0
    macros += [("WitnessChecks", f"{wit['checks']:,}"), ("WitnessRefuted", str(wit["refuted"])), ("WitnessResolves", str(wit["resolves"])), ("WitnessBrackets", f"{wit['brackets']:,}")]
    if os.path.exists(args.enum):
        enum_macros(json.load(open(args.enum)), macros)
    else:
        print("no enumeration file")
    open(os.path.join(HERE, "sections", "results_studies.tex"), "w").write(format_table_currency(tex))
    with open(os.path.join(HERE, "sections", "results_numbers_study.tex"), "w") as fh:
        fh.write("% generated by make_study_tables.py; do not edit\n")
        for k, v in macros:
            k = "".join(DIGITS[c] if c.isdigit() else c for c in k)        # LaTeX macro names cannot contain digits
            fh.write(f"\\newcommand{{\\study{k}}}{{{v}}}\n")
    print(f"wrote results_studies.tex and {len(macros)} macros")
