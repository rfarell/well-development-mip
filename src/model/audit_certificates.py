"""Audit every stored certificate. A bracket's upper end is refuted when

  1. within its own solve it lies below its lower end, below a later exact incumbent (round bounds in hist), or the solver
     flagged it (bound_valid False);
  2. across solves of the same problem it lies below a feasible value found by another solve. In a run file every row of a
     block is scored on one model, so (a) the upper end of an unrestricted two-stage solve must cover every feasible value of
     the block, and (b) the upper end of a fixed-first-stage solve must cover the value of every stored plan with the same
     first stage. Blocks: the method rows, the fine-tree rows, and each truth of the re-optimized perturbed-truth table;
  3. in a study file (scale_study*.json) the unrestricted solve on a tree lies below a fixed-first-stage value on that tree,
     or two solves of one problem (master types, ranks) have disjoint brackets.

Kinds 2 and 3 were added on 2026-10-05: kind 1 cannot see a bound that agrees with its own solve's incumbent while a better
plan exists, which is how a solver's wrong dual bound shows up (Section S3.4 of the Supplementary Materials).

Usage: audit_certificates.py file.json ... [--json summary.json]     (exit code 1 when any refuted bound is found)
Importable: refuted_in_run(run) for the reporting scripts, cross_validate_nodes(study) for the study tables. No solver.
"""
import json
import math
import sys

UNRESTRICTED = ("ce", "ce_cold", "ce_1round")          # rows whose upper end bounds every two-stage plan of the model


def _tol(x):
    return 1e-6 * max(1.0, abs(x))


def _num(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _value(row):
    """The exact value of the row's plan (a feasible value of its problem), or None."""
    if not isinstance(row, dict) or row.get("feasible") is False:
        return None
    v = row.get("ce_lb", row.get("ce"))
    return v if _num(v) else None


def _first_stage(plans, name):
    for cand in (name, name.split("+")[0]):
        pl = plans.get(cand)
        if isinstance(pl, dict) and "z0" in pl:
            return json.dumps([pl.get("z0"), pl.get("y"), pl.get("w0")])
    return None


def _blocks(run):
    out = [("methods", run.get("methods", {}))]
    fine = run.get("fine", {}).get("methods")
    if isinstance(fine, dict):
        out.append(("fine", fine))
    for truth, rows in run.get("perturbed_reopt", {}).get("table", {}).items():
        if isinstance(rows, dict):
            out.append(("reopt:" + truth, rows))
    return out


def within_solve(row):
    """Kind 1. Returns a reason string or None."""
    if not isinstance(row, dict) or "ce_lb" not in row or "ce_ub" not in row:
        return None
    lb, ub = row["ce_lb"], row["ce_ub"]
    if row.get("bound_valid", True) is False:
        return "flagged by the solver"
    if _num(lb) and _num(ub) and ub < lb - _tol(lb):
        return "inverted"
    for h in row.get("hist", []):
        if _num(lb) and h.get("round_bound", float("inf")) < lb - _tol(lb):
            return "round bound below the final incumbent"
    return None


def cross_solve(run, stats=None):
    """Kind 2. Returns a list of dicts (block, row, ub, witness, value, excess) for one run of a run file."""
    plans, out = run.get("plans", {}), []
    for bname, rows in _blocks(run):
        vals = {n: _value(r) for n, r in rows.items()}
        vals = {n: v for n, v in vals.items() if v is not None}
        sig = {n: _first_stage(plans, n) for n in rows}
        for name, row in rows.items():
            if not isinstance(row, dict) or not _num(row.get("ce_ub")) or "committed" in name:
                continue
            ub = row["ce_ub"]
            if bname == "methods" and name in UNRESTRICTED:
                wit = {n: v for n, v in vals.items() if n != name}
            elif sig.get(name) is not None and (bname != "methods" or name.endswith("+exact_recourse")):
                wit = {n: v for n, v in vals.items() if n != name and sig.get(n) == sig[name]}
            else:
                continue
            if stats is not None and wit:
                stats["rows_with_a_cross_check"] = stats.get("rows_with_a_cross_check", 0) + 1
                stats["cross_pairs"] = stats.get("cross_pairs", 0) + len(wit)
            if not wit:
                continue
            w = max(wit, key=wit.get)
            if wit[w] > ub + _tol(ub):
                out.append(dict(block=bname, row=name, ub=ub, witness=w, value=wit[w], excess=wit[w] - ub, rel_pct=100 * (wit[w] - ub) / abs(ub)))
    return out


def refuted_in_run(run):
    """Names of the method rows of one run whose upper end is refuted by any check (kinds 1 and 2), with the reason."""
    bad = {}
    for name, row in run.get("methods", {}).items():
        why = within_solve(row)
        if why:
            bad[name] = why
    for c in cross_solve(run):
        if c["block"] == "methods":
            bad.setdefault(c["row"], f"below the value of {c['witness']} by {c['excess']:.4f}")
    return bad


def cross_validate_nodes(study):
    """Kind 3 for the node study. Returns {(seed, n): dict(ce_lb, ce_ub, gap_pct, refuted, witness, excess, dropped)} with the
    refuted round bounds dropped and the best feasible value known on that tree as the lower end."""
    fine = study.get("nodes_fine_test", {})
    out = {}
    for r in study.get("nodes", []):
        ce = r.get("ce", {})
        if not _num(ce.get("ce_lb")) or not _num(ce.get("ce_ub")):
            continue
        seed, n = r.get("seed", 100), r["n"]
        ft = fine.get(str(seed), {})
        wit = [row["ce_lb"] for row in ft.get("rows", {}).values() if _num(row.get("ce_lb"))] if ft.get("fine") == n else []
        lb = max([ce["ce_lb"]] + wit)
        rounds = [h["round_bound"] for h in ce.get("hist", []) if _num(h.get("round_bound"))] or [ce["ce_ub"]]
        kept = [b for b in rounds if b >= lb - _tol(lb)]
        refuted = ce["ce_ub"] < lb - _tol(lb)
        ub = min(kept) if kept else float("inf")
        out[(seed, n)] = dict(ce_lb=lb, ce_ub=ub, gap_pct=100 * (ub - lb) / max(1e-9, abs(lb)), refuted=refuted,
                              reported_ub=ce["ce_ub"], witness=max(wit) if wit else None, excess=(lb - ce["ce_ub"]) if refuted else 0.0,
                              dropped=[b for b in rounds if b < lb - _tol(lb)])
    return out


def audit_study(fn, d, bad, stats):
    for (seed, n), c in cross_validate_nodes(d).items():
        stats["brackets"] += 1
        if c["witness"] is not None:
            stats["rows_with_a_cross_check"] += 1; stats["cross_pairs"] += 1
        if c["refuted"]:
            bad.append(dict(file=fn, where=f"node study seed {seed} {n}x{n} unrestricted", ub=c["reported_ub"], witness="fixed first stage on the same tree",
                            value=c["witness"], excess=c["excess"], rel_pct=100 * c["excess"] / abs(c["reported_ub"]), kind="cross-solve"))
    for seed, t in d.get("nodes_fine_test", {}).items():
        for n, row in t.get("rows", {}).items():
            stats["brackets"] += 1
            why = within_solve(row)
            if why:
                bad.append(dict(file=fn, where=f"node study seed {seed} first stage of {n}x{n} on the fine tree", ub=row.get("ce_ub"), kind=why))
    for part in ("master", "rank"):
        rows = [r for r in d.get(part, []) if isinstance(r, dict) and _num(r.get("ce_lb")) and _num(r.get("ce_ub"))]
        stats["brackets"] += len(rows)
        for a in rows:
            for b in rows:
                if a is not b:
                    stats["cross_pairs"] += 1
                    if a["ce_lb"] > b["ce_ub"] + _tol(b["ce_ub"]):
                        bad.append(dict(file=fn, where=f"{part} study row {b.get(part)}", ub=b["ce_ub"], witness=f"{part} row {a.get(part)}", value=a["ce_lb"],
                                        excess=a["ce_lb"] - b["ce_ub"], rel_pct=100 * (a["ce_lb"] - b["ce_ub"]) / abs(b["ce_ub"]), kind="cross-solve"))
    for r in d.get("scale", []):
        ce = r.get("ce", {})
        if "ce_lb" in ce:
            stats["brackets"] += 1
            why = within_solve(ce)
            if why:
                bad.append(dict(file=fn, where=f"scale study U={r.get('U')} rho={r.get('rho')}", ub=ce.get("ce_ub"), kind=why))


def audit(files):
    bad = []
    stats = dict(files=0, runs=0, brackets=0, rows_with_a_cross_check=0, cross_pairs=0, like_for_like_runs=0)
    for fn in files:
        d = json.load(open(fn)); stats["files"] += 1
        short = fn.split("/results/")[-1]
        for r in d.get("runs", []):
            stats["runs"] += 1
            tag = f"seed {r.get('seed')} rho {r.get('rho')}"
            for bname, rows in _blocks(r):
                for name, row in rows.items():
                    if isinstance(row, dict) and "ce_lb" in row and "ce_ub" in row:
                        stats["brackets"] += 1
                        why = within_solve(row)
                        if why:
                            bad.append(dict(file=short, where=f"{tag} {bname}:{name}", ub=row.get("ce_ub"), kind=why))
            L = r.get("ladder", {})
            for k in ("committed", "two_stage", "pi"):
                if _num(L.get(k + "_lb")) and _num(L.get(k + "_ub")):
                    stats["brackets"] += 1
                    if L[k + "_ub"] < L[k + "_lb"] - _tol(L[k + "_lb"]):
                        bad.append(dict(file=short, where=f"{tag} ladder:{k}", ub=L[k + "_ub"], kind="inverted"))
            if L.get("node_bound_valid") is not None and not all(L["node_bound_valid"]):
                bad.append(dict(file=short, where=f"{tag} perfect-information nodes", ub=None, kind="flagged by the solver"))
            cs = cross_solve(r, stats)
            plans = r.get("plans", {})
            sig_a = _first_stage(plans, "ce")
            if sig_a and any(n.endswith("+exact_recourse") and _first_stage(plans, n) == sig_a for n in r.get("methods", {})):
                stats["like_for_like_runs"] += 1
            for c in cs:
                bad.append(dict(file=short, where=f"{tag} {c['block']}:{c['row']}", ub=c["ub"], witness=c["witness"], value=c["value"], excess=c["excess"],
                                rel_pct=c["rel_pct"], kind="cross-solve"))
        if any(k in d for k in ("nodes", "master", "rank", "scale")):
            audit_study(short, d, bad, stats)
        if "solve_ce" in d and "enum_ce" in d:                              # the enumeration check
            stats["brackets"] += 1
            s = d["solve_ce"]
            if not (s["ce_lb"] - _tol(s["ce_lb"]) <= d["enum_ce"] <= s["ce_ub"] + _tol(s["ce_ub"])):
                bad.append(dict(file=short, where="enumeration", ub=s["ce_ub"], witness="enumerated optimum", value=d["enum_ce"],
                                excess=d["enum_ce"] - s["ce_ub"], rel_pct=100 * (d["enum_ce"] - s["ce_ub"]) / abs(s["ce_ub"]), kind="cross-solve"))
    cross = [b for b in bad if b["kind"] == "cross-solve"]
    stats.update(refuted=len(bad), refuted_cross_solve=len(cross), refuted_within_solve=len(bad) - len(cross),
                 refuted_files=len({b["file"] for b in bad}), max_rel_pct=max([b["rel_pct"] for b in cross], default=0.0),
                 max_excess=max([b["excess"] for b in cross], default=0.0))
    return dict(stats=stats, refuted=bad)


if __name__ == "__main__":
    args = sys.argv[1:]
    out = None
    if "--json" in args:
        i = args.index("--json"); out = args[i + 1]; args = args[:i] + args[i + 2:]
    res = audit(args)
    s = res["stats"]
    print(f"files {s['files']}, runs {s['runs']}, brackets checked {s['brackets']}; brackets with a cross-solve check {s['rows_with_a_cross_check']} "
          f"({s['cross_pairs']} comparisons); refuted {s['refuted']} ({s['refuted_within_solve']} within a solve, {s['refuted_cross_solve']} across solves)")
    for b in res["refuted"]:
        if b["kind"] == "cross-solve":
            print(f"   {b['file']}: {b['where']}: upper bound {b['ub']:.6f} < {b['value']:.6f} ({b['witness']}), by {b['excess']:.6f} ({b['rel_pct']:.4f}%)")
        else:
            print(f"   {b['file']}: {b['where']}: {b['kind']}")
    if out:
        json.dump(res, open(out, "w"), indent=1)
    sys.exit(1 if res["refuted"] else 0)
