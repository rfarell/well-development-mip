"""Score the metaheuristic runs under certified recourse (needs Gurobi).

Usage:  GRB_MEMLIMIT=1.2 OMP_NUM_THREADS=2 python score_meta.py result_dir
            [--pkl results/meta/instances.pkl] [--out results/meta/scored_<name>.json] [--instances 0,1] [--rhos 100]
            [--max-solves N]

For every run JSON written by run_meta.py in result_dir: rebuild the instance (from the pickle) and Model(I, 1/rho),
check the option table's SHA-1 against the run's, and compute the certified CE of the plan's first stage (z0, y, w0)
with solve_recourse_exact at bench_v3.OA (the recourse re-optimized to a certificate; the run's own plan is the
seed). The certificate is computed once per distinct first stage per (instance, rho), cached by the bytes of
z0, y and w0, and so is the certificate of each heuristic seed's first stage (heuristic_plan(M, **fill)).

Output <out> (default results/meta/scored_<basename of result_dir>.json), rewritten atomically after every solve so
the scoring is resumable: dict(rows, refs, cert, oa) with
  rows  one per run: file, instance, rho, method, mult, seed, with_seed, long, objective, evals, max_evals, run wall
        and stop reason, in-sample ce / enpv (the shared recourse rule), the heuristic seed's in-sample ce, whether the
        run returned the seed's plan, the certified ce_lb / ce_ub of its first stage and of the seed's first stage,
        the solve time, and the gap to z_A: gap_lo = zA_ce_lb - ce_ub and gap_hi = zA_ce_ub - ce_lb (an interval for
        CE(z_A) - CE(first stage), both under certified recourse), also as a percent of E[NPV](rn); the gap is left
        out (ref_error) when the reference JSON's option table differs from the rebuilt Model's;
  refs  per instance and rho: z_A = methods.ce (ce_lb, ce_ub) of results/b40v32_s<seed>_r<rho>.json for seeds 0-3 or of
        results/replicates/rep_s<seed>_r<rho>.json for seeds 100-103, with E[NPV](rn) and the rule's certified
        first stage (heur_ce+exact_recourse) when present; None when the file is missing;
  cert  the certificate cache, key "<instance>|<rho>|<sha1 of the first stage>".
"""
import os

os.environ.setdefault("GRB_MEMLIMIT", "1.2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse                                                          # noqa: E402
import glob                                                              # noqa: E402
import hashlib                                                           # noqa: E402
import json                                                              # noqa: E402
import pickle                                                            # noqa: E402
import sys                                                               # noqa: E402
import time                                                              # noqa: E402


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Model                                                  # noqa: E402
import baselines as BL                                                   # noqa: E402
import bench_v3 as B3                                                    # noqa: E402  (imports solve, gurobipy)
from solve import solve_recourse_exact                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def fs_key(plan):
    return hashlib.sha1(plan["z0"].tobytes() + plan["y"].tobytes() + plan["w0"].tobytes()).hexdigest()


def reference(inst_seed, rho):
    """z_A and context for (instance, rho) from the v3.2 or replicate JSON, or None."""
    path = (os.path.join(RES, f"b40v33_s{inst_seed}_r{rho:g}.json") if inst_seed < 100
            else os.path.join(RES, "replicates", f"rep_s{inst_seed}_r{rho:g}.json"))
    if not os.path.exists(path):
        return None
    with open(path) as f:
        runs = json.load(f).get("runs", [])
    if not runs or "ce" not in runs[0].get("methods", {}) or "ce_lb" not in runs[0]["methods"]["ce"]:
        return dict(source=os.path.relpath(path, HERE), status="no certified ce row yet")
    m = runs[0]["methods"]
    ref = dict(source=os.path.relpath(path, HERE), options_sha1=runs[0].get("plan_format", {}).get("options_sha1"),
               zA_ce_lb=m["ce"]["ce_lb"], zA_ce_ub=m["ce"]["ce_ub"], enpv_rn=m.get("rn", {}).get("enpv"),
               heur_ce_ce=m.get("heur_ce", {}).get("ce"))
    h = m.get("heur_ce+exact_recourse")
    if h and "ce_lb" in h:
        ref.update(heur_ce_exact_lb=h["ce_lb"], heur_ce_exact_ub=h["ce_ub"])
    return ref


def dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(B3.jsonable(obj), f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir")
    ap.add_argument("--pkl", default=os.path.join(RES, "meta", "instances.pkl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--instances", type=lambda s: [int(v) for v in s.split(",")], default=None)
    ap.add_argument("--rhos", type=lambda s: [float(v) for v in s.split(",")], default=None)
    ap.add_argument("--max-solves", type=int, default=None, help="stop after this many new certified solves")
    args = ap.parse_args()
    name = os.path.basename(os.path.normpath(args.result_dir))
    out = args.out or os.path.join(RES, "meta", f"scored_{name}.json")
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    res = dict(rows=[], refs={}, cert={}, oa=B3.OA, result_dir=os.path.abspath(args.result_dir), pkl=os.path.abspath(args.pkl))
    if os.path.exists(out):
        with open(out) as f:
            old = json.load(f)
        res["cert"], res["refs"] = old.get("cert", {}), old.get("refs", {})
    runs = []
    for p in sorted(glob.glob(os.path.join(args.result_dir, "i*_r*_*.json"))):
        with open(p) as f:
            r = json.load(f)
        if args.instances is not None and r["instance"] not in args.instances:
            continue
        if args.rhos is not None and r["rho"] not in args.rhos:
            continue
        runs.append((os.path.basename(p), r))
    runs.sort(key=lambda x: (x[1]["instance"], x[1]["rho"], x[0]))
    print(f"{len(runs)} runs in {args.result_dir}; {len(res['cert'])} cached certificates; output {out}", flush=True)
    solves, M, cur = 0, None, None
    for fname, r in runs:
        inst, rho = r["instance"], float(r["rho"])
        entry = data["instances"][inst]
        if cur != (inst, rho):
            M, cur = Model(entry["I"], 1.0 / rho), (inst, rho)
            rk = f"{inst}_r{rho:g}"
            if rk not in res["refs"]:
                res["refs"][rk] = reference(entry["seed"], rho)
        ref = res["refs"].get(f"{inst}_r{rho:g}") or {}
        row = dict(file=fname, instance=inst, rho=rho, method=r["method"], mult=r["mult"], seed=r["seed"],
                   with_seed=r["with_seed"], long=r.get("long", False), objective=r["objective"], evals=r["info"]["evals"],
                   max_evals=r["max_evals"], run_wall=r["wall"], stop=r["info"].get("stop"), ce=r["ce"], enpv=r["enpv"],
                   seed_ce=r["seed_ce"], returned_seed=r.get("returned_seed"))
        sha = B3.plan_format(M)["options_sha1"]
        if sha != r["options_sha1"]:
            row["error"] = f"option table SHA-1 {sha[:12]} != run's {r['options_sha1'][:12]}"
            res["rows"].append(row)
            print(f"  {fname}: {row['error']}", flush=True)
            continue
        plan = B3.dense_plan(M, r["plan"])
        seed_plan = BL.heuristic_plan(M, **r["fill"])
        row["run_model_version"] = r.get("model_version"); row["model_version"] = B3.MODEL_VERSION; row["solver_rev"] = B3.SOLVER_REV
        row["plan_feasible_now"] = bool(M.feasible(plan))
        if row["plan_feasible_now"]:                                        # in-sample scores re-evaluated under the current model
            e_now = M.evaluate(plan); row["ce"], row["enpv"] = e_now["ce"], e_now["enpv"]
        row["seed_ce"] = M.evaluate(seed_plan)["ce"]
        for tag, pl in (("", plan), ("seed_", seed_plan)):
            key = f"{inst}|{rho:g}|{fs_key(pl)}"
            if key not in res["cert"]:
                if args.max_solves is not None and solves >= args.max_solves:
                    break
                t0 = time.time()
                c = solve_recourse_exact(M, pl, **B3.OA)
                res["cert"][key] = dict(ce_lb=c["ce_lb"], ce_ub=c["ce_ub"], rounds=c["rounds"], time=time.time() - t0)
                solves += 1
                print(f"  certified {tag or 'run '}first stage of {fname}: [{c['ce_lb']:.3f}, {c['ce_ub']:.3f}] "
                      f"{time.time() - t0:.1f}s", flush=True)
                dump(res, out)
            c = res["cert"][key]
            row.update({f"{tag}ce_lb": c["ce_lb"], f"{tag}ce_ub": c["ce_ub"], f"{tag}cert_time": c["time"]})
        if ref.get("options_sha1") not in (None, sha):                      # the reference is a different instance
            row["ref_error"] = f"reference option table SHA-1 {ref['options_sha1'][:12]} != {sha[:12]}"
        elif "ce_lb" in row and "zA_ce_lb" in ref:
            row["gap_lo"], row["gap_hi"] = ref["zA_ce_lb"] - row["ce_ub"], ref["zA_ce_ub"] - row["ce_lb"]
            if ref.get("enpv_rn"):
                row["gap_lo_pct"] = 100 * row["gap_lo"] / ref["enpv_rn"]
                row["gap_hi_pct"] = 100 * row["gap_hi"] / ref["enpv_rn"]
        res["rows"].append(row)
        if args.max_solves is not None and solves >= args.max_solves and "ce_lb" not in row:
            print(f"stopping after {solves} new solves (--max-solves)", flush=True)
            break
    dump(res, out)
    done = sum("ce_lb" in r for r in res["rows"])
    print(f"wrote {out}: {len(res['rows'])} rows, {done} with a certificate, {len(res['cert'])} distinct first stages, "
          f"{solves} new solves", flush=True)


if __name__ == "__main__":
    main()
