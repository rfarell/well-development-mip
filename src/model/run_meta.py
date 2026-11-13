"""Run the metaheuristic study on a machine without Gurobi (numpy only).

Usage:
  run_meta.py instances.pkl out_dir [--methods bpso_v,rk_pso,ga2,bpso_sigmoid,ga] [--seeds 10] [--mults 1,5,10]
              [--workers 8] [--instances 0,1,2,3,100,101,102,103] [--rhos 100,300] [--objective ce]
              [--no-seed-runs 3] [--no-seed-mults 1,5,10] [--long] [--time-cap SECONDS] [--dry-run]

instances.pkl holds the instances, the tuned rule settings and the evaluation budget (evals_1x) per instance and rho.
For every (instance, rho, method, mult, seed) one run with the heuristic seed (seeds 0..--seeds-1) and, for seeds
0..--no-seed-runs-1 at the --no-seed-mults (default: every mult), one run started without it. The
evaluation budget is mult x evals_1x of that instance and rho; the objective is --objective, and the recourse rule
and the seed are the heuristic tuned on the same objective (heur_ce for 'ce', heur_ev for 'enpv').

Methods: bpso_v, rk_pso, ga2 (metaheuristics.py), and the original sigmoid BPSO (bpso_sigmoid = baselines.pso_plan)
and GA (ga = baselines.ga_plan), both given the evaluation budget through their max_evals option.

The run's random seed is rng_seed = seed + 1000 * mult + 100000 * (not with_seed), so the runs at different budgets
are independent draws. --long adds one rk_pso and one bpso_v run at 50x on instance 0 at rho 100 (seed 0, history
every 200 evaluations): the run-to-target curve.

Output: one JSON per run in out_dir, named i{instance}_r{rho}_{method}_x{mult}_s{seed}_{seed|noseed}[_long].json,
written atomically and skipped when it exists (the run is resumable). Each holds the run's identity and budget,
the sparse plan, info (iters, evals, time, seed_score, best_score, best_history = (evals, seconds, best), stop,
calls), the in-sample CE, E[NPV] and lambda*sigma from Model.evaluate, the heuristic seed's in-sample CE and
E[NPV], feasibility, the option table's SHA-1 (as bench_v3.plan_format) and the wall-clock and CPU time.
Workers are separate processes (spawn) with OMP_NUM_THREADS=1 set before numpy is imported.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"                                                 # before numpy is imported, here and in workers

import argparse                                                          # noqa: E402
import json                                                              # noqa: E402
import multiprocessing as mp                                             # noqa: E402
import pickle                                                            # noqa: E402
import platform                                                          # noqa: E402
import sys                                                               # noqa: E402
import time                                                              # noqa: E402
import traceback                                                         # noqa: E402

import numpy as np                                                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Model, MODEL_VERSION                                   # noqa: E402
import baselines as BL                                                   # noqa: E402
import metaheuristics as MH                                              # noqa: E402

ALL_METHODS = ("bpso_v", "rk_pso", "ga2", "bpso_sigmoid", "ga")
LONG_MULT, LONG_HIST = 50, 200


def bpso_sigmoid(M, max_evals, time_budget=None, hist_every=None, **kw):
    """The original sigmoid BPSO (baselines.pso_plan) under an evaluation budget; history as (evals, seconds, best)."""
    plan, info = BL.pso_plan(M, time_budget=time_budget, max_evals=max_evals, **kw)
    return plan, _uniform_info(info, max_evals)


def ga(M, max_evals, time_budget=None, hist_every=None, **kw):
    """The original GA (baselines.ga_plan) under an evaluation budget; history as (evals, seconds, best)."""
    plan, info = BL.ga_plan(M, time_budget=time_budget, max_evals=max_evals, **kw)
    return plan, _uniform_info(info, max_evals)


def _uniform_info(info, max_evals):
    """The baselines' info in the metaheuristics' format: best_history as (evals, seconds, best) and a stop reason."""
    hist = [(int(e), round(t, 3), float(b)) for e, t, b in info.pop("eval_history")]
    stop = "evals" if info["evals"] >= max_evals else "stall or time"
    return dict(info, best_history=hist, stop=stop)


FUNCS = dict(MH.METHODS, bpso_sigmoid=bpso_sigmoid, ga=ga)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def dump(obj, path):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(jsonable(obj), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def out_name(t):
    tag = "seed" if t["with_seed"] else "noseed"
    return (f"i{t['instance']}_r{t['rho']:g}_{t['method']}_x{t['mult']:g}_s{t['seed']}_{tag}"
            f"{'_long' if t['long'] else ''}.json")


# ------------------------------------------------------------------------------------------------ worker side
_W = {}


def _init_worker(pkl):
    with open(pkl, "rb") as f:
        _W["data"] = pickle.load(f)
    _W["models"] = {}


def _model(inst, rho):
    key = (inst, rho)
    if key not in _W["models"]:
        _W["models"].clear()                                             # tasks are grouped; keep one Model per worker
        I = _W["data"]["instances"][inst]["I"]
        _W["models"][key] = Model(I, 1.0 / rho)
    return _W["models"][key]


def run_task(t):
    path = os.path.join(t["out_dir"], out_name(t))
    if os.path.exists(path):
        return dict(name=out_name(t), skipped=True)
    try:
        entry = _W["data"]["instances"][t["instance"]]
        er = entry["rho"][t["rho"]]
        M = _model(t["instance"], t["rho"])
        fill = er["heur_ce"] if t["objective"] == "ce" else er["heur_ev"]
        max_evals = int(round(t["mult"] * er["evals_1x"]))
        seed_plan = BL.heuristic_plan(M, **fill)
        seed_ev = M.evaluate(seed_plan)
        w0, c0 = time.time(), time.process_time()
        plan, info = FUNCS[t["method"]](M, max_evals, time_budget=t["time_cap"], seed=t["rng_seed"],
                                        objective=t["objective"], fill=fill, with_seed=t["with_seed"],
                                        hist_every=t["hist_every"])
        wall, cpu = time.time() - w0, time.process_time() - c0
        ev = M.evaluate(plan)
        sha = MH.options_sha1(M)
        res = dict(instance=t["instance"], instance_seed=entry["seed"], knobs=entry["knobs"], rho=t["rho"], lam=1.0 / t["rho"],
                   method=t["method"], mult=t["mult"], seed=t["seed"], rng_seed=t["rng_seed"], with_seed=t["with_seed"],
                   long=t["long"], objective=t["objective"], fill=fill, evals_1x=er["evals_1x"], max_evals=max_evals, model_version=MODEL_VERSION,
                   time_cap=t["time_cap"], hist_every=t["hist_every"],
                   plan=MH.sparse_plan(plan), info=info, ce=ev["ce"], enpv=ev["enpv"], lam_sigma=ev["lam_sigma"],
                   seed_ce=seed_ev["ce"], seed_enpv=seed_ev["enpv"], returned_seed=bool(MH.sparse_plan(plan) == MH.sparse_plan(seed_plan)),
                   feasible=bool(M.feasible(plan)), options_sha1=sha, sha1_matches_export=(sha == entry["options_sha1"]),
                   wall=wall, cpu=cpu, host=platform.node(), numpy=np.__version__,
                   finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        dump(res, path)
        return dict(name=out_name(t), skipped=False, ce=ev["ce"], seed_ce=seed_ev["ce"], evals=info["evals"],
                    max_evals=max_evals, wall=wall, stop=info.get("stop"), feasible=res["feasible"])
    except Exception as e:                                               # report and go on; the task can be rerun
        return dict(name=out_name(t), error=repr(e), traceback=traceback.format_exc()[-3000:])


# ------------------------------------------------------------------------------------------------ main side
def ints(s):
    return [int(v) for v in s.split(",") if v.strip()]


def floats(s):
    return [float(v) for v in s.split(",") if v.strip()]


def build_tasks(args, data):
    insts = args.instances if args.instances is not None else list(data["instances"])
    rhos = args.rhos if args.rhos is not None else [float(r) for r in data["rhos"]]
    ns_mults = args.no_seed_mults if args.no_seed_mults is not None else args.mults
    base = dict(out_dir=args.out_dir, objective=args.objective, time_cap=args.time_cap, hist_every=None, long=False)
    tasks = []
    for inst in insts:
        if inst not in data["instances"]:
            sys.exit(f"instance {inst} not in {args.pkl} (have {list(data['instances'])})")
        for rho in rhos:
            if rho not in data["instances"][inst]["rho"]:
                sys.exit(f"rho {rho:g} not exported for instance {inst}")
            for method in args.methods:
                for mult in args.mults:
                    for seed in range(args.seeds):
                        tasks.append(dict(base, instance=inst, rho=rho, method=method, mult=mult, seed=seed, with_seed=True))
                for mult in ns_mults:
                    for seed in range(args.no_seed_runs):
                        tasks.append(dict(base, instance=inst, rho=rho, method=method, mult=mult, seed=seed, with_seed=False))
    if args.long:
        inst0 = 0 if 0 in data["instances"] else list(data["instances"])[0]
        for method in ("rk_pso", "bpso_v"):
            tasks.append(dict(base, instance=inst0, rho=100.0, method=method, mult=LONG_MULT, seed=0, with_seed=True,
                              long=True, hist_every=LONG_HIST))
    for t in tasks:
        t["rng_seed"] = int(t["seed"] + 1000 * t["mult"] + 100000 * (not t["with_seed"]))
        t["budget"] = int(round(t["mult"] * data["instances"][t["instance"]]["rho"][t["rho"]]["evals_1x"]))
    # longest first (balances the pool), grouped by (instance, rho) within a budget so workers reuse their Model
    tasks.sort(key=lambda t: (-t["budget"], t["instance"], t["rho"], t["method"], t["seed"]))
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkl")
    ap.add_argument("out_dir")
    ap.add_argument("--methods", type=lambda s: [m.strip() for m in s.split(",") if m.strip()], default=list(ALL_METHODS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--mults", type=ints, default=[1, 5, 10])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--instances", type=ints, default=None)
    ap.add_argument("--rhos", type=floats, default=None)
    ap.add_argument("--objective", choices=("ce", "enpv"), default="ce")
    ap.add_argument("--no-seed-runs", type=int, default=3, help="seeds per cell run without the heuristic seed")
    ap.add_argument("--no-seed-mults", type=ints, default=None, help="budgets of the no-seed runs (default: --mults)")
    ap.add_argument("--long", action="store_true", help="add the two 50x run-to-target runs (instance 0, rho 100)")
    ap.add_argument("--time-cap", type=float, default=None, help="optional wall-clock cap per run, seconds")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    bad = [m for m in args.methods if m not in FUNCS]
    if bad:
        sys.exit(f"unknown methods {bad}; choose from {list(FUNCS)}")
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    tasks = build_tasks(args, data)
    todo = [t for t in tasks if not os.path.exists(os.path.join(args.out_dir, out_name(t)))]
    tot = sum(t["budget"] for t in todo)
    print(f"{len(tasks)} runs, {len(tasks) - len(todo)} already done, {len(todo)} to do, {tot:,} evaluations; "
          f"{args.workers} workers; host {platform.node()}, numpy {np.__version__}", flush=True)
    if args.dry_run or not todo:
        return
    with open(os.path.join(args.out_dir, "run_meta_args.json"), "a") as f:
        f.write(json.dumps(dict(vars(args), started=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node())) + "\n")
    ctx = mp.get_context("spawn")
    t0, done, errors, evals_done = time.time(), 0, 0, 0
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.pkl,), maxtasksperchild=200) as pool:
        for r in pool.imap_unordered(run_task, todo, chunksize=1):
            done += 1
            el = time.time() - t0
            if "error" in r:
                errors += 1
                print(f"[{done}/{len(todo)} {el / 60:.1f} min] ERROR {r['name']}: {r['error']}\n{r['traceback']}", flush=True)
                continue
            if r.get("skipped"):
                continue
            evals_done += r["max_evals"]
            eta = el / max(1, evals_done) * (tot - evals_done)
            print(f"[{done}/{len(todo)} {el / 60:.1f} min, eta {eta / 60:.0f} min] {r['name']}: ce {r['ce']:.3f} "
                  f"(seed {r['seed_ce']:.3f}, {r['ce'] - r['seed_ce']:+.3f}) evals {r['evals']}/{r['max_evals']} "
                  f"stop {r['stop']} {r['wall']:.0f}s{'' if r['feasible'] else ' INFEASIBLE'}", flush=True)
    print(f"done: {done} runs in {(time.time() - t0) / 60:.1f} min, {errors} error(s)", flush=True)


if __name__ == "__main__":
    main()
