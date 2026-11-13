"""Draw three supplement figures from the stored runs: the deck curve, the regime map
and the seed-0 placement map.

Run from anywhere: python src/model/fig_si.py [--output-dir DIR]
Uses the plotting functions in supplement_figures.py; solves no model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import supplement_figures as reporting


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "src/model/results"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_inputs():
    runs, inputs = [], {}
    for seed in range(4):
        for rho in (100, 300):
            path = RESULTS / f"b40v33_s{seed}_r{rho}.json"
            data = json.loads(path.read_text())
            require(not data.get("errors"), f"Stage errors in {path}")
            require(len(data.get("runs", [])) == 1, f"Expected one run in {path}")
            run = data["runs"][0]
            require((run["U"], run["seed"], run["rho"]) == (40, seed, rho),
                    f"Unexpected run identity in {path}")
            require(run.get("model_version") == "v3.3" and run.get("solver_rev") == "s2",
                    f"Expected a v3.3/s2 run in {path}")
            require(run.get("status") == "done" and not run.get("errors"),
                    f"Incomplete run in {path}")
            curve = run["methods"]["rn_deck"]["deck_curve"]
            require(len(curve) == 25 and all(abs(e["deck"] - (0.5 + i * 0.025)) < 1e-9
                    for i, e in enumerate(sorted(curve, key=lambda e: e["deck"]))),
                    f"Unexpected deck grid in {path}")
            runs.append(run)
            inputs[str(path.relative_to(ROOT))] = sha256(path)

    sweeps = []
    for seed in (0, 1):
        path = RESULTS / f"rho_sweep40_v33_s{seed}.json"
        sweep = json.loads(path.read_text())
        require(sweep.get("model_version") == "v3.3" and sweep.get("solver_rev") == "s2",
                f"Expected a v3.3/s2 sweep in {path}")
        require((sweep["seed"], sweep["nx"], sweep["ny"], sweep["U"]) == (seed, 8, 10, 40),
                f"Unexpected sweep identity in {path}")
        require(sorted(row["rho"] for row in sweep["rows"]) ==
                [25, 50, 100, 200, 300, 500, 1000, 3000, 10000], f"Incomplete sweep in {path}")
        require(all(row.get("rn_lam_sigma", 0) > 0 and row.get("rn_enpv") for row in sweep["rows"]),
                f"Missing regime numbers in {path}")
        if seed == 0:
            require(all("dev_year0" in row and "dev_any" in row for row in sweep["rows"]),
                    f"Missing placement sets in {path}")
        sweep["_file"] = str(path)
        inputs[str(path.relative_to(ROOT))] = sha256(path)
        sweeps.append(sweep)
    return runs, sweeps, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/figures")
    args = parser.parse_args()
    runs, sweeps, inputs = load_inputs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt = reporting.mpl()
    outputs = {}
    jobs = (("deck_curve.pdf", lambda path: reporting.fig_deck_curve(plt, runs, path)),
            ("regime_map.pdf", lambda path: reporting.fig_regime_map(plt, sweeps, runs, path)),
            ("placement_s0.pdf", lambda path: reporting.fig_placement(plt, sweeps[0], path)))
    for name, plot in jobs:
        path = args.output_dir / name
        require(plot(str(path)), f"Failed to render {name}")
        outputs[name] = sha256(path)
        print(f"wrote {path}")
    require(not reporting.WARNINGS, f"Figure warnings: {reporting.WARNINGS}")
    sources = [Path(__file__), Path(reporting.__file__), ROOT / "src/model/instance.py"]
    manifest = {
        "command": "python src/model/fig_si.py",
        "model_version": "v3.3", "solver_rev": "s2",
        "inputs": inputs,
        "sources": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "outputs": outputs,
        "stress_runs": len(runs), "sweep_rows": {s["seed"]: len(s["rows"]) for s in sweeps},
    }
    (args.output_dir / "si_figures_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
