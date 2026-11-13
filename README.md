# Risk-averse development planning in pooled shale spacing units

Code and run files for the paper "Risk-Averse Development Planning in Pooled Shale Spacing Units: What
Optimization and Risk Modeling Add" by R. Farell, J. E. Bickel, and C. Bajaj.

The folder holds the instance generators, the solver with its optimality bound, the comparators, the experiment
drivers, the run files behind the results in the paper, and the scripts that turn those run files into the
paper's tables, numbers, and figures.

## Contents

| Path | What it is |
|---|---|
| `src/model/instance.py` | synthetic Bakken-like instance generator |
| `src/model/model.py` | option tables, node coefficients, exact evaluation, feasibility checks |
| `src/model/solve.py` | outer-approximation solver with its optimality bound; risk-neutral and tilted MIPs |
| `src/model/baselines.py`, `metaheuristics.py` | rank-and-fill rule; particle-swarm and genetic-algorithm searches |
| `src/model/bench_v3.py`, `exp_*.py`, `run_meta.py`, `score_meta.py`, `merge_scored.py` | experiment drivers (see Runs) |
| `src/model/audit_certificates.py` | audit of stored bounds across solves |
| `src/model/summarize_*.py`, `rescore_net_frozen_cov.py`, `fig_*.py`, `supplement_figures.py`, `solver_figure_data.py`, `report_currency.py`, `paper/make_*.py` | reporting: tables, numbers, figures |
| `src/model/selftest.py` | consistency checks of the model and solver on a small instance |
| `src/ndic/build_ndic_instance.py` | builds the North Dakota instance from a local copy of the public NDIC layers (the layers are not included) |
| `src/model/results/` | run files (see Runs) |
| `data/derived/ndic_mckenzie_40u.pkl` | the North Dakota instance |
| `reference/headline-data.json` | data extract that `fig_headline.py` checks its statistics against |
| `reproduce.sh` | regenerates the tables, numbers, and figures from the run files |

Each script's docstring gives its usage.

## Requirements

Python 3.12 and the packages in `requirements.txt`. Solving needs a Gurobi license; the runs in the paper used
Gurobi 11.0.3 and 12.0.3. The reporting scripts run no solver, but some of them import `gurobipy`.

## Regenerating the paper's numbers

```
./reproduce.sh
```

or `PYTHON=/path/to/python ./reproduce.sh`. It takes a few minutes and runs no solver. The LaTeX tables and
number macros are written to `paper/sections/`, the figures to `paper/figures/`, and plain-text summaries to
`paper/`.

## Runs

All paths are under `src/model/results/`, and all commands are run from `src/model/`. Synthetic areas are
regenerated from the generator seed and settings given on the command line; every run file records them.

| Run files | In the paper | Command |
|---|---|---|
| `replicates/rep_s{100..129}_r{50,100,300}.json` | base study: 30 areas at three risk tolerances | `python exp_replicates.py 8 10 SEED SEED+1 RHO OUT.json --tag base` |
| `replicates/codev_s{100..109}_r100.json` | co-development control | add `--tag codevelop --no-pi --no-committed --codevelop` |
| `replicates/net_s{100..109}_r100.json` | lower-margin control | add `--knobs mu_scale=0.75 --tag net --no-pi --no-committed` |
| `replicates/pm_s{100..109}_r100.json` | full-hedge control | add `--tag pricemarket --no-pi --no-committed --price-market` |
| `replicates/oat_NAME_VALUE_s{100..109}_r100.json` | one generator setting changed at a time | add `--knobs NAME=VALUE --tag oat:NAME=VALUE --no-pi --no-committed` |
| `replicates/stress_s{0..3}_r{25,50}.json` | stress cases at the two lowest tolerances | `python exp_replicates.py 8 10 SEED SEED+1 RHO OUT.json --knobs SETTINGS --tag stress` |
| `b40v33_s{0..3}_r{100,300}.json` | stress cases at USD 100 and 300 million | `python bench_v3.py 8 10 SEED SEED+1 RHO OUT.json --budget 60 --pso_seeds 0 --ev_seeds 0 --fine 4 --perturb_methods ce,rn,rn_tilt2,heur_ce` |
| `replicates/ndic_r{100,300,1000}.json` | North Dakota case | `python exp_replicates.py 8 10 0 1 RHO OUT.json --tag ndic --instance ../../data/derived/ndic_mckenzie_40u.pkl` |
| `rho_sweep40_v33_s{0,1}.json` | two areas across nine risk tolerances | `python exp_rho_sweep.py 8 10 SEED OUT.json 25,50,100,200,300,500,1000,3000,10000` |
| `conventions40_v33_s100.json` | credit-share and hedge conventions | `python exp_conventions.py 8 10 100 OUT.json` |
| `scale/scale_study.json`, `scale/scale_study_ns.json` | scale, rank, and tree-size studies | `python exp_scale.py OUT.json --parts master,rank,tol` and `--parts nodes,scale` |
| `scale/enum_check.json` | enumeration check on a one-unit instance | `python exp_enum.py OUT.json` |
| `rule_check/synth_s{100..129}_r{50,100,300}.json` | rule comparison on the base areas | `python exp_rule_check.py --seeds SEED --rhos RHO --grid registered --reuse-dir results/replicates --out OUT.json` |
| `ndic_rule_check.json` | rule comparison in the North Dakota case | `python exp_rule_check.py --instance ../../data/derived/ndic_mckenzie_40u.pkl --out OUT.json` |
| `meta/instances.pkl`, `meta/out/`, `meta/scored_all.json` | search-method study | `python run_meta.py results/meta/instances.pkl OUT_DIR`, then `score_meta.py` and `merge_scored.py` |
| `audit_summary.json`, `audit_summary_s1.json` | counts from the audit of stored bounds (Section 5.4) | `python audit_certificates.py FILE.json ... --json OUT.json` |

The stress-case settings by seed are

| Seed | `--knobs` |
|---|---|
| 0 | `own_frac=0.5,deadline_frac=0.1,parent_frac=0.2,rig_mult=2.0,range_mi=2.0` |
| 1 | `own_frac=0.65,deadline_frac=0.1,parent_frac=0.35,rig_mult=2.5,range_mi=2.0` |
| 2 | `own_frac=0.8,deadline_frac=0.1,parent_frac=0.5,rig_mult=3.0,range_mi=3.0` |
| 3 | `own_frac=0.5,deadline_frac=0.25,parent_frac=0.35,rig_mult=3.0,range_mi=3.0` |

`meta/scored_all.json` holds the score of every search run. `meta/out/` holds the complete search runs for the
area shown in the supplement's search figure; `meta/instances.pkl` holds the instances, rule settings, and
evaluation budgets needed to rerun the rest.

The runs in the paper used 32 solver threads on one TACC Vista node (`GRB_THREADS=32 GRB_MEMLIMIT=60`).
`python selftest.py 6 6` checks an installation in about two minutes.

## Names

The paper's NPV-tuned rule and CE-tuned rule are `heur_ev` and `heur_ce` in the run files, and `practice` and
`oracle` in the rule-check files and in some stored labels. The CE-tuned deck is `rn_deck`; the practice deck is
the 0.9 entry of its stored curve. The tilted MIP after one and two iterations is `rn_tilt1` and `rn_tilt2`.

## License

MIT. See `LICENSE`.
