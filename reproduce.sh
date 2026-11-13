#!/bin/bash
# Regenerate the tables, numbers and figures of the paper from the stored run files.
# No solver is run. Takes a few minutes.
#
#   ./reproduce.sh
#   PYTHON=/path/to/python ./reproduce.sh
set -e
PY=${PYTHON:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MPLBACKEND=Agg
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/paper/sections" "$ROOT/paper/figures"

cd "$ROOT/src/model"
# Base study, controls, generator-setting study, stress cases and the North Dakota case.
$PY summarize_replicates.py 'results/replicates/rep_*.json' 'results/replicates/oat_*.json' \
    'results/replicates/pm_*.json' 'results/replicates/ndic_*.json' 'results/replicates/codev_*.json' \
    'results/replicates/net_*.json' --stress 'results/b40v33_s*_r*.json' 'results/replicates/stress_*.json' \
    --boot 4000 --paper ../../paper --md ../../paper/summary_replicates.md
# Rule comparison on the 30 base areas and on the North Dakota case.
$PY exp_rule_check.py --summarize --glob 'results/rule_check/synth_s*_r*.json' \
    --md ../../paper/summary_rule_check.md --macros ../../paper/sections/results_numbers_synthrule.tex
$PY exp_rule_check.py --summarize --out results/ndic_rule_check.json \
    --macros ../../paper/sections/results_numbers_ndicrule.tex
# Search-method study.
$PY summarize_meta.py results/meta/scored_all.json --runs results/meta/out --paper ../../paper \
    --md ../../paper/summary_meta.md
# Lower-margin control re-scored with the base covariance.
$PY rescore_net_frozen_cov.py --paper ../../paper
# Figures of the main text and the supplement.
$PY fig_headline.py
$PY fig_spe.py
$PY fig_ndic_placement.py
$PY fig_si.py

cd "$ROOT/paper"
# Scale, tree-size and enumeration studies; stress-case tables; sweep numbers.
$PY make_study_tables.py
$PY make_v32_extra.py
$PY make_sweep_macros.py
echo "done: see paper/sections and paper/figures"
