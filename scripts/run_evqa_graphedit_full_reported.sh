#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${EVOGRAPH_GRAPHEDIT_OUTPUT_ROOT:-/home/data/dataset/wjz/EvoGraph-R1/expr_mm/evqa_graphedit_full1891_3b_epoch1_v1}"
experiment_name="${EVOGRAPH_GRAPHEDIT_EXPERIMENT_NAME:-Qwen2.5-VL-3B_E-VQA_GraphEdit_full1891_epoch1_v1}"
train_exit=0
"$project_dir/scripts/run_evqa_graphedit_full_3b.sh" || train_exit=$?
report_exit=0
/home/data/env/wjz/evograph-r1/bin/python "$project_dir/scripts/report_evqa_graph_edit_training.py" \
    --output "$output_root" --experiment "$experiment_name" --process-exit "$train_exit" || report_exit=$?
if (( train_exit != 0 )); then
    exit "$train_exit"
fi
exit "$report_exit"
