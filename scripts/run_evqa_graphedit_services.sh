#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/data/env/wjz/evograph-r1/bin/python"
isolate_root="${1:?usage: run_evqa_graphedit_services.sh ISOLATE_ROOT}"
log_dir="$isolate_root/service_logs"
runtime_encoder="${EVOGRAPH_SERVICE_RUNTIME_ENCODER:-gme}"

test -x "$python_bin"
for name in train_0 train_1 train_2 train_3 val_0 val_1; do
  test -d "$isolate_root/$name"
done
for port in 8010 8011 8012 8013 8014 8015; do
  if curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "refusing to start: port $port already has a healthy graph service" >&2
    exit 1
  fi
done

mkdir -p "$log_dir"
cd "$project_dir"
pids=()
for index in 0 1 2 3; do
  "$python_bin" -u scripts/serve_evqa_graph_edit_copy.py \
    --working-dir "$isolate_root/train_$index" \
    --port "$((8010 + index))" \
    --runtime-encoder "$runtime_encoder" \
    >"$log_dir/train_$index.log" 2>&1 &
  pids+=("$!")
done
for index in 0 1; do
  "$python_bin" -u scripts/serve_evqa_graph_edit_copy.py \
    --working-dir "$isolate_root/val_$index" \
    --port "$((8014 + index))" \
    --runtime-encoder "$runtime_encoder" \
    >"$log_dir/val_$index.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" >"$isolate_root/service_pids.txt"

cleanup() {
  kill "${pids[@]}" 2>/dev/null || true
  wait "${pids[@]}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT
printf 'GraphEdit services started: %s\n' "${pids[*]}"
wait
