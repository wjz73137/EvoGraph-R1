#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$script_dir/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$script_dir/.env"
  set +a
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is required. Run nvidia-smi, choose two idle GPUs, then export CUDA_VISIBLE_DEVICES=<gpu_a>,<gpu_b>." >&2
  exit 1
fi

IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#visible_gpus[@]}" -ne 2 ]]; then
  echo "This first-run profile requires exactly two explicitly selected GPUs; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

export N_GPUS=2
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1

export TRAIN_BATCH_SIZE=2
export VAL_BATCH_SIZE=2
export PPO_MINI_BATCH_SIZE=2
export PPO_MICRO_BATCH_SIZE_PER_GPU=1

export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=1024
export MAX_START_LENGTH=2048
export MAX_TOOL_RESPONSE_LENGTH=1024

export ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.35
export ROLLOUT_N=2
export ROLLOUT_DTYPE=bfloat16

export ACTOR_LR=5e-7
export TOTAL_EPOCHS=1
export TOOL_MAX_TURNS=2
export CUDA_LAUNCH_BLOCKING=0

export MM_DATA_ROOT="${MM_DATA_ROOT:-/home/data/dataset/wjz/EvoGraph-R1}"
export MM_DATASET="${MM_DATASET:-E-VQA}"
export POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/home/data/dataset/wjz/models/Qwen2.5-VL-3B-Instruct}"
export POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-Qwen2.5-VL-3B-Instruct}"

if [[ -z "${MM_SUBSET:-}" || "$MM_SUBSET" == '<'*'>' ]]; then
  echo "MM_SUBSET must name a completed processed subset; refusing to start Ray or training." >&2
  exit 1
fi

if [[ ! -d "$POLICY_MODEL_PATH" ]]; then
  echo "Policy model directory does not exist: $POLICY_MODEL_PATH" >&2
  exit 1
fi

train_file="${TRAIN_FILE:-${MM_DATA_ROOT}/datasets_mm/${MM_DATASET}/processed/${MM_SUBSET}/train.parquet}"
val_file="${VAL_FILE:-${MM_DATA_ROOT}/datasets_mm/${MM_DATASET}/processed/${MM_SUBSET}/test.parquet}"

if [[ ! -f "$train_file" ]]; then
  echo "Training data does not exist: $train_file" >&2
  exit 1
fi
if [[ ! -f "$val_file" ]]; then
  echo "Validation data does not exist: $val_file" >&2
  exit 1
fi

export TRAIN_FILE="$train_file"
export VAL_FILE="$val_file"

exec bash "$script_dir/run_mm_grpo.sh" \
  -p "$POLICY_MODEL_PATH" \
  -m "$POLICY_MODEL_NAME" \
  -d "$MM_DATASET" \
  -s "$MM_SUBSET" \
  "$@"
