#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/data/env/wjz/evograph-r1/bin/python"
data_root="/home/data/dataset/wjz/EvoGraph-R1"
subset="paper_grpo_smoke63_v1"
model_path="${EVOGRAPH_GRPO_MODEL_PATH:-/home/data/dataset/wjz/models/Qwen2.5-VL-3B-Instruct}"
model_name="${EVOGRAPH_GRPO_MODEL_NAME:-$(basename -- "$model_path")}"
output_root="${EVOGRAPH_GRPO_OUTPUT_ROOT:-$data_root/expr_mm/evqa_grpo_smoke63_v1}"
experiment_name="${EVOGRAPH_GRPO_EXPERIMENT_NAME:-${model_name}_E-VQA_strict63_grpo_smoke}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must explicitly select the two approved idle GPUs." >&2
  exit 1
fi
IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#visible_gpus[@]}" -ne 2 ]]; then
  echo "This smoke profile requires exactly two GPUs; got $CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi
if [[ ! -x "$python_bin" || ! -d "$model_path" ]]; then
  echo "The pinned environment or local model is missing." >&2
  exit 1
fi

train_file="${EVOGRAPH_GRPO_TRAIN_FILE:-$data_root/datasets_mm/E-VQA/processed/$subset/train.parquet}"
val_file="${EVOGRAPH_GRPO_VAL_FILE:-$data_root/datasets_mm/E-VQA/processed/$subset/test.parquet}"
max_turns="${EVOGRAPH_GRPO_MAX_TURNS:-3}"
n_repeat="${EVOGRAPH_GRPO_N_REPEAT:-2}"
test -f "$train_file"
test -f "$val_file"
ray_tmp="${EVOGRAPH_RAY_TMPDIR:-/home/data/dataset/wjz/.ray/e63}"
mkdir -p "$ray_tmp" "$output_root/hydra" "$output_root/checkpoints"

export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/data/env/wjz/evograph-r1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_INIT_ADDRESS=local
export RAY_TMPDIR="$ray_tmp"
export RAY_NUM_CPUS=8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MAX_JOBS=4
export EVOGRAPH_MM_MAX_IMAGE_PIXELS="${EVOGRAPH_MM_MAX_IMAGE_PIXELS:-200704}"
export EVOGRAPH_MM_MIN_IMAGE_PIXELS="${EVOGRAPH_MM_MIN_IMAGE_PIXELS:-50176}"
export MM_SEARCH_API_URL="http://127.0.0.1:8005/search"
export TEXT_SEARCH_API_URL="$MM_SEARCH_API_URL"
export MM_API_URL="$MM_SEARCH_API_URL"
export MM_SEARCH_TIMEOUT=120
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export TOOL_RESPONSE_IMAGE_LIMIT=1
export ROLLOUT_REPEAT_INTERLEAVE=true
export PRINT_SAMPLE_PROMPT=0

cd "$output_root"
exec "$python_bin" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.train_files="$train_file" \
  data.val_files="$val_file" \
  data.val_batch_size=2 \
  data.image_key=image_path \
  data.train_batch_size=2 \
  data.max_prompt_length=2048 \
  data.max_response_length=1536 \
  data.max_start_length=2048 \
  data.max_tool_response_length=512 \
  data.use_custom_tool_format_func=true \
  actor_rollout_ref.model.path="$model_path" \
  +actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.name=vllm \
  +actor_rollout_ref.rollout.micro_batch_size=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
  actor_rollout_ref.rollout.max_model_len=6144 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.n_repeat="$n_repeat" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  "trainer.logger=['console']" \
  trainer.project_name=EvoGraph-R1-MM-Smoke \
  trainer.experiment_name="$experiment_name" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.resume_mode=disable \
  trainer.default_local_dir="$output_root/checkpoints" \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.val_before_train=false \
  tool.env=mm_search \
  tool.max_turns="$max_turns" \
  tool.use_batch_tool_calls=True \
  +tool.response_guidance=true \
  +tool.max_turn_response_length=320 \
  +data.num_workers=0 \
  +data.pin_memory=False \
  +data.persistent_workers=False \
  hydra.run.dir="$output_root/hydra" \
  "$@"
