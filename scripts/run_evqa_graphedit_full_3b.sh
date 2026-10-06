#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/data/env/wjz/evograph-r1/bin/python"
data_root="/home/data/dataset/wjz/EvoGraph-R1"
model_path="$data_root/../models/Qwen2.5-VL-3B-Instruct"
checkpoint_default="$data_root/expr_mm/evqa_grpo_graph_covered_full_3b_epoch1_v1/checkpoints/global_step_326"
checkpoint="${EVOGRAPH_GRAPHEDIT_CHECKPOINT:-$checkpoint_default}"
dataset_dir="${EVOGRAPH_GRAPHEDIT_DATASET_DIR:-$data_root/datasets_mm/E-VQA/processed/paper_grpo_graph_edit_full1891_v1}"
output_root="${EVOGRAPH_GRAPHEDIT_OUTPUT_ROOT:-$data_root/expr_mm/evqa_graphedit_full1891_3b_epoch1_v1}"
experiment_name="${EVOGRAPH_GRAPHEDIT_EXPERIMENT_NAME:-Qwen2.5-VL-3B_E-VQA_GraphEdit_full1891_epoch1_v1}"
actor_param_offload="${EVOGRAPH_ACTOR_PARAM_OFFLOAD:-true}"
total_training_steps="${EVOGRAPH_TOTAL_TRAINING_STEPS:-1258}"
reset_dataloader_on_resume="${EVOGRAPH_RESET_DATALOADER_ON_RESUME:-true}"
max_prompt_length="${EVOGRAPH_MAX_PROMPT_LENGTH:-2048}"
max_response_length="${EVOGRAPH_MAX_RESPONSE_LENGTH:-3072}"
max_tool_response_length="${EVOGRAPH_MAX_TOOL_RESPONSE_LENGTH:-512}"
max_turn_response_length="${EVOGRAPH_MAX_TURN_RESPONSE_LENGTH:-320}"

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "2,3" && "${CUDA_VISIBLE_DEVICES:-}" != "3,2" ]]; then
  echo "CUDA_VISIBLE_DEVICES must select only approved physical GPUs 2 and 3." >&2
  exit 1
fi
test -x "$python_bin"
test -d "$model_path"
test -d "$checkpoint/actor"
test -f "$dataset_dir/train.parquet"
test -f "$dataset_dir/test.parquet"

for gpu in 2 3; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")"
  if (( used > 512 )); then
    echo "refusing to start: physical GPU $gpu already uses ${used} MiB" >&2
    exit 1
  fi
done

"$python_bin" - "$project_dir/.env" <<'PY'
import os
import sys
from dotenv import load_dotenv

load_dotenv(sys.argv[1], override=False)
required = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "").strip(),
    "OPENAI_BASE_URL/OPENAI_API_BASE": (
        os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("OPENAI_API_BASE", "").strip()
    ),
    "QUALITY_JUDGE_MODEL/OPENAI_MODEL": (
        os.getenv("QUALITY_JUDGE_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
    ),
}
missing = [name for name, value in required.items() if not value]
if missing:
    raise SystemExit("missing GraphEdit API configuration: " + ", ".join(missing))
print("GraphEdit API preflight: configured")
PY

for port in 8010 8011 8012 8013 8014 8015; do
  curl -fsS --max-time 10 "http://127.0.0.1:$port/health" >/dev/null
done

mkdir -p "$output_root/hydra" "$output_root/checkpoints"

export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/data/env/wjz/evograph-r1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_INIT_ADDRESS=local
export RAY_TMPDIR="${EVOGRAPH_RAY_TMPDIR:-/home/data/dataset/wjz/.ray/ge_full1891_v1}"
export RAY_NUM_CPUS=8
export RAY_OBJECT_STORE_MEMORY_BYTES="${RAY_OBJECT_STORE_MEMORY_BYTES:-8589934592}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MAX_JOBS=4
# vLLM sleep mode relies on its CuMemAllocator pool. PyTorch expandable
# segments bypass that pool, leaving rollout weights resident during backward.
unset PYTORCH_CUDA_ALLOC_CONF
export EVOGRAPH_MM_MAX_IMAGE_PIXELS=100352
export EVOGRAPH_MM_MIN_IMAGE_PIXELS=50176
export MM_SEARCH_API_URL="http://127.0.0.1:8010/search"
export TEXT_SEARCH_API_URL="$MM_SEARCH_API_URL"
export MM_TRAIN_SEARCH_API_URLS="http://127.0.0.1:8010/search,http://127.0.0.1:8011/search,http://127.0.0.1:8012/search,http://127.0.0.1:8013/search"
export MM_VAL_SEARCH_API_URLS="http://127.0.0.1:8014/search,http://127.0.0.1:8015/search"
export MM_SEARCH_TIMEOUT=300
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7890}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export TOOL_RESPONSE_IMAGE_LIMIT=1
export ROLLOUT_REPEAT_INTERLEAVE=true
export PRINT_SAMPLE_PROMPT=0
export EVOGRAPH_SAVE_TRAIN_TRAJECTORIES=true
export TOOL_USE_DEFERRED_EXECUTION=false
export WEBSEARCH_FUZZY_ENABLED=false
export WEBSEARCH_WIKIPEDIA_AUGMENT=true
export EVOGRAPH_GRAPH_EDIT_COMMIT_GATE="${EVOGRAPH_GRAPH_EDIT_COMMIT_GATE:-api}"
export GRAPH_EDIT_GATE_MIN_CONFIDENCE="${GRAPH_EDIT_GATE_MIN_CONFIDENCE:-0.85}"

mkdir -p "$RAY_TMPDIR"
cd "$output_root"
exec "$python_bin" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.train_files="$dataset_dir/train.parquet" \
  data.val_files="$dataset_dir/test.parquet" \
  data.val_batch_size=2 \
  data.image_key=image_path \
  data.train_batch_size=2 \
  data.max_prompt_length="$max_prompt_length" \
  data.max_response_length="$max_response_length" \
  data.max_start_length="$max_prompt_length" \
  data.max_tool_response_length="$max_tool_response_length" \
  data.use_custom_tool_format_func=true \
  actor_rollout_ref.model.path="$model_path" \
  +actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  +actor_rollout_ref.actor.optim.foreach=false \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  +actor_rollout_ref.actor.entropy_chunk_size=8 \
  +actor_rollout_ref.actor.empty_cache_before_backward=true \
  actor_rollout_ref.actor.fsdp_config.param_offload="$actor_param_offload" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.sleep_level=1 \
  +actor_rollout_ref.rollout.micro_batch_size=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.33 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.n_repeat=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  +actor_rollout_ref.ref.fsdp_config.native_cpu_offload="${EVOGRAPH_REF_NATIVE_CPU_OFFLOAD:-true}" \
  trainer.critic_warmup=0 \
  "trainer.logger=['console']" \
  trainer.project_name=EvoGraph-R1-MM-GraphEdit-Full1891 \
  trainer.experiment_name="$experiment_name" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.resume_mode="$checkpoint" \
  trainer.reset_dataloader_on_resume="$reset_dataloader_on_resume" \
  trainer.default_local_dir="$output_root/checkpoints" \
  trainer.save_freq="${EVOGRAPH_SAVE_FREQ:-200}" \
  trainer.remove_previous_ckpt_in_save="${EVOGRAPH_REMOVE_PREVIOUS_CHECKPOINT:-true}" \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$total_training_steps" \
  trainer.val_before_train=false \
  +trainer.val_only=false \
  tool.env=mm_all \
  tool.max_turns=8 \
  tool.use_batch_tool_calls=true \
  tool.force_graph_edit_verification=true \
  +tool.response_guidance=true \
  +tool.max_turn_response_length="$max_turn_response_length" \
  +data.num_workers=0 \
  +data.pin_memory=false \
  +data.persistent_workers=false \
  hydra.run.dir="$output_root/hydra"
