#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

GPU_IDS="${GPU_IDS:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

CONFIG_PATH="${CONFIG_PATH:-configs/lora/sft.yaml}"
ACCELERATE_CONFIG_PATH="${ACCELERATE_CONFIG_PATH:-configs/accelerate.yaml}"

export WANDB_PROJECT="${WANDB_PROJECT:-arcane}"

# accelerate's default config has distributed_type=NO; enable real multi-GPU
# (DDP) explicitly instead of relying on auto-detection when >1 process.
ACCELERATE_ARGS=(
  --config_file "$ACCELERATE_CONFIG_PATH"
  --gpu_ids "$GPU_IDS"
  --num_processes "$NUM_PROCESSES"
)
if [ "$NUM_PROCESSES" -gt 1 ]; then
  ACCELERATE_ARGS+=(--multi_gpu)
fi

uv run accelerate launch \
  "${ACCELERATE_ARGS[@]}" \
  arcane/sft_train.py \
  --config "$CONFIG_PATH" \
  "$@"
