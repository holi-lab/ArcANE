#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# rollout.sh - generate completions from a base model or fine-tuned checkpoint
#
# Usage examples:
#
#   # Base model
#   DATA_FILE=./data/test.jsonl bash scripts/rollout.sh
#
#   # Fine-tuned checkpoint (full run directory)
#   CHECKPOINT=./outputs/<model>/<dataset>/full/<run> bash scripts/rollout.sh
#
#   # Specific checkpoint step
#   CHECKPOINT=./outputs/<model>/<dataset>/full/<run>/checkpoint-100 bash scripts/rollout.sh
#
#   # Extra CLI flags are forwarded to rollout.py
#   bash scripts/rollout.sh --max-samples 50 --greedy
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# --- GPU selection ---------------------------------------------------------
GPU_IDS="${GPU_IDS:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# --- Model selection -------------------------------------------------------
# Priority: CHECKPOINT > MODEL_NAME_OR_PATH > default base model
BASE_MODEL="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B}"
CHECKPOINT="${CHECKPOINT:-}"

if [[ -n "$CHECKPOINT" ]]; then
    MODEL_ARG="--checkpoint $CHECKPOINT"
else
    MODEL_ARG="--model-name-or-path $BASE_MODEL"
fi

# --- Data / output ---------------------------------------------------------
DATA_FILE="${DATA_FILE:-./data/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./rollouts}"

# --- Generation defaults ---------------------------------------------------
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"

echo "[rollout] GPU_IDS        : $GPU_IDS"
echo "[rollout] model/checkpoint: ${CHECKPOINT:-$BASE_MODEL}"
echo "[rollout] data file      : $DATA_FILE"
echo "[rollout] output dir     : $OUTPUT_DIR"

uv run python arcane/rollout.py \
    $MODEL_ARG \
    --data-file "$DATA_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --min-p "$MIN_P" \
    "$@"
