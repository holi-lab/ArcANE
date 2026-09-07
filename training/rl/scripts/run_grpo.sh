#!/usr/bin/env bash
# Launch ArcANE GRPO with verl. Run this script from an activated RL environment.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/arcane-ray-${UID:-user}}"
mkdir -p "${RAY_TMPDIR}"

DEFAULT_BASE_MODEL="holi-lab/ArcANE-32B-DPO"
DEFAULT_BASE_MODEL_REVISION="9b1dc3d21e44692cf1b5edd0ab55af499696c059"
BASE_MODEL="${BASE_MODEL:-${DEFAULT_BASE_MODEL}}"
HF_MODEL_REVISION="${HF_MODEL_REVISION:-}"
if [[ "${BASE_MODEL}" == "${DEFAULT_BASE_MODEL}" && -z "${HF_MODEL_REVISION}" ]]; then
  HF_MODEL_REVISION="${DEFAULT_BASE_MODEL_REVISION}"
fi
TRAIN="${TRAIN:-}"
VAL="${VAL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/grpo}"
EXP_NAME="${EXP_NAME:-arcane-grpo-$(date +%Y%m%d_%H%M%S)}"
N_GPUS="${N_GPUS:-2}"
ROLLOUT_TP="${ROLLOUT_TP:-${N_GPUS}}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  export ARCANE_REWARD_MODE="${ARCANE_REWARD_MODE:-heuristic}"
else
  export ARCANE_REWARD_MODE="${ARCANE_REWARD_MODE:-remote}"
fi
if [[ "${ARCANE_REWARD_MODE}" == "remote" || "${ARCANE_REWARD_MODE}" == "rm" ]]; then
  : "${ARCANE_REWARD_URL:?Set ARCANE_REWARD_URL for remote reward mode}"
fi

if [[ -z "${TRAIN}" && -z "${VAL}" ]]; then
  DEFAULT_HF_DATASET="holi-lab/ArcANE-Data"
  DEFAULT_HF_REVISION="e86e5630a32037e1605bca533f822401076f0354"
  HF_DATASET="${HF_DATASET:-${DEFAULT_HF_DATASET}}"
  HF_DATASET_CONFIG="${HF_DATASET_CONFIG:-rl}"
  HF_TRAIN_SPLIT="${HF_TRAIN_SPLIT:-train}"
  HF_VAL_SPLIT="${HF_VAL_SPLIT:-test}"
  HF_REVISION="${HF_REVISION:-}"
  if [[ "${HF_DATASET}" == "${DEFAULT_HF_DATASET}" && -z "${HF_REVISION}" ]]; then
    HF_REVISION="${DEFAULT_HF_REVISION}"
  elif [[ -z "${HF_REVISION}" ]]; then
    echo "ERROR: set HF_REVISION when HF_DATASET is overridden."
    exit 1
  fi
  HF_ARGS=(
    --repo-id "${HF_DATASET}"
    --config "${HF_DATASET_CONFIG}"
    --train-split "${HF_TRAIN_SPLIT}"
    --val-split "${HF_VAL_SPLIT}"
    --revision "${HF_REVISION}"
  )
  USE_HF_DATA=1
elif [[ -z "${TRAIN}" || -z "${VAL}" ]]; then
  echo "ERROR: set both TRAIN and VAL, or leave both unset to use Hugging Face."
  exit 1
else
  USE_HF_DATA=0
fi

MODEL_PATH="${BASE_MODEL}"
if [[ ! -e "${BASE_MODEL}" ]]; then
  if [[ -z "${HF_MODEL_REVISION}" ]]; then
    echo "ERROR: set HF_MODEL_REVISION when BASE_MODEL is a Hugging Face repo ID."
    exit 1
  fi
  MODEL_ARGS=(--repo-id "${BASE_MODEL}" --revision "${HF_MODEL_REVISION}")
  [[ -n "${HF_MODEL_CACHE:-}" ]] && MODEL_ARGS+=(--cache-dir "${HF_MODEL_CACHE}")
  MODEL_PATH="$(python scripts/download_hf_model.py "${MODEL_ARGS[@]}")"
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: resolved model path is not a directory: ${MODEL_PATH}"
  exit 1
fi

if [[ "${USE_HF_DATA}" == "1" ]]; then
  [[ -n "${HF_DATA_CACHE:-}" ]] && HF_ARGS+=(--cache-dir "${HF_DATA_CACHE}")
  DATA_OUTPUT="$(python scripts/download_hf_data.py "${HF_ARGS[@]}")"
  DATA_FILES=()
  while IFS= read -r data_file; do
    DATA_FILES+=("${data_file}")
  done <<< "${DATA_OUTPUT}"
  if [[ "${#DATA_FILES[@]}" -ne 2 ]]; then
    echo "ERROR: expected two parquet paths from the Hugging Face downloader."
    exit 1
  fi
  TRAIN="${DATA_FILES[0]}"
  VAL="${DATA_FILES[1]}"
fi

if [[ ! -f "${TRAIN}" || ! -f "${VAL}" ]]; then
  echo "ERROR: resolved train and validation parquet files are not available."
  exit 1
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  TRAIN_BS="${TRAIN_BS:-8}"
  MINI_BS="${MINI_BS:-8}"
  ROLLOUT_N="${ROLLOUT_N:-4}"
  TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
  MAX_STEPS="${MAX_STEPS:-2}"
  SAVE_FREQ="${SAVE_FREQ:--1}"
  TEST_FREQ="${TEST_FREQ:--1}"
  LOGGER="${LOGGER:-[\"console\"]}"
else
  TRAIN_BS="${TRAIN_BS:-64}"
  MINI_BS="${MINI_BS:-32}"
  ROLLOUT_N="${ROLLOUT_N:-8}"
  TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
  MAX_STEPS="${MAX_STEPS:--1}"
  SAVE_FREQ="${SAVE_FREQ:-20}"
  TEST_FREQ="${TEST_FREQ:-20}"
  LOGGER="${LOGGER:-[\"console\",\"wandb\"]}"
fi

if [[ "${MAX_STEPS}" -gt 0 ]]; then
  set -- "trainer.total_training_steps=${MAX_STEPS}" "$@"
fi

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN}" \
  data.val_files="${VAL}" \
  data.train_batch_size="${TRAIN_BS}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-2048}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_shm="${MODEL_USE_SHM:-False}" \
  actor_rollout_ref.model.lora_rank="${LORA_RANK:-64}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA:-128}" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE:-1e-5}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BS}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.6}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LENGTH:-10240}" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  custom_reward_function.path="${REPO_DIR}/verl_arcane/reward.py" \
  custom_reward_function.name=compute_score \
  trainer.logger="${LOGGER}" \
  trainer.project_name="${WANDB_PROJECT:-arcane-grpo}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.default_local_dir="${OUTPUT_DIR}/${EXP_NAME}" \
  "$@"
