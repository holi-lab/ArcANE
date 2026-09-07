#!/usr/bin/env bash
# Create an isolated conda environment and install the pinned verl checkout.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="${VERL_DIR:-${REPO_DIR}/third_party/verl}"
ENV_PREFIX="${ARCANE_RL_ENV:-${REPO_DIR}/.conda/verl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VERL_REVISION="081df509ca9fe02d524f913a7a3f96eefe9ca568"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "ERROR: this pinned RL environment requires Linux x86_64 with NVIDIA CUDA GPUs."
  exit 1
fi
if [[ ! "${PYTHON_VERSION}" =~ ^3\.12(\.[0-9]+)?$ ]]; then
  echo "ERROR: the pinned FlashAttention wheel requires Python 3.12; got ${PYTHON_VERSION}."
  exit 1
fi
for required_command in conda git nvidia-smi; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    echo "ERROR: ${required_command} is required before installing the RL environment."
    exit 1
  fi
done
if ! GPU_LIST="$(nvidia-smi --list-gpus)" || [[ -z "${GPU_LIST}" ]]; then
  echo "ERROR: no accessible NVIDIA GPU was found. Check the driver and GPU visibility."
  exit 1
fi
if [[ ! -f "${VERL_DIR}/pyproject.toml" ]]; then
  echo "ERROR: verl is not available at ${VERL_DIR}"
  echo "Clone the pinned revision described in rl/README.md first."
  exit 1
fi
if ! ACTUAL_REVISION="$(git -C "${VERL_DIR}" rev-parse HEAD)" || \
    [[ "${ACTUAL_REVISION}" != "${VERL_REVISION}" ]]; then
  echo "ERROR: verl must be checked out at ${VERL_REVISION}."
  exit 1
fi
if ! git -C "${VERL_DIR}" diff --quiet HEAD --; then
  echo "ERROR: the verl checkout has tracked changes. Use a clean pinned checkout."
  exit 1
fi
if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  ENV_PYTHON_VERSION="$("${ENV_PREFIX}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "${ENV_PYTHON_VERSION}" != "3.12" ]]; then
    echo "ERROR: ${ENV_PREFIX} uses Python ${ENV_PYTHON_VERSION}; choose a new ARCANE_RL_ENV for Python 3.12."
    exit 1
  fi
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda create --yes --prefix "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip
fi
conda activate "${ENV_PREFIX}"
if [[ ! "$(command -v python)" -ef "${ENV_PREFIX}/bin/python" ]]; then
  echo "ERROR: conda activation did not select ${ENV_PREFIX}/bin/python."
  exit 1
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'

# Resolve verl and the CUDA runtime together so later installs cannot undo
# the NumPy/OpenCV and Transformers/PEFT compatibility constraints.
python -m pip install --no-cache-dir \
  --constraint "${REPO_DIR}/constraints.txt" \
  --editable "${VERL_DIR}" vllm flashinfer-python hf-transfer

# Match the wheel to the installed PyTorch ABI, not the upstream helper's
# hard-coded cxx11abiFALSE wheel. pip checks the release asset's SHA256.
FLASH_ATTN_ABI="$(python -c 'import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())')"
case "${FLASH_ATTN_ABI}" in
  TRUE) FLASH_ATTN_SHA256="55331d797171973c8babf2b5e5fce5f78859b1cd298d54a136e8994147fe9e95" ;;
  FALSE) FLASH_ATTN_SHA256="15db5bb6524dcbf292c3c116aa4c2fa823b80abff0eb5bc58454107bca1ba0c2" ;;
  *) echo "ERROR: unsupported PyTorch CXX11 ABI value: ${FLASH_ATTN_ABI}"; exit 1 ;;
esac
FLASH_ATTN_WHEEL="flash_attn-2.8.1+cu12torch2.8cxx11abi${FLASH_ATTN_ABI}-cp312-cp312-linux_x86_64.whl"
python -m pip install --no-cache-dir --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/${FLASH_ATTN_WHEEL}#sha256=${FLASH_ATTN_SHA256}"
python -m pip check

python - <<'PY'
import importlib
from importlib.metadata import version

import torch

for module in ("verl.trainer.main_ppo", "vllm", "flash_attn", "flashinfer", "peft", "datasets"):
    importlib.import_module(module)
if version("vllm") != "0.11.0":
    raise RuntimeError("The pinned installer requires vllm==0.11.0")
if not torch.__version__.startswith("2.8."):
    raise RuntimeError("The pinned FlashAttention wheel requires PyTorch 2.8")
if version("flash-attn").split("+", 1)[0] != "2.8.1":
    raise RuntimeError("The RL recipe requires FlashAttention 2.8.1")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available to PyTorch; check the NVIDIA driver/runtime")
if not torch.cuda.is_bf16_supported(including_emulation=False):
    raise RuntimeError("The default RL recipe requires native BF16 support (Ampere or newer)")
print("Verified torch", torch.__version__, "| CUDA", torch.version.cuda, "| vllm", version("vllm"))
PY

echo "Installed ArcANE RL environment at ${ENV_PREFIX}"
