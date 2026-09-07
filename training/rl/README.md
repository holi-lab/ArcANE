# ArcANE Training: RLVR (GRPO)

The GRPO stage of the [ArcANE training stack](../README.md), applied to ArcANE-32B only.

## Setup

Run the following from `training/rl` on a Linux x86_64 CUDA host. Install
conda and Git first, and check that `nvidia-smi` can see the GPUs.
The pinned stack uses Python 3.12, PyTorch 2.8, vLLM 0.11.0, and the
CPython 3.12 FlashAttention 2.8.1 wheel. The BF16 recipe requires NVIDIA
Ampere or newer GPUs; macOS, CPU-only hosts, and Python 3.11 are not supported.

Clone the pinned [verl](https://github.com/verl-project/verl) revision and
build the isolated environment:

~~~bash
git clone https://github.com/verl-project/verl.git third_party/verl
git -C third_party/verl checkout 081df509ca9fe02d524f913a7a3f96eefe9ca568

bash scripts/install_env.sh
conda activate ./.conda/verl   # prefix env created in-tree (override: ARCANE_RL_ENV)
~~~

`VERL_DIR` and `ARCANE_RL_ENV` can override the checkout and environment
paths. The installer requires a clean checkout at the exact revision above,
checks dependency consistency and imports, and verifies CUDA/BF16 before
reporting success. An existing Python 3.11 environment is left unchanged:
select a new prefix, for example `ARCANE_RL_ENV=/path/to/arcane-rl-py312`,
and activate that same prefix afterward. The installer downloads CUDA
packages, but does not download model weights or start training.

The installer resolves verl and vLLM dependencies together using
[`constraints.txt`](constraints.txt), including Transformers 4.57.6,
PEFT 0.17.1, and NumPy/OpenCV versions compatible with verl's NumPy 1.x
requirement. These are compatibility pins for the public recipe, not a
complete dependency lock.
It then selects the FlashAttention wheel matching the installed PyTorch
CXX11 ABI and verifies the release asset's SHA256.

## Training Data

The launcher downloads the `rl` configuration of
[holi-lab/ArcANE-Data](https://huggingface.co/datasets/holi-lab/ArcANE-Data)
at commit `e86e5630a32037e1605bca533f822401076f0354`
(`rl/train.parquet`, `rl/test.parquet`), which follows the verl RLHFDataset
schema. Override with `HF_DATASET` / `HF_DATASET_CONFIG` / `HF_REVISION`, or
point `TRAIN` / `VAL` at local files.

The public split contains 2,513 training prompts and 678 validation prompts
from ten novels, as described in the
[training-data scope](../README.md#dataset-configurations).

## Released Models and Model Preparation

Both public 32B repositories contain complete merged Transformers checkpoints
(14 safetensors shards each), not PEFT adapters:

| Purpose | Repository | Pinned revision |
| --- | --- | --- |
| Start a new RLVR run | `holi-lab/ArcANE-32B-DPO` | `9b1dc3d21e44692cf1b5edd0ab55af499696c059` |
| Use the released final RLVR model | `holi-lab/ArcANE-32B-RLVR` | `bd731d129b2ce057d39a95370f5a27e0e7ad8974` |

`run_grpo.sh` defaults to the pinned DPO snapshot and resolves it to the local
Hugging Face cache before invoking verl. Set `HF_MODEL_CACHE` to choose that
cache; the first resolution downloads about 61 GiB of model shards. For a
different Hub model, set both `BASE_MODEL` and
`HF_MODEL_REVISION`; a local `BASE_MODEL` directory needs no revision.

Download the final RLVR checkpoint for inference without starting training:

~~~bash
python scripts/download_hf_model.py \
  --repo-id holi-lab/ArcANE-32B-RLVR \
  --revision bd731d129b2ce057d39a95370f5a27e0e7ad8974
~~~

Merge custom DPO adapters first so verl's adapter-disabled KL reference uses
the DPO weights:

~~~bash
python scripts/merge_lora.py --adapter /path/to/dpo-adapter --out /path/to/dpo-merged
~~~

The merger uses the base-model revision recorded by the current SFT/DPO
recipes. If `--base` selects a different Hub repository, also provide an
explicit `--base-revision`; local base paths do not need one. Pass the merged
output directory through `BASE_MODEL`.

## Reward Modes

verl loads `verl_arcane/reward.py` through its custom reward interface.

Remote mode calls a user-provided HTTP reward service. The paper's reward
uses Qwen3.6-27B with `evaluation/per_response/prompt.py` and returns
`mean(APF, RPF, RAE) / 100`. `SMOKE=1` defaults to a reference-free heuristic
reward for pipeline checks, not the paper's reward.

Each request contains the rollout and the context exposed by verl:

~~~json
{"data_source": "dataset identifier", "response": "policy response", "ground_truth": "value from the parquet row", "extra_info": {}}
~~~

In the public `rl` split, `ground_truth` is a JSON string with exactly the keys
`action`, `speech`, and `thought`; each value is a string or null. The same
values are available as `extra_info.gt_action`, `extra_info.gt_speech`, and
`extra_info.gt_thought`.

The service must return a JSON object with a numeric score; additional numeric keys are forwarded as logged reward metrics (`{"score": 0.82, "fidelity": 0.79, "format": 1.0}`).

Use `ARCANE_REWARD_TIMEOUT`, `ARCANE_REWARD_RETRIES`, and `ARCANE_REWARD_RETRY_DELAY` to tune failure handling.

## Launch

Normal training defaults to remote reward mode. Set `ARCANE_REWARD_URL` to
your service's scoring endpoint:

~~~bash
ARCANE_REWARD_MODE=remote \
ARCANE_REWARD_URL=http://127.0.0.1:8000/score \
bash scripts/run_grpo.sh
~~~

This uses the pinned public DPO model and public `rl` dataset by default. A
custom merged policy can instead be selected with `BASE_MODEL=/path/to/model`.

Extra arguments are forwarded to verl, so Hydra settings can be overridden at launch.

## Runtime and Tuning

| Area | Environment variables |
| --- | --- |
| Hardware | N_GPUS, ROLLOUT_TP, GPU_MEMORY_UTILIZATION, MODEL_USE_SHM |
| Data | HF_DATASET, HF_DATASET_CONFIG, HF_TRAIN_SPLIT, HF_VAL_SPLIT |
| Hub/cache | HF_REVISION, HF_DATA_CACHE, HF_MODEL_REVISION, HF_MODEL_CACHE |
| Local data | TRAIN, VAL |
| Batching | TRAIN_BS, MINI_BS, ROLLOUT_N |
| Optimization | LEARNING_RATE, KL_LOSS_COEF |
| Length | MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH, MAX_MODEL_LENGTH |
| Output | OUTPUT_DIR, EXP_NAME, WANDB_PROJECT, LOGGER |

Defaults use two CUDA GPUs for the actor and rollout (`N_GPUS=2`,
`ROLLOUT_TP=2`). The remote reward judge runs separately. Adjust GPU count,
sequence lengths, and memory utilization to the available memory.

Maximum prompt and response lengths default to 8,192 and 2,048 tokens. Chat templates
explicitly set `enable_thinking=False`, and actor/reference weights load in
BF16. Model snapshots stay in the disk cache by default (`MODEL_USE_SHM=False`)
so the 61 GiB checkpoint is not copied into `/dev/shm`; enable shared-memory
loading only when enough shared memory is provisioned.
