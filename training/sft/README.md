# ArcANE Training: SFT and DPO

Supervised fine-tuning and DPO preference tuning for `ArcANE-8B` and
`ArcANE-32B`. Released hyperparameters, dataset configurations and the RLVR
stage are documented in [../README.md](../README.md).

## Setup

Install [uv](https://docs.astral.sh/uv/) and create the locked environment:

~~~bash
uv sync

uv run hf auth login     # optional
uv run wandb login       # optional; the configs default to report_to: wandb
~~~

All ArcANE dataset and model repositories used here are public and ungated, so
Hugging Face authentication is not required. To train without Weights & Biases,
pass `--report-to none`.

The shipped configs set `attn_implementation: flash_attention_2`, which is not
part of the locked environment. Install it, or switch the attention backend:

~~~bash
uv pip install flash-attn --no-build-isolation   # needs a CUDA toolchain
# or, without flash-attn:
bash scripts/train_sft_lora.sh --attn-implementation sdpa
~~~

## Data

The configs load the `sft` and `dpo` configurations of
[holi-lab/ArcANE-Data](https://huggingface.co/datasets/holi-lab/ArcANE-Data)
at revision `e86e5630a32037e1605bca533f822401076f0354`: `train` for training and
`test` for evaluation. This public snapshot contains 20,560 SFT training rows,
2,285 SFT test rows, 12,746 DPO training rows, and 375 DPO test rows across 10
novels.

The paper and model cards describe the original 12-novel training pool. The
public dataset excludes *East Lynne* and *The Underdogs*. The recipes therefore
reproduce the released pipeline and hyperparameters on the public subset, but
they are not an exact retraining of the published checkpoints. Use the released
model repositories below when the original checkpoint itself is required.

SFT examples contain a conversational messages column:

~~~json
{
  "messages": [
    {"role": "system", "content": "You are a helpful character."},
    {"role": "user", "content": "What would you do?"},
    {"role": "assistant", "content": "I would begin by listening."}
  ]
}
~~~

DPO examples contain conversational chosen and rejected columns: role/content
message lists with an identical prompt and different final assistant responses.

## Published checkpoints

| Checkpoint | Immutable revision | Published format |
| --- | --- | --- |
| [ArcANE-8B-SFT](https://huggingface.co/holi-lab/ArcANE-8B-SFT) | `3b21f160d130741634b07e4a18c79460e73a829f` | Full BF16 weights |
| [ArcANE-8B-DPO](https://huggingface.co/holi-lab/ArcANE-8B-DPO) | `67b9bea604bea65a5245928ed87103545814b4f7` | Full BF16 weights |
| [ArcANE-32B-SFT](https://huggingface.co/holi-lab/ArcANE-32B-SFT) | `21f6f98fae38b34f420d48a25a50d6e0e654d649` | Merged full BF16 weights |
| [ArcANE-32B-DPO](https://huggingface.co/holi-lab/ArcANE-32B-DPO) | `9b1dc3d21e44692cf1b5edd0ab55af499696c059` | Merged full BF16 weights |

The 32B recipes train LoRA adapters, but the published 32B repositories are
self-contained merged checkpoints. They contain sharded `model-*.safetensors`
files and an index, not `adapter_config.json` or `adapter_model.safetensors`.

## Training

| Objective | LoRA (32B recipe) | Full tuning (8B recipe) |
| --- | --- | --- |
| SFT | scripts/train_sft_lora.sh | scripts/train_sft_full.sh |
| DPO | scripts/train_dpo_lora.sh | scripts/train_dpo_full.sh |

Each script defaults to its matching config under `configs/lora/` or
`configs/full/`. The SFT configs pin the corresponding Qwen3 base revision. The
DPO configs pin the released, full-weight ArcANE SFT checkpoint and start a new
DPO update from it. Running the SFT command first does not automatically change
the DPO input to the newly created local checkpoint.

~~~bash
bash scripts/train_sft_lora.sh
bash scripts/train_dpo_lora.sh

# Multi-GPU: set both the visible devices and the Accelerate process count.
GPU_IDS=0,1 NUM_PROCESSES=2 bash scripts/train_sft_lora.sh

# Another maintained config, plus CLI overrides.
CONFIG_PATH=configs/full/sft.yaml \
bash scripts/train_sft_full.sh \
  --model-name-or-path Qwen/Qwen3-8B \
  --model-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --learning-rate 2e-5 \
  --max-length 4096
~~~

To chain a custom 8B full-tuning run, pass its completed output directory to
DPO. Existing local paths ignore the Hub revision stored in the YAML:

~~~bash
bash scripts/train_dpo_full.sh \
  --model-name-or-path /absolute/path/to/8b-sft-output
~~~

The 32B SFT recipe saves an adapter. Merge it into a stable full checkpoint
before using it as the base of a fresh DPO adapter:

~~~bash
uv run python ../rl/scripts/merge_lora.py \
  --adapter /absolute/path/to/32b-sft-adapter \
  --out /absolute/path/to/32b-sft-merged

bash scripts/train_dpo_lora.sh \
  --model-name-or-path /absolute/path/to/32b-sft-merged
~~~

Keep the merged SFT directory with the resulting local DPO adapter, or merge
the DPO adapter as well before moving or publishing it. For a different remote
model or dataset, override its path and revision together. `main` is accepted
when intentionally following a moving branch, but it is not reproducible.

Most YAML keys have a matching flag; `uv run python arcane/sft_train.py --help`
(or `arcane/dpo_train.py`) lists them. Qwen thinking mode is off by default, as
in the paper: `formatting.enable_thinking` / `--enable-thinking`.

Checkpoints go to a per-run directory below `training.output_root`, which
defaults to outputs. During SFT, an evaluation dataset also enables fixed-prompt
sample generations in `<run>/eval_samples/eval_responses_step<N>.json`.

## Rollouts

The rollout utility reads JSONL records with the ArcANE generation columns
(`gen_system`, `gen_user`, and optional `meta_data`). It does not read the
conversational `messages` column of the `sft` configuration.

~~~bash
# Base model
DATA_FILE=./data/test.jsonl \
MODEL_NAME_OR_PATH=Qwen/Qwen3-8B \
bash scripts/rollout.sh

# Trained checkpoint
DATA_FILE=./data/test.jsonl \
CHECKPOINT=outputs/model/dataset/lora/run/checkpoint-500 \
bash scripts/rollout.sh --max-samples 50
~~~

Outputs are stored under rollouts unless `OUTPUT_DIR` or `--output-file` is set.
