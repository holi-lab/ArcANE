# ArcANE Training

The **SFT → DPO → RLVR** stack behind `ArcANE-8B` and `ArcANE-32B`, part of the
[ArcANE](../README.md) release. Training data:
[holi-lab/ArcANE-Data](https://huggingface.co/datasets/holi-lab/ArcANE-Data).
All five model checkpoints are available in the
[ArcANE collection](https://huggingface.co/collections/holi-lab/arcane).
They are full checkpoints, including the already-merged 32B SFT, DPO, and
RLVR weights. Inference does not require running these training recipes.
Published input revisions are pinned in the recipes and recorded in
[hf_release_manifest.json](../hf_release_manifest.json).

| Directory | Stage | Environment |
| --- | --- | --- |
| [`sft/`](sft/) | Supervised fine-tuning and DPO preference tuning | `uv` (`sft/pyproject.toml`) |
| [`rl/`](rl/) | RLVR with GRPO on a pinned [verl](https://github.com/verl-project/verl) checkout | conda (`rl/scripts/install_env.sh`) |

## SFT and DPO

~~~bash
cd sft
uv sync
uv pip install flash-attn --no-build-isolation   # the configs request flash_attention_2,
# which uv does not install; without a CUDA toolchain, pass --attn-implementation sdpa

bash scripts/train_sft_lora.sh    # 32B / LoRA   (full-tuning: train_sft_full.sh → 8B)
bash scripts/train_dpo_lora.sh    # 32B / LoRA   (full-tuning: train_dpo_full.sh → 8B)
~~~

The DPO configs start from pinned revisions of the published
`holi-lab/ArcANE-{8B,32B}-SFT` checkpoints. To chain your own SFT run, use the
full-model or local-LoRA instructions in [sft/README.md](sft/README.md).

Configs, multi-GPU and rollouts: [sft/README.md](sft/README.md).

## RLVR

Applied to **ArcANE-32B only**; there is no 8B counterpart. The launcher defaults
to the pinned, already-merged
[ArcANE-32B-DPO](https://huggingface.co/holi-lab/ArcANE-32B-DPO) checkpoint.
Use `rl/scripts/merge_lora.py` only if starting from your own local LoRA adapter.
The final [ArcANE-32B-RLVR](https://huggingface.co/holi-lab/ArcANE-32B-RLVR)
checkpoint is also published and can be loaded directly for inference.

Normal training requires a reward service at `ARCANE_REWARD_URL`.
`SMOKE=1` defaults to a heuristic reward for pipeline checks, not the paper's reward.

~~~bash
cd rl

git clone https://github.com/verl-project/verl.git third_party/verl
git -C third_party/verl checkout 081df509ca9fe02d524f913a7a3f96eefe9ca568

bash scripts/install_env.sh
conda activate ./.conda/verl        # prefix env created in-tree by install_env.sh

ARCANE_REWARD_MODE=remote \
ARCANE_REWARD_URL=http://127.0.0.1:8000/score bash scripts/run_grpo.sh
~~~

Reward interface and runtime knobs: [rl/README.md](rl/README.md).

## Training Configuration

Values reported for the original checkpoints (paper Appendix D, Table 8).
`configs/full/` is the 8B recipe and `configs/lora/` the 32B one.

| | **ArcANE-8B** | **ArcANE-32B** |
| --- | --- | --- |
| Base model | Qwen3-8B | Qwen3-32B |
| Fine-tuning | Full | LoRA (r=64, α=128) |
| **SFT** — epochs | 1 | 1 |
| SFT — learning rate | 1e-5 | 1e-4 |
| SFT — effective batch | 64 | 32 |
| **DPO** — epochs | 1 | 1 |
| DPO — learning rate | 5e-6 | 1e-5 |
| DPO — batch | 64 | 64 |
| Max length | 8192 | 8192 |
| **RLVR (GRPO)** — steps / epochs | — | 86 / 2 |
| RLVR — learning rate | — | 1e-5 |
| RLVR — KL β / clip ε | — | 0.001 / 0.2 |
| RLVR — train / mini batch | — | 64 / 32 |
| RLVR — rollouts per prompt | — | 8 |
| RLVR — prompt / response length | — | 8192 / 2048 |

For SFT and DPO, effective batch is `per_device_train_batch_size ×
gradient_accumulation_steps × NUM_PROCESSES`, and the YAMLs assume the paper's
1 × B200 192GB setup: on more GPUs, raise `NUM_PROCESSES` and lower the
accumulation steps to match. The RLVR hyperparameters are exposed by
`run_grpo.sh`; the number of steps for two epochs changes with the public
subset's size and prompt-length filtering.

## Dataset Configurations

[holi-lab/ArcANE-Data](https://huggingface.co/datasets/holi-lab/ArcANE-Data)
carries three configurations, each with a `train` and a `test` split:

| Configuration | Train | Test | Used by | Schema |
| --- | ---: | ---: | --- | --- |
| `sft` | 20,560 | 2,285 | `sft/arcane/sft_train.py` | conversational `messages` |
| `dpo` | 12,746 | 375 | `sft/arcane/dpo_train.py` | conversational `chosen` / `rejected` |
| `rl` | 2,513 | 678 | `rl/scripts/run_grpo.sh` | verl `RLHFDataset` columns |

These splits cover ten novels, excluding *East Lynne* and *The Underdogs* from
the twelve-novel training pool used for the published checkpoints. The `rl`
configuration contains policy prompts and reward references.

From the SFT environment, load the pinned release with:

~~~python
from datasets import load_dataset

repo_id = "holi-lab/ArcANE-Data"
revision = "e86e5630a32037e1605bca533f822401076f0354"
sft = load_dataset(repo_id, "sft", revision=revision)
dpo = load_dataset(repo_id, "dpo", revision=revision)
rl = load_dataset(repo_id, "rl", revision=revision)
print({name: {split: len(rows) for split, rows in data.items()}
       for name, data in {"sft": sft, "dpo": dpo, "rl": rl}.items()})
~~~
