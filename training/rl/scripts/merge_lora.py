#!/usr/bin/env python3
"""Merge a custom PEFT LoRA adapter into its base model and save full HF weights.

The public ArcANE-32B-DPO checkpoint is already a complete, merged model, so it
does not need this script. Use this only when starting from a custom DPO
adapter. `run_grpo.sh` attaches a fresh LoRA adapter to `BASE_MODEL`, and verl
disables that active adapter when computing reference log probabilities. A
custom DPO adapter must therefore be merged first so the KL reference remains
anchored to the intended DPO policy.

  python scripts/merge_lora.py \
      --adapter outputs/dpo/checkpoint-500 \
      --out     outputs/dpo-merged
"""
import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def hub_revision_kwargs(name_or_path: str, revision: str | None) -> dict[str, str]:
    """Apply a Hub revision only when the source is not a local path."""
    if revision is None or Path(name_or_path).expanduser().exists():
        return {}
    return {"revision": revision}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="dir with adapter_config.json + adapter_model.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None, help="override base model (default: read from adapter_config.json)")
    ap.add_argument(
        "--base-revision",
        default=None,
        help="base-model commit/tag/branch (default: revision saved in adapter_config.json)",
    )
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    with open(os.path.join(args.adapter, "adapter_config.json")) as f:
        adapter_config = json.load(f)
    saved_base = adapter_config["base_model_name_or_path"]
    base = args.base or saved_base
    base_changed = args.base is not None and args.base != saved_base
    if args.base_revision is not None:
        base_revision = args.base_revision
    elif base_changed:
        base_revision = None
    else:
        base_revision = adapter_config.get("revision")

    if base_changed and not Path(base).expanduser().exists() and base_revision is None:
        ap.error(
            "--base-revision is required when --base overrides the saved Hub repository"
        )

    base_revision_kwargs = hub_revision_kwargs(base, base_revision)
    revision_label = base_revision_kwargs.get("revision", "local/default")
    print(
        f"[merge] base={base}  base_revision={revision_label}  "
        f"adapter={args.adapter}"
    )

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=dtype,
        device_map="cpu",
        **base_revision_kwargs,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")
    # prefer the adapter's tokenizer (it may carry added tokens / chat template)
    tok_src = args.adapter if os.path.exists(os.path.join(args.adapter, "tokenizer_config.json")) else base
    tokenizer_revision_kwargs = {} if tok_src == args.adapter else base_revision_kwargs
    AutoTokenizer.from_pretrained(
        tok_src,
        **tokenizer_revision_kwargs,
    ).save_pretrained(args.out)
    print(f"[merge] wrote merged model -> {args.out}")


if __name__ == "__main__":
    main()
