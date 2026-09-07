"""Rollout: generate model completions on a test dataset and save results."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from arcane.formatting import apply_chat_template_if_available


DEFAULT_DATA_FILE = "./data/test.jsonl"
DEFAULT_OUTPUT_DIR = "./rollouts"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_BATCH_SIZE = 1


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {exc}") from exc
    return records


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompt(example: dict[str, Any], tokenizer: AutoTokenizer, enable_thinking: bool) -> str:
    """Build the generation prompt from a test example.

    Prefers ``gen_system`` / ``gen_user`` fields (matching the existing rollout
    format) and falls back to the raw ``scenario`` / ``question`` fields.
    """
    system_text: str | None = example.get("gen_system") or None
    user_text: str | None = example.get("gen_user") or None

    if user_text is None:
        meta = example.get("meta_data", {})
        scenario = meta.get("scenario", "")
        question = meta.get("question", "")
        user_text = f"Scenario:\n{scenario}\n\nQuestion:\n{question}" if scenario else question

    messages: list[dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text or ""})

    prompt = apply_chat_template_if_available(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if prompt is None:
        # Fallback: plain text
        parts = []
        for msg in messages:
            parts.append(f"{msg['role']}: {msg['content']}")
        prompt = "\n\n".join(parts) + "\n\nassistant: "
    return prompt


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_name_or_path: str,
    torch_dtype: str = "auto",
    attn_implementation: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"[rollout] Loading tokenizer from {model_name_or_path!r} …")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, fix_mistral_regex=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batch generation

    print(f"[rollout] Loading model from {model_name_or_path!r} …")
    dtype_map: dict[str, Any] = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    resolved_dtype = dtype_map.get(torch_dtype, "auto")

    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolved_dtype,
        "trust_remote_code": True,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    do_sample: bool,
) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_lengths = inputs["input_ids"].shape[1]

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        if top_k > 0:
            gen_kwargs["top_k"] = top_k
        if min_p > 0.0:
            gen_kwargs["min_p"] = min_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Slice off the prompt tokens
    new_ids = output_ids[:, prompt_lengths:]
    completions = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
    return completions


# ---------------------------------------------------------------------------
# Main rollout loop
# ---------------------------------------------------------------------------


def run_rollout(
    model_name_or_path: str,
    data_file: Path,
    output_file: Path,
    *,
    torch_dtype: str = "auto",
    attn_implementation: str | None = None,
    enable_thinking: bool = False,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    do_sample: bool = True,
    max_samples: int | None = None,
    resume: bool = False,
) -> None:
    # Load data
    print(f"[rollout] Loading data from {data_file} …")
    examples = load_jsonl(data_file)
    if max_samples is not None:
        examples = examples[:max_samples]
    print(f"[rollout] {len(examples)} examples loaded.")

    # Resume: skip already-done indices
    done_indices: set[int] = set()
    existing_records: list[dict[str, Any]] = []
    if resume and output_file.exists():
        existing_records = load_jsonl(output_file)
        done_indices = {r["index"] for r in existing_records if "index" in r}
        print(f"[rollout] Resuming — {len(done_indices)} examples already done.")

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path, torch_dtype=torch_dtype, attn_implementation=attn_implementation
    )

    # Open output file (append if resuming, otherwise overwrite)
    mode = "a" if resume and output_file.exists() else "w"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_fh = output_file.open(mode, encoding="utf-8")

    try:
        pending = [
            (idx, ex)
            for idx, ex in enumerate(examples)
            if idx not in done_indices
        ]

        total = len(pending)
        with tqdm(total=total, unit="ex", desc="rollout", dynamic_ncols=True) as pbar:
            for batch_start in range(0, total, batch_size):
                batch = pending[batch_start : batch_start + batch_size]
                indices = [item[0] for item in batch]
                batch_examples = [item[1] for item in batch]

                prompts = [
                    build_prompt(ex, tokenizer, enable_thinking) for ex in batch_examples
                ]

                completions = generate_batch(
                    model,
                    tokenizer,
                    prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    do_sample=do_sample,
                )

                ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                for idx, ex, prompt, completion in zip(indices, batch_examples, prompts, completions):
                    record: dict[str, Any] = {
                        "index": idx,
                        "id": ex.get("id", ""),
                        "source": ex.get("source", ""),
                        "meta_data": ex.get("meta_data", {}),
                        "prompt": prompt,
                        "reference": ex.get("response", ex.get("completion", "")),
                        "completion": completion,
                        "ts": ts,
                        "model": model_name_or_path,
                    }
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_fh.flush()

                pbar.update(len(batch))
                pbar.set_postfix(idx=f"{indices[0]}–{indices[-1]}")
    finally:
        out_fh.close()

    print(f"[rollout] Saved to {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_output_file(
    model_name_or_path: str,
    output_dir: str,
    output_file: str | None,
) -> Path:
    if output_file:
        return Path(output_file)

    model_slug = Path(model_name_or_path).name.lower().replace("_", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rollout_{timestamp}.jsonl"
    return Path(output_dir) / model_slug / filename


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate model rollouts on a test dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model ---
    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--model-name-or-path",
        default=None,
        help=(
            "HuggingFace model ID or local path. "
            "For a fine-tuned checkpoint pass the checkpoint directory "
            "(e.g. ./outputs/<model>/<dataset>/full/<run>/checkpoint-100)."
        ),
    )
    model_group.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Shortcut: path to a checkpoint directory under ./outputs/. "
            "Equivalent to --model-name-or-path. Takes precedence if both are set."
        ),
    )
    model_group.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    model_group.add_argument("--attn-implementation", default=None)

    # --- Data ---
    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--data-file",
        default=DEFAULT_DATA_FILE,
        help="Path to the test JSONL file.",
    )
    data_group.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of examples (useful for debugging).",
    )

    # --- Output ---
    out_group = parser.add_argument_group("output")
    out_group.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write rollout JSONL files.",
    )
    out_group.add_argument(
        "--output-file",
        default=None,
        help="Explicit output file path (overrides --output-dir auto-naming).",
    )
    out_group.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Append to an existing output file, skipping already-completed indices.",
    )

    # --- Generation ---
    gen_group = parser.add_argument_group("generation")
    gen_group.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    gen_group.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    gen_group.add_argument("--temperature", type=float, default=0.7)
    gen_group.add_argument("--top-p", type=float, default=0.9)
    gen_group.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling (0 = disabled).",
    )
    gen_group.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Min-p sampling threshold (0.0 = disabled).",
    )
    gen_group.add_argument(
        "--greedy",
        action="store_true",
        default=False,
        help="Use greedy decoding (overrides --temperature / --top-p / --top-k / --min-p).",
    )
    gen_group.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass enable_thinking=True to chat templates that support thinking mode.",
    )

    args = parser.parse_args()

    # Resolve model path
    model_path: str | None = args.checkpoint or args.model_name_or_path
    if model_path is None:
        parser.error("Provide --model-name-or-path or --checkpoint.")

    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"[rollout] ERROR: data file not found: {data_file}", file=sys.stderr)
        sys.exit(1)

    output_file = resolve_output_file(model_path, args.output_dir, args.output_file)

    print(f"[rollout] model         : {model_path}")
    print(f"[rollout] data file     : {data_file}")
    print(f"[rollout] output file   : {output_file}")
    print(f"[rollout] max new tokens: {args.max_new_tokens}")
    print(f"[rollout] batch size    : {args.batch_size}")
    print(f"[rollout] greedy        : {args.greedy}")
    print(f"[rollout] temperature   : {args.temperature}")
    print(f"[rollout] top_p         : {args.top_p}")
    print(f"[rollout] top_k         : {args.top_k}")
    print(f"[rollout] min_p         : {args.min_p}")
    print(f"[rollout] enable_thinking: {args.enable_thinking}")

    run_rollout(
        model_name_or_path=model_path,
        data_file=data_file,
        output_file=output_file,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        enable_thinking=args.enable_thinking,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        do_sample=not args.greedy,
        max_samples=args.max_samples,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
