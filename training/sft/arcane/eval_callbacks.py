"""Trainer callbacks for inspecting evaluation behaviour during training."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import TrainerCallback

from arcane.formatting import (
    MESSAGES_COLUMNS,
    PROBLEM_COLUMNS,
    SOLUTION_COLUMNS,
    apply_chat_template_if_available,
    first_present,
    first_present_value_with_name,
    validate_messages,
)


DEFAULT_NUM_SAMPLES = 5
DEFAULT_MAX_NEW_TOKENS = 512

# Qwen3 recommended sampling settings, keyed by whether thinking mode is enabled.
THINKING_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
NON_THINKING_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


def _last_assistant_content(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message["content"]
    return ""


def _prompt_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    last_assistant_index = None
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            last_assistant_index = index
    if last_assistant_index is None:
        return messages
    return messages[:last_assistant_index]


def build_eval_prompt(
    example: dict,
    tokenizer: Any,
    enable_thinking: bool,
    conversational: bool,
) -> dict[str, str]:
    """Build the generation prompt and reference answer for one raw eval example."""
    if conversational:
        messages = validate_messages(first_present_value_with_name(example, MESSAGES_COLUMNS)[0])
        prompt_messages = _prompt_messages(messages)
        reference = _last_assistant_content(messages)
    else:
        prompt_messages = [{"role": "user", "content": first_present(example, PROBLEM_COLUMNS)}]
        reference = first_present(example, SOLUTION_COLUMNS)

    prompt = apply_chat_template_if_available(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if prompt is None:
        prompt = "\n\n".join(f"{message['role']}: {message['content']}" for message in prompt_messages)
    return {"prompt": prompt, "reference": reference}


def select_eval_samples(
    raw_eval_dataset: Dataset,
    tokenizer: Any,
    enable_thinking: bool,
    conversational: bool,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Randomly pick ``num_samples`` raw eval examples and build their prompts."""
    total = len(raw_eval_dataset)
    if total == 0:
        return []

    indices = random.Random(seed).sample(range(total), min(num_samples, total))
    samples = []
    for index in indices:
        built = build_eval_prompt(raw_eval_dataset[index], tokenizer, enable_thinking, conversational)
        samples.append({"index": index, **built})
    return samples


class SaveEvalResponsesCallback(TrainerCallback):
    """Generate and save model responses for a fixed set of prompts at every evaluation.

    The prompt set is chosen once (see :func:`select_eval_samples`) and never changes,
    so the per-step files can be compared side by side to follow how each response
    evolves over training. Paired with ``eval_on_start`` the first file captures the
    baseline before any optimizer step.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer: Any,
        output_dir: str,
        *,
        enable_thinking: bool,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.max_new_tokens = max_new_tokens
        self.sampling = THINKING_SAMPLING if enable_thinking else NON_THINKING_SAMPLING

    def on_evaluate(self, args, state, control, **kwargs):  # noqa: ANN001
        if not self.samples or not state.is_world_process_zero:
            return
        model = kwargs.get("model")
        if model is None:
            return

        records = self._generate(model)
        self._save(records, state.global_step)

    def _generate(self, model: Any) -> list[dict[str, Any]]:
        target = model.module if hasattr(model, "module") else model
        device = next(target.parameters()).device
        was_training = target.training
        prev_use_cache = getattr(target.config, "use_cache", None)
        target.eval()
        if hasattr(target, "config"):
            target.config.use_cache = True

        records: list[dict[str, Any]] = []
        try:
            for sample in self.samples:
                inputs = self.tokenizer(sample["prompt"], return_tensors="pt").to(device)
                with torch.no_grad():
                    output_ids = target.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        **self.sampling,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
                response = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                records.append(
                    {
                        "index": sample["index"],
                        "prompt": sample["prompt"],
                        "reference": sample["reference"],
                        "response": response,
                    }
                )
        finally:
            if prev_use_cache is not None and hasattr(target, "config"):
                target.config.use_cache = prev_use_cache
            target.train(was_training)
        return records

    def _save(self, records: list[dict[str, Any]], global_step: int) -> None:
        out_dir = self.output_dir / "eval_samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"eval_responses_step{global_step}.json"
        with out_file.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
        print(f"[eval] Saved {len(records)} sample responses to {out_file}")
