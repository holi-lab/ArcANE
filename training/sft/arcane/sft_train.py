from __future__ import annotations

import argparse

from datasets import Dataset
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from arcane.config import apply_overrides, finalize_run_config, load_config
from arcane.formatting import (
    MESSAGES_COLUMNS,
    PROBLEM_COLUMNS,
    apply_chat_template_if_available,
    first_present,
    first_present_value_with_name,
    format_conversational_sft_example,
    format_text_sft_example,
    validate_messages,
)
from arcane.modeling import build_peft_config, load_model, load_tokenizer
from arcane.data import load_dataset_split
from arcane.eval_callbacks import SaveEvalResponsesCallback, select_eval_samples
from arcane.trainer_utils import select_train_dataset


def is_conversational_sft_dataset(dataset: Dataset) -> bool:
    if not dataset.column_names:
        return False

    example = next(iter(dataset))
    try:
        messages, _ = first_present_value_with_name(example, MESSAGES_COLUMNS)
        validate_messages(messages)
    except (KeyError, TypeError):
        return False
    return True


def prompt_messages_from_conversation(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    last_assistant_index = None
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            last_assistant_index = index

    if last_assistant_index is None:
        return messages
    return messages[:last_assistant_index]


def token_count(text: str, tokenizer: object) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def conversational_prompt_token_count(
    example: dict,
    tokenizer: object,
    enable_thinking: bool,
) -> int:
    messages, _ = first_present_value_with_name(example, MESSAGES_COLUMNS)
    prompt_messages = prompt_messages_from_conversation(validate_messages(messages))
    prompt = apply_chat_template_if_available(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if prompt is None:
        prompt = "\n\n".join(f"{message['role']}: {message['content']}" for message in prompt_messages)
    return token_count(prompt, tokenizer)


def text_prompt_token_count(example: dict, tokenizer: object) -> int:
    return token_count(first_present(example, PROBLEM_COLUMNS), tokenizer)


def filter_by_max_prompt_length(
    dataset: Dataset,
    tokenizer: object,
    enable_thinking: bool,
    max_prompt_length: int | None,
    *,
    desc_prefix: str,
    num_proc: int | None,
    conversational: bool,
) -> Dataset:
    if max_prompt_length is None:
        return dataset

    before = len(dataset)
    if before == 0:
        print(f"{desc_prefix}: max_prompt_length={max_prompt_length} dropped=0/0 (0.00%) kept=0")
        return dataset

    map_kwargs = {"num_proc": num_proc} if num_proc and num_proc > 1 else {}
    length_fn = conversational_prompt_token_count if conversational else text_prompt_token_count
    length_dataset = dataset.map(
        lambda example: {
            "_prompt_token_length": length_fn(example, tokenizer, enable_thinking)
            if conversational
            else length_fn(example, tokenizer)
        },
        desc=f"Measuring {desc_prefix} prompt lengths",
        **map_kwargs,
    )
    filtered = length_dataset.filter(
        lambda example: example["_prompt_token_length"] <= max_prompt_length,
        desc=f"Filtering {desc_prefix} by max prompt length",
        **map_kwargs,
    )
    filtered = filtered.remove_columns(["_prompt_token_length"])

    after = len(filtered)
    dropped = before - after
    dropped_ratio = dropped / before
    print(
        f"{desc_prefix}: max_prompt_length={max_prompt_length} "
        f"dropped={dropped}/{before} ({dropped_ratio:.2%}) kept={after}"
    )
    if after == 0:
        raise ValueError(
            f"{desc_prefix} dataset is empty after max_prompt_length={max_prompt_length} filtering"
        )
    return filtered


def prepare_sft_dataset(
    dataset: Dataset,
    tokenizer: object,
    enable_thinking: bool,
    *,
    desc_prefix: str,
    num_proc: int | None,
    max_prompt_length: int | None,
) -> Dataset:
    map_kwargs = {"num_proc": num_proc} if num_proc and num_proc > 1 else {}
    conversational = is_conversational_sft_dataset(dataset)
    dataset = filter_by_max_prompt_length(
        dataset,
        tokenizer,
        enable_thinking,
        max_prompt_length,
        desc_prefix=desc_prefix,
        num_proc=num_proc,
        conversational=conversational,
    )

    if conversational:
        return dataset.map(
            lambda example: format_conversational_sft_example(
                first_present_value_with_name(example, MESSAGES_COLUMNS)[0],
                enable_thinking,
            ),
            remove_columns=dataset.column_names,
            desc=f"Formatting {desc_prefix} conversational examples",
            **map_kwargs,
        )

    return dataset.map(
        lambda example: format_text_sft_example(example, tokenizer, enable_thinking),
        remove_columns=dataset.column_names,
        desc=f"Formatting {desc_prefix} text examples",
        **map_kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)

    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--torch-dtype", choices=("auto", "bfloat16", "float16", "float32"), default=None)
    parser.add_argument("--attn-implementation", default=None)

    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--dataset-data-files", default=None)
    parser.add_argument("--dataset-num-proc", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=None)

    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--report-to", default=None)
    parser.add_argument("--assistant-only-loss", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--use-peft", "--peft", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)

    parser.add_argument("--eval-dataset-name", default=None)
    parser.add_argument("--eval-dataset-revision", default=None)
    parser.add_argument("--eval-dataset-config", default=None)
    parser.add_argument("--eval-dataset-split", default=None)
    parser.add_argument("--eval-dataset-data-files", default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--trainer-eval-steps", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-accumulation-steps", type=int, default=None)
    parser.add_argument("--eval-on-start", action=argparse.BooleanOptionalAction, default=None)

    args = parser.parse_args()
    config = load_config(args.config, task="sft")
    overrides = vars(args).copy()
    overrides.pop("config")
    config = finalize_run_config(apply_overrides(config, overrides), task="sft")

    set_seed(config.seed)
    tokenizer = load_tokenizer(config)

    raw_train_dataset = select_train_dataset(config)
    train_dataset = prepare_sft_dataset(
        raw_train_dataset,
        tokenizer,
        config.enable_thinking,
        desc_prefix="train",
        num_proc=config.dataset_num_proc,
        max_prompt_length=config.max_prompt_length,
    )

    raw_eval_dataset = None
    has_eval_config = (
        config.eval_dataset_name is not None
        or config.eval_dataset_config is not None
        or config.eval_dataset_data_files is not None
        or config.trainer_eval_steps is not None
    )
    if has_eval_config:
        eval_dataset_name = config.eval_dataset_name or config.dataset_name
        eval_dataset_config = (
            config.eval_dataset_config
            if config.eval_dataset_config is not None
            else config.dataset_config
        )
        if config.eval_dataset_data_files is not None:
            eval_data_files = config.eval_dataset_data_files
        elif config.eval_dataset_name is None:
            eval_data_files = config.dataset_data_files
        else:
            eval_data_files = None
        if config.eval_dataset_revision is not None:
            eval_revision = config.eval_dataset_revision
        elif config.eval_dataset_name is None:
            eval_revision = config.dataset_revision
        else:
            eval_revision = None
        raw_eval_dataset = load_dataset_split(
            eval_dataset_name,
            eval_dataset_config,
            config.eval_dataset_split,
            eval_data_files,
            revision=eval_revision,
        )
        if config.max_eval_samples is not None:
            raw_eval_dataset = raw_eval_dataset.select(
                range(min(config.max_eval_samples, len(raw_eval_dataset)))
            )

    eval_dataset = None
    eval_callbacks = []
    if raw_eval_dataset is not None:
        eval_dataset = prepare_sft_dataset(
            raw_eval_dataset,
            tokenizer,
            config.enable_thinking,
            desc_prefix="eval",
            num_proc=config.dataset_num_proc,
            max_prompt_length=config.max_prompt_length,
        )
        eval_response_samples = select_eval_samples(
            raw_eval_dataset,
            tokenizer,
            config.enable_thinking,
            is_conversational_sft_dataset(raw_eval_dataset),
            seed=config.seed,
        )
        if eval_response_samples:
            eval_callbacks.append(
                SaveEvalResponsesCallback(
                    eval_response_samples,
                    tokenizer,
                    config.output_dir,
                    enable_thinking=config.enable_thinking,
                )
            )

    model = load_model(config)
    trainer_args = SFTConfig(
        output_dir=config.output_dir,
        max_length=config.max_length,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config.trainer_eval_steps,
        eval_on_start=config.eval_on_start,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        eval_accumulation_steps=config.eval_accumulation_steps,
        report_to=config.report_to,
        assistant_only_loss=config.assistant_only_loss,
        dataset_num_proc=config.dataset_num_proc,
        run_name=config.run_name,
        seed=config.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=build_peft_config(config),
        callbacks=eval_callbacks or None,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
