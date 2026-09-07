from __future__ import annotations

from datasets import Dataset
from transformers import set_seed
from trl import DPOConfig, DPOTrainer

from arcane.config import TrainConfig
from arcane.formatting import (
    CHOSEN_COLUMNS,
    apply_chat_template_if_available,
    first_present_value_with_name,
    validate_messages,
)
from arcane.modeling import build_peft_config, load_model, load_tokenizer, resolve_precision
from arcane.trainer_utils import select_eval_dataset, select_train_dataset


def drop_metadata(dataset: Dataset) -> Dataset:
    if "metadata" in dataset.column_names:
        dataset = dataset.remove_columns(["metadata"])
    return dataset


def prompt_messages_from_conversation(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    last_assistant_index = None
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            last_assistant_index = index

    if last_assistant_index is None:
        return messages
    return messages[:last_assistant_index]


def prompt_token_count(example: dict, tokenizer: object, enable_thinking: bool) -> int:
    messages, _ = first_present_value_with_name(example, CHOSEN_COLUMNS)
    prompt_messages = prompt_messages_from_conversation(validate_messages(messages))
    prompt = apply_chat_template_if_available(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if prompt is None:
        prompt = "\n\n".join(f"{message['role']}: {message['content']}" for message in prompt_messages)
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def filter_by_max_prompt_length(
    dataset: Dataset,
    tokenizer: object,
    enable_thinking: bool,
    max_prompt_length: int | None,
    *,
    desc_prefix: str,
    num_proc: int | None,
) -> Dataset:
    if max_prompt_length is None:
        return dataset

    before = len(dataset)
    if before == 0:
        print(f"{desc_prefix}: max_prompt_length={max_prompt_length} dropped=0/0 (0.00%) kept=0")
        return dataset

    map_kwargs = {"num_proc": num_proc} if num_proc and num_proc > 1 else {}
    measured = dataset.map(
        lambda example: {"_prompt_token_length": prompt_token_count(example, tokenizer, enable_thinking)},
        desc=f"Measuring {desc_prefix} prompt lengths",
        **map_kwargs,
    )
    filtered = measured.filter(
        lambda example: example["_prompt_token_length"] <= max_prompt_length,
        desc=f"Filtering {desc_prefix} by max prompt length",
        **map_kwargs,
    )
    filtered = filtered.remove_columns(["_prompt_token_length"])

    after = len(filtered)
    dropped = before - after
    print(
        f"{desc_prefix}: max_prompt_length={max_prompt_length} "
        f"dropped={dropped}/{before} ({dropped / before:.2%}) kept={after}"
    )
    if after == 0:
        raise ValueError(
            f"{desc_prefix} dataset is empty after max_prompt_length={max_prompt_length} filtering"
        )
    return filtered


def prepare_dpo_dataset(dataset: Dataset, tokenizer: object, config: TrainConfig, *, desc_prefix: str) -> Dataset:
    dataset = drop_metadata(dataset)
    dataset = filter_by_max_prompt_length(
        dataset,
        tokenizer,
        config.enable_thinking,
        config.max_prompt_length,
        desc_prefix=desc_prefix,
        num_proc=config.dataset_num_proc,
    )

    # Match the SFT pipeline: tag each example so the Qwen3 chat template applied by
    # DPOTrainer respects enable_thinking instead of defaulting to thinking mode.
    enable_thinking = config.enable_thinking
    num_proc = config.dataset_num_proc
    map_kwargs = {"num_proc": num_proc} if num_proc and num_proc > 1 else {}
    return dataset.map(
        lambda example: {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        desc=f"Tagging {desc_prefix} chat_template_kwargs",
        **map_kwargs,
    )


def prepare_train_dataset(config: TrainConfig, tokenizer: object) -> Dataset:
    return prepare_dpo_dataset(select_train_dataset(config), tokenizer, config, desc_prefix="train")


def prepare_trainer_eval_dataset(config: TrainConfig, tokenizer: object) -> Dataset | None:
    dataset = select_eval_dataset(config)
    if dataset is None:
        return None
    return prepare_dpo_dataset(dataset, tokenizer, config, desc_prefix="eval")


def build_dpo_config(config: TrainConfig) -> DPOConfig:
    bf16, fp16 = resolve_precision(config.mixed_precision)
    return DPOConfig(
        output_dir=config.output_dir,
        max_length=config.max_length,
        truncation_mode=config.dpo_truncation_mode,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=bf16,
        fp16=fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_strategy="steps",
        eval_strategy=config.trainer_eval_strategy,
        eval_steps=config.trainer_eval_steps,
        eval_delay=config.trainer_eval_delay,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        eval_accumulation_steps=config.eval_accumulation_steps,
        eval_on_start=config.eval_on_start,
        report_to=config.report_to,
        run_name=config.run_name,
        seed=config.seed,
        dataset_num_proc=config.dataset_num_proc,
        beta=config.dpo_beta,
        loss_type=config.dpo_loss_type,
        loss_weights=config.dpo_loss_weights,
        label_smoothing=config.dpo_label_smoothing,
        ld_alpha=config.dpo_ld_alpha,
        precompute_ref_log_probs=config.dpo_precompute_ref_log_probs,
        precompute_ref_batch_size=config.dpo_precompute_ref_batch_size,
    )


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    tokenizer = load_tokenizer(config)
    train_dataset = prepare_train_dataset(config, tokenizer)
    eval_dataset = prepare_trainer_eval_dataset(config, tokenizer)
    model = load_model(config)

    trainer = DPOTrainer(
        model=model,
        args=build_dpo_config(config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=build_peft_config(config),
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
