from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_DATASET = "holi-lab/ArcANE-Data"
DEFAULT_DATASET_CONFIG = "sft"
DEFAULT_OUTPUT_ROOT = "outputs"


@dataclass
class TrainConfig:
    model_name_or_path: str = DEFAULT_MODEL
    model_revision: str | None = None
    torch_dtype: str = "auto"
    attn_implementation: str | None = None

    dataset_name: str = DEFAULT_DATASET
    dataset_revision: str | None = None
    dataset_config: str | None = DEFAULT_DATASET_CONFIG
    dataset_split: str = "train"
    dataset_data_files: Any | None = None
    dataset_num_proc: int = 4
    max_train_samples: int | None = None
    max_prompt_length: int | None = None

    enable_thinking: bool = False

    output_root: str = DEFAULT_OUTPUT_ROOT
    output_dir: str = ""
    max_length: int = 2048
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    seed: int = 42
    resume_from_checkpoint: str | None = None
    report_to: str = "wandb"
    assistant_only_loss: bool = False
    run_name: str = ""
    run_id: str | None = None
    packing: bool = False
    mixed_precision: str = "auto"

    dpo_beta: float = 0.1
    dpo_loss_type: str | list[str] = "sigmoid"
    dpo_loss_weights: list[float] | None = None
    dpo_label_smoothing: float = 0.0
    dpo_ld_alpha: float | None = None
    dpo_precompute_ref_log_probs: bool = False
    dpo_precompute_ref_batch_size: int | None = None
    dpo_truncation_mode: str = "keep_start"

    use_peft: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    eval_dataset_name: str | None = None
    eval_dataset_revision: str | None = None
    eval_dataset_config: str | None = None
    eval_dataset_split: str = "test"
    eval_dataset_data_files: Any | None = None
    max_eval_samples: int | None = 128
    trainer_eval_strategy: str = "no"
    trainer_eval_steps: int | None = None
    trainer_eval_delay: float = 0.0
    per_device_eval_batch_size: int = 1
    eval_accumulation_steps: int | None = None
    eval_on_start: bool = False


SECTION_KEYS = {
    "model": {
        "model_name_or_path",
        "model_revision",
        "torch_dtype",
        "attn_implementation",
    },
    "dataset": {
        "dataset_name",
        "dataset_revision",
        "dataset_config",
        "dataset_split",
        "dataset_data_files",
        "dataset_num_proc",
        "max_train_samples",
        "max_prompt_length",
    },
    "formatting": {"enable_thinking"},
    "evaluation": {
        "eval_dataset_name",
        "eval_dataset_revision",
        "eval_dataset_config",
        "eval_dataset_split",
        "eval_dataset_data_files",
        "max_eval_samples",
        "trainer_eval_strategy",
        "trainer_eval_steps",
        "trainer_eval_delay",
        "per_device_eval_batch_size",
        "eval_accumulation_steps",
        "eval_on_start",
    },
    "training": {
        "output_root",
        "max_length",
        "num_train_epochs",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "logging_steps",
        "save_steps",
        "save_total_limit",
        "seed",
        "resume_from_checkpoint",
        "report_to",
        "assistant_only_loss",
        "packing",
        "mixed_precision",
    },
    "dpo": {
        "dpo_beta",
        "dpo_loss_type",
        "dpo_loss_weights",
        "dpo_label_smoothing",
        "dpo_ld_alpha",
        "dpo_precompute_ref_log_probs",
        "dpo_precompute_ref_batch_size",
        "dpo_truncation_mode",
    },
    "peft": {"use_peft", "lora_r", "lora_alpha", "lora_dropout"},
}


SFT_REMOVED_CONFIG_KEYS = {
    "dataset": set(),
    "evaluation": {"trainer_eval_strategy", "trainer_eval_delay"},
    "training": {"weight_decay", "packing", "mixed_precision"},
}


def section_keys_for_task(task: str | None) -> dict[str, set[str]]:
    section_keys = {section: set(keys) for section, keys in SECTION_KEYS.items()}
    if task == "sft":
        for section, removed_keys in SFT_REMOVED_CONFIG_KEYS.items():
            section_keys[section] -= removed_keys
    return section_keys


def load_config(path: str | Path | None, task: str | None = None) -> TrainConfig:
    config = TrainConfig()
    if path is None:
        return config

    config_path = Path(path)
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path} must contain a YAML mapping")

    values: dict[str, Any] = {}
    for section, allowed_keys in section_keys_for_task(task).items():
        section_values = raw.pop(section, {})
        if not isinstance(section_values, dict):
            raise TypeError(f"{section} must be a mapping in {config_path}")
        unknown = sorted(set(section_values) - allowed_keys)
        if unknown:
            joined = ", ".join(str(key) for key in unknown)
            raise KeyError(f"Unknown keys in {section} of {config_path}: {joined}")
        values.update(section_values)

    if raw:
        joined = ", ".join(str(key) for key in sorted(raw))
        raise KeyError(f"Unknown top-level sections in {config_path}: {joined}")

    return dataclasses.replace(config, **values)


def apply_overrides(config: TrainConfig, overrides: dict[str, Any]) -> TrainConfig:
    values = {key: value for key, value in overrides.items() if value is not None}
    if not values:
        return config
    return dataclasses.replace(config, **values)


def timestamped_name(name: str, run_id: str) -> str:
    return f"{name}-{run_id}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", value.strip().lower()).strip("-")
    return slug or "run"


def model_slug(model_name_or_path: str) -> str:
    return slugify(Path(model_name_or_path).name)


def dataset_slug(dataset_name: str, dataset_config: str | None) -> str:
    slug = slugify(Path(dataset_name).name.replace("_", "-"))
    if dataset_config:
        slug = f"{slug}-{slugify(dataset_config)}"
    return slug


def tuning_slug(config: TrainConfig) -> str:
    return "lora" if config.use_peft else "full"


def compact_float(value: float) -> str:
    abs_value = abs(value)
    if value != 0.0 and (abs_value < 1e-3 or abs_value >= 1e3):
        formatted = f"{value:.2e}"
        mantissa, exponent = formatted.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        sign = exponent[0]
        digits = exponent[1:].lstrip("0") or "0"
        return f"{mantissa}e{sign}{digits}".replace(".", "p")

    formatted = f"{value:.4g}"
    if "e" in formatted:
        mantissa, exponent = formatted.split("e")
        sign = exponent[0]
        digits = exponent[1:].lstrip("0") or "0"
        formatted = f"{mantissa}e{sign}{digits}"
    return formatted.replace(".", "p")


def effective_train_batch_size(config: TrainConfig) -> int:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world_size = 1
    world_size = max(world_size, 1)
    return (
        config.per_device_train_batch_size
        * config.gradient_accumulation_steps
        * world_size
    )


def run_params_suffix(config: TrainConfig) -> str:
    return (
        f"ebs{effective_train_batch_size(config)}"
        f"-seq{config.max_length}"
        f"-lr{compact_float(config.learning_rate)}"
    )


def checkpoint_output_dir(checkpoint: str) -> str:
    path = Path(checkpoint)
    if path.name.startswith("checkpoint-"):
        return str(path.parent)
    return str(path)


def finalize_run_config(config: TrainConfig, task: str) -> TrainConfig:
    if config.resume_from_checkpoint:
        output_dir = checkpoint_output_dir(config.resume_from_checkpoint)
        run_name = Path(output_dir).name
        return dataclasses.replace(config, output_dir=output_dir, run_name=run_name)

    run_id = config.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    params_suffix = run_params_suffix(config)
    model = model_slug(config.model_name_or_path)
    benchmark = dataset_slug(config.dataset_name, config.dataset_config)
    tuning = tuning_slug(config)
    base_run_name = f"{model}-{benchmark}-{task}-{tuning}"
    run_name = timestamped_name(f"{base_run_name}-{params_suffix}", run_id)
    output_leaf = f"{task}-{params_suffix}"
    output_dir = str(
        Path(config.output_root or DEFAULT_OUTPUT_ROOT)
        / model
        / benchmark
        / tuning
        / timestamped_name(output_leaf, run_id)
    )
    return dataclasses.replace(config, run_id=run_id, run_name=run_name, output_dir=output_dir)


def as_dict(config: TrainConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)
