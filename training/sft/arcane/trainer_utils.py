from __future__ import annotations

from typing import Any

from datasets import Dataset

from arcane.config import TrainConfig
from arcane.data import load_dataset_split, load_train_dataset


def select_train_dataset(config: TrainConfig) -> Dataset:
    dataset = load_train_dataset(
        config.dataset_name,
        config.dataset_config,
        config.dataset_split,
        config.dataset_data_files,
        revision=config.dataset_revision,
    )
    return maybe_limit_dataset(dataset, config.max_train_samples)


def select_eval_dataset(config: TrainConfig) -> Dataset | None:
    if config.trainer_eval_strategy == "no":
        return None

    dataset_name = config.eval_dataset_name or config.dataset_name
    dataset_config = (
        config.eval_dataset_config
        if config.eval_dataset_config is not None
        else config.dataset_config
    )
    data_files: Any | None
    if config.eval_dataset_data_files is not None:
        data_files = config.eval_dataset_data_files
    elif config.eval_dataset_name is None:
        data_files = config.dataset_data_files
    else:
        data_files = None

    if config.eval_dataset_revision is not None:
        revision = config.eval_dataset_revision
    elif config.eval_dataset_name is None:
        revision = config.dataset_revision
    else:
        revision = None

    dataset = load_dataset_split(
        dataset_name,
        dataset_config,
        config.eval_dataset_split,
        data_files,
        revision=revision,
    )
    return maybe_limit_dataset(dataset, config.max_eval_samples)


def maybe_limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))
