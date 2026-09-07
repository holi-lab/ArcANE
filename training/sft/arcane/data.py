from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset


def load_dataset_split(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    data_files: Any | None = None,
    revision: str | None = None,
) -> Dataset:
    if not dataset_name:
        raise ValueError("dataset_name must be set")

    local_path = Path(dataset_name)
    revision_kwargs = {} if local_path.exists() or revision is None else {"revision": revision}

    if data_files is not None:
        return load_dataset(
            dataset_name,
            dataset_config,
            data_files=data_files,
            split=split,
            **revision_kwargs,
        )

    if local_path.exists():
        return load_local_dataset_split(local_path, dataset_config, split)

    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split, **revision_kwargs)

    try:
        return load_dataset(dataset_name, split=split, **revision_kwargs)
    except Exception as default_error:
        configs = get_dataset_config_names(dataset_name, **revision_kwargs)
        if not configs or configs == ["default"]:
            raise default_error

        datasets = [
            load_dataset(dataset_name, config, split=split, **revision_kwargs)
            for config in configs
        ]
        return concatenate_datasets(datasets)


def load_local_dataset_split(dataset_path: Path, dataset_config: str | None, split: str) -> Dataset:
    data_files = resolve_local_data_files(dataset_path, dataset_config, split)
    if local_json_data_files(data_files):
        return load_dataset("json", data_files={split: data_files}, split=split)

    paths = [Path(data_files)] if isinstance(data_files, str) else [Path(path) for path in data_files]
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(read_json_records(path))
    return Dataset.from_list(records)


def local_json_data_files(data_files: str | list[str]) -> bool:
    paths = [Path(data_files)] if isinstance(data_files, str) else [Path(path) for path in data_files]
    return all(path.suffix in {".json", ".jsonl"} for path in paths)


def read_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number} must be a JSON object")
                records.append(value)
        return records

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        if not all(isinstance(item, dict) for item in value):
            raise TypeError(f"{path} must contain a list of JSON objects")
        return value
    if isinstance(value, dict):
        return [value]
    raise TypeError(f"{path} must contain a JSON object or a list of JSON objects")


def resolve_local_data_files(
    dataset_path: Path,
    dataset_config: str | None,
    split: str,
) -> str | list[str]:
    candidates: list[Path] = []
    search_roots = [dataset_path / dataset_config] if dataset_config else [dataset_path]

    for root in search_roots:
        candidates.extend(
            [
                root / f"{split}.jsonl",
                root / f"{split}.json",
                root / f"{split}-00000-of-00001.jsonl",
                root / f"{split}-00000-of-00001.json",
            ]
        )

    existing = [path for path in candidates if path.is_file()]
    if existing:
        return str(existing[0])

    root = search_roots[0]
    if root.is_file():
        return str(root)

    if root.is_dir():
        json_files = sorted(str(path) for path in root.glob("*.jsonl"))
        json_files.extend(str(path) for path in sorted(root.glob("*.json")))
        if json_files:
            return json_files

    config_hint = f"/{dataset_config}" if dataset_config else ""
    raise FileNotFoundError(
        f"Could not find split {split!r} under {dataset_path}{config_hint}. "
        "Expected files like train.jsonl and test.jsonl."
    )


def load_train_dataset(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    data_files: Any | None = None,
    revision: str | None = None,
) -> Dataset:
    return load_dataset_split(
        dataset_name,
        dataset_config,
        split,
        data_files,
        revision=revision,
    )
