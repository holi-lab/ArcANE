#!/usr/bin/env python3
"""Resolve ArcANE RL parquet splits from a Hugging Face dataset repository."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "holi-lab/ArcANE-Data"
DEFAULT_CONFIG = "rl"
DEFAULT_REVISION = "e86e5630a32037e1605bca533f822401076f0354"


def download_split(
    repo_id: str,
    config: str,
    split: str,
    *,
    revision: str,
    cache_dir: str | None,
) -> str:
    filename = f"{config}/{split}.parquet"
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        cache_dir=cache_dir,
    )
    return str(Path(path).absolute())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and print the local paths of two RL parquet splits."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "dataset commit, tag, or branch (defaults to the pinned public "
            "release commit only for the default repository)"
        ),
    )
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    revision = args.revision
    if revision is None:
        if args.repo_id != DEFAULT_REPO_ID:
            parser.error("--revision is required when --repo-id is overridden")
        revision = DEFAULT_REVISION

    print(
        download_split(
            args.repo_id,
            args.config,
            args.train_split,
            revision=revision,
            cache_dir=args.cache_dir,
        )
    )
    print(
        download_split(
            args.repo_id,
            args.config,
            args.val_split,
            revision=revision,
            cache_dir=args.cache_dir,
        )
    )


if __name__ == "__main__":
    main()
