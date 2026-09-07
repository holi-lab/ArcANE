#!/usr/bin/env python3
"""Resolve a Hugging Face model snapshot at an explicit revision."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "holi-lab/ArcANE-32B-DPO"
DEFAULT_REVISION = "9b1dc3d21e44692cf1b5edd0ab55af499696c059"


def download_model(repo_id: str, revision: str, cache_dir: str | None) -> str:
    """Download or reuse a complete model snapshot and return its local path."""
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
    )
    return str(Path(path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a complete model snapshot and print its local path."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "model commit, tag, or branch (defaults to the pinned "
            "ArcANE-32B-DPO commit only for the default repository)"
        ),
    )
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    revision = args.revision
    if revision is None:
        if args.repo_id != DEFAULT_REPO_ID:
            parser.error("--revision is required when --repo-id is overridden")
        revision = DEFAULT_REVISION

    print(download_model(args.repo_id, revision, args.cache_dir))


if __name__ == "__main__":
    main()
