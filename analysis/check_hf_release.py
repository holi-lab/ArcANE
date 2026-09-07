"""Check the pinned public Hugging Face release without downloading weights.

Run ``python -m analysis.check_hf_release`` from the repository root. Requests
are anonymous and read only. The check inspects repository metadata, dataset
split declarations and LFS hashes, model configs, and weight indexes. It does
not download or hash file contents, validate parquet rows, or run models.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote
from urllib.request import Request, urlopen


HF_URL = "https://huggingface.co"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "hf_release_manifest.json"


def _get_json(url: str, timeout: float) -> dict:
    request = Request(url, headers={"User-Agent": "ArcANE-public-release-check/1"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _check_repo(kind: str, entry: dict, timeout: float) -> str:
    repo_id, revision = entry["repo_id"], entry["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{repo_id}: revision must be an immutable commit SHA")
    repo_path = quote(repo_id, safe="/")
    plural = "datasets" if kind == "dataset" else "models"
    info = _get_json(f"{HF_URL}/api/{plural}/{repo_path}/revision/{revision}?blobs=true", timeout)
    if info.get("private") is not False or info.get("gated") not in (False, None):
        raise ValueError(f"{repo_id}: repository is private or gated")
    if info.get("sha") != revision:
        raise ValueError(f"{repo_id}: resolved commit differs from the manifest")
    file_info = {item["rfilename"]: item for item in info["siblings"]}
    files = set(file_info)
    required = {"README.md"}
    if kind == "dataset":
        required.update(
            f"{config}/{split}.parquet"
            for config, splits in entry["configs"].items()
            for split in splits
        )
        declared = {
            config["config_name"]: {
                split["split"]: split["path"] for split in config["data_files"]
            }
            for config in info.get("cardData", {}).get("configs", [])
        }
        expected = {
            config: {split: f"{config}/{split}.parquet" for split in splits}
            for config, splits in entry["configs"].items()
        }
        if declared != expected:
            raise ValueError(f"{repo_id}: dataset card configurations differ from the manifest")
        for filename, expected_file in entry.get("files", {}).items():
            actual_file = file_info.get(filename, {})
            if (actual_file.get("size") != expected_file["size_bytes"]
                    or actual_file.get("lfs", {}).get("sha256") != expected_file["sha256"]):
                raise ValueError(f"{repo_id}: size or LFS hash differs for {filename}")
        detail = f"{len(required) - 1} parquet splits"
    else:
        required.update({"config.json", "tokenizer_config.json", "tokenizer.json",
                         "model.safetensors.index.json"})
        if "adapter_config.json" in files:
            raise ValueError(f"{repo_id}: expected a full checkpoint, found adapter_config.json")
        resolve = f"{HF_URL}/{repo_path}/resolve/{revision}"
        config = _get_json(f"{resolve}/config.json", timeout)
        if config.get("model_type") != "qwen3":
            raise ValueError(f"{repo_id}: expected a Qwen3 checkpoint")
        index = _get_json(f"{resolve}/model.safetensors.index.json", timeout)
        shards = set(index.get("weight_map", {}).values())
        if len(shards) != entry["weight_shards"]:
            raise ValueError(f"{repo_id}: unexpected number of indexed weight shards")
        required.update(shards)
        detail = f"{len(shards)} indexed weight shards"
    if missing := required - files:
        raise ValueError(f"{repo_id}: missing files: {', '.join(sorted(missing))}")
    return f"PASS {repo_id}@{revision[:12]}: public, {detail}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="timeout in seconds for each HTTP request")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported release manifest schema")
        entries = [(kind, entry) for kind, key in (("dataset", "datasets"), ("model", "models"))
                   for entry in manifest[key]]
        if not entries:
            raise ValueError("release manifest has no repositories")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL manifest: {exc}", file=sys.stderr)
        return 1
    failed = False
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_check_repo, kind, entry, args.timeout) for kind, entry in entries]
        for (_, entry), future in zip(entries, futures):
            try:
                print(future.result())
            except Exception as exc:
                print(f"FAIL {entry.get('repo_id', '<unknown>')}: {exc}", file=sys.stderr)
                failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
