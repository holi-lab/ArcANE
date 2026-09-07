"""Download the benchmark's public-domain source editions from Project Gutenberg.

Run from the release root: python -m arc_construction.download_novels --all
Downloads use Project Gutenberg's own bulk-download mirror, not its website.
Existing files are left unchanged. This module requires only the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


# Slugs match the directories containing the released arcs and probes.
SOURCES = {
    "Anna_Kareina": 1399,
    "Monte_Cristo": 1184,
    "don_quixote": 996,
    "Benjamin_Franklin": 20203,
    "He Knew He Was Right": 5140,
    "The Odd Women": 4313,
    "Anne of Green Gables": 45,
    "Dorian Gray": 174,
    "East Lynne": 3322,
    "Great Expectations": 1400,
    "Jane_Eyre": 1260,
    "Lady Audley's Secret": 8954,
    "Little Women": 37106,
    "Olaudah Equiano": 15399,
    "Persuasion": 105,
    "pride_and_prejudice": 1342,
    "The Underdogs": 549,
    "Treasure Island": 120,
}
MIRROR = "https://gutenberg.pglaf.org/cache/epub"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "novels"
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024


def validate_text(content: bytes, ebook_id: int) -> None:
    """Reject truncated responses, HTML error pages, and wrong source editions."""
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise ValueError("source text exceeds the download size limit")
    text = content.decode("utf-8-sig")
    header = text[:10_000]
    identifier = re.search(r"\bEBook\s*(?:#|No\.?\s*:)\s*(\d+)\b", header, re.I)
    if identifier is None or int(identifier.group(1)) != ebook_id:
        raise ValueError(f"response does not identify Project Gutenberg eBook #{ebook_id}")
    marker = r"\*\*\*\s+%s OF (?:THE|THIS) PROJECT GUTENBERG E(?:BOOK|TEXT)\b"
    start = re.search(marker % "START", text, re.I)
    end = re.search(marker % "END", text, re.I)
    if start is None or end is None or end.start() <= start.end():
        raise ValueError("response is missing the Gutenberg start/end markers")
    if len(text[start.end():end.start()].strip()) < 1_000:
        raise ValueError("response does not contain a complete novel")


def download_novel(novel: str, output_dir: Path, timeout: float = 30.0) -> str:
    """Save a validated source file without replacing any existing destination."""
    ebook_id = SOURCES[novel]
    destination = output_dir / f"{novel}.txt"
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ValueError(f"destination is not a regular file: {destination}")
    if destination.exists():
        return f"KEEP {destination} (existing file left unchanged)"

    url = f"{MIRROR}/{ebook_id}/pg{ebook_id}.txt"
    request = Request(url, headers={"User-Agent": "ArcANE-source-downloader/1.0"})
    with urlopen(request, timeout=timeout) as response:
        if response.headers.get_content_type() == "text/html":
            raise ValueError("mirror returned HTML instead of plain text")
        content = response.read(MAX_DOWNLOAD_BYTES + 1)
    validate_text(content, ebook_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    # A hard link publishes the complete file atomically and cannot overwrite it.
    with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".download-", suffix=".part") as temp:
        temp.write(content)
        temp.flush()
        os.link(temp.name, destination)
    digest = hashlib.sha256(content).hexdigest()
    return f"SAVED {destination} (Gutenberg #{ebook_id}, sha256={digest})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--novel", choices=sorted(SOURCES), action="append",
                           help="novel slug; repeat to download multiple titles")
    selection.add_argument("--all", action="store_true", help="download all 18 public titles")
    selection.add_argument("--list", action="store_true", help="list slugs and Gutenberg IDs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="HTTP timeout in seconds (default: 30)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="seconds between titles (default: 2)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("--delay must be a finite nonnegative number")
    if args.list:
        for novel, ebook_id in SOURCES.items():
            print(f"{novel}\thttps://www.gutenberg.org/ebooks/{ebook_id}")
        return 0

    novels = list(SOURCES) if args.all else list(dict.fromkeys(args.novel))
    failed = False
    for index, novel in enumerate(novels):
        if index:
            time.sleep(args.delay)
        try:
            print(download_novel(novel, args.output_dir, timeout=args.timeout), flush=True)
        except (OSError, URLError, ValueError) as exc:
            print(f"FAIL {novel}: {exc}", file=sys.stderr, flush=True)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
