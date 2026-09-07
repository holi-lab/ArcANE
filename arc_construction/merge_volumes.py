"""Merge multiple novel volumes into a single clean text file.

Usage — Gutenberg 2-volume novels:
    python -m arc_construction.merge_volumes \\
        --vol1 data/novels/Hung_Lou_Meng_1.txt \\
        --vol2 data/novels/Hung_Lou_Meng_2.txt \\
        --out  data/novels/Hung_Lou_Meng.txt

Usage — Multiple volumes with chapter heading normalization:
    python -m arc_construction.merge_volumes \\
        --vols data/novels/Hung_Lou_Meng_1.txt data/novels/Hung_Lou_Meng_2.txt \\
        --out data/novels/Hung_Lou_Meng.txt \\
        --normalize-chapters

--normalize-chapters rewrites every chapter heading to "Chapter N:" with
global sequential numbering across all volumes. This lets phase0_preprocess.py
split chapters correctly when volumes use inconsistent heading formats.
"""

import argparse
import re
from pathlib import Path


# ── Gutenberg strip ───────────────────────────────────────────────────────────

def strip_gutenberg(text: str) -> str:
    start = re.search(
        r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .+? \*\*\*",
        text,
        re.IGNORECASE,
    )
    if start:
        text = text[start.end():]
    end = re.search(
        r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK",
        text,
        re.IGNORECASE,
    )
    if end:
        text = text[:end.start()]
    return text.strip()


# ── Chapter heading normalization ─────────────────────────────────────────────

# Patterns that detect chapter headings, ordered from most specific to least.
# Each pattern must match exactly one heading per chapter.
_HEADING_PATTERNS = [
    # Arabic numerals with optional inline titles: "Chapter 1: Title".
    re.compile(r"^Chapter\s+\d+[^\n]*$", re.MULTILINE),
    # Uppercase headings: "CHAPTER ONE" / "- CHAPTER THREE -".
    # Uses [^\n]+ so it handles mixed-case numerals and inline titles without dashes.
    re.compile(r"^[\s\-]*CHAPTER\s+[^\n]+$", re.MULTILINE),
    # Mixed-case word numerals without inline titles: "Chapter Twenty-Three".
    re.compile(r"^Chapter\s+[A-Z][a-z]+(?:[\s\-][A-Z][a-z]+)?\s*$", re.MULTILINE),
]


def _detect_heading_pattern(text: str) -> re.Pattern | None:
    """Return the pattern with the most matches (minimum 3)."""
    best_pat, best_count = None, 0
    for pat in _HEADING_PATTERNS:
        count = len(pat.findall(text))
        if count > best_count:
            best_pat, best_count = pat, count
    return best_pat if best_count >= 3 else None


def normalize_chapters(text: str, global_start: int) -> tuple[str, int]:
    """Rewrite all chapter headings in text to 'Chapter N:' with global numbering.

    Returns (normalized_text, next_global_chapter_num).
    """
    pat = _detect_heading_pattern(text)
    if pat is None:
        print("  Warning: no chapter pattern detected — text merged without normalization.")
        return text, global_start

    chapter_num = global_start
    result_parts = []
    last_end = 0

    for m in pat.finditer(text):
        # Keep text before this heading unchanged
        result_parts.append(text[last_end:m.start()])
        # Replace heading with normalized form
        result_parts.append(f"Chapter {chapter_num}:")
        chapter_num += 1
        last_end = m.end()

    result_parts.append(text[last_end:])
    return "".join(result_parts), chapter_num


# ── Merge logic ───────────────────────────────────────────────────────────────

def merge(vol_paths: list[str], out_path: str, normalize: bool = False) -> None:
    texts = []
    for path in vol_paths:
        print(f"Reading {path}...")
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        texts.append(strip_gutenberg(raw))

    if normalize:
        print("\nNormalizing chapter headings to 'Chapter N:' format...")
        normalized = []
        global_num = 1
        for i, text in enumerate(texts, 1):
            before = global_num
            text, global_num = normalize_chapters(text, global_num)
            chapters_found = global_num - before
            print(f"  Volume {i}: {chapters_found} chapters detected (→ Chapter {before}–{global_num-1}:)")
            normalized.append(text)
        texts = normalized

    merged = "\n\n\n".join(texts)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(merged)

    print(f"\nMerged file saved to: {out_path}")
    for i, (path, text) in enumerate(zip(vol_paths, texts), 1):
        print(f"  Vol {i}: {len(text):,} chars")
    print(f"  Total: {len(merged):,} chars (~{len(merged)//4:,} tokens)")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge novel volumes into one file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Original 2-volume interface (backward compatible)
    parser.add_argument("--vol1", help="Volume 1 path (2-volume shorthand)")
    parser.add_argument("--vol2", help="Volume 2 path (2-volume shorthand)")

    # Multi-volume interface
    parser.add_argument("--vols", nargs="+", help="Paths to all volumes in order")

    parser.add_argument("--out", required=True, help="Output merged txt file path")
    parser.add_argument(
        "--normalize-chapters",
        action="store_true",
        help="Rewrite chapter headings to 'Chapter N:' with global sequential numbering",
    )
    args = parser.parse_args()

    if args.vols:
        vol_paths = args.vols
    elif args.vol1 and args.vol2:
        vol_paths = [args.vol1, args.vol2]
    else:
        parser.error("Provide either --vols (multiple) or --vol1 + --vol2")

    merge(vol_paths, args.out, normalize=args.normalize_chapters)
