"""Phase 0: Split the novel into individual chapter files."""

import os
import re
import shlex

from arc_construction.config import NOVEL_NAME, NOVEL_PATH, CHAPTERS_DIR
from arc_construction.download_novels import SOURCES
from arc_construction.merge_volumes import strip_gutenberg

# Chapters larger than this are sub-chunked (~50k tokens, well under any context limit)
MAX_CHUNK_CHARS = 200_000
# Chunks shorter than this are TOC artifacts and skipped
MIN_CHUNK_CHARS = 300

# Chapter heading patterns in priority order (most specific first).
# The pattern with the most matches wins at runtime.
_CHAPTER_PATTERNS = [
    # "Chapter 1" / "Chapter 1. Title"  — Anna Karenina, Monte Cristo, HP6
    re.compile(r"^Chapter\s+\d+[^\n]*$", re.MULTILINE),
    # "CHAPTER I." / "CHAPTER I TITLE"  — P&P, Don Quixote, Hung Lou Meng
    re.compile(r"^CHAPTER\s+[IVXLC]+[^\n]*$", re.MULTILINE),
    # Standalone Roman numeral on its own line — Benjamin Franklin
    re.compile(r"^\n[IVXLC]+\n", re.MULTILINE),
    # "CHAPTER ONE" / "　　CHAPTER ONE - TITLE" / "- CHAPTER ONE -" — HP 1-5
    # [\s\-]* handles leading ideographic spaces (books 2-4) and dashes (book 5)
    re.compile(r"^[\s\-]*CHAPTER\s+[A-Z]{2,}[^\n]*$", re.MULTILINE),
    # "Chapter Two" / "Chapter Twenty-Three" — HP7 (mixed-case word numerals)
    re.compile(r"^Chapter\s+[A-Z][a-z]+(?:[\s\-][A-Z][a-z]+)?\s*$", re.MULTILINE),
    # "   I." — indented Roman numeral alone on a line — Little Women
    re.compile(r"^\s+[IVXLC]+\.\s*$", re.MULTILINE),
]


def load_and_strip_gutenberg(path: str) -> str:
    """Load novel text and strip Gutenberg header/footer."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return strip_gutenberg(f.read())


def _select_pattern(text: str) -> re.Pattern | None:
    """Return the pattern with the most matches (minimum 3)."""
    best_pattern, best_count = None, 0
    for pat in _CHAPTER_PATTERNS:
        count = len(pat.findall(text))
        if count > best_count:
            best_pattern, best_count = pat, count
    return best_pattern if best_count >= 3 else None


def _sub_chunk(text: str) -> list[str]:
    """Split a large chunk at paragraph boundaries to stay under MAX_CHUNK_CHARS."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + MAX_CHUNK_CHARS
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        split_at = text.rfind("\n\n", start, end)
        if split_at <= start:
            split_at = end
        chunks.append(text[start:split_at].strip())
        start = split_at
    return [c for c in chunks if c]


def split_chapters(text: str) -> list[tuple[int, str]]:
    """Split text into chapters.

    - Chunks shorter than MIN_CHUNK_CHARS are treated as TOC artifacts and skipped.
    - Chunks longer than MAX_CHUNK_CHARS are sub-split at paragraph boundaries.
    - Falls back to size-based splitting if no pattern matches.
    """
    pattern = _select_pattern(text)

    if pattern is None:
        print("  Warning: no chapter pattern matched — splitting by size.")
        raw = _sub_chunk(text.strip())
    else:
        matches = list(pattern.finditer(text))
        raw = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            ch_text = text[start:end].strip()
            if len(ch_text) < MIN_CHUNK_CHARS:
                continue  # skip TOC artifacts and near-empty chunks
            raw.extend(_sub_chunk(ch_text))

    chapters = [(i + 1, ch) for i, ch in enumerate(raw)]
    return chapters


def run_phase0() -> int:
    """Execute Phase 0: split novel into chapter files. Returns chapter count."""
    if not os.path.exists(NOVEL_PATH):
        if NOVEL_NAME in SOURCES:
            guidance = (
                "Download this edition first:\n"
                "  python -m arc_construction.download_novels --novel "
                f"{shlex.quote(NOVEL_NAME)}"
            )
        else:
            guidance = "Place a legally obtained plain-text copy at the path above."
        raise SystemExit(
            f"novel text not found: {NOVEL_PATH}\n"
            f"{guidance}\nSee data/novels/README.md for source editions and slugs."
        )
    os.makedirs(CHAPTERS_DIR, exist_ok=True)

    text = load_and_strip_gutenberg(NOVEL_PATH)
    chapters = split_chapters(text)

    total_words = 0
    for ch_num, ch_text in chapters:
        out_path = os.path.join(CHAPTERS_DIR, f"chapter_{ch_num:02d}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(ch_text)
        total_words += len(ch_text.split())

    n = len(chapters)
    avg_words = total_words / n if n > 0 else 0
    print(f"Phase 0 complete: {n} chapters extracted")
    print(f"  Average word count per chapter: {avg_words:.0f}")
    return n


if __name__ == "__main__":
    run_phase0()
