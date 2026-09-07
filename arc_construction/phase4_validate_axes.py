"""Phase 4 — Aggregate human annotations and keep majority-validated axes.

Each character's arc axes are evaluated by several annotators, who give every
axis a ``validity`` label ("valid" / "invalid" / ...). This script collects all
annotator files for a novel, counts the "valid" votes per axis, and keeps only
the axes that reached ``MIN_VALID_VOTES``. Surviving axes — with their full
definitions copied from the source axes file — are written per character to:

    results/arc_extraction/{NOVEL_NAME}/final_validated/

To run on another novel, change NOVEL_NAME in the config block below (or set
the NOVEL_NAME environment variable). Nothing else needs to change.

Usage:
    python -m arc_construction.phase4_validate_axes
    NOVEL_NAME=Anna_Kareina python -m arc_construction.phase4_validate_axes
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

# ── Config — change these to run on a different novel ───────────────────────
NOVEL_NAME      = os.environ.get("NOVEL_NAME", "don_quixote")
MIN_VALID_VOTES = 2                  # keep an axis if >= this many annotators say "valid"
VALID_LABEL     = "valid"            # the validity value that counts as a positive vote

# Older annotation files carry a numeric/date suffix (e.g. *_annotations_0511.json)
# and used a 3-way scale that includes "partial". In those numbered files a
# "partial" vote is counted as a valid vote. Newer files only use valid/invalid.
TREAT_PARTIAL_AS_VALID_IN_NUMBERED = True
PARTIAL_LABEL                      = "partial"

SOURCE_DIR_NAME = "final_grounded"   # where full axis definitions come from;
                                     # falls back to "final" if a character is missing
OUTPUT_DIR_NAME = "final_validated"

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
NOVEL_DIR      = PROJECT_ROOT / "results" / "arc_extraction" / NOVEL_NAME
ANNOTATION_DIR = NOVEL_DIR / "annotation"
OUTPUT_DIR     = NOVEL_DIR / OUTPUT_DIR_NAME

AXIS_LIST_KEYS = ["intrapersonal_axes", "relational_axes"]


def slugify(name: str) -> str:
    """Fallback slug for a character display name ('Don Quixote' -> 'don_quixote')."""
    return name.strip().lower().replace(" ", "_")


def is_numbered_file(path: Path) -> bool:
    """True if the filename has a numeric suffix after '_annotations'.

    e.g. 'don_quixote_A01_annotations_0511.json' -> True
         'don_quixote_A02_annotations.json'       -> False
    """
    suffix = path.stem.split("_annotations", 1)[-1]
    return any(ch.isdigit() for ch in suffix)


def load_annotations() -> dict[str, dict[str, dict[str, str]]]:
    """Scan every annotation file and collect validity votes.

    Returns votes[character][axis_id][annotator_id] = validity_label.
    Character and annotator id are read from the file *contents*, so a date
    suffix in the filename (e.g. ``*_annotations_0511.json``) does not matter.
    """
    votes: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    files = sorted(ANNOTATION_DIR.glob("*_annotations*.json"))
    if not files:
        raise FileNotFoundError(f"No annotation files found under {ANNOTATION_DIR}")

    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            print(f"  [skip] empty annotation file: {path.name}")
            continue
        numbered = is_numbered_file(path)
        remapped = 0
        for axis_key, entry in data.items():
            character = entry.get("character")
            annotator = entry.get("annotator_id")
            axis_id   = entry.get("axis_id", axis_key)
            validity  = entry.get("validity")
            if not (character and annotator and axis_id):
                print(f"  [warn] incomplete entry '{axis_key}' in {path.name} — skipped")
                continue
            if (TREAT_PARTIAL_AS_VALID_IN_NUMBERED and numbered
                    and validity == PARTIAL_LABEL):
                validity = VALID_LABEL          # numbered-file 'partial' counts as valid
                remapped += 1
            prev = votes[character][axis_id].get(annotator)
            if prev is not None and prev != validity:
                print(f"  [warn] {character}/{axis_id}: annotator '{annotator}' has "
                      f"conflicting labels ('{prev}' vs '{validity}') — "
                      f"using '{validity}' from {path.name}")
            votes[character][axis_id][annotator] = validity
        if remapped:
            print(f"  [info] {path.name}: {remapped} 'partial' vote(s) "
                  f"counted as '{VALID_LABEL}' (numbered file)")
    return votes


def load_source_axes() -> dict[str, tuple[str, dict]]:
    """Load full axis definitions, keyed by character display name.

    Returns {character: (slug, axes_data)}. Files from SOURCE_DIR_NAME take
    priority; ``final/`` is used as a fallback for any character missing there.
    """
    source: dict[str, tuple[str, dict]] = {}
    for dir_name in (SOURCE_DIR_NAME, "final"):
        directory = NOVEL_DIR / dir_name
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*_axes.json")):
            if path.name.startswith("all_"):          # skip aggregate files
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            character = data.get("character")
            if not character or character in source:
                continue
            slug = "_".join(path.stem.split("_")[:-2])  # strip '_<src>_axes' suffix
            source[character] = (slug, data)
    if not source:
        raise FileNotFoundError(
            f"No source axis files found in {NOVEL_DIR / SOURCE_DIR_NAME} or "
            f"{NOVEL_DIR / 'final'}")
    return source


def validate() -> None:
    votes  = load_annotations()
    source = load_source_axes()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nNovel: {NOVEL_NAME}")
    print(f"Rule : keep axes with >= {MIN_VALID_VOTES} '{VALID_LABEL}' votes\n")

    all_characters: list[dict] = []

    for character, (slug, data) in source.items():
        char_votes = votes.get(character)
        if not char_votes:
            print(f"  [skip] {character}: no annotations found")
            continue

        annotators = sorted({a for ax in char_votes.values() for a in ax})
        kept: dict[str, list] = {k: [] for k in AXIS_LIST_KEYS}
        axis_votes_summary: dict[str, dict] = {}
        n_total = n_kept = 0

        for list_key in AXIS_LIST_KEYS:
            for axis in data.get(list_key, []):
                axis_id   = axis["axis_id"]
                ann_votes = char_votes.get(axis_id, {})
                valid_n   = sum(1 for v in ann_votes.values() if v == VALID_LABEL)
                passed    = valid_n >= MIN_VALID_VOTES
                n_total  += 1

                axis_votes_summary[axis_id] = {
                    "axis_name":    axis.get("axis_name", ""),
                    "valid_votes":  valid_n,
                    "n_annotators": len(ann_votes),
                    "votes":        dict(sorted(ann_votes.items())),
                    "passed":       passed,
                }
                if passed:
                    n_kept += 1
                    axis_out = dict(axis)
                    axis_out["annotation"] = {
                        "valid_votes":  valid_n,
                        "n_annotators": len(ann_votes),
                        "votes":        dict(sorted(ann_votes.items())),
                    }
                    kept[list_key].append(axis_out)

        # Axes that were annotated but no longer exist in the source file.
        for aid in sorted(set(char_votes) - set(axis_votes_summary)):
            print(f"  [warn] {character}: annotated axis '{aid}' not in source file")

        validated = {
            "character":           character,
            "arc_richness":        data.get("arc_richness", ""),
            "intrapersonal_axes":  kept["intrapersonal_axes"],
            "relational_axes":     kept["relational_axes"],
            "validation": {
                "novel":            NOVEL_NAME,
                "min_valid_votes":  MIN_VALID_VOTES,
                "annotators":       annotators,
                "n_total_axes":     n_total,
                "n_validated_axes": n_kept,
                "n_dropped_axes":   n_total - n_kept,
                "axis_votes":       axis_votes_summary,
            },
        }

        out_path = OUTPUT_DIR / f"{slug or slugify(character)}_validated_axes.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(validated, f, indent=2, ensure_ascii=False)
        all_characters.append(validated)
        print(f"  [ok] {character:<22} kept {n_kept}/{n_total} axes  "
              f"(annotators: {', '.join(annotators)})  -> {out_path.name}")

    if all_characters:
        agg_path = OUTPUT_DIR / "all_characters_validated.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(all_characters, f, indent=2, ensure_ascii=False)
        print(f"\n  [ok] aggregate -> {agg_path.name}")

    print(f"\nDone. {len(all_characters)} character file(s) written to {OUTPUT_DIR}")


if __name__ == "__main__":
    validate()
