"""Single-character probe generation runner.

Example:
    uv run python -m probe_generation \
        --novel don_quixote --character "Don Quixote" \
        --axes final_intra_01 final_rel_02
"""

import asyncio
import json
import logging
from typing import Optional

from openai import AsyncOpenAI

from .config import (
    require_openai_key, MAX_CONCURRENT_API_CALLS, PROBE_SCHEMA_VERSION,
    axes_path, probes_out_dir, characters_path,
)
from .pipeline import generate_arc_family

logger = logging.getLogger(__name__)


def _arc_passes_filter(arc: dict, axis_source: str) -> bool:
    """Human-validated axes are final; other sources use the critic filter."""
    if axis_source == "final_validated":
        return True
    tag = arc.get("literary_validation", {}).get("validation_tag", "")
    if tag == "none_axis":
        logger.warning(
            "skipping %s: literary_validation.validation_tag=none_axis",
            arc.get("axis_id"),
        )
        return False
    return True


def _resolve_char_slug(novel: str, character: str) -> str:
    """Return the on-disk slug for a character.

    Prefers the explicit `character_dirs` mapping in characters.json when
    present (e.g. "Philip Pirrip (Pip)" → "pip"); otherwise falls back to
    a defensive lowercase + punctuation-strip transformation.
    """
    cp = characters_path(novel)
    if cp.exists():
        try:
            with open(cp, encoding="utf-8") as f:
                chars_data = json.load(f)
            dirs = chars_data.get("character_dirs") or {}
            if character in dirs:
                return dirs[character]
        except Exception:
            pass
    return (
        character.lower()
        .replace(",", "")
        .replace(".", "")
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
    )


async def run_one_character(client: AsyncOpenAI, sem: asyncio.Semaphore,
                            novel: str, character: str,
                            axes_filter: Optional[list[str]] = None,
                            probe_types: Optional[list[str]] = None,
                            out_name: Optional[str] = None,
                            axes_source: str = "auto") -> None:
    char_slug = _resolve_char_slug(novel, character)
    ap, axis_source = axes_path(novel, char_slug, source=axes_source)
    if not ap.exists():
        logger.warning(f"no axes file for {character}: {ap}")
        return
    logger.info(f"using axes from {axis_source}: {ap}")

    with open(ap, encoding="utf-8") as f:
        data = json.load(f)

    all_arcs = data.get("intrapersonal_axes", []) + data.get("relational_axes", [])
    if axes_filter:
        wanted = set(axes_filter)
        all_arcs = [a for a in all_arcs if a["axis_id"] in wanted]
    all_arcs = [a for a in all_arcs if _arc_passes_filter(a, axis_source)]

    if not all_arcs:
        logger.warning(f"no arcs survived filter for {character}")
        return

    logger.info(f"generating {len(all_arcs)} arcs for {character} "
                f"(types={probe_types or 'all'})")

    family_results = await asyncio.gather(*[
        generate_arc_family(client, sem, character, char_slug, arc, novel,
                            probe_types=probe_types)
        for arc in all_arcs
    ])

    out_dir = probes_out_dir(novel)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (out_name or f"{char_slug}_probes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "character": character,
            "novel": novel,
            "version": PROBE_SCHEMA_VERSION,
            "axis_source": axis_source,
            "families": family_results,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"wrote {out_path}")


async def run(novel: str, character: Optional[str],
              all_characters: bool,
              axes_filter: Optional[list[str]],
              probe_types: Optional[list[str]],
              out_name: Optional[str],
              axes_source: str = "auto") -> None:
    client = AsyncOpenAI(api_key=require_openai_key())
    sem = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

    if all_characters:
        cp = characters_path(novel)
        if not cp.exists():
            raise ValueError(f"no characters.json for {novel}")
        with open(cp, encoding="utf-8") as f:
            chars_data = json.load(f)
        targets = chars_data.get("target_characters", [])
        for ch in targets:
            try:
                await run_one_character(
                    client, sem, novel, ch,
                    axes_filter=axes_filter, probe_types=probe_types,
                    out_name=None, axes_source=axes_source,
                )
            except Exception as e:
                logger.error(f"character '{ch}' failed: {e}")
    else:
        if not character:
            raise ValueError("--character is required when --all-characters is not set")
        await run_one_character(
            client, sem, novel, character,
            axes_filter=axes_filter, probe_types=probe_types,
            out_name=out_name, axes_source=axes_source,
        )


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        prog="probe_generation",
        description="Generate probes (In-Text + In-World + Out-of-World), "
                    "multi-phase per probe with gt_thought.",
    )
    p.add_argument("--novel", required=True)
    p.add_argument("--character", help="single character name (e.g. 'Don Quixote')")
    p.add_argument("--all-characters", action="store_true")
    p.add_argument("--axes", nargs="*", help="restrict to these axis_ids")
    p.add_argument(
        "--probe-types", nargs="*",
        choices=["in_text", "in_world", "out_of_world"],
        default=None,
        help="restrict probe types (default: all three)",
    )
    p.add_argument("--out-name", default=None,
                   help="output filename (default: {slug}_probes.json)")
    p.add_argument(
        "--axes-source",
        choices=["auto", "final", "final_auto", "final_validated"],
        default="auto",
        help=("which axes file to read. 'auto' (default) prefers final_validated "
              "then the existing probe source, then final_auto. 'final' uses pre-validation axes "
              "(all candidates) — useful when filtering will be done post-hoc."),
    )
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(run(
        novel=args.novel,
        character=args.character,
        all_characters=args.all_characters,
        axes_filter=args.axes,
        probe_types=args.probe_types,
        out_name=args.out_name,
        axes_source=args.axes_source,
    ))


if __name__ == "__main__":
    main()
