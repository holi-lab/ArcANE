"""Orchestrator: runs all phases of the character arc extraction pipeline.

Execution order:
  Phase 0  — sequential (chapter splitting)
  Phase 1  — two concurrent streams per character (event and psychology)
             Within each stream, chapters are sequential.
             After all streams finish, axis extraction runs concurrently.
  Phase 2  — one cross-validation call per character
  Phase 3  — literary grounding with three critic perspectives
"""

import asyncio
import sys
import time
import logging
from datetime import datetime

from openai import AsyncOpenAI

from arc_construction import config
from arc_construction.config import (
    MODEL_NAME, require_openai_key, MAX_CONCURRENT_API_CALLS, NOVEL_NAME,
)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_banner(msg: str):
    print(f"\n{'='*60}")
    print(f"[{timestamp()}] {msg}")
    print(f"{'='*60}")


def estimate_cost():
    """Print estimated cost and ask for confirmation."""
    estimated_calls = config.ESTIMATED_CALLS
    est_input_tokens = estimated_calls * 2000
    est_output_tokens = estimated_calls * 500
    # gpt-5.4-mini rough pricing: $0.40/1M input, $1.60/1M output
    est_cost_input = est_input_tokens / 1_000_000 * 0.40
    est_cost_output = est_output_tokens / 1_000_000 * 1.60
    est_total = est_cost_input + est_cost_output

    print("\n--- Cost Estimate ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Estimated API calls: ~{estimated_calls}")
    print(f"Estimated input tokens: ~{est_input_tokens:,}")
    print(f"Estimated output tokens: ~{est_output_tokens:,}")
    print(f"Estimated cost: ~${est_total:.2f}")
    print("(Actual cost may vary based on chapter lengths and response sizes)")
    print()

    answer = input("Proceed? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("Aborted.")
        sys.exit(0)


async def run_pipeline():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    start_time = time.time()

    # ── Character Extraction ──────────────────────────────────
    print_banner("Character Extraction")
    from arc_construction.character_extractor import extract_characters
    client = AsyncOpenAI(api_key=require_openai_key())
    target_chars, char_dirs = await extract_characters(
        client, MODEL_NAME, NOVEL_NAME,
        config.NOVEL_PATH, config.RESULTS_BASE,
        n_characters=config.N_EXTRACT_CHARACTERS,
    )
    config.init(NOVEL_NAME, characters=target_chars, character_dirs=char_dirs)
    TARGET_CHARACTERS = config.TARGET_CHARACTERS

    # ── Phase 0 ──────────────────────────────────────────────
    print_banner("PHASE 0: Preprocessing — Chapter Splitting")
    from arc_construction.phase0_preprocess import run_phase0
    num_chapters = run_phase0()

    # Update chapter count and estimated calls, then confirm cost
    config.NUM_CHAPTERS = num_chapters
    config.ESTIMATED_CALLS = config.estimate_api_calls(num_chapters)
    estimate_cost()

    # ── Phase 1: two streams per character ───────────────────
    print_banner(
        f"PHASE 1: Event Graph + Psychological Summary ({2 * len(TARGET_CHARACTERS)} streams)"
    )

    sem = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

    from arc_construction.phase1a_event_graph import run_phase1a_events_async
    from arc_construction.phase1b_psych_summary import run_phase1b_psych_async

    # Launch both approaches concurrently across the selected characters.
    task_1a = asyncio.create_task(
        run_phase1a_events_async(client, sem, TARGET_CHARACTERS, num_chapters)
    )
    task_1b = asyncio.create_task(
        run_phase1b_psych_async(client, sem, TARGET_CHARACTERS, num_chapters)
    )

    event_counts, _ = await asyncio.gather(task_1a, task_1b)

    # ── Phase 1 Axis Extraction (after event/psych data is ready) ──
    print_banner("PHASE 1: Axis Extraction (events + psych, concurrent)")

    from arc_construction.phase1a_axis_extract import run_phase1a_axes_async
    from arc_construction.phase1b_axis_extract import run_phase1b_axes_async

    task_axes_a = asyncio.create_task(
        run_phase1a_axes_async(client, sem, TARGET_CHARACTERS)
    )
    task_axes_b = asyncio.create_task(
        run_phase1b_axes_async(client, sem, TARGET_CHARACTERS)
    )

    axes_a_counts, axes_b_counts = await asyncio.gather(task_axes_a, task_axes_b)

    # ── Phase 2: Cross-Validation ────────────────────────────
    print_banner("PHASE 2: Cross-Validation & Merge")

    from arc_construction.phase2_cross_validate import run_phase2_async
    final_counts = await run_phase2_async(client, sem, TARGET_CHARACTERS)

    # ── Phase 3: Literary Grounding ──────────────────────────
    print_banner("PHASE 3: Literary Grounding Validation")
    from arc_construction import phase3_literary_grounding
    await phase3_literary_grounding.run_silent()

    # ── Summary ──────────────────────────────────────────────
    elapsed = time.time() - start_time
    print_banner("PIPELINE COMPLETE")
    print(f"Total time: {elapsed/60:.1f} minutes\n")

    header = (
        f"{'Character':<20} {'Events':<8} "
        f"{'Intra(A)':<10} {'Rel(A)':<8} "
        f"{'Intra(B)':<10} {'Rel(B)':<8} "
        f"{'Final-I':<9} {'Final-R':<9}"
    )
    print(header)
    print("-" * len(header))
    for char in TARGET_CHARACTERS:
        ev = event_counts.get(char, "?")
        ia = axes_a_counts.get(char, {}).get("intrapersonal", "?")
        ra = axes_a_counts.get(char, {}).get("relational", "?")
        ib = axes_b_counts.get(char, {}).get("intrapersonal", "?")
        rb = axes_b_counts.get(char, {}).get("relational", "?")
        fi = final_counts.get(char, {}).get("intrapersonal", "?")
        fr = final_counts.get(char, {}).get("relational", "?")
        print(
            f"{char:<20} {str(ev):<8} "
            f"{str(ia):<10} {str(ra):<8} "
            f"{str(ib):<10} {str(rb):<8} "
            f"{str(fi):<9} {str(fr):<9}"
        )

    print(f"\nResults saved to {config.FINAL_DIR}")


def main():
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
