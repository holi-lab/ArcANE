"""Phase 2: Cross-validate and merge axes from both approaches (async).

Compares intrapersonal axes separately from relational axes.
"""

import asyncio
import json
import os
import logging

from openai import AsyncOpenAI

from arc_construction.config import (
    require_openai_key, MODEL_NAME, CHARACTER_DIRS,
    OUTPUTS_DIR, FINAL_DIR, MAX_RETRIES, RETRY_BACKOFF_BASE,
    MAX_CONCURRENT_API_CALLS, NOVEL_NAME,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a literary analyst performing cross-validation of character arc analyses. "
    "You compare independently generated axis sets and produce a merged, high-confidence result."
)


def build_merge_prompt(character_name: str, axes_events: str, axes_psych: str) -> str:
    novel_display = NOVEL_NAME.replace("_", " ").title()
    return f"""Below are two independently generated sets of character arc axes for {character_name} in {novel_display}.
Each set contains INTRAPERSONAL axes (internal psychological change) and RELATIONAL axes (interpersonal dynamics change).

SET A (from event graph analysis):
{axes_events}

SET B (from psychological summary analysis):
{axes_psych}

Compare these two sets and produce a merged result. Compare intrapersonal axes with intrapersonal axes, and relational axes with relational axes separately.

For each category:
1. MATCHED: Axes that appear in both sets (possibly different names but same concept). HIGH CONFIDENCE.
2. UNIQUE TO A: Axes only in Set A. Assess: valid_missed / artifact / borderline.
3. UNIQUE TO B: Axes only in Set B. Same assessment.
4. MERGED FINAL SET: Your recommended final axes, combining the best from both.

For each final axis, synthesize trajectory information from both sources.
Every final axis MUST include: axis_type, dimension_label (2-4 words), axis_name, pole_start, pole_end, confidence, source, arc_direction, trajectory, evidence_summary.
Relational axes must also include source_character and target_character.

arc_direction classifies the overall trajectory shape of this axis:
- "positive": character moves from a flawed position toward a healthier/wiser one
- "negative": character entrenches or deepens a flaw
- "flat": character's position on this axis remains essentially stable
- "disillusionment": character loses a positive belief without gaining a replacement
- "ambivalent": trajectory is non-linear with no clear net direction

Return JSON only:
{{
  "character": "{character_name}",
  "comparison": {{
    "intrapersonal": {{
      "matched": [
        {{
          "set_a_axis": "axis name from A",
          "set_b_axis": "axis name from B",
          "overlap_description": "How they correspond"
        }}
      ],
      "unique_to_a": [{{"axis": "name", "assessment": "valid_missed/artifact/borderline"}}],
      "unique_to_b": [{{"axis": "name", "assessment": "valid_missed/artifact/borderline"}}]
    }},
    "relational": {{
      "matched": [
        {{
          "set_a_axis": "axis name from A",
          "set_b_axis": "axis name from B",
          "overlap_description": "How they correspond"
        }}
      ],
      "unique_to_a": [{{"axis": "name", "assessment": "valid_missed/artifact/borderline"}}],
      "unique_to_b": [{{"axis": "name", "assessment": "valid_missed/artifact/borderline"}}]
    }}
  }},
  "intrapersonal_axes": [
    {{
      "axis_type": "intrapersonal",
      "axis_id": "final_intra_01",
      "dimension_label": "2-4 word dimension name",
      "axis_name": "Merged descriptive name",
      "pole_start": "...",
      "pole_end": "...",
      "confidence": "high/medium/low",
      "source": "both/event_only/psych_only",
      "arc_direction": "positive/negative/flat/disillusionment/ambivalent",
      "trajectory": [
        {{
          "phase": "...",
          "chapter_range": [0, 0],
          "position_description": "...",
          "key_moments": ["descriptions"]
        }}
      ],
      "evidence_summary": "..."
    }}
  ],
  "relational_axes": [
    {{
      "axis_type": "relational",
      "axis_id": "final_rel_01",
      "source_character": "{character_name}",
      "target_character": "Other Character",
      "dimension_label": "2-4 word dimension name",
      "axis_name": "Merged descriptive name",
      "pole_start": "...",
      "pole_end": "...",
      "confidence": "high/medium/low",
      "source": "both/event_only/psych_only",
      "arc_direction": "positive/negative/flat/disillusionment/ambivalent",
      "trajectory": [
        {{
          "phase": "...",
          "chapter_range": [0, 0],
          "position_description": "...",
          "key_moments": ["descriptions"]
        }}
      ],
      "evidence_summary": "..."
    }}
  ]
}}"""


async def call_api_with_retry(client: AsyncOpenAI, sem: asyncio.Semaphore,
                               system: str, user: str) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            wait = RETRY_BACKOFF_BASE ** (attempt + 1)
            logger.warning(f"API error (attempt {attempt+1}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(wait)
            else:
                logger.error(f"Failed after {MAX_RETRIES} attempts: {e}")
                raise


async def cross_validate_character(
    client: AsyncOpenAI, sem: asyncio.Semaphore,
    character_name: str,
) -> dict:
    """Cross-validate axes for a single character."""
    char_dir = os.path.join(OUTPUTS_DIR, CHARACTER_DIRS[character_name])

    with open(os.path.join(char_dir, "axes_from_events.json"), "r") as f:
        axes_events = json.load(f)
    with open(os.path.join(char_dir, "axes_from_psych.json"), "r") as f:
        axes_psych = json.load(f)

    prompt = build_merge_prompt(
        character_name,
        json.dumps(axes_events, indent=2, ensure_ascii=False),
        json.dumps(axes_psych, indent=2, ensure_ascii=False),
    )

    logger.info(f"[Phase 2] Cross-validating axes for {character_name}...")
    result = await call_api_with_retry(client, sem, SYSTEM_PROMPT, prompt)

    # Save individual character result
    os.makedirs(FINAL_DIR, exist_ok=True)
    out_path = os.path.join(FINAL_DIR, f"{CHARACTER_DIRS[character_name]}_final_axes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    n_intra = len(result.get("intrapersonal_axes", []))
    n_rel = len(result.get("relational_axes", []))
    logger.info(f"[Phase 2] {character_name}: {n_intra} intrapersonal, {n_rel} relational final axes")
    return result


async def run_phase2_async(
    client: AsyncOpenAI, sem: asyncio.Semaphore,
    characters: list[str],
) -> dict[str, dict]:
    """Run cross-validation for all characters concurrently."""
    tasks = {
        char: asyncio.create_task(cross_validate_character(client, sem, char))
        for char in characters
    }

    all_results = {}
    final_counts = {}

    for char, task in tasks.items():
        result = await task
        all_results[char] = result
        n_intra = len(result.get("intrapersonal_axes", []))
        n_rel = len(result.get("relational_axes", []))
        final_counts[char] = {"intrapersonal": n_intra, "relational": n_rel}
        print(f"  [Phase 2] {char}: {n_intra} intrapersonal, {n_rel} relational final axes")

    # Save combined file
    combined_path = os.path.join(FINAL_DIR, "all_characters_axes.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nCombined results saved to {combined_path}")

    return final_counts


if __name__ == "__main__":
    import asyncio as _asyncio
    import config as _config
    from arc_construction.character_extractor import load_cached_characters

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _chars, _dirs = load_cached_characters(_config.RESULTS_BASE)
    _config.init(_config.NOVEL_NAME, characters=_chars, character_dirs=_dirs)
    client = AsyncOpenAI(api_key=require_openai_key())
    sem = _asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
    _asyncio.run(run_phase2_async(client, sem, _config.TARGET_CHARACTERS))
