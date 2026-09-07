"""Phase 1A Step 2: Extract intrapersonal + relational axes from event data (async)."""

import asyncio
import json
import os
import logging

from openai import AsyncOpenAI

from arc_construction.config import (
    require_openai_key, MODEL_NAME, CHARACTER_DIRS,
    OUTPUTS_DIR, MAX_RETRIES, RETRY_BACKOFF_BASE,
    MAX_CONCURRENT_API_CALLS, NOVEL_NAME,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a literary analyst specializing in character psychology and narrative structure. "
    "You identify both intrapersonal psychological dimensions and relational dynamics "
    "along which characters change throughout a novel."
)


def build_axis_prompt(character_name: str, events_json: str) -> str:
    novel_display = NOVEL_NAME.replace("_", " ").title()
    return f"""Below is a chronological list of psychologically impactful events for {character_name} across {novel_display}.

{events_json}

Based on these events, identify TWO types of character arc axes:

## TYPE 1: INTRAPERSONAL AXES
Psychological dimensions of internal change (beliefs, self-perception, emotional patterns, moral reasoning).
- Identify 3-5 axes for characters with rich arcs, fewer for flat characters.
- REQUIRED: The first axis (intra_01) MUST be an Agency–Communion axis (Bakan 1966).
  Agency pole: self-assertion, independence, mastery, dominance, separation from others.
  Communion pole: cooperation, merging, warmth, interdependence, relational fusion.
  Track where {character_name} starts and ends on this motivational dimension.

## TYPE 2: RELATIONAL AXES
Dimensions of change in how {character_name} relates to specific other characters (trust, esteem, intimacy, antagonism).
- Only include relationships with meaningful, trackable change (1-2 axes per significant relationship).
- Do NOT force relational axes for minor or static relationships.

For EVERY axis:
1. Name it clearly and specifically (not generic like "personal growth")
2. Provide a short dimension_label (2-4 words) naming the psychological or relational dimension
3. Define the two poles (where the character starts vs. where they end on this dimension)
4. Trace the trajectory with specific chapter ranges and key events
5. Rate confidence (high/medium/low)

If {character_name} shows little meaningful change (flat arc), state this explicitly and keep axes minimal.

Return JSON only:
{{
  "character": "{character_name}",
  "arc_richness": "rich/moderate/flat",
  "intrapersonal_axes": [
    {{
      "axis_type": "intrapersonal",
      "axis_id": "intra_01",
      "dimension_label": "2-4 word dimension name",
      "axis_name": "From X to Y (descriptive arc name)",
      "pole_start": "Starting position description",
      "pole_end": "Ending position description",
      "confidence": "high/medium/low",
      "trajectory": [
        {{
          "phase": "Phase label",
          "chapter_range": [start, end],
          "position_description": "Where on the axis and why",
          "key_events": ["event descriptions"]
        }}
      ],
      "evidence_summary": "Brief justification"
    }}
  ],
  "relational_axes": [
    {{
      "axis_type": "relational",
      "axis_id": "rel_01",
      "source_character": "{character_name}",
      "target_character": "Other Character",
      "dimension_label": "2-4 word dimension name",
      "axis_name": "From X to Y (descriptive arc name)",
      "pole_start": "Starting position description",
      "pole_end": "Ending position description",
      "confidence": "high/medium/low",
      "trajectory": [
        {{
          "phase": "Phase label",
          "chapter_range": [start, end],
          "position_description": "Where on the axis and why",
          "key_events": ["event descriptions"]
        }}
      ],
      "evidence_summary": "Brief justification"
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


async def extract_axes_from_events(
    client: AsyncOpenAI, sem: asyncio.Semaphore,
    character_name: str,
) -> dict:
    """Extract intrapersonal + relational axes from the merged events file."""
    events_path = os.path.join(OUTPUTS_DIR, CHARACTER_DIRS[character_name], "events_all.json")
    with open(events_path, "r", encoding="utf-8") as f:
        all_events = json.load(f)

    # Condense for prompt
    condensed = []
    for chapter_data in all_events:
        ch = chapter_data.get("chapter", "?")
        events = chapter_data.get("events", [])
        if isinstance(events, dict):
            continue
        for evt in events:
            if not isinstance(evt, dict):
                continue
            condensed.append({
                "chapter": ch,
                "event_id": evt.get("event_id"),
                "description": evt.get("description"),
                "psychological_impact": evt.get("psychological_impact"),
                "relationship_changes": evt.get("relationship_changes"),
                "intensity": evt.get("intensity"),
            })

    events_json = json.dumps(condensed, indent=2, ensure_ascii=False)
    prompt = build_axis_prompt(character_name, events_json)

    logger.info(f"[1A-Axis] Extracting axes from events for {character_name}...")
    result = await call_api_with_retry(client, sem, SYSTEM_PROMPT, prompt)

    out_path = os.path.join(OUTPUTS_DIR, CHARACTER_DIRS[character_name], "axes_from_events.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    n_intra = len(result.get("intrapersonal_axes", []))
    n_rel = len(result.get("relational_axes", []))
    logger.info(f"[1A-Axis] {character_name}: {n_intra} intrapersonal, {n_rel} relational axes")
    return result


async def run_phase1a_axes_async(
    client: AsyncOpenAI, sem: asyncio.Semaphore,
    characters: list[str],
) -> dict[str, dict]:
    """Run axis extraction for given characters concurrently."""
    tasks = {
        char: asyncio.create_task(extract_axes_from_events(client, sem, char))
        for char in characters
    }

    axis_counts = {}
    for char, task in tasks.items():
        result = await task
        n_intra = len(result.get("intrapersonal_axes", []))
        n_rel = len(result.get("relational_axes", []))
        axis_counts[char] = {"intrapersonal": n_intra, "relational": n_rel}
        print(f"  [1A-Axis] {char}: {n_intra} intrapersonal, {n_rel} relational")

    return axis_counts


if __name__ == "__main__":
    import asyncio as _asyncio
    import config as _config
    from arc_construction.character_extractor import load_cached_characters

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _chars, _dirs = load_cached_characters(_config.RESULTS_BASE)
    _config.init(_config.NOVEL_NAME, characters=_chars, character_dirs=_dirs)
    client = AsyncOpenAI(api_key=require_openai_key())
    sem = _asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
    _asyncio.run(run_phase1a_axes_async(client, sem, _config.TARGET_CHARACTERS))
