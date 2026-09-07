"""Phase 1A Step 1: Chapter-by-chapter event extraction for each character (async)."""

import asyncio
import json
import os
import logging

from openai import AsyncOpenAI

from arc_construction.config import (
    require_openai_key, MODEL_NAME, CHARACTER_DIRS,
    CHAPTERS_DIR, OUTPUTS_DIR, MAX_RETRIES, RETRY_BACKOFF_BASE,
    MAX_CONCURRENT_API_CALLS, NOVEL_NAME,
)

logger = logging.getLogger(__name__)

def get_system_prompt() -> str:
    novel_display = NOVEL_NAME.replace("_", " ").title()
    return (
        "You are a literary analyst specializing in character psychology. "
        f"You are analyzing {novel_display} chapter by chapter to track "
        "psychologically impactful events for a specific character."
    )


def build_user_prompt(character_name: str, chapter_num: int,
                      chapter_text: str, running_context: str) -> str:
    return f"""TARGET CHARACTER: {character_name}

RUNNING CONTEXT (events extracted from previous chapters):
{running_context}

CURRENT CHAPTER ({chapter_num}):
{chapter_text}

Extract events from this chapter that have psychological impact on {character_name}. Include events where:
- {character_name}'s emotions, beliefs, or self-perception shift
- {character_name}'s relationship with another character changes
- {character_name} makes a significant decision or judgment
- Something happens that will later affect {character_name}'s development

If {character_name} does not appear in this chapter or nothing psychologically relevant happens to them, return an empty events array.

IMPORTANT: Be selective. Only include events with genuine psychological weight, not every minor interaction.

Return JSON only, no other text:
{{
  "chapter": {chapter_num},
  "character": "{character_name}",
  "events": [
    {{
      "event_id": "ch{chapter_num}_evt01",
      "description": "Brief description of what happens",
      "characters_involved": ["list", "of", "characters"],
      "psychological_impact": {{
        "emotional_state": "What {character_name} feels",
        "belief_change": "How their beliefs/views shift (or 'none')",
        "self_perception_change": "How they see themselves differently (or 'none')",
        "sequence_type": "redemptive (bad→good outcome) / contaminating (good→bad outcome) / neutral",
        "meaning_made": "Does {character_name} extract a lesson or update their self-narrative from this event? Describe briefly, or 'none'"
      }},
      "relationship_changes": [
        {{
          "target": "OtherCharacterName",
          "direction": "negative/positive/neutral/ambivalent",
          "change_description": "How the relationship changes",
          "dimensions_affected": ["trust", "esteem", "intimacy", "antagonism"]
        }}
      ],
      "intensity": "low/medium/high"
    }}
  ],
  "updated_running_summary": "Updated 2-3 sentence summary of {character_name}'s psychological trajectory so far, incorporating this chapter's events"
}}"""


async def call_api_with_retry(client: AsyncOpenAI, sem: asyncio.Semaphore,
                               system: str, user: str) -> dict:
    """Call OpenAI API with retry logic and semaphore."""
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


async def extract_events_for_character(
    client: AsyncOpenAI, sem: asyncio.Semaphore,
    character_name: str, num_chapters: int,
) -> list[dict]:
    """Extract events for a single character across all chapters (sequential per character)."""
    char_dir = os.path.join(OUTPUTS_DIR, CHARACTER_DIRS[character_name], "events")
    os.makedirs(char_dir, exist_ok=True)

    # Load already-completed chapters so we can resume mid-run
    completed = set()
    for fname in os.listdir(char_dir):
        if fname.startswith("chapter_") and fname.endswith("_events.json"):
            try:
                completed.add(int(fname[len("chapter_"):-len("_events.json")]))
            except ValueError:
                pass

    all_events = []
    running_context = "No previous events. This is the beginning of the novel."

    if completed:
        for ch_num in sorted(completed):
            path = os.path.join(char_dir, f"chapter_{ch_num:02d}_events.json")
            with open(path, encoding="utf-8") as f:
                result = json.load(f)
            all_events.append(result)
            running_context = result.get("updated_running_summary", running_context)
        logger.info(
            f"[1A] {character_name}: Resuming from chapter {max(completed)+1} "
            f"({len(completed)} chapters already done)"
        )

    for ch_num in range(1, num_chapters + 1):
        if ch_num in completed:
            continue

        ch_path = os.path.join(CHAPTERS_DIR, f"chapter_{ch_num:02d}.txt")
        if not os.path.exists(ch_path):
            logger.warning(f"Chapter file not found: {ch_path}")
            continue

        with open(ch_path, "r", encoding="utf-8") as f:
            chapter_text = f.read()

        logger.info(f"[1A] Chapter {ch_num}/{num_chapters} for {character_name}")

        prompt = build_user_prompt(character_name, ch_num, chapter_text, running_context)
        result = await call_api_with_retry(client, sem, get_system_prompt(), prompt)

        # Save per-chapter result
        out_path = os.path.join(char_dir, f"chapter_{ch_num:02d}_events.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Update running context
        running_context = result.get("updated_running_summary", running_context)
        all_events.append(result)

    # Save merged events
    merged_path = os.path.join(OUTPUTS_DIR, CHARACTER_DIRS[character_name], "events_all.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_events, f, indent=2, ensure_ascii=False)

    total_events = sum(
        sum(1 for e in r.get("events", []) if isinstance(e, dict))
        for r in all_events if isinstance(r.get("events"), list)
    )
    logger.info(f"[1A] {character_name}: {total_events} events extracted")
    return all_events


async def run_phase1a_events_async(
    client: AsyncOpenAI, sem: asyncio.Semaphore,
    characters: list[str], num_chapters: int,
) -> dict[str, int]:
    """Run Phase 1A event extraction for given characters concurrently."""
    tasks = {
        char: asyncio.create_task(
            extract_events_for_character(client, sem, char, num_chapters)
        )
        for char in characters
    }

    event_counts = {}
    for char, task in tasks.items():
        results = await task
        total = sum(len(r.get("events", [])) for r in results if isinstance(r.get("events"), list))
        event_counts[char] = total
        print(f"  [1A] {char}: {total} events")

    return event_counts


# Standalone entry point
if __name__ == "__main__":
    import asyncio as _asyncio
    import config as _config
    from arc_construction.character_extractor import load_cached_characters

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _chars, _dirs = load_cached_characters(_config.RESULTS_BASE)
    _config.init(_config.NOVEL_NAME, characters=_chars, character_dirs=_dirs)
    client = AsyncOpenAI(api_key=require_openai_key())
    sem = _asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
    _asyncio.run(run_phase1a_events_async(client, sem, _config.TARGET_CHARACTERS, _config.count_chapters()))
