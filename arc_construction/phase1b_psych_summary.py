"""Phase 1B Step 1: Chapter-by-chapter psychological change summaries (async)."""

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
        f"You are a literary psychologist analyzing character development in {novel_display}. "
        "You track psychological states, belief shifts, desires, intentions, and relationship dynamics chapter by chapter."
    )


def build_psych_prompt(character_name: str, chapter_num: int,
                       chapter_text: str, running_profile: str) -> str:
    return f"""TARGET CHARACTER: {character_name}

RUNNING PSYCHOLOGICAL PROFILE (accumulated from previous chapters):
{running_profile}

CURRENT CHAPTER ({chapter_num}):
{chapter_text}

Analyze this chapter for {character_name}'s psychological state and any changes. Describe:

1. PRESENCE: Is {character_name} present or mentioned in this chapter? If not, return minimal response.
2. EMOTIONAL STATE: What is {character_name} feeling in this chapter? How does it compare to the previous chapter?
3. BELIEFS & VALUES: Any shifts in what {character_name} believes about themselves, others, society, morality?
4. DESIRES: What does {character_name} want to be true about the world or their situation? How have their desires shifted from the previous chapter?
5. INTENTIONS: What is {character_name} actively trying to bring about? What plans or goals are they pursuing?
6. KEY RELATIONSHIPS: How does {character_name} view/feel about other characters in this chapter? Any shifts?
7. SELF-AWARENESS: Does {character_name} show any new self-knowledge or remain unaware of something about themselves?
8. CHANGE MAGNITUDE: Rate overall psychological change in this chapter as: none / subtle / moderate / significant / transformative

Return JSON only:
{{
  "chapter": {chapter_num},
  "character": "{character_name}",
  "present_in_chapter": true,
  "emotional_state": "Description",
  "belief_shifts": "Description or 'none'",
  "current_desires": "What {character_name} wants to be true about the world or their situation, or 'none'",
  "current_intentions": "What {character_name} is actively trying to bring about, or 'none'",
  "relationship_states": [
    {{
      "target": "OtherCharacterName",
      "current_attitude": "Description of current attitude/feeling",
      "direction_of_change": "warming/cooling/stable/conflicted",
      "dimensions": {{
        "trust": "low/medium/high",
        "esteem": "low/medium/high",
        "intimacy": "low/medium/high"
      }}
    }}
  ],
  "self_awareness_notes": "Description or 'none'",
  "change_magnitude": "none/subtle/moderate/significant/transformative",
  "chapter_summary": "1-2 sentence summary of psychological state at end of this chapter",
  "updated_running_profile": "Updated comprehensive psychological profile of {character_name} as of this chapter, max 300 words. This should capture their current emotional state, active beliefs, desires, intentions, relationship stances, and ongoing internal conflicts."
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


async def extract_psych_for_character(
    client: AsyncOpenAI, sem: asyncio.Semaphore,
    character_name: str, num_chapters: int,
) -> list[dict]:
    """Extract psychological summaries for a character (sequential per character)."""
    char_dir = os.path.join(OUTPUTS_DIR, CHARACTER_DIRS[character_name], "psych")
    os.makedirs(char_dir, exist_ok=True)

    # Load already-completed chapters so we can resume mid-run
    completed = set()
    for fname in os.listdir(char_dir):
        if fname.startswith("chapter_") and fname.endswith("_psych.json"):
            try:
                completed.add(int(fname[len("chapter_"):-len("_psych.json")]))
            except ValueError:
                pass

    all_psych = []
    running_profile = "No previous analysis. This is the beginning of the novel."

    if completed:
        for ch_num in sorted(completed):
            path = os.path.join(char_dir, f"chapter_{ch_num:02d}_psych.json")
            with open(path, encoding="utf-8") as f:
                result = json.load(f)
            all_psych.append(result)
            running_profile = result.get("updated_running_profile", running_profile)
        logger.info(
            f"[1B] {character_name}: Resuming from chapter {max(completed)+1} "
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

        logger.info(f"[1B] Chapter {ch_num}/{num_chapters} for {character_name}")

        prompt = build_psych_prompt(character_name, ch_num, chapter_text, running_profile)
        result = await call_api_with_retry(client, sem, get_system_prompt(), prompt)

        # Save per-chapter result
        out_path = os.path.join(char_dir, f"chapter_{ch_num:02d}_psych.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        running_profile = result.get("updated_running_profile", running_profile)
        all_psych.append(result)

    # Save merged
    merged_path = os.path.join(OUTPUTS_DIR, CHARACTER_DIRS[character_name], "psych_all.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_psych, f, indent=2, ensure_ascii=False)

    significant = sum(
        1 for r in all_psych
        if r.get("change_magnitude") in ("moderate", "significant", "transformative")
    )
    logger.info(f"[1B] {character_name}: {significant} chapters with moderate+ change")
    return all_psych


async def run_phase1b_psych_async(
    client: AsyncOpenAI, sem: asyncio.Semaphore,
    characters: list[str], num_chapters: int,
) -> dict[str, int]:
    """Run Phase 1B psych summaries for given characters concurrently."""
    tasks = {
        char: asyncio.create_task(
            extract_psych_for_character(client, sem, char, num_chapters)
        )
        for char in characters
    }

    change_counts = {}
    for char, task in tasks.items():
        results = await task
        significant = sum(
            1 for r in results
            if r.get("change_magnitude") in ("moderate", "significant", "transformative")
        )
        change_counts[char] = significant
        print(f"  [1B] {char}: {significant} chapters with moderate+ change")

    return change_counts


if __name__ == "__main__":
    import asyncio as _asyncio
    import config as _config
    from arc_construction.character_extractor import load_cached_characters

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _chars, _dirs = load_cached_characters(_config.RESULTS_BASE)
    _config.init(_config.NOVEL_NAME, characters=_chars, character_dirs=_dirs)
    client = AsyncOpenAI(api_key=require_openai_key())
    sem = _asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
    _asyncio.run(run_phase1b_psych_async(client, sem, _config.TARGET_CHARACTERS, _config.count_chapters()))
