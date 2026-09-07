"""Text grounding: locate a key_moment in the novel's chapter texts and
return the exact passage that depicts it.

Two stages, both using structured outputs:
  locate_chapter   — match a key_moment to one event in events_all.json
                     within the relevant chapter range (no chapter text loaded).
  extract_passage  — load that chapter's text and ask the model for the
                     verbatim passage.
"""

import asyncio
import json
import logging
from typing import Optional

from openai import AsyncOpenAI

from ._api import parse_call
from .config import (
    MODEL_LOCATOR, MODEL_EXTRACTOR,
    MAX_CHAPTER_CHARS, MAX_LOCATOR_CANDIDATES,
    GEN_TEMPERATURE,
    events_path, chapter_file,
)
from .prompts import (
    SYSTEM_LITERARY_LOCATOR, SYSTEM_LITERARY_EXTRACTOR,
    step2_locator_user, step3_extractor_user,
)
from .schemas import LocatorOutput, ExtractorOutput

logger = logging.getLogger(__name__)


# ── Event index helpers ────────────────────────────────────────────
def load_events_index(novel: str, character_slug: str) -> list[dict]:
    """Returns a flat list of {chapter, event_id, description, characters_involved}."""
    p = events_path(novel, character_slug)
    if not p.exists():
        logger.warning(f"events file not found: {p}")
        return []
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    flat: list[dict] = []
    for ch_block in data:
        if not isinstance(ch_block, dict):
            continue
        ch = ch_block.get("chapter")
        for ev in ch_block.get("events", []):
            if not isinstance(ev, dict):
                continue
            flat.append({
                "chapter": ch,
                "event_id": ev.get("event_id", ""),
                "description": ev.get("description", ""),
                "characters_involved": ev.get("characters_involved", []),
            })
    return flat


def filter_candidates(events: list[dict], chapter_range: list[int],
                      max_candidates: int = MAX_LOCATOR_CANDIDATES) -> list[dict]:
    lo, hi = chapter_range[0], chapter_range[1]
    in_range = [e for e in events if lo <= e["chapter"] <= hi]
    return in_range[:max_candidates]


# ── Stage 2: locate chapter ────────────────────────────────────────
async def locate_chapter(client: AsyncOpenAI, sem: asyncio.Semaphore,
                         novel: str, character_slug: str, key_moment: str,
                         chapter_range: list[int]) -> Optional[dict]:
    """Returns {chapter, event_id, event_description, reasoning} or None."""
    events = load_events_index(novel, character_slug)
    candidates = filter_candidates(events, chapter_range)
    if not candidates:
        logger.warning(
            f"no candidate events for {character_slug} in ch {chapter_range}; "
            f"fallback to midpoint"
        )
        mid = (chapter_range[0] + chapter_range[1]) // 2
        return {"chapter": mid, "event_id": None, "event_description": "",
                "reasoning": "fallback: no events_all entries in range"}

    user = step2_locator_user(key_moment, chapter_range, candidates)
    try:
        result = await parse_call(client, sem, MODEL_LOCATOR,
                                  SYSTEM_LITERARY_LOCATOR, user,
                                  LocatorOutput, GEN_TEMPERATURE)
    except Exception as e:
        logger.warning(f"locator failed: {e}; using first candidate")
        first = candidates[0]
        return {"chapter": first["chapter"], "event_id": first["event_id"],
                "event_description": first["description"],
                "reasoning": "fallback: locator call failed"}

    ch = result.get("chapter")
    ev_id = result.get("event_id")
    if ev_id in (None, "", "null"):
        logger.info(
            f"  locator rejected all candidates for key_moment='{key_moment}' "
            f"(reasoning: {result.get('reasoning','')[:120]})"
        )
        return None

    ev_desc = next((c["description"] for c in candidates if c["event_id"] == ev_id), "")
    if not ev_desc and candidates:
        ev_desc = next((c["description"] for c in candidates if c["chapter"] == ch), "")

    return {
        "chapter": ch if isinstance(ch, int) else candidates[0]["chapter"],
        "event_id": ev_id,
        "event_description": ev_desc,
        "reasoning": result.get("reasoning", ""),
    }


# ── Stage 3: extract verbatim passage ──────────────────────────────
def _load_chapter_text(novel: str, ch: int) -> str:
    p = chapter_file(novel, ch)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8")
    if len(text) > MAX_CHAPTER_CHARS:
        text = text[:MAX_CHAPTER_CHARS]
    return text


async def extract_passage(client: AsyncOpenAI, sem: asyncio.Semaphore,
                          novel: str, chapter_num: int, key_moment: str,
                          event_description: str) -> Optional[dict]:
    chapter_text = _load_chapter_text(novel, chapter_num)
    if not chapter_text:
        logger.warning(f"chapter {chapter_num} text not found for {novel}")
        return None

    user = step3_extractor_user(novel, chapter_num, key_moment,
                                event_description, chapter_text)
    try:
        result = await parse_call(client, sem, MODEL_EXTRACTOR,
                                  SYSTEM_LITERARY_EXTRACTOR, user,
                                  ExtractorOutput, GEN_TEMPERATURE)
    except Exception as e:
        logger.warning(f"extractor failed for ch {chapter_num}: {e}")
        return None

    passage = result.get("verbatim_passage")
    if not passage:
        logger.info(f"extractor returned null passage for ch {chapter_num}: "
                    f"{result.get('note','(no note)')}")
        return None
    return {
        "verbatim_passage": passage,
        "first_words": result.get("first_words", ""),
        "last_words":  result.get("last_words", ""),
        "chapter": chapter_num,
    }


# ── Combined: locate + extract ─────────────────────────────────────
async def ground_scene(client: AsyncOpenAI, sem: asyncio.Semaphore,
                       novel: str, character_slug: str, key_moment: str,
                       chapter_range: list[int]) -> Optional[dict]:
    loc = await locate_chapter(client, sem, novel, character_slug,
                               key_moment, chapter_range)
    if not loc or loc.get("chapter") is None:
        return None
    pas = await extract_passage(client, sem, novel, loc["chapter"],
                                key_moment, loc.get("event_description", ""))
    if not pas:
        return None
    return {
        "chapter": loc["chapter"],
        "event_id": loc.get("event_id"),
        "event_description": loc.get("event_description", ""),
        "locator_reasoning": loc.get("reasoning", ""),
        "verbatim_passage": pas["verbatim_passage"],
        "first_words": pas["first_words"],
        "last_words":  pas["last_words"],
    }
