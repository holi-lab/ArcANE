"""Extract major characters from a novel using LLM, then verify them.

Two stages, both run inside extract_characters():

  1. EXTRACTION — the LLM picks the N most psychologically important characters
     from the opening of the novel.
  2. VERIFICATION — every extracted character is checked against the FULL novel
     text. A name found verbatim (case-insensitive substring, like Ctrl+F) is
     confirmed. A name NOT found is judged by a single LLM call — given the
     novel text — as either a real character under a different spelling /
     romanisation / nickname (kept) or not a character at all (dropped).

Usage:
    # Called automatically at the start of run_all.py
    # Results are cached to results/arc_extraction/{novel_name}/characters.json
    # Re-run with force=True to override cache.

    python -m arc_construction.character_extractor <Novel_Name> [results_base]
"""

import json
import logging
import os

from openai import AsyncOpenAI

from arc_construction.config import require_openai_key

logger = logging.getLogger(__name__)

EXTRACTION_CHAR_LIMIT = 30_000  # opening characters sent to the LLM for extraction
VERIFY_EXCERPT_CHARS  = 20_000  # opening text shown to the LLM when judging names


# ── Cache ────────────────────────────────────────────────────────────────────
def _cache_path(results_base: str) -> str:
    return os.path.join(results_base, "characters.json")


def _load_cache(results_base: str) -> tuple[list[str], dict[str, str]] | None:
    path = _cache_path(results_base)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["target_characters"], data["character_dirs"]


def _save_cache(
    results_base: str,
    target_characters: list[str],
    character_dirs: dict[str, str],
) -> None:
    os.makedirs(results_base, exist_ok=True)
    with open(_cache_path(results_base), "w", encoding="utf-8") as f:
        json.dump(
            {"target_characters": target_characters, "character_dirs": character_dirs},
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_cached_characters(results_base: str) -> tuple[list[str], dict[str, str]]:
    """Load characters from cache. Raises FileNotFoundError if cache doesn't exist.

    Use this in stages that run after character extraction,
    where the cache is guaranteed to exist from a prior run_all.py execution.
    """
    cached = _load_cache(results_base)
    if cached is None:
        raise FileNotFoundError(
            f"Character cache not found at {_cache_path(results_base)}.\n"
            "Run the full pipeline (run_all.py) first to extract characters."
        )
    logger.info(f"[CharacterExtractor] Loaded from cache: {cached[0]}")
    return cached


# ── Stage 1: extraction ──────────────────────────────────────────────────────
_EXTRACT_SYSTEM = (
    "You are a literary scholar with comprehensive knowledge of classic and contemporary fiction. "
    "Your task is to identify the most psychologically important characters in a novel."
)


def _build_extract_prompt(novel_display: str, text: str, n_characters: int) -> str:
    approx_kb = EXTRACTION_CHAR_LIMIT // 1000
    return f"""Novel: {novel_display}

Below is the opening portion of the novel (approximately the first {approx_kb}k characters).
This is NOT the full novel — it is only the beginning. When selecting characters, you must
consider their importance throughout the ENTIRE novel, including characters who appear more
prominently in later chapters not shown here.

---
{text}
---

Identify the {n_characters} most psychologically important characters in "{novel_display}" who:
- Play a central role in the overall plot and themes
- Undergo meaningful psychological development or change
- Appear significantly throughout the full novel (not just the opening)

For each character provide:
1. Their full name as it appears in the novel (canonical form)
2. A short, lowercase, underscore-separated directory slug (e.g. "firstname_lastname", "short_name")

Return JSON only, no other text:
{{
  "characters": [
    {{"name": "Full Character Name", "dir": "short_dir_slug"}},
    ...
  ]
}}"""


async def _extract_raw(
    client: AsyncOpenAI, model_name: str,
    novel_display: str, opening: str, n_characters: int,
) -> list[dict]:
    """Ask the LLM for the top N characters. Returns [{name, dir}, ...]."""
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": _build_extract_prompt(
                novel_display, opening, n_characters)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    data = json.loads(response.choices[0].message.content)
    picked: list[dict] = []
    for c in data.get("characters", []):
        name = (c.get("name") or "").strip()
        slug = (c.get("dir") or "").strip()
        if name and slug:
            picked.append({"name": name, "dir": slug})
    return picked


# ── Stage 2: verification ────────────────────────────────────────────────────
_VERIFY_SYSTEM = (
    "You verify whether a name is a real character of a given novel. "
    "You judge only from the novel itself, never from outside knowledge."
)


def _in_text(name: str, text_lower: str) -> bool:
    """Ctrl+F: case-insensitive substring search, whitespace-normalised."""
    needle = " ".join(name.lower().split())
    return bool(needle) and needle in text_lower


def _build_judge_prompt(novel_display: str, names: list[str], excerpt: str) -> str:
    listing = "\n".join(f"  - {n}" for n in names)
    return f"""Novel: {novel_display}

The names below were extracted as characters of this novel, but none of them
appears verbatim in the novel's text. For EACH name decide one of:

  "real"            — it IS a genuine character of this novel; the text merely
                      spells the name differently (another romanisation,
                      translation, nickname, title, honorific, or name form).
  "not_a_character" — it does NOT correspond to any real character of this
                      novel (an invented or wrong name, or not a person).

Be conservative: if a name plausibly refers to a real character under a
different spelling, judge it "real". Only use "not_a_character" when you are
confident no such character exists in this novel.

Opening of the novel (for reference):
---
{excerpt}
---

Names to judge:
{listing}

Return JSON only, no other text:
{{"verdicts": [{{"name": "<name exactly as given>", "verdict": "real" | "not_a_character", "reason": "<brief>"}}]}}"""


async def _verify_characters(
    client: AsyncOpenAI, model_name: str,
    novel_display: str, picked: list[dict], full_text: str,
) -> tuple[list[dict], list[dict]]:
    """Check each picked character against the full novel text.

    Returns (kept, flagged): `kept` is the subset of `picked` to keep; `flagged`
    is a {name, verdict, reason} record for every name not found verbatim.
    """
    text_lower = " ".join(full_text.lower().split())
    flagged_names = [c["name"] for c in picked if not _in_text(c["name"], text_lower)]

    verdicts: dict[str, tuple[str, str]] = {}
    if flagged_names:
        prompt = _build_judge_prompt(
            novel_display, flagged_names, full_text[:VERIFY_EXCERPT_CHARS])
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": _VERIFY_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            for v in json.loads(response.choices[0].message.content)["verdicts"]:
                verdicts[v["name"]] = (v.get("verdict"), v.get("reason", ""))
        except Exception as e:
            # On any failure, keep everything — never drop a character on error.
            logger.warning(f"[CharacterExtractor] verify call failed: {e}; keeping all.")
            verdicts = {}

    flagged: list[dict] = []
    dropped: set[str] = set()
    for name in flagged_names:
        verdict, reason = verdicts.get(name, ("real", "judge unavailable — kept"))
        if verdict != "not_a_character":
            verdict = "real"
        else:
            dropped.add(name)
        flagged.append({"name": name, "verdict": verdict, "reason": reason})

    kept = [c for c in picked if c["name"] not in dropped]
    return kept, flagged


# ── Public entry point ───────────────────────────────────────────────────────
async def extract_characters(
    client: AsyncOpenAI,
    model_name: str,
    novel_name: str,
    novel_path: str,
    results_base: str,
    n_characters: int = 10,
    force: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Extract the top N major characters from the novel, then verify them.

    Characters whose name cannot be found in the novel text and which an LLM
    judges to be no real character are dropped. Results are cached to
    characters.json and reused on subsequent runs (pass force=True to bypass).

    Returns:
        (target_characters, character_dirs)
    """
    if not force:
        cached = _load_cache(results_base)
        if cached is not None:
            logger.info(f"[CharacterExtractor] Loaded from cache: {cached[0]}")
            print(f"[CharacterExtractor] Using cached characters: {cached[0]}")
            return cached

    if not os.path.exists(novel_path):
        raise FileNotFoundError(
            f"Novel text not found: {novel_path}\n"
            f"Place the novel as a .txt file at that path before running."
        )

    with open(novel_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    opening = full_text[:EXTRACTION_CHAR_LIMIT]

    novel_display = novel_name.replace("_", " ").title()
    print(f"[CharacterExtractor] Extracting top {n_characters} characters for '{novel_display}'...")

    # Stage 1 — extraction
    picked = await _extract_raw(client, model_name, novel_display, opening, n_characters)
    if not picked:
        raise RuntimeError(
            f"[CharacterExtractor] LLM returned no characters for '{novel_display}'."
        )
    print(f"[CharacterExtractor] Extracted: {[c['name'] for c in picked]}")

    # Stage 2 — verification against the full text
    kept, flagged = await _verify_characters(
        client, model_name, novel_display, picked, full_text)
    for f in flagged:
        tag = "DROP" if f["verdict"] == "not_a_character" else "keep"
        print(f"[CharacterExtractor]   verify [{tag}] {f['name']}  —  {f['reason']}")

    if not kept:
        raise RuntimeError(
            f"[CharacterExtractor] Every extracted character for '{novel_display}' "
            f"was rejected by verification."
        )

    target_characters = [c["name"] for c in kept]
    character_dirs = {c["name"]: c["dir"] for c in kept}

    _save_cache(results_base, target_characters, character_dirs)
    logger.info(f"[CharacterExtractor] Verified characters: {target_characters}")
    print(f"[CharacterExtractor] Final verified characters: {target_characters}")
    return target_characters, character_dirs


if __name__ == "__main__":
    import asyncio
    import sys

    novel = sys.argv[1] if len(sys.argv) > 1 else "Benjamin_Franklin"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    novel_path = os.path.join(project_root, "data", "novels", f"{novel}.txt")
    results_base = (
        sys.argv[2] if len(sys.argv) > 2
        else os.path.join(project_root, "results", "arc_extraction", novel)
    )
    n_chars = int(os.environ.get("N_CHARACTERS", "5"))
    model = os.environ.get("ARC_MODEL", "gpt-5.4-mini")

    async def _main() -> None:
        client = AsyncOpenAI(api_key=require_openai_key())
        print(f"=== character_extractor: '{novel}' (model={model}, n={n_chars}) ===")
        chars, dirs = await extract_characters(
            client, model, novel, novel_path, results_base,
            n_characters=n_chars, force=True,
        )
        print("\n=== RESULT ===")
        for name in chars:
            print(f"  {name}  ->  {dirs[name]}")

    asyncio.run(_main())
