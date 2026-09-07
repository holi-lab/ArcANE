"""Phase 3: Literary grounding validation via 3-LLM ensemble.

Three independent LLM critics — each with a different scholarly perspective —
evaluate every axis from the final extraction. Each critic must cite specific
published scholarship (author, title, year, URL when known). Verdicts are
aggregated:

  valid_axis           2 or 3 critics confirm with specific citations
  partially_valid_axis exactly 1 critic confirms
  none_axis            0 critics confirm (no scholarly grounding found)

Saves two output sets to results/arc_extraction/{novel}/:
  final_grounded/  — all axes with literary_validation tags (transparent record)
  final_auto/      — only valid_axis axes (0–1 critic confirmed axes removed)

Original final/ files are NOT modified.

Usage:
    python -m arc_construction.phase3_literary_grounding
    NOVEL_NAME=pride_and_prejudice python -m arc_construction.phase3_literary_grounding

Steps (each requires your confirmation before proceeding):
  Cost estimate — show API calls + estimated cost, confirm to proceed
  Step 1 — Load final axes and show summary
  Step 2 — Preview axes to evaluate
  Step 3 — Run 3-LLM ensemble (concurrent per axis)
  Step 4 — Review aggregated verdicts and citations
  Step 5 — Save tagged axes to final_grounded/ and filtered axes to final_auto/
"""

import asyncio
import json
import os
import logging
from urllib.parse import quote_plus

from openai import AsyncOpenAI

from arc_construction import config
from arc_construction.config import (
    require_openai_key, MODEL_NAME,
    MAX_CONCURRENT_API_CALLS, MAX_RETRIES, RETRY_BACKOFF_BASE,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
EVALUATION_TEMPERATURE = 0.4
GROUNDED_DIR_NAME = "final_grounded"
AUTO_DIR_NAME = "final_auto"

# Rough token estimates per call (system + prompt + response)
EST_INPUT_TOKENS_PER_CALL = 1_000
EST_OUTPUT_TOKENS_PER_CALL = 400

# Three independent critical lenses
EVALUATORS = [
    {
        "id": "structuralist",
        "label": "Structuralist / Narratologist",
        "system": (
            "You are a structuralist literary critic with expertise in narratology and "
            "character function. You evaluate character arc claims by consulting your "
            "knowledge of published narratological and structuralist analyses. "
            "You must cite specific, real, verifiable works. "
            "For each citation include the direct URL to the work if you know it "
            "(e.g. JSTOR, Google Books, publisher page, Academia.edu), otherwise null. "
            "If you cannot recall any real published citation for this axis, "
            "set verdict=false and leave citations as an empty list."
        ),
    },
    {
        "id": "psychological",
        "label": "Psychological Critic",
        "system": (
            "You are a psychological literary critic specializing in depth psychology, "
            "ego development, and psychoanalytic approaches to character. "
            "You evaluate character arc claims using your knowledge of published "
            "psychological criticism of the novel. "
            "You must cite specific, real, verifiable works. "
            "For each citation include the direct URL to the work if you know it, "
            "otherwise null. "
            "If you cannot recall any real published citation for this axis, "
            "set verdict=false and leave citations as an empty list."
        ),
    },
    {
        "id": "historical_cultural",
        "label": "Historical / Cultural Critic",
        "system": (
            "You are a historical and cultural literary critic. You evaluate character arc "
            "claims using your knowledge of published cultural, historical, and "
            "reception-oriented scholarship on the novel. "
            "You must cite specific, real, verifiable works. "
            "For each citation include the direct URL to the work if you know it, "
            "otherwise null. "
            "If you cannot recall any real published citation for this axis, "
            "set verdict=false and leave citations as an empty list."
        ),
    },
]


# ── UI helpers ─────────────────────────────────────────────────────────────────

def confirm(prompt: str) -> bool:
    answer = input(f"\n{prompt} [y/N] ").strip().lower()
    return answer in ("y", "yes")


def divider(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


# ── Citation helpers ───────────────────────────────────────────────────────────

def _make_search_url(author: str, title: str, year: int | str) -> str:
    q = quote_plus(f"{author} {title} {year}")
    return f"https://scholar.google.com/scholar?q={q}"


def _enrich_citations(citations: list[dict]) -> list[dict]:
    enriched = []
    for cit in citations:
        c = dict(cit)
        c.setdefault("url", None)
        c["search_url"] = _make_search_url(
            c.get("author", ""),
            c.get("title", ""),
            c.get("year", ""),
        )
        enriched.append(c)
    return enriched


# ── Cost estimation ────────────────────────────────────────────────────────────

def _count_axes(axes: dict[str, dict]) -> int:
    return sum(
        len(d.get("intrapersonal_axes", [])) + len(d.get("relational_axes", []))
        for d in axes.values()
    )


def _estimate_cost(n_axes: int) -> tuple[int, float]:
    """Return (total_api_calls, estimated_usd)."""
    n_calls = n_axes * len(EVALUATORS)
    total_input = n_calls * EST_INPUT_TOKENS_PER_CALL
    total_output = n_calls * EST_OUTPUT_TOKENS_PER_CALL

    # gpt-5.4-mini pricing: $0.40/1M input, $1.60/1M output
    cost = (total_input / 1_000_000 * 0.40) + (total_output / 1_000_000 * 1.60)
    return n_calls, cost


def show_cost_estimate(axes: dict[str, dict]) -> bool:
    divider("Cost Estimate")

    n_axes = _count_axes(axes)
    n_calls, est_cost = _estimate_cost(n_axes)

    novel = config.NOVEL_NAME.replace("_", " ").title()
    print(f"  Novel            : {novel}")
    print(f"  Characters       : {len(axes)}")
    print(f"  Total axes       : {n_axes}")
    print(f"  Evaluators       : {len(EVALUATORS)}")
    print("  ─────────────────────────────────────")
    print(f"  Total API calls  : {n_calls}  ({n_axes} axes × {len(EVALUATORS)} critics)")
    print(f"  Est. input tokens: ~{n_calls * EST_INPUT_TOKENS_PER_CALL:,}")
    print(f"  Est. output tokens: ~{n_calls * EST_OUTPUT_TOKENS_PER_CALL:,}")
    print(f"  Model            : {MODEL_NAME}")
    print("  ─────────────────────────────────────")
    print(f"  Estimated cost   : ~${est_cost:.3f}  (actual may vary)")
    print()
    print("  Note: axes with no verifiable citations are automatically tagged none_axis.")

    return confirm("Proceed with Phase 3?")


# ── Step 1: Load axes ──────────────────────────────────────────────────────────

def load_final_axes() -> dict[str, dict]:
    if not os.path.exists(config.FINAL_DIR):
        raise FileNotFoundError(
            f"Final directory not found: {config.FINAL_DIR}\n"
            "Run run_all.py first to generate final axes."
        )
    axes: dict[str, dict] = {}
    for fname in sorted(os.listdir(config.FINAL_DIR)):
        if fname.endswith("_final_axes.json") and not fname.startswith("all_"):
            path = os.path.join(config.FINAL_DIR, fname)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            char = data.get("character", fname.replace("_final_axes.json", ""))
            axes[char] = data
    return axes


def step1_load_and_show(axes: dict[str, dict]) -> bool:
    divider("STEP 1: Load Final Axes")

    for char, data in axes.items():
        n_i = len(data.get("intrapersonal_axes", []))
        n_r = len(data.get("relational_axes", []))
        print(f"  {char}: {n_i} intrapersonal + {n_r} relational = {n_i + n_r} axes")

    print(
        "\n  Aggregation rule:\n"
        "    valid_axis           — 2–3 critics confirm with citations\n"
        "    partially_valid_axis — exactly 1 critic confirms\n"
        "    none_axis            — 0 critics confirm\n\n"
        "  Output:\n"
        "    final_grounded/ — all axes with validation tags (transparent record)\n"
        "    final_auto/     — only valid_axis axes (0–1 critic confirmed axes removed)\n"
        "  Original final/ files are NOT changed."
    )
    return confirm("Proceed with Step 1?")


# ── Step 2: Preview axes ───────────────────────────────────────────────────────

def step2_preview(axes: dict[str, dict]) -> bool:
    divider("STEP 2: Preview Axes to Evaluate")

    for char, data in axes.items():
        print(f"\n  [{char}]")
        for ax in data.get("intrapersonal_axes", []):
            print(f"    [{ax['axis_id']}] (intra)      {ax['axis_name']}")
        for ax in data.get("relational_axes", []):
            target = ax.get("target_character", "?")
            print(f"    [{ax['axis_id']}] (rel→{target})  {ax['axis_name']}")

    return confirm("Proceed with Step 2 (axes look correct)?")


# ── Step 3: 3-LLM ensemble evaluation ─────────────────────────────────────────

def _build_eval_prompt(novel_title: str, character: str, axis: dict) -> str:
    is_relational = axis.get("axis_type") == "relational"

    if is_relational:
        target = axis.get("target_character", "?")
        axis_type_block = (
            f"  Target character : {target}\n\n"
            f"AXIS TYPE: RELATIONAL\n"
            f"  This axis tracks how {character}'s relationship with {target} changes\n"
            f"  across the novel (e.g. trust, power dynamics, emotional intimacy, hostility).\n"
            f"  Relevant scholarship includes: relationship studies, gender criticism,\n"
            f"  social dynamics analysis, or any criticism discussing the {character}–{target}\n"
            f"  relationship arc specifically."
        )
        evidence_guidance = (
            f"3. For relational axes, a work supports the axis if it discusses the "
            f"evolving dynamic between {character} and {target} — even if it uses "
            f"different terms (e.g. 'power struggle', 'romantic tension', 'class conflict')."
        )
    else:
        axis_type_block = (
            f"AXIS TYPE: INTRAPERSONAL\n"
            f"  This axis tracks an internal psychological change in {character}\n"
            f"  (beliefs, self-perception, moral reasoning, emotional patterns).\n"
            f"  Relevant scholarship includes: psychological criticism, character\n"
            f"  development studies, ego/identity analysis, or psychoanalytic readings."
        )
        evidence_guidance = (
            f"3. For intrapersonal axes, a work supports the axis if it discusses "
            f"the same internal dimension of {character} — even under different terminology "
            f"(e.g. 'pride' vs 'arrogance', 'humility' vs 'self-knowledge')."
        )

    return f"""Novel: {novel_title}
Character: {character}

AXIS UNDER EVALUATION
  ID          : {axis.get('axis_id')}
  Dimension   : {axis.get('dimension_label')}
  Name        : {axis.get('axis_name')}
  Pole start  : {axis.get('pole_start')}
  Pole end    : {axis.get('pole_end')}
  Confidence  : {axis.get('confidence')}

{axis_type_block}

YOUR TASK
Assess whether this axis is grounded in published literary scholarship on "{novel_title}".

Rules:
1. Cite only real, verifiable published works. Never invent citations.
2. Each citation MUST include: author, title, year.
   Include the direct URL (JSTOR, Google Books, publisher page, Academia.edu, etc.)
   if you know it; otherwise set url to null.
{evidence_guidance}
4. If you cannot recall any real published work supporting this axis,
   set "verdict": false and "citations": [].

Return JSON only:
{{
  "verdict": true or false,
  "reasoning": "2-3 sentences from your critical perspective",
  "citations": [
    {{
      "author": "Surname, Firstname (or just Surname)",
      "title": "Full work title",
      "year": 1990,
      "publication": "Journal name / book publisher / essay collection",
      "url": "direct URL or null",
      "relevance": "one sentence: how this work supports the axis"
    }}
  ]
}}"""


async def _call_api(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    system: str,
    user: str,
) -> dict:
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
                    temperature=EVALUATION_TEMPERATURE,
                )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            wait = RETRY_BACKOFF_BASE ** (attempt + 1)
            logger.warning(f"API retry {attempt+1}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(wait)
            else:
                raise


async def _evaluate_axis(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    novel_title: str,
    character: str,
    axis: dict,
) -> dict:
    prompt = _build_eval_prompt(novel_title, character, axis)
    tasks = {
        ev["id"]: asyncio.create_task(_call_api(client, sem, ev["system"], prompt))
        for ev in EVALUATORS
    }
    results = {}
    for ev_id, task in tasks.items():
        raw = await task
        raw["citations"] = _enrich_citations(raw.get("citations", []))
        results[ev_id] = raw
    return results


def _aggregate(ev_results: dict) -> str:
    confirmed = sum(
        1
        for ev in EVALUATORS
        if ev_results.get(ev["id"], {}).get("verdict") is True
        and len(ev_results.get(ev["id"], {}).get("citations", [])) > 0
    )
    if confirmed >= 2:
        return "valid_axis"
    elif confirmed == 1:
        return "partially_valid_axis"
    return "none_axis"


async def step3_evaluate(axes: dict[str, dict]) -> dict[str, dict] | None:
    divider("STEP 3: 3-LLM Ensemble Evaluation")

    n_axes = _count_axes(axes)
    n_calls, _ = _estimate_cost(n_axes)
    print(f"  {n_axes} axes × {len(EVALUATORS)} evaluators = {n_calls} API calls")
    print(f"  Model: {MODEL_NAME}  |  Max concurrent: {MAX_CONCURRENT_API_CALLS}")

    if not confirm("Proceed with Step 3 (start LLM evaluation)?"):
        return None

    client = AsyncOpenAI(api_key=require_openai_key())
    sem = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
    novel_title = config.NOVEL_NAME.replace("_", " ").title()

    all_results: dict[str, dict] = {}

    for char, data in axes.items():
        print(f"\n  Evaluating [{char}]...")
        char_results: dict[str, dict] = {}
        all_axes = data.get("intrapersonal_axes", []) + data.get("relational_axes", [])

        for axis in all_axes:
            axis_id = axis["axis_id"]
            ev_results = await _evaluate_axis(client, sem, novel_title, char, axis)
            char_results[axis_id] = ev_results

            tag = _aggregate(ev_results)
            verdicts = "".join(
                "✓" if ev_results[ev["id"]].get("verdict") and ev_results[ev["id"]].get("citations") else "✗"
                for ev in EVALUATORS
            )
            print(f"    [{axis_id}] {verdicts} → {tag}")

        all_results[char] = char_results

    return all_results


# ── Step 4: Review results ─────────────────────────────────────────────────────

def step4_review(axes: dict[str, dict], all_eval_results: dict[str, dict]) -> bool:
    divider("STEP 4: Review Aggregated Verdicts")

    for char, data in axes.items():
        print(f"\n  [{char}]")
        all_axes = data.get("intrapersonal_axes", []) + data.get("relational_axes", [])
        for axis in all_axes:
            axis_id = axis["axis_id"]
            ev_results = all_eval_results.get(char, {}).get(axis_id, {})
            tag = _aggregate(ev_results)
            tag_label = {
                "valid_axis": "VALID        ",
                "partially_valid_axis": "PARTIAL      ",
                "none_axis": "NONE (remove)",
            }[tag]
            print(f"\n    [{axis_id}] {tag_label}  {axis['axis_name']}")
            for ev in EVALUATORS:
                ev_id = ev["id"]
                result = ev_results.get(ev_id, {})
                verdict = result.get("verdict", False)
                citations = result.get("citations", [])
                icon = "✓" if verdict and citations else "✗"
                print(f"      {icon} {ev['label']:<35} ({len(citations)} citation(s))")
                reasoning = result.get("reasoning", "")[:90]
                if reasoning:
                    print(f"        {reasoning}")
                for cit in citations:
                    url_display = cit.get("url") or cit.get("search_url", "")
                    print(
                        f"        → {cit.get('author','?')}, "
                        f"\"{cit.get('title','?')}\" ({cit.get('year','?')})"
                    )
                    if url_display:
                        print(f"          {url_display}")

    return confirm("Proceed with Step 4 (results look correct)?")


# ── Step 5: Save ───────────────────────────────────────────────────────────────

def step5_save(axes: dict[str, dict], all_eval_results: dict[str, dict]) -> bool:
    divider("STEP 5: Save Tagged Axes to final_grounded/ and final_auto/")

    for char, data in axes.items():
        all_ax = data.get("intrapersonal_axes", []) + data.get("relational_axes", [])
        c = {"valid_axis": 0, "partially_valid_axis": 0, "none_axis": 0}
        for ax in all_ax:
            c[_aggregate(all_eval_results.get(char, {}).get(ax["axis_id"], {}))] += 1
        removed = c["partially_valid_axis"] + c["none_axis"]
        print(
            f"  {char}: valid={c['valid_axis']}  "
            f"partial={c['partially_valid_axis']}  none={c['none_axis']}  "
            f"(→ final_auto keeps {c['valid_axis']}, removes {removed})"
        )

    grounded_dir = os.path.join(config.RESULTS_BASE, GROUNDED_DIR_NAME)
    auto_dir = os.path.join(config.RESULTS_BASE, AUTO_DIR_NAME)
    print(f"\n  Output (full)     : {grounded_dir}/")
    print(f"  Output (filtered) : {auto_dir}/")
    print("  final_grounded/ : all axes with validation_tag (transparent record).")
    print("  final_auto/     : only valid_axis axes (partially_valid + none removed).")
    print("  Original final/ files are untouched.")

    if not confirm("Proceed with Step 5 (save)?"):
        return False

    _save_no_confirm(axes, all_eval_results)
    return True


# ── Non-interactive entry point (called from run_all.py) ─────────────────────

async def run_silent():
    """Run Phase 3 end-to-end with no confirmation prompts."""
    from arc_construction.character_extractor import load_cached_characters
    target_chars, char_dirs = load_cached_characters(config.RESULTS_BASE)
    config.init(config.NOVEL_NAME, characters=target_chars, character_dirs=char_dirs)

    axes = load_final_axes()
    n_axes = _count_axes(axes)
    n_calls, est_cost = _estimate_cost(n_axes)
    novel_title = config.NOVEL_NAME.replace("_", " ").title()
    print(f"\n  Phase 3: {novel_title}  |  {n_axes} axes × {len(EVALUATORS)} critics "
          f"= {n_calls} API calls  (~${est_cost:.3f})")

    client = AsyncOpenAI(api_key=require_openai_key())
    sem = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
    all_results: dict[str, dict] = {}

    for char, data in axes.items():
        print(f"\n  Evaluating [{char}]...")
        char_results: dict[str, dict] = {}
        for axis in data.get("intrapersonal_axes", []) + data.get("relational_axes", []):
            axis_id = axis["axis_id"]
            ev_results = await _evaluate_axis(client, sem, novel_title, char, axis)
            char_results[axis_id] = ev_results
            tag = _aggregate(ev_results)
            verdicts = "".join(
                "✓" if ev_results[ev["id"]].get("verdict") and ev_results[ev["id"]].get("citations") else "✗"
                for ev in EVALUATORS
            )
            print(f"    [{axis_id}] {verdicts} → {tag}")
        all_results[char] = char_results

    _save_no_confirm(axes, all_results)


def _save_no_confirm(axes: dict[str, dict], all_eval_results: dict[str, dict]):
    """Write complete and critic-filtered axis artifacts without prompting."""
    grounded_dir = os.path.join(config.RESULTS_BASE, GROUNDED_DIR_NAME)
    auto_dir = os.path.join(config.RESULTS_BASE, AUTO_DIR_NAME)
    os.makedirs(grounded_dir, exist_ok=True)
    os.makedirs(auto_dir, exist_ok=True)

    for char, data in axes.items():
        ev_char = all_eval_results.get(char, {})
        dir_slug = config.CHARACTER_DIRS.get(char, char.lower().replace(" ", "_"))

        def tag_axes(axis_list: list) -> list:
            tagged = []
            for axis in axis_list:
                axis_id = axis["axis_id"]
                ev_results = ev_char.get(axis_id, {})
                axis_copy = dict(axis)
                axis_copy["literary_validation"] = {
                    "validation_tag": _aggregate(ev_results),
                    "evaluators": {
                        ev["id"]: {
                            "label": ev["label"],
                            "verdict": ev_results.get(ev["id"], {}).get("verdict", False),
                            "reasoning": ev_results.get(ev["id"], {}).get("reasoning", ""),
                            "citations": ev_results.get(ev["id"], {}).get("citations", []),
                        }
                        for ev in EVALUATORS
                    },
                }
                tagged.append(axis_copy)
            return tagged

        grounded_data = {
            "character": data.get("character", char),
            "arc_richness": data.get("arc_richness", ""),
            "intrapersonal_axes": tag_axes(data.get("intrapersonal_axes", [])),
            "relational_axes": tag_axes(data.get("relational_axes", [])),
        }
        grounded_path = os.path.join(grounded_dir, f"{dir_slug}_grounded_axes.json")
        with open(grounded_path, "w", encoding="utf-8") as f:
            json.dump(grounded_data, f, indent=2, ensure_ascii=False)
        print(f"  [grounded] Saved: {grounded_path}")

        def filter_valid(tagged_list: list) -> list:
            return [
                ax for ax in tagged_list
                if ax.get("literary_validation", {}).get("validation_tag") == "valid_axis"
            ]

        auto_data = {
            "character": data.get("character", char),
            "arc_richness": data.get("arc_richness", ""),
            "intrapersonal_axes": filter_valid(grounded_data["intrapersonal_axes"]),
            "relational_axes": filter_valid(grounded_data["relational_axes"]),
        }
        auto_path = os.path.join(auto_dir, f"{dir_slug}_auto_axes.json")
        with open(auto_path, "w", encoding="utf-8") as f:
            json.dump(auto_data, f, indent=2, ensure_ascii=False)
        n_kept = len(auto_data["intrapersonal_axes"]) + len(auto_data["relational_axes"])
        print(f"  [auto]     Saved: {auto_path}  ({n_kept} axes kept)")

    combined_grounded: dict = {}
    for fname in sorted(os.listdir(grounded_dir)):
        if fname.endswith("_grounded_axes.json"):
            with open(os.path.join(grounded_dir, fname), encoding="utf-8") as f:
                d = json.load(f)
            combined_grounded[d.get("character", fname)] = d
    with open(os.path.join(grounded_dir, "all_characters_grounded.json"), "w", encoding="utf-8") as f:
        json.dump(combined_grounded, f, indent=2, ensure_ascii=False)

    combined_auto: dict = {}
    for fname in sorted(os.listdir(auto_dir)):
        if fname.endswith("_auto_axes.json"):
            with open(os.path.join(auto_dir, fname), encoding="utf-8") as f:
                d = json.load(f)
            combined_auto[d.get("character", fname)] = d
    with open(os.path.join(auto_dir, "all_characters_auto.json"), "w", encoding="utf-8") as f:
        json.dump(combined_auto, f, indent=2, ensure_ascii=False)

    print("\n  Phase 3 complete.")
    print(f"    full record   → {grounded_dir}/")
    print(f"    auto-filtered → {auto_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    print("\nPhase 3: Literary Grounding Validation")
    print(f"Novel : {config.NOVEL_NAME.replace('_', ' ').title()}")
    print(f"Model : {MODEL_NAME}")

    from arc_construction.character_extractor import load_cached_characters
    target_chars, char_dirs = load_cached_characters(config.RESULTS_BASE)
    config.init(config.NOVEL_NAME, characters=target_chars, character_dirs=char_dirs)

    # Load axes early — needed for cost estimate
    axes = load_final_axes()

    if not show_cost_estimate(axes):
        print("Aborted."); return
    if not step1_load_and_show(axes):
        print("Aborted."); return
    if not step2_preview(axes):
        print("Aborted."); return

    all_eval_results = await step3_evaluate(axes)
    if all_eval_results is None:
        print("Aborted."); return

    if not step4_review(axes, all_eval_results):
        print("Aborted."); return
    if not step5_save(axes, all_eval_results):
        print("Aborted."); return


if __name__ == "__main__":
    asyncio.run(main())
