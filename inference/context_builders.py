"""Build the `<context>` block for each mode.

A builder's signature is `(trial) -> Optional[str]`. Returning None (or
empty) makes the prompt renderer skip the context block entirely — useful
for vanilla and for trials where the relevant cache is missing.

`arc`     reads `final_auto/{slug}_auto_axes.json` and serialises the curated
          trajectory + pole information **truncated at `trial.query_chapter`**:
          only phases whose `chapter_range` has started by that chapter are
          surfaced. When later phases are hidden, `pole_end` and `arc_direction`
          are also stripped (both describe the endpoint and are future
          information at this time-point). `literary_validation` and
          `evidence_summary` are always stripped — the former is a bibliography
          for human review; the latter summarises the *whole* arc and so leaks
          the ending.

`summary` concatenates precomputed per-chapter summaries for the most
          recent `SUMMARY_MAX_CHAPTERS` chapters up to and including
          `query_chapter` (default: last 5). Earlier chapters are dropped to
          keep the context tractable on long novels — running the whole
          history through every probe was costing 80k+ tokens on Jane Eyre /
          Pride and Prejudice. Build the per-chapter cache with
          `inference.build_summaries`.

`rag`     retrieves top-k chunks from the precomputed embedding index.
          Build the index with `inference.build_index`.

`timechara` is a hint-injection mode (no system-side context block). It runs
          a two-stage classifier — temporal (which chapter is this scene?)
          and spatial (is the character present?) — using the same chat
          model, and appends a "(HINT: ...)" warning to the user message when
          the scene appears to be in the character's future or the character
          was absent. Port of ahnjaewoo/timechara's narrative-experts.

`lifechoice` is a port of CHARMAP (arxiv:2404.12138 §4.1). Description =
          chapter summaries up to `query_chapter` (reuses `build_summary`).
          Memory     = top-k RAG chunks retrieved with a *description-
                       augmented* query — `embed(description + scenario +
                       question)` — which is CHARMAP's contribution over
                       plain `rag`. Both halves are concatenated under
                       `[Character Description]` / `[Relevant Memory]`
                       section headers inside the `<context>` block.
"""

import json
import re
from functools import lru_cache
from typing import Optional

import numpy as np

from . import config
from .loaders import Trial, load_axes_file, list_chapters


# ── vanilla ────────────────────────────────────────────────────────
def build_vanilla(trial: Trial) -> Optional[str]:
    return None


# ── arc ────────────────────────────────────────────────────────────
def _curate_axes(raw: dict, query_chapter: int) -> dict:
    """Drop spoiler-prone fields and any phase information that postdates
    `query_chapter`. `literary_validation` and `evidence_summary` are always
    removed (the latter summarises the whole arc, ending included). A phase is
    kept iff its `chapter_range` has already started by `query_chapter`
    (i.e. `chapter_range[0] <= query_chapter`).

    When at least one later phase is hidden, `pole_end` and `arc_direction`
    are also stripped from that axis — both describe where the trajectory
    lands, which is future information at this time-point. `axis_name` and
    `dimension_label` are retained as high-level framing; they serve as the
    axis identifier and would be awkward to redact.
    """
    out: dict = {
        "character": raw.get("character"),
        "intrapersonal_axes": [],
        "relational_axes": [],
    }
    for key in ("intrapersonal_axes", "relational_axes"):
        for ax in raw.get(key, []):
            curated = {k: v for k, v in ax.items()
                       if k not in ("literary_validation", "evidence_summary")}
            trajectory = curated.get("trajectory") or []
            kept = [
                ph for ph in trajectory
                if (ph.get("chapter_range") or [10**9, 10**9])[0] <= query_chapter
            ]
            curated["trajectory"] = kept
            if len(kept) < len(trajectory):
                curated.pop("pole_end", None)
                curated.pop("arc_direction", None)
            out[key].append(curated)
    return out


@lru_cache(maxsize=512)
def _arc_json(novel: str, character_slug: str, query_chapter: int) -> str:
    raw = load_axes_file(novel, character_slug)
    return json.dumps(_curate_axes(raw, query_chapter), indent=2, ensure_ascii=False)


def build_arc(trial: Trial) -> Optional[str]:
    return _arc_json(trial.novel, trial.character_slug, trial.query_chapter)


# ── summary ────────────────────────────────────────────────────────
def build_summary(trial: Trial) -> Optional[str]:
    """Concatenate chapter summaries for the last `SUMMARY_MAX_CHAPTERS`
    chapters up to and including `query_chapter`.

    Older chapters are intentionally dropped: passing the entire chapter
    history (50+ chapters on long novels) blew past 80k tokens per probe.
    The recent-window heuristic keeps prompts under a couple of thousand
    tokens (5-chapter window) while still giving the model the most
    relevant scene context.

    Skips any chapter whose summary file is missing (caller should run the
    `build_summaries` CLI first; missing chapters in-window are reported
    inline but non-fatal).
    """
    start_ch = max(1, trial.query_chapter - config.SUMMARY_MAX_CHAPTERS + 1)
    pieces: list[str] = []
    missing: list[int] = []
    for ch in range(start_ch, trial.query_chapter + 1):
        p = config.summary_path(trial.novel, ch)
        if not p.exists():
            missing.append(ch)
            continue
        pieces.append(f"[Chapter {ch}]\n{p.read_text(encoding='utf-8').strip()}")
    if missing:
        # surface the gap rather than silently degrading
        pieces.append(f"[note] Missing summaries for chapters: {missing}")
    return "\n\n".join(pieces) if pieces else None


# ── rag ────────────────────────────────────────────────────────────
@lru_cache(maxsize=8)
def _load_rag_index(novel: str) -> tuple[list[dict], np.ndarray]:
    """Return (chunks, embeddings). chunks[i] mirrors embeddings[i]."""
    rdir = config.rag_dir(novel)
    chunks_p = rdir / "chunks.jsonl"
    embs_p   = rdir / "embeddings.npy"
    if not chunks_p.exists() or not embs_p.exists():
        raise FileNotFoundError(
            f"RAG index missing for novel={novel!r}. Build with "
            f"`uv run python -m inference.build_index --novel {novel!r}`"
        )
    chunks = [json.loads(l) for l in chunks_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    embs = np.load(embs_p)
    assert len(chunks) == embs.shape[0], "chunk/embedding length mismatch"
    return chunks, embs


def _cosine_topk(query_vec: np.ndarray, embs: np.ndarray, k: int,
                 chapter_mask: np.ndarray) -> list[int]:
    """Top-k cosine indices, restricted to rows where chapter_mask is True."""
    if not chapter_mask.any():
        return []
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    en = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    sims = en @ qn
    sims = np.where(chapter_mask, sims, -np.inf)
    return np.argsort(-sims)[:k].tolist()


async def build_rag_async(embed_client, sem, trial: Trial) -> Optional[str]:
    """RAG context requires an async embedding call for the query string.

    The runner has a sync context-builder interface for vanilla/arc/summary,
    so RAG is handled specially via `RAG_REQUIRES_EMBED`. The runner calls
    this coroutine directly with the embed backend's client (which may
    differ from the chat client when chat is served by vLLM and embeddings
    by OpenAI).
    """
    from ._api import embed_batch

    chunks, embs = _load_rag_index(trial.novel)
    mask = np.array([c["chapter"] <= trial.query_chapter for c in chunks])
    query = f"{trial.scenario}\n\n{trial.question}"
    [qv_raw] = await embed_batch(embed_client, sem, config.EXP_EMBED_MODEL, [query])
    qv = np.array(qv_raw, dtype=np.float32)
    idxs = _cosine_topk(qv, embs, config.RAG_TOP_K, mask)
    if not idxs:
        return None
    pieces = []
    for i in idxs:
        c = chunks[i]
        pieces.append(f"[Chapter {c['chapter']} excerpt]\n{c['text'].strip()}")
    return "\n\n".join(pieces)


# ── lifechoice ─────────────────────────────────────────────────────
# Port of CHARMAP (Xu et al. 2024, arxiv:2404.12138 §4.1). Two-block context:
# the chapter-summary description plays the role of CHARMAP's character
# Description, and we retrieve Memory chunks with a description-augmented
# embedding query (CHARMAP's contribution over plain RAG).

# text-embedding-3-small caps at 8192 tokens (~32k chars). Truncate the
# augmented query as a safety net — last 10 chapter summaries on long novels
# can be a few thousand tokens already; description + scenario + question
# could brush the limit.
_LIFECHOICE_MAX_QUERY_CHARS = 30_000


async def build_lifechoice_async(embed_client, sem, trial: Trial) -> Optional[str]:
    from ._api import embed_batch

    description = build_summary(trial)
    if not description:
        return None  # precheck warns; downstream renders without context

    query = (
        f"Character profile:\n{description}\n\n"
        f"Scenario:\n{trial.scenario}\n\n"
        f"Question:\n{trial.question}"
    )
    if len(query) > _LIFECHOICE_MAX_QUERY_CHARS:
        query = query[:_LIFECHOICE_MAX_QUERY_CHARS]

    chunks, embs = _load_rag_index(trial.novel)
    mask = np.array([c["chapter"] <= trial.query_chapter for c in chunks])
    [qv_raw] = await embed_batch(embed_client, sem, config.EXP_EMBED_MODEL, [query])
    qv = np.array(qv_raw, dtype=np.float32)
    idxs = _cosine_topk(qv, embs, config.RAG_TOP_K, mask)

    parts: list[str] = [f"[Character Description]\n{description}"]
    if idxs:
        memory_pieces = [
            f"[Chapter {chunks[i]['chapter']} excerpt]\n{chunks[i]['text'].strip()}"
            for i in idxs
        ]
        parts.append("[Relevant Memory]\n" + "\n\n".join(memory_pieces))
    return "\n\n".join(parts)


# ── timechara ──────────────────────────────────────────────────────
# Two-stage hint pipeline ported from ahnjaewoo/timechara's narrative-experts:
# (1) temporal expert maps the scenario to a chapter and compares to the
# character's query_chapter; future → "you don't know this yet" hint, and we
# short-circuit (skip stage 2). (2) spatial expert decides presence; absent →
# "don't claim you were there" hint. Both classifier calls use the same chat
# model as the final response (per user preference), at temperature 0.

_CHAPTER_RE = re.compile(r"chapter\s*[:\-]?\s*(\d+|unknown)", re.IGNORECASE)
_PRESENCE_RE = re.compile(r"presence\s*[:\-]?\s*(present|absent)", re.IGNORECASE)


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _parse_chapter(text: str) -> Optional[int]:
    """Return the predicted chapter as int, or None for 'unknown' / unparseable."""
    last = _last_nonempty_line(text)
    m = _CHAPTER_RE.search(last)
    if not m:
        m = _CHAPTER_RE.search(text)
    if not m:
        return None
    tok = m.group(1).lower()
    if tok == "unknown":
        return None
    try:
        return int(tok)
    except ValueError:
        return None


def _parse_presence(text: str) -> Optional[str]:
    """Return 'present' / 'absent' / None."""
    last = _last_nonempty_line(text)
    m = _PRESENCE_RE.search(last)
    if not m:
        m = _PRESENCE_RE.search(text)
    if m:
        return m.group(1).lower()
    low = last.lower()
    has_present = "present" in low
    has_absent = "absent" in low
    if has_present and not has_absent:
        return "present"
    if has_absent and not has_present:
        return "absent"
    return None


@lru_cache(maxsize=64)
def _total_chapters(novel: str) -> int:
    chapters = list_chapters(novel)
    if not chapters:
        raise FileNotFoundError(f"no chapter files under {config.chapter_dir(novel)}")
    return max(chapters)


async def build_timechara_async(chat_client, sem, model: str, trial: Trial) -> Optional[str]:
    """Run the two expert classifiers and return a HINT string (or None).

    Self-referential: the same `model` used for the role-play call is used
    for classification. Classifier temperature is fixed at 0.
    """
    from ._api import chat_text
    from .prompts import (
        TIMECHARA_EXPERT_SYSTEM,
        render_timechara_spatial_user,
        render_timechara_temporal_user,
        timechara_absent_hint,
        timechara_future_hint,
    )

    total_ch = _total_chapters(trial.novel)
    hints: list[str] = []

    # 1) temporal expert
    temporal_user = render_timechara_temporal_user(trial, total_ch)
    text, _, _ = await chat_text(
        chat_client, sem, model, TIMECHARA_EXPERT_SYSTEM, temporal_user,
        max_tokens=config.TIMECHARA_MAX_TOKENS,
        temperature=0.0,
    )
    predicted_chapter = _parse_chapter(text)
    is_future = predicted_chapter is not None and predicted_chapter > trial.query_chapter
    if is_future:
        hints.append(timechara_future_hint(trial.character))

    # 2) spatial expert (skipped when future, mirroring timechara)
    if not is_future:
        spatial_user = render_timechara_spatial_user(trial)
        text, _, _ = await chat_text(
            chat_client, sem, model, TIMECHARA_EXPERT_SYSTEM, spatial_user,
            max_tokens=config.TIMECHARA_MAX_TOKENS,
            temperature=0.0,
        )
        presence = _parse_presence(text)
        if presence == "absent":
            hints.append(timechara_absent_hint(trial.character))

    return " ".join(hints) if hints else None


# ── dispatch ───────────────────────────────────────────────────────
SYNC_BUILDERS = {
    "vanilla":   build_vanilla,
    "arc":       build_arc,
    "summary":   build_summary,
}
RAG_REQUIRES_EMBED = "rag"
LIFECHOICE_REQUIRES_EMBED = "lifechoice"
TIMECHARA_REQUIRES_CHAT = "timechara"


def build_context_sync(mode: str, trial: Trial) -> Optional[str]:
    if mode == RAG_REQUIRES_EMBED:
        raise RuntimeError("rag mode must be built via build_rag_async")
    if mode == LIFECHOICE_REQUIRES_EMBED:
        raise RuntimeError("lifechoice mode must be built via build_lifechoice_async")
    if mode == TIMECHARA_REQUIRES_CHAT:
        raise RuntimeError("timechara mode must be built via build_timechara_async")
    try:
        builder = SYNC_BUILDERS[mode]
    except KeyError:
        raise ValueError(
            f"unknown mode: {mode!r}; supported: "
            f"{list(SYNC_BUILDERS) + [RAG_REQUIRES_EMBED, LIFECHOICE_REQUIRES_EMBED, TIMECHARA_REQUIRES_CHAT]}"
        )
    return builder(trial)


def precheck_mode(mode: str, novel: str, character_slug: str) -> None:
    """Raise early with a helpful message if the mode's preconditions are unmet."""
    if mode == "arc":
        if not config.axes_file(novel, character_slug).exists():
            raise FileNotFoundError(
                f"{mode} mode requires {config.axes_file(novel, character_slug)}"
            )
    elif mode == "summary":
        chapters = list_chapters(novel)
        if not chapters:
            raise FileNotFoundError(f"no chapter files under {config.chapter_dir(novel)}")
        # warn-only: missing per-chapter summaries are surfaced inline in the prompt
    elif mode == "rag":
        rdir = config.rag_dir(novel)
        if not (rdir / "chunks.jsonl").exists() or not (rdir / "embeddings.npy").exists():
            raise FileNotFoundError(
                f"rag index missing under {rdir}; run `python -m inference.build_index --novel {novel!r}`"
            )
    elif mode == "timechara":
        chapters = list_chapters(novel)
        if not chapters:
            raise FileNotFoundError(
                f"timechara mode requires chapter files under {config.chapter_dir(novel)}"
            )
    elif mode == "lifechoice":
        chapters = list_chapters(novel)
        if not chapters:
            raise FileNotFoundError(
                f"lifechoice mode requires chapter files under {config.chapter_dir(novel)}"
            )
        rdir = config.rag_dir(novel)
        if not (rdir / "chunks.jsonl").exists() or not (rdir / "embeddings.npy").exists():
            raise FileNotFoundError(
                f"lifechoice mode requires rag index under {rdir}; "
                f"run `python -m inference.build_index --novel {novel!r}`"
            )
        # per-chapter summaries are warn-only (build_summary surfaces missing inline)
