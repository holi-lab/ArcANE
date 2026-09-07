"""Runner: fan out (trial × mode × model) over an AsyncOpenAI client and
append results to a JSONL file per (novel, character, mode, model).

Resume semantics: each JSONL row carries `trial_id`. On restart we scan the
file once, build the set of completed ids, and skip them. Incomplete rows are
ignored and retried. New records start on a separate line, preserving any
incomplete tail for inspection.
"""

import asyncio
import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

from openai import AsyncOpenAI

from . import config, context_builders
from ._api import chat_text, strip_think_block
from .loaders import Trial, trial_to_dict
from .prompts import render_system, render_user
from .records import is_dry_run_record

logger = logging.getLogger(__name__)


def _completed_trial_ids(path: Path) -> set[str]:
    """Read append-only JSONL and return ids successfully written.

    A non-preview row that JSON-parses and has `error is None` counts as complete.
    Errored rows are kept on disk for debugging but are NOT skipped — the
    runner will re-attempt them. This is intentional: transient API errors
    are common and worth retrying on the next invocation.
    """
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_bytes().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(row, dict) or row.get("error") or is_dry_run_record(row):
            continue
        tid = row.get("trial_id")
        if tid:
            done.add(tid)
    return done


def _append_record(path: Path, row: dict) -> None:
    """Append a complete JSONL row without joining an unterminated tail."""
    payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("ab+") as fh:
        if fh.seek(0, 2):
            fh.seek(-1, 2)
            if fh.read(1) != b"\n":
                fh.write(b"\n")
        fh.write(payload)


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _build_full_trial_id(trial: Trial, mode: str, model: str) -> str:
    safe_model = model.replace("/", "_").replace(":", "_")
    return f"{trial.trial_id}__{mode}__{safe_model}"


async def _run_one(chat_client: AsyncOpenAI, embed_client: AsyncOpenAI,
                   sem: asyncio.Semaphore,
                   trial: Trial, mode: str, model: str,
                   record_prompts: bool) -> dict:
    """Build context → render prompts → call model → return JSONL row."""
    full_id = _build_full_trial_id(trial, mode, model)

    # context (system-side) and hint (user-side, timechara-only)
    context: Optional[str] = None
    hint: Optional[str] = None
    try:
        if mode == context_builders.RAG_REQUIRES_EMBED:
            context = await context_builders.build_rag_async(embed_client, sem, trial)
        elif mode == context_builders.LIFECHOICE_REQUIRES_EMBED:
            context = await context_builders.build_lifechoice_async(embed_client, sem, trial)
        elif mode == context_builders.TIMECHARA_REQUIRES_CHAT:
            hint = await context_builders.build_timechara_async(
                chat_client, sem, model, trial,
            )
        else:
            context = context_builders.build_context_sync(mode, trial)
    except Exception as e:
        return _row(trial, mode, model, full_id, None, None, None,
                    None, None, None, None, None, error=f"context_build: {e}",
                    record_prompts=record_prompts)

    system = render_system(trial, context)
    user   = render_user(trial, hint=hint)
    ctx_len = len(context or "") + len(hint or "")
    p_hash = _hash(system + "\n---\n" + user)

    try:
        text, usage, latency_ms = await chat_text(
            chat_client, sem, model, system, user,
            max_tokens=config.ROLEPLAY_MAX_TOKENS)
        text = strip_think_block(text)
        return _row(trial, mode, model, full_id, system, user, p_hash, ctx_len,
                    text, usage, latency_ms, hint, error=None,
                    record_prompts=record_prompts)
    except Exception as e:
        return _row(trial, mode, model, full_id, system, user, p_hash, ctx_len,
                    None, None, None, hint, error=f"chat_call: {e}",
                    record_prompts=record_prompts)


def _row(trial: Trial, mode: str, model: str, full_id: str,
         system: Optional[str], user: Optional[str], prompt_hash: Optional[str],
         context_char_len: Optional[int], response: Optional[str],
         usage: Optional[dict], latency_ms: Optional[int],
         hint: Optional[str], *,
         error: Optional[str], record_prompts: bool) -> dict:
    row: dict = {
        "trial_id": full_id,
        **trial_to_dict(trial),
        "mode": mode,
        "model": model,
        "context_char_len": context_char_len,
        "prompt_hash": prompt_hash,
        "response": response,
        "usage": usage,
        "latency_ms": latency_ms,
        "error": error,
        "ts": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    if mode == "timechara":
        row["timechara_hint"] = hint
    if record_prompts:
        row["prompt_system"] = system
        row["prompt_user"]   = user
    return row


_DEFERRED_CONTEXT = {
    context_builders.RAG_REQUIRES_EMBED: "query embedding and RAG retrieval",
    context_builders.LIFECHOICE_REQUIRES_EMBED: "description-augmented embedding and memory retrieval",
    context_builders.TIMECHARA_REQUIRES_CHAT: "temporal/spatial expert calls and hint generation",
}


def _preview_one(trial: Trial, mode: str, model: str, *,
                 record_prompts: bool, precheck_error: Optional[str]) -> dict:
    """Render only local context; mark API-dependent prompts as incomplete."""
    context = None
    error = None
    deferred = _DEFERRED_CONTEXT.get(mode)
    if deferred is None:
        try:
            context = context_builders.build_context_sync(mode, trial)
        except Exception as e:
            error = f"context_build: {e}"
    system = render_system(trial, context)
    user = render_user(trial)
    complete = not (deferred or error or precheck_error)
    row = _row(
        trial, mode, model, _build_full_trial_id(trial, mode, model),
        system, user, _hash(system + "\n---\n" + user) if complete else None,
        len(context or "") if complete else None,
        None, None, None, None, error=error, record_prompts=record_prompts,
    )
    row.update(dry_run=True, prompt_complete=complete,
               deferred_context=deferred, precheck_error=precheck_error)
    return row


def _preview(novel: str, character_slug: str, trials: list[Trial],
             modes: list[str], models: list[str], *, record_prompts: bool) -> None:
    """Print JSONL previews without creating clients, caches or result files."""
    for mode in modes:
        precheck_error = None
        try:
            context_builders.precheck_mode(mode, novel, character_slug)
        except Exception as e:
            precheck_error = str(e)
        for model in models:
            path = config.results_path(novel, character_slug, mode, model)
            completed = _completed_trial_ids(path)
            for trial in trials:
                row = _preview_one(trial, mode, model, record_prompts=record_prompts,
                                   precheck_error=precheck_error)
                row["already_completed"] = row["trial_id"] in completed
                print(json.dumps(row, ensure_ascii=False))
    logger.info("dry-run: previews printed to stdout; no API calls or result files written. "
                "API-dependent context is deferred; inspect precheck_error for missing assets.")


async def run(novel: str, character_slug: str, trials: Iterable[Trial],
              modes: list[str], models: list[str], *,
              variant: str = "main",                 # noqa: ARG001 (kept for future fan-out by variant)
              record_prompts: bool = True,
              dry_run: bool = False) -> None:
    """Execute (trial × mode × model) and append rows to one JSONL per
    (mode, model). Existing trial_ids in those files are skipped.
    With dry_run, print local previews to stdout without API calls or writes.
    """
    trials = list(trials)
    if not trials:
        logger.warning("no trials to run")
        return

    if dry_run:
        _preview(novel, character_slug, trials, modes, models,
                 record_prompts=record_prompts)
        return

    # precheck modes once (fail fast on missing arc file / rag index)
    for mode in modes:
        context_builders.precheck_mode(mode, novel, character_slug)

    chat_client  = config.make_chat_client()
    embed_client = config.make_embed_client()
    sem = asyncio.Semaphore(config.MAX_CONCURRENT_API_CALLS)

    # per-(mode,model) output handles + completed-id sets
    out_files: dict[tuple[str, str], Path] = {}
    completed: dict[tuple[str, str], set[str]] = {}
    for mode in modes:
        for model in models:
            p = config.results_path(novel, character_slug, mode, model)
            p.parent.mkdir(parents=True, exist_ok=True)
            out_files[(mode, model)] = p
            completed[(mode, model)] = _completed_trial_ids(p)

    # build task list
    tasks: list[tuple[tuple[str, str], asyncio.Task]] = []
    skipped = 0
    for mode in modes:
        for model in models:
            for trial in trials:
                full_id = _build_full_trial_id(trial, mode, model)
                if full_id in completed[(mode, model)]:
                    skipped += 1
                    continue
                tasks.append((
                    (mode, model),
                    asyncio.create_task(_run_one(
                        chat_client, embed_client, sem, trial, mode, model,
                        record_prompts=record_prompts,
                    )),
                ))

    total = len(tasks)
    logger.info(f"runner: trials={len(trials)} modes={modes} models={models} "
                f"planned={total} skipped(already done)={skipped}")
    if not tasks:
        return

    # stream completions back to disk in arrival order
    pending = {t for _, t in tasks}
    task_key = {t: key for key, t in tasks}
    done_n = err_n = 0
    while pending:
        finished, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in finished:
            key = task_key[t]
            row = t.result()
            out_path = out_files[key]
            _append_record(out_path, row)
            done_n += 1
            if row.get("error"):
                err_n += 1
            if done_n % 25 == 0 or done_n == total:
                logger.info(f"runner: {done_n}/{total} done (errors={err_n})")

    logger.info(f"runner: finished. wrote {done_n} rows, {err_n} errored")
