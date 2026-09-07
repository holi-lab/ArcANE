"""Run the LLM judge over evaluation units, then aggregate.

Each (trial, model, mode) is judged in its own call: the judge receives one
response with no id/label and no sibling candidate, and returns one verdict.
So every score is an independent absolute verdict — not a relative ranking
within a batch, and not biased by the mode name. Judge replies request JSON
output and are locally validated before their scores are accepted.

Verdicts are staged until all judge calls succeed, then published as
`eval_records_{judge_tag}.jsonl` and aggregated into
`eval_summary_{judge_tag}.json`. Failed attempts remain in separate .partial
files without replacing prior outputs.

Reporting utilities:
  • `estimate_cost` / `print_estimate` — show projected token use and cost
    before any judge call is made.
  • `print_summary` — print the final per-(model, mode) table to the terminal.

Judge settings — model, backend, pricing — live in `evaluation/per_response/config.py`
(env-overridable), kept separate from the role-play config so the judge can be
tuned independently.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from openai import OpenAI

from evaluation.scoring import EvaluationRunError, scores_with_average, validate_scale
from inference import config
from inference.config import MAX_RETRIES, RETRY_BACKOFF_BASE
from . import config as eval_config
from .loader import EvalUnit
from .prompt import (DIMENSIONS, evaluator_response_format,
                     render_evaluator_messages)

logger = logging.getLogger(__name__)

# Judge models that 400 on a custom `temperature` (gpt-5.x family); the
# rejection is detected once and cached so later calls drop the parameter.
_MODELS_REJECT_TEMPERATURE: set[str] = set()


# ── output paths ───────────────────────────────────────────────────
# Output filenames use short judge-family tags:
#   deepseek/deepseek-v4-flash   -> "deepseek"
#   claude-opus-4-5              -> "claude"
#   gpt-5.5                      -> "gpt"
# Models in the same family share a tag.
def judge_tag(judge_model: str) -> str:
    """Short, filename-safe judge-family tag."""
    m = (judge_model or "").lower()
    for family in ("gpt", "deepseek", "claude", "qwen", "llama"):
        if family in m:
            return family
    tail = m.rsplit("/", 1)[-1] or "judge"
    safe = "".join(c if (c.isalnum() or c == "-") else "-" for c in tail)
    return safe[:32] or "judge"


def _thought_suffix(include_thought: bool) -> str:
    """File-name suffix for the include_thought ablation.

    With-thought is the default and keeps the unsuffixed names; the no-thought
    ablation lands in `*_nothought.*` siblings so the two never share a JSONL.
    """
    return "" if include_thought else "_nothought"


def records_path(novel: str, character_slug: str, judge_model: str,
                 include_thought: bool = True) -> Path:
    name = (f"eval_records_{judge_tag(judge_model)}"
            f"{_thought_suffix(include_thought)}.jsonl")
    return config.RESULTS_ROOT / novel / character_slug / name


def summary_path(novel: str, character_slug: str, judge_model: str,
                 include_thought: bool = True) -> Path:
    name = (f"eval_summary_{judge_tag(judge_model)}"
            f"{_thought_suffix(include_thought)}.json")
    return config.RESULTS_ROOT / novel / character_slug / name


# ── judge-output parsing ───────────────────────────────────────────
def _parse_verdict(text: str) -> Optional[dict]:
    """Parse the judge's structured reply into the verdict dict.

    Strip an optional code fence and parse JSON; scores are validated separately.
    """
    s = (text or "").strip()
    if s.startswith("```"):
        s = "\n".join(ln for ln in s.splitlines()
                      if not ln.strip().startswith("```")).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _scores_of(obj: Optional[dict], scale: Optional[int] = None) -> Optional[dict]:
    """Return valid dimension scores and their computed mean, or None."""
    return scores_with_average(
        obj, DIMENSIONS, eval_config.SCORE_SCALE if scale is None else scale)


def _record(unit: EvalUnit, mode: str, obj: Optional[dict],
            error: Optional[str], scale: int) -> dict:
    """One judge verdict for one response."""
    scores = _scores_of(obj, scale) if error is None else None
    if scores is None and error is None:
        error = f"invalid judge scores: expected all dimensions in 1..{scale}"
    return {
        "novel": unit.novel,
        "character": unit.character,
        "probe_id": unit.probe_id,
        "phase_idx": unit.phase_idx,
        "model": unit.model,
        "mode": mode,
        "scale": scale,
        "scores": {d: scores[d] for d in DIMENSIONS} if scores else None,
        "average": scores["average"] if scores else None,
        "parse_ok": scores is not None,
        "error": error,
    }


# ── one judge call ─────────────────────────────────────────────────
def _judge_chat(client: OpenAI, model: str, system: str, user: str,
                response_format: dict,
                temperature: float = 0.0, *,
                scale: Optional[int] = None) -> tuple[str, dict]:
    """One synchronous judge completion → (text, usage).

    Validate the reply locally and retry invalid scores with
    backoff; if the model rejects a custom `temperature` (gpt-5.x family) the
    parameter is dropped and remembered for later calls.
    """
    cur_temp: Optional[float] = (
        None if model in _MODELS_REJECT_TEMPERATURE else temperature)
    for attempt in range(MAX_RETRIES):
        try:
            kwargs: dict = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "response_format": response_format,
            }
            if cur_temp is not None:
                kwargs["temperature"] = cur_temp
            # Apply the configured OpenRouter routing and reasoning settings.
            if eval_config.JUDGE_BACKEND == "openrouter":
                extra_body: dict = {}
                if eval_config.JUDGE_PROVIDER_ORDER:
                    extra_body["provider"] = {
                        "order": eval_config.JUDGE_PROVIDER_ORDER,
                        "allow_fallbacks":
                            eval_config.JUDGE_PROVIDER_ALLOW_FALLBACK,
                    }
                if eval_config.JUDGE_DISABLE_REASONING:
                    extra_body["reasoning"] = {"enabled": False}
                if extra_body:
                    kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            if getattr(msg, "refusal", None):
                raise RuntimeError(f"model refused: {msg.refusal}")
            usage = resp.usage.model_dump() if resp.usage else {}
            # Validate scores even when a structured response was requested.
            content = msg.content or ""
            probe = _parse_verdict(content)
            if _scores_of(probe, scale) is None:
                raise RuntimeError(
                    "judge returned missing, malformed, or out-of-range scores"
                )
            return content, usage
        except Exception as e:                       # noqa: BLE001
            if "temperature" in str(e) and cur_temp is not None:
                _MODELS_REJECT_TEMPERATURE.add(model)
                logger.info(f"[judge] {model} rejects custom temperature; dropping")
                cur_temp = None
                continue
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE ** (attempt + 1))
            else:
                raise
    raise RuntimeError("unreachable")


def _score_one(client: OpenAI, judge_model: str, scale: int,
               unit: EvalUnit, mode: str, text: str,
               include_thought: bool = True) -> tuple[dict, dict]:
    """Judge one (trial, model, mode), returning (record, usage).

    A failed judge call yields an error record.
    """
    messages = render_evaluator_messages(unit.trial, text, scale,
                                          include_thought)
    try:
        out, usage = _judge_chat(
            client, judge_model,
            messages[0]["content"], messages[1]["content"],
            evaluator_response_format(),
            temperature=0.0,
            scale=scale,
        )
    except Exception as e:                           # noqa: BLE001
        logger.warning(
            f"judge failed [{unit.probe_id}/p{unit.phase_idx}/"
            f"{unit.model}/{mode}]: {e}"
        )
        return _record(unit, mode, None, str(e), scale), {}

    obj = _parse_verdict(out)
    return _record(unit, mode, obj, None, scale), (usage or {})


# ── aggregation ────────────────────────────────────────────────────
def aggregate(rec_path: Path, novel: str, character: str,
              judge_model: str, scale: int) -> dict:
    """Read judge records into per-(model, mode) average scores, over
    records scored at `scale`. On a re-scored (trial, model, mode) the last
    record wins."""
    latest: dict[tuple, dict] = {}
    if rec_path.exists():
        for line in rec_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict) or r.get("scale") != scale:
                continue
            key = (r.get("probe_id"), r.get("phase_idx"),
                   r.get("model"), r.get("mode"))
            latest[key] = r

    # {model: {mode: [records]}}
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    invalid = 0
    for r in latest.values():
        if r.get("parse_ok") and not r.get("error"):
            scores = _scores_of(r, scale)
            if scores is None:
                invalid += 1
                continue
            grouped[r["model"]][r["mode"]].append(
                {**r, "average": scores["average"]})
    if invalid:
        logger.warning("skipped %d records with invalid scores in %s", invalid, rec_path)

    by_model: dict[str, dict[str, dict]] = {}
    for model, modes in grouped.items():
        by_model[model] = {}
        for mode, recs in modes.items():
            agg: dict = {}
            for dim in DIMENSIONS:
                agg[dim] = round(
                    sum(r["scores"][dim] for r in recs) / len(recs), 3)
            agg["average"] = round(
                sum(r["average"] for r in recs) / len(recs), 3)
            agg["n"] = len(recs)
            by_model[model][mode] = agg

    return {
        "novel": novel,
        "character": character,
        "judge_model": judge_model,
        "score_scale": scale,
        "by_model": by_model,
    }


# ── run ────────────────────────────────────────────────────────────
def run(novel: str, character_slug: str, todo: list[EvalUnit], *,
        judge_model: str, include_thought: bool = True) -> dict:
    """Score every (unit, mode), publishing fresh records only on full success.
    Failed attempts remain in a separate .partial file; prior outputs are kept.

    One judge call per (trial, model, mode); calls run concurrently on a
    `ThreadPoolExecutor` sized by `EVAL_CONCURRENCY` (200 by default). `todo`
    contains all requested units; an empty `todo` just rebuilds the summary.
    `include_thought` picks the prompt variant *and* the output filename — the
    two ablations land in separate `*[_nothought].*` files.
    """
    scale = eval_config.SCORE_SCALE
    validate_scale(scale)
    rec_path = records_path(novel, character_slug, judge_model,
                             include_thought)
    rec_path.parent.mkdir(parents=True, exist_ok=True)

    # one job = one (trial, model, mode) judge call
    jobs = [(u, mode, text)
            for u in todo
            for mode, text in u.responses.items()]

    in_tokens = out_tokens = ok = bad = 0
    if jobs:
        client = eval_config.make_judge_client()
        workers = max(1, min(eval_config.EVAL_CONCURRENCY, len(jobs)))
        # Keep failed attempts separate from the last successful results.
        with ThreadPoolExecutor(max_workers=workers) as pool, \
                tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=rec_path.parent,
                    prefix=rec_path.name + ".", suffix=".partial",
                    delete=False) as fh:
            tmp_path = Path(fh.name)
            futures = [pool.submit(_score_one, client, judge_model, scale,
                                   u, m, t, include_thought)
                       for (u, m, t) in jobs]
            for fut in as_completed(futures):
                rec, usage = fut.result()
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                in_tokens += usage.get("prompt_tokens", 0) or 0
                out_tokens += usage.get("completion_tokens", 0) or 0
                if rec["parse_ok"]:
                    ok += 1
                else:
                    bad += 1
                if (ok + bad) % 100 == 0:
                    logger.info(f"eval: {ok + bad}/{len(jobs)} responses scored")
        if bad:
            raise EvaluationRunError(
                f"eval: {bad}/{ok + bad} judge calls failed; previous outputs "
                f"unchanged. Attempt records: {tmp_path}")
        tmp_path.replace(rec_path)
        logger.info(f"eval: finished — {ok} ok, {bad} with call/parse issues")

    summary = aggregate(rec_path, novel, character_slug, judge_model, scale)
    summary["include_thought"] = include_thought
    summary["this_run"] = {
        "responses_scored": ok + bad,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "cost_usd": round(_cost(in_tokens, out_tokens), 4),
    }
    sp = summary_path(novel, character_slug, judge_model, include_thought)
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    logger.info(f"eval: wrote {sp}")
    return summary



# ── cost estimate ──────────────────────────────────────────────────
def _cost(in_tokens: int, out_tokens: int) -> float:
    return (in_tokens / 1e6) * eval_config.JUDGE_PRICE_INPUT \
        + (out_tokens / 1e6) * eval_config.JUDGE_PRICE_OUTPUT


def estimate_cost(units: list[EvalUnit], judge_model: str,
                  scale: int, include_thought: bool = True) -> dict:
    """Project token use / cost over `units` before any judge call is made.

    One call per (trial, model, mode). Token counts are heuristic (≈ chars / 4
    for input; a fixed budget per call for output), so treat the figure as a
    ballpark.
    """
    in_tokens = out_tokens = n_calls = 0
    for unit in units:
        for _mode, text in unit.responses.items():
            messages = render_evaluator_messages(unit.trial, text, scale,
                                                  include_thought)
            chars = sum(len(m["content"]) for m in messages)
            in_tokens += chars // eval_config.EST_CHARS_PER_TOKEN
            out_tokens += eval_config.EST_OUTPUT_TOKENS_PER_MODE
            n_calls += 1
    return {
        "judge_model": judge_model,
        "include_thought": include_thought,
        "n_calls": n_calls,
        "est_input_tokens": in_tokens,
        "est_output_tokens": out_tokens,
        "est_cost_usd": round(_cost(in_tokens, out_tokens), 4),
    }


def print_estimate(est: dict) -> None:
    """Print the pre-run cost projection to the terminal."""
    print("─" * 72)
    ablation = "with-thought" if est.get("include_thought", True) else "NO-THOUGHT"
    print(f"COST ESTIMATE — judge: {est['judge_model']} "
          f"(backend: {eval_config.JUDGE_BACKEND}, {ablation})")
    if (eval_config.JUDGE_BACKEND == "openrouter"
            and eval_config.JUDGE_PROVIDER_ORDER):
        order = ",".join(eval_config.JUDGE_PROVIDER_ORDER)
        fb = "fallback ON" if eval_config.JUDGE_PROVIDER_ALLOW_FALLBACK \
            else "fallback OFF"
        print(f"  pinned provider   : {order}  ({fb})")
    print(f"  judge calls       : {est['n_calls']:,}")
    print(f"  input tokens  (~) : {est['est_input_tokens']:,}")
    print(f"  output tokens (~) : {est['est_output_tokens']:,}")
    print(f"  estimated cost    : ~${est['est_cost_usd']}")
    print(f"  (rate: ${eval_config.JUDGE_PRICE_INPUT}/1M in, "
          f"${eval_config.JUDGE_PRICE_OUTPUT}/1M out — set in "
          f"evaluation/per_response/config.py or via EXP_JUDGE_PRICE_* env)")
    print("─" * 72)


# ── final terminal summary ─────────────────────────────────────────
def print_summary(summary: dict) -> None:
    """Print the per-(model, mode) score table to the terminal."""
    print()
    print("=" * 72)
    print(f"EVAL SUMMARY — {summary['novel']} / {summary['character']}"
          f"   (judge: {summary['judge_model']}, "
          f"scale: 1-{summary.get('score_scale', '?')})")
    print("=" * 72)

    by_model = summary.get("by_model") or {}
    if not by_model:
        print("(no scored records found)")
        return

    # 7-char abbreviations keep the table aligned regardless of dim count.
    headers = [d[:6] for d in DIMENSIONS]
    for model in sorted(by_model):
        modes = by_model[model]
        print(f"\nmodel: {model}")
        cols = " ".join(f"{h:>7}" for h in headers)
        print(f"  {'mode':12} {cols} {'AVG':>7} {'n':>6}")
        for mode in sorted(modes):
            a = modes[mode]
            vals = " ".join(f"{a[d]:7.2f}" for d in DIMENSIONS)
            print(f"  {mode:12} {vals} {a['average']:7.2f} {a['n']:6d}")
        best = max(modes, key=lambda m: modes[m]["average"])
        print(f"  -> best by AVG: {best} ({modes[best]['average']})")

    run = summary.get("this_run") or {}
    if run.get("responses_scored"):
        print(f"\nthis run: {run['responses_scored']} responses scored, "
              f"{run['input_tokens']:,} in / {run['output_tokens']:,} out tokens, "
              f"actual ~${run['cost_usd']}")
    print("=" * 72)
