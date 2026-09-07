"""CLI entry point for response evaluation.

    python -m evaluation.per_response --novel Anna_Kareina --character anna_karenina

Scores one character's role-play responses with an LLM judge. Each
(trial, model, mode) response is judged in its own call — independently, with
no mode label shown to the judge. After all judge calls succeed, writes two
files into `results/inference/{novel}/{character}/`:

  eval_records_{judge_tag}.jsonl  — one verdict per (trial, model, mode)
  eval_summary_{judge_tag}.json   — per-(model, mode) average scores

Before any judge call the projected cost is printed; `--estimate-only` stops
there. After a run the per-(model, mode) table is printed to the terminal.
"""

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from evaluation.scoring import EvaluationRunError
from . import config as eval_config
from .loader import build_eval_units
from .runner import (estimate_cost, print_estimate, print_summary,
                     records_path, run)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evaluation.per_response")
    p.add_argument("--novel", default=eval_config.EVAL_NOVEL,
                   help=f"novel (default: {eval_config.EVAL_NOVEL}, via EXP_EVAL_NOVEL)")
    p.add_argument("--character", default=eval_config.EVAL_CHARACTER,
                   help=f"character slug (default: {eval_config.EVAL_CHARACTER}, "
                        f"via EXP_EVAL_CHARACTER)")
    p.add_argument("--variant", default="main",
                   help="probe file variant (default: main)")
    p.add_argument("--judge-model", default=eval_config.JUDGE_MODEL,
                   help=f"judge model (default: {eval_config.JUDGE_MODEL}, "
                        f"override via EXP_JUDGE_MODEL; backend currently "
                        f"'{eval_config.JUDGE_BACKEND}' via EXP_JUDGE_BACKEND)")
    p.add_argument("--include-thought", action=argparse.BooleanOptionalAction,
                   default=eval_config.INCLUDE_THOUGHT,
                   help=f"include ref_thought in the reference shown to the "
                        f"judge (default: {eval_config.INCLUDE_THOUGHT}, via "
                        f"EXP_EVAL_INCLUDE_THOUGHT). Use --no-include-thought "
                        f"to run the no-thought ablation; outputs land in "
                        f"`*_nothought.*` sibling files.")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of eval units (smoke test)")
    p.add_argument("--manifest", type=Path, default=None,
                   help="pairwise manifest .json; restrict scoring to its "
                        "(probe_id, phase_idx, arc_mode|other_mode) triples")
    p.add_argument("--model", default=None,
                   help="restrict to this exact model name (e.g. 'Qwen/Qwen3-32B')")
    p.add_argument("--estimate-only", action="store_true",
                   help="print the cost estimate and exit without calling the judge")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    try:
        units = build_eval_units(args.novel, args.character, args.variant)
    except FileNotFoundError as e:
        raise SystemExit(f"could not load probe file: {e}")
    if args.model:
        units = [u for u in units if u.model == args.model]

    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        allowed: set[tuple[str, int, str]] = set()
        for m in manifest:
            pid, ph = m["probe_id"], int(m["phase_idx"])
            allowed.add((pid, ph, m["arc_mode"]))
            allowed.add((pid, ph, m["other_mode"]))
        filtered: list = []
        for u in units:
            kept = {mode: text for mode, text in u.responses.items()
                    if (u.probe_id, u.phase_idx, mode) in allowed}
            if kept:
                filtered.append(u if len(kept) == len(u.responses)
                                else replace(u, responses=kept))
        units = filtered

    if args.limit is not None:
        units = units[:args.limit]
    if not units:
        raise SystemExit(
            "no eval units — check the probe file and the "
            "{mode}__{model}.jsonl result files under "
            f"results/inference/{args.novel}/{args.character}/"
        )

    # Re-score all requested units; replace outputs only after full success.
    todo = list(units)
    rec_path_ = records_path(args.novel, args.character,
                             args.judge_model, args.include_thought)
    if rec_path_.exists():
        print(f"note: {rec_path_.name} already exists and will be replaced "
              "after a successful run.")

    print_estimate(estimate_cost(todo, args.judge_model,
                                 eval_config.SCORE_SCALE,
                                 args.include_thought))
    if args.estimate_only:
        return

    if todo and eval_config.JUDGE_API_KEY == "EMPTY":
        backend_key = ("OPENROUTER_API_KEY"
                       if eval_config.JUDGE_BACKEND == "openrouter"
                       else "OPENAI_API_KEY")
        logger.warning("no judge API key (EXP_JUDGE_API_KEY / %s); "
                       "set credentials required by the judge endpoint.",
                       backend_key)

    try:
        summary = run(args.novel, args.character, todo,
                      judge_model=args.judge_model,
                      include_thought=args.include_thought)
    except EvaluationRunError as exc:
        raise SystemExit(str(exc)) from exc
    print_summary(summary)


if __name__ == "__main__":
    main()
