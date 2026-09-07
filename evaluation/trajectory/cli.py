"""CLI entry point for trajectory evaluation.

    python -m evaluation.trajectory --novel Anna_Kareina --character anna_karenina

Scores one character's *trajectories* with an LLM judge. For every
(probe, model, mode) the judge sees the model's responses across every valid
phase of that probe at once, and scores how faithfully the response sequence
mirrors the reference sequence (per-phase anchoring, direction, shape).

After all judge calls succeed, writes into the character's results folder
(`results/inference/{novel}/{character}/`):

  traj_records_{judge_tag}.jsonl  — one line per (probe, model, mode)
  traj_summary_{judge_tag}.json   — per-(model, mode) average + N-bucket
                                     breakdown

Before any judge call the projected cost is printed; `--estimate-only` stops
there. After a run the per-(model, mode) trajectory table is printed.
"""

import argparse
import logging

from evaluation.scoring import EvaluationRunError
from . import config as eval_config
from .loader import build_traj_units
from .runner import (estimate_cost, print_estimate, print_summary,
                     records_path, run)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evaluation.trajectory")
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
                   help=f"include ref_thought in each phase block shown to "
                        f"the judge (default: {eval_config.INCLUDE_THOUGHT}, "
                        f"via EXP_EVAL_INCLUDE_THOUGHT). Use "
                        f"--no-include-thought to run the no-thought "
                        f"ablation; outputs land in `*_nothought.*` sibling "
                        f"files.")
    p.add_argument("--min-phases", type=int,
                   default=eval_config.TRAJ_MIN_PHASES,
                   help=f"skip probes whose mode has fewer than this many valid "
                        f"phases (default: {eval_config.TRAJ_MIN_PHASES}, via "
                        f"EXP_TRAJ_MIN_PHASES)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of trajectory calls (smoke test)")
    p.add_argument("--model", default=None,
                   help="restrict to this exact model name (e.g. 'Qwen/Qwen3-32B')")
    p.add_argument("--mode", default=None,
                   help="restrict to this exact mode name (e.g. 'arc', 'vanilla')")
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
        units = build_traj_units(args.novel, args.character, args.variant,
                                  min_phases=args.min_phases)
    except FileNotFoundError as e:
        raise SystemExit(f"could not load probe file: {e}")
    if args.model:
        units = [u for u in units if u.model == args.model]
    if args.mode:
        for u in units:
            u.phases_by_mode = {m: refs for m, refs in u.phases_by_mode.items()
                                if m == args.mode}
        units = [u for u in units if u.phases_by_mode]

    if not units:
        raise SystemExit(
            "no trajectory units — check the probe file, --min-phases, and "
            "the {mode}__{model}.jsonl result files under "
            f"results/inference/{args.novel}/{args.character}/"
        )

    # one job = one (unit, mode) — i.e. one judge call
    all_jobs: list[tuple] = [(u, m)
                             for u in units
                             for m in u.phases_by_mode]

    # Re-score all requested jobs; replace outputs only after full success.
    todo = list(all_jobs)
    rec_path_ = records_path(args.novel, args.character,
                             args.judge_model, args.include_thought)
    if rec_path_.exists():
        print(f"note: {rec_path_.name} already exists and will be replaced "
              "after a successful run.")

    if args.limit is not None:
        todo = todo[:args.limit]

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
