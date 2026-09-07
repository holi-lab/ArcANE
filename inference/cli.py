"""CLI entrypoint:  uv run python -m inference  ...

Example:
    uv run python -m inference \
        --novel "Anne of Green Gables" \
        --character anne_shirley \
        --mode vanilla,arc \
        --model gpt-5.4 \
        --limit 8 \
        --dry-run
"""

import argparse
import asyncio
import logging

from . import config
from .loaders import iter_trials
from .runner import run


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="inference")
    p.add_argument("--novel", required=True, help="e.g. 'Anne of Green Gables'")
    p.add_argument("--character", required=True,
                   help="character slug, e.g. anne_shirley")
    p.add_argument("--variant", default="main",
                   help="probe file variant (default: main)")
    p.add_argument("--mode", default="vanilla",
                   help=f"comma-separated; one or more of {','.join(config.SUPPORTED_MODES)}")
    p.add_argument("--model", default=config.EXP_MODEL_DEFAULT,
                   help="comma-separated chat-completion models")
    p.add_argument("--probe-types", default=None,
                   help="comma-separated subset of in_text,in_world,out_of_world")
    p.add_argument("--limit", type=int, default=None,
                   help="cap on number of trials (after filtering); useful for smoke tests")
    p.add_argument("--include-unavailable", action="store_true",
                   help="also run trials whose ground-truth response was marked unavailable")
    p.add_argument("--no-record-prompts", action="store_true",
                   help="do not persist prompt_system/prompt_user (only their hash + len)")
    p.add_argument("--dry-run", action="store_true",
                   help="print JSONL previews to stdout without API calls or result-file writes; "
                        "API-dependent context is deferred")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    modes  = _parse_csv(args.mode)
    models = _parse_csv(args.model)
    for m in modes:
        if m not in config.SUPPORTED_MODES:
            raise SystemExit(f"unsupported mode: {m!r}; supported: {config.SUPPORTED_MODES}")

    probe_types = set(_parse_csv(args.probe_types)) if args.probe_types else None

    trials = list(iter_trials(
        novel=args.novel,
        character_slug=args.character,
        variant=args.variant,
        probe_types=probe_types,
        skip_unavailable=not args.include_unavailable,
    ))
    if args.limit is not None:
        trials = trials[:args.limit]
    if not trials:
        raise SystemExit("no trials matched the filters")

    asyncio.run(run(
        novel=args.novel,
        character_slug=args.character,
        trials=trials,
        modes=modes,
        models=models,
        variant=args.variant,
        record_prompts=not args.no_record_prompts,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
