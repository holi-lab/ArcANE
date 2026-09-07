"""Build summary and RAG caches for every novel that has a per-chapter split
under `results/arc_extraction/{novel}/chapters/`.

Runs sequentially per novel (each builder internally fans out under
`MAX_CONCURRENT_API_CALLS`). Already-built caches are skipped unless
--force is passed. Per-novel failures are logged but do not stop the run.

Usage:
    uv run python -m inference.build_all                       # both, all novels
    uv run python -m inference.build_all --only summary
    uv run python -m inference.build_all --novels Persuasion,Jane_Eyre
    uv run python -m inference.build_all --skip "Great Expectations,Monte_Cristo"
"""

import argparse
import asyncio
import logging

from . import config, build_summaries, build_index

logger = logging.getLogger(__name__)


def _discover_novels() -> list[str]:
    """All novels with a non-empty chapters/ directory."""
    out: list[str] = []
    for chdir in sorted(config.ARC_RESULTS.glob("*/chapters")):
        if any(chdir.glob("chapter_*.txt")):
            out.append(chdir.parent.name)
    return out


def _summary_already_built(novel: str) -> bool:
    """True iff every chapter under this novel has a summary file."""
    sdir = config.summary_dir(novel)
    if not sdir.exists():
        return False
    from .loaders import list_chapters
    expected = list_chapters(novel)
    if not expected:
        return False
    return all(config.summary_path(novel, ch).exists() for ch in expected)


def _rag_already_built(novel: str) -> bool:
    rdir = config.rag_dir(novel)
    return (rdir / "chunks.jsonl").exists() and (rdir / "embeddings.npy").exists()


async def _run_summary(novel: str, args: argparse.Namespace) -> None:
    ns = argparse.Namespace(
        novel=novel,
        model=args.summary_model,
        force=args.force,
    )
    await build_summaries._amain(ns)


async def _run_index(novel: str, args: argparse.Namespace) -> None:
    ns = argparse.Namespace(
        novel=novel,
        model=args.embed_model,
        chunk_chars=config.RAG_CHUNK_CHARS,
        stride_chars=config.RAG_CHUNK_STRIDE,
        force=args.force,
    )
    await build_index._amain(ns)


async def _amain(args: argparse.Namespace) -> None:
    discovered = _discover_novels()
    selected = [n for n in (args.novels or discovered) if n in discovered]
    if args.skip:
        skip = set(args.skip)
        selected = [n for n in selected if n not in skip]
    if args.novels:
        missing = [n for n in args.novels if n not in discovered]
        for n in missing:
            logger.warning(f"skip {n!r}: no chapters/ dir under {config.ARC_RESULTS}/{n}")
    if not selected:
        raise SystemExit("no novels matched")
    logger.info(f"build_all: {len(selected)} novels — {selected}")

    do_summary = args.only in (None, "summary")
    do_rag     = args.only in (None, "rag")

    summary_failures: list[tuple[str, str]] = []
    rag_failures:     list[tuple[str, str]] = []

    for i, novel in enumerate(selected, 1):
        logger.info(f"[{i}/{len(selected)}] {novel}")
        if do_summary:
            if _summary_already_built(novel) and not args.force:
                logger.info("  summary: cached, skip")
            else:
                try:
                    await _run_summary(novel, args)
                except Exception as e:
                    logger.error(f"  summary: FAILED — {e}")
                    summary_failures.append((novel, str(e)))
        if do_rag:
            if _rag_already_built(novel) and not args.force:
                logger.info("  rag: cached, skip")
            else:
                try:
                    await _run_index(novel, args)
                except Exception as e:
                    logger.error(f"  rag: FAILED — {e}")
                    rag_failures.append((novel, str(e)))

    logger.info("==== build_all done ====")
    if summary_failures:
        logger.warning(f"summary failures ({len(summary_failures)}):")
        for n, e in summary_failures:
            logger.warning(f"  {n}: {e}")
    if rag_failures:
        logger.warning(f"rag failures ({len(rag_failures)}):")
        for n, e in rag_failures:
            logger.warning(f"  {n}: {e}")


def _parse_csv(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(prog="inference.build_all")
    p.add_argument("--novels", type=_parse_csv, default=None,
                   help="comma-separated subset (default: all discovered)")
    p.add_argument("--skip", type=_parse_csv, default=None,
                   help="comma-separated novels to skip")
    p.add_argument("--only", choices=("summary", "rag"), default=None,
                   help="build only one cache type (default: both)")
    p.add_argument("--summary-model", default=config.EXP_MODEL_SUMMARY)
    p.add_argument("--embed-model",   default=config.EXP_EMBED_MODEL)
    p.add_argument("--force", action="store_true",
                   help="rebuild even if cache already present")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
