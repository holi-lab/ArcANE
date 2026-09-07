"""Precompute per-chapter summaries used by `summary` mode.

Cache layout:
  cache/inference/{novel}/chapter_summaries/chapter_XX.txt

Usage:
  uv run python -m inference.build_summaries --novel "Anne of Green Gables"

Idempotent: existing summary files are left in place unless --force is passed.
Failed chapters are reported and skipped (the summary mode prompt surfaces
missing chapters inline so they can be regenerated later).
"""

import argparse
import asyncio
import logging

from openai import AsyncOpenAI

from . import config
from ._api import chat_text
from .loaders import list_chapters, read_chapter
from .prompts import SUMMARY_SYSTEM, render_summary_user

logger = logging.getLogger(__name__)


async def _summarize_one(client: AsyncOpenAI, sem: asyncio.Semaphore,
                         novel: str, model: str,
                         chapter_idx: int) -> tuple[int, str]:
    text = read_chapter(novel, chapter_idx)
    user = render_summary_user(novel, chapter_idx, text)
    summary, _, _ = await chat_text(client, sem, model, SUMMARY_SYSTEM, user)
    return chapter_idx, summary.strip()


async def _amain(args: argparse.Namespace) -> None:
    chapters = list_chapters(args.novel)
    if not chapters:
        raise SystemExit(f"no chapter files under {config.chapter_dir(args.novel)}")

    out_dir = config.summary_dir(args.novel)
    out_dir.mkdir(parents=True, exist_ok=True)

    to_do = [
        ch for ch in chapters
        if args.force or not config.summary_path(args.novel, ch).exists()
    ]
    if not to_do:
        logger.info("all chapter summaries already present")
        return

    logger.info(f"summarizing {len(to_do)} chapters (model={args.model})")
    client = config.make_chat_client()
    sem = asyncio.Semaphore(config.MAX_CONCURRENT_API_CALLS)

    tasks = [
        asyncio.create_task(_summarize_one(
            client, sem, args.novel, args.model, ch,
        )) for ch in to_do
    ]
    done = 0
    for coro in asyncio.as_completed(tasks):
        try:
            ch, summary = await coro
        except Exception as e:
            logger.warning(f"chapter summarization failed: {e}")
            continue
        config.summary_path(args.novel, ch).write_text(summary, encoding="utf-8")
        done += 1
        if done % 5 == 0 or done == len(tasks):
            logger.info(f"  {done}/{len(tasks)} chapters summarized")


def main() -> None:
    p = argparse.ArgumentParser(prog="inference.build_summaries")
    p.add_argument("--novel", required=True)
    p.add_argument("--model", default=config.EXP_MODEL_SUMMARY)
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
