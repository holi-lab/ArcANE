"""Precompute the RAG index used by `rag` mode.

Cache layout:
  cache/inference/{novel}/rag/chunks.jsonl     # one chunk per line
  cache/inference/{novel}/rag/embeddings.npy   # (N, D) float32

Each chunk is a dict {chapter, chunk_idx, text}. The runner filters by
chapter so we only retrieve from chapters at-or-before query_chapter.

Usage:
  uv run python -m inference.build_index --novel "Anne of Green Gables"
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from . import config
from ._api import embed_batch
from .loaders import list_chapters, read_chapter

logger = logging.getLogger(__name__)


def _chunk_text(text: str, size: int, stride: int) -> list[str]:
    """Split text into overlapping character windows."""
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        if i + size >= len(text):
            break
        i += stride
    return out


async def _amain(args: argparse.Namespace) -> None:
    chapters = list_chapters(args.novel)
    if not chapters:
        raise SystemExit(f"no chapter files under {config.chapter_dir(args.novel)}")

    out_dir: Path = config.rag_dir(args.novel)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_p = out_dir / "chunks.jsonl"
    embs_p   = out_dir / "embeddings.npy"
    if chunks_p.exists() and embs_p.exists() and not args.force:
        logger.info(f"index already exists at {out_dir}; pass --force to rebuild")
        return

    # build chunks
    all_chunks: list[dict] = []
    for ch in chapters:
        text = read_chapter(args.novel, ch)
        for i, piece in enumerate(_chunk_text(text, args.chunk_chars, args.stride_chars)):
            all_chunks.append({"chapter": ch, "chunk_idx": i, "text": piece})
    logger.info(f"chunked {len(chapters)} chapters into {len(all_chunks)} chunks "
                f"(size={args.chunk_chars} stride={args.stride_chars})")

    # embed in batches
    client = config.make_embed_client()
    sem = asyncio.Semaphore(config.MAX_CONCURRENT_API_CALLS)
    batch_size = 96

    async def _do_batch(start: int) -> tuple[int, list[list[float]]]:
        inputs = [c["text"] for c in all_chunks[start:start + batch_size]]
        vecs = await embed_batch(client, sem, args.model, inputs)
        return start, vecs

    tasks = [asyncio.create_task(_do_batch(s)) for s in range(0, len(all_chunks), batch_size)]
    results: list[tuple[int, list[list[float]]]] = []
    done_batches = 0
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)
        done_batches += 1
        if done_batches % 5 == 0 or done_batches == len(tasks):
            logger.info(f"  embedded {done_batches}/{len(tasks)} batches")

    # assemble in order
    results.sort(key=lambda x: x[0])
    flat: list[list[float]] = []
    for _, vecs in results:
        flat.extend(vecs)
    assert len(flat) == len(all_chunks), "embedding/chunk count mismatch"

    embs = np.array(flat, dtype=np.float32)
    np.save(embs_p, embs)
    with chunks_p.open("w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info(f"wrote {embs.shape[0]} embeddings (dim={embs.shape[1]}) to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(prog="inference.build_index")
    p.add_argument("--novel", required=True)
    p.add_argument("--model", default=config.EXP_EMBED_MODEL)
    p.add_argument("--chunk-chars", type=int, default=config.RAG_CHUNK_CHARS)
    p.add_argument("--stride-chars", type=int, default=config.RAG_CHUNK_STRIDE)
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
