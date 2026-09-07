"""Experiment configuration.

Env overrides:
  EXP_MODEL_DEFAULT     — completion model for role-play (default: gpt-5.4)
  EXP_MODEL_SUMMARY     — model used when building chapter summaries
  EXP_EMBED_MODEL       — embedding model for RAG (default: text-embedding-3-small)
  OPENAI_API_KEY        — fallback API key for both chat and embed backends

Backend selection (OpenAI by default; point at a vLLM/OpenAI-compatible
server to switch). Chat and embedding backends are configured independently
so the common "vLLM for chat, OpenAI for embeddings" setup works out of the
box:

  EXP_CHAT_BASE_URL     — e.g. http://localhost:8000/v1  (unset → openai.com)
  EXP_CHAT_API_KEY      — defaults to OPENAI_API_KEY, else "EMPTY"
  EXP_EMBED_BASE_URL    — e.g. http://localhost:8001/v1  (unset → openai.com)
  EXP_EMBED_API_KEY     — defaults to OPENAI_API_KEY, else "EMPTY"

OpenRouter convenience:
  EXP_OPENROUTER_PROVIDER          — when chatting through OpenRouter, pin an
                                     upstream provider (e.g. "alibaba",
                                     "deepseek"). Forces allow_fallbacks=False
                                     so routing is deterministic across runs
                                     and quantisations.
  EXP_ROLEPLAY_MAX_TOKENS          — completion cap for the role-play call
                                     (default: 8192, the paper's setting).
  EXP_TIMECHARA_MAX_TOKENS         — completion cap for the TimeChara expert
                                     hint calls (default: 512).
  EXP_DISABLE_THINKING             — "1"/"true" to force non-thinking mode on
                                     every model, "0"/"false" to never send the
                                     flag. Unset: sent to Qwen3-family and
                                     ArcANE checkpoints only.
  EXP_OPENROUTER_DISABLE_REASONING — "1"/"true" to send reasoning.enabled=false
                                     on every chat call (skips the thinking
                                     pass for hybrid reasoning models like
                                     DeepSeek v4). Saves billed reasoning
                                     tokens, not just hides them.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────
# Official/ layout (all artifacts under the repo root):
#   results/arc_extraction/{novel}/...        ← arc_construction outputs
#   results/inference/{novel}/{character}/    ← role-play JSONLs + judge outputs
#   cache/inference/{novel}/...               ← chapter summaries + RAG index
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARC_RESULTS  = PROJECT_ROOT / "results" / "arc_extraction"
CACHE_ROOT   = PROJECT_ROOT / "cache" / "inference"
RESULTS_ROOT = PROJECT_ROOT / "results" / "inference"


def probe_file(novel: str, character_slug: str, variant: str = "main") -> Path:
    """Path to a probe JSON for (novel, character).

    `variant=="main"` resolves to `{slug}_probes.json`; otherwise
    `{slug}_probes_{variant}.json`. Some novels store probes under a versioned
    folder (e.g. `probes_v1/`); the loader will fall back to those.
    """
    base = ARC_RESULTS / novel
    name = f"{character_slug}_probes.json" if variant == "main" \
        else f"{character_slug}_probes_{variant}.json"
    for folder in ("probes", "probes_v1", "probes_v2", "probes_v3"):
        candidate = base / folder / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No probe file for novel={novel!r} character={character_slug!r} variant={variant!r}"
    )


def axes_file(novel: str, character_slug: str) -> Path:
    """`final_auto/{slug}_auto_axes.json` — character arc used by `arc` mode."""
    return ARC_RESULTS / novel / "final_auto" / f"{character_slug}_auto_axes.json"


def chapter_dir(novel: str) -> Path:
    return ARC_RESULTS / novel / "chapters"


def chapter_path(novel: str, chapter_idx: int) -> Path:
    return chapter_dir(novel) / f"chapter_{chapter_idx:02d}.txt"


def summary_dir(novel: str) -> Path:
    return CACHE_ROOT / novel / "chapter_summaries"


def summary_path(novel: str, chapter_idx: int) -> Path:
    return summary_dir(novel) / f"chapter_{chapter_idx:02d}.txt"


def rag_dir(novel: str) -> Path:
    return CACHE_ROOT / novel / "rag"


def results_path(novel: str, character_slug: str, mode: str, model: str) -> Path:
    safe_model = model.replace("/", "_").replace(":", "_")
    return RESULTS_ROOT / novel / character_slug / f"{mode}__{safe_model}.jsonl"


# ── API ────────────────────────────────────────────────────────────
# OPENAI_API_KEY is the fallback key for both backends. Optional now:
# a vLLM-only setup can leave it unset and the per-backend keys fall back
# to "EMPTY" (which vLLM accepts).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

EXP_MODEL_DEFAULT = os.environ.get("EXP_MODEL_DEFAULT", "gpt-5.4")
EXP_MODEL_SUMMARY = os.environ.get("EXP_MODEL_SUMMARY", EXP_MODEL_DEFAULT)
EXP_EMBED_MODEL   = os.environ.get("EXP_EMBED_MODEL", "text-embedding-3-small")

# Per-backend base_url / key. `None` for base_url ⇒ default OpenAI endpoint.
EXP_CHAT_BASE_URL: Optional[str]  = os.environ.get("EXP_CHAT_BASE_URL") or None
EXP_CHAT_API_KEY:  str            = os.environ.get("EXP_CHAT_API_KEY") or OPENAI_API_KEY or "EMPTY"
EXP_EMBED_BASE_URL: Optional[str] = os.environ.get("EXP_EMBED_BASE_URL") or None
EXP_EMBED_API_KEY:  str           = os.environ.get("EXP_EMBED_API_KEY") or OPENAI_API_KEY or "EMPTY"

# OpenRouter provider pin. Empty string ⇒ no pin (default OpenRouter routing).
OPENROUTER_PROVIDER: str = os.environ.get("EXP_OPENROUTER_PROVIDER", "").strip()

# When truthy, send `reasoning: {enabled: false}` on every chat call so hybrid
# reasoning models (e.g. DeepSeek v4) skip the thinking pass — for roleplay we
# only want the in-character reply, not the chain-of-thought.
# Completion caps. The paper generates role-play answers under an 8,192-token
# ceiling (App. D "Inference Details"); the TimeChara expert pass only produces
# a short temporal/spatial hint, so it gets a much smaller one.
ROLEPLAY_MAX_TOKENS: int = int(os.environ.get("EXP_ROLEPLAY_MAX_TOKENS", "8192"))
TIMECHARA_MAX_TOKENS: int = int(os.environ.get("EXP_TIMECHARA_MAX_TOKENS", "512"))

# Qwen3-family chat templates wrap the reply in <think>…</think>. Role-play
# runs use non-thinking mode (paper App. D), which the template exposes as
# chat_template_kwargs={"enable_thinking": False}. "1"/"0" forces the flag on
# or off for every model; unset falls back to a family-name match that covers
# Qwen3 and the Qwen3-derived ArcANE checkpoints.
DISABLE_THINKING: str = os.environ.get("EXP_DISABLE_THINKING", "").strip().lower()

OPENROUTER_DISABLE_REASONING: bool = os.environ.get(
    "EXP_OPENROUTER_DISABLE_REASONING", ""
).strip().lower() in ("1", "true", "yes", "on")


def make_chat_client() -> AsyncOpenAI:
    """AsyncOpenAI client for chat completions (chat backend)."""
    return AsyncOpenAI(api_key=EXP_CHAT_API_KEY, base_url=EXP_CHAT_BASE_URL)


def make_embed_client() -> AsyncOpenAI:
    """AsyncOpenAI client for embeddings (embed backend)."""
    return AsyncOpenAI(api_key=EXP_EMBED_API_KEY, base_url=EXP_EMBED_BASE_URL)

# ── Concurrency / retry ───────────────────────────────────────────
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2
MAX_CONCURRENT_API_CALLS = 64

# ── Mode whitelist ─────────────────────────────────────────────────
SUPPORTED_MODES = ("vanilla", "arc", "summary", "rag", "timechara", "lifechoice")

# ── RAG defaults ───────────────────────────────────────────────────
RAG_CHUNK_CHARS  = 1800          # ~ paragraphs; characters not tokens, kept rough on purpose
RAG_CHUNK_STRIDE = 1500          # 300-char overlap
RAG_TOP_K        = 6

# ── Summary defaults ───────────────────────────────────────────────
SUMMARY_MAX_CHAPTERS = 5         # `summary` mode keeps only the last N chapters
                                 # before the query point (full history blew past
                                 # 80k tokens on long novels). Tightened from 10
                                 # → 5 once we saw that the recent-window already
                                 # dominated probe-relevant evidence.
