"""Configuration for evaluation.per_response (the LLM judge).

Kept separate from the role-play config (`inference/config.py`) so the judge
— model, backend, pricing — can be tuned on its own. Every value is overridable
by an environment variable.

Two judge backends are supported, selected via `EXP_JUDGE_BACKEND`:
  openrouter  — OpenRouter, deepseek/deepseek-v4-flash by default. **Default
                backend.** This is the paper's main judge. Any OpenRouter
                model can be picked via EXP_JUDGE_MODEL.
  openai      — direct OpenAI API, gpt-5.4-mini by default. Used for the
                GPT-5.5 leg of the cross-judge replication
                (EXP_JUDGE_BACKEND=openai EXP_JUDGE_MODEL=gpt-5.5).

The backend setting only flips the *defaults* (base URL / model / pricing /
API-key fallback). Anthropic cross-judges (claude-opus-4-5, claude-sonnet-4-5)
are not served by this module; the paper produced those scores via a
separate validation runner using the native Anthropic SDK.

Env overrides:
  EXP_EVAL_NOVEL         — default novel (default: Anna_Kareina)
  EXP_EVAL_CHARACTER     — default character slug (default: anna_karenina)
  EXP_EVAL_CONCURRENCY   — judge thread-pool size (default: 200)
  EXP_EVAL_SCORE_SCALE   — judge score scale, e.g. 100 or 5 (default: 100)
  EXP_EVAL_INCLUDE_THOUGHT — include ref_thought in the reference shown to
                              the judge (default: true). When false, the
                              prompt instructs the judge to infer reference
                              reasoning from ref_action/ref_speech, and
                              outputs land in `*_nothought.*` sibling files
                              so the two ablations never mix.
  EXP_JUDGE_BACKEND      — "openrouter" (default) or "openai"
  EXP_JUDGE_MODEL        — judge model (backend-dependent default)
  EXP_JUDGE_BASE_URL     — judge endpoint (backend-dependent default)
  EXP_JUDGE_API_KEY      — judge API key; falls back to OPENAI_API_KEY /
                            OPENROUTER_API_KEY depending on backend
  EXP_JUDGE_PRICE_INPUT  — judge price, USD per 1M input tokens
  EXP_JUDGE_PRICE_OUTPUT — judge price, USD per 1M output tokens
  EXP_JUDGE_PROVIDER_ORDER        — (openrouter only) comma-separated provider
                                     preference order (default:
                                     "DeepSeek,Alibaba"). Empty omits the
                                     provider order from the request.
  EXP_JUDGE_PROVIDER_ALLOW_FALLBACK — "true"/"false". When false (default), the
                                     request disables fallback routing beyond
                                     the configured provider order.
  EXP_JUDGE_DISABLE_REASONING — (openrouter only) "true"/"false". When true
                                 (default), the judge call sends
                                 `reasoning: {enabled: false}`.
"""

import os
from typing import Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── evaluation target ──────────────────────────────────────────────
# Default (novel, character); the CLI --novel / --character override these.
EVAL_NOVEL:     str = os.environ.get("EXP_EVAL_NOVEL", "Anna_Kareina")
EVAL_CHARACTER: str = os.environ.get("EXP_EVAL_CHARACTER", "anna_karenina")

# ── run ────────────────────────────────────────────────────────────
# Judge thread-pool size — the runner fires this many judge calls at once
# via a ThreadPoolExecutor. Kept separate from the role-play runner's
# config.MAX_CONCURRENT_API_CALLS.
EVAL_CONCURRENCY: int = int(os.environ.get("EXP_EVAL_CONCURRENCY", "200"))

# ── scoring ────────────────────────────────────────────────────────
# Judge score scale: each dimension is scored 1..SCORE_SCALE. 100 by default;
# set 5 for the original 1-5 rubric.
SCORE_SCALE: int = int(os.environ.get("EXP_EVAL_SCORE_SCALE", "100"))

# Whether the reference shown to the judge includes ref_thought. When False,
# the user prompt omits the `ref_thought:` line and the system prompt switches
# to the "infer reasoning from ref_action/ref_speech" variant. Result files
# get a `_nothought` suffix so the two conditions never share a JSONL.
INCLUDE_THOUGHT: bool = (
    os.environ.get("EXP_EVAL_INCLUDE_THOUGHT", "true")
    .lower() in ("true", "1", "yes"))

# ── judge backend & defaults ───────────────────────────────────────
# Backend picks the *default* model / base URL / pricing / API-key fallback.
# Any of those can still be overridden per-env-var below.
JUDGE_BACKEND: str = os.environ.get("EXP_JUDGE_BACKEND", "openrouter").lower()

# Backend-specific defaults. Override EXP_JUDGE_PRICE_* for the selected
# model and provider when estimating costs.
_BACKEND_DEFAULTS: dict[str, dict] = {
    "openai": {
        "model":         "gpt-5.4-mini",
        "base_url":      None,                          # openai.com
        "api_key_env":   "OPENAI_API_KEY",
        "price_input":   0.75,
        "price_output":  4.50,
    },
    "openrouter": {
        "model":         "deepseek/deepseek-v4-flash",
        "base_url":      "https://openrouter.ai/api/v1",
        "api_key_env":   "OPENROUTER_API_KEY",
        # Cost-estimate defaults; actual rates depend on the model and provider.
        "price_input":   0.14,
        "price_output":  0.28,
    },
}
if JUDGE_BACKEND not in _BACKEND_DEFAULTS:
    raise ValueError(
        f"unknown EXP_JUDGE_BACKEND={JUDGE_BACKEND!r}; "
        f"expected one of {sorted(_BACKEND_DEFAULTS)}"
    )
_defaults = _BACKEND_DEFAULTS[JUDGE_BACKEND]

JUDGE_MODEL:    str           = os.environ.get("EXP_JUDGE_MODEL", _defaults["model"])
JUDGE_BASE_URL: Optional[str] = os.environ.get("EXP_JUDGE_BASE_URL") or _defaults["base_url"]
JUDGE_API_KEY:  str           = (os.environ.get("EXP_JUDGE_API_KEY")
                                 or os.environ.get(_defaults["api_key_env"])
                                 or "EMPTY")

# ── OpenRouter provider pinning ────────────────────────────────────
# Comma-separated provider preference order; empty omits the provider order.
JUDGE_PROVIDER_ORDER: list[str] = [
    p.strip() for p in
    os.environ.get("EXP_JUDGE_PROVIDER_ORDER", "DeepSeek,Alibaba").split(",")
    if p.strip()
]

# Disable fallback routing beyond the configured provider order by default.
JUDGE_PROVIDER_ALLOW_FALLBACK: bool = (
    os.environ.get("EXP_JUDGE_PROVIDER_ALLOW_FALLBACK", "false")
    .lower() in ("true", "1", "yes"))

# Request non-reasoning judge responses on OpenRouter by default.
JUDGE_DISABLE_REASONING: bool = (
    os.environ.get("EXP_JUDGE_DISABLE_REASONING", "true")
    .lower() in ("true", "1", "yes"))


# ── judge pricing (USD per 1M tokens) ──────────────────────────────
# Rates used for token-cost estimates. Override with EXP_JUDGE_PRICE_* for
# the selected model and provider.
JUDGE_PRICE_INPUT:  float = float(os.environ.get("EXP_JUDGE_PRICE_INPUT",
                                                  str(_defaults["price_input"])))
JUDGE_PRICE_OUTPUT: float = float(os.environ.get("EXP_JUDGE_PRICE_OUTPUT",
                                                  str(_defaults["price_output"])))

# ── cost-estimate heuristics ───────────────────────────────────────
# Rough token approximations for the pre-run estimate (no real tokenizer).
EST_CHARS_PER_TOKEN        = 4      # English chars per token
EST_OUTPUT_TOKENS_PER_MODE = 30     # size of one JSONL score object


def make_judge_client() -> OpenAI:
    """Synchronous OpenAI-compatible client for the evaluation judge.

    Works for both `openai` and `openrouter` backends since OpenRouter speaks
    the OpenAI Chat Completions protocol — only `base_url` and the API key
    differ. The httpx connection pool is sized to EVAL_CONCURRENCY so every
    judge thread can hold a live connection at once (httpx defaults to 100).
    """
    limit = max(EVAL_CONCURRENCY, 1)
    return OpenAI(
        api_key=JUDGE_API_KEY,
        base_url=JUDGE_BASE_URL,
        http_client=httpx.Client(
            limits=httpx.Limits(max_connections=limit,
                                 max_keepalive_connections=limit),
            timeout=httpx.Timeout(120.0, connect=10.0),
        ),
    )
