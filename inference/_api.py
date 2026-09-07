"""Async OpenAI helpers used by the experiment runner.

Two surfaces:

  chat_text(...)        — free-form chat completion → returns (text, usage_dict)
                          Used for the role-play call and for chapter
                          summarization. No structured-output schema; the
                          character is allowed to respond however it wants.

  embed_batch(...)      — embeddings batched up to 96 inputs per call.

`gpt-5.x` family rejects custom `temperature`; the rejection is detected on
the first 400 and cached so subsequent calls drop the parameter automatically.
"""

import asyncio
import logging
import re
import time
from typing import Optional

from openai import AsyncOpenAI

from .config import (
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    DISABLE_THINKING,
    OPENROUTER_PROVIDER,
    OPENROUTER_DISABLE_REASONING,
)

logger = logging.getLogger(__name__)

_MODELS_REJECT_TEMPERATURE: set[str] = set()


_THINK_BLOCK_RE = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_think_block(text: Optional[str]) -> Optional[str]:
    """Drop a leading <think>…</think> block from a model reply.

    Qwen3-family templates emit the block even in non-thinking mode (usually
    empty), and a checkpoint fine-tuned on those targets keeps emitting it
    whether or not the flag reaches the server. The judge scores the visible
    reply, so the block is removed before the response is stored — otherwise
    a served-model name that misses the flag silently changes what is judged.
    """
    if not text:
        return text
    return _THINK_BLOCK_RE.sub("", text, count=1)


def _disable_thinking_for(model: str) -> bool:
    """Should this call send chat_template_kwargs={"enable_thinking": False}?

    `EXP_DISABLE_THINKING=1|0` decides outright. Unset, we match on family
    name: Qwen3 and the Qwen3-derived ArcANE checkpoints understand the flag,
    while other backends (OpenAI, Anthropic, DeepSeek, Qwen2, Llama) reject an
    unknown chat_template_kwargs with a 400, so they must not receive it.
    """
    if DISABLE_THINKING in {"1", "true", "yes", "on"}:
        return True
    if DISABLE_THINKING in {"0", "false", "no", "off"}:
        return False
    m = model.lower()
    return "qwen3" in m or "qwen-3" in m or "arcane" in m


def _build_extra_body(model: str) -> Optional[dict]:
    """Merge per-call extra_body for OpenAI-compat chat completions.

    - Non-thinking mode: see `_disable_thinking_for` (Qwen3 / ArcANE by
      default, forced either way by EXP_DISABLE_THINKING).
    - OpenRouter: when EXP_OPENROUTER_PROVIDER is set, pin the upstream
      provider so we don't fan out across mirrors that may have different
      quantisation / sampling behaviour (e.g. fp8 vs fp4). `allow_fallbacks`
      is forced False so we fail loudly rather than silently routing to a
      different provider.
    - OpenRouter reasoning: when EXP_OPENROUTER_DISABLE_REASONING is set,
      skip the model's reasoning pass entirely. Uses `reasoning.enabled=false`
      (the universal OpenRouter flag), which is honoured by hybrid reasoning
      models such as DeepSeek v4. Distinct from `exclude:true`, which would
      still run reasoning and only hide the trace — we want to actually save
      the tokens, not just the bytes on the wire.
    """
    body: dict = {}
    if _disable_thinking_for(model):
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if OPENROUTER_PROVIDER:
        body["provider"] = {
            "only": [OPENROUTER_PROVIDER],
            "allow_fallbacks": False,
        }
    if OPENROUTER_DISABLE_REASONING:
        body["reasoning"] = {"enabled": False}
    return body or None


async def chat_text(client: AsyncOpenAI, sem: asyncio.Semaphore,
                    model: str, system: str, user: str,
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None) -> tuple[str, dict, int]:
    """One chat completion → (text, usage, latency_ms).

    `usage` is the raw `.usage.model_dump()` dict so the caller can persist
    prompt/completion token counts. `latency_ms` is wall time of the API leg.
    """
    cur_temp = None if model in _MODELS_REJECT_TEMPERATURE else temperature
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                t0 = time.perf_counter()
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                }
                if cur_temp is not None:
                    kwargs["temperature"] = cur_temp
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                extra_body = _build_extra_body(model)
                if extra_body is not None:
                    kwargs["extra_body"] = extra_body
                resp = await client.chat.completions.create(**kwargs)
                latency_ms = int((time.perf_counter() - t0) * 1000)
            msg = resp.choices[0].message
            if getattr(msg, "refusal", None):
                raise RuntimeError(f"model refused: {msg.refusal}")
            text = msg.content or ""
            usage = resp.usage.model_dump() if resp.usage else {}
            return text, usage, latency_ms
        except Exception as e:
            err = str(e)
            if "temperature" in err and cur_temp is not None:
                _MODELS_REJECT_TEMPERATURE.add(model)
                logger.info(f"[api] {model} rejects custom temperature; dropping")
                cur_temp = None
                continue
            wait = RETRY_BACKOFF_BASE ** (attempt + 1)
            logger.warning(f"[api] error attempt {attempt+1}: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("unreachable")


async def embed_batch(client: AsyncOpenAI, sem: asyncio.Semaphore,
                      model: str, inputs: list[str]) -> list[list[float]]:
    """Embed up to 96 inputs in a single call; caller batches if needed."""
    assert 1 <= len(inputs) <= 96, "OpenAI accepts up to ~2048; we cap at 96 to keep retries cheap"
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.embeddings.create(model=model, input=inputs)
            return [d.embedding for d in resp.data]
        except Exception as e:
            wait = RETRY_BACKOFF_BASE ** (attempt + 1)
            logger.warning(f"[embed] error attempt {attempt+1}: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("unreachable")
