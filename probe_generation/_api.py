"""Shared OpenAI structured-output call helper.

`parse_call` issues one chat-completions request with `response_format` set
to a Pydantic schema and returns the parsed instance as a dict. Models that
reject custom temperature (gpt-5.x reasoning models) are auto-detected on
first 400 and the parameter is dropped on retry.

Lives in its own module so pipeline.py and text_grounding.py share both the
implementation and the `_MODELS_REJECT_TEMPERATURE` cache — one rejection
teaches every stage.
"""

import asyncio
import logging
from typing import Type

from openai import AsyncOpenAI
from pydantic import BaseModel

from .config import MAX_RETRIES, RETRY_BACKOFF_BASE

logger = logging.getLogger(__name__)

_MODELS_REJECT_TEMPERATURE: set[str] = set()


async def parse_call(client: AsyncOpenAI, sem: asyncio.Semaphore,
                     model: str, system: str, user: str,
                     schema: Type[BaseModel], temperature: float) -> dict:
    cur_temp = None if model in _MODELS_REJECT_TEMPERATURE else temperature
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "response_format": schema,
                }
                if cur_temp is not None:
                    kwargs["temperature"] = cur_temp
                resp = await client.beta.chat.completions.parse(**kwargs)
            msg = resp.choices[0].message
            if getattr(msg, "refusal", None):
                raise RuntimeError(f"model refused: {msg.refusal}")
            parsed = msg.parsed
            if parsed is None:
                raise RuntimeError("structured-output parse returned None")
            return parsed.model_dump()
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
