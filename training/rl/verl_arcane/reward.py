"""Portable reward functions for ArcANE GRPO training.

verl calls compute_score once per generated response. The default heuristic
mode is intended for smoke tests. Release training should normally use remote
mode and provide an HTTP reward service through ARCANE_REWARD_URL.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any


_REFUSAL = re.compile(
    r"\b(as an ai|i cannot|i can't|i am unable|i'm unable|language model|"
    r"cannot fulfill|i'm sorry, but)\b",
    re.IGNORECASE,
)


def _heuristic_score(solution_str: str) -> dict[str, float]:
    """Return a cheap, reference-free score for pipeline smoke tests."""
    text = (solution_str or "").strip()
    words = text.split()
    n_words = len(words)
    result = {
        "score": 0.0,
        "h_empty": 0.0,
        "h_refusal": 0.0,
        "h_length": 0.0,
        "h_variety": 0.0,
        "h_substance": 0.0,
        "h_n_words": float(n_words),
    }
    if not text:
        result["h_empty"] = 1.0
        return result
    if _REFUSAL.search(text):
        result["score"] = 0.05
        result["h_refusal"] = 1.0
        return result

    if n_words < 40:
        length_score = n_words / 40.0
    elif n_words <= 400:
        length_score = 1.0
    else:
        length_score = max(0.0, 1.0 - (n_words - 400) / 400.0)

    variety_score = len({word.lower() for word in words}) / n_words
    has_speech = float(bool(re.search(r"[\"“”']", text)))
    has_action = float(
        bool(re.search(r"\*[^*]+\*|\b(turn|step|look|hand|eye|voice|walk|rise|stand)\w*", text, re.I))
    )
    substance_score = 0.5 * (has_speech + has_action)
    result.update(
        score=float(min(1.0, 0.5 * length_score + 0.3 * variety_score + 0.2 * substance_score)),
        h_length=round(length_score, 3),
        h_variety=round(variety_score, 3),
        h_substance=round(substance_score, 3),
    )
    return result


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    timeout = float(os.environ.get("ARCANE_REWARD_TIMEOUT", "120"))
    retries = max(0, int(os.environ.get("ARCANE_REWARD_RETRIES", "3")))
    retry_delay = float(os.environ.get("ARCANE_REWARD_RETRY_DELAY", "2"))
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
            if not isinstance(value, dict):
                raise TypeError("reward service must return a JSON object")
            return value
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == retries:
                raise RuntimeError(f"reward request to {url} failed") from error
            time.sleep(retry_delay)

    raise AssertionError("unreachable")


def _remote_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Score a response with a user-provided HTTP service.

    The service receives the same context exposed by verl and must return a JSON
    object containing a numeric score. Extra fields are logged by verl.
    """
    url = os.environ.get("ARCANE_REWARD_URL")
    if not url:
        raise RuntimeError(
            "remote reward mode requires ARCANE_REWARD_URL, for example "
            "http://127.0.0.1:8000/score"
        )
    result = _post_json(
        url,
        {
            "data_source": data_source,
            "response": solution_str or "",
            "ground_truth": ground_truth,
            "extra_info": extra_info or {},
        },
    )
    score = result.get("score")
    if not isinstance(score, (int, float)):
        raise TypeError("reward service response must contain a numeric 'score'")
    result["score"] = float(score)
    return result


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **reward_kwargs: Any,
) -> float | dict[str, Any]:
    """Compute a reward using heuristic, remote, or zero mode."""
    mode = str(reward_kwargs.get("mode") or os.environ.get("ARCANE_REWARD_MODE", "heuristic")).lower()
    if mode == "heuristic":
        return _heuristic_score(solution_str)
    if mode in {"remote", "rm"}:
        return _remote_score(data_source, solution_str, ground_truth, extra_info)
    if mode == "zero":
        return 0.0
    raise ValueError(f"unknown ArcANE reward mode: {mode!r}; expected heuristic, remote, or zero")
