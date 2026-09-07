"""Shared validation for judge scores."""

import math
from collections.abc import Sequence


class EvaluationRunError(RuntimeError):
    """An evaluation attempt failed without replacing its published outputs."""


def validate_scale(scale: int) -> None:
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        raise ValueError("score scale must be a positive integer")


def scores_with_average(verdict: object, dimensions: Sequence[str],
                        scale: int) -> dict[str, float] | None:
    """Validate dimension scores and derive their mean, ignoring a supplied mean."""
    validate_scale(scale)
    if not isinstance(verdict, dict) or not dimensions:
        return None
    raw = verdict.get("scores")
    if not isinstance(raw, dict):
        return None
    values: dict[str, float] = {}
    for dim in dimensions:
        value = raw.get(dim)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not 1 <= value <= scale or not math.isfinite(value)):
            return None
        values[dim] = float(value)
    values["average"] = round(sum(values.values()) / len(values), 2)
    return values
