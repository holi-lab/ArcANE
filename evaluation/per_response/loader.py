"""Gather evaluation inputs.

Ground truth lives in the probe file
(`results/arc_extraction/{novel}/probes/{slug}_probes.json`); the model
responses live in the experiment results
(`results/inference/{novel}/{slug}/{mode}__{model}.jsonl`). Neither side
carries the other, so they are joined here on `(probe_id, phase_idx)`.

`build_eval_units` produces one `EvalUnit` per (trial, model): the trial's
ground truth plus that model's response under every mode it was run with.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from inference import config
from inference.records import is_dry_run_record
from .prompt import EvalTrial


@dataclass
class EvalUnit:
    """One trial scored for one model: the trial ground truth plus that
    model's response text under each mode."""

    novel: str
    character: str
    probe_id: str
    phase_idx: int
    model: str                       # real model name, e.g. "Qwen/Qwen3-8B"
    trial: EvalTrial                 # scenario / question / GT
    responses: dict[str, str]        # mode -> response text


# ── ground truth (probe file) ──────────────────────────────────────
def load_ground_truth(novel: str, character_slug: str,
                       variant: str = "main") -> dict[tuple[str, int], EvalTrial]:
    """Read the probe file → {(probe_id, phase_idx): EvalTrial}.

    `unavailable` phase responses (no salvageable ground truth) are skipped;
    they were never run, so no response references them either.
    """
    path = config.probe_file(novel, character_slug, variant)
    data = json.loads(path.read_text(encoding="utf-8"))
    gt: dict[tuple[str, int], EvalTrial] = {}
    for fam in data.get("families", []):
        for probe in fam.get("probes", []):
            probe_id = probe["probe_id"]
            scenario = probe.get("scenario", "")
            question = probe.get("question", "")
            for ph in probe.get("phase_responses", []):
                if ph.get("unavailable"):
                    continue
                gt[(probe_id, ph["phase_idx"])] = EvalTrial(
                    scenario=scenario,
                    question=question,
                    gt_action=ph.get("gt_action"),
                    gt_speech=ph.get("gt_speech"),
                    gt_thought=ph.get("gt_thought"),
                    phase=ph.get("phase_label"),
                )
    return gt


# ── model responses (experiment results) ───────────────────────────
def results_dir(novel: str, character_slug: str) -> Path:
    return config.RESULTS_ROOT / novel / character_slug


def discover_result_files(novel: str,
                          character_slug: str) -> list[tuple[str, str, Path]]:
    """Return [(mode, safe_model, path)] for every `{mode}__{model}.jsonl`
    role-play result file. The eval's own outputs (`eval_records.jsonl`,
    `eval_summary.json`) have no `__` / are not `.jsonl`, so they are skipped.
    """
    rdir = results_dir(novel, character_slug)
    if not rdir.is_dir():
        return []
    out: list[tuple[str, str, Path]] = []
    for p in sorted(rdir.glob("*.jsonl")):
        stem = p.stem                          # e.g. "arc__Qwen_Qwen3-8B"
        if "__" not in stem:                   # eval_records.jsonl etc.
            continue
        mode, safe_model = stem.split("__", 1)
        out.append((mode, safe_model, p))
    return out


EMPTY_RESPONSE_SENTINEL = "[empty response — model returned nothing]"

_THINK_BLOCK_RE = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_think(text):
    """Drop a leading <think>…</think> block before the judge sees the reply.

    Qwen3-family chat templates emit the block even in non-thinking mode, and a
    checkpoint fine-tuned on those targets keeps emitting it. The judge scores
    the visible in-character reply, so the block never reaches the prompt.
    """
    if not text:
        return text
    return _THINK_BLOCK_RE.sub("", str(text), count=1)




def _load_responses(path: Path) -> tuple[Optional[str],
                                         dict[tuple[str, int], str]]:
    """Read one result JSONL → (real_model_name, {(probe_id, phase_idx): text}).

    Dry-run placeholders and rows with an `error` are skipped (neither is model
    behavior). Rows where the model returned an empty `response` are kept and
    surfaced to the judge with a sentinel string — silence/refusal is part of
    the model's output and should be scored (low), not discarded.
    """
    model: Optional[str] = None
    out: dict[tuple[str, int], str] = {}
    for line in path.read_bytes().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(row, dict) or is_dry_run_record(row):
            continue
        model = row.get("model", model)
        if row.get("error"):
            continue
        probe_id, phase_idx = row.get("probe_id"), row.get("phase_idx")
        if probe_id is None or phase_idx is None:
            continue
        resp = _strip_think(row.get("response"))
        out[(probe_id, phase_idx)] = (resp if resp and str(resp).strip()
                                      else EMPTY_RESPONSE_SENTINEL)
    return model, out


def build_eval_units(novel: str, character_slug: str,
                     variant: str = "main") -> list[EvalUnit]:
    """Join ground truth with every model's per-mode responses.

    One `EvalUnit` per (trial, model). A trial is included for a model only if
    that model produced at least one response for it and the probe file has
    ground truth for it.
    """
    gt = load_ground_truth(novel, character_slug, variant)

    # {safe_model: {mode: {(probe_id, phase_idx): response_text}}}
    per_model: dict[str, dict[str, dict[tuple[str, int], str]]] = {}
    real_name: dict[str, str] = {}
    for mode, safe_model, path in discover_result_files(novel, character_slug):
        model, responses = _load_responses(path)
        per_model.setdefault(safe_model, {})[mode] = responses
        if model:
            real_name[safe_model] = model

    units: list[EvalUnit] = []
    for safe_model, mode_map in per_model.items():
        model = real_name.get(safe_model, safe_model)
        keys: set[tuple[str, int]] = set()
        for responses in mode_map.values():
            keys |= set(responses)
        for key in sorted(keys):
            if key not in gt:                  # no ground truth -> cannot score
                continue
            responses = {mode: rs[key] for mode, rs in mode_map.items()
                         if key in rs}
            if not responses:
                continue
            units.append(EvalUnit(
                novel=novel, character=character_slug,
                probe_id=key[0], phase_idx=key[1],
                model=model, trial=gt[key], responses=responses,
            ))
    return units
