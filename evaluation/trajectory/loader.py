"""Gather trajectory evaluation inputs.

For trajectory evaluation the unit of judgment is one (probe, model): the judge sees the
model's responses across every valid phase of that probe at once. This
loader joins the probe file (ground truth across phases) with each model's
per-phase responses (one JSONL per mode), then groups them per (probe_id,
model). A TrajUnit then carries one ordered list of `PhaseRef`s *per mode* —
each mode is judged in its own call but the per-phase ordering is shared.

The probe file lives at `results/arc_extraction/{novel}/probes/{slug}_probes.json`;
model responses live at `results/inference/{novel}/{slug}/{mode}__{model}.jsonl`.
Both inputs are shared with per-response evaluation; only the assembly differs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from inference import config
from inference.records import is_dry_run_record
from evaluation.per_response.loader import (EMPTY_RESPONSE_SENTINEL,
                                           _strip_think,
                                           discover_result_files)
from .prompt import PhaseRef


@dataclass
class TrajUnit:
    """One probe scored for one model: the probe's scenario/question plus,
    per mode, the ordered list of (reference, response) per phase."""

    novel: str
    character: str
    probe_id: str
    model: str                                  # real model name
    scenario: str
    question: str
    phases_by_mode: dict[str, list[PhaseRef]]   # mode -> ordered PhaseRefs

    def n_phases(self, mode: str) -> int:
        return len(self.phases_by_mode.get(mode, []))


# ── probe file → per-phase reference fields ────────────────────────
@dataclass(frozen=True)
class _ProbeGT:
    scenario: str
    question: str
    phases: dict[int, dict]      # phase_idx -> {label, gt_action, gt_speech, gt_thought}


def load_probe_ground_truth(novel: str, character_slug: str,
                             variant: str = "main") -> dict[str, _ProbeGT]:
    """Read the probe file → {probe_id: _ProbeGT}.

    `unavailable` phases are dropped (no salvageable reference); their
    phase_idx simply does not appear in `_ProbeGT.phases`.
    """
    path = config.probe_file(novel, character_slug, variant)
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, _ProbeGT] = {}
    for fam in data.get("families", []):
        for probe in fam.get("probes", []):
            pid = probe["probe_id"]
            phases: dict[int, dict] = {}
            for ph in probe.get("phase_responses", []):
                if ph.get("unavailable"):
                    continue
                phases[ph["phase_idx"]] = {
                    "phase_label": ph.get("phase_label"),
                    "gt_action": ph.get("gt_action"),
                    "gt_speech": ph.get("gt_speech"),
                    "gt_thought": ph.get("gt_thought"),
                }
            out[pid] = _ProbeGT(
                scenario=probe.get("scenario", ""),
                question=probe.get("question", ""),
                phases=phases,
            )
    return out


# ── result files → per (probe_id, mode, model) → {phase_idx: response} ─
def _load_responses(path: Path) -> tuple[Optional[str],
                                          dict[tuple[str, int], str]]:
    """Read one role-play result JSONL → (model, {(probe_id, phase_idx): text}).

    Mirrors `evaluation.per_response.loader._load_responses`: dry-run and API-error
    rows are skipped (not model behavior), but empty model responses are
    surfaced with `EMPTY_RESPONSE_SENTINEL` so trajectory scoring counts
    silence as a failed anchor instead of dropping it.
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


# ── assembly ───────────────────────────────────────────────────────
def build_traj_units(novel: str, character_slug: str,
                     variant: str = "main",
                     min_phases: int = 2) -> list[TrajUnit]:
    """Build one TrajUnit per (probe_id, model).

    For each model, group all its mode results by (probe_id, mode); for each
    (probe, model) build a TrajUnit whose `phases_by_mode[mode]` is the
    ordered (by phase_idx) list of PhaseRefs that exist for both the probe
    file (reference present) and the role-play result (response present).

    A mode whose response list is shorter than `min_phases` is dropped from
    the unit. A unit with no surviving modes is dropped entirely.
    """
    probe_gt = load_probe_ground_truth(novel, character_slug, variant)

    # {safe_model: {mode: {(probe_id, phase_idx): response}}}
    per_model: dict[str, dict[str, dict[tuple[str, int], str]]] = {}
    real_name: dict[str, str] = {}
    for mode, safe_model, path in discover_result_files(novel, character_slug):
        model, responses = _load_responses(path)
        per_model.setdefault(safe_model, {})[mode] = responses
        if model:
            real_name[safe_model] = model

    units: list[TrajUnit] = []
    for safe_model, mode_map in per_model.items():
        model = real_name.get(safe_model, safe_model)
        # collect every (probe_id, phase_idx) touched by any mode of this model
        per_probe: dict[str, set[int]] = defaultdict(set)
        for responses in mode_map.values():
            for pid, ph_idx in responses:
                per_probe[pid].add(ph_idx)

        for pid, _phases_touched in per_probe.items():
            gt = probe_gt.get(pid)
            if gt is None:
                continue
            # Only phase indices that BOTH the probe file and at least one
            # mode have. Modes that don't cover a phase will see fewer
            # entries in their own PhaseRef list.
            phases_by_mode: dict[str, list[PhaseRef]] = {}
            for mode, responses in mode_map.items():
                refs: list[PhaseRef] = []
                for ph_idx in sorted(gt.phases):
                    if (pid, ph_idx) not in responses:
                        continue
                    meta = gt.phases[ph_idx]
                    refs.append(PhaseRef(
                        phase_idx=ph_idx,
                        phase_label=meta["phase_label"],
                        gt_action=meta["gt_action"],
                        gt_speech=meta["gt_speech"],
                        gt_thought=meta["gt_thought"],
                        response=responses[(pid, ph_idx)],
                    ))
                if len(refs) >= min_phases:
                    phases_by_mode[mode] = refs

            if not phases_by_mode:
                continue
            units.append(TrajUnit(
                novel=novel, character=character_slug,
                probe_id=pid, model=model,
                scenario=gt.scenario, question=gt.question,
                phases_by_mode=phases_by_mode,
            ))

    return units
