"""Pydantic schemas.

A probe is ONE (Scenario, Question) anchored at a specific chapter, plus
phase_responses[] — one entry per trajectory phase. Each phase response
includes its own knowledge cutoff (query_chapter), what the character does
(gt_action), what they say (gt_speech), AND how they cognitively process
the situation (gt_thought) — the construal dimension that lets adjacent
phases differentiate even when constrained scenes limit action variance.

For In-Scenario probes, the anchor response is grounded in a verbatim source
passage; other phase responses project their respective phases onto the same
scenario. For In-World and Out-of-World probes, all phase responses are generated,
with the scenario built around the anchor phase but plausibly encountered by
any phase.
"""

from typing import Literal, Optional

from pydantic import BaseModel


# ── S1: decision variable extraction ────────────────────────────────
class PhaseContrast(BaseModel):
    from_phase_idx: int
    to_phase_idx: int
    contrast: str


class Step1Output(BaseModel):
    decision_variable: str
    phase_contrasts: list[PhaseContrast]


# ── S-LifeStage: per-phase life-stage tag ──────────────────────────
class PhaseLifeStage(BaseModel):
    phase_idx: int
    life_stage: Literal["child", "adolescent", "young_adult", "adult", "older_adult"]
    approx_age: str
    rationale: str


class LifeStageOutput(BaseModel):
    phases: list[PhaseLifeStage]


# ── S-AxisRe: era-agnostic axis re-expression (③ scaffold) ─────────
class EraAgnosticAxis(BaseModel):
    abstract_axis: str
    abstract_phases: list[str]


# ── Text grounding (① only) ─────────────────────────────────────────
class LocatorOutput(BaseModel):
    chapter: Optional[int]
    event_id: Optional[str]
    reasoning: str


class ExtractorOutput(BaseModel):
    verbatim_passage: Optional[str]
    first_words: str
    last_words: str
    note: Optional[str]


# ── Per-phase response inside a Probe (multi-phase) ─────────────────
class PhaseResponseOut(BaseModel):
    """One phase's response to the shared (scenario, question).

    Four content fields. `gt_thought` is the construal/processing field —
    1-2 sentences describing how THIS phase's character interprets, weighs,
    or ruminates on the situation. This is what lets adjacent phases
    differentiate even when the scene's action affordance is narrow
    (Mischel & Shoda 1995 CAPS: situational encoding is the first link in
    the trait chain; two phases can encode the same situation differently).
    """
    phase_idx: int
    gt_action: str                              # ≤ 30 words
    gt_speech: Optional[str]                    # ≤ 25 words or None
    gt_thought: str                             # 1–2 sentences (construal)
    gt_typicality: Literal["typical", "plausible_tail"]


# ── Generator output (multi-phase per probe) ───────────────────────
class ProbeOutput(BaseModel):
    """One probe = (scenario, question, phase_responses[N]).

    For ③ Out-of-World, the generator also receives a fixed era_label and
    the per-phase life-stage table; phase responses must each embody their
    own life-stage within the shared era.
    """
    scenario: str
    question: str
    phase_responses: list[PhaseResponseOut]


# ── Per-phase-response validator outputs ────────────────────────────
class VoiceValidatorOutput(BaseModel):
    """Q-Voice — does THIS phase's response sound like the character at
    this phase, in this setting? Folds in-character + anachronism."""
    verdict: Literal["pass", "fail"]
    note: str


class PhaseFitValidatorOutput(BaseModel):
    """Q-PhaseFit — categorical: which trajectory phase is THIS phase's
    response most diagnostic of? The validator is NOT told the target —
    the wrapper compares against target_phase_idx.

    Wrapper verdicts:
      pass     — most_likely_phase_idx == target_phase_idx
      adjacent — |most_likely - target| ≤ 1
      off_phase— |most_likely - target| ≥ 2
    """
    most_likely_phase_idx: int
    confidence: Literal["low", "medium", "high"]
    note: str


# ── Per-probe validator outputs ─────────────────────────────────────
class AnchorValidatorOutput(BaseModel):
    """Q-Anchor — ① only: scenario + anchor phase's response faithful
    to source passage."""
    verdict: Literal["pass", "fail"]
    note: str


class WorldValidatorOutput(BaseModel):
    """Q-World — ②③ only: world/era rules respected; no source-world
    leakage in ③; no canonical-scene reproduction in ②; life-stages
    consistent in ③ phase responses."""
    verdict: Literal["pass", "fail"]
    note: str


class DiscrimPairOutput(BaseModel):
    """Q-Discrim — within a probe, one adjacent phase pair.

    Annotation only — never drops a probe. Adjacent overlap is
    theoretically expected under Fleeson 2001.
    """
    from_phase_idx: int
    to_phase_idx: int
    separation: Literal["separated", "similar", "ambiguous"]
    note: str


class DiscrimOutput(BaseModel):
    """Q-Discrim — within a probe, ALL adjacent phase pairs."""
    weak_pairs: list[DiscrimPairOutput]
    note: str
