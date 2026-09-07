"""Trajectory-fidelity prompt and output schema for ArcANE evaluation.

Where per-response evaluation scores one (trial, model, mode) response, this
evaluator scores the *trajectory* of a model's responses across all valid
phases of a single probe — one judge call sees every phase together. The
verdict captures whether the model's sequence of responses moves along the
same arc as the reference sequence (per-phase anchoring, direction, shape).

The judge returns JSON dimension scores, which are validated by the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Three sub-checks scored 1..scale, in output order.
DIMENSIONS = ("ptf_alignment", "ptf_direction", "ptf_shape")


@dataclass(frozen=True)
class PhaseRef:
    """Reference + candidate response for one phase of one trajectory unit."""

    phase_idx: int
    phase_label: Optional[str]
    gt_action: Optional[str]
    gt_speech: Optional[str]
    gt_thought: Optional[str]
    response: str            # candidate response at this phase (one mode)


# ── Evaluator system prompt (with-thought variant) ─────────────────
# `__SCALE__` (score maximum) is filled in per call by render_evaluator_system().
_EVALUATOR_TRAJECTORY_SYSTEM_WITH_THOUGHT = """You are an evaluator of character-grounded narrative trajectories. Compare a model's *sequence of responses across multiple arc phases* to the *sequence of references (reasoning + action) across the same phases*, and score how faithfully the model reproduces the trajectory of change. Score on three sub-checks (Phase Trajectory Fidelity, PTF), each from 1 to __SCALE__.

Each Reference at phase i represents how this character behaves at that arc phase. Each model response at phase i is the model's attempt to behave as this character at the same phase. Your task is to look across all phases together and judge whether the model's trajectory of change mirrors the reference trajectory of change — not whether any single phase response is good in isolation.

Input format:
You will receive SCENARIO, QUESTION, and then N phase blocks in chronological order. Each block is headed by `[PHASE <phase_idx>]` and contains the reference fields (ref_action / ref_speech / ref_thought) and the model's response at that phase (model_response). Phase indices are taken from the source probe and may have gaps (an "unavailable" reference phase is omitted) — use the indices as given.

Principles:
- Look across phases, not within one. A response that is locally fine at phase i can still belong to the wrong phase in the trajectory; that is the failure mode this evaluation is designed to catch.
- Reward differentiation only when it is grounded. A model that produces different responses across phases scores well only if those differences track the reference's differences — random or stylistic variation does not count.
- Penalize phase collapse. A model that gives near-identical responses across phases has failed regardless of how well any single response matches its own phase.
- Penalize phase swapping. A response that better fits another phase's reference than its own is a trajectory error.
- Penalize fabrication and stereotype regression. Reference-absent props/backstory, or a fallback to the character's canonical template against a reference that resists it, should pull scores down.
- Respect the character's epistemic state at each phase — no future-event awareness, no later-phase information leaking back into earlier-phase responses.
- An empty / refused response at a phase ("[empty response — model returned nothing]" or similar) is part of the trajectory: treat it as a failed anchor at that phase and as a missing contribution to direction/shape.

Sub-checks:

1. ptf_alignment — Per-phase anchoring. For each phase i, is the model's response at phase i closer in mechanism to the reference at phase i than to references at other phases? A high score means most responses are correctly anchored to their own phase; a low score means responses systematically drift toward the wrong phase or collapse into one undifferentiated voice.

2. ptf_direction — Direction of change. Treating the reference sequence as defining a direction (e.g. fearful → confident, dependent → autonomous), does the model's sequence move along the same axis and in the same direction? A high score means the model's overall direction-of-travel matches the reference's; a low score means the model moves on a different axis, in the opposite direction, or shows no movement.

3. ptf_shape — Shape and pacing of change. Beyond direction, does the model reproduce where on the trajectory the largest shifts occur and how gradual or abrupt those shifts are? A high score means inflection points and pacing line up with the reference; a low score means the model's trajectory has a clearly different internal structure (e.g. linear ramp where reference shows a sharp turning point, or inflection at the wrong phase). When N=2 there is no inflection structure to compare — score shape on whether the magnitude of the single transition is similar to the reference's.

Output Format:
You score one (character, scenario) item across N phases. Return a single JSON object with this schema:

{"scores": {"ptf_alignment": <1-__SCALE__>, "ptf_direction": <1-__SCALE__>, "ptf_shape": <1-__SCALE__>}, "average": <float>}

Rules:
- Every score is an integer from 1 to __SCALE__; use the full range. Higher = better.
- Phase collapse — near-identical responses across phases — drives all three sub-checks down, not just ptf_alignment. A collapsed trajectory has no direction and no shape to evaluate.
- average is the mean of the three sub-check scores, rounded to 2 decimals.
- Score this trajectory on its own merits against the reference only. You are not comparing it to any other model or method, and you are not told which method produced these responses."""


# ── Evaluator system prompt (no-thought ablation variant) ──────────
# Phase blocks will OMIT ref_thought. The judge must infer the reference
# reasoning at each phase from ref_action / ref_speech (plus the scenario and
# the surrounding-phase context) and use that inferred reasoning anywhere a
# previous version of this rubric would have cited ref_thought. The three
# sub-checks themselves (alignment / direction / shape) are unchanged — only
# the basis for "mechanism" judgement shifts.
_EVALUATOR_TRAJECTORY_SYSTEM_NO_THOUGHT = """You are an evaluator of character-grounded narrative trajectories. Compare a model's *sequence of responses across multiple arc phases* to the *sequence of references (action + speech) across the same phases*, and score how faithfully the model reproduces the trajectory of change. Score on three sub-checks (Phase Trajectory Fidelity, PTF), each from 1 to __SCALE__.

In this run each Reference block contains ONLY ref_action and ref_speech — ref_thought is intentionally withheld. For every phase you must infer the reference reasoning (the character's appraisal, goal, and strategy at that phase) from ref_action / ref_speech together with the scenario, the question, and the surrounding phases, and use that inferred reasoning anywhere a previous version of this rubric would have cited ref_thought. Your task is still to look across all phases together and judge whether the model's trajectory of change mirrors the reference trajectory of change — not whether any single phase response is good in isolation.

Input format:
You will receive SCENARIO, QUESTION, and then N phase blocks in chronological order. Each block is headed by `[PHASE <phase_idx>]` and contains the reference fields (ref_action / ref_speech only — ref_thought is omitted) and the model's response at that phase (model_response). Phase indices are taken from the source probe and may have gaps (an "unavailable" reference phase is omitted) — use the indices as given.

Principles:
- Look across phases, not within one. A response that is locally fine at phase i can still belong to the wrong phase in the trajectory; that is the failure mode this evaluation is designed to catch.
- Reward differentiation only when it is grounded. A model that produces different responses across phases scores well only if those differences track the reference's differences — random or stylistic variation does not count.
- Penalize phase collapse. A model that gives near-identical responses across phases has failed regardless of how well any single response matches its own phase.
- Penalize phase swapping. A response that better fits another phase's reference than its own is a trajectory error.
- Penalize fabrication and stereotype regression. Reference-absent props/backstory, or a fallback to the character's canonical template against a reference that resists it, should pull scores down.
- Respect the character's epistemic state at each phase — no future-event awareness, no later-phase information leaking back into earlier-phase responses.
- An empty / refused response at a phase ("[empty response — model returned nothing]" or similar) is part of the trajectory: treat it as a failed anchor at that phase and as a missing contribution to direction/shape.
- Inferred reasoning carries more uncertainty than stated reasoning. When ref_action / ref_speech are sparse or ambiguous at a phase, prefer moderate scores over confident high or low ones.

Sub-checks:

1. ptf_alignment — Per-phase anchoring. For each phase i, is the model's response at phase i closer in mechanism to the reference at phase i (action + speech + the reasoning you inferred from them) than to references at other phases? A high score means most responses are correctly anchored to their own phase; a low score means responses systematically drift toward the wrong phase or collapse into one undifferentiated voice.

2. ptf_direction — Direction of change. Treating the reference sequence (using inferred reasoning at each step) as defining a direction (e.g. fearful → confident, dependent → autonomous), does the model's sequence move along the same axis and in the same direction? A high score means the model's overall direction-of-travel matches the reference's; a low score means the model moves on a different axis, in the opposite direction, or shows no movement.

3. ptf_shape — Shape and pacing of change. Beyond direction, does the model reproduce where on the trajectory the largest shifts occur and how gradual or abrupt those shifts are? A high score means inflection points and pacing line up with the reference; a low score means the model's trajectory has a clearly different internal structure (e.g. linear ramp where reference shows a sharp turning point, or inflection at the wrong phase). When N=2 there is no inflection structure to compare — score shape on whether the magnitude of the single transition is similar to the reference's.

Output Format:
You score one (character, scenario) item across N phases. Return a single JSON object with this schema:

{"scores": {"ptf_alignment": <1-__SCALE__>, "ptf_direction": <1-__SCALE__>, "ptf_shape": <1-__SCALE__>}, "average": <float>}

Rules:
- Every score is an integer from 1 to __SCALE__; use the full range. Higher = better.
- Phase collapse — near-identical responses across phases — drives all three sub-checks down, not just ptf_alignment. A collapsed trajectory has no direction and no shape to evaluate.
- average is the mean of the three sub-check scores, rounded to 2 decimals.
- Score this trajectory on its own merits against the reference only. You are not comparing it to any other model or method, and you are not told which method produced these responses."""


def render_evaluator_system(scale: int = 100,
                             include_thought: bool = True) -> str:
    """Trajectory evaluator instruction for a 1-to-`scale` scoring scale.

    `include_thought=False` swaps in the no-thought ablation prompt, which
    instructs the judge to infer reference reasoning at each phase from
    ref_action / ref_speech. The three sub-checks themselves are unchanged
    across both variants.
    """
    src = (_EVALUATOR_TRAJECTORY_SYSTEM_WITH_THOUGHT if include_thought
           else _EVALUATOR_TRAJECTORY_SYSTEM_NO_THOUGHT)
    return src.replace("__SCALE__", str(scale))


# ── User prompt formatting ─────────────────────────────────────────
def _format_optional(value: Optional[str], default: str = "none") -> str:
    if value is None or not str(value).strip():
        return default
    return str(value).strip()


def render_evaluator_user(scenario: str, question: str,
                          phases: list[PhaseRef],
                          include_thought: bool = True) -> str:
    """Render one trajectory evaluation instance.

    Phases must be ordered by `phase_idx`. Each block carries the reference
    fields and the model's response for one mode at that phase. The judge sees
    no mode name or model name. When `include_thought=False`, the
    `ref_thought:` line is omitted from every phase block (and the system
    prompt is the matching no-thought variant).
    """
    if len(phases) < 2:
        raise ValueError("trajectory eval requires at least 2 phases.")

    parts: list[str] = [
        f"SCENARIO: {(scenario or '').strip()}",
        f"QUESTION: {(question or '').strip()}",
        f"PHASES (N={len(phases)}, in chronological order):",
    ]
    for ph in phases:
        label = f" — {ph.phase_label}" if ph.phase_label else ""
        lines = [
            f"[PHASE {ph.phase_idx}]{label}",
            f"  ref_action: {_format_optional(ph.gt_action)}",
            f"  ref_speech: {_format_optional(ph.gt_speech)}",
        ]
        if include_thought:
            lines.append(f"  ref_thought: {_format_optional(ph.gt_thought)}")
        lines.append(f"  model_response: {str(ph.response).strip()}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def render_evaluator_messages(
    scenario: str,
    question: str,
    phases: list[PhaseRef],
    scale: int = 100,
    include_thought: bool = True,
) -> list[dict[str, str]]:
    """Messages list ready for chat-completion style evaluation."""
    return [
        {"role": "system",
         "content": render_evaluator_system(scale, include_thought)},
        {"role": "user",
         "content": render_evaluator_user(scenario, question, phases,
                                          include_thought)},
    ]


# ── Structured output schema ───────────────────────────────────────
def evaluator_json_schema() -> dict:
    """JSON schema for one trajectory verdict (three sub-checks + average).

    The 1..scale score range is enforced by the prompt text, not the schema.
    Schema is sent non-strict — strict mode compiles a constrained grammar on
    the provider side and noticeably slows generation; the runner already
    re-tries on empty/no-scores replies so a non-strict schema is enough.
    """
    dim_int = {
        "type": "object",
        "properties": {d: {"type": "integer"} for d in DIMENSIONS},
        "required": list(DIMENSIONS),
    }
    return {
        "type": "object",
        "properties": {
            "scores": dim_int,
            "average": {"type": "number"},
        },
        "required": ["scores", "average"],
    }


def evaluator_response_format() -> dict:
    """`response_format` payload that asks the judge reply to match the
    trajectory verdict schema (non-strict — see `evaluator_json_schema`)."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "persona_traj_verdict",
            "schema": evaluator_json_schema(),
        },
    }
