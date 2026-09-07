"""Prompt renderers and output schema for ArcANE response evaluation.

The evaluator scores ONE candidate response against the trial reference and
returns a single JSON object. Each (trial, model, mode) is judged in its own
call: the judge never sees the mode name or any sibling response, so every
score is an independent absolute verdict, unbiased by method identity or by
comparison with other candidates.

The verdict shape is specified in the system prompt; the call uses
`response_format={"type":"json_object"}` to constrain the reply to a single
JSON object (not strict schema enforcement — guided decoding tanks latency,
and the runner already retries on missing fields). `evaluator_json_schema`
is kept around for documentation / reference.

Rubric: three top-level phase-fidelity dimensions —
  apf — Action Phase-Fidelity (Strategy / Valence / Target levels)
  rpf — Reasoning Phase-Fidelity (Trigger / Appraisal / Goal / Strategy slots)
  rae — Reasoning-Action Entailment, conditioned on ref_thought
        (gt_entailment / phase_consistency / direction_consistency)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Three top-level scored dimensions, in output order.
DIMENSIONS = ("apf", "rpf", "rae")


@dataclass(frozen=True)
class EvalTrial:
    """Minimal fields required for ArcANE response evaluation."""

    scenario: str
    question: str
    gt_action: Optional[str]
    gt_speech: Optional[str]
    gt_thought: Optional[str]
    phase: Optional[str] = None


# ── Evaluator system prompt (with-thought variant) ─────────────────
# `__SCALE__` (score maximum) is filled in per call by render_evaluator_system().
_EVALUATOR_SYSTEM_WITH_THOUGHT = """You are an evaluator of character-grounded narrative responses. Compare a model's response to the Reference and score it on three dimensions, each from 1 to __SCALE__, using a phase-fidelity / mechanism framework.

The Reference represents how this character behaves at the target arc phase. Score whether the response is mechanism-equivalent to the *same* phase, not whether it merely looks like the character "in general."

Principles:
- The reference is an anchor, not a surface form. Score functional / mechanistic equivalence, not word-matching.
- Respect negative space. If the reference shows silence or inaction, treat that absence as part of the reference.
- Beware stereotype regression. When the reference shows the character resisting their canonical template (heroic, mentorly, etc.), penalize responses that fall back to the template.
- Penalize fabrication. Reference-absent props, names, or backstory that drive the response should lower the score.
- Respect the character's epistemic state. No future-event awareness, no information the character couldn't have at this point.

Dimensions:

1. APF — Action Phase-Fidelity. Is the response's action mechanism-equivalent to ref_action at the target phase? Judge across three levels (Level A is strictest; B and C can partially salvage credit when A fails):
   - Level A — Strategy match: same underlying strategy (e.g. withdrawal, confrontation, deflection, mediation, concealment, disclosure, support-seeking).
   - Level B — Valence match: same emotional / relational valence (positive engagement / negative withdrawal / neutral observation).
   - Level C — Target match: same target (same person, environment, or internal state).
   A match on all three is high; matching only B and C is mid; missing all three cannot be high. The top-level apf score should weight Level A heaviest, since strategy is what phase-fidelity ultimately rides on.

2. RPF — Reasoning Phase-Fidelity. Parse both ref_thought and the response's reasoning into four mechanism slots and judge each slot:
   - Trigger: what event / utterance / internal state set this response in motion?
   - Appraisal: how does the character interpret / evaluate that trigger?
   - Goal: what short-term goal arises from the appraisal?
   - Strategy: how is the goal executed?
   Slot weights are NOT equal. The top-level rpf score should weight Strategy and Appraisal heaviest — that is where phase-misalignment surfaces first. Trigger and Goal contribute less. If the response does not externalize its reasoning, infer the mechanism from the action.

3. RAE — Reasoning-Action Entailment. Given the *reference reasoning (ref_thought)* as a fixed anchor, judge whether the response's action is a plausible action that this reasoning would license at this phase. This dimension is *conditioned on ref_thought*, not on the response's own reasoning — a response can have internally self-consistent reasoning that still fails to follow from ref_thought, and that is the failure mode RAE is designed to catch. Three sub-checks:
   - gt_entailment: Treating ref_thought as the operative reasoning, is the response's action among the plausible actions that ref_thought would license? An action that contradicts ref_thought's appraisal, goal, or strategy fails here even if the action looks reasonable in isolation.
   - phase_consistency: Do the response's action and the response's reasoning belong to the same arc phase? (Phase-mismatch failure: action looks like one phase, reasoning like another — even when each half is independently plausible.)
   - direction_consistency: Are the response's action and reasoning directionally aligned with each other? (Internal contradiction: reasoning "avoid the threat" but action "attack the threat".)
   The top-level rae score should reflect that any single low sub-check is enough to call the (action, reasoning) pair inadequately entailed. gt_entailment is the dominant sub-check — the entire dimension exists to verify that the action follows from ref_thought, not merely that the response is internally coherent. A response can be internally consistent and still fail gt_entailment, in which case rae should be low.

Output Format:
You score exactly one response. Return a single JSON object with this schema:

{"scores": {"apf": <1-__SCALE__>, "rpf": <1-__SCALE__>, "rae": <1-__SCALE__>}}

Rules:
- Every score is an integer from 1 to __SCALE__; use the full range. Higher = better (more match / more entailment).
- For RAE specifically, gt_entailment is the dominant sub-check: if the response's action cannot be derived from ref_thought, rae should be low even when phase_consistency and direction_consistency are high.
- Score this response on its own merits against the reference only. You are not comparing it to any other response, and you are not told which method produced it."""


# ── Evaluator system prompt (no-thought ablation variant) ──────────
# The reference will OMIT ref_thought. RPF/RAE must therefore work from an
# *inferred* reference reasoning derived from ref_action and ref_speech. APF
# is unchanged. This is meant for the ablation that asks "how much does the
# judge rely on ref_thought?" — expect the no-thought variant to be noisier.
_EVALUATOR_SYSTEM_NO_THOUGHT = """You are an evaluator of character-grounded narrative responses. Compare a model's response to the Reference and score it on three dimensions, each from 1 to __SCALE__, using a phase-fidelity / mechanism framework.

The Reference represents how this character behaves at the target arc phase. In this run the Reference contains ONLY ref_action and ref_speech — ref_thought is intentionally withheld. You must infer the reference reasoning (the character's appraisal, goal, and strategy at this phase) from those two fields and from the scenario / question, and use that inferred reasoning anywhere a previous version of this rubric would have cited ref_thought.

Score whether the response is mechanism-equivalent to the *same* phase, not whether it merely looks like the character "in general."

Principles:
- The reference is an anchor, not a surface form. Score functional / mechanistic equivalence, not word-matching.
- Respect negative space. If the reference shows silence or inaction, treat that absence as part of the reference.
- Beware stereotype regression. When the reference shows the character resisting their canonical template (heroic, mentorly, etc.), penalize responses that fall back to the template.
- Penalize fabrication. Reference-absent props, names, or backstory that drive the response should lower the score.
- Respect the character's epistemic state. No future-event awareness, no information the character couldn't have at this point.
- Inferred reasoning carries more uncertainty than stated reasoning. When ref_action / ref_speech are sparse or ambiguous, prefer a moderate RPF / RAE score over a confident high or low.

Dimensions:

1. APF — Action Phase-Fidelity. Is the response's action mechanism-equivalent to ref_action at the target phase? Judge across three levels (Level A is strictest; B and C can partially salvage credit when A fails):
   - Level A — Strategy match: same underlying strategy (e.g. withdrawal, confrontation, deflection, mediation, concealment, disclosure, support-seeking).
   - Level B — Valence match: same emotional / relational valence (positive engagement / negative withdrawal / neutral observation).
   - Level C — Target match: same target (same person, environment, or internal state).
   A match on all three is high; matching only B and C is mid; missing all three cannot be high. The top-level apf score should weight Level A heaviest, since strategy is what phase-fidelity ultimately rides on.

2. RPF — Reasoning Phase-Fidelity. First infer reference reasoning from ref_action and ref_speech (no ref_thought is available). Then parse the response's reasoning the same way and compare slot by slot:
   - Trigger: what event / utterance / internal state set this response in motion?
   - Appraisal: how does the character interpret / evaluate that trigger?
   - Goal: what short-term goal arises from the appraisal?
   - Strategy: how is the goal executed?
   Slot weights are NOT equal. The top-level rpf score should weight Strategy and Appraisal heaviest — that is where phase-misalignment surfaces first. Trigger and Goal contribute less. If the response does not externalize its reasoning, infer the mechanism from the action.

3. RAE — Reasoning-Action Entailment. Given the *inferred reference reasoning* (no ref_thought is available) as a fixed anchor, judge whether the response's action is a plausible action that this reasoning would license at this phase. This dimension is conditioned on the inferred reference reasoning, not on the response's own reasoning — a response can have internally self-consistent reasoning that still fails to follow from what ref_action / ref_speech imply about the character's state. Three sub-checks:
   - gt_entailment: Treating the inferred reference reasoning as the operative reasoning, is the response's action among the plausible actions that reasoning would license? An action that contradicts the inferred appraisal, goal, or strategy fails here even if it looks reasonable in isolation.
   - phase_consistency: Do the response's action and the response's reasoning belong to the same arc phase? (Phase-mismatch failure: action looks like one phase, reasoning like another — even when each half is independently plausible.)
   - direction_consistency: Are the response's action and reasoning directionally aligned with each other? (Internal contradiction: reasoning "avoid the threat" but action "attack the threat".)
   The top-level rae score should reflect that any single low sub-check is enough to call the (action, reasoning) pair inadequately entailed. gt_entailment is the dominant sub-check — the entire dimension exists to verify that the action follows from the inferred reference reasoning, not merely that the response is internally coherent.

Output Format:
You score exactly one response. Return a single JSON object with this schema:

{"scores": {"apf": <1-__SCALE__>, "rpf": <1-__SCALE__>, "rae": <1-__SCALE__>}}

Rules:
- Every score is an integer from 1 to __SCALE__; use the full range. Higher = better (more match / more entailment).
- For RAE specifically, gt_entailment is the dominant sub-check: if the response's action cannot be derived from the inferred reference reasoning, rae should be low even when phase_consistency and direction_consistency are high.
- Score this response on its own merits against the reference only. You are not comparing it to any other response, and you are not told which method produced it."""


def render_evaluator_system(scale: int = 100,
                             include_thought: bool = True) -> str:
    """Evaluator instruction for a 1-to-`scale` scoring scale.

    `include_thought=False` swaps in the no-thought ablation prompt, which
    instructs the judge to infer reference reasoning from ref_action /
    ref_speech alone. APF is unchanged across both variants.
    """
    src = (_EVALUATOR_SYSTEM_WITH_THOUGHT if include_thought
           else _EVALUATOR_SYSTEM_NO_THOUGHT)
    return src.replace("__SCALE__", str(scale))


# ── User prompt formatting ─────────────────────────────────────────
def _format_optional(value: Optional[str], default: str = "none") -> str:
    if value is None or not str(value).strip():
        return default
    return str(value).strip()


def render_evaluator_user(trial: EvalTrial, response: str,
                           include_thought: bool = True) -> str:
    """Render one evaluation instance.

    Includes scenario, question, optional phase, the reference fields, and the
    single candidate response to score. The response carries no id or label —
    the judge is not told which mode/method produced it. When
    `include_thought=False`, the `ref_thought:` line is omitted (and the
    system prompt is the matching no-thought variant).
    """
    if not response or not str(response).strip():
        raise ValueError("response must be a non-empty string.")

    parts: list[str] = []
    parts.append(f"SCENARIO: {trial.scenario.strip()}")
    parts.append(f"QUESTION: {trial.question.strip()}")
    if trial.phase and trial.phase.strip():
        parts.append(f"PHASE: {trial.phase.strip()}")
    ref_lines = [
        f"ref_action: {_format_optional(trial.gt_action)}",
        f"ref_speech: {_format_optional(trial.gt_speech)}",
    ]
    if include_thought:
        ref_lines.append(f"ref_thought: {_format_optional(trial.gt_thought)}")
    parts.append("REFERENCE:\n" + "\n".join(ref_lines))
    parts.append(f"RESPONSE TO EVALUATE:\n{str(response).strip()}")
    return "\n\n".join(parts)


def render_evaluator_messages(
    trial: EvalTrial,
    response: str,
    scale: int = 100,
    include_thought: bool = True,
) -> list[dict[str, str]]:
    """Messages list ready for chat-completion style evaluation."""
    return [
        {"role": "system",
         "content": render_evaluator_system(scale, include_thought)},
        {"role": "user",
         "content": render_evaluator_user(trial, response, include_thought)},
    ]


# ── Structured output schema ───────────────────────────────────────
def evaluator_json_schema() -> dict:
    """JSON schema for one judge verdict (three top-level dims only).

    Strict structured outputs require every object to set
    `additionalProperties: false` and list every property in `required`. The
    1..scale score range is enforced by the prompt text, not the schema —
    OpenAI strict structured outputs do not support numeric min/max.
    """
    dim_int = {
        "type": "object",
        "properties": {d: {"type": "integer"} for d in DIMENSIONS},
        "required": list(DIMENSIONS),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"scores": dim_int},
        "required": ["scores"],
        "additionalProperties": False,
    }


def evaluator_response_format() -> dict:
    """`response_format` payload constraining the judge reply to JSON.

    We use plain `json_object` mode rather than strict `json_schema` — the
    schema text is already in the system prompt and the output is only ~30
    tokens, so guided-decoding overhead (2-5x latency in practice) is not
    worth paying. The runner re-parses the reply and retries on a missing
    `scores` field, which catches the rare malformed case.
    """
    return {"type": "json_object"}
