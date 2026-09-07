"""Probe-quality validators.

Per phase response (parallel):
  Q-Voice      voice / anachronism / knowledge-cutoff respect
  Q-PhaseFit   categorical "most-likely phase" judgment (BLIND — target
               idx not shown in prompt)

Per probe (parallel, after phase-response validation):
  Q-Anchor   ① only: scenario + anchor-phase response vs source
  Q-World    ②③ only: world / era / life-stage rules
  Q-Discrim  within-probe: all adjacent phase pairs in one call
             (annotation only — NEVER drops)

Q-PhaseFit verdict logic (applied here, not by the LLM):
  pass     — most_likely_phase_idx == target_phase_idx
  adjacent — |most_likely - target| ≤ 1
  off_phase— |most_likely - target| ≥ 2
"""

import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI

from ._api import parse_call
from .config import MODEL_VALIDATOR, VAL_TEMPERATURE
from .prompts import (
    SYSTEM_VOICE_VALIDATOR, SYSTEM_PHASE_FIT_VALIDATOR,
    SYSTEM_ANCHOR_VALIDATOR, SYSTEM_WORLD_VALIDATOR,
    SYSTEM_DISCRIM_VALIDATOR,
    arc_system_block, compose_system,
    voice_validator_user, phase_fit_validator_user,
    anchor_validator_user, world_validator_user,
    discrim_validator_user,
)
from .schemas import (
    VoiceValidatorOutput, PhaseFitValidatorOutput,
    AnchorValidatorOutput, WorldValidatorOutput,
    DiscrimOutput,
)

logger = logging.getLogger(__name__)


# ── Q-Voice (per phase response) ────────────────────────────────────
async def q_voice(client: AsyncOpenAI, sem: asyncio.Semaphore,
                  character: str, arc: dict, novel: str,
                  decision_variable: str, phase_contrasts: list[dict],
                  life_stages: dict[int, dict],
                  phase_idx: int, probe_type: str,
                  era_label: Optional[str], life_stage: Optional[str],
                  query_chapter: int,
                  scenario: str, phase_response: dict) -> dict:
    system = compose_system(
        SYSTEM_VOICE_VALIDATOR,
        arc_system_block(character, arc, novel, decision_variable,
                         phase_contrasts, life_stages),
    )
    user = voice_validator_user(character, arc, phase_idx, probe_type,
                                era_label, life_stage, query_chapter,
                                scenario, phase_response)
    try:
        return await parse_call(client, sem, MODEL_VALIDATOR, system, user,
                                VoiceValidatorOutput, VAL_TEMPERATURE)
    except Exception as e:
        logger.warning(f"  Q-Voice threw: {e}")
        return {"verdict": "fail", "note": f"validator error: {e}"}


# ── Q-PhaseFit (per phase response, blind to target) ───────────────
def _phase_fit_verdict(most_likely: int, target: int, n_phases: int) -> str:
    if not (0 <= most_likely < n_phases):
        return "off_phase"
    if most_likely == target:
        return "pass"
    if abs(most_likely - target) == 1:
        return "adjacent"
    return "off_phase"


async def q_phase_fit(client: AsyncOpenAI, sem: asyncio.Semaphore,
                      character: str, arc: dict, novel: str,
                      decision_variable: str, phase_contrasts: list[dict],
                      life_stages: dict[int, dict],
                      target_phase_idx: int,
                      scenario: str, question: str,
                      phase_response: dict) -> dict:
    system = compose_system(
        SYSTEM_PHASE_FIT_VALIDATOR,
        arc_system_block(character, arc, novel, decision_variable,
                         phase_contrasts, life_stages),
    )
    user = phase_fit_validator_user(arc, target_phase_idx, decision_variable,
                                    scenario, question, phase_response)
    try:
        raw = await parse_call(client, sem, MODEL_VALIDATOR, system, user,
                               PhaseFitValidatorOutput, VAL_TEMPERATURE)
    except Exception as e:
        logger.warning(f"  Q-PhaseFit threw: {e}")
        return {"verdict": "off_phase",
                "most_likely_phase_idx": None,
                "target_phase_idx": target_phase_idx,
                "confidence": "low",
                "note": f"validator error: {e}"}
    most_likely = raw.get("most_likely_phase_idx")
    n = len(arc["trajectory"])
    verdict = _phase_fit_verdict(
        most_likely if isinstance(most_likely, int) else -1,
        target_phase_idx, n,
    )
    return {
        "verdict": verdict,
        "most_likely_phase_idx": most_likely,
        "target_phase_idx": target_phase_idx,
        "confidence": raw.get("confidence", "low"),
        "note": raw.get("note", ""),
    }


# ── Q-Anchor (① only, anchor phase response) ────────────────────────
async def q_anchor(client: AsyncOpenAI, sem: asyncio.Semaphore,
                   character: str, arc: dict, novel: str,
                   decision_variable: str, phase_contrasts: list[dict],
                   life_stages: dict[int, dict],
                   scenario: str, question: str,
                   anchor_phase_idx: int, anchor_response: dict,
                   source_passage: str) -> dict:
    system = compose_system(
        SYSTEM_ANCHOR_VALIDATOR,
        arc_system_block(character, arc, novel, decision_variable,
                         phase_contrasts, life_stages),
    )
    user = anchor_validator_user(scenario, question, anchor_phase_idx,
                                 anchor_response, source_passage)
    try:
        return await parse_call(client, sem, MODEL_VALIDATOR, system, user,
                                AnchorValidatorOutput, VAL_TEMPERATURE)
    except Exception as e:
        logger.warning(f"  Q-Anchor threw: {e}")
        return {"verdict": "fail", "note": f"validator error: {e}"}


# ── Q-World (②③ only, all phase responses considered together) ─────
async def q_world(client: AsyncOpenAI, sem: asyncio.Semaphore,
                  character: str, arc: dict, novel: str,
                  decision_variable: str, phase_contrasts: list[dict],
                  life_stages: dict[int, dict],
                  anchor_phase_idx: int, anchor_query_chapter: int,
                  probe_type: str,
                  era_label: Optional[str],
                  era_description: Optional[str],
                  scenario: str, phase_responses: list[dict]) -> dict:
    system = compose_system(
        SYSTEM_WORLD_VALIDATOR,
        arc_system_block(character, arc, novel, decision_variable,
                         phase_contrasts, life_stages),
    )
    user = world_validator_user(character, novel, arc, anchor_phase_idx,
                                anchor_query_chapter, probe_type,
                                era_label, era_description, life_stages,
                                scenario, phase_responses)
    try:
        return await parse_call(client, sem, MODEL_VALIDATOR, system, user,
                                WorldValidatorOutput, VAL_TEMPERATURE)
    except Exception as e:
        logger.warning(f"  Q-World threw: {e}")
        return {"verdict": "fail", "note": f"validator error: {e}"}


# ── Q-Discrim (within-probe, annotation only) ──────────────────────
async def q_discrim(client: AsyncOpenAI, sem: asyncio.Semaphore,
                    character: str, arc: dict, novel: str,
                    decision_variable: str, phase_contrasts: list[dict],
                    life_stages: dict[int, dict],
                    scenario: str, question: str,
                    phase_responses: list[dict]) -> dict:
    system = compose_system(
        SYSTEM_DISCRIM_VALIDATOR,
        arc_system_block(character, arc, novel, decision_variable,
                         phase_contrasts, life_stages),
    )
    user = discrim_validator_user(arc, decision_variable, scenario, question,
                                  phase_responses)
    try:
        return await parse_call(client, sem, MODEL_VALIDATOR, system, user,
                                DiscrimOutput, VAL_TEMPERATURE)
    except Exception as e:
        logger.warning(f"  Q-Discrim threw: {e}")
        return {"weak_pairs": [], "note": f"validator error: {e}"}
