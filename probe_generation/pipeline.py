"""Multi-phase probe generation pipeline.

Per Character Arc:
  S1          decision_variable + phase_contrasts                    (analyst)
  S-Life      per-phase life-stage tags                              (analyst)
  S-AxisRe    era-agnostic axis re-expression (if ③ requested)       (analyst)

Per anchor phase × per probe_type:
  Generator → ProbeOutput (scenario + question + N phase_responses each
              with gt_action + gt_speech + gt_thought + gt_typicality).
              Each phase response carries its own knowledge cutoff
              (query_chapter — midpoint of phase.chapter_range, except
              the ① anchor phase which uses the grounded scene's chapter).

Per probe (parallel):
  for each phase_response (parallel):
    Q-Voice + Q-PhaseFit
    if voice fail OR phase_fit off_phase → retry once; still failing →
      mark `unavailable=true` on that phase response (probe is still kept).
  Probe-level (parallel):
    Q-Anchor (① only) / Q-World (②③ only)
    Q-Discrim (within probe; annotation only — never drops)

Per axis:
  Drop a probe ONLY if its Q-Anchor (①) or Q-World (②③) failed.
  Auto-mark gt_typicality="plausible_tail" for any phase_response whose
  Q-PhaseFit verdict is "adjacent".
  Roll up every kept probe's Q-Discrim weak_pairs into family.weak_pairs
  (probe_id + probe_type + anchor_phase_idx attached) so a later analysis
  pass can report family-level "indistinguishability influence".
"""

import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI

from ._api import parse_call
from .config import (
    DROP_TRIGGERS,
    ERA_DESCRIPTIONS,
    GEN_TEMPERATURE,
    MAX_PHASE_RETRY,
    MODEL_ANALYST, MODEL_DESIGNER,
    era_for_phase,
)
from .prompts import (
    SYSTEM_CONSTRUCT_ANALYST, SYSTEM_LIFE_STAGE, SYSTEM_AXIS_RE_EXP,
    SYSTEM_IN_TEXT_DESIGNER, SYSTEM_IN_WORLD_DESIGNER, SYSTEM_OOW_DESIGNER,
    arc_system_block, compose_system,
    step1_user, life_stage_user, axis_re_exp_user,
    in_text_user, in_world_user, out_of_world_user,
    regen_phase_user,
)
from .schemas import (
    Step1Output, LifeStageOutput, EraAgnosticAxis, ProbeOutput,
    PhaseResponseOut,
)
from .text_grounding import ground_scene
from .validators import (
    q_voice, q_phase_fit, q_anchor, q_world, q_discrim,
)

logger = logging.getLogger(__name__)


# ── Per-phase representative chapter ────────────────────────────────
def phase_midpoint_chapter(phase: dict) -> int:
    cr = phase.get("chapter_range") or [1, 1]
    return (cr[0] + cr[1]) // 2


def _phase_query_chapters(arc: dict, anchor_phase_idx: int,
                          anchor_chapter_override: Optional[int]) -> dict[int, int]:
    """Per-phase representative chapter (the phase's knowledge cutoff).

    Default: phase_midpoint_chapter(phase). For ① the anchor phase's
    query_chapter is replaced by the grounded scene's actual chapter.
    """
    out = {}
    for i, ph in enumerate(arc["trajectory"]):
        out[i] = phase_midpoint_chapter(ph)
    if anchor_chapter_override is not None:
        out[anchor_phase_idx] = anchor_chapter_override
    return out


# ── S-LifeStage ─────────────────────────────────────────────────────
async def derive_life_stages(client, sem, character, arc, novel) -> dict[int, dict]:
    n = len(arc["trajectory"])
    system = compose_system(
        SYSTEM_LIFE_STAGE,
        arc_system_block(character, arc, novel),
    )
    user = life_stage_user(arc)
    try:
        raw = await parse_call(client, sem, MODEL_ANALYST, system, user,
                               LifeStageOutput, GEN_TEMPERATURE)
    except Exception as e:
        logger.warning(f"  [{arc['axis_id']}/S-Life] failed: {e}; fallback 'adult'")
        return {k: {"life_stage": "adult", "approx_age": "?",
                    "rationale": "fallback (S-Life call failed)"}
                for k in range(n)}
    out: dict[int, dict] = {}
    for entry in raw.get("phases", []):
        k = entry.get("phase_idx")
        if isinstance(k, int) and 0 <= k < n:
            out[k] = {
                "life_stage": entry.get("life_stage", "adult"),
                "approx_age": entry.get("approx_age", "?"),
                "rationale":  entry.get("rationale", ""),
            }
    for k in range(n):
        out.setdefault(k, {"life_stage": "adult", "approx_age": "?",
                           "rationale": "missing from LLM output"})
    return out


# ── S-AxisRe ────────────────────────────────────────────────────────
async def derive_axis_re_expression(client, sem, character, arc, novel,
                                    decision_variable, phase_contrasts,
                                    life_stages) -> Optional[dict]:
    system = compose_system(
        SYSTEM_AXIS_RE_EXP,
        arc_system_block(character, arc, novel, decision_variable,
                         phase_contrasts, life_stages),
    )
    user = axis_re_exp_user(arc)
    try:
        return await parse_call(client, sem, MODEL_ANALYST, system, user,
                                EraAgnosticAxis, GEN_TEMPERATURE)
    except Exception as e:
        logger.warning(f"  [{arc['axis_id']}/S-AxisRe] failed: {e}")
        return None


# ── Phase-response normalization ───────────────────────────────────
def _normalise_phase_responses(raw_responses: list[dict], n_phases: int,
                               phase_query_chapters: dict[int, int],
                               arc: dict) -> list[dict]:
    by_idx = {}
    for r in raw_responses:
        idx = r.get("phase_idx")
        if isinstance(idx, int) and 0 <= idx < n_phases and idx not in by_idx:
            by_idx[idx] = r
    if len(by_idx) != n_phases:
        return []
    out = []
    for k in range(n_phases):
        r = by_idx[k]
        out.append({
            "phase_idx": k,
            "phase_label": arc["trajectory"][k]["phase"],
            "query_chapter": phase_query_chapters.get(k, 0),
            "gt_action": r.get("gt_action", ""),
            "gt_speech": r.get("gt_speech"),
            "gt_thought": r.get("gt_thought", ""),
            "gt_typicality": r.get("gt_typicality", "typical"),
        })
    return out


# ── Probe generators (multi-phase) ─────────────────────────────────
async def gen_in_text(client, sem, character, char_slug, arc, novel,
                      anchor_phase_idx, decision_variable, phase_contrasts,
                      life_stages) -> Optional[dict]:
    phase = arc["trajectory"][anchor_phase_idx]
    aid = arc["axis_id"]
    key_moments = phase.get("key_moments", [])
    if not key_moments:
        logger.info(f"  [{aid}/a{anchor_phase_idx}/①] no key_moments; skip")
        return None
    scene = None
    for km in key_moments:
        scene = await ground_scene(client, sem, novel, char_slug, km,
                                   phase["chapter_range"])
        if scene:
            scene["key_moment"] = km
            break
    if not scene:
        logger.info(f"  [{aid}/a{anchor_phase_idx}/①] grounding failed; skip")
        return None
    pqc = _phase_query_chapters(arc, anchor_phase_idx, scene["chapter"])

    system = compose_system(
        SYSTEM_IN_TEXT_DESIGNER,
        arc_system_block(character, arc, novel, decision_variable,
                         phase_contrasts, life_stages),
    )
    user = in_text_user(character, arc, anchor_phase_idx, scene["chapter"],
                        scene["verbatim_passage"], pqc, life_stages)
    try:
        raw = await parse_call(client, sem, MODEL_DESIGNER, system, user,
                               ProbeOutput, GEN_TEMPERATURE)
    except Exception as e:
        logger.error(f"  [{aid}/a{anchor_phase_idx}/①] designer failed: {e}")
        return None

    n = len(arc["trajectory"])
    prs = _normalise_phase_responses(
        raw.get("phase_responses", []), n, pqc, arc,
    )
    if not prs:
        logger.error(f"  [{aid}/a{anchor_phase_idx}/①] incomplete phase_responses")
        return None
    return {
        "probe_type": "in_text",
        "anchor_phase_idx": anchor_phase_idx,
        "anchor_phase_label": phase["phase"],
        "anchor_query_chapter": scene["chapter"],
        "era_label": None,
        "scenario": raw["scenario"],
        "question": raw["question"],
        "phase_responses": prs,
        "anchor_source": {
            "chapter": scene["chapter"],
            "event_id": scene.get("event_id"),
            "event_description": scene.get("event_description", ""),
            "first_words": scene["first_words"],
            "last_words":  scene["last_words"],
            "key_moment":  scene["key_moment"],
            "verbatim_passage": scene["verbatim_passage"],
        },
    }


async def gen_in_world(client, sem, character, arc, novel,
                       anchor_phase_idx, decision_variable, phase_contrasts,
                       life_stages) -> Optional[dict]:
    phase = arc["trajectory"][anchor_phase_idx]
    aid = arc["axis_id"]
    anchor_query_chapter = phase_midpoint_chapter(phase)
    pqc = _phase_query_chapters(arc, anchor_phase_idx, None)
    system = compose_system(
        SYSTEM_IN_WORLD_DESIGNER,
        arc_system_block(character, arc, novel, decision_variable,
                         phase_contrasts, life_stages),
    )
    user = in_world_user(character, arc, anchor_phase_idx,
                         anchor_query_chapter, pqc, life_stages)
    try:
        raw = await parse_call(client, sem, MODEL_DESIGNER, system, user,
                               ProbeOutput, GEN_TEMPERATURE)
    except Exception as e:
        logger.error(f"  [{aid}/a{anchor_phase_idx}/②] designer failed: {e}")
        return None
    n = len(arc["trajectory"])
    prs = _normalise_phase_responses(
        raw.get("phase_responses", []), n, pqc, arc,
    )
    if not prs:
        logger.error(f"  [{aid}/a{anchor_phase_idx}/②] incomplete phase_responses")
        return None
    return {
        "probe_type": "in_world",
        "anchor_phase_idx": anchor_phase_idx,
        "anchor_phase_label": phase["phase"],
        "anchor_query_chapter": anchor_query_chapter,
        "era_label": None,
        "scenario": raw["scenario"],
        "question": raw["question"],
        "phase_responses": prs,
        "anchor_source": None,
    }


async def gen_out_of_world(client, sem, character, arc, novel,
                           anchor_phase_idx, decision_variable, phase_contrasts,
                           life_stages, axis_re_exp: dict) -> Optional[dict]:
    phase = arc["trajectory"][anchor_phase_idx]
    aid = arc["axis_id"]
    n = len(arc["trajectory"])
    abstract_phases = axis_re_exp.get("abstract_phases", [])
    if len(abstract_phases) != n:
        logger.error(f"  [{aid}/a{anchor_phase_idx}/③] axis_re_exp phase count mismatch")
        return None
    era_label = era_for_phase(aid, anchor_phase_idx)
    era_desc = ERA_DESCRIPTIONS.get(era_label, "")
    anchor_query_chapter = phase_midpoint_chapter(phase)
    pqc = _phase_query_chapters(arc, anchor_phase_idx, None)
    system = compose_system(
        SYSTEM_OOW_DESIGNER,
        arc_system_block(character, arc, novel, decision_variable,
                         phase_contrasts, life_stages),
    )
    user = out_of_world_user(character, arc, anchor_phase_idx,
                             anchor_query_chapter,
                             axis_re_exp.get("abstract_axis", ""),
                             abstract_phases,
                             era_label, era_desc,
                             pqc, life_stages)
    try:
        raw = await parse_call(client, sem, MODEL_DESIGNER, system, user,
                               ProbeOutput, GEN_TEMPERATURE)
    except Exception as e:
        logger.error(f"  [{aid}/a{anchor_phase_idx}/③] designer failed: {e}")
        return None
    prs = _normalise_phase_responses(
        raw.get("phase_responses", []), n, pqc, arc,
    )
    if not prs:
        logger.error(f"  [{aid}/a{anchor_phase_idx}/③] incomplete phase_responses")
        return None
    return {
        "probe_type": "out_of_world",
        "anchor_phase_idx": anchor_phase_idx,
        "anchor_phase_label": phase["phase"],
        "anchor_query_chapter": anchor_query_chapter,
        "era_label": era_label,
        "era_description": era_desc,
        "abstract_axis": axis_re_exp.get("abstract_axis", ""),
        "abstract_phases": abstract_phases,
        "scenario": raw["scenario"],
        "question": raw["question"],
        "phase_responses": prs,
        "anchor_source": None,
    }


# ── Per-phase regen (single phase response) ────────────────────────
async def _regen_one_phase(client, sem, character, arc, novel,
                           decision_variable, phase_contrasts, life_stages,
                           probe, phase_idx, feedback) -> Optional[dict]:
    pt = probe["probe_type"]
    if pt == "in_text":
        system_role = SYSTEM_IN_TEXT_DESIGNER
    elif pt == "in_world":
        system_role = SYSTEM_IN_WORLD_DESIGNER
    else:
        system_role = SYSTEM_OOW_DESIGNER
    orig_pr = next(p for p in probe["phase_responses"]
                   if p["phase_idx"] == phase_idx)
    qc = orig_pr.get("query_chapter", 0)
    system = compose_system(
        system_role,
        arc_system_block(character, arc, novel, decision_variable,
                         phase_contrasts, life_stages),
    )
    user = regen_phase_user(character, arc, phase_idx, qc,
                            probe["scenario"], probe["question"],
                            orig_pr, probe["phase_responses"], feedback)
    try:
        return await parse_call(client, sem, MODEL_DESIGNER, system, user,
                                PhaseResponseOut, GEN_TEMPERATURE)
    except Exception as e:
        logger.warning(f"  regen phase {phase_idx} failed: {e}")
        return None


# ── Per-probe validation + regen ────────────────────────────────────
async def _validate_phase_responses(client, sem, character, arc, novel,
                                    decision_variable, phase_contrasts,
                                    life_stages, probe) -> None:
    pt = probe["probe_type"]
    era_label = probe.get("era_label")

    tasks = []
    meta = []
    for pr in probe["phase_responses"]:
        k = pr["phase_idx"]
        life_stage = (life_stages.get(k) or {}).get("life_stage") if pt == "out_of_world" else None
        tasks.append(q_voice(
            client, sem, character, arc, novel,
            decision_variable, phase_contrasts, life_stages,
            k, pt, era_label, life_stage, pr["query_chapter"],
            probe["scenario"], pr,
        ))
        tasks.append(q_phase_fit(
            client, sem, character, arc, novel,
            decision_variable, phase_contrasts, life_stages,
            k, probe["scenario"], probe["question"], pr,
        ))
        meta.append((k, "voice"))
        meta.append((k, "phase_fit"))
    results = await asyncio.gather(*tasks, return_exceptions=False)
    per_phase = {}
    for (k, kind), r in zip(meta, results):
        per_phase.setdefault(k, {})[kind] = r
    for pr in probe["phase_responses"]:
        pr["quality"] = per_phase.get(pr["phase_idx"], {})
        pr.setdefault("retry_count", 0)
        pr.setdefault("unavailable", False)


async def _retry_failing_phases(client, sem, character, arc, novel,
                                decision_variable, phase_contrasts,
                                life_stages, probe) -> None:
    failing: list[int] = []
    for pr in probe["phase_responses"]:
        q = pr.get("quality") or {}
        voice_v = (q.get("voice") or {}).get("verdict")
        fit_v = (q.get("phase_fit") or {}).get("verdict")
        if voice_v == "fail" or fit_v == "off_phase":
            failing.append(pr["phase_idx"])
    if not failing:
        return

    feedbacks = {}
    for k in failing:
        pr = next(p for p in probe["phase_responses"] if p["phase_idx"] == k)
        q = pr.get("quality") or {}
        lines = []
        voice = q.get("voice") or {}
        if voice.get("verdict") == "fail":
            lines.append(f"VOICE: {voice.get('note','(no note)')}")
        fit = q.get("phase_fit") or {}
        if fit.get("verdict") == "off_phase":
            lines.append(
                f"PHASE-FIT (off): read as phase "
                f"{fit.get('most_likely_phase_idx')}, target {k}. "
                f"Note: {fit.get('note','(no note)')}"
            )
        feedbacks[k] = "\n".join(lines) or "(generic regen)"

    regens = await asyncio.gather(
        *[_regen_one_phase(client, sem, character, arc, novel,
                           decision_variable, phase_contrasts, life_stages,
                           probe, k, feedbacks[k])
          for k in failing],
        return_exceptions=True,
    )

    revalidate = []
    for k, r in zip(failing, regens):
        pr = next(p for p in probe["phase_responses"] if p["phase_idx"] == k)
        pr["retry_count"] = 1
        if isinstance(r, Exception) or r is None:
            pr["unavailable"] = True
            pr["unavailable_reason"] = "regen call failed"
            continue
        pr["previous_attempt"] = {
            "gt_action": pr["gt_action"],
            "gt_speech": pr.get("gt_speech"),
            "gt_thought": pr.get("gt_thought", ""),
            "gt_typicality": pr.get("gt_typicality", "typical"),
            "quality": pr.get("quality", {}),
        }
        pr["gt_action"] = r["gt_action"]
        pr["gt_speech"] = r.get("gt_speech")
        pr["gt_thought"] = r.get("gt_thought", "")
        pr["gt_typicality"] = r.get("gt_typicality", "typical")
        revalidate.append(k)

    if not revalidate:
        return

    pt = probe["probe_type"]
    era_label = probe.get("era_label")
    reval_tasks = []
    for k in revalidate:
        pr = next(p for p in probe["phase_responses"] if p["phase_idx"] == k)
        life_stage = (life_stages.get(k) or {}).get("life_stage") if pt == "out_of_world" else None
        reval_tasks.append(asyncio.gather(
            q_voice(client, sem, character, arc, novel,
                    decision_variable, phase_contrasts, life_stages,
                    k, pt, era_label, life_stage, pr["query_chapter"],
                    probe["scenario"], pr),
            q_phase_fit(client, sem, character, arc, novel,
                        decision_variable, phase_contrasts, life_stages,
                        k, probe["scenario"], probe["question"], pr),
        ))
    reval_results = await asyncio.gather(*reval_tasks, return_exceptions=False)
    for k, (new_voice, new_fit) in zip(revalidate, reval_results):
        pr = next(p for p in probe["phase_responses"] if p["phase_idx"] == k)
        pr["quality"] = {"voice": new_voice, "phase_fit": new_fit}
        if (new_voice.get("verdict") == "fail" or
                new_fit.get("verdict") == "off_phase"):
            pr["unavailable"] = True
            pr["unavailable_reason"] = "validator still fails after retry"


async def _probe_level_validation(client, sem, character, arc, novel,
                                  decision_variable, phase_contrasts,
                                  life_stages, probe) -> None:
    pt = probe["probe_type"]
    era_label = probe.get("era_label")
    era_desc = probe.get("era_description")

    if pt == "in_text":
        anchor_idx = probe["anchor_phase_idx"]
        anchor_pr = next(p for p in probe["phase_responses"]
                         if p["phase_idx"] == anchor_idx)
        src_passage = (probe.get("anchor_source") or {}).get("verbatim_passage", "")
        whole_task = q_anchor(
            client, sem, character, arc, novel,
            decision_variable, phase_contrasts, life_stages,
            probe["scenario"], probe["question"],
            anchor_idx, anchor_pr, src_passage,
        )
        whole_label = "anchor"
    else:
        whole_task = q_world(
            client, sem, character, arc, novel,
            decision_variable, phase_contrasts, life_stages,
            probe["anchor_phase_idx"], probe["anchor_query_chapter"],
            pt, era_label, era_desc,
            probe["scenario"], probe["phase_responses"],
        )
        whole_label = "world"

    disc_task = q_discrim(
        client, sem, character, arc, novel,
        decision_variable, phase_contrasts, life_stages,
        probe["scenario"], probe["question"], probe["phase_responses"],
    )
    whole_r, disc_r = await asyncio.gather(whole_task, disc_task,
                                            return_exceptions=False)
    probe["probe_check"] = {
        whole_label: whole_r,
        "discrimination": disc_r,
    }


async def validate_probe(client, sem, character, arc, novel,
                         decision_variable, phase_contrasts, life_stages,
                         probe) -> None:
    # Round 1: per-phase Q-Voice + Q-PhaseFit
    await _validate_phase_responses(
        client, sem, character, arc, novel,
        decision_variable, phase_contrasts, life_stages, probe,
    )
    # Round 2: retry failing phases (one regen)
    if MAX_PHASE_RETRY > 0:
        await _retry_failing_phases(
            client, sem, character, arc, novel,
            decision_variable, phase_contrasts, life_stages, probe,
        )
    # Round 3: probe-level Q-Anchor / Q-World + within-probe Q-Discrim
    await _probe_level_validation(
        client, sem, character, arc, novel,
        decision_variable, phase_contrasts, life_stages, probe,
    )


# ── Drop policy + annotation post-processing ───────────────────────
def _should_drop(probe: dict) -> tuple[bool, list[str]]:
    pt = probe["probe_type"]
    triggers = DROP_TRIGGERS.get(pt, set())
    pc = probe.get("probe_check") or {}
    failed = []
    for key in ("anchor", "world"):
        if key in triggers:
            r = pc.get(key)
            if isinstance(r, dict) and r.get("verdict") == "fail":
                failed.append(key)
    return (bool(failed), failed)


def _post_process_probe(probe: dict) -> None:
    """Auto-mark plausible_tail for phase responses whose Q-PhaseFit is
    'adjacent'. Add annotation for ① non-anchor responses with phase_fit
    'off_phase' (counterfactual that didn't quite land — kept for analysis)."""
    pt = probe["probe_type"]
    annotations = []
    for pr in probe["phase_responses"]:
        q = pr.get("quality") or {}
        fit = q.get("phase_fit") or {}
        verdict = fit.get("verdict")
        if verdict == "adjacent":
            pr["gt_typicality"] = "plausible_tail"
            annotations.append(
                f"P{pr['phase_idx']}: auto-marked plausible_tail "
                f"(Q-PhaseFit placed response at P{fit.get('most_likely_phase_idx')})"
            )
        elif (verdict == "off_phase" and pt == "in_text"
              and pr["phase_idx"] != probe.get("anchor_phase_idx")):
            annotations.append(
                f"P{pr['phase_idx']}: ① non-anchor counterfactual "
                f"placed at P{fit.get('most_likely_phase_idx')} — kept "
                f"for analysis, not dropped"
            )
    if annotations:
        probe["annotations"] = annotations


def _collect_family_weak_pairs(kept: list[dict]) -> list[dict]:
    """Flatten every kept probe's Q-Discrim weak_pairs into a family-level list.

    Each entry carries enough probe identifiers (probe_id, probe_type,
    anchor_phase_idx) that downstream influence reporting can attribute
    indistinguishability back to its source probe without consulting the
    per-probe `probe_check.discrimination.weak_pairs` block.
    """
    out: list[dict] = []
    for p in kept:
        pc = p.get("probe_check") or {}
        disc = pc.get("discrimination") or {}
        wps = disc.get("weak_pairs") if isinstance(disc, dict) else None
        if not wps:
            continue
        for wp in wps:
            out.append({
                "probe_id":          p.get("probe_id"),
                "probe_type":        p.get("probe_type"),
                "anchor_phase_idx":  p.get("anchor_phase_idx"),
                "from_phase_idx":    wp.get("from_phase_idx"),
                "to_phase_idx":      wp.get("to_phase_idx"),
                "separation":        wp.get("separation"),
                "note":              wp.get("note", ""),
            })
    return out


# ── Per-arc orchestrator ────────────────────────────────────────────
async def generate_arc_family(
    client: AsyncOpenAI, sem: asyncio.Semaphore,
    character: str, char_slug: str, arc: dict, novel: str,
    probe_types: Optional[list[str]] = None,
) -> dict:
    if probe_types is None:
        probe_types = ["in_text", "in_world", "out_of_world"]

    aid = arc["axis_id"]
    n_phases = len(arc["trajectory"])
    logger.info(f"[{aid}] start: {arc['axis_name']} ({n_phases} phases, "
                f"types={probe_types})")

    # S1
    step1 = await parse_call(
        client, sem, MODEL_ANALYST,
        compose_system(SYSTEM_CONSTRUCT_ANALYST,
                       arc_system_block(character, arc, novel)),
        step1_user(n_phases=n_phases),
        Step1Output, GEN_TEMPERATURE,
    )
    decision_variable = step1.get("decision_variable", "")
    phase_contrasts = step1.get("phase_contrasts", [])
    logger.info(f"  [{aid}] dv='{decision_variable[:80]}…'")

    # S-LifeStage
    life_stages = await derive_life_stages(client, sem, character, arc, novel)
    logger.info(
        f"  [{aid}] life-stages: "
        f"{[(k, life_stages[k]['life_stage']) for k in sorted(life_stages)]}"
    )

    # S-AxisRe (only if ③ requested)
    axis_re_exp = None
    if "out_of_world" in probe_types:
        axis_re_exp = await derive_axis_re_expression(
            client, sem, character, arc, novel,
            decision_variable, phase_contrasts, life_stages,
        )

    # Generation (per anchor phase × per type, all multi-phase output)
    gen_tasks = []
    task_meta = []
    for anchor in range(n_phases):
        if "in_text" in probe_types:
            gen_tasks.append(gen_in_text(
                client, sem, character, char_slug, arc, novel, anchor,
                decision_variable, phase_contrasts, life_stages,
            ))
            task_meta.append((anchor, "in_text"))
        if "in_world" in probe_types:
            gen_tasks.append(gen_in_world(
                client, sem, character, arc, novel, anchor,
                decision_variable, phase_contrasts, life_stages,
            ))
            task_meta.append((anchor, "in_world"))
        if "out_of_world" in probe_types and axis_re_exp:
            gen_tasks.append(gen_out_of_world(
                client, sem, character, arc, novel, anchor,
                decision_variable, phase_contrasts, life_stages, axis_re_exp,
            ))
            task_meta.append((anchor, "out_of_world"))

    gen_results = await asyncio.gather(*gen_tasks, return_exceptions=True)
    probes: list[dict] = []
    for (anchor, pt), r in zip(task_meta, gen_results):
        if isinstance(r, Exception):
            logger.error(f"  [{aid}/a{anchor}/{pt}] exception: {r}")
            continue
        if r is None:
            continue
        probes.append(r)
    logger.info(f"  [{aid}] generated {len(probes)}/{len(gen_tasks)} probes")

    # Probe IDs
    pt_suffix = {"in_text": "intext", "in_world": "inworld",
                 "out_of_world": "outworld"}
    for p in probes:
        p["probe_id"] = (
            f"{char_slug}_{aid}_{pt_suffix[p['probe_type']]}_a{p['anchor_phase_idx']}"
        )

    # Per-probe validation (parallel)
    val_tasks = [
        validate_probe(client, sem, character, arc, novel,
                       decision_variable, phase_contrasts, life_stages, p)
        for p in probes
    ]
    val_results = await asyncio.gather(*val_tasks, return_exceptions=True)
    for p, r in zip(probes, val_results):
        if isinstance(r, Exception):
            logger.error(f"  validate exception for {p.get('probe_id','?')}: {r}")

    # Post-process: auto-mark plausible_tail; ① off_phase non-anchor annotations
    for p in probes:
        _post_process_probe(p)

    # Drop probes failing probe-level triggers (① anchor / ②③ world)
    kept: list[dict] = []
    dropped: list[dict] = []
    for p in probes:
        drop, failed_keys = _should_drop(p)
        if drop:
            dropped.append({
                "probe_id": p.get("probe_id"),
                "probe_type": p["probe_type"],
                "anchor_phase_idx": p["anchor_phase_idx"],
                "failed_verdicts": failed_keys,
            })
        else:
            kept.append(p)
    if dropped:
        logger.info(
            f"  [{aid}] dropped {len(dropped)} probe(s): "
            f"{[(d['probe_id'], d['failed_verdicts']) for d in dropped]}"
        )

    family_weak_pairs = _collect_family_weak_pairs(kept)

    by_type_counts = {pt: sum(1 for p in kept if p["probe_type"] == pt)
                      for pt in ("in_text", "in_world", "out_of_world")}
    unavail_count = sum(1 for p in kept for pr in p["phase_responses"]
                        if pr.get("unavailable"))
    logger.info(
        f"[{aid}] done: kept={by_type_counts}, dropped={len(dropped)}, "
        f"unavailable_phase_resp={unavail_count}, "
        f"weak_pairs={len(family_weak_pairs)}"
    )

    return {
        "axis_id": aid,
        "axis_name": arc["axis_name"],
        "dimension_label": arc.get("dimension_label", ""),
        "axis_type": arc.get("axis_type"),
        "target_character": arc.get("target_character"),
        "decision_variable": decision_variable,
        "phase_contrasts": phase_contrasts,
        "axis_re_expression": axis_re_exp,
        "life_stages": life_stages,
        "n_phases": n_phases,
        "probes": kept,
        "dropped_probes": dropped,
        "weak_pairs": family_weak_pairs,
    }
