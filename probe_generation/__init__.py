"""ArcANE probe generation pipeline.

Theoretical basis:
  1. The arc is a McAdams (1995) narrative-identity unit, not a numeric trait scale.
  2. Phases are Fleeson (2001) density distributions of states; adjacent overlap is expected.
  3. Personality is realized in person × situation (Mischel & Shoda 1995, CAPS).

Probe types (all three retained):
  ① In-Scenario — source-grounded role-play (verbatim source anchor)
  ② In-World    — within-distribution generalization (plausible-unwritten scene)
  ③ Out-of-World — cross-context narrative-identity transfer (non-source era,
                   life-stage locked)

Each probe = ONE (Scenario, Question) anchored at a specific chapter, plus
phase_responses[] — one response per trajectory phase. Each phase response
includes its knowledge cutoff (query_chapter) and four fields:
  gt_action + (optional) gt_speech + gt_thought + gt_typicality.

The thought field captures HOW this phase's character construes/processes
the situation — the cognitive dimension that lets adjacent phases
differentiate even when constrained scenes limit action variance.

Per-axis pipeline:
  S1         decision_variable + phase_contrasts
  S-Life     life-stage tag per phase (child/adolescent/young_adult/adult/older_adult)
  S-AxisRe   era-agnostic axis re-expression (③ scaffold; only if ③ requested)
  Gen-①      per anchor phase: locate key_moment → extract verbatim →
             multi-phase probe (anchor response grounded in the source passage)
  Gen-②      per anchor phase: plausible-unwritten in source world →
             multi-phase probe
  Gen-③      per anchor phase: deterministic-era-from-menu, scenario is
             age-agnostic so each phase response embodies its own life-stage
  Per-phase validators (parallel, per phase response):
    Q-Voice      — anachronism / tone / knowledge-cutoff respect
                   (retry 1× → mark unavailable on persistent fail; probe
                   stays — just that phase response is flagged)
    Q-PhaseFit   — categorical, BLIND — which phase is this response most
                   diagnostic of? (retry 1× on off_phase; still failing →
                   unavailable + flag; "adjacent" → auto-mark plausible_tail)
  Per-probe validators (parallel):
    Q-Anchor (①)  — anchor-phase response faithful to source passage  (drop)
    Q-World  (②③) — world/era/life-stage rules                        (drop)
    Q-Discrim    — within-probe adjacent-pair separation
                   (ANNOTATE ONLY — never drops. Adjacent overlap is
                   theoretically expected under Fleeson 2001.)
                   Flagged pairs roll up to family.weak_pairs for
                   downstream influence reporting.

Entry point:
    uv run python -m probe_generation --novel X --character Y
    uv run python -m probe_generation --novel X --all-characters

See cli.py for the full argument surface.
"""

__version__ = "1.0"
