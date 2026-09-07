"""Configuration for evaluation.trajectory (trajectory judge).

Shares all judge settings (model, backend, pricing, concurrency, score scale)
with `evaluation.per_response.config`. Trajectory-specific settings control
the minimum phase count and the cost-estimate output budget.

Trajectory-only knob:
  EXP_TRAJ_MIN_PHASES  — minimum N to include a probe (default: 2). Probes
                          with fewer valid phases are skipped.
"""

import os

# Re-export the shared judge settings and client factory.
from evaluation.per_response.config import *  # noqa: F401,F403


# Cost-estimate output budget per call. Verdict is just three scores +
# average, so the reply is small. Override via env if needed.
EST_OUTPUT_TOKENS_PER_CALL: int = int(
    os.environ.get("EXP_TRAJ_EST_OUTPUT_TOKENS", "60"))


# Minimum phases for a probe to be eligible for trajectory eval. A
# single-phase probe has no trajectory to compare, so we skip it. Two phases
# is enough for alignment/direction; shape is degenerate at N=2 and the
# prompt handles that explicitly.
TRAJ_MIN_PHASES: int = int(os.environ.get("EXP_TRAJ_MIN_PHASES", "2"))
