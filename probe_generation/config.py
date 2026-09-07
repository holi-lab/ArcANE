"""Probe generation configuration.

Per-stage models can be overridden via env vars:
  PROBE_MODEL_ANALYST       (S1, S-LifeStage, S-AxisRe — small structured outputs)
  PROBE_MODEL_LOCATOR       (① text-grounding: chapter locator)
  PROBE_MODEL_EXTRACTOR     (① text-grounding: verbatim passage extractor)
  PROBE_MODEL_DESIGNER      (① ② ③ probe designers + per-phase regen)
  PROBE_MODEL_VALIDATOR     (Q-Voice / Q-PhaseFit / Q-Anchor / Q-World / Q-Discrim)

A single PROBE_MODEL_DEFAULT can be used as a fallback for any unset stage.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARC_RESULTS  = PROJECT_ROOT / "results" / "arc_extraction"


AxisSource = Literal["auto", "final", "final_auto", "final_validated"]

_AXIS_SOURCE_FILES: dict[str, tuple[str, str]] = {
    # source_label -> (subdir, filename_suffix)
    "final":           ("final",           "_final_axes.json"),
    "final_auto":      ("final_auto",      "_auto_axes.json"),
    "final_validated": ("final_validated", "_validated_axes.json"),
}


def axes_path(novel: str, character_slug: str,
              source: AxisSource = "auto") -> tuple[Path, str]:
    """Resolve per-character axis file.

    `source` selects which stage to read:
      - "final"           : pre-validation axes (all candidates kept)
      - "final_auto"      : Phase-3 critic-filtered axes (valid_axis only)
      - "final_validated" : Phase-4 human-validated axes
      - "auto" (default)  : prefer final_validated, then existing probe source,
                            then final_auto

    Returns (path, source_label) where source_label is stamped into probe
    output so downstream consumers can tell which provenance the probes
    came from. With source="auto", source_label reflects which stage was
    actually found on disk.
    """
    if source == "auto":
        validated = ARC_RESULTS / novel / "final_validated" / f"{character_slug}_validated_axes.json"
        if validated.exists():
            return validated, "final_validated"

        # Preserve an existing probe source only when human-validated axes
        # are absent. Historical provenance must not override validation.
        shipped = ARC_RESULTS / novel / "probes" / f"{character_slug}_probes.json"
        if shipped.exists():
            try:
                metadata = json.loads(shipped.read_text(encoding="utf-8"))
                stamped = metadata.get("axis_source") if isinstance(metadata, dict) else None
            except (json.JSONDecodeError, OSError):
                stamped = None
            if isinstance(stamped, str) and stamped in _AXIS_SOURCE_FILES:
                subdir, suffix = _AXIS_SOURCE_FILES[stamped]
                path = ARC_RESULTS / novel / subdir / f"{character_slug}{suffix}"
                if path.exists():
                    return path, stamped
        return ARC_RESULTS / novel / "final_auto" / f"{character_slug}_auto_axes.json", "final_auto"
    if source not in _AXIS_SOURCE_FILES:
        raise ValueError(f"unknown axis source: {source!r}")
    subdir, suffix = _AXIS_SOURCE_FILES[source]
    return ARC_RESULTS / novel / subdir / f"{character_slug}{suffix}", source


def characters_path(novel: str) -> Path:
    return ARC_RESULTS / novel / "characters.json"


def events_path(novel: str, character_slug: str) -> Path:
    return ARC_RESULTS / novel / "outputs" / character_slug / "events_all.json"


def chapter_file(novel: str, ch: int) -> Path:
    return ARC_RESULTS / novel / "chapters" / f"chapter_{ch:02d}.txt"


def probes_out_dir(novel: str) -> Path:
    return ARC_RESULTS / novel / "probes"


# Schema version stamped into each output JSON. Bump on breaking changes.
PROBE_SCHEMA_VERSION = "1.0"


# ── API ────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

def require_openai_key() -> str:
    """Return the OpenAI key, or fail with an actionable message.

    The key is read lazily so that API-free stages -- notably
    `phase0_preprocess`, which only splits the novel into chapters -- import
    and run without one.
    """
    if not OPENAI_API_KEY:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it (or put it in .env) before "
            "running the stages that call the API. Chapter splitting "
            "(arc_construction.phase0_preprocess) needs no key."
        )
    return OPENAI_API_KEY


PROBE_MODEL_DEFAULT = os.environ.get("PROBE_MODEL_DEFAULT", "gpt-5.4")

MODEL_ANALYST   = os.environ.get("PROBE_MODEL_ANALYST",   PROBE_MODEL_DEFAULT)
MODEL_LOCATOR   = os.environ.get("PROBE_MODEL_LOCATOR",   PROBE_MODEL_DEFAULT)
MODEL_EXTRACTOR = os.environ.get("PROBE_MODEL_EXTRACTOR", PROBE_MODEL_DEFAULT)
MODEL_DESIGNER  = os.environ.get("PROBE_MODEL_DESIGNER",  PROBE_MODEL_DEFAULT)
MODEL_VALIDATOR = os.environ.get("PROBE_MODEL_VALIDATOR", PROBE_MODEL_DEFAULT)

# ── Retry / concurrency ────────────────────────────────────────────
MAX_RETRIES = 3              # API-level retry on transient errors (in _api.py)
RETRY_BACKOFF_BASE = 2
MAX_CONCURRENT_API_CALLS = 64

# ── Generation parameters ──────────────────────────────────────────
GEN_TEMPERATURE = 0.7
VAL_TEMPERATURE = 0.2

# ── Per-phase-response regen policy ────────────────────────────────
# When a phase response fails Q-Voice or Q-PhaseFit (off_phase), regenerate
# that single phase response once. Still failing → mark `unavailable=true`
# on that phase response (probe is still kept; the response is just flagged).
MAX_PHASE_RETRY = 1

# ── Type-aware probe-level drop policy ─────────────────────────────
# Probe-level verdicts that drop the WHOLE PROBE from the family output.
# Per-phase-response failures (Q-Voice / Q-PhaseFit) do NOT drop the probe;
# they only mark that phase's response unavailable.
# - Q-Discrim NEVER drops anywhere — adjacent-phase overlap is expected
#   (Fleeson 2001). Always annotation-only.
DROP_TRIGGERS: dict[str, set[str]] = {
    "in_text":      {"anchor"},
    "in_world":     {"world"},
    "out_of_world": {"world"},
}

# ── Text-grounding limits ──────────────────────────────────────────
MAX_CHAPTER_CHARS = 60_000
MAX_LOCATOR_CANDIDATES = 30

# ── Life-stage taxonomy ────────────────────────────────────────────
# Used in OoW transposition (Erikson 1968; McAdams & Olson 2010).
LifeStage = Literal["child", "adolescent", "young_adult", "adult", "older_adult"]
LIFE_STAGE_TAGS: list[str] = [
    "child", "adolescent", "young_adult", "adult", "older_adult",
]

# ── Out-of-World era menu ──────────────────────────────────────────
# Broad enough to generalize across source novels; specific enough that the
# generator commits to one setting. Pre-assigned deterministically per
# (axis_id, phase_idx) to vary across the family without sequential dependency.
ERA_MENU: list[str] = [
    "modern_urban",              # contemporary metropolis
    "industrial_early_20th",     # 1900–1940 industrial society
    "mid_century",               # 1950–1970
    "pre_industrial_agrarian",   # pre-1800 rural (NOT source's era — generator checks)
    "speculative_near_future",   # 50–100 years forward
    "pre_modern_imperial",       # ancient or medieval empire (NOT source's era)
    "diasporic_contemporary",    # modern, explicitly cross-cultural
]

ERA_DESCRIPTIONS: dict[str, str] = {
    "modern_urban":             "Contemporary city, ~2010s–2020s. Smartphones, public transit, office/school institutions.",
    "industrial_early_20th":    "Early 20th-century industrial society (~1900–1940). Factories, telegrams, urbanization, before mass media.",
    "mid_century":              "Mid-20th-century (~1950–1970). Suburbs, landlines, early television, postwar institutions.",
    "pre_industrial_agrarian":  "Rural pre-industrial society (pre-1800), NOT the source's era. Farming, hand tools, local kinship, no modern tech.",
    "speculative_near_future":  "50–100 years forward. Plausible tech extrapolation, no magic; recognizable institutions evolved.",
    "pre_modern_imperial":      "Ancient or medieval empire (NOT the source's era). Court politics, hierarchy, religious authority.",
    "diasporic_contemporary":   "Contemporary but explicitly cross-cultural — diaspora community, migrant family, refugee context.",
}


def era_for_phase(axis_id: str, phase_idx: int) -> str:
    """Deterministic era assignment per (axis_id, phase_idx).

    Cyclic walk through ERA_MENU starting from a hash-derived offset. This
    guarantees that for any n_phases ≤ len(ERA_MENU), every phase in the
    same axis gets a DISTINCT era — preventing the v1 modern-default collapse
    while avoiding adjacent-pair collisions that a pure hash assignment can
    produce on small N. Phase generators run in parallel because each looks
    up its own slot.
    """
    h = hashlib.sha1(axis_id.encode()).hexdigest()
    base = int(h[:8], 16) % len(ERA_MENU)
    return ERA_MENU[(base + phase_idx) % len(ERA_MENU)]
