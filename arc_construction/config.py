"""Configuration for the character arc extraction pipeline.

Usage:
    NOVEL_NAME="Great Expectations" python -m arc_construction.run_all

    # Configure a novel before importing pipeline stages in Python.
    from arc_construction import config
    config.init("Great Expectations")
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Release root ──────────────────────────────────────────────
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# ── API settings ─────────────────────────────────────────────────
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

MODEL_NAME = os.environ.get("ARC_MODEL", "gpt-5.4-mini")

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds
MAX_CONCURRENT_API_CALLS = 6  # asyncio.Semaphore limit

# ── Novel / character settings ───────────────────────────────
NOVEL_NAME: str = os.environ.get("NOVEL_NAME", "The Underdogs")

N_EXTRACT_CHARACTERS: int = int(os.environ.get("N_CHARACTERS", "5"))

TARGET_CHARACTERS: list[str] = []   # populated by character_extractor at runtime
CHARACTER_DIRS: dict[str, str] = {} # populated by character_extractor at runtime

# ── Derived paths (all results live under results/arc_extraction/<novel>/) ──
NOVEL_PATH: str = os.path.join(PROJECT_ROOT, "data", "novels", f"{NOVEL_NAME}.txt")
RESULTS_BASE: str = os.path.join(PROJECT_ROOT, "results", "arc_extraction", NOVEL_NAME)
CHAPTERS_DIR: str = os.path.join(RESULTS_BASE, "chapters")
OUTPUTS_DIR: str = os.path.join(RESULTS_BASE, "outputs")
FINAL_DIR: str = os.path.join(RESULTS_BASE, "final")


def init(
    novel_name: str,
    characters: list[str] | None = None,
    character_dirs: dict[str, str] | None = None,
    model: str | None = None,
):
    """Re-initialize config for a different novel at runtime."""
    global NOVEL_NAME, TARGET_CHARACTERS, CHARACTER_DIRS, MODEL_NAME
    global NOVEL_PATH, RESULTS_BASE, CHAPTERS_DIR, OUTPUTS_DIR, FINAL_DIR

    NOVEL_NAME = novel_name
    if characters is not None:
        TARGET_CHARACTERS = characters
    if character_dirs is not None:
        CHARACTER_DIRS = character_dirs
    if model is not None:
        MODEL_NAME = model

    NOVEL_PATH = os.path.join(PROJECT_ROOT, "data", "novels", f"{NOVEL_NAME}.txt")
    RESULTS_BASE = os.path.join(PROJECT_ROOT, "results", "arc_extraction", NOVEL_NAME)
    CHAPTERS_DIR = os.path.join(RESULTS_BASE, "chapters")
    OUTPUTS_DIR = os.path.join(RESULTS_BASE, "outputs")
    FINAL_DIR = os.path.join(RESULTS_BASE, "final")


# ── Cost estimation helper ───────────────────────────────────────
def estimate_api_calls(num_chapters: int = 0) -> int:
    """Estimate total API calls based on current character config and chapter count."""
    n_chars = len(TARGET_CHARACTERS)
    # Phase 1A events + Phase 1B psych: n_chars × num_chapters each
    # Phase 1A axes + Phase 1B axes + Phase 2: n_chars each
    return n_chars * num_chapters * 2 + n_chars * 3


ESTIMATED_CALLS: int = 0  # updated after Phase 0 with actual chapter count
NUM_CHAPTERS: int = 0    # set after Phase 0


def count_chapters() -> int:
    """Count chapter files already split in CHAPTERS_DIR."""
    if not os.path.exists(CHAPTERS_DIR):
        return 0
    return len([
        f for f in os.listdir(CHAPTERS_DIR)
        if f.startswith("chapter_") and f.endswith(".txt")
    ])
