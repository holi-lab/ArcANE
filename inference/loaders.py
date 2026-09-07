"""Read probe / arc / chapter assets and flatten probes into trials.

`iter_trials` yields trials for inference. Each trial carries
exactly the inputs needed to render a prompt and (later) attribute the
response back to a (probe, phase) cell.
"""

import json
from dataclasses import asdict, dataclass
from typing import Iterator, Optional

from .config import axes_file, chapter_dir, chapter_path, probe_file


@dataclass(frozen=True)
class Trial:
    novel: str
    character: str
    character_slug: str
    axis_id: str
    axis_name: str
    axis_type: str                     # "intrapersonal" | "relational"
    target_character: Optional[str]    # only set for relational axes
    probe_id: str
    probe_type: str                    # "in_text" | "in_world" | "out_of_world"
    era_label: Optional[str]
    phase_idx: int
    phase_label: str
    query_chapter: int
    scenario: str
    question: str

    @property
    def trial_id(self) -> str:
        # mode + model are appended by the runner since one Trial may run
        # across several modes/models without changing identity-on-disk
        return f"{self.probe_id}__p{self.phase_idx}"


def load_probe_file(novel: str, character_slug: str, variant: str = "main") -> dict:
    return json.loads(probe_file(novel, character_slug, variant).read_text(encoding="utf-8"))


def load_axes_file(novel: str, character_slug: str) -> dict:
    return json.loads(axes_file(novel, character_slug).read_text(encoding="utf-8"))


def list_chapters(novel: str) -> list[int]:
    d = chapter_dir(novel)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("chapter_*.txt")):
        try:
            out.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return out


def read_chapter(novel: str, chapter_idx: int) -> str:
    p = chapter_path(novel, chapter_idx)
    if not p.exists():
        raise FileNotFoundError(f"missing chapter file: {p}")
    return p.read_text(encoding="utf-8")


def iter_trials(novel: str, character_slug: str, variant: str = "main",
                probe_types: Optional[set[str]] = None,
                skip_unavailable: bool = True) -> Iterator[Trial]:
    """Flatten a probe file into Trials.

    `skip_unavailable=True` drops phase responses the validator could not
    salvage (the canonical pipeline still writes them to disk with
    `unavailable=true`); we never want to query the model with a probe whose
    own ground truth is missing.
    """
    probe = load_probe_file(novel, character_slug, variant)
    novel_name = probe.get("novel", novel)
    character  = probe.get("character", character_slug)
    for fam in probe.get("families", []):
        axis_id   = fam["axis_id"]
        axis_name = fam["axis_name"]
        axis_type = fam.get("axis_type", "intrapersonal")
        target    = fam.get("target_character")
        for pr in fam.get("probes", []):
            ptype = pr["probe_type"]
            if probe_types is not None and ptype not in probe_types:
                continue
            for ph in pr.get("phase_responses", []):
                if skip_unavailable and ph.get("unavailable"):
                    continue
                yield Trial(
                    novel=novel_name,
                    character=character,
                    character_slug=character_slug,
                    axis_id=axis_id,
                    axis_name=axis_name,
                    axis_type=axis_type,
                    target_character=target,
                    probe_id=pr["probe_id"],
                    probe_type=ptype,
                    era_label=pr.get("era_label"),
                    phase_idx=ph["phase_idx"],
                    phase_label=ph["phase_label"],
                    query_chapter=ph["query_chapter"],
                    scenario=pr["scenario"],
                    question=pr["question"],
                )


def load_trials(novel: str, character_slug: str, **kwargs) -> list[Trial]:
    return list(iter_trials(novel, character_slug, **kwargs))


def trial_to_dict(t: Trial) -> dict:
    return asdict(t)
