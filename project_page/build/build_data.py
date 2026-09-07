"""Build the static JSON bundle that powers the ArcANE project page.

Reads the released benchmark artifacts (arcs, probes, judge outputs) from
``results/`` and writes ``project_page/data/``:

    data/catalog.json
        Novels, characters, models, modes, and pre-aggregated score tables
        (novel-, character-, and slice-level) using the same pooling and
        equal-weight-across-novels rule as ``analysis/build_paper_table.py``.
    data/characters/<novel>__<character>.json.gz
        One file per character: its arcs (phases, poles, validation), probes
        (scenario, question, phase-keyed references, validator verdicts), and
        every judge score at (probe, phase, model, mode) plus (probe, model,
        mode) trajectory scores.
    data/responses/<novel>__<character>/<probe_id>.json.gz
        Model responses per (phase, model, mode) for the six main-table
        models. Only written when ``--responses-root`` points at a directory
        holding the raw ``{mode}__{model}.jsonl`` inference files (these are
        not bundled in the repository).
    data/paper.json
        The paper-reported tables parsed from the LaTeX source, when
        ``--paper-tables`` is given (defaults to build/paper_tables.json).

Run from the repository root:

    python project_page/build/build_data.py \
        --responses-root /path/to/experiments/results
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_INF = PROJECT_ROOT / "results" / "inference"
RESULTS_ARC = PROJECT_ROOT / "results" / "arc_extraction"
OUT_ROOT = PROJECT_ROOT / "project_page" / "data"
JUDGE_TAG = "deepseek"
JUDGE_MODEL = "deepseek/deepseek-v4-flash"

# ── static vocab ──────────────────────────────────────────────────

MODELS = [
    # id, record key, label, short, family, size, group
    ("ds-v4-flash", "deepseek/deepseek-v4-flash", "DeepSeek-V4-Flash", "DS-V4-Flash", "DeepSeek-V4", "Flash", "main"),
    ("ds-v4-pro", "deepseek/deepseek-v4-pro", "DeepSeek-V4-Pro", "DS-V4-Pro", "DeepSeek-V4", "Pro", "main"),
    ("qwen3-8b", "Qwen/Qwen3-8B", "Qwen3-8B", "Qwen3-8B", "Qwen3", "8B", "main"),
    ("qwen3-32b", "Qwen/Qwen3-32B", "Qwen3-32B", "Qwen3-32B", "Qwen3", "32B", "main"),
    ("arcane-8b-dpo", "arcane-8B-dpo", "ArcANE-8B-DPO", "ArcANE-8B", "ArcANE-DPO", "8B", "main"),
    ("arcane-32b-dpo", "arcane-32B-dpo", "ArcANE-32B-DPO", "ArcANE-32B", "ArcANE-DPO", "32B", "main"),
    ("her-32b", "HER-32B", "HER-32B", "HER-32B", "HER", "32B", "added"),
    ("coser-8b", "Neph0s/CoSER-Llama-3.1-8B", "CoSER-8B", "CoSER-8B", "CoSER", "8B", "added"),
    ("coser-70b", "Neph0s/CoSER-Llama-3.1-70B", "CoSER-70B", "CoSER-70B", "CoSER", "70B", "added"),
    ("arcane-8b-sft", "arcane-8B-sft", "ArcANE-8B-SFT", "ArcANE-8B-SFT", "ArcANE-SFT", "8B", "added"),
    ("arcane-32b-sft", "arcane-32B-sft", "ArcANE-32B-SFT", "ArcANE-32B-SFT", "ArcANE-SFT", "32B", "added"),
]
MODEL_ALIASES = {
    "ChengyuDu0123/HER-32B": "HER-32B",
    "qwen/qwen3-8b": "Qwen/Qwen3-8B",
    "qwen/qwen3-32b": "Qwen/Qwen3-32B",
}
KEY_TO_ID = {key: mid for mid, key, *_ in MODELS}
RESPONSE_MODELS = {mid for mid, *_, group in MODELS if group == "main"}

MODES = [
    ("vanilla", "Vanilla", "Character identity and query chapter only."),
    ("summary", "Summary", "Vanilla plus summaries of the five most recent chapters up to the query chapter."),
    ("rag", "RAG", "Vanilla plus the top-6 source-text chunks retrieved for the scenario and question, masked to chapters up to the query chapter."),
    ("lifechoice", "LifeChoice", "CHARMAP port: chapter-summary description plus memory retrieved with that description as the query."),
    ("timechara", "TimeChara", "Narrative-experts port: a first call predicts the chapter and the character's presence, and the prediction is injected as a hint."),
    ("arc", "Arc", "Vanilla plus the automatically constructed Character Arc, truncated at the phase of the query chapter (ours)."),
]
MODE_IDS = [m for m, _, _ in MODES]

PROBE_TYPES = [
    ("in_text", "In-Scenario", "The scenario is lifted from a verbatim passage of the novel."),
    ("in_world", "In-World", "A new situation inside the novel's own world that the text never shows."),
    ("out_of_world", "Out-of-World", "The same axis transposed to a setting outside the novel's world."),
]
PTYPE_IDS = [p for p, _, _ in PROBE_TYPES]
_PROBE_TYPE_RE = re.compile(r"_(intext|inworld|outworld)_a(\d+)$")
_TYPE_MAP = {"intext": "in_text", "inworld": "in_world", "outworld": "out_of_world"}

METRICS = ["apf", "rpf", "rae", "ptf"]

NOVELS = [
    # dir slug, url id, title, author, year, slice, gutenberg id
    ("Anna_Kareina", "anna-karenina", "Anna Karenina", "Leo Tolstoy", 1878, "validated", 1399),
    ("Monte_Cristo", "monte-cristo", "The Count of Monte Cristo", "Alexandre Dumas", 1846, "validated", 1184),
    ("don_quixote", "don-quixote", "Don Quixote", "Miguel de Cervantes", 1615, "validated", 996),
    ("Benjamin_Franklin", "benjamin-franklin", "The Autobiography of Benjamin Franklin", "Benjamin Franklin", 1791, "validated", 20203),
    ("He Knew He Was Right", "he-knew-he-was-right", "He Knew He Was Right", "Anthony Trollope", 1869, "lowpop", 5140),
    ("The Odd Women", "the-odd-women", "The Odd Women", "George Gissing", 1893, "lowpop", 4313),
    ("East Lynne", "east-lynne", "East Lynne", "Ellen Wood", 1861, "training", 3322),
    ("The Underdogs", "the-underdogs", "The Underdogs", "Mariano Azuela", 1915, "training", 549),
]
SLICES = {
    "validated": {
        "label": "Validated slice",
        "desc": "Human-validated evaluation novels. The paper's slice also includes Harry Potter, whose artifacts are not distributed, so these aggregates cover four of the five novels.",
    },
    "lowpop": {
        "label": "Low-popularity slice",
        "desc": "Held-out low-popularity novels used as a memorization control. Arcs are automatically constructed and not human-validated; neither novel is in the training pool.",
    },
    "training": {
        "label": "Training-novel reference",
        "desc": "Both novels are in the SFT/DPO training pool, so ArcANE results here are in-distribution and reported for reference only.",
    },
}
# Title character of each novel (plus Levin as Anna Karenina's co-protagonist),
# matching analysis/build_per_character_analysis.py.
CENTRAL_CHARS = {
    "Anna_Kareina": {"anna_karenina", "konstantin_levin"},
    "don_quixote": {"don_quixote"},
    "Benjamin_Franklin": {"benjamin_franklin"},
    "Monte_Cristo": {"edmond_dantes"},
    "He Knew He Was Right": {"louis_trevelyan"},
    "The Odd Women": {"rhoda_nunn"},
    "East Lynne": {"lady_isabel_vane"},
    "The Underdogs": {"demetrio_macias"},
}

RESPONSE_CHAR_CAP = 6000
THINK_CHAR_CAP = 2500


def write_gz_json(path: Path, obj) -> None:
    """Write compact JSON as <path>.gz; the page decompresses it in the browser."""
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(str(path) + ".gz", "wb", compresslevel=9) as fh:
        fh.write(data)


# ── helpers ───────────────────────────────────────────────────────

def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def probe_type_of(probe_id: str):
    m = _PROBE_TYPE_RE.search(probe_id or "")
    return _TYPE_MAP[m.group(1)] if m else None


def r1(v):
    return None if v is None else round(v, 2)


def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


class Pool:
    """scores[model_id][mode][ptype][metric] -> list of probe-level values."""

    def __init__(self):
        self.d = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    def add(self, model_id, mode, ptype, metric, val):
        self.d[model_id][mode][ptype][metric].append(val)

    def table(self):
        """-> {model_id: {mode: {ptype: [apf, rpf, rae, ptf]}}} with per-cell means."""
        out = {}
        for mid in self.d:
            out[mid] = {}
            for mode in self.d[mid]:
                out[mid][mode] = {}
                for pt in PTYPE_IDS:
                    cell = self.d[mid][mode].get(pt)
                    if not cell:
                        continue
                    out[mid][mode][pt] = [r1(mean_or_none(cell.get(m) or [])) for m in METRICS]
        return out

    def counts(self):
        """-> {model_id: {mode: {ptype: [n_phase_records, n_trajectories]}}}"""
        out = {}
        for mid in self.d:
            out[mid] = {}
            for mode in self.d[mid]:
                out[mid][mode] = {}
                for pt in PTYPE_IDS:
                    cell = self.d[mid][mode].get(pt)
                    if not cell:
                        continue
                    out[mid][mode][pt] = [len(cell.get("apf", [])), len(cell.get("ptf", []))]
        return out


def merge_novel_tables(tables):
    """Equal-weight mean across novel-level tables (the paper's aggregation rule)."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [[] for _ in METRICS])))
    for t in tables:
        for mid, by_mode in t.items():
            for mode, by_pt in by_mode.items():
                for pt, vals in by_pt.items():
                    for i, v in enumerate(vals):
                        if v is not None:
                            acc[mid][mode][pt][i].append(v)
    out = {}
    for mid in acc:
        out[mid] = {}
        for mode in acc[mid]:
            out[mid][mode] = {}
            for pt in acc[mid][mode]:
                out[mid][mode][pt] = [r1(mean_or_none(v)) for v in acc[mid][mode][pt]]
    return out


def overall_of(cell_table_for_model_mode):
    vals = []
    for pt in PTYPE_IDS:
        for v in cell_table_for_model_mode.get(pt, []) or []:
            if v is not None:
                vals.append(v)
    return r1(mean(vals)) if vals else None


# ── arcs ──────────────────────────────────────────────────────────

def load_arc_file(novel_dir: Path, char: str):
    """Prefer the human-validated arc file, then the critic-grounded one, then the merged one."""
    for sub, suffix in (("final_validated", "validated"), ("final_grounded", "grounded"), ("final", "final")):
        p = novel_dir / sub / f"{char}_{suffix}_axes.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")), sub
    return None, None


def trim_citation(c):
    return {k: c.get(k) for k in ("author", "title", "year", "publication") if c.get(k) is not None}


def build_arc(axis: dict, family: dict | None, votes: dict | None):
    phases = []
    for i, ph in enumerate(axis.get("trajectory", [])):
        entry = {
            "idx": i,
            "label": ph.get("phase"),
            "chapters": ph.get("chapter_range"),
            "description": ph.get("position_description"),
            "key_moments": ph.get("key_moments") or [],
        }
        if family and family.get("life_stages"):
            ls = family["life_stages"].get(str(i)) or {}
            if ls.get("life_stage"):
                entry["life_stage"] = ls.get("life_stage")
                entry["approx_age"] = ls.get("approx_age")
        phases.append(entry)
    critics = []
    lv = axis.get("literary_validation") or {}
    for key, ev in (lv.get("evaluators") or {}).items():
        critics.append({
            "id": key,
            "label": ev.get("label"),
            "verdict": ev.get("verdict"),
            "reasoning": ev.get("reasoning"),
            "citations": [trim_citation(c) for c in (ev.get("citations") or [])][:4],
        })
    out = {
        "axis_id": axis["axis_id"],
        "axis_type": axis.get("axis_type"),
        "axis_name": axis.get("axis_name"),
        "dimension_label": axis.get("dimension_label"),
        "pole_start": axis.get("pole_start"),
        "pole_end": axis.get("pole_end"),
        "arc_direction": axis.get("arc_direction"),
        "confidence": axis.get("confidence"),
        "source": axis.get("source"),
        "target_character": axis.get("target_character"),
        "evidence_summary": axis.get("evidence_summary"),
        "n_phases": len(phases),
        "phases": phases,
        "critic_tag": lv.get("validation_tag"),
        "critics": critics,
        "validation": None,
    }
    ann = axis.get("annotation")
    if ann:
        out["validation"] = {"valid_votes": ann.get("valid_votes"), "n_annotators": ann.get("n_annotators")}
    elif votes:
        out["validation"] = {"valid_votes": votes.get("valid_votes"), "n_annotators": votes.get("n_annotators")}
    if family:
        out["decision_variable"] = family.get("decision_variable")
        out["phase_contrasts"] = family.get("phase_contrasts") or []
        ax_re = family.get("axis_re_expression") or {}
        out["abstract_axis"] = ax_re.get("abstract_axis")
        out["abstract_phases"] = ax_re.get("abstract_phases")
        out["weak_pairs"] = [
            {k: wp.get(k) for k in ("probe_id", "from_phase_idx", "to_phase_idx", "separation")}
            for wp in (family.get("weak_pairs") or [])
        ]
    return out


# ── probes ────────────────────────────────────────────────────────

def build_probe(family: dict, q: dict):
    refs = []
    for r in q.get("phase_responses", []):
        qual = r.get("quality") or {}
        refs.append({
            "phase_idx": r.get("phase_idx"),
            "phase_label": r.get("phase_label"),
            "query_chapter": r.get("query_chapter"),
            "action": r.get("gt_action"),
            "speech": r.get("gt_speech"),
            "thought": r.get("gt_thought"),
            "typicality": r.get("gt_typicality"),
            "unavailable": bool(r.get("unavailable")),
            "voice": (qual.get("voice") or {}).get("verdict"),
            "phase_fit": (qual.get("phase_fit") or {}).get("verdict"),
            "retry_count": r.get("retry_count", 0),
        })
    pc = q.get("probe_check") or {}
    check = {}
    for key in ("anchor", "world"):
        if pc.get(key):
            check[key] = {"verdict": pc[key].get("verdict"), "note": pc[key].get("note")}
    disc = pc.get("discrimination") or {}
    check["weak_pairs"] = [
        {"from": wp.get("from_phase_idx"), "to": wp.get("to_phase_idx"), "separation": wp.get("separation")}
        for wp in (disc.get("weak_pairs") or [])
    ]
    out = {
        "probe_id": q["probe_id"],
        "axis_id": family["axis_id"],
        "probe_type": q.get("probe_type"),
        "anchor_phase_idx": q.get("anchor_phase_idx"),
        "anchor_phase_label": q.get("anchor_phase_label"),
        "anchor_query_chapter": q.get("anchor_query_chapter"),
        "era_label": q.get("era_label"),
        "era_description": q.get("era_description"),
        "scenario": q.get("scenario"),
        "question": q.get("question"),
        "refs": refs,
        "check": check,
        "scores": {},   # filled from judge records
    }
    src = q.get("anchor_source")
    if isinstance(src, dict):
        out["anchor_source"] = {"chapter": src.get("chapter"), "event": src.get("event_description")}
    return out


# ── responses ─────────────────────────────────────────────────────

_THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*", re.S)


def split_response(text: str):
    text = text or ""
    think = None
    m = _THINK_RE.match(text)
    if m:
        inner = m.group(1).strip()
        think = inner if inner else None
        text = text[m.end():]
    text = text.strip()
    rec = {"text": text[:RESPONSE_CHAR_CAP], "chars": len(text)}
    if len(text) > RESPONSE_CHAR_CAP:
        rec["truncated"] = True
    if think:
        rec["think"] = think[:THINK_CHAR_CAP]
        if len(think) > THINK_CHAR_CAP:
            rec["think_truncated"] = True
    return rec


def load_responses(char_dir: Path):
    """-> {(probe_id, phase_idx, model_id, mode): record}; latest timestamp wins on duplicates."""
    out = {}
    if not char_dir.is_dir():
        return out
    for path in sorted(char_dir.iterdir()):
        name = path.name
        if not name.endswith(".jsonl") or "__" not in name:
            continue
        if name.startswith(("eval_", "traj_")):
            continue
        mode = name.split("__", 1)[0]
        if mode not in MODE_IDS:
            continue
        for r in iter_jsonl(path):
            key = MODEL_ALIASES.get(r.get("model"), r.get("model"))
            mid = KEY_TO_ID.get(key)
            if mid not in RESPONSE_MODELS or r.get("mode") != mode:
                continue
            if r.get("error"):
                continue
            k = (r.get("probe_id"), r.get("phase_idx"), mid, mode)
            prev = out.get(k)
            if prev is None or (r.get("ts") or "") >= (prev.get("ts") or ""):
                out[k] = r
    return out


# ── main build ────────────────────────────────────────────────────

def build(responses_root: Path | None, paper_tables: Path | None, out_root: Path):
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "characters").mkdir(parents=True)

    catalog_novels = []
    slice_tables = defaultdict(list)
    coverage_notes = []
    totals = {"novels": 0, "characters": 0, "arcs": 0, "probes": 0, "phase_slots": 0,
              "judge_records": 0, "trajectory_records": 0, "responses": 0,
              "responses_missing": 0}

    for slug, nid, title, author, year, slice_id, gutenberg in NOVELS:
        inf_dir = RESULTS_INF / slug
        arc_dir = RESULTS_ARC / slug
        if not inf_dir.is_dir():
            coverage_notes.append(f"no judge outputs for {slug}; skipped")
            continue
        chars_meta = json.loads((arc_dir / "characters.json").read_text(encoding="utf-8"))
        dir_to_name = {v: k for k, v in chars_meta.get("character_dirs", {}).items()}
        novel_pool = Pool()
        novel_entry = {
            "id": nid, "slug": slug, "title": title, "author": author, "year": year,
            "slice": slice_id, "gutenberg": gutenberg, "characters": [],
        }
        for char_dir in sorted(p for p in inf_dir.iterdir() if p.is_dir()):
            char = char_dir.name
            probes_path = arc_dir / "probes" / f"{char}_probes.json"
            if not probes_path.exists():
                coverage_notes.append(f"{slug}/{char}: no probe file; skipped")
                continue
            pj = json.loads(probes_path.read_text(encoding="utf-8"))
            arc_json, arc_source = load_arc_file(arc_dir, char)
            axes_by_id = {}
            votes = {}
            if arc_json:
                for ax in arc_json.get("intrapersonal_axes", []) + arc_json.get("relational_axes", []):
                    axes_by_id[ax["axis_id"]] = ax
                votes = ((arc_json.get("validation") or {}).get("axis_votes") or {})

            arcs, probes, probe_index = [], [], {}
            for fam in pj.get("families", []):
                axis = axes_by_id.get(fam["axis_id"])
                if axis is None:
                    coverage_notes.append(f"{slug}/{char}: axis {fam['axis_id']} missing in arc file; built from probe family only")
                    axis = {"axis_id": fam["axis_id"], "axis_name": fam.get("axis_name"),
                            "axis_type": fam.get("axis_type"), "dimension_label": fam.get("dimension_label"),
                            "target_character": fam.get("target_character"), "trajectory": []}
                arc = build_arc(axis, fam, votes.get(fam["axis_id"]))
                arc["probe_ids"] = []
                for q in fam.get("probes", []):
                    pr = build_probe(fam, q)
                    probes.append(pr)
                    probe_index[pr["probe_id"]] = pr
                    arc["probe_ids"].append(pr["probe_id"])
                    totals["phase_slots"] += len(pr["refs"])
                arc["n_probes"] = len(arc["probe_ids"])
                arcs.append(arc)

            # judge records -----------------------------------------------
            char_pool = Pool()
            arc_pools = defaultdict(Pool)
            n_eval = n_traj = 0
            for r in iter_jsonl(char_dir / f"eval_records_{JUDGE_TAG}.jsonl"):
                if not r.get("parse_ok"):
                    continue
                key = MODEL_ALIASES.get(r.get("model"), r.get("model"))
                mid = KEY_TO_ID.get(key)
                mode = r.get("mode")
                pid = r.get("probe_id")
                ptype = probe_type_of(pid)
                if mid is None or mode not in MODE_IDS or ptype is None:
                    continue
                sc = r.get("scores") or {}
                if not all(k in sc for k in ("apf", "rpf", "rae")):
                    continue
                n_eval += 1
                for metric in ("apf", "rpf", "rae"):
                    for pool in (novel_pool, char_pool):
                        pool.add(mid, mode, ptype, metric, sc[metric])
                pr = probe_index.get(pid)
                if pr is not None:
                    arc_pools[pr["axis_id"]].add(mid, mode, ptype, "apf", sc["apf"])
                    arc_pools[pr["axis_id"]].add(mid, mode, ptype, "rpf", sc["rpf"])
                    arc_pools[pr["axis_id"]].add(mid, mode, ptype, "rae", sc["rae"])
                    cell = pr["scores"].setdefault(mid, {}).setdefault(mode, {"phases": {}, "ptf": None})
                    cell["phases"][str(r.get("phase_idx"))] = [sc["apf"], sc["rpf"], sc["rae"]]
            for r in iter_jsonl(char_dir / f"traj_records_{JUDGE_TAG}.jsonl"):
                if not r.get("parse_ok"):
                    continue
                key = MODEL_ALIASES.get(r.get("model"), r.get("model"))
                mid = KEY_TO_ID.get(key)
                mode = r.get("mode")
                pid = r.get("probe_id")
                ptype = probe_type_of(pid)
                if mid is None or mode not in MODE_IDS or ptype is None:
                    continue
                sc = r.get("scores") or {}
                dims = [sc[k] for k in ("ptf_alignment", "ptf_direction", "ptf_shape") if k in sc]
                if not dims:
                    continue
                n_traj += 1
                val = sum(dims) / len(dims)
                for pool in (novel_pool, char_pool):
                    pool.add(mid, mode, ptype, "ptf", val)
                pr = probe_index.get(pid)
                if pr is not None:
                    arc_pools[pr["axis_id"]].add(mid, mode, ptype, "ptf", val)
                    cell = pr["scores"].setdefault(mid, {}).setdefault(mode, {"phases": {}, "ptf": None})
                    cell["ptf"] = [sc.get("ptf_alignment"), sc.get("ptf_direction"), sc.get("ptf_shape")]
            totals["judge_records"] += n_eval
            totals["trajectory_records"] += n_traj

            for arc in arcs:
                arc["table"] = arc_pools[arc["axis_id"]].table()

            # responses ---------------------------------------------------
            resp_written = 0
            resp_missing = 0
            if responses_root is not None:
                raw = load_responses(responses_root / slug / char)
                rdir = out_root / "responses" / f"{nid}__{char}"
                rdir.mkdir(parents=True, exist_ok=True)
                for pr in probes:
                    bundle = {}
                    for mid, by_mode in pr["scores"].items():
                        if mid not in RESPONSE_MODELS:
                            continue
                        for mode, cell in by_mode.items():
                            for ph in cell["phases"]:
                                rec = raw.get((pr["probe_id"], int(ph), mid, mode))
                                if rec is None:
                                    resp_missing += 1
                                    continue
                                bundle.setdefault(mid, {}).setdefault(mode, {})[ph] = split_response(rec.get("response"))
                                resp_written += 1
                    write_gz_json(rdir / f"{pr['probe_id']}.json",
                                  {"probe_id": pr["probe_id"], "responses": bundle})
                    pr["has_responses"] = bool(bundle)
            totals["responses"] += resp_written
            totals["responses_missing"] += resp_missing

            char_name = dir_to_name.get(char, char.replace("_", " ").title())
            char_table = char_pool.table()
            char_file = {
                "novel": {"id": nid, "slug": slug, "title": title, "slice": slice_id},
                "character": {"id": char, "name": char_name, "central": char in CENTRAL_CHARS.get(slug, set()),
                              "arc_source": arc_source},
                "arcs": arcs,
                "probes": probes,
                "table": char_table,
                "counts": char_pool.counts(),
            }
            write_gz_json(out_root / "characters" / f"{nid}__{char}.json", char_file)

            novel_entry["characters"].append({
                "id": char, "name": char_name,
                "central": char in CENTRAL_CHARS.get(slug, set()),
                "n_arcs": len(arcs), "n_probes": len(probes),
                "n_phase_slots": sum(len(p["refs"]) for p in probes),
                "arcs": [{"axis_id": a["axis_id"], "axis_name": a["axis_name"], "axis_type": a["axis_type"],
                          "dimension_label": a["dimension_label"], "arc_direction": a["arc_direction"],
                          "n_phases": a["n_phases"], "n_probes": a["n_probes"],
                          "target_character": a["target_character"]} for a in arcs],
                "table": char_table,
                "responses": resp_written,
            })
            totals["characters"] += 1
            totals["arcs"] += len(arcs)
            totals["probes"] += len(probes)

        novel_entry["table"] = novel_pool.table()
        novel_entry["counts"] = novel_pool.counts()
        novel_entry["n_characters"] = len(novel_entry["characters"])
        novel_entry["n_arcs"] = sum(c["n_arcs"] for c in novel_entry["characters"])
        novel_entry["n_probes"] = sum(c["n_probes"] for c in novel_entry["characters"])
        catalog_novels.append(novel_entry)
        slice_tables[slice_id].append(novel_entry["table"])
        totals["novels"] += 1

    slices = {}
    for sid, meta in SLICES.items():
        tables = slice_tables.get(sid, [])
        if not tables:
            continue
        slices[sid] = {
            **meta,
            "novels": [n["id"] for n in catalog_novels if n["slice"] == sid],
            "table": merge_novel_tables(tables),
        }

    catalog = {
        "judge": JUDGE_MODEL,
        "judge_tag": JUDGE_TAG,
        "models": [
            {"id": mid, "key": key, "label": label, "short": short, "family": fam, "size": size,
             "group": group, "responses": mid in RESPONSE_MODELS and responses_root is not None}
            for mid, key, label, short, fam, size, group in MODELS
        ],
        "modes": [{"id": m, "label": l, "desc": d} for m, l, d in MODES],
        "probe_types": [{"id": p, "label": l, "desc": d} for p, l, d in PROBE_TYPES],
        "metrics": [
            {"id": "apf", "label": "APF", "name": "Action Phase-Fidelity", "level": "phase"},
            {"id": "rpf", "label": "RPF", "name": "Reasoning Phase-Fidelity", "level": "phase"},
            {"id": "rae", "label": "RAE", "name": "Reasoning–Action Entailment", "level": "phase"},
            {"id": "ptf", "label": "PTF", "name": "Phase Trajectory Fidelity", "level": "trajectory"},
        ],
        "slices": slices,
        "novels": catalog_novels,
        "totals": totals,
        "notes": coverage_notes,
        "response_cap_chars": RESPONSE_CHAR_CAP,
    }
    (out_root / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    if paper_tables and paper_tables.exists():
        shutil.copyfile(paper_tables, out_root / "paper.json")

    return catalog


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses-root", type=Path, default=None,
                    help="directory holding <novel>/<character>/{mode}__{model}.jsonl inference outputs")
    ap.add_argument("--paper-tables", type=Path, default=Path(__file__).resolve().parent / "paper_tables.json")
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    args = ap.parse_args(argv)
    catalog = build(args.responses_root, args.paper_tables, args.out)
    t = catalog["totals"]
    print(f"wrote {args.out}")
    print(f"  novels={t['novels']} characters={t['characters']} arcs={t['arcs']} probes={t['probes']} phase_slots={t['phase_slots']}")
    print(f"  judge_records={t['judge_records']} trajectory_records={t['trajectory_records']}")
    if args.responses_root:
        print(f"  responses={t['responses']} missing={t['responses_missing']}")
    for n in catalog["notes"]:
        print("  note:", n, file=sys.stderr)


if __name__ == "__main__":
    main()
