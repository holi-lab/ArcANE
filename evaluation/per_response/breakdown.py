"""Break eval scores down by question type (probe_type).

    python -m evaluation.per_response.breakdown --novel Anna_Kareina --character anna_karenina

The judge writes `eval_records.jsonl` / `eval_summary.json` aggregated only by
(model, mode); the question type — `in_text` / `in_world` / `out_of_world` —
is dropped. This re-reads the *already-scored* records, joins each to its
`probe_type` from the probe file, and re-aggregates per (model, mode, type).

It calls no judge — it only re-slices existing scores, so it is free and
instant. Writes `eval_summary_by_type.json` next to `eval_summary.json` and
prints a per-question-type table to the terminal.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from inference import config
from . import config as eval_config
from .prompt import DIMENSIONS
from .runner import _thought_suffix, judge_tag, records_path

# Stable column / row order; anything unmapped lands in "unknown".
TYPE_ORDER = ("in_text", "in_world", "out_of_world", "unknown")


# ── probe_id -> probe_type ─────────────────────────────────────────
def probe_type_map(novel: str, character: str,
                   variant: str = "main") -> dict[str, str]:
    """Read the probe file -> {probe_id: probe_type}."""
    path = config.probe_file(novel, character, variant)
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for fam in data.get("families", []):
        for probe in fam.get("probes", []):
            pid = probe.get("probe_id")
            if pid:
                out[pid] = probe.get("probe_type") or "unknown"
    return out


# ── aggregation ────────────────────────────────────────────────────
def aggregate_by_type(rec_path: Path, type_map: dict[str, str],
                      scale: int) -> tuple[dict, int]:
    """eval_records.jsonl -> per-(model, mode, probe_type) average scores.

    Mirrors `runner.aggregate`: only records at `scale` count, the last record
    for a (probe_id, phase_idx, model, mode) wins, parse failures are dropped.
    Returns (by_model, n_unmapped_probe_ids).
    """
    latest: dict[tuple, dict] = {}
    if rec_path.exists():
        for line in rec_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("scale") != scale:
                continue
            key = (r.get("probe_id"), r.get("phase_idx"),
                   r.get("model"), r.get("mode"))
            latest[key] = r

    # {model: {mode: {ptype: [records]}}}
    grouped: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    unmapped: set[str] = set()
    for r in latest.values():
        if not (r.get("parse_ok") and r.get("scores")):
            continue
        pid = r.get("probe_id")
        ptype = type_map.get(pid)
        if ptype is None:
            unmapped.add(pid)
            ptype = "unknown"
        grouped[r["model"]][r["mode"]][ptype].append(r)

    by_model: dict[str, dict] = {}
    for model, modes in grouped.items():
        by_model[model] = {}
        for mode, types in modes.items():
            by_model[model][mode] = {}
            for ptype, recs in types.items():
                agg = {dim: round(sum(x["scores"][dim] for x in recs)
                                  / len(recs), 3)
                       for dim in DIMENSIONS}
                agg["average"] = round(
                    sum(x["average"] for x in recs) / len(recs), 3)
                agg["n"] = len(recs)
                by_model[model][mode][ptype] = agg
    return by_model, len(unmapped)


# ── terminal table ─────────────────────────────────────────────────
def _present_types(by_model: dict) -> list[str]:
    seen = {t for modes in by_model.values()
            for types in modes.values() for t in types}
    return [t for t in TYPE_ORDER if t in seen]


def print_by_type(summary: dict) -> None:
    """Per-(model, mode) row, one AVG/n column block per question type."""
    by_model = summary.get("by_model") or {}
    types = _present_types(by_model)
    print()
    print("=" * (24 + 16 * len(types)))
    print(f"EVAL BY QUESTION TYPE — {summary['novel']} / {summary['character']}"
          f"   (judge: {summary['judge_model']}, "
          f"scale: 1-{summary.get('score_scale', '?')})")
    print("=" * (24 + 16 * len(types)))
    if not by_model:
        print("(no scored records found)")
        return

    header = f"  {'mode':14}" + "".join(f"{t:>16}" for t in types)
    sub = f"  {'':14}" + "".join(f"{'AVG':>9}{'n':>7}" for _ in types)
    for model in sorted(by_model):
        print(f"\nmodel: {model}")
        print(header)
        print(sub)
        for mode in sorted(by_model[model]):
            row = by_model[model][mode]
            cells = ""
            for t in types:
                if t in row:
                    cells += f"{row[t]['average']:>9.2f}{row[t]['n']:>7d}"
                else:
                    cells += f"{'-':>9}{'-':>7}"
            print(f"  {mode:14}{cells}")
    print("=" * (24 + 16 * len(types)))


# ── CLI ────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evaluation.per_response.breakdown")
    p.add_argument("--novel", default=eval_config.EVAL_NOVEL)
    p.add_argument("--character", default=eval_config.EVAL_CHARACTER)
    p.add_argument("--variant", default="main",
                   help="probe file variant (default: main)")
    p.add_argument("--judge-model", default=eval_config.JUDGE_MODEL,
                   help=f"judge model (default: {eval_config.JUDGE_MODEL})")
    p.add_argument("--include-thought", action=argparse.BooleanOptionalAction,
                   default=eval_config.INCLUDE_THOUGHT,
                   help=f"select which ablation's records to break down "
                        f"(default: {eval_config.INCLUDE_THOUGHT}). "
                        f"--no-include-thought reads the `*_nothought.*` "
                        f"sibling files.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    scale = eval_config.SCORE_SCALE

    rec_path = records_path(args.novel, args.character, args.judge_model,
                             args.include_thought)
    if not rec_path.exists():
        raise SystemExit(
            f"no eval records at {rec_path} — run "
            f"`python -m evaluation.per_response --novel {args.novel} "
            f"--character {args.character}"
            f"{'' if args.include_thought else ' --no-include-thought'}` first.")

    type_map = probe_type_map(args.novel, args.character, args.variant)
    by_model, n_unmapped = aggregate_by_type(rec_path, type_map, scale)
    if n_unmapped:
        print(f"warning: {n_unmapped} probe_id(s) in records not found in the "
              f"probe file — counted as 'unknown'.")

    summary = {
        "novel": args.novel,
        "character": args.character,
        "judge_model": args.judge_model,
        "include_thought": args.include_thought,
        "score_scale": scale,
        "by_model": by_model,
    }
    out_name = (f"eval_summary_by_type_{judge_tag(args.judge_model)}"
                f"{_thought_suffix(args.include_thought)}.json")
    out_path = (config.RESULTS_ROOT / args.novel / args.character / out_name)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print_by_type(summary)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
