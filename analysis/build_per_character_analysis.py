"""Per-character stratified analysis for §5.3 of the paper.

Splits the available validated-slice characters into a *central* tier
(title character or canonical co-protagonist) and a *supporting* tier.
Reports the Arc-minus-Vanilla Overall lift for each model and tier.

Same pipeline as analysis.build_paper_table, but the per-(model, mode,
ptype, metric) cell is aggregated *within each character class* instead of
within each novel. Two aggregation paths are reported:
    1. probe-pool:  pool all probe-level scores across characters in the
                    class, take the mean.
    2. char-mean:   take per-character means first, then average characters
                    in the class. Less affected by characters with many
                    probes.
We report both because the two can diverge when probe counts are uneven.

Outputs:
    paper_results/per_character_analysis.json   — full per-(model, mode,
                                                  class) breakdown.
    paper_results/per_character_lift.pdf        — figure: Arc-minus-Vanilla lift
                                                  per model, central vs
                                                  supporting (paired bars).

Run:
    uv run python -m analysis.build_per_character_analysis
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from analysis.build_paper_table import (
    MODELS_MAIN, MODES, NOVELS_MAIN, METRICS_PER_SECTION, SECTIONS,
    RESULTS_ROOT, _iter_jsonl, probe_type_of, _split_available,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "paper_results"

JUDGE_TAG = "deepseek"

# Canonical central-character mapping. Title character of each novel, plus
# co-protagonist where the novel is built around two equally weighted leads.
# Anna Karenina is the standard dual-protagonist case (Tolstoy's parallel
# Anna / Levin construction). All other novels have a single titular lead.
CENTRAL_CHARS: dict[str, set[str]] = {
    "Harry_Potter":      {"harry_potter"},
    "Anna_Kareina":      {"anna_karenina", "konstantin_levin"},
    "don_quixote":       {"don_quixote"},
    "Benjamin_Franklin": {"benjamin_franklin"},
    "Monte_Cristo":      {"edmond_dantes"},
}


def collect_per_character(judge_tag: str, novels, models) -> dict:
    """Return scores[novel][char][model][mode][ptype][metric] = list[float]."""
    target_models = {m for m, _, _ in models}
    target_modes  = {m for m, _ in MODES}
    bucket: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))))))

    for novel in novels:
        novel_dir = RESULTS_ROOT / novel
        if not novel_dir.exists():
            continue
        for char_dir in sorted(p for p in novel_dir.iterdir() if p.is_dir()):
            char = char_dir.name

            for r in _iter_jsonl(char_dir / f"eval_records_{judge_tag}.jsonl"):
                if not r.get("parse_ok"):
                    continue
                if (r.get("model") not in target_models
                        or r.get("mode") not in target_modes):
                    continue
                ptype = probe_type_of(r.get("probe_id") or "")
                if ptype is None:
                    continue
                scores = r.get("scores") or {}
                for label, _src, dims in METRICS_PER_SECTION:
                    if _src != "eval1":
                        continue
                    vals = [scores[d] for d in dims if d in scores]
                    if vals:
                        bucket[novel][char][r["model"]][r["mode"]][ptype][label].append(
                            sum(vals) / len(vals))

            for r in _iter_jsonl(char_dir / f"traj_records_{judge_tag}.jsonl"):
                if not r.get("parse_ok"):
                    continue
                if (r.get("model") not in target_models
                        or r.get("mode") not in target_modes):
                    continue
                ptype = probe_type_of(r.get("probe_id") or "")
                if ptype is None:
                    continue
                scores = r.get("scores") or {}
                for label, _src, dims in METRICS_PER_SECTION:
                    if _src != "eval2":
                        continue
                    vals = [scores[d] for d in dims if d in scores]
                    if vals:
                        bucket[novel][char][r["model"]][r["mode"]][ptype][label].append(
                            sum(vals) / len(vals))
    return bucket


def per_character_means(bucket: dict) -> dict:
    """Return mean per (novel, char, model, mode, ptype, metric)."""
    out: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
            lambda: defaultdict(dict)))))
    for novel, by_char in bucket.items():
        for char, by_model in by_char.items():
            for model, by_mode in by_model.items():
                for mode, by_ptype in by_mode.items():
                    for ptype, by_metric in by_ptype.items():
                        for metric, vals in by_metric.items():
                            out[novel][char][model][mode][ptype][metric] = (
                                mean(vals) if vals else None)
    return out


def char_overall(per_char: dict, novel: str, char: str,
                 model: str, mode: str) -> float | None:
    """Mean of 12 metric cells for one (novel, char, model, mode)."""
    vals: list[float] = []
    cell = per_char.get(novel, {}).get(char, {}).get(model, {}).get(mode, {})
    for ptype, _ in SECTIONS:
        for metric, _src, _dims in METRICS_PER_SECTION:
            v = cell.get(ptype, {}).get(metric)
            if v is not None:
                vals.append(v)
    return mean(vals) if vals else None


def class_overall_probepool(bucket: dict, novels, chars_in_class,
                            model: str, mode: str) -> float | None:
    """Pool all probe-level scores in the class, then mean over the 12 cells.

    chars_in_class: iterable of (novel, char) tuples that belong to the class.
    """
    per_cell: list[float] = []
    for ptype, _ in SECTIONS:
        for metric, _src, _dims in METRICS_PER_SECTION:
            pooled: list[float] = []
            for novel, char in chars_in_class:
                vals = bucket.get(novel, {}).get(char, {}).get(model, {})\
                                            .get(mode, {}).get(ptype, {})\
                                            .get(metric, [])
                pooled.extend(vals)
            if pooled:
                per_cell.append(mean(pooled))
    return mean(per_cell) if per_cell else None


def class_overall_charmean(per_char: dict, chars_in_class,
                           model: str, mode: str) -> float | None:
    per_char_overalls = [
        char_overall(per_char, novel, char, model, mode)
        for novel, char in chars_in_class
    ]
    per_char_overalls = [v for v in per_char_overalls if v is not None]
    return mean(per_char_overalls) if per_char_overalls else None


def build_classes(bucket: dict, novels) -> tuple[list, list]:
    central: list[tuple[str, str]] = []
    supporting: list[tuple[str, str]] = []
    for novel in novels:
        for char in sorted(bucket.get(novel, {}).keys()):
            if char in CENTRAL_CHARS.get(novel, set()):
                central.append((novel, char))
            else:
                supporting.append((novel, char))
    return central, supporting


def model_lift(bucket, per_char, chars, model: str, agg: str) -> dict:
    """Per-(class, model) Arc Overall, Vanilla Overall, and lift."""
    by_mode: dict[str, float | None] = {}
    for mode_id, _ in MODES:
        if agg == "probepool":
            by_mode[mode_id] = class_overall_probepool(
                bucket, NOVELS_MAIN, chars, model, mode_id)
        else:
            by_mode[mode_id] = class_overall_charmean(
                per_char, chars, model, mode_id)
    arc = by_mode.get("arc")
    vanilla = by_mode.get("vanilla")
    lift = (arc - vanilla) if (arc is not None and vanilla is not None) else None
    return {"by_mode": by_mode, "arc": arc, "vanilla": vanilla, "lift": lift}


# ── figure ────────────────────────────────────────────────────────

MODEL_DISPLAY = {
    "deepseek/deepseek-v4-flash": "DS-V4-Flash",
    "deepseek/deepseek-v4-pro":   "DS-V4-Pro",
    "Qwen/Qwen3-8B":              "Qwen3-8B",
    "Qwen/Qwen3-32B":             "Qwen3-32B",
    "arcane-8B-dpo":              "ArcANE-8B",
    "arcane-32B-dpo":             "ArcANE-32B",
}

# Matplotlib rc for the paper figure: ACL single column (~3.25in).
def _set_paper_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def make_lift_figure(results: dict, out_path: Path, agg: str = "probepool") -> None:
    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    models = [m for m, _, _ in MODELS_MAIN]
    labels = [MODEL_DISPLAY[m] for m in models]
    x = list(range(len(models)))
    bar_w = 0.36

    central_lifts = [results[m][agg]["central"]["lift"] for m in models]
    support_lifts = [results[m][agg]["supporting"]["lift"] for m in models]

    ax.bar([xi - bar_w / 2 for xi in x], central_lifts, width=bar_w,
           color="#c98c1a", edgecolor="#7a5410", linewidth=0.5,
           label="Central")
    ax.bar([xi + bar_w / 2 for xi in x], support_lifts, width=bar_w,
           color="#f6c84c", edgecolor="#c98c1a", linewidth=0.5,
           label="Supporting")

    ax.axhline(0, color="black", linewidth=0.6, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Arc lift over Vanilla (pts)")
    ymax = max(central_lifts + support_lifts) + 1.5
    ymin = min(central_lifts + support_lifts + [0]) - 0.5
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#bbb", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.4,
              borderpad=0.2, labelspacing=0.2)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    novels, missing = _split_available(NOVELS_MAIN)
    if missing:
        print(f"[build_per_character_analysis] no judge outputs for "
              f"{', '.join(missing)}; analysing {len(novels)} of "
              f"{len(NOVELS_MAIN)} validated novels. The character tiers and "
              f"lifts below are therefore a subset of the paper's.",
              file=sys.stderr)
    if not novels:
        raise SystemExit("no judge outputs found for the validated slice")

    bucket = collect_per_character(JUDGE_TAG, novels, MODELS_MAIN)
    per_char = per_character_means(bucket)
    central, supporting = build_classes(bucket, novels)

    print(f"Central characters ({len(central)}):")
    for n, c in central:
        print(f"  - {n}/{c}")
    print(f"Supporting characters ({len(supporting)}):")
    for n, c in supporting:
        print(f"  - {n}/{c}")

    results: dict = {}
    for model, _, _ in MODELS_MAIN:
        results[model] = {}
        for agg in ("probepool", "charmean"):
            results[model][agg] = {
                "central":    model_lift(bucket, per_char, central, model, agg),
                "supporting": model_lift(bucket, per_char, supporting, model, agg),
            }

    summary = {
        "judge_tag": JUDGE_TAG,
        "novels": novels,
        "central_characters": [{"novel": n, "char": c} for n, c in central],
        "supporting_characters": [{"novel": n, "char": c} for n, c in supporting],
        "results": results,
    }
    (OUT_DIR / "per_character_analysis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'per_character_analysis.json'}")

    # Pretty terminal summary
    print()
    header = f"{'model':<14}{'C-arc':>8}{'C-van':>8}{'C-lift':>8}"\
             f"{'S-arc':>8}{'S-van':>8}{'S-lift':>8}"
    for agg in ("probepool", "charmean"):
        print(f"\n=== aggregation: {agg} ===")
        print(header)
        for model, _, _ in MODELS_MAIN:
            c = results[model][agg]["central"]
            s = results[model][agg]["supporting"]
            def f(v): return f"{v:>8.2f}" if v is not None else "    --  "
            print(f"{MODEL_DISPLAY[model]:<14}"
                  f"{f(c['arc'])}{f(c['vanilla'])}{f(c['lift'])}"
                  f"{f(s['arc'])}{f(s['vanilla'])}{f(s['lift'])}")

    _set_paper_rc()
    make_lift_figure(results, OUT_DIR / "per_character_lift.pdf",
                     agg="probepool")
    print(f"wrote {OUT_DIR / 'per_character_lift.pdf'}")


if __name__ == "__main__":
    main()
