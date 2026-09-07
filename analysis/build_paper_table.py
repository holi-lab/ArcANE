"""Build the main LaTeX table for the paper from eval1 + eval2 records.

Layout (transposed from the first cut):
    rows = 6 models × 6 modes = 36 rows
           (modes: vanilla, summary, rag, lifechoice, timechara, arc)
    cols = 3 categories (In-Scenario, In-world, Out-of-world)
              × 4 metrics (APF, RPF, RAE from eval1; PTF from eval2 = mean of
                           ptf_alignment / ptf_direction / ptf_shape)
         = 12 columns

Within each model's 6-mode block, the max in each metric column is bolded —
so the table answers "for this model, which mode wins on this metric?"

Aggregation:
    1. Within each novel, pool all probe-level scores across characters and
       take the mean per (model, mode, probe_type, metric).
    2. Across the 5 novels, take a simple mean of the novel-level means;
       novels with no records (e.g. Benjamin Franklin has no eval2) are
       skipped from that metric's across-novel mean.

Run:
    uv run python -m analysis.build_paper_table
    uv run python -m analysis.build_paper_table --judge-tag deepseek --print
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "inference"
OUT_DEFAULT  = PROJECT_ROOT / "paper_results" / "main_table.tex"

# Human-validated evaluation slice (paper Tab. 2). Harry Potter is part of the
# paper's slice but no Harry Potter artifact is distributed, so a public clone
# resolves this preset to the remaining four novels and says so on stderr.
NOVELS_MAIN = ["Harry_Potter", "don_quixote", "Benjamin_Franklin",
               "Monte_Cristo", "Anna_Kareina"]

# Held-out low-popularity slice of the paper's Additional Results
# (App. "Additional Novels"): neither novel is in the training pool.
NOVELS_LOWPOP = ["He Knew He Was Right", "The Odd Women"]

# Training-novel reference table. Both titles ARE in the SFT/DPO training pool,
# so ArcANE numbers on them are in-distribution and are reported for reference
# only -- this is NOT the low-popularity control (use NOVELS_LOWPOP for that).
NOVELS_EXTRA3 = ["The Underdogs", "East Lynne"]

# (model_id_in_records, family, size). Family/size are split so the table
# can group by family with a `\multirow` over both sizes — much tidier than
# stuffing "DeepSeek-V4-flash" into one narrow column.
MODELS_MAIN: list[tuple[str, str, str]] = [
    ("deepseek/deepseek-v4-flash", "DeepSeek-V4",   "Flash"),
    ("deepseek/deepseek-v4-pro",   "DeepSeek-V4",   "Pro"),
    ("Qwen/Qwen3-8B",              "Qwen3",         "8B"),
    ("Qwen/Qwen3-32B",             "Qwen3",         "32B"),
    ("arcane-8B-dpo",              "ArcANE",        "8B"),
    ("arcane-32B-dpo",             "ArcANE",        "32B"),
]

# Added baselines + SFT ablation, shown in a supplementary table that
# complements the main 6-model table.
MODELS_ADDED: list[tuple[str, str, str]] = [
    ("HER-32B",                    "HER",           "32B"),
    ("Neph0s/CoSER-Llama-3.1-8B",  "CoSER",         "8B"),
    ("Neph0s/CoSER-Llama-3.1-70B", "CoSER",         "70B"),
    ("arcane-8B-sft",              "ArcANE-SFT",    "8B"),
    ("arcane-32B-sft",             "ArcANE-SFT",    "32B"),
]

# Some models were re-run with a shorter display name; canonicalize on load
# so old and new records merge into one row.
MODEL_ALIASES: dict[str, str] = {
    "ChengyuDu0123/HER-32B": "HER-32B",
}

MODEL_PRESETS = {
    "main": MODELS_MAIN,
    "added": MODELS_ADDED,
}

NOVEL_PRESETS = {
    "main":   NOVELS_MAIN,
    "lowpop": NOVELS_LOWPOP,
    "extra3": NOVELS_EXTRA3,
}

# (mode_id_in_records, display name)
MODES: list[tuple[str, str]] = [
    ("vanilla",    "Vanilla"),
    ("summary",    "Summary"),
    ("rag",        "RAG"),
    ("lifechoice", "LifeChoice"),
    ("timechara", "TimeChara"),
    ("arc",        "Arc"),
]

# (probe_type_in_record, section_header_in_table)
SECTIONS: list[tuple[str, str]] = [
    ("in_text",      "In-Scenario"),
    ("in_world",     "In-world"),
    ("out_of_world", "Out-of-world"),
]

# Per section: which metrics to show, where to read them from.
# source: "eval1" reads scores from eval_records ("apf"/"rpf"/"rae")
#         "eval2" reads scores from traj_records ("ptf_alignment" etc).
EVAL1_DIMS = ("apf", "rpf", "rae")
EVAL2_DIMS = ("ptf_alignment", "ptf_direction", "ptf_shape")

METRICS_PER_SECTION: list[tuple[str, str, tuple[str, ...]]] = [
    ("APF", "eval1", ("apf",)),
    ("RPF", "eval1", ("rpf",)),
    ("RAE", "eval1", ("rae",)),
    ("PTF", "eval2", EVAL2_DIMS),  # mean of the 3 trajectory dims
]


_PROBE_TYPE_RE = re.compile(r"_(intext|inworld|outworld)_")
_TYPE_MAP = {"intext": "in_text", "inworld": "in_world",
             "outworld": "out_of_world"}


def probe_type_of(probe_id: str) -> str | None:
    m = _PROBE_TYPE_RE.search(probe_id or "")
    return _TYPE_MAP.get(m.group(1)) if m else None


# ── loaders ───────────────────────────────────────────────────────

def _iter_jsonl(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def collect(judge_tag: str, novels: list[str],
            models: list[tuple[str, str, str]]) -> dict:
    """Return nested dict:
        scores[novel][model][mode][ptype][metric_label] = list[float]
    """
    target_models = {m for m, _, _ in models}
    target_modes  = {m for m, _ in MODES}
    bucket: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))))

    for novel in novels:
        novel_dir = RESULTS_ROOT / novel
        if not novel_dir.exists():
            continue
        for char_dir in sorted(p for p in novel_dir.iterdir() if p.is_dir()):
            # eval1 ── apf / rpf / rae
            for r in _iter_jsonl(char_dir / f"eval_records_{judge_tag}.jsonl"):
                if not r.get("parse_ok"):
                    continue
                model = MODEL_ALIASES.get(r.get("model"), r.get("model"))
                if (model not in target_models
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
                        bucket[novel][model][r["mode"]][ptype][label].append(
                            sum(vals) / len(vals))

            # eval2 ── ptf trajectory metrics
            for r in _iter_jsonl(char_dir / f"traj_records_{judge_tag}.jsonl"):
                if not r.get("parse_ok"):
                    continue
                model = MODEL_ALIASES.get(r.get("model"), r.get("model"))
                if (model not in target_models
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
                        bucket[novel][model][r["mode"]][ptype][label].append(
                            sum(vals) / len(vals))
    return bucket


def aggregate(bucket: dict, novels: list[str],
              models: list[tuple[str, str, str]]) -> tuple[dict, dict]:
    """Two-step aggregation:
        1. novel mean   = mean over all pooled probe-level scores in the novel
        2. overall mean = simple mean across novels that have a value
    Returns:
        per_novel[novel][model][mode][ptype][metric] -> float | None
        overall[model][mode][ptype][metric]          -> float | None
    """
    per_novel: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(
            lambda: defaultdict(dict))))
    for novel, by_model in bucket.items():
        for model, by_mode in by_model.items():
            for mode, by_ptype in by_mode.items():
                for ptype, by_metric in by_ptype.items():
                    for metric, vals in by_metric.items():
                        per_novel[novel][model][mode][ptype][metric] = (
                            mean(vals) if vals else None)

    overall: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict)))
    for model, _, _ in models:
        for mode, _ in MODES:
            for ptype, _ in SECTIONS:
                for label, _src, _dims in METRICS_PER_SECTION:
                    vals = []
                    for novel in novels:
                        v = (per_novel.get(novel, {}).get(model, {})
                             .get(mode, {}).get(ptype, {}).get(label))
                        if v is not None:
                            vals.append(v)
                    overall[model][mode][ptype][label] = (
                        mean(vals) if vals else None)
    return per_novel, overall


# ── LaTeX rendering ───────────────────────────────────────────────

def _fmt(v: float | None, bold: bool) -> str:
    if v is None:
        return "--"
    s = f"{v:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def _row_cells(overall: dict, model_id: str, mode_id: str
               ) -> list[tuple[str, float | None]]:
    """Build the data cells for one (model, mode) row in render order.

    Returns list of (cell_kind, value) where cell_kind is one of:
        "metric"     — APF/RPF/RAE/PTF in a section
        "overall_avg"— mean of all 12 metrics across the 3 sections
    """
    out: list[tuple[str, float | None]] = []
    all_vals: list[float] = []
    for ptype, _ in SECTIONS:
        for metric, _src, _dims in METRICS_PER_SECTION:
            v = overall[model_id][mode_id][ptype].get(metric)
            out.append(("metric", v))
            if v is not None:
                all_vals.append(v)
    out.append(("overall_avg", mean(all_vals) if all_vals else None))
    return out


def render_latex(overall: dict, judge_tag: str, *,
                 novels: list[str],
                 models: list[tuple[str, str, str]],
                 caption: str,
                 label: str,
                 ours_arcane: bool = False) -> str:
    n_metrics = len(METRICS_PER_SECTION)
    n_sections = len(SECTIONS)
    # 12 metric cols (3 sections × 4) + 1 trailing Overall column.
    n_data_cols = n_sections * n_metrics + 1

    # 3 fixed cols: Family, Size, Mode.
    n_fixed_cols = 3
    col_spec = "l" * n_fixed_cols + "c" * n_data_cols
    n_total_cols = n_fixed_cols + n_data_cols

    # Section header spans + Overall single cell.
    section_headers = " & ".join(
        f"\\multicolumn{{{n_metrics}}}{{c}}{{{name}}}"
        for _, name in SECTIONS)
    # cmidrules under each section span. Data cols start at column 4
    # (after Family/Size/Mode). The Overall col gets its own cmidrule.
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{n_fixed_cols + 1 + i * n_metrics}-"
        f"{n_fixed_cols + (i + 1) * n_metrics}}}"
        for i in range(n_sections))
    cmidrules += (f" \\cmidrule(lr){{{n_total_cols}-{n_total_cols}}}")

    metric_headers = []
    for _ in SECTIONS:
        for m, _src, _dims in METRICS_PER_SECTION:
            metric_headers.append(m)
    metric_headers.append("Overall")

    lines: list[str] = []
    lines.append("% Auto-generated by analysis/build_paper_table.py")
    lines.append(f"% judge={judge_tag}  novels={','.join(novels)}")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    # Plain `tabular` instead of `tabular*` + `\extracolsep{\fill}`: the
    # elastic gap that `\extracolsep` introduces is not reliably tinted
    # by `\rowcolor`/`\cellcolor`, which makes the Arc-row highlight
    # look spotty. We pick a `\tabcolsep` wide enough to get the table
    # close to `\textwidth` without breaking the continuous yellow band.
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\renewcommand{\\arraystretch}{0.85}")
    # Tighten the gap booktabs adds around \toprule/\midrule/\bottomrule.
    lines.append("\\setlength{\\aboverulesep}{0.2ex}")
    lines.append("\\setlength{\\belowrulesep}{0.2ex}")
    lines.append("% \\cellcolor below needs \\usepackage[table]{xcolor} "
                 "(or colortbl).")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("Family & Size & Mode & "
                 f"{section_headers} & Overall \\\\")
    lines.append(cmidrules)
    lines.append(" & & & " + " & ".join(metric_headers) + " \\\\")
    lines.append("\\midrule")

    # Two separator styles:
    #   • between sizes within a family: thin partial rule skipping the
    #     Family column (which is a `\multirow` spanning both sizes)
    #   • between families: `\midrule` so family boundaries pop visually
    sep_between_sizes    = f"\\cmidrule(l){{2-{n_total_cols}}}"
    sep_between_families = "\\midrule"

    # Group consecutive models by family so we can emit one `\multirow`
    # over the family's combined row span.
    i = 0
    while i < len(models):
        family = models[i][1]
        j = i
        while j < len(models) and models[j][1] == family:
            j += 1
        family_models = models[i:j]
        family_rowspan = len(family_models) * len(MODES)
        family_display = (f"\\ours{{{family}}}"
                          if ours_arcane and family == "ArcANE" else family)

        if i > 0:
            lines.append(sep_between_families)

        for size_idx_in_family, (model_id, _family, size_name) \
                in enumerate(family_models):
            # Per-column max within this (family, size)'s 6-mode block.
            rows_cells = [_row_cells(overall, model_id, mode_id)
                          for mode_id, _ in MODES]
            n_cols = len(rows_cells[0])
            col_max: list[float | None] = []
            for c in range(n_cols):
                present = [r[c][1] for r in rows_cells if r[c][1] is not None]
                col_max.append(max(present) if present else None)

            if size_idx_in_family > 0:
                lines.append(sep_between_sizes)

            for mode_idx, (mode_id, mode_name) in enumerate(MODES):
                # Yellow highlight on Arc rows. With plain `tabular`
                # there are no `\extracolsep` rubber gaps, so adjacent
                # `\cellcolor` cells form a continuous band — and the
                # Family/Size columns stay untouched (no overlap with
                # their `\multirow` text).
                hl = ("\\cellcolor{yellow!15}"
                      if mode_id == "arc" else "")
                cells: list[str] = []
                for c, (_kind, v) in enumerate(rows_cells[mode_idx]):
                    mx = col_max[c]
                    is_max = (v is not None and mx is not None
                              and abs(v - mx) < 1e-9)
                    s = _fmt(v, bold=is_max)
                    cells.append(f"{hl} {s}" if hl else s)

                # Top-align `\multirow` so Family/Size text sits near
                # Vanilla (top of the block) — keeps it well away from
                # the Arc row even if a future highlight crept in.
                if size_idx_in_family == 0 and mode_idx == 0:
                    family_cell = (
                        f"\\multirow[t]{{{family_rowspan}}}{{*}}{{{family_display}}}")
                else:
                    family_cell = ""
                if mode_idx == 0:
                    size_cell = (
                        f"\\multirow[t]{{{len(MODES)}}}{{*}}{{{size_name}}}")
                else:
                    size_cell = ""

                mode_cell = f"{hl} {mode_name}" if hl else mode_name
                lines.append(
                    f"{family_cell} & {size_cell} & "
                    f"{mode_cell} & " + " & ".join(cells) + " \\\\")
        i = j

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


# ── terminal summary ──────────────────────────────────────────────

def print_terminal(overall: dict,
                   models: list[tuple[str, str, str]]) -> None:
    col_labels = [(ptype, section, metric)
                  for ptype, section in SECTIONS
                  for metric, _src, _dims in METRICS_PER_SECTION]
    print("\nOVERALL means by (model, mode):")
    header1 = f"  {'model':<24}{'mode':<12}" + "".join(
        f"{s[:6]:>8}" for _p, s, _m in col_labels)
    header2 = f"  {'':<24}{'':<12}" + "".join(
        f"{m:>8}" for _p, _s, m in col_labels)
    for model_id, family, size in models:
        model_name = f"{family} {size}"
        print()
        print(header1)
        print(header2)
        for mode_id, mode_name in MODES:
            cells = []
            for ptype, _section, metric in col_labels:
                v = overall.get(model_id, {}).get(mode_id, {}).get(
                    ptype, {}).get(metric)
                cells.append(f"{v:>8.2f}" if v is not None else f"{'--':>8}")
            print(f"  {model_name:<24}{mode_name:<12}" + "".join(cells))


# ── CLI ───────────────────────────────────────────────────────────

NOVEL_NICE = {
    "Harry_Potter":      "Harry Potter",
    "don_quixote":       "Don Quixote",
    "Benjamin_Franklin": "Benjamin Franklin",
    "Monte_Cristo":      "The Count of Monte Cristo",
    "Anna_Kareina":      "Anna Karenina",
    "Hung_Lou_Meng":     "Dream of the Red Chamber",
    "The Underdogs":     "The Underdogs",
    "East Lynne":        "East Lynne",
    "He Knew He Was Right": "He Knew He Was Right",
    "The Odd Women":     "The Odd Women",
}


def _default_caption(novels: list[str], extended: bool) -> str:
    nice = [NOVEL_NICE.get(n, n.replace("_", " ")) for n in novels]
    if len(nice) == 1:
        novel_str = nice[0]
    elif len(nice) == 2:
        novel_str = f"{nice[0]} and {nice[1]}"
    else:
        novel_str = ", ".join(nice[:-1]) + ", and " + nice[-1]
    extra = (" This supplementary table reports additional RPA baselines "
             "(HER, CoSER) and the SFT-only ablation of our ArcANE model; "
             "compare against Tab.~\\ref{tab:main_results} for the DPO "
             "version of ArcANE." if extended else "")
    return (f"Per-(model, mode) results aggregated over {len(novels)} "
            f"novels ({novel_str}). Probes are categorized by their relation "
            "to the source text: \\textbf{In-Scenario} (probe scenario "
            "appears in the text), \\textbf{In-world} (consistent with the "
            "world but not in text), \\textbf{Out-of-world} "
            "(modern/external). APF / RPF / RAE are per-response judgments "
            "(eval-1); PTF is the mean of trajectory alignment / direction "
            "/ shape (eval-2). \\emph{Overall} averages all twelve metrics. "
            f"For each model, the best mode per column is bolded.{extra}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="analysis.build_paper_table")
    p.add_argument("--judge-tag", default="deepseek",
                   help="judge tag suffix in eval_records_*.jsonl / "
                        "traj_records_*.jsonl (default: deepseek)")
    p.add_argument("--novels", default="main",
                   help="comma-separated novel ids OR a preset name: "
                        f"{','.join(NOVEL_PRESETS)} (default: main)")
    p.add_argument("--models", default="main",
                   help=f"model preset: {','.join(MODEL_PRESETS)} "
                        "(default: main)")
    p.add_argument("--out", default=str(OUT_DEFAULT),
                   help=f"output .tex path (default: {OUT_DEFAULT})")
    p.add_argument("--caption", default=None,
                   help="custom \\caption{...} text; default is auto-built")
    p.add_argument("--label", default="tab:main_results",
                   help="LaTeX \\label{...} (default: tab:main_results)")
    p.add_argument("--ours-arcane", action="store_true",
                   help="wrap the ArcANE family name in \\ours{...}")
    p.add_argument("--strict", action="store_true",
                   help="fail before writing if requested judge files or "
                        "per-novel model/mode/probe-type/metric cells are missing")
    p.add_argument("--print", action="store_true",
                   help="also print the overall (model × mode × metric) "
                        "grid to stdout")
    return p


def _split_available(novels: list[str]) -> tuple[list[str], list[str]]:
    """Split a requested novel list into (present, missing) on disk.

    `results/inference/` does not carry every novel the paper reports: the
    public release ships no Harry Potter artifacts. Silently skipping the gap
    would emit a table whose caption claims more novels than it aggregates, so
    callers report the split and build the caption from what was found.
    """
    present = [n for n in novels if (RESULTS_ROOT / n).is_dir()]
    missing = [n for n in novels if n not in present]
    return present, missing


def _resolve_novels(arg: str) -> list[str]:
    if arg in NOVEL_PRESETS:
        return NOVEL_PRESETS[arg]
    return [s.strip() for s in arg.split(",") if s.strip()]


def _judge_file_gaps(novels: list[str], judge_tag: str,
                     models: list[tuple[str, str, str]]) -> tuple[list[Path], list[Path]]:
    target_models = {model for model, _, _ in models}
    target_modes = {mode for mode, _ in MODES}
    missing, empty = [], []
    for novel in novels:
        for character in sorted(p for p in (RESULTS_ROOT / novel).iterdir()
                                if p.is_dir()):
            for kind, dims in (("eval", EVAL1_DIMS), ("traj", EVAL2_DIMS)):
                path = character / f"{kind}_records_{judge_tag}.jsonl"
                if not path.is_file():
                    missing.append(path.relative_to(RESULTS_ROOT))
                    continue
                if not any(
                    isinstance(row, dict) and row.get("parse_ok")
                    and MODEL_ALIASES.get(row.get("model"), row.get("model")) in target_models
                    and row.get("mode") in target_modes
                    and probe_type_of(row.get("probe_id") or "") is not None
                    and isinstance(row.get("scores"), dict)
                    and any(dim in row["scores"] for dim in dims)
                    for row in _iter_jsonl(path)
                ):
                    empty.append(path.relative_to(RESULTS_ROOT))
    return missing, empty


def _missing_cells(per_novel: dict, novels: list[str], models: list) -> list[tuple]:
    return [
        (novel, model, mode, ptype, metric)
        for novel in novels
        for model, _, _ in models
        for mode, _ in MODES
        for ptype, _ in SECTIONS
        for metric, _, _ in METRICS_PER_SECTION
        if per_novel.get(novel, {}).get(model, {}).get(mode, {}).get(ptype, {})
        .get(metric) is None
    ]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.models not in MODEL_PRESETS:
        raise SystemExit(f"unknown --models preset: {args.models!r}; "
                         f"expected one of {list(MODEL_PRESETS)}")
    models = MODEL_PRESETS[args.models]
    requested = _resolve_novels(args.novels)
    present, _ = _split_available(requested)
    missing_files, empty_files = _judge_file_gaps(present, args.judge_tag, models)
    bucket = collect(args.judge_tag, present, models)
    novels = [n for n in present if bucket.get(n)]
    missing = [n for n in requested if n not in novels]
    if not novels:
        raise SystemExit(f"no usable judge records for tag {args.judge_tag!r} "
                         f"and models {args.models!r} in: {', '.join(requested)} "
                         f"(looked under {RESULTS_ROOT})")
    if missing:
        note = (f"[build_paper_table] {len(missing)} of {len(requested)} "
                f"requested novels have no usable judge outputs for "
                f"{args.judge_tag!r} and are EXCLUDED "
                f"from this table: {', '.join(missing)}.\n"
                f"[build_paper_table] Aggregating {len(novels)} novels: "
                f"{', '.join(novels)}.\n"
                f"[build_paper_table] Numbers therefore differ from the "
                f"paper, which averages over {len(requested)}.")
        if args.strict:
            raise SystemExit(note + "\n[build_paper_table] --strict is set; "
                             "refusing to build a partial table.")
        print(note, file=sys.stderr)
    per_novel, overall = aggregate(bucket, novels, models)
    missing_cells = _missing_cells(per_novel, novels, models)
    gaps = []
    if missing_files:
        examples = "; ".join(str(path) for path in missing_files[:3])
        gaps.append(f"{len(missing_files)} missing judge file(s): {examples}")
    if empty_files:
        examples = "; ".join(str(path) for path in empty_files[:3])
        gaps.append(f"{len(empty_files)} judge file(s) have no usable selected-model records: {examples}")
    if missing_cells:
        examples = "; ".join(" / ".join(cell) for cell in missing_cells[:3])
        gaps.append(f"{len(missing_cells)} missing per-novel metric cell(s): {examples}")
    if gaps:
        note = "[build_paper_table] " + ". ".join(gaps)
        if args.strict:
            raise SystemExit(note + "\n[build_paper_table] --strict is set; "
                             "refusing to build a partial table.")
        print(note, file=sys.stderr)
    caption = args.caption or _default_caption(novels, extended=args.models == "added")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_latex(
        overall, args.judge_tag, novels=novels, models=models,
        caption=caption, label=args.label, ours_arcane=args.ours_arcane,
    ), encoding="utf-8")
    print(f"wrote {out_path}")

    # Numeric dump for sanity checks.
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "judge_tag": args.judge_tag,
        "novels": novels,
        "models": [m for m, _, _ in models],
        "modes":  [m for m, _ in MODES],
        "per_novel": {
            n: {m: {md: {p: dict(per_novel.get(n, {}).get(m, {})
                                  .get(md, {}).get(p, {}))
                          for p, _ in SECTIONS}
                     for md, _ in MODES}
                for m, _, _ in models}
            for n in novels},
        "overall": {m: {md: {p: dict(overall.get(m, {}).get(md, {}).get(p, {}))
                              for p, _ in SECTIONS}
                         for md, _ in MODES}
                    for m, _, _ in models},
    }, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")

    if args.__dict__["print"]:
        print_terminal(overall, models)


if __name__ == "__main__":
    main()
