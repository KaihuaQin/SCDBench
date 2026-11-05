from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT.parent / "dataset/scdbench.jsonl"

RUNS: List[Tuple[str, Path]] = [
    ("Opus 4.7", ROOT / "runs/semantic_prep/eval_semanticprep_claude_opus47_reasoning_high_before.jsonl"),
    ("Opus 4.7$^\\dagger$", ROOT / "runs/semantic_prep/eval_semanticprep_claude_opus47_reasoning_high_after_repair.jsonl"),
    ("GPT-5.3-Codex", ROOT / "runs/semantic_prep/eval_semanticprep_gpt53_codex_reasoning_high_before.jsonl"),
    ("GPT-5.3-Codex$^\\dagger$", ROOT / "runs/semantic_prep/eval_semanticprep_gpt53_codex_reasoning_high_after_repair.jsonl"),
    ("GLM-5", ROOT / "runs/semantic_prep/eval_semanticprep_glm5_reasoning_high_before.jsonl"),
    ("GLM-5$^\\dagger$", ROOT / "runs/semantic_prep/eval_semanticprep_glm5_reasoning_high_after_repair.jsonl"),
    ("GLM-5 (instruct)", ROOT / "runs/semantic_prep/eval_semanticprep_glm5_reasoning_none_before.jsonl"),
    ("GLM-5 (instruct)$^\\dagger$", ROOT / "runs/semantic_prep/eval_semanticprep_glm5_reasoning_none_after_repair.jsonl"),
]

BEST_ABI_MODEL = "GPT-5.3-Codex$^\\dagger$"
DIFFICULTIES = [("easy", "Easy"), ("medium", "Medium"), ("hard", "Hard")]
METRICS = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]

BINS = [
    ("zero", "0", 0.0, 0.0),
    ("low", "(0,0.5)", 0.0, 0.5),
    ("mid", "[0.5,0.9)", 0.5, 0.9),
    ("high", "[0.9,1)", 0.9, 1.0),
    ("perfect", "1", 1.0, 1.0),
]

COLORS = {
    "zero": "#d9d9d9",
    "low": "#f4b6b6",
    "mid": "#ffd58a",
    "high": "#a8d5ff",
    "perfect": "#7bc87c",
}


def empty_metric_bins() -> Dict[str, Dict[str, int]]:
    return {
        metric: {name: 0 for name, _, _, _ in BINS}
        for metric, _ in METRICS
    }


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_metadata() -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(DATASET):
        functions = [
            entry
            for entry in (row.get("test_cases") or {}).values()
            if isinstance(entry, dict) and entry.get("entry_type") == "function"
        ]
        meta[row["benchmark_id"]] = {
            "difficulty": row.get("difficulty_bin", ""),
            "ground_truth_count": len(functions),
        }
    return meta


def div(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def f1(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def bin_name(value: float) -> str:
    if value == 0:
        return "zero"
    if value < 0.5:
        return "low"
    if value < 0.9:
        return "mid"
    if value < 1:
        return "high"
    return "perfect"


def escape_tex(text: str) -> str:
    dagger_token = "SCDDAGGERMARK"
    text = text.replace("$^\\dagger$", dagger_token)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace(dagger_token, "\\ensuremath{^{\\dagger}}")


def escape_svg(text: str) -> str:
    return (
        text.replace("$^\\dagger$", "&#8224;")
        .replace("&", "&amp;")
        .replace("&amp;#8224;", "&#8224;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def compute_rows() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = load_metadata()
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"models": []}

    for model, path in RUNS:
        metric_bins = empty_metric_bins()
        by_difficulty = {
            diff: {
                "difficulty": diff,
                "contracts": 0,
                "abi_evaluated_contracts": 0,
                "perfect_abi_contracts": 0,
                "bins": empty_metric_bins(),
            }
            for diff, _ in DIFFICULTIES
        }
        perfect_contracts = 0
        evaluated_contracts = 0

        for row in read_jsonl(path):
            benchmark_id = row["benchmark_id"]
            info = meta[benchmark_id]
            difficulty = str(info["difficulty"])
            abi = row.get("abi") or {}
            evaluated = bool(abi.get("evaluated"))
            gt = int(info["ground_truth_count"])
            tp = int(abi.get("true_positive") or 0) if evaluated else 0
            fp = int(abi.get("false_positive") or 0) if evaluated else 0

            precision = div(tp, tp + fp)
            recall = div(tp, gt)
            score = f1(precision, recall)
            is_perfect = gt > 0 and tp == gt and fp == 0
            if evaluated:
                evaluated_contracts += 1
            if is_perfect:
                perfect_contracts += 1
            if difficulty in by_difficulty:
                by_difficulty[difficulty]["contracts"] += 1
                if evaluated:
                    by_difficulty[difficulty]["abi_evaluated_contracts"] += 1
                if is_perfect:
                    by_difficulty[difficulty]["perfect_abi_contracts"] += 1

            for metric, value in (
                ("precision", precision),
                ("recall", recall),
                ("f1", score),
            ):
                bucket = bin_name(value)
                metric_bins[metric][bucket] += 1
                if difficulty in by_difficulty:
                    by_difficulty[difficulty]["bins"][metric][bucket] += 1

            rows.append(
                {
                    "model": model,
                    "benchmark_id": benchmark_id,
                    "difficulty": info["difficulty"],
                    "ground_truth_count": gt,
                    "evaluated": evaluated,
                    "true_positive": tp,
                    "false_positive": fp,
                    "precision": precision,
                    "recall": recall,
                    "f1": score,
                    "perfect_abi": is_perfect,
                }
            )

        summary["models"].append(
            {
                "model": model,
                "contracts": 600,
                "abi_evaluated_contracts": evaluated_contracts,
                "perfect_abi_contracts": perfect_contracts,
                "perfect_abi_rate": perfect_contracts / 600,
                "bins": metric_bins,
                "by_difficulty": by_difficulty,
            }
        )

    return rows, summary


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "benchmark_id",
        "difficulty",
        "ground_truth_count",
        "evaluated",
        "true_positive",
        "false_positive",
        "precision",
        "recall",
        "f1",
        "perfect_abi",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tikz(
    summary: Dict[str, Any],
    path: Path,
    denominator: int = 600,
    title_prefix: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    models = summary["models"]
    row_h = 0.34
    bar_w = 5.0
    panel_gap = 0.55
    x0 = 2.95
    lines = [
        "% Generated by scripts/summarize_abi_distribution.py",
        "\\begin{tikzpicture}[x=1cm,y=1cm,font=\\scriptsize]",
        "\\definecolor{abiZero}{HTML}{D9D9D9}",
        "\\definecolor{abiLow}{HTML}{F4B6B6}",
        "\\definecolor{abiMid}{HTML}{FFD58A}",
        "\\definecolor{abiHigh}{HTML}{A8D5FF}",
        "\\definecolor{abiPerfect}{HTML}{7BC87C}",
        "\\def\\barw{5.0}",
    ]
    color_map = {
        "zero": "abiZero",
        "low": "abiLow",
        "mid": "abiMid",
        "high": "abiHigh",
        "perfect": "abiPerfect",
    }
    for panel_idx, (metric, title) in enumerate(METRICS):
        y_base = -panel_idx * (len(models) * row_h + panel_gap)
        panel_title = f"{title_prefix} {title}".strip()
        lines.append(f"\\node[anchor=west,font=\\bfseries] at (0,{y_base + 0.22:.3f}) {{{escape_tex(panel_title)}}};")
        for i, model in enumerate(models):
            y = y_base - (i + 1) * row_h
            if panel_idx == 0:
                lines.append(f"\\node[anchor=east] at ({x0 - 0.12:.3f},{y + 0.09:.3f}) {{{escape_tex(model['model'])}}};")
            x = x0
            for name, _, _, _ in BINS:
                count = model["bins"][metric][name]
                width = bar_w * count / denominator
                lines.append(
                    f"\\fill[{color_map[name]}] ({x:.3f},{y:.3f}) rectangle ({x + width:.3f},{y + 0.18:.3f});"
                )
                x += width
            perfect = model["bins"][metric]["perfect"]
            lines.append(f"\\node[anchor=west] at ({x0 + bar_w + 0.1:.3f},{y + 0.09:.3f}) {{{perfect}}};")
        lines.append(f"\\draw[black!30] ({x0:.3f},{y_base - (len(models)+1)*row_h + 0.12:.3f}) -- ({x0 + bar_w:.3f},{y_base - (len(models)+1)*row_h + 0.12:.3f});")
    legend_y = -len(METRICS) * (len(models) * row_h + panel_gap) + 0.20
    legend_x = x0
    lines.append(f"\\node[anchor=east] at ({legend_x - 0.12:.3f},{legend_y + 0.09:.3f}) {{Bins}};")
    for idx, (name, label, _, _) in enumerate(BINS):
        x = legend_x + idx * 1.15
        lines.append(f"\\fill[{color_map[name]}] ({x:.3f},{legend_y:.3f}) rectangle ({x + 0.22:.3f},{legend_y + 0.18:.3f});")
        lines.append(f"\\node[anchor=west] at ({x + 0.26:.3f},{legend_y + 0.09:.3f}) {{{label}}};")
    lines.append(f"\\node[anchor=west] at ({x0 + bar_w + 0.1:.3f},{0.22:.3f}) {{perfect count}};")
    lines.append("\\end{tikzpicture}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(
    summary: Dict[str, Any],
    path: Path,
    denominator: int = 600,
    title_prefix: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    models = summary["models"]
    width = 1040
    row_h = 22
    panel_gap = 38
    left = 205
    bar_w = 560
    top = 34
    height = top + len(METRICS) * (len(models) * row_h + panel_gap) + 44
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-weight:700;font-size:14px}</style>',
    ]
    for pidx, (metric, title) in enumerate(METRICS):
        y0 = top + pidx * (len(models) * row_h + panel_gap)
        panel_title = f"{title_prefix} {title}".strip()
        parts.append(f'<text class="title" x="20" y="{y0 - 10}">{escape_svg(panel_title)}</text>')
        for i, model in enumerate(models):
            y = y0 + i * row_h
            if pidx == 0:
                parts.append(f'<text x="{left - 10}" y="{y + 13}" text-anchor="end">{escape_svg(model["model"])}</text>')
            x = left
            for name, label, _, _ in BINS:
                count = model["bins"][metric][name]
                w = bar_w * count / denominator
                parts.append(f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="15" fill="{COLORS[name]}"/>')
                x += w
            perfect = model["bins"][metric]["perfect"]
            parts.append(f'<text x="{left + bar_w + 10}" y="{y + 12}">{perfect}</text>')
    legend_y = height - 28
    x = left
    for name, label, _, _ in BINS:
        parts.append(f'<rect x="{x}" y="{legend_y}" width="18" height="14" fill="{COLORS[name]}"/>')
        parts.append(f'<text x="{x + 24}" y="{legend_y + 12}">{label}</text>')
        x += 100
    parts.append(f'<text x="{left + bar_w + 10}" y="{top - 10}">perfect count</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def difficulty_summary(summary: Dict[str, Any], difficulty: str) -> Dict[str, Any]:
    return {
        "models": [
            {
                "model": model["model"],
                "contracts": model["by_difficulty"][difficulty]["contracts"],
                "abi_evaluated_contracts": model["by_difficulty"][difficulty]["abi_evaluated_contracts"],
                "perfect_abi_contracts": model["by_difficulty"][difficulty]["perfect_abi_contracts"],
                "bins": model["by_difficulty"][difficulty]["bins"],
            }
            for model in summary["models"]
        ]
    }


def write_tikz_difficulty_grid(summary: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_h = 0.36
    bar_w = 2.00
    col_gap = 0.82
    panel_gap = 0.72
    label_x = 2.95
    x0 = 3.15
    models = summary["models"]
    columns = DIFFICULTIES + [("total", "Total")]
    plot_right_x = x0 + (len(columns) - 1) * (bar_w + col_gap) + bar_w + 0.72
    color_map = {
        "zero": "abiZero",
        "low": "abiLow",
        "mid": "abiMid",
        "high": "abiHigh",
        "perfect": "abiPerfect",
    }
    lines = [
        "% Generated by scripts/summarize_abi_distribution.py",
        "\\begin{tikzpicture}[x=1cm,y=1cm,font=\\fontsize{7.0pt}{7.8pt}\\selectfont]",
        "\\definecolor{abiZero}{HTML}{D9D9D9}",
        "\\definecolor{abiLow}{HTML}{F4B6B6}",
        "\\definecolor{abiMid}{HTML}{FFD58A}",
        "\\definecolor{abiHigh}{HTML}{A8D5FF}",
        "\\definecolor{abiPerfect}{HTML}{7BC87C}",
        "\\definecolor{abiBest}{HTML}{E6F4EA}",
    ]
    for col_idx, (column, diff_label) in enumerate(columns):
        x = x0 + col_idx * (bar_w + col_gap)
        lines.append(
            f"\\node[anchor=center,font=\\bfseries] at ({x + bar_w / 2:.3f},0.340) {{{escape_tex(diff_label)}}};"
        )
        contract_count = 600 if column == "total" else 200
        lines.append(
            f"\\node[anchor=center] at ({x + bar_w / 2:.3f},0.110) {{{contract_count} contracts}};"
        )
    for panel_idx, (metric, metric_label) in enumerate(METRICS):
        y_base = -panel_idx * (len(models) * row_h + panel_gap)
        lines.append(f"\\node[anchor=west,font=\\bfseries] at (0,{y_base + 0.02:.3f}) {{{escape_tex(metric_label)}}};")
        for i, model in enumerate(models):
            y = y_base - (i + 1) * row_h
            is_best = model["model"] == BEST_ABI_MODEL
            if is_best:
                lines.append(
                    f"\\fill[abiBest] (0.000,{y - 0.055:.3f}) rectangle ({plot_right_x:.3f},{y + 0.225:.3f});"
                )
            label_font = ",font=\\bfseries" if is_best else ""
            lines.append(f"\\node[anchor=east{label_font}] at ({label_x:.3f},{y + 0.085:.3f}) {{{escape_tex(model['model'])}}};")
            for col_idx, (column, _) in enumerate(columns):
                x = x0 + col_idx * (bar_w + col_gap)
                cursor = x
                denominator = 600 if column == "total" else 200
                split_bins = (
                    model["bins"][metric]
                    if column == "total"
                    else model["by_difficulty"][column]["bins"][metric]
                )
                for name, _, _, _ in BINS:
                    count = split_bins[name]
                    width = bar_w * count / denominator
                    lines.append(
                        f"\\fill[{color_map[name]}] ({cursor:.3f},{y:.3f}) rectangle ({cursor + width:.3f},{y + 0.170:.3f});"
                    )
                    cursor += width
                perfect = split_bins["perfect"]
                lines.append(f"\\node[anchor=west] at ({x + bar_w + 0.055:.3f},{y + 0.085:.3f}) {{{perfect}}};")
        for col_idx, _ in enumerate(columns):
            x = x0 + col_idx * (bar_w + col_gap)
            lines.append(
                f"\\draw[black!25] ({x:.3f},{y_base - (len(models)+1)*row_h + 0.120:.3f}) -- ({x + bar_w:.3f},{y_base - (len(models)+1)*row_h + 0.120:.3f});"
            )
    legend_y = -len(METRICS) * (len(models) * row_h + panel_gap) + 0.18
    legend_x = x0
    lines.append(f"\\node[anchor=east] at ({legend_x - 0.12:.3f},{legend_y + 0.085:.3f}) {{Bins}};")
    for idx, (name, label, _, _) in enumerate(BINS):
        x = legend_x + idx * 1.35
        lines.append(f"\\fill[{color_map[name]}] ({x:.3f},{legend_y:.3f}) rectangle ({x + 0.200:.3f},{legend_y + 0.170:.3f});")
        lines.append(f"\\node[anchor=west,font=\\fontsize{{6.3pt}}{{7.0pt}}\\selectfont] at ({x + 0.240:.3f},{legend_y + 0.085:.3f}) {{{label}}};")
    right_x = x0 + len(columns) * bar_w + (len(columns) - 1) * col_gap
    lines.append(f"\\node[anchor=west] at ({right_x + 0.055:.3f},{0.110:.3f}) {{perfect count}};")
    lines.append("\\end{tikzpicture}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# ABI Distribution Summary",
        "",
        "Per-contract precision uses zero when a model predicts no ABI signatures. A contract is counted as perfectly ABI recovered only when it has at least one ground-truth ABI function, all ground-truth signatures are recovered, and no extra signatures are produced.",
        "",
        "| Model | Perfect ABI contracts | Perfect rate | F1=0 contracts | F1 in [0.9,1) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in summary["models"]:
        bins = model["bins"]["f1"]
        lines.append(
            f"| {model['model']} | {model['perfect_abi_contracts']}/600 | {model['perfect_abi_rate']*100:.1f}% | {bins['zero']} | {bins['high']} |"
        )
    lines.extend(
        [
            "",
            "## Perfect ABI Recovery by Difficulty",
            "",
            "Each difficulty split contains 200 contracts.",
            "",
            "| Model | Easy | Medium | Hard |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for model in summary["models"]:
        counts = []
        for difficulty, _ in DIFFICULTIES:
            split = model["by_difficulty"][difficulty]
            count = split["bins"]["f1"]["perfect"]
            counts.append(f"{count}/200 ({count / 2:.1f}%)")
        lines.append(f"| {model['model']} | {counts[0]} | {counts[1]} | {counts[2]} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows, summary = compute_rows()
    write_csv(rows, ROOT / "runs/abi_per_contract_metrics.csv")
    (ROOT / "runs/abi_distribution_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_summary_md(summary, ROOT / "runs/abi_distribution_summary.md")
    write_tikz_difficulty_grid(
        summary, ROOT / "latex/figures/abi_metric_distribution_by_difficulty.tex"
    )
    print(json.dumps({
        "csv": "runs/abi_per_contract_metrics.csv",
        "summary_json": "runs/abi_distribution_summary.json",
        "summary_md": "runs/abi_distribution_summary.md",
        "by_difficulty_grid": "latex/figures/abi_metric_distribution_by_difficulty.tex",
    }, indent=2))


if __name__ == "__main__":
    main()
