#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


DEFAULT_DATASET = Path("../dataset/scdbench.jsonl")
DEFAULT_OUTPUT_JSON = Path("runs/first_three_metrics_summary.json")
DEFAULT_OUTPUT_MD = Path("runs/first_three_metrics_summary.md")

MODEL_RUNS = [
    {
        "slug": "claude_opus47_reasoning_high",
        "label": "Claude Opus 4.7 high",
    },
    {
        "slug": "gpt53_codex_reasoning_high",
        "label": "GPT-5.3 Codex high",
    },
    {
        "slug": "glm5_reasoning_high",
        "label": "GLM-5 high",
    },
    {
        "slug": "glm5_reasoning_none",
        "label": "GLM-5 none",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize SCDBench format, compilation, and ABI reconstruction metrics "
            "before and after zero-shot compilation repair."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--prefer-format-compile",
        action="store_true",
        help=(
            "Use existing eval_format_compile_* files before eval_first3_* files. "
            "This is useful for quick compilation-only checks, but ABI metrics will be absent."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                yield json.loads(raw)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def dedupe_latest_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        bid = row.get("benchmark_id")
        if bid:
            rows[str(bid)] = row
    return rows


def candidate_paths(runs_dir: Path, slug: str, repair: bool, prefer_format_compile: bool) -> list[Path]:
    if repair:
        full = runs_dir / f"eval_first3_repair_{slug}.jsonl"
        compile_only = runs_dir / f"eval_format_compile_repair_{slug}.jsonl"
        candidates = [compile_only, full] if prefer_format_compile else [full, compile_only]
    else:
        full = runs_dir / f"eval_first3_{slug}.jsonl"
        compile_only = runs_dir / f"eval_format_compile_{slug}.jsonl"
        legacy = runs_dir / f"eval_model_{slug}.jsonl"
        candidates = [compile_only, full, legacy] if prefer_format_compile else [full, legacy, compile_only]
    return candidates


def select_existing_path(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def group_by(records: Sequence[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record)
    return dict(groups)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def summarize_group(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    format_ok = sum(1 for row in records if (row.get("format") or {}).get("ok"))
    compile_ok = sum(1 for row in records if (row.get("compilation") or {}).get("success"))
    abi_rows = [row.get("abi") or {} for row in records if (row.get("abi") or {}).get("evaluated")]
    semantic_eligible = sum(1 for row in records if (row.get("semantic") or {}).get("eligible"))

    tp = sum(int(row.get("true_positive") or 0) for row in abi_rows)
    fp = sum(int(row.get("false_positive") or 0) for row in abi_rows)
    fn = sum(int(row.get("false_negative") or 0) for row in abi_rows)
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    return {
        "contracts": total,
        "format_ok": format_ok,
        "format_rate": format_ok / total if total else 0.0,
        "compile_ok": compile_ok,
        "compile_rate_all": compile_ok / total if total else 0.0,
        "compile_rate_format_eligible": compile_ok / format_ok if format_ok else 0.0,
        "abi_evaluated": len(abi_rows),
        "abi_macro_precision": mean(float(row.get("precision") or 0.0) for row in abi_rows),
        "abi_macro_recall": mean(float(row.get("recall") or 0.0) for row in abi_rows),
        "abi_macro_f1": mean(float(row.get("f1") or 0.0) for row in abi_rows),
        "abi_micro_precision": micro_precision,
        "abi_micro_recall": micro_recall,
        "abi_micro_f1": micro_f1,
        "semantic_eligible": semantic_eligible,
    }


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(records),
        "overall": summarize_group(records),
        "by_difficulty_bin": {
            diff: summarize_group(group)
            for diff, group in sorted(
                group_by(records, lambda row: str(row.get("difficulty_bin") or "unknown")).items()
            )
        },
    }


def ordered_records(dataset_rows: Sequence[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in dataset_rows:
        bid = str(row.get("benchmark_id") or "")
        record = by_id.get(bid)
        if record is None:
            missing.append(bid)
        else:
            records.append(record)
    return records, missing


def combine_after_repair(
    dataset_rows: Sequence[dict[str, Any]],
    before_by_id: dict[str, dict[str, Any]],
    repair_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    combined_by_id = dict(before_by_id)
    replaced_ids = sorted(set(before_by_id) & set(repair_by_id))
    for bid in replaced_ids:
        combined_by_id[bid] = repair_by_id[bid]

    records, missing = ordered_records(dataset_rows, combined_by_id)
    recovered_compile = 0
    lost_compile = 0
    recovered_abi = 0
    lost_abi = 0
    for bid in replaced_ids:
        before = before_by_id.get(bid) or {}
        after = repair_by_id.get(bid) or {}
        before_compile = bool((before.get("compilation") or {}).get("success"))
        after_compile = bool((after.get("compilation") or {}).get("success"))
        before_abi = bool((before.get("abi") or {}).get("evaluated"))
        after_abi = bool((after.get("abi") or {}).get("evaluated"))
        if not before_compile and after_compile:
            recovered_compile += 1
        if before_compile and not after_compile:
            lost_compile += 1
        if not before_abi and after_abi:
            recovered_abi += 1
        if before_abi and not after_abi:
            lost_abi += 1

    return records, {
        "replaced": len(replaced_ids),
        "missing_after_combination": missing,
        "recovered_compile": recovered_compile,
        "lost_compile": lost_compile,
        "recovered_abi": recovered_abi,
        "lost_abi": lost_abi,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def delta_pp(after: float, before: float) -> str:
    delta = 100.0 * (after - before)
    return f"{delta:+.1f}"


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def stage_row(model: str, stage: str, summary: dict[str, Any], before: dict[str, Any] | None = None) -> list[str]:
    overall = summary["overall"]
    before_overall = before["overall"] if before else overall
    return [
        model,
        stage,
        f"{overall['format_ok']}/{overall['contracts']} ({pct(overall['format_rate'])})",
        f"{overall['compile_ok']}/{overall['contracts']} ({pct(overall['compile_rate_all'])})",
        delta_pp(overall["compile_rate_all"], before_overall["compile_rate_all"]),
        f"{overall['abi_evaluated']}",
        f"{overall['abi_micro_precision']:.3f}",
        f"{overall['abi_micro_recall']:.3f}",
        f"{overall['abi_micro_f1']:.3f}",
        f"{overall['abi_macro_f1']:.3f}",
    ]


def build_markdown(result: dict[str, Any]) -> str:
    lines = ["# First Three Metrics Summary", ""]
    lines.append(
        "The after-repair rows are computed over the same benchmark contracts as the before rows: "
        "a repaired evaluation replaces the corresponding original evaluation when available, "
        "and all other contracts keep their original evaluation."
    )
    lines.append("")

    main_rows: list[list[str]] = []
    repair_rows: list[list[str]] = []
    for run in result["models"]:
        before = run["before"]["summary"]
        after = run["after_repair_all"]["summary"]
        main_rows.append(stage_row(run["label"], "before", before))
        main_rows.append(stage_row(run["label"], "after repair", after, before))
        repair_subset = run["repair_subset"]["summary"]
        repair_overall = repair_subset["overall"]
        replacement = run["after_repair_all"]["replacement"]
        repair_rows.append(
            [
                run["label"],
                str(replacement["replaced"]),
                f"{repair_overall['format_ok']}/{repair_overall['contracts']} ({pct(repair_overall['format_rate'])})",
                f"{repair_overall['compile_ok']}/{repair_overall['contracts']} ({pct(repair_overall['compile_rate_all'])})",
                str(replacement["recovered_compile"]),
                str(replacement["lost_compile"]),
                str(replacement["recovered_abi"]),
            ]
        )

    lines.append("## All-Contract Results")
    lines.append(
        table(
            [
                "Model",
                "Stage",
                "Format OK",
                "Compile OK",
                "Compile Δ pp",
                "ABI eval",
                "ABI micro P",
                "ABI micro R",
                "ABI micro F1",
                "ABI macro F1",
            ],
            main_rows,
        )
    )
    lines.append("")

    lines.append("## Repair Subset")
    lines.append(
        table(
            [
                "Model",
                "Repairs",
                "Repair format OK",
                "Repair compile OK",
                "Recovered compiles",
                "Lost compiles",
                "Recovered ABI evals",
            ],
            repair_rows,
        )
    )
    lines.append("")

    for diff in ("easy", "medium", "hard"):
        rows: list[list[str]] = []
        for run in result["models"]:
            before = run["before"]["summary"]["by_difficulty_bin"].get(diff, {"contracts": 0})
            after = run["after_repair_all"]["summary"]["by_difficulty_bin"].get(diff, {"contracts": 0})
            if not before.get("contracts") and not after.get("contracts"):
                continue
            rows.append(
                [
                    run["label"],
                    f"{before.get('compile_ok', 0)}/{before.get('contracts', 0)} ({pct(before.get('compile_rate_all', 0.0))})",
                    f"{after.get('compile_ok', 0)}/{after.get('contracts', 0)} ({pct(after.get('compile_rate_all', 0.0))})",
                    delta_pp(after.get("compile_rate_all", 0.0), before.get("compile_rate_all", 0.0)),
                    f"{before.get('abi_micro_f1', 0.0):.3f}",
                    f"{after.get('abi_micro_f1', 0.0):.3f}",
                ]
            )
        lines.append(f"## {diff.title()} Difficulty")
        lines.append(table(["Model", "Compile before", "Compile after", "Δ pp", "ABI micro F1 before", "ABI micro F1 after"], rows))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    dataset_rows = load_dataset(args.dataset)
    result: dict[str, Any] = {
        "dataset": str(args.dataset),
        "contracts": len(dataset_rows),
        "models": [],
    }

    for run in MODEL_RUNS:
        slug = run["slug"]
        before_path = select_existing_path(candidate_paths(args.runs_dir, slug, repair=False, prefer_format_compile=args.prefer_format_compile))
        repair_path = select_existing_path(candidate_paths(args.runs_dir, slug, repair=True, prefer_format_compile=args.prefer_format_compile))
        if before_path is None:
            raise SystemExit(f"missing original evaluation for {slug}")
        if repair_path is None:
            raise SystemExit(f"missing repair evaluation for {slug}")

        before_by_id = dedupe_latest_by_id(before_path)
        repair_by_id = dedupe_latest_by_id(repair_path)
        before_records, before_missing = ordered_records(dataset_rows, before_by_id)
        repair_records = list(repair_by_id.values())
        after_records, replacement = combine_after_repair(dataset_rows, before_by_id, repair_by_id)

        result["models"].append(
            {
                "slug": slug,
                "label": run["label"],
                "before": {
                    "path": str(before_path),
                    "missing": before_missing,
                    "summary": summarize_records(before_records),
                },
                "repair_subset": {
                    "path": str(repair_path),
                    "summary": summarize_records(repair_records),
                },
                "after_repair_all": {
                    "replacement": replacement,
                    "summary": summarize_records(after_records),
                },
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(result), encoding="utf-8")
    print(f"[summary] wrote {args.output_json}")
    print(f"[summary] wrote {args.output_md}")


if __name__ == "__main__":
    main()
