#!/usr/bin/env python3

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from summarize_compilation_errors import compilation_category
except ImportError:
    from scripts.summarize_compilation_errors import compilation_category


ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    (
        "Opus 4.7",
        ROOT / "runs/semantic_prep/eval_semanticprep_claude_opus47_reasoning_high_before.jsonl",
        ROOT / "runs/semantic_prep/eval_semanticprep_claude_opus47_reasoning_high_after_repair.jsonl",
    ),
    (
        "GPT-5.3-Codex",
        ROOT / "runs/semantic_prep/eval_semanticprep_gpt53_codex_reasoning_high_before.jsonl",
        ROOT / "runs/semantic_prep/eval_semanticprep_gpt53_codex_reasoning_high_after_repair.jsonl",
    ),
    (
        "GLM-5",
        ROOT / "runs/semantic_prep/eval_semanticprep_glm5_reasoning_high_before.jsonl",
        ROOT / "runs/semantic_prep/eval_semanticprep_glm5_reasoning_high_after_repair.jsonl",
    ),
    (
        "GLM-5 (instruct)",
        ROOT / "runs/semantic_prep/eval_semanticprep_glm5_reasoning_none_before.jsonl",
        ROOT / "runs/semantic_prep/eval_semanticprep_glm5_reasoning_none_after_repair.jsonl",
    ),
]

DIFFICULTIES = ("easy", "medium", "hard")

CATEGORY_ORDER = [
    "Invalid address literal/checksum",
    "Duplicate declarations/name collisions",
    "Type/mutability errors",
    "Unresolved identifiers/members",
    "Parser errors",
    "Invalid compiler settings",
    "Stack too deep",
    "Inheritance/override/abstractness",
    "Syntax errors",
    "Target artifact/bytecode errors",
    "Other solc errors",
    "Other",
    "Unknown compiler failure",
]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                yield json.loads(raw)


def by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["benchmark_id"]): row for row in read_jsonl(path)}


def summarize() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model, before_path, after_path in RUNS:
        before = by_id(before_path)
        after = by_id(after_path)
        rows = []
        by_category: dict[str, dict[str, Any]] = {}

        for bid, row in before.items():
            fmt = row.get("format") or {}
            comp = row.get("compilation") or {}
            if not fmt.get("ok") or comp.get("success"):
                continue

            category = compilation_category(str(comp.get("error") or ""))
            diff = str(row.get("difficulty_bin") or "unknown")
            fixed = bool((after.get(bid) or {}).get("compilation", {}).get("success"))

            bucket = by_category.setdefault(
                category,
                {
                    "easy": 0,
                    "medium": 0,
                    "hard": 0,
                    "easy_fixed": 0,
                    "medium_fixed": 0,
                    "hard_fixed": 0,
                    "total": 0,
                    "fixed": 0,
                },
            )
            if diff in DIFFICULTIES:
                bucket[diff] += 1
                if fixed:
                    bucket[f"{diff}_fixed"] += 1
            bucket["total"] += 1
            if fixed:
                bucket["fixed"] += 1
            rows.append({"benchmark_id": bid, "difficulty": diff, "category": category, "fixed": fixed})

        ordered_categories = [c for c in CATEGORY_ORDER if c in by_category]
        ordered_categories.extend(sorted(c for c in by_category if c not in CATEGORY_ORDER))
        out[model] = {
            "compile_failures_after_format": len(rows),
            "fixed_after_repair": sum(1 for row in rows if row["fixed"]),
            "categories": {category: by_category[category] for category in ordered_categories},
        }
    return out


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    def cell(total: int, fixed: int) -> str:
        if total == 0 and fixed == 0:
            return "--"
        return f"{total} ({fixed})"

    lines = [
        "# Compilation Failure Reasons and Repair Fixes",
        "",
        "Rows include before-repair outputs that passed format validation but failed Solidity compilation. `Fixed` counts failures that compile after one same-model repair attempt.",
        "",
        "| Model | Failure reason | Easy | Medium | Hard | Total |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model, payload in summary.items():
        first = True
        for category, counts in payload["categories"].items():
            label = model if first else ""
            first = False
            lines.append(
                f"| {label} | {category} | "
                f"{cell(counts['easy'], counts['easy_fixed'])} | "
                f"{cell(counts['medium'], counts['medium_fixed'])} | "
                f"{cell(counts['hard'], counts['hard_fixed'])} | "
                f"{cell(counts['total'], counts['fixed'])} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = summarize()
    json_path = ROOT / "runs/compilation_failure_repair_summary.json"
    md_path = ROOT / "runs/compilation_failure_repair_summary.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, md_path)
    print(json.dumps({"json": str(json_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
