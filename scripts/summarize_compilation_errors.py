#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_INPUTS = {
    "Claude Opus 4.7 high": Path("runs/eval_format_compile_claude_opus47_reasoning_high.jsonl"),
    "GPT-5.3 Codex high": Path("runs/eval_format_compile_gpt53_codex_reasoning_high.jsonl"),
    "GLM-5 high": Path("runs/eval_format_compile_glm5_reasoning_high.jsonl"),
    "GLM-5 none": Path("runs/eval_format_compile_glm5_reasoning_none.jsonl"),
}

FORMAT_CATEGORY_LABELS = {
    "missing_or_misordered_xml_blocks": "Malformed response blocks",
    "contains_placeholder_text": "Placeholder/incomplete code",
    "contains_import": "External import",
    "compiler_block_invalid_json": "Invalid compiler JSON",
    "compiler_block_not_json_object": "Invalid compiler JSON",
    "missing_compiler_json": "Missing compiler JSON",
    "missing_compiler_version": "Missing compiler version",
    "missing_compiler_settings": "Missing compiler settings",
    "empty_code": "Empty code",
    "missing_pragma": "Missing pragma",
    "missing_contract_interface_or_library_declaration": "Missing contract declaration",
    "invalid_contract_name": "Invalid contract name",
}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                yield json.loads(raw)


def clean_error(error: str) -> str:
    return str(error or "").strip()


def first_error_line(error: str) -> str:
    for raw in clean_error(error).splitlines():
        line = raw.strip()
        if line:
            return line
    return ""


def compilation_category(error: str) -> str:
    text = clean_error(error)
    first = first_error_line(text)
    lower = text.lower()

    if not text:
        return "Unknown compiler failure"
    if "stack too deep" in lower:
        return "Stack too deep"
    if (
        "syntaxerror:" in lower
        and "address" in lower
        and ("invalid checksum" in lower or "exactly 40 hex digits" in lower or "address-literal" in lower or "address literal" in lower)
    ):
        return "Invalid address literal/checksum"
    if (
        "invalid evm version requested" in lower
        or "setting must be" in lower
        or "unable to extract solc version" in lower
        or "failed to install solc" in lower
    ):
        return "Invalid compiler settings"
    if (
        "identifier already declared" in lower
        or "cannot be re-defined" in lower
        or "duplicate function signature" in lower
        or "same name and parameter types defined twice" in lower
        or "function signature hash collision" in lower
    ):
        return "Duplicate declarations/name collisions"
    if (
        "undeclared identifier" in lower
        or "identifier not found or not unique" in lower
        or "member " in lower and "not found or not visible" in lower
    ):
        return "Unresolved identifiers/members"
    if (
        "linearization of inheritance graph impossible" in lower
        or "must override" in lower
        or "trying to override non-virtual" in lower
        or "should be marked as abstract" in lower
        or "contract should be abstract" in lower
        or "missing implementation" in lower
    ):
        return "Inheritance/override/abstractness"
    if "typeerror:" in lower:
        return "Type/mutability errors"
    if "parsererror:" in lower:
        return "Parser errors"
    if "syntaxerror:" in lower:
        return "Syntax errors"
    if (
        "target contract" in lower
        or "target artifact" in lower
        or "runtime bytecode" in lower
        or "bytecode is empty" in lower
        or "unresolved library" in lower
    ):
        return "Target artifact/bytecode errors"
    if re.search(r"\b(Error|CompilerError)\b", first):
        return "Other solc errors"
    return "Other"


def format_categories(errors: List[str]) -> List[str]:
    labels: List[str] = []
    for error in errors:
        labels.append(FORMAT_CATEGORY_LABELS.get(error, error))
    return labels


def pct(count: int, total: int) -> float:
    return count / total if total else 0.0


def summarize_run(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    format_ok = [r for r in rows if (r.get("format") or {}).get("ok")]
    compile_ok = [r for r in rows if (r.get("compilation") or {}).get("success")]
    format_failures = [r for r in rows if not (r.get("format") or {}).get("ok")]
    compilation_failures = [
        r
        for r in rows
        if (r.get("format") or {}).get("ok") and not (r.get("compilation") or {}).get("success")
    ]

    format_error_occurrences: collections.Counter[str] = collections.Counter()
    format_primary_records: collections.Counter[str] = collections.Counter()
    for row in format_failures:
        cats = format_categories((row.get("format") or {}).get("errors") or [])
        for cat in cats:
            format_error_occurrences[cat] += 1
        format_primary_records[cats[0] if cats else "Unknown format failure"] += 1

    compile_categories: collections.Counter[str] = collections.Counter()
    compile_by_difficulty: Dict[str, collections.Counter[str]] = {
        "easy": collections.Counter(),
        "medium": collections.Counter(),
        "hard": collections.Counter(),
    }
    examples: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in compilation_failures:
        category = compilation_category((row.get("compilation") or {}).get("error") or "")
        compile_categories[category] += 1
        diff = str(row.get("difficulty_bin") or "unknown")
        compile_by_difficulty.setdefault(diff, collections.Counter())[category] += 1
        if len(examples[category]) < 3:
            examples[category].append(
                {
                    "benchmark_id": row.get("benchmark_id"),
                    "difficulty_bin": row.get("difficulty_bin"),
                    "first_error_line": first_error_line((row.get("compilation") or {}).get("error") or ""),
                }
            )

    by_difficulty: Dict[str, Dict[str, Any]] = {}
    for diff in ("easy", "medium", "hard"):
        group = [r for r in rows if r.get("difficulty_bin") == diff]
        group_format_ok = [r for r in group if (r.get("format") or {}).get("ok")]
        group_compile_ok = [r for r in group if (r.get("compilation") or {}).get("success")]
        by_difficulty[diff] = {
            "contracts": len(group),
            "format_ok": len(group_format_ok),
            "format_fail": len(group) - len(group_format_ok),
            "compile_ok": len(group_compile_ok),
            "compile_fail_after_format": len(group_format_ok) - len(group_compile_ok),
            "compile_rate_all": pct(len(group_compile_ok), len(group)),
            "compile_rate_format_eligible": pct(len(group_compile_ok), len(group_format_ok)),
            "compile_error_categories": dict(compile_by_difficulty.get(diff, collections.Counter()).most_common()),
        }

    return {
        "contracts": total,
        "format_ok": len(format_ok),
        "format_fail": len(format_failures),
        "format_rate": pct(len(format_ok), total),
        "compile_ok": len(compile_ok),
        "compile_fail_after_format": len(compilation_failures),
        "compile_rate_all": pct(len(compile_ok), total),
        "compile_rate_format_eligible": pct(len(compile_ok), len(format_ok)),
        "format_error_occurrences": dict(format_error_occurrences.most_common()),
        "format_primary_records": dict(format_primary_records.most_common()),
        "compile_error_categories": dict(compile_categories.most_common()),
        "compile_error_examples": dict(examples),
        "by_difficulty": by_difficulty,
    }


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def build_markdown(summary: Dict[str, Any]) -> str:
    model_names = list(summary)
    lines: List[str] = ["# Compilation Error Summary", ""]

    lines.append("## Pipeline Summary")
    lines.append(
        markdown_table(
            ["Model", "Format OK", "Compile OK", "Compile rate", "Compile rate given format"],
            [
                [
                    model,
                    f"{summary[model]['format_ok']}/{summary[model]['contracts']}",
                    f"{summary[model]['compile_ok']}/{summary[model]['contracts']}",
                    f"{100 * summary[model]['compile_rate_all']:.1f}%",
                    f"{100 * summary[model]['compile_rate_format_eligible']:.1f}%",
                ]
                for model in model_names
            ],
        )
    )
    lines.append("")

    categories = sorted(
        {
            category
            for model in model_names
            for category in summary[model]["compile_error_categories"]
        }
    )
    ordered = [
        "Duplicate declarations/name collisions",
        "Type/mutability errors",
        "Parser errors",
        "Syntax errors",
        "Unresolved identifiers/members",
        "Invalid compiler settings",
        "Invalid address literal/checksum",
        "Stack too deep",
        "Inheritance/override/abstractness",
        "Target artifact/bytecode errors",
        "Other solc errors",
        "Other",
        "Unknown compiler failure",
    ]
    categories = [c for c in ordered if c in categories] + [c for c in categories if c not in ordered]

    lines.append("## Compilation-Failure Categories")
    lines.append(
        markdown_table(
            ["Category", *model_names],
            [
                [
                    category,
                    *[
                        str(summary[model]["compile_error_categories"].get(category, 0))
                        for model in model_names
                    ],
                ]
                for category in categories
            ],
        )
    )
    lines.append("")

    lines.append("## Format-Failure Categories")
    fmt_categories = sorted(
        {
            category
            for model in model_names
            for category in summary[model]["format_primary_records"]
        }
    )
    lines.append(
        markdown_table(
            ["Primary format failure", *model_names],
            [
                [
                    category,
                    *[
                        str(summary[model]["format_primary_records"].get(category, 0))
                        for model in model_names
                    ],
                ]
                for category in fmt_categories
            ],
        )
    )
    lines.append("")

    lines.append("## By Difficulty")
    for model in model_names:
        lines.append(f"### {model}")
        lines.append(
            markdown_table(
                ["Difficulty", "Format OK", "Compile OK", "Compile failures after format", "Compile rate"],
                [
                    [
                        diff,
                        f"{summary[model]['by_difficulty'][diff]['format_ok']}/200",
                        f"{summary[model]['by_difficulty'][diff]['compile_ok']}/200",
                        str(summary[model]["by_difficulty"][diff]["compile_fail_after_format"]),
                        f"{100 * summary[model]['by_difficulty'][diff]['compile_rate_all']:.1f}%",
                    ]
                    for diff in ("easy", "medium", "hard")
                ],
            )
        )
        lines.append("")

    lines.append("## Representative Compiler Errors")
    for model in model_names:
        lines.append(f"### {model}")
        for category, examples in summary[model]["compile_error_examples"].items():
            lines.append(f"- {category}:")
            for example in examples[:2]:
                lines.append(
                    f"  - `{example['benchmark_id']}` ({example['difficulty_bin']}): {example['first_error_line']}"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize format and Solidity compilation errors.")
    parser.add_argument("--output-json", type=Path, default=Path("runs/compilation_error_categorized_summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("runs/compilation_error_categorized_summary.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = {label: summarize_run(list(read_jsonl(path))) for label, path in DEFAULT_INPUTS.items()}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({"json": str(args.output_json), "md": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
