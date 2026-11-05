#!/usr/bin/env python3
"""Compare SCDBench and ItyFuzz on the submission's target-line universe."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path(__file__).resolve().parent / "recorded/contract_metadata.jsonl"
DEFAULT_ITYFUZZ_RESULTS = Path(__file__).resolve().parent / "recorded/ityfuzz_results.json"
DEFAULT_PREFIX_DIR = (
    Path(__file__).resolve().parent / "recorded/budgeted_prefix"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/matched_coverage.json"
DIFFICULTIES = ("easy", "medium", "hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ityfuzz-results", type=Path, default=DEFAULT_ITYFUZZ_RESULTS)
    parser.add_argument("--budgeted-prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def coverage_line_sets(coverage: dict[str, Any]) -> tuple[set[int], set[int]] | None:
    if not coverage.get("available"):
        return None
    lines = coverage.get("lines")
    if not isinstance(lines, dict):
        target = coverage.get("target_contract")
        lines = target.get("lines") if isinstance(target, dict) else None
    if not isinstance(lines, dict):
        return None
    covered = {int(value) for value in lines.get("covered_line_numbers") or []}
    uncovered = {int(value) for value in lines.get("uncovered_line_numbers") or []}
    universe = covered | uncovered
    reported_total = lines.get("total")
    if reported_total is not None and int(reported_total) != len(universe):
        raise ValueError(
            f"coverage line total {reported_total} does not match {len(universe)} lines"
        )
    return universe, covered


def load_rows(dataset: Path, selected_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with dataset.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            benchmark_id = str(row.get("benchmark_id") or "")
            if benchmark_id in selected_ids:
                rows[benchmark_id] = row
    missing = sorted(selected_ids - set(rows))
    if missing:
        raise ValueError(f"selected benchmark IDs missing from dataset: {missing}")
    return rows


def pooled_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    covered = sum(int(row[f"{prefix}_covered"]) for row in rows)
    total = sum(int(row["line_total"]) for row in rows)
    return {
        "contracts": len(rows),
        "covered": covered,
        "total": total,
        "ratio": covered / total if total else None,
    }


def main() -> None:
    args = parse_args()
    ityfuzz = read_json(args.ityfuzz_results)
    cases = ityfuzz.get("cases")
    if not isinstance(cases, list):
        raise ValueError("ItyFuzz results have no case list")
    selected_ids = {str(case.get("benchmark_id") or "") for case in cases}
    if len(cases) != len(selected_ids):
        raise ValueError("ItyFuzz results contain duplicate benchmark IDs")
    rows_by_id = load_rows(args.dataset, selected_ids)

    comparable: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    for case in cases:
        benchmark_id = str(case["benchmark_id"])
        dataset_row = rows_by_id[benchmark_id]
        submitted = coverage_line_sets(dataset_row.get("semantic_coverage") or {})
        if submitted is None:
            unavailable.append(
                {
                    "benchmark_id": benchmark_id,
                    "difficulty": str(case.get("difficulty_bin") or ""),
                    "reason": str(
                        (dataset_row.get("semantic_coverage") or {}).get("reason")
                        or "submission_target_line_universe_unavailable"
                    ),
                }
            )
            continue
        universe, scdbench_covered = submitted

        prefix_coverage_path = (
            args.budgeted_prefix_dir / "contracts" / benchmark_id / "coverage.json"
        )
        if prefix_coverage_path.exists():
            prefix_lines = coverage_line_sets(read_json(prefix_coverage_path))
            if prefix_lines is None:
                unavailable.append(
                    {
                        "benchmark_id": benchmark_id,
                        "difficulty": str(case.get("difficulty_bin") or ""),
                        "reason": "budget_capped_target_line_coverage_unavailable",
                    }
                )
                continue
            _, prefix_covered = prefix_lines
            scdbench_covered = prefix_covered & universe

        ityfuzz_coverage = case.get("coverage") or {}
        ityfuzz_lines = {
            int(value) for value in ityfuzz_coverage.get("covered_source_lines") or []
        }
        comparable.append(
            {
                "benchmark_id": benchmark_id,
                "difficulty": str(case.get("difficulty_bin") or ""),
                "line_total": len(universe),
                "scdbench_covered": len(scdbench_covered & universe),
                "ityfuzz_covered": len(ityfuzz_lines & universe),
                "ityfuzz_status": case.get("status"),
                "scdbench_budget_capped": prefix_coverage_path.exists(),
            }
        )

    by_difficulty: dict[str, dict[str, Any]] = {}
    for difficulty in DIFFICULTIES:
        group = [row for row in comparable if row["difficulty"] == difficulty]
        by_difficulty[difficulty] = {
            "scdbench": pooled_summary(group, "scdbench"),
            "ityfuzz": pooled_summary(group, "ityfuzz"),
        }
    by_difficulty["all"] = {
        "scdbench": pooled_summary(comparable, "scdbench"),
        "ityfuzz": pooled_summary(comparable, "ityfuzz"),
    }

    status_counts: dict[str, int] = defaultdict(int)
    for case in cases:
        status_counts[str(case.get("status") or "")] += 1
    payload = {
        "schema": "scdbench-ityfuzz-matched-target-line-coverage-v1",
        "selected_contracts": len(cases),
        "comparable_contracts": len(comparable),
        "submission_line_universe_unavailable": unavailable,
        "budgets_seconds": {"easy": 4, "medium": 9, "hard": 16},
        "summary": by_difficulty,
        "ityfuzz_status_counts": dict(sorted(status_counts.items())),
        "cases": comparable,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(by_difficulty, sort_keys=True))


if __name__ == "__main__":
    main()
