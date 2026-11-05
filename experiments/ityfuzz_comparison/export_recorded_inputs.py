#!/usr/bin/env python3
"""Project archived experiment records into the public coverage-only inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PINNED_COMMIT = "80e3b44170573b592cd53b2c5ee9f36b21032b16"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coverage_projection(record: dict) -> dict:
    result = {key: record[key] for key in ("available", "kind", "reason") if key in record}
    lines = record.get("lines")
    if not isinstance(lines, dict):
        lines = (record.get("target_contract") or {}).get("lines")
    if isinstance(lines, dict):
        result["lines"] = {
            key: lines[key]
            for key in ("total", "covered", "uncovered", "ratio", "covered_line_numbers", "uncovered_line_numbers")
            if key in lines
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ityfuzz-results", type=Path, required=True)
    parser.add_argument("--timing-results", type=Path, required=True)
    parser.add_argument("--timing-manifest", type=Path, required=True)
    parser.add_argument("--budgeted-prefix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "recorded")
    parser.add_argument("--timing-output-dir", type=Path, default=HERE.parent / "semantic_checkpoint_timing/recorded")
    args = parser.parse_args()
    raw = read_json(args.ityfuzz_results)
    selected = {case["benchmark_id"] for case in raw["cases"]}
    assert len(selected) == len(raw["cases"]) == 60
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    with args.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["benchmark_id"] not in selected:
                continue
            metadata.append({
                "benchmark_id": row["benchmark_id"],
                "difficulty_bin": row["difficulty_bin"],
                "semantic_metadata": {"deployment_mode": row["semantic_metadata"]["deployment_mode"]},
                "semantic_coverage": coverage_projection(row.get("semantic_coverage") or {}),
            })
    assert {row["benchmark_id"] for row in metadata} == selected
    metadata.sort(key=lambda row: row["benchmark_id"])
    with (args.output_dir / "contract_metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    cases = []
    for case in raw["cases"]:
        projected = {key: case[key] for key in (
            "benchmark_id", "difficulty_bin", "status", "seed",
            "runtime_matches_submission", "wall_clock_budget_seconds", "coverage_replay_budget_seconds"
        ) if key in case}
        coverage = case.get("coverage") or {}
        projected["coverage"] = {key: coverage[key] for key in (
            "available", "covered_source_lines", "coverable_source_lines",
            "instruction_total", "instruction_covered", "branch_total", "branch_covered"
        ) if key in coverage}
        sequence_export = case.get("sequence_export") or {}
        projected["sequence_export"] = {key: sequence_export[key] for key in (
            "candidate_replayable_files", "duplicate_sequences", "unique_sequences", "max_sequence_length"
        ) if key in sequence_export}
        cases.append(projected)
    write_json(args.output_dir / "ityfuzz_results.json", {
        "schema": "scdbench-public-ityfuzz-coverage-inputs-v1",
        "ityfuzz_commit": PINNED_COMMIT,
        "budgets_seconds": {"easy": 4, "medium": 9, "hard": 16},
        "cases": cases,
    })
    prefix_files = {}
    for path in sorted((args.budgeted_prefix / "contracts").glob("*/coverage.json")):
        bid = path.parent.name
        if bid not in selected:
            continue
        write_json(args.output_dir / "budgeted_prefix/contracts" / bid / "coverage.json", coverage_projection(read_json(path)))
        prefix_files[bid] = sha256(path)
    timing = read_json(args.timing_results)
    records = [{key: row[key] for key in (
        "schema", "benchmark_id", "difficulty", "success", "checkpoint_count",
        "function_group_count", "max_transaction_sequence_depth", "timing_seconds",
        "checkpoint_validation", "submitted_coverage", "coverage", "deployment_mode"
    ) if key in row} for row in timing["results"]]
    assert {row["benchmark_id"] for row in records} == selected
    write_json(args.timing_output_dir / "timing_results.json", {
        "schema": timing["schema"], "expected_contracts": 60, "results": records,
    })
    write_json(args.timing_output_dir / "coverage_protocol.json", read_json(args.timing_manifest)["coverage_protocol"])
    identifiers = "\n".join(sorted(selected)) + "\n"
    (args.output_dir.parent / "selected_ids.txt").write_text(identifiers, encoding="ascii")
    (args.timing_output_dir.parent / "selected_ids.txt").write_text(identifiers, encoding="ascii")
    write_json(args.output_dir / "provenance.json", {
        "source_sha256": {
            "original_complete_dataset": sha256(args.dataset),
            "ityfuzz_results": sha256(args.ityfuzz_results),
            "sequential_timing_results": sha256(args.timing_results),
            "sequential_timing_manifest": sha256(args.timing_manifest),
        },
        "budgeted_prefix_coverage_sha256": prefix_files,
        "projection": "Only fields used by coverage/timing analyses and deployment preparation are retained. Coverage line sets, measured timings, baseline validation and status labels are unchanged. Logs, RPC endpoints, paths and vulnerability outputs are omitted.",
    })


if __name__ == "__main__":
    main()
