#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from copy import deepcopy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

semantic: Any = None


def load_semantic() -> None:
    global semantic
    import run_foundry_semantic
    semantic = run_foundry_semantic


SCHEMA = "scdbench-sequential-checkpoint-timing-v1"
FORK_BLOCK = 24136053
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
DEFAULT_DATASET = ROOT.parent / "dataset/scdbench.jsonl"
DEFAULT_IDS = Path(__file__).resolve().parent / "selected_ids.txt"
DEFAULT_METADATA = Path(__file__).resolve().parent.parent / "ityfuzz_comparison/recorded/contract_metadata.jsonl"
DEFAULT_RECORDED_RESULTS = Path(__file__).resolve().parent / "recorded/timing_results.json"
PROTOCOL_PATH = Path(__file__).resolve().parent / "recorded/coverage_protocol.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute stored SCDBench semantic checkpoints sequentially and "
            "measure per-contract wall time and source coverage."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--recorded-results", type=Path, default=DEFAULT_RECORDED_RESULTS)
    parser.add_argument("--ids-file", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fork-url", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-coverage", action="store_true")
    parser.add_argument("--compile-timeout", type=int, default=60)
    parser.add_argument("--replay-timeout", type=int, default=600)
    parser.add_argument("--coverage-timeout", type=int, default=300)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_ids(path: Path) -> list[str]:
    identifiers = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate benchmark IDs in {path}")
    return identifiers


def load_selected_rows(
    dataset: Path, identifiers: list[str], metadata: Path = DEFAULT_METADATA
) -> list[dict[str, Any]]:
    wanted = set(identifiers)
    rows: dict[str, dict[str, Any]] = {}
    with dataset.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            benchmark_id = str(row.get("benchmark_id") or "")
            if benchmark_id not in wanted:
                continue
            if benchmark_id in rows:
                raise ValueError(f"duplicate selected benchmark ID: {benchmark_id}")
            row.setdefault("_dataset_line", line_number)
            rows[benchmark_id] = row
    missing = [benchmark_id for benchmark_id in identifiers if benchmark_id not in rows]
    if missing:
        raise ValueError(f"selected IDs missing from {dataset}: {missing}")
    archived = {
        row["benchmark_id"]: row
        for row in (json.loads(line) for line in metadata.read_text().splitlines() if line.strip())
    }
    for benchmark_id, row in rows.items():
        if benchmark_id not in archived:
            raise ValueError(f"missing recorded metadata for {benchmark_id}")
        for key in ("semantic_metadata", "semantic_coverage"):
            row[key] = archived[benchmark_id][key]
    return [rows[benchmark_id] for benchmark_id in identifiers]


def interleave_difficulties(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_difficulty = {
        difficulty: [
            row
            for row in rows
            if str(row.get("difficulty_bin") or "") == difficulty
        ]
        for difficulty in ("easy", "medium", "hard")
    }
    ordered: list[dict[str, Any]] = []
    for index in range(max((len(group) for group in by_difficulty.values()), default=0)):
        for difficulty in ("easy", "medium", "hard"):
            group = by_difficulty[difficulty]
            if index < len(group):
                ordered.append(group[index])
    if len(ordered) != len(rows):
        known_ids = {id(row) for row in ordered}
        ordered.extend(row for row in rows if id(row) not in known_ids)
    return ordered


def frozen_replay_args(
    forge_bin: str,
    fork_url: str,
    compile_timeout: int,
    replay_timeout: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        forge_bin=forge_bin,
        fork_url=fork_url,
        fork_block=FORK_BLOCK,
        timeout=replay_timeout,
        chunk_size=10,
        compile_timeout=compile_timeout,
        coverage_report="lcov",
        run_coverage=False,
    )


def frozen_coverage_args(
    forge_bin: str,
    fork_url: str,
    compile_timeout: int,
    coverage_timeout: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        forge_bin=forge_bin,
        fork_url=fork_url,
        fork_block=FORK_BLOCK,
        compile_timeout=compile_timeout,
        coverage_synthetic_constructor_args=True,
        coverage_batch_size=10,
        coverage_call_gas_limit=5_000_000,
        coverage_sender_variants=1,
        coverage_auth_storage_slots=0,
        coverage_legacy_solidity=True,
        coverage_ir_minimum_retry=True,
        coverage_allow_failure=True,
        coverage_trace_timeout=90,
        coverage_max_trace_tests=0,
        coverage_augment_calls=False,
        coverage_augment_fallback_calls=False,
        coverage_augment_runs=24,
        coverage_augment_static_cases_per_function=32,
        coverage_augment_setup_functions=0,
        coverage_synthetic_setup_functions=0,
        coverage_augment_stateful_cases_per_function=4,
        max_dynamic_array_len=3,
        max_fixed_array_len=16,
        max_static_values_per_type=12,
        max_static_int_abs=10**24,
        max_msg_value_wei=10**17,
        timeout=coverage_timeout,
        seed=20260504,
        chunk_size=10,
        value_wei=0,
    )


def tool_version(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        return f"unavailable:{exc}"
    output = (completed.stdout or completed.stderr or "").strip().splitlines()
    return output[0] if output else f"exit_code={completed.returncode}"


def compact_forge_runs(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for run in runs:
        compact.append(
            {
                "exit_code": run.get("exit_code"),
                "latency_sec": run.get("latency_sec"),
                "chunk_index": run.get("chunk_index"),
                "chunk_start": run.get("chunk_start"),
                "chunk_calls": run.get("chunk_calls"),
                "split_depth": run.get("split_depth"),
                "case_ids": run.get("case_ids") or [],
            }
        )
    return compact


def compact_failed_run(run: dict[str, Any] | None) -> dict[str, Any] | None:
    if not run:
        return None
    compact = compact_forge_runs([run])[0]
    compact["stdout"] = str(run.get("stdout") or "")[:8000]
    compact["stderr"] = str(run.get("stderr") or "")[:8000]
    return compact


def checkpoint_sequence_depth(call: dict[str, Any]) -> int:
    setup = call.get("setup")
    return 1 + (len(setup) if isinstance(setup, list) else 0)


def original_init_code(
    row: dict[str, Any],
    compile_result: Any,
) -> tuple[str, str, bool]:
    metadata = row.get("semantic_metadata") or {}
    deployment_mode = str(metadata.get("deployment_mode") or "")
    source_matches = semantic.source_bytecode_matches_input(
        row,
        compile_result.deployed_bytecode,
    )
    stored_source_matches = metadata.get("source_deployed_bytecode_matches_input_logic")
    if stored_source_matches is not None and bool(stored_source_matches) != source_matches:
        raise ValueError(
            "current ground-truth source/runtime comparison disagrees with "
            "the submitted semantic metadata"
        )
    if deployment_mode == "ground_truth_source_creation":
        if not compile_result.bytecode:
            raise ValueError("ground-truth creation bytecode is unavailable")
        return semantic.ensure0x(compile_result.bytecode), deployment_mode, source_matches
    if deployment_mode == "input_runtime_wrapper":
        runtime = semantic.ensure0x(str(row.get("bytecode") or ""))
        return semantic.make_init_code_from_runtime(runtime), deployment_mode, source_matches
    raise ValueError(f"unsupported submitted deployment mode: {deployment_mode!r}")


def validate_observations(
    calls: list[dict[str, Any]],
    observations: dict[str, dict[str, dict[str, Any]]],
    skipped_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    skipped_ids = {str(item.get("case_id") or "") for item in skipped_cases}
    missing: list[str] = []
    mismatched: list[str] = []
    matched = 0
    for call in calls:
        case_id = str(call.get("case_id") or "")
        if case_id in skipped_ids:
            continue
        actual = (observations.get(case_id) or {}).get("original")
        if not actual:
            missing.append(case_id)
            continue
        expected = call.get("baseline") or {}
        if semantic.observation_equal(expected, actual, str(call.get("state") or "")):
            matched += 1
        else:
            mismatched.append(case_id)
    return {
        "matched": matched,
        "missing": missing,
        "mismatched": mismatched,
        "skipped": sorted(skipped_ids),
        "valid": not missing and not mismatched and not skipped_ids,
    }


def compact_coverage(coverage: dict[str, Any]) -> dict[str, Any]:
    lines = coverage.get("lines") if isinstance(coverage.get("lines"), dict) else {}
    functions = (
        coverage.get("functions")
        if isinstance(coverage.get("functions"), dict)
        else {}
    )
    return {
        "available": bool(coverage.get("available")),
        "kind": coverage.get("kind"),
        "instrumentation_schema": coverage.get("instrumentation_schema"),
        "reason": coverage.get("reason"),
        "source_deployment_mode": coverage.get("source_deployment_mode"),
        "checkpoint_call_count": coverage.get("checkpoint_call_count"),
        "supplemental_call_count": coverage.get("supplemental_call_count"),
        "lines_covered": lines.get("covered"),
        "lines_total": lines.get("total"),
        "line_ratio": lines.get("ratio"),
        "functions_covered": functions.get("covered"),
        "functions_total": functions.get("total"),
        "function_ratio": functions.get("ratio"),
    }


def compact_submitted_coverage(row: dict[str, Any]) -> dict[str, Any]:
    coverage = row.get("semantic_coverage")
    if not isinstance(coverage, dict):
        return {
            "available": False,
            "kind": None,
            "reason": "missing_semantic_coverage",
            "lines_covered": None,
            "lines_total": None,
            "line_ratio": None,
        }
    lines = coverage.get("lines") if isinstance(coverage.get("lines"), dict) else {}
    functions = (
        coverage.get("functions")
        if isinstance(coverage.get("functions"), dict)
        else {}
    )
    return {
        "available": bool(coverage.get("available")),
        "kind": coverage.get("kind"),
        "reason": coverage.get("reason"),
        "lines_covered": lines.get("covered"),
        "lines_total": lines.get("total"),
        "line_ratio": lines.get("ratio"),
        "functions_covered": functions.get("covered"),
        "functions_total": functions.get("total"),
        "function_ratio": functions.get("ratio"),
    }


def sanitized_coverage(coverage: dict[str, Any], fork_url: str) -> dict[str, Any]:
    sanitized = deepcopy(coverage)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and item == fork_url:
                    value[key] = "<redacted>"
                else:
                    visit(item)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, str) and item == fork_url:
                    value[index] = "<redacted>"
                else:
                    visit(item)

    visit(sanitized)
    return sanitized


def run_contract(
    row: dict[str, Any],
    output_dir: Path,
    replay_args: SimpleNamespace,
    coverage_args: SimpleNamespace,
    skip_coverage: bool,
) -> dict[str, Any]:
    benchmark_id = str(row.get("benchmark_id") or "")
    contract_dir = output_dir / "contracts" / benchmark_id
    contract_dir.mkdir(parents=True, exist_ok=True)
    contract_started = time.monotonic()
    calls = [
        dict(call)
        for call in semantic.flatten_test_case_groups(row.get("test_cases") or {})
    ]
    calls.sort(key=lambda call: str(call.get("case_id") or ""))
    if not calls:
        raise ValueError(f"{benchmark_id} has no stored semantic checkpoints")

    compile_started = time.monotonic()
    compile_result = semantic.compile_ground_truth(
        row,
        replay_args.compile_timeout,
    )
    compile_seconds = time.monotonic() - compile_started
    if not compile_result.success:
        raise RuntimeError(
            f"ground-truth compilation failed: {compile_result.error}"
        )
    init_code, deployment_mode, source_matches = original_init_code(
        row,
        compile_result,
    )

    replay_started = time.monotonic()
    observations, forge_runs, failed_run, skipped_cases = (
        semantic.run_forge_observations(
            replay_args,
            benchmark_id,
            init_code,
            calls,
        )
    )
    replay_seconds = time.monotonic() - replay_started
    validation = validate_observations(calls, observations, skipped_cases)

    coverage: dict[str, Any]
    coverage_seconds: float | None
    if skip_coverage:
        coverage = {
            "kind": "foundry_source_lcov",
            "instrumentation_schema": semantic.COVERAGE_SCHEMA,
            "available": False,
            "reason": "skipped_by_cli",
        }
        coverage_seconds = None
    else:
        coverage_started = time.monotonic()
        coverage = semantic.run_source_coverage(
            row,
            calls,
            deployment_mode,
            coverage_args,
            abi=compile_result.abi,
            source_matches=source_matches,
        )
        coverage_seconds = time.monotonic() - coverage_started
    atomic_write_json(
        contract_dir / "coverage.json",
        sanitized_coverage(coverage, replay_args.fork_url),
    )

    record = {
        "schema": SCHEMA,
        "benchmark_id": benchmark_id,
        "difficulty": row.get("difficulty_bin"),
        "contract_name": row.get("contract_name"),
        "deployment_mode": deployment_mode,
        "checkpoint_count": len(calls),
        "function_group_count": len(row.get("test_cases") or {}),
        "max_transaction_sequence_depth": max(
            checkpoint_sequence_depth(call) for call in calls
        ),
        "timing_seconds": {
            "ground_truth_compile": compile_seconds,
            "checkpoint_execution": replay_seconds,
            "coverage_measurement": coverage_seconds,
            "total": time.monotonic() - contract_started,
            "forge_process_sum": sum(
                float(run.get("latency_sec") or 0.0) for run in forge_runs
            ),
        },
        "checkpoint_validation": validation,
        "forge_run_count": len(forge_runs),
        "forge_runs": compact_forge_runs(forge_runs),
        "fatal_forge_run": compact_failed_run(failed_run),
        "submitted_coverage": compact_submitted_coverage(row),
        "coverage": compact_coverage(coverage),
        "coverage_artifact": str(
            (contract_dir / "coverage.json").relative_to(output_dir)
        ),
    }
    record["success"] = failed_run is None and validation["valid"]
    atomic_write_json(contract_dir / "result.json", record)
    return record


def percentile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(proportion * len(ordered)) - 1))
    return ordered[index]


def aggregate_group(
    records: list[dict[str, Any]],
    coverage_field: str = "submitted_coverage",
) -> dict[str, Any]:
    replay_times = [
        float(record["timing_seconds"]["checkpoint_execution"])
        for record in records
    ]
    checkpoint_count = sum(int(record.get("checkpoint_count") or 0) for record in records)
    measurable = [
        record
        for record in records
        if bool((record.get(coverage_field) or {}).get("available"))
        and (record.get(coverage_field) or {}).get("lines_total") is not None
    ]
    covered = sum(
        int(record[coverage_field]["lines_covered"]) for record in measurable
    )
    total = sum(int(record[coverage_field]["lines_total"]) for record in measurable)
    ratios = [
        float(record[coverage_field]["line_ratio"])
        for record in measurable
        if record[coverage_field].get("line_ratio") is not None
    ]
    return {
        "contracts": len(records),
        "successful_contracts": sum(bool(record.get("success")) for record in records),
        "checkpoints": checkpoint_count,
        "checkpoint_execution_seconds": {
            "total": sum(replay_times),
            "mean_per_contract": statistics.fmean(replay_times) if replay_times else None,
            "median_per_contract": statistics.median(replay_times) if replay_times else None,
            "p90_per_contract": percentile(replay_times, 0.90),
            "mean_per_checkpoint": (
                sum(replay_times) / checkpoint_count if checkpoint_count else None
            ),
        },
        "coverage": {
            "field": coverage_field,
            "measurable_contracts": len(measurable),
            "lines_covered": covered,
            "lines_total": total,
            "pooled_ratio": covered / total if total else None,
            "mean_contract_ratio": statistics.fmean(ratios) if ratios else None,
            "median_contract_ratio": statistics.median(ratios) if ratios else None,
        },
        "max_transaction_sequence_depth": max(
            (int(record.get("max_transaction_sequence_depth") or 0) for record in records),
            default=0,
        ),
    }


def collect_results(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    contracts_dir = output_dir / "contracts"
    if not contracts_dir.exists():
        return records
    for path in contracts_dir.glob("*/result.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            isinstance(payload, dict)
            and payload.get("schema") == SCHEMA
            and isinstance(payload.get("timing_seconds"), dict)
        ):
            records.append(payload)
    return sorted(
        records,
        key=lambda record: (
            DIFFICULTY_ORDER.get(str(record.get("difficulty")), 99),
            str(record.get("benchmark_id") or ""),
        ),
    )


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "benchmark_id",
        "difficulty",
        "success",
        "checkpoint_count",
        "function_group_count",
        "max_transaction_sequence_depth",
        "compile_seconds",
        "checkpoint_execution_seconds",
        "coverage_measurement_seconds",
        "total_seconds",
        "submitted_coverage_available",
        "submitted_coverage_reason",
        "submitted_lines_covered",
        "submitted_lines_total",
        "submitted_line_coverage",
        "diagnostic_lcov_available",
        "diagnostic_lcov_reason",
        "diagnostic_lcov_lines_covered",
        "diagnostic_lcov_lines_total",
        "diagnostic_lcov_line_coverage",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            timing = record.get("timing_seconds") or {}
            submitted = record.get("submitted_coverage") or {}
            diagnostic = record.get("coverage") or {}
            writer.writerow(
                {
                    "benchmark_id": record.get("benchmark_id"),
                    "difficulty": record.get("difficulty"),
                    "success": record.get("success"),
                    "checkpoint_count": record.get("checkpoint_count"),
                    "function_group_count": record.get("function_group_count"),
                    "max_transaction_sequence_depth": record.get(
                        "max_transaction_sequence_depth"
                    ),
                    "compile_seconds": timing.get("ground_truth_compile"),
                    "checkpoint_execution_seconds": timing.get(
                        "checkpoint_execution"
                    ),
                    "coverage_measurement_seconds": timing.get(
                        "coverage_measurement"
                    ),
                    "total_seconds": timing.get("total"),
                    "submitted_coverage_available": submitted.get("available"),
                    "submitted_coverage_reason": submitted.get("reason"),
                    "submitted_lines_covered": submitted.get("lines_covered"),
                    "submitted_lines_total": submitted.get("lines_total"),
                    "submitted_line_coverage": submitted.get("line_ratio"),
                    "diagnostic_lcov_available": diagnostic.get("available"),
                    "diagnostic_lcov_reason": diagnostic.get("reason"),
                    "diagnostic_lcov_lines_covered": diagnostic.get("lines_covered"),
                    "diagnostic_lcov_lines_total": diagnostic.get("lines_total"),
                    "diagnostic_lcov_line_coverage": diagnostic.get("line_ratio"),
                }
            )
    os.replace(temporary, path)


def format_seconds(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}"


def format_ratio(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.1f}%"


def write_summary(
    output_dir: Path,
    records: list[dict[str, Any]],
    expected_contracts: int,
) -> None:
    groups = {
        difficulty: aggregate_group(
            [record for record in records if record.get("difficulty") == difficulty]
        )
        for difficulty in ("easy", "medium", "hard")
    }
    groups["all"] = aggregate_group(records)
    diagnostic_groups = {
        difficulty: aggregate_group(
            [record for record in records if record.get("difficulty") == difficulty],
            coverage_field="coverage",
        )
        for difficulty in ("easy", "medium", "hard")
    }
    diagnostic_groups["all"] = aggregate_group(
        records,
        coverage_field="coverage",
    )
    payload = {
        "schema": SCHEMA,
        "expected_contracts": expected_contracts,
        "completed_contracts": len(records),
        "by_difficulty": groups,
        "diagnostic_lcov_by_difficulty": diagnostic_groups,
        "results": records,
    }
    atomic_write_json(output_dir / "results.json", payload)
    write_csv(output_dir / "results.csv", records)

    lines = [
        "# Sequential Semantic Checkpoint Timing",
        "",
        f"Completed: **{len(records)}/{expected_contracts} contracts**.",
        "",
        "| Bin | Contracts | Checkpoints | Checkpoint time | Median/contract | Submitted coverage | Measurable | Max depth |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("easy", "medium", "hard", "all"):
        group = groups[label]
        timing = group["checkpoint_execution_seconds"]
        coverage = group["coverage"]
        lines.append(
            "| "
            + " | ".join(
                [
                    label.capitalize(),
                    f"{group['successful_contracts']}/{group['contracts']}",
                    str(group["checkpoints"]),
                    f"{format_seconds(timing['total'])}s",
                    f"{format_seconds(timing['median_per_contract'])}s",
                    format_ratio(coverage["pooled_ratio"]),
                    f"{coverage['measurable_contracts']}/{group['contracts']}",
                    str(group["max_transaction_sequence_depth"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Checkpoint time excludes ground-truth compilation and coverage "
            "instrumentation. Submitted coverage is pooled executable-line "
            "coverage from the frozen benchmark artifact over contracts with "
            "numeric measurements. Fresh stored-cases-only LCOV is retained "
            "separately in results.json as a diagnostic.",
            "",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.summarize_only:
        payload = json.loads(args.recorded_results.read_text(encoding="utf-8"))
        write_summary(args.output_dir, payload["results"], payload["expected_contracts"])
        return
    load_semantic()
    args.dataset = args.dataset.expanduser().resolve()
    args.ids_file = args.ids_file.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    semantic.load_env_file(ROOT / ".env")
    fork_url = args.fork_url or os.environ.get("ARCHIVE_RPC_URL")
    if not fork_url:
        raise SystemExit("missing ARCHIVE_RPC_URL in .env or --fork-url")
    forge_bin = shutil.which("forge")
    if not forge_bin:
        raise SystemExit("forge is not available on PATH")

    identifiers = load_ids(args.ids_file)
    rows = load_selected_rows(args.dataset, identifiers, args.metadata)
    counts = Counter(str(row.get("difficulty_bin") or "") for row in rows)
    if counts != Counter({"easy": 20, "medium": 20, "hard": 20}):
        raise SystemExit(f"expected 20/20/20 subset, found {dict(counts)}")
    rows = interleave_difficulties(rows)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        rows = rows[: args.limit]

    replay_args = frozen_replay_args(
        forge_bin,
        fork_url,
        args.compile_timeout,
        args.replay_timeout,
    )
    coverage_args = frozen_coverage_args(
        forge_bin,
        fork_url,
        args.compile_timeout,
        args.coverage_timeout,
    )
    warmup_started = time.monotonic()
    warmup = subprocess.run(
        [forge_bin, "build"],
        cwd=semantic.FOUNDRY_ROOT,
        capture_output=True,
        text=True,
        timeout=args.compile_timeout,
        check=False,
    )
    warmup_seconds = time.monotonic() - warmup_started
    if warmup.returncode != 0:
        raise SystemExit(
            "Foundry harness warm-up failed: "
            + (warmup.stderr or warmup.stdout or "").strip()[:2000]
        )

    manifest = {
        "schema": SCHEMA,
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "ids_file": str(args.ids_file),
        "ids_file_sha256": sha256_file(args.ids_file),
        "selected_contracts": len(rows),
        "full_subset_difficulty_counts": dict(sorted(counts.items())),
        "execution_order": [str(row.get("benchmark_id") or "") for row in rows],
        "fork_block": FORK_BLOCK,
        "chunk_size": replay_args.chunk_size,
        "harness_warmup_seconds": warmup_seconds,
        "coverage_profile": "stored-cases-lcov",
        "coverage_protocol_digest": hashlib.sha256(
            json.dumps(json.loads(PROTOCOL_PATH.read_text()), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "coverage_protocol": json.loads(PROTOCOL_PATH.read_text()),
        "skip_coverage": args.skip_coverage,
        "tool_versions": {
            "forge": tool_version([forge_bin, "--version"]),
            "python": sys.version.splitlines()[0],
        },
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)

    for index, row in enumerate(rows, start=1):
        benchmark_id = str(row.get("benchmark_id") or "")
        result_path = (
            args.output_dir / "contracts" / benchmark_id / "result.json"
        )
        if result_path.exists() and not args.force:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("schema") == SCHEMA and existing.get("success"):
                print(
                    f"[sequential] {index}/{len(rows)} {benchmark_id} "
                    "skipped=successful_result_exists",
                    flush=True,
                )
                continue
        try:
            record = run_contract(
                row,
                args.output_dir,
                replay_args,
                coverage_args,
                args.skip_coverage,
            )
            coverage = record["coverage"]
            print(
                f"[sequential] {index}/{len(rows)} {benchmark_id} "
                f"checkpoints={record['checkpoint_count']} "
                f"replay_sec={record['timing_seconds']['checkpoint_execution']:.3f} "
                f"coverage={coverage.get('line_ratio')} "
                f"success={record['success']}",
                flush=True,
            )
        except Exception as exc:
            error_record = {
                "schema": SCHEMA,
                "benchmark_id": benchmark_id,
                "difficulty": row.get("difficulty_bin"),
                "success": False,
                "error": repr(exc),
            }
            atomic_write_json(result_path, error_record)
            print(
                f"[sequential] {index}/{len(rows)} {benchmark_id} "
                f"error={exc!r}",
                flush=True,
            )
        records = collect_results(args.output_dir)
        write_summary(args.output_dir, records, len(rows))

    records = collect_results(args.output_dir)
    write_summary(args.output_dir, records, len(rows))
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "completed_contracts": len(records),
                "expected_contracts": len(rows),
            }
        )
    )


if __name__ == "__main__":
    main()
