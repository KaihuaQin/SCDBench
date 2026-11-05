#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import run_sequential as experiment


SCHEMA = "scdbench-budgeted-checkpoint-prefix-v1"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "budgeted_prefix"
BUDGETS = {
    "scdbench-easy-154": 4.0,
    "scdbench-medium-120": 9.0,
    "scdbench-medium-163": 9.0,
    "scdbench-medium-164": 9.0,
    "scdbench-hard-142": 16.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute over-budget SCDBench checkpoint suites under hard "
            "per-bin deadlines and measure coverage of completed prefixes."
        )
    )
    parser.add_argument("--dataset", type=Path, default=experiment.DEFAULT_DATASET)
    parser.add_argument("--metadata", type=Path, default=experiment.DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fork-url", default=None)
    parser.add_argument("--compile-timeout", type=int, default=60)
    parser.add_argument("--coverage-timeout", type=int, default=300)
    parser.add_argument("--id", action="append", choices=sorted(BUDGETS))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_rows(
    dataset: Path,
    selected_ids: list[str] | None = None,
    metadata: Path = experiment.DEFAULT_METADATA,
) -> list[dict[str, Any]]:
    selected = set(selected_ids or BUDGETS)
    identifiers = [benchmark_id for benchmark_id in BUDGETS if benchmark_id in selected]
    return experiment.load_selected_rows(dataset, identifiers, metadata)


def attempt_summary(
    *,
    chunk: list[dict[str, Any]],
    start: int,
    depth: int,
    elapsed: float,
    exit_code: int | None,
    timed_out: bool,
    failure: str | None = None,
) -> dict[str, Any]:
    return {
        "case_ids": [str(call.get("case_id") or "") for call in chunk],
        "chunk_start": start,
        "chunk_calls": len(chunk),
        "split_depth": depth,
        "elapsed_seconds": elapsed,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "failure": failure,
    }


def execute_budgeted(
    replay_args: SimpleNamespace,
    benchmark_id: str,
    init_code: str,
    calls: list[dict[str, Any]],
    budget_seconds: float,
) -> dict[str, Any]:
    observations: dict[str, dict[str, dict[str, Any]]] = {}
    attempts: list[dict[str, Any]] = []
    skipped: list[str] = []
    fatal: dict[str, Any] | None = None
    deadline_hit = False
    started = time.monotonic()
    deadline = started + budget_seconds

    def run_chunk(
        chunk: list[dict[str, Any]],
        start: int,
        depth: int,
    ) -> bool:
        nonlocal deadline_hit, fatal
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            deadline_hit = True
            return False
        local_args = SimpleNamespace(**vars(replay_args))
        local_args.timeout = max(0.01, remaining)
        env = experiment.semantic.foundry_env(
            benchmark_id,
            init_code,
            chunk,
        )
        attempt_started = time.monotonic()
        try:
            run = experiment.semantic.run_forge(
                local_args,
                env,
                coverage=False,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - attempt_started
            attempts.append(
                attempt_summary(
                    chunk=chunk,
                    start=start,
                    depth=depth,
                    elapsed=elapsed,
                    exit_code=None,
                    timed_out=True,
                    failure="hard_budget_timeout",
                )
            )
            deadline_hit = True
            return False
        elapsed = time.monotonic() - attempt_started
        exit_code = int(run.get("exit_code") or 0)
        if exit_code == 0:
            experiment.semantic.merge_observations(
                observations,
                experiment.semantic.parse_foundry_observations(
                    str(run.get("stdout") or "")
                ),
            )
            attempts.append(
                attempt_summary(
                    chunk=chunk,
                    start=start,
                    depth=depth,
                    elapsed=elapsed,
                    exit_code=exit_code,
                    timed_out=False,
                )
            )
            return True

        failure = experiment.semantic.classify_forge_failure(run)
        attempts.append(
            attempt_summary(
                chunk=chunk,
                start=start,
                depth=depth,
                elapsed=elapsed,
                exit_code=exit_code,
                timed_out=False,
                failure=failure,
            )
        )
        if len(chunk) > 1:
            midpoint = len(chunk) // 2
            if not run_chunk(chunk[:midpoint], start, depth + 1):
                return False
            return run_chunk(
                chunk[midpoint:],
                start + midpoint,
                depth + 1,
            )
        if "SCDBENCH_DEPLOY" in str(run.get("stdout") or ""):
            skipped.append(str(chunk[0].get("case_id") or ""))
            return True
        fatal = attempts[-1]
        return False

    chunk_size = max(1, int(replay_args.chunk_size))
    for start in range(0, len(calls), chunk_size):
        if deadline - time.monotonic() <= 0:
            deadline_hit = True
            break
        if not run_chunk(calls[start : start + chunk_size], start, 0):
            break

    elapsed = time.monotonic() - started
    completed_ids = {
        case_id
        for case_id, by_target in observations.items()
        if isinstance(by_target, dict) and by_target.get("original")
    }
    completed_calls = [
        call
        for call in calls
        if str(call.get("case_id") or "") in completed_ids
    ]
    validation = experiment.validate_observations(
        completed_calls,
        observations,
        [{"case_id": case_id} for case_id in skipped],
    )
    return {
        "budget_seconds": budget_seconds,
        "elapsed_seconds": elapsed,
        "deadline_hit": deadline_hit,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "fatal": fatal,
        "completed_calls": completed_calls,
        "completed_checkpoint_count": len(completed_calls),
        "validation": validation,
    }


def coverage_for_prefix(
    row: dict[str, Any],
    calls: list[dict[str, Any]],
    deployment_mode: str,
    source_matches: bool,
    compile_result: Any,
    coverage_args: SimpleNamespace,
) -> tuple[dict[str, Any], float]:
    coverage_args = SimpleNamespace(**vars(coverage_args))
    coverage_args.coverage_max_trace_tests = 0
    started = time.monotonic()
    coverage = experiment.semantic.run_source_coverage(
        row,
        calls,
        deployment_mode,
        coverage_args,
        abi=compile_result.abi,
        source_matches=source_matches,
    )
    return coverage, time.monotonic() - started


def run_one(
    row: dict[str, Any],
    budget_seconds: float,
    replay_args: SimpleNamespace,
    coverage_args: SimpleNamespace,
    output_dir: Path,
) -> dict[str, Any]:
    benchmark_id = str(row.get("benchmark_id") or "")
    compile_started = time.monotonic()
    compile_result = experiment.semantic.compile_ground_truth(
        row,
        replay_args.compile_timeout,
    )
    compile_seconds = time.monotonic() - compile_started
    if not compile_result.success:
        raise RuntimeError(
            f"ground-truth compilation failed: {compile_result.error}"
        )
    init_code, deployment_mode, source_matches = experiment.original_init_code(
        row,
        compile_result,
    )
    calls = [
        dict(call)
        for call in experiment.semantic.flatten_test_case_groups(
            row.get("test_cases") or {}
        )
    ]
    calls.sort(key=lambda call: str(call.get("case_id") or ""))
    budgeted = execute_budgeted(
        replay_args,
        benchmark_id,
        init_code,
        calls,
        budget_seconds,
    )
    completed_calls = budgeted.pop("completed_calls")
    coverage, coverage_seconds = coverage_for_prefix(
        row,
        completed_calls,
        deployment_mode,
        source_matches,
        compile_result,
        coverage_args,
    )
    contract_dir = output_dir / "contracts" / benchmark_id
    contract_dir.mkdir(parents=True, exist_ok=True)
    experiment.atomic_write_json(
        contract_dir / "coverage.json",
        experiment.sanitized_coverage(coverage, replay_args.fork_url),
    )
    record = {
        "schema": SCHEMA,
        "benchmark_id": benchmark_id,
        "difficulty": row.get("difficulty_bin"),
        "budget_seconds": budget_seconds,
        "full_checkpoint_count": len(calls),
        "completed_checkpoint_count": budgeted["completed_checkpoint_count"],
        "completed_fraction": (
            budgeted["completed_checkpoint_count"] / len(calls) if calls else None
        ),
        "max_completed_transaction_sequence_depth": max(
            (
                experiment.checkpoint_sequence_depth(call)
                for call in completed_calls
            ),
            default=0,
        ),
        "compile_seconds": compile_seconds,
        "checkpoint_execution": budgeted,
        "coverage_measurement_seconds": coverage_seconds,
        "coverage": experiment.compact_coverage(coverage),
        "submitted_full_suite_coverage": experiment.compact_submitted_coverage(
            row
        ),
        "success": (
            budgeted["validation"]["valid"]
            and budgeted["fatal"] is None
            and budgeted["completed_checkpoint_count"] > 0
        ),
    }
    experiment.atomic_write_json(contract_dir / "result.json", record)
    return record


def attach_full_lcov_reference(
    record: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    benchmark_id = str(record.get("benchmark_id") or "")
    full_result_path = (
        experiment.DEFAULT_OUTPUT
        / "contracts"
        / benchmark_id
        / "result.json"
    )
    if full_result_path.exists():
        full_result = json.loads(full_result_path.read_text(encoding="utf-8"))
        record["full_homogeneous_lcov"] = full_result.get("coverage")
    else:
        record["full_homogeneous_lcov"] = {
            "available": False,
            "reason": "full_homogeneous_lcov_result_missing",
        }
    result_path = output_dir / "contracts" / benchmark_id / "result.json"
    experiment.atomic_write_json(result_path, record)
    return record


def write_summary(output_dir: Path, records: list[dict[str, Any]]) -> None:
    records = sorted(
        records,
        key=lambda record: (
            experiment.DIFFICULTY_ORDER.get(str(record.get("difficulty")), 99),
            str(record.get("benchmark_id") or ""),
        ),
    )
    experiment.atomic_write_json(
        output_dir / "results.json",
        {
            "schema": SCHEMA,
            "contracts": len(records),
            "results": records,
        },
    )
    lines = [
        "# Budgeted Semantic Checkpoint Prefixes",
        "",
        "| Contract | Budget | Completed checkpoints | Time | Prefix LCOV | Full LCOV | Covered lines retained |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        coverage = record.get("coverage") or {}
        full_coverage = record.get("full_homogeneous_lcov") or {}
        covered = coverage.get("lines_covered")
        total = coverage.get("lines_total")
        ratio = coverage.get("line_ratio")
        full_covered = full_coverage.get("lines_covered")
        full_total = full_coverage.get("lines_total")
        full_ratio = full_coverage.get("line_ratio")
        retained = (
            float(covered) / float(full_covered)
            if covered is not None and full_covered
            else None
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.get("benchmark_id") or ""),
                    f"{float(record['budget_seconds']):.0f}s",
                    (
                        f"{record['completed_checkpoint_count']}/"
                        f"{record['full_checkpoint_count']}"
                    ),
                    f"{float(record['checkpoint_execution']['elapsed_seconds']):.3f}s",
                    (
                        f"{covered}/{total} ({100.0 * float(ratio):.1f}%)"
                        if ratio is not None
                        else "NA"
                    ),
                    (
                        f"{full_covered}/{full_total} "
                        f"({100.0 * float(full_ratio):.1f}%)"
                        if full_ratio is not None
                        else "NA"
                    ),
                    (
                        f"{100.0 * retained:.1f}%"
                        if retained is not None
                        else "NA"
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Coverage is measured after the deadline over only checkpoints that "
            "completed before it. Coverage instrumentation time is excluded "
            "from the checkpoint budget. Only homogeneous Foundry LCOV is "
            "reported; contracts rejected by Foundry's source parser remain "
            "unavailable and are not treated as zero coverage.",
            "",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiment.load_semantic()
    dataset = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment.semantic.load_env_file(experiment.ROOT / ".env")
    fork_url = args.fork_url or os.environ.get("ARCHIVE_RPC_URL")
    if not fork_url:
        raise SystemExit("missing ARCHIVE_RPC_URL in .env or --fork-url")
    forge_bin = shutil.which("forge")
    if not forge_bin:
        raise SystemExit("forge is not available on PATH")

    warmup = subprocess.run(
        [forge_bin, "build"],
        cwd=experiment.semantic.FOUNDRY_ROOT,
        capture_output=True,
        text=True,
        timeout=args.compile_timeout,
        check=False,
    )
    if warmup.returncode != 0:
        raise SystemExit(
            "Foundry harness warm-up failed: "
            + (warmup.stderr or warmup.stdout or "").strip()[:2000]
        )

    replay_args = experiment.frozen_replay_args(
        forge_bin,
        fork_url,
        args.compile_timeout,
        max(BUDGETS.values()),
    )
    coverage_args = experiment.frozen_coverage_args(
        forge_bin,
        fork_url,
        args.compile_timeout,
        args.coverage_timeout,
    )
    records: list[dict[str, Any]] = []
    selected_rows = load_rows(dataset, args.id, args.metadata)
    for index, row in enumerate(selected_rows, start=1):
        benchmark_id = str(row.get("benchmark_id") or "")
        result_path = output_dir / "contracts" / benchmark_id / "result.json"
        if result_path.exists() and not args.force:
            record = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            record = run_one(
                row,
                BUDGETS[benchmark_id],
                replay_args,
                coverage_args,
                output_dir,
            )
        record = attach_full_lcov_reference(record, output_dir)
        records.append(record)
        coverage = record.get("coverage") or {}
        print(
            f"[budgeted-prefix] {index}/{len(selected_rows)} {benchmark_id} "
            f"completed={record['completed_checkpoint_count']}/"
            f"{record['full_checkpoint_count']} "
            f"elapsed={record['checkpoint_execution']['elapsed_seconds']:.3f}s "
            f"coverage={coverage.get('line_ratio')} "
            f"backend={coverage.get('kind')} "
            f"success={record.get('success')}",
            flush=True,
        )
        write_summary(output_dir, records)
    write_summary(output_dir, records)
    print(json.dumps({"output_dir": str(output_dir), "contracts": len(records)}))


if __name__ == "__main__":
    main()
