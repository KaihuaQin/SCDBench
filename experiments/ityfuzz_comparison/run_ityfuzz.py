#!/usr/bin/env python3
"""Run bounded, local ItyFuzz campaigns and summarize reproducibility metrics."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


PINNED_ITYFUZZ_COMMIT = "80e3b44170573b592cd53b2c5ee9f36b21032b16"
MULTILINE_DECL_RE = re.compile(r"^(?:library|contract|function)(?:.|\n)*\}$")
SKIPPED_OPCODES = {0x00, 0x5B, 0xFE}  # STOP, JUMPDEST, INVALID
DIFFICULTY_BINS = {"easy", "medium", "hard"}
SOURCE_LINE_DEFINITION = (
    "unique source lines mapped to ItyFuzz-counted runtime instructions. "
    "STOP, JUMPDEST, INVALID, unmapped spans, and whole declaration spans "
    "are excluded to mirror ItyFuzz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ItyFuzz only against locally compiled SCDBench artifacts. The "
            "campaign has a fixed wall-clock budget; coverage is obtained by "
            "replaying the emitted coverage corpus."
        )
    )
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--ityfuzz-repo", type=Path, default=Path("/tmp/ityfuzz"))
    parser.add_argument("--ityfuzz-bin", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--wall-seconds",
        type=float,
        help="Default wall-clock campaign budget per contract.",
    )
    parser.add_argument(
        "--wall-seconds-by-difficulty",
        default="",
        help="Comma-separated per-bin budgets, for example easy=4,medium=9,hard=16.",
    )
    parser.add_argument(
        "--replay-timeout",
        type=float,
        help="Coverage-replay timeout. Defaults to max(60, 2 * wall-seconds).",
    )
    parser.add_argument("--seed", type=int, default=1667840158231589000)
    parser.add_argument("--termination-grace", type=float, default=5.0)
    parser.add_argument(
        "--id",
        action="append",
        default=[],
        help="Run only the selected benchmark ID. May be repeated.",
    )
    parser.add_argument("--limit", type=int, help="Run only the first N selected cases.")
    parser.add_argument(
        "--expected-commit",
        default=PINNED_ITYFUZZ_COMMIT,
        help="Abort if the local ItyFuzz checkout is not at this commit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing per-case output directories.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_difficulty_budgets(raw: str) -> dict[str, float]:
    budgets: dict[str, float] = {}
    if not raw.strip():
        return budgets
    for item in raw.split(","):
        key, separator, value = item.strip().partition("=")
        if not separator or key not in DIFFICULTY_BINS:
            raise ValueError(
                "--wall-seconds-by-difficulty must use easy=N,medium=N,hard=N"
            )
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError(f"invalid wall-clock budget for {key}: {value!r}")
        budgets[key] = parsed
    return budgets


def git_checkout_info(repo: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"repo": str(repo.resolve()), "commit": commit, "dirty": dirty}


def run_bounded(
    command: list[str],
    timeout_seconds: float,
    grace_seconds: float,
    log_path: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_wall = time.time()
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        timed_out = False
        termination_signal: str | None = None
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            termination_signal = "SIGTERM"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                termination_signal = "SIGKILL"
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
    return {
        "command": command,
        "started_unix": started_wall,
        "elapsed_seconds": time.monotonic() - started,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "termination_signal": termination_signal,
        "returncode": returncode,
        "log": str(log_path.resolve()),
    }


def ityfuzz_failure_zero_reason(campaign: dict[str, Any]) -> str | None:
    if campaign.get("timed_out"):
        return None
    log_value = campaign.get("log")
    if not isinstance(log_value, str):
        return None
    log_path = Path(log_value)
    if not log_path.is_file():
        return None
    log = log_path.read_text(encoding="utf-8", errors="replace")
    if "has overflowed its stack" in log:
        return "ItyFuzz stack overflow before completing the campaign budget"
    if (
        "Failed to deploy contract" in log
        and "get runtime bytecode failed" in log
    ):
        return "ItyFuzz could not deploy the contract in its local EVM"
    return None


def canonical_sequence(path: Path) -> tuple[str | None, str | None]:
    transactions: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            return None, f"line {line_no}: invalid JSON: {exc.msg}"
        if not isinstance(payload, dict):
            return None, f"line {line_no}: transaction is not a JSON object"
        transactions.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    if not transactions:
        return None, "empty sequence"
    return "\n".join(transactions) + "\n", None


def export_sequences(corpus_dir: Path, sequence_dir: Path) -> dict[str, Any]:
    sequence_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(corpus_dir.glob("*_replayable")) if corpus_dir.exists() else []
    unique: dict[str, tuple[str, list[str]]] = {}
    rejected: list[dict[str, str]] = []
    for path in candidates:
        sequence, error = canonical_sequence(path)
        if sequence is None:
            rejected.append({"source": str(path), "reason": error or "unknown"})
            continue
        digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        if digest not in unique:
            unique[digest] = (sequence, [])
        unique[digest][1].append(str(path))

    sequence_rows: list[dict[str, Any]] = []
    for index, (digest, (sequence, sources)) in enumerate(sorted(unique.items()), start=1):
        output = sequence_dir / f"sequence_{index:06d}.jsonl"
        output.write_text(sequence, encoding="utf-8")
        transaction_count = len(sequence.splitlines())
        sequence_rows.append(
            {
                "sequence_id": f"sequence_{index:06d}",
                "path": str(output.resolve()),
                "sha256": digest,
                "transaction_count": transaction_count,
                "source_corpus_files": sources,
            }
        )
    manifest = {
        "candidate_replayable_files": len(candidates),
        "unique_sequences": len(sequence_rows),
        "duplicate_sequences": max(0, len(candidates) - len(rejected) - len(sequence_rows)),
        "rejected_sequences": rejected,
        "max_sequence_length": max(
            (row["transaction_count"] for row in sequence_rows), default=0
        ),
        "sequences": sequence_rows,
    }
    write_json(sequence_dir / "manifest.json", manifest)
    return manifest


def instruction_pcs(bytecode_hex: str) -> list[tuple[int, int]]:
    bytecode = bytes.fromhex(bytecode_hex)
    result: list[tuple[int, int]] = []
    pc = 0
    while pc < len(bytecode):
        opcode = bytecode[pc]
        result.append((pc, opcode))
        pc += 2 + opcode - 0x60 if 0x60 <= opcode <= 0x7F else 1
    return result


def decompress_source_map(source_map: str) -> list[tuple[int, int, int | None]]:
    result: list[tuple[int, int, int | None]] = []
    previous = (0, 0, None)
    for index, entry in enumerate(source_map.split(";")):
        fields = entry.split(":")
        if index == 0 and (
            len(fields) < 4 or any(not fields[pos] for pos in range(4))
        ):
            current = (0, 0, None)
        else:
            offset = (
                int(fields[0])
                if len(fields) > 0 and fields[0]
                else previous[0]
            )
            length = (
                int(fields[1])
                if len(fields) > 1 and fields[1]
                else previous[1]
            )
            if len(fields) > 2 and fields[2]:
                parsed_file = int(fields[2])
                file_index = parsed_file if parsed_file >= 0 else None
            else:
                file_index = previous[2]
            current = (offset, length, file_index)
        result.append(current)
        previous = current
    return result


def lines_for_range(source_bytes: bytes, offset: int, length: int) -> set[int]:
    if length <= 0 or offset < 0 or offset >= len(source_bytes):
        return set()
    end = min(len(source_bytes), offset + length)
    newline_offsets = [
        index for index, value in enumerate(source_bytes) if value == ord("\n")
    ]
    start_line = bisect.bisect_left(newline_offsets, offset) + 1
    end_line = bisect.bisect_left(newline_offsets, max(offset, end - 1)) + 1
    return set(range(start_line, end_line + 1))


def source_map_coverable_lines(metadata: dict[str, Any], source_code: str) -> set[int]:
    source_bytes = source_code.encode("utf-8")
    mappings = decompress_source_map(str(metadata["runtime_source_map"]))
    coverable: set[int] = set()
    for instruction_index, (_, opcode) in enumerate(
        instruction_pcs(str(metadata["runtime_bytecode"]))
    ):
        if opcode in SKIPPED_OPCODES or instruction_index >= len(mappings):
            continue
        offset, length, file_index = mappings[instruction_index]
        if file_index != 0 or length <= 0 or offset + length > len(source_bytes):
            continue
        snippet = source_bytes[offset : offset + length].decode("utf-8", errors="replace")
        if MULTILINE_DECL_RE.fullmatch(snippet):
            continue
        coverable.update(lines_for_range(source_bytes, offset, length))
    return coverable


def has_ityfuzz_target(metadata: dict[str, Any]) -> bool:
    abi = metadata.get("ityfuzz_abi")
    if not isinstance(abi, list):
        return False
    return any(
        isinstance(entry, dict)
        and (
            (
                entry.get("type") == "function"
                and entry.get("stateMutability") != "view"
            )
            or entry.get("type") == "fallback"
            or (
                entry.get("type") == "receive"
                and entry.get("stateMutability") == "payable"
            )
        )
        for entry in abi
    )


def zero_coverage(
    metadata: dict[str, Any], source_path: Path, reason: str
) -> dict[str, Any]:
    instructions = instruction_pcs(str(metadata["runtime_bytecode"]))
    source_code = source_path.read_text(encoding="utf-8")
    coverable_lines = source_map_coverable_lines(metadata, source_code)
    branch_total = 2 * sum(opcode == 0x57 for _, opcode in instructions)
    return {
        "available": True,
        "reason": reason,
        "instruction_covered": 0,
        "instruction_total": len(instructions),
        "instruction_coverage": 0.0 if instructions else None,
        "branch_covered": 0,
        "branch_total": branch_total,
        "branch_coverage": 0.0 if branch_total else None,
        "source_line_covered": 0,
        "source_line_total": len(coverable_lines),
        "source_line_coverage": 0.0 if coverable_lines else None,
        "source_line_definition": SOURCE_LINE_DEFINITION,
        "covered_source_lines": [],
        "coverable_source_lines": sorted(coverable_lines),
        "contracts": [],
        "raw_coverage": None,
    }


def aggregate_coverage(
    coverage_path: Path, metadata: dict[str, Any], source_path: Path
) -> dict[str, Any]:
    if not coverage_path.exists():
        return {
            "available": False,
            "reason": "ItyFuzz replay did not produce coverage.json",
        }
    raw = read_json(coverage_path)
    contracts = raw.get("coverage") if isinstance(raw, dict) else None
    if not isinstance(contracts, dict):
        return {"available": False, "reason": "coverage.json has no coverage mapping"}

    instruction_covered = 0
    instruction_total = 0
    branch_covered = 0
    branch_total = 0
    covered_ranges: set[tuple[int, int, int | None]] = set()
    per_contract: list[dict[str, Any]] = []
    for name, result in sorted(contracts.items()):
        if not isinstance(result, dict):
            continue
        instruction_covered += int(result.get("instruction_coverage") or 0)
        instruction_total += int(result.get("total_instructions") or 0)
        branch_covered += int(result.get("branch_coverage") or 0)
        branch_total += int(result.get("total_branches") or 0)
        for item in result.get("covered_code") or []:
            if not isinstance(item, dict):
                continue
            covered_ranges.add(
                (
                    int(item.get("offset") or 0),
                    int(item.get("length") or 0),
                    item.get("file_idx"),
                )
            )
        per_contract.append(
            {
                "name": name,
                "address": result.get("address"),
                "instruction_covered": int(result.get("instruction_coverage") or 0),
                "instruction_total": int(result.get("total_instructions") or 0),
                "branch_covered": int(result.get("branch_coverage") or 0),
                "branch_total": int(result.get("total_branches") or 0),
            }
        )

    source_code = source_path.read_text(encoding="utf-8")
    source_bytes = source_code.encode("utf-8")
    coverable_lines = source_map_coverable_lines(metadata, source_code)
    covered_lines: set[int] = set()
    for offset, length, file_index in covered_ranges:
        if file_index == 0:
            covered_lines.update(lines_for_range(source_bytes, offset, length))
    covered_lines &= coverable_lines

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "available": True,
        "instruction_covered": instruction_covered,
        "instruction_total": instruction_total,
        "instruction_coverage": ratio(instruction_covered, instruction_total),
        "branch_covered": branch_covered,
        "branch_total": branch_total,
        "branch_coverage": ratio(branch_covered, branch_total),
        "source_line_covered": len(covered_lines),
        "source_line_total": len(coverable_lines),
        "source_line_coverage": ratio(len(covered_lines), len(coverable_lines)),
        "source_line_definition": SOURCE_LINE_DEFINITION,
        "covered_source_lines": sorted(covered_lines),
        "coverable_source_lines": sorted(coverable_lines),
        "contracts": per_contract,
        "raw_coverage": str(coverage_path.resolve()),
    }


def weighted_ratio(rows: Iterable[dict[str, Any]], numerator: str, denominator: str) -> float | None:
    row_list = list(rows)
    den = sum(int(row.get(denominator) or 0) for row in row_list)
    if not den:
        return None
    return sum(int(row.get(numerator) or 0) for row in row_list) / den


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [
        case
        for case in cases
        if isinstance(case.get("coverage"), dict)
        and case["coverage"].get("available")
    ]
    return {
        "requested_cases": len(cases),
        "campaigns_started": sum(
            isinstance(case.get("campaign"), dict)
            and not bool(case["campaign"].get("skipped"))
            for case in cases
        ),
        "campaigns_timed_out_as_planned": sum(
            bool(case.get("campaign", {}).get("timed_out")) for case in cases
        ),
        "no_fuzzable_abi_entry_zero_coverage_cases": sum(
            case.get("status") == "measured_zero_no_fuzzable_abi_entry"
            for case in cases
        ),
        "ityfuzz_failure_zero_coverage_cases": sum(
            case.get("status") == "measured_zero_ityfuzz_failure"
            for case in cases
        ),
        "ityfuzz_failure_partial_coverage_cases": sum(
            case.get("status") == "measured_partial_ityfuzz_failure"
            for case in cases
        ),
        "run_failed_cases": sum(
            case.get("status") == "run_failed" for case in cases
        ),
        "coverage_measured_cases": len(measured),
        "source_line_coverage_micro": weighted_ratio(
            (case["coverage"] for case in measured),
            "source_line_covered",
            "source_line_total",
        ),
        "instruction_coverage_micro": weighted_ratio(
            (case["coverage"] for case in measured),
            "instruction_covered",
            "instruction_total",
        ),
        "branch_coverage_micro": weighted_ratio(
            (case["coverage"] for case in measured),
            "branch_covered",
            "branch_total",
        ),
    }


def summarize_by_difficulty(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        difficulty: summarize(
            [case for case in cases if case.get("difficulty_bin") == difficulty]
        )
        for difficulty in sorted(DIFFICULTY_BINS)
    }


def run_case(
    case: dict[str, Any],
    binary: Path,
    checkout: dict[str, Any],
    output_root: Path,
    wall_seconds: float,
    replay_timeout: float,
    seed: int,
    grace: float,
    overwrite: bool,
) -> dict[str, Any]:
    benchmark_id = str(case["benchmark_id"])
    case_output = output_root / benchmark_id
    result_path = case_output / "result.json"
    if result_path.exists() and not overwrite:
        return read_json(result_path)
    if case_output.exists() and overwrite:
        shutil.rmtree(case_output)
    case_output.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(case["artifact_meta"])
    metadata = read_json(metadata_path)
    artifact_dir = Path(case["artifact_dir"])
    fuzz_work = case_output / "fuzz"
    sequence_dir = case_output / "sequences"
    replay_work = case_output / "replay"
    source_path = artifact_dir / "contract.sol"

    if not has_ityfuzz_target(metadata):
        sequence_export = export_sequences(case_output / "no_corpus", sequence_dir)
        coverage = zero_coverage(
            metadata,
            source_path,
            "ABI has no entry fuzzed by default ItyFuzz",
        )
        result = {
            "schema_version": 1,
            "benchmark_id": benchmark_id,
            "difficulty_bin": case.get("difficulty_bin"),
            "status": "measured_zero_no_fuzzable_abi_entry",
            "ityfuzz": checkout,
            "binary": str(binary.resolve()),
            "seed": seed,
            "wall_clock_budget_seconds": wall_seconds,
            "coverage_replay_budget_seconds": replay_timeout,
            "artifact_meta": str(metadata_path.resolve()),
            "runtime_matches_submission": metadata.get("runtime_matches_submission"),
            "campaign": {
                "skipped": True,
                "reason": "no_default_ityfuzz_abi_entry",
                "timeout_seconds": wall_seconds,
            },
            "sequence_export": sequence_export,
            "replay": None,
            "coverage": coverage,
        }
        write_json(result_path, result)
        return result

    target_glob = str(artifact_dir / "*")
    env = dict(os.environ)
    env.setdefault("RUST_LOG", "warn")

    base_command = [
        str(binary),
        "evm",
        "--target",
        target_glob,
        "--target-type",
        "glob",
        "--detectors",
        "",
        "--seed",
        str(seed),
    ]
    campaign_command = base_command + [
        "--work-dir",
        str(fuzz_work),
        "--run-forever",
    ]
    campaign = run_bounded(
        campaign_command,
        wall_seconds,
        grace,
        case_output / "campaign.log",
        env,
    )
    if not campaign["timed_out"]:
        failure_reason = ityfuzz_failure_zero_reason(campaign)
        if failure_reason is None:
            raise RuntimeError(
                "ItyFuzz campaign exited before its wall-clock budget "
                f"with return code {campaign['returncode']}"
            )
        sequence_export = export_sequences(fuzz_work / "corpus", sequence_dir)
        coverage = aggregate_coverage(
            fuzz_work / "coverage.json",
            metadata,
            source_path,
        )
        if coverage.get("available"):
            coverage["partial"] = True
            coverage["partial_reason"] = failure_reason
            status = "measured_partial_ityfuzz_failure"
        else:
            coverage = zero_coverage(metadata, source_path, failure_reason)
            status = "measured_zero_ityfuzz_failure"
        result = {
            "schema_version": 1,
            "benchmark_id": benchmark_id,
            "difficulty_bin": case.get("difficulty_bin"),
            "status": status,
            "ityfuzz": checkout,
            "binary": str(binary.resolve()),
            "seed": seed,
            "wall_clock_budget_seconds": wall_seconds,
            "coverage_replay_budget_seconds": replay_timeout,
            "artifact_meta": str(metadata_path.resolve()),
            "runtime_matches_submission": metadata.get("runtime_matches_submission"),
            "campaign": campaign,
            "sequence_export": sequence_export,
            "replay": None,
            "coverage": coverage,
        }
        write_json(result_path, result)
        return result
    sequence_export = export_sequences(fuzz_work / "corpus", sequence_dir)

    replay: dict[str, Any] | None = None
    if sequence_export["unique_sequences"] == 0:
        coverage = zero_coverage(
            metadata,
            source_path,
            "bounded campaign produced no replayable coverage sequence",
        )
    else:
        replay_command = base_command + [
            "--work-dir",
            str(replay_work),
            "--replay-file",
            str(sequence_dir / "*.jsonl"),
        ]
        replay = run_bounded(
            replay_command,
            replay_timeout,
            grace,
            case_output / "replay.log",
            env,
        )
        if replay["timed_out"] or replay["returncode"] != 0:
            raise RuntimeError(
                "ItyFuzz coverage replay did not complete successfully "
                f"(return code {replay['returncode']}, timed out={replay['timed_out']})"
            )
        coverage = aggregate_coverage(
            replay_work / "coverage.json",
            metadata,
            source_path,
        )
        if not coverage.get("available"):
            raise RuntimeError(str(coverage.get("reason") or "coverage unavailable"))
    result = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "difficulty_bin": case.get("difficulty_bin"),
        "status": "measured" if coverage.get("available") else "campaign_complete",
        "ityfuzz": checkout,
        "binary": str(binary.resolve()),
        "seed": seed,
        "wall_clock_budget_seconds": wall_seconds,
        "coverage_replay_budget_seconds": replay_timeout,
        "artifact_meta": str(metadata_path.resolve()),
        "runtime_matches_submission": metadata.get("runtime_matches_submission"),
        "campaign": campaign,
        "sequence_export": sequence_export,
        "replay": replay,
        "coverage": coverage,
    }
    write_json(result_path, result)
    return result


def main() -> int:
    args = parse_args()
    difficulty_budgets = parse_difficulty_budgets(args.wall_seconds_by_difficulty)
    if args.wall_seconds is None and not difficulty_budgets:
        raise ValueError(
            "provide --wall-seconds or --wall-seconds-by-difficulty"
        )
    if args.wall_seconds is not None and (
        not math.isfinite(args.wall_seconds) or args.wall_seconds <= 0
    ):
        raise ValueError("--wall-seconds must be a positive finite number")
    checkout = git_checkout_info(args.ityfuzz_repo)
    if checkout["commit"] != args.expected_commit:
        raise RuntimeError(
            f"ItyFuzz commit {checkout['commit']} does not match expected {args.expected_commit}"
        )
    binary = (
        args.ityfuzz_bin
        if args.ityfuzz_bin is not None
        else args.ityfuzz_repo / "target/release/ityfuzz"
    )
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(
            f"ItyFuzz binary not found or not executable: {binary}; run cargo build --release"
        )

    prepared = read_json(args.prepared_manifest)
    runnable = [
        case for case in prepared.get("cases", []) if case.get("status") == "prepared"
    ]
    if args.id:
        selected_ids = set(args.id)
        runnable = [
            case for case in runnable if str(case.get("benchmark_id")) in selected_ids
        ]
        found_ids = {str(case.get("benchmark_id")) for case in runnable}
        missing_ids = sorted(selected_ids - found_ids)
        if missing_ids:
            raise ValueError(f"selected benchmark IDs are not prepared: {missing_ids}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        runnable = runnable[: args.limit]
    case_budgets: dict[str, float] = {}
    for case in runnable:
        difficulty = str(case.get("difficulty_bin") or "")
        wall_seconds = difficulty_budgets.get(difficulty, args.wall_seconds)
        if wall_seconds is None:
            raise ValueError(f"no wall-clock budget configured for {difficulty!r}")
        case_budgets[str(case["benchmark_id"])] = wall_seconds
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, case in enumerate(runnable, start=1):
        wall_seconds = case_budgets[str(case["benchmark_id"])]
        replay_timeout = (
            args.replay_timeout
            if args.replay_timeout is not None
            else max(60.0, 2.0 * wall_seconds)
        )
        print(
            f"[{index}/{len(runnable)}] {case['benchmark_id']} "
            f"({wall_seconds:g}s campaign)",
            flush=True,
        )
        try:
            result = run_case(
                case,
                binary,
                checkout,
                args.output_dir,
                wall_seconds,
                replay_timeout,
                args.seed,
                args.termination_grace,
                args.overwrite,
            )
        except Exception as exc:
            result = {
                "schema_version": 1,
                "benchmark_id": case["benchmark_id"],
                "difficulty_bin": case.get("difficulty_bin"),
                "status": "run_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_json(args.output_dir / case["benchmark_id"] / "result.json", result)
            print(f"failed: {result['error']}", file=sys.stderr, flush=True)
        results.append(result)
        write_json(
            args.output_dir / "results.json",
            {
                "schema_version": 1,
                "prepared_manifest": str(args.prepared_manifest.resolve()),
                "ityfuzz": checkout,
                "wall_clock_budget_seconds_default": args.wall_seconds,
                "wall_clock_budget_seconds_by_difficulty": difficulty_budgets,
                "coverage_replay_budget_seconds_override": args.replay_timeout,
                "summary": summarize(results),
                "summary_by_difficulty": summarize_by_difficulty(results),
                "cases": results,
            },
        )
    return 0 if all(result.get("status") != "run_failed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
