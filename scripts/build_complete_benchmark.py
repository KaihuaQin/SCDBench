from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_DATASET = Path("data/scdbench_sample.jsonl")
DEFAULT_CHECKPOINTS = Path("data/scdbench_semantic_checkpoints_v8.json")
DEFAULT_OUTPUT = Path("data/scdbench_complete.jsonl")

SEMANTIC_REPLAY_SCHEMA = "scdbench-semantic-replay-v1"
DEFAULT_REPLAY_ADDRESS = "0x7fa9385be102ac3eac297483dd6233d62b3e1496"
REPLAY_INITIAL_BALANCE_WEI = "340282366920938463463374607431768211455"
REPLAY_CALL_GAS_LIMIT = 30_000_000


METADATA_KEYS = (
    "deployment_mode",
    "constructor_input_count",
    "source_deployed_bytecode_matches_input_logic",
    "abi_function_count",
    "abi_executable_entry_count",
    "functions_with_testcases",
    "entries_with_testcases",
    "calls_generated",
    "calls_total",
    "unsupported_entries",
    "skipped_test_cases",
    "function_signatures",
)

COVERAGE_KEYS = (
    "available",
    "kind",
    "reason",
    "source_file",
    "solc_version",
    "lines",
    "functions",
    "branches",
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def compact_coverage(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"available": False, "reason": "coverage_missing"}
    return {key: deepcopy(raw[key]) for key in COVERAGE_KEYS if key in raw}


def semantic_replay_spec(fork_block: Any) -> Dict[str, Any]:
    return {
        "schema": SEMANTIC_REPLAY_SCHEMA,
        "fork_block": fork_block,
        "default_deployer": DEFAULT_REPLAY_ADDRESS,
        "default_sender": DEFAULT_REPLAY_ADDRESS,
        "default_setup_sender": DEFAULT_REPLAY_ADDRESS,
        "initial_balance_wei": REPLAY_INITIAL_BALANCE_WEI,
        "call_gas_limit": REPLAY_CALL_GAS_LIMIT,
        "execution_order": [
            "deploy_candidate",
            "snapshot",
            "apply_setup_calls",
            "execute_main_call",
            "compare_against_baseline",
            "revert_snapshot",
        ],
        "case_fields": {
            "calldata": "hex calldata for the main call",
            "value": "msg.value in wei for the main call",
            "setup": "optional ordered setup calls, each with calldata and value",
            "baseline.call.ok": "true if the ground-truth execution succeeded, false if it reverted",
            "baseline.call.result": "return data on success or revert data on failure",
            "baseline.logs": "ordered emitted logs, with the evaluated contract normalized to <target>",
            "baseline.storage_changes": "write slots whose before and after values differ",
            "baseline.storage_touched": "write slots touched by the execution, including unchanged writes",
        },
        "comparison_fields": [
            "baseline.call.ok",
            "baseline.call.result",
            "baseline.logs",
            "baseline.storage_changes",
        ],
    }


def normalize_setup_call(setup_call: Dict[str, Any]) -> Dict[str, Any]:
    item = deepcopy(setup_call)
    item.setdefault("sender", DEFAULT_REPLAY_ADDRESS)
    return item


def normalize_case(case: Dict[str, Any], benchmark_id: str) -> Dict[str, Any]:
    item = deepcopy(case)
    required = ("case_id", "calldata", "value", "baseline")
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"{benchmark_id} testcase is missing required fields: {', '.join(missing)}")
    baseline = item.get("baseline")
    if not isinstance(baseline, dict) or not isinstance(baseline.get("call"), dict):
        raise ValueError(f"{benchmark_id} testcase has no baseline call observation")
    item.setdefault("sender", DEFAULT_REPLAY_ADDRESS)
    setup = item.get("setup")
    if isinstance(setup, list):
        item["setup"] = [
            normalize_setup_call(setup_call)
            for setup_call in setup
            if isinstance(setup_call, dict)
        ]
    return item


def grouped_test_cases_to_map(groups: List[Dict[str, Any]], benchmark_id: str) -> Dict[str, Dict[str, Any]]:
    mapped: Dict[str, Dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError(f"{benchmark_id} testcase group is not an object")
        signature = str(group.get("signature") or "")
        if not signature:
            raise ValueError(f"{benchmark_id} testcase group has no signature")
        if signature in mapped:
            raise ValueError(f"{benchmark_id} has duplicate testcase signature group: {signature}")
        mapped[signature] = {
            "signature": signature,
            "entry_type": group.get("entry_type"),
            "state": group.get("state"),
            "func_index": group.get("func_index"),
            "case_count": group.get("case_count", len(group.get("cases") or [])),
            "cases": [
                normalize_case(case, benchmark_id)
                for case in (group.get("cases") or [])
                if isinstance(case, dict)
            ],
        }
    return mapped


def load_checkpoint_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    if not isinstance(payload.get("results"), list):
        raise ValueError(f"{path} is missing a results list")
    return payload


def index_checkpoints(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    duplicates: List[str] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        benchmark_id = str(item.get("benchmark_id") or "")
        if not benchmark_id:
            continue
        if benchmark_id in index:
            duplicates.append(benchmark_id)
        index[benchmark_id] = item
    if duplicates:
        dup_preview = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"duplicate checkpoint benchmark_id values: {dup_preview}")
    return index


def add_semantic_fields(
    row: Dict[str, Any],
    checkpoint: Dict[str, Any],
    checkpoint_schema: Any,
    fork_block: Any,
) -> Dict[str, Any]:
    merged = dict(row)
    if checkpoint.get("error"):
        raise ValueError(f"{row.get('benchmark_id')} checkpoint has error={checkpoint.get('error')}")

    test_cases = checkpoint.get("test_cases")
    if not isinstance(test_cases, list):
        raise ValueError(f"{row.get('benchmark_id')} checkpoint has no grouped test_cases list")

    merged["semantic_schema"] = checkpoint_schema
    merged["semantic_fork_block"] = fork_block
    merged["semantic_replay"] = semantic_replay_spec(fork_block)
    merged["semantic_metadata"] = {
        key: deepcopy(checkpoint[key])
        for key in METADATA_KEYS
        if key in checkpoint
    }
    merged["semantic_coverage"] = compact_coverage(checkpoint.get("coverage"))
    merged["test_cases"] = grouped_test_cases_to_map(test_cases, str(row.get("benchmark_id") or ""))
    return merged


def count_cases(groups: Any) -> int:
    if isinstance(groups, dict):
        iterable = groups.values()
    elif isinstance(groups, list):
        iterable = groups
    else:
        return 0
    total = 0
    for group in iterable:
        if isinstance(group, dict) and isinstance(group.get("cases"), list):
            total += len(group["cases"])
    return total


def iter_testcase_groups(groups: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(groups, dict):
        for signature, group in groups.items():
            if isinstance(group, dict):
                item = group
                item.setdefault("signature", signature)
                yield item
    elif isinstance(groups, list):
        for group in groups:
            if isinstance(group, dict):
                yield group


def build_summary(
    rows: List[Dict[str, Any]],
    output: Path,
    checkpoint_payload: Dict[str, Any],
) -> Dict[str, Any]:
    by_difficulty: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "contracts": 0,
        "test_cases": 0,
        "function_groups": 0,
        "coverage_available": 0,
    })
    case_kinds: Counter[str] = Counter()
    entry_types: Counter[str] = Counter()
    unsupported_entries = 0
    skipped_test_cases = 0

    for row in rows:
        difficulty = str(row.get("difficulty_bin") or "unknown")
        groups = row.get("test_cases") or []
        by_difficulty[difficulty]["contracts"] += 1
        by_difficulty[difficulty]["test_cases"] += count_cases(groups)
        by_difficulty[difficulty]["function_groups"] += len(groups) if isinstance(groups, (dict, list)) else 0
        if (row.get("semantic_coverage") or {}).get("available"):
            by_difficulty[difficulty]["coverage_available"] += 1
        meta = row.get("semantic_metadata") or {}
        unsupported_entries += len(meta.get("unsupported_entries") or [])
        skipped_test_cases += len(meta.get("skipped_test_cases") or [])
        for group in iter_testcase_groups(groups):
            entry_types[str(group.get("entry_type") or "unknown")] += 1
            for case in group.get("cases") or []:
                if isinstance(case, dict):
                    case_kinds[str(case.get("case_kind") or "unknown")] += 1

    ordered_difficulty = {
        key: by_difficulty[key]
        for key in ("easy", "medium", "hard")
        if key in by_difficulty
    }
    for key in sorted(k for k in by_difficulty if k not in ordered_difficulty):
        ordered_difficulty[key] = by_difficulty[key]

    return {
        "schema": "scdbench-complete-jsonl-v1",
        "semantic_replay_schema": SEMANTIC_REPLAY_SCHEMA,
        "output": str(output),
        "source_dataset": str(DEFAULT_DATASET),
        "source_checkpoints_schema": checkpoint_payload.get("schema"),
        "fork_block": checkpoint_payload.get("fork_block"),
        "contracts": len(rows),
        "test_cases": sum(count_cases(row.get("test_cases")) for row in rows),
        "function_groups": sum(len(row.get("test_cases") or {}) for row in rows),
        "coverage_available": sum(1 for row in rows if (row.get("semantic_coverage") or {}).get("available")),
        "unsupported_entries": unsupported_entries,
        "skipped_test_cases": skipped_test_cases,
        "entry_types": dict(sorted(entry_types.items())),
        "case_kinds": dict(sorted(case_kinds.items())),
        "by_difficulty": ordered_difficulty,
    }


def build_complete_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    rows = read_jsonl(args.dataset)
    checkpoint_payload = load_checkpoint_payload(args.checkpoints)
    checkpoint_index = index_checkpoints(checkpoint_payload)

    merged_rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for row in rows:
        benchmark_id = str(row.get("benchmark_id") or "")
        checkpoint = checkpoint_index.get(benchmark_id)
        if checkpoint is None:
            missing.append(benchmark_id)
            continue
        merged_rows.append(
            add_semantic_fields(
                row=row,
                checkpoint=checkpoint,
                checkpoint_schema=checkpoint_payload.get("schema"),
                fork_block=checkpoint_payload.get("fork_block"),
            )
        )

    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"missing checkpoints for {len(missing)} benchmark rows: {preview}")

    extra = sorted(set(checkpoint_index) - {str(row.get("benchmark_id") or "") for row in rows})
    if extra and args.strict:
        preview = ", ".join(extra[:10])
        raise ValueError(f"checkpoint file contains {len(extra)} extra benchmark ids: {preview}")

    write_jsonl(args.output, merged_rows)
    summary = build_summary(merged_rows, args.output, checkpoint_payload)
    summary["source_dataset"] = str(args.dataset)
    summary["source_checkpoints"] = str(args.checkpoints)
    summary["extra_checkpoint_ids"] = len(extra)

    summary_path = args.summary or args.output.with_name(args.output.stem + "_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge SCDBench benchmark rows with grouped semantic test cases and checkpoints."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if the checkpoint file contains benchmark ids not present in the dataset.",
    )
    return parser.parse_args()


def main() -> None:
    summary = build_complete_dataset(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
