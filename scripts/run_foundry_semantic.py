#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

from eth_abi import encode as abi_encode
from eth_utils import keccak

try:
    from evaluate_model_outputs import compile_ground_truth, function_abi_map
    from foundry_compile import CompileResult, extract_solc_version
except ImportError:
    from scripts.evaluate_model_outputs import compile_ground_truth, function_abi_map
    from scripts.foundry_compile import CompileResult, extract_solc_version


FOUNDRY_ROOT = Path(__file__).resolve().parent / "foundry_semantic"
DEFAULT_BLOCK_NUMBER = 24136053
DEFAULT_DATASET = Path("../dataset/scdbench.jsonl")
SCHEMA = "scdbench-foundry-semantic-v8"
COVERAGE_SCHEMA = "scdbench-source-coverage-v4"
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
BRANCH_CASE_VARIANTS = ("zero_empty", "one_true", "small_threshold", "moderate_threshold")
SETUP_NAME_HINTS = (
    "initialize",
    "init",
    "setup",
    "set",
    "grant",
    "approve",
    "mint",
    "deposit",
    "enable",
    "add",
    "register",
    "configure",
    "create",
    "open",
    "unpause",
)
AUTHORITY_PARAM_HINTS = (
    "owner",
    "admin",
    "authority",
    "governance",
    "governor",
    "manager",
    "operator",
    "controller",
    "guardian",
    "pauser",
    "minter",
    "upgrader",
    "caller",
)
CONSTRUCTOR_CODE_ADDRESS_HINTS = (
    "token",
    "asset",
    "feed",
    "aggregator",
    "oracle",
    "router",
    "factory",
    "pair",
    "pool",
    "vault",
    "registry",
    "config",
    "implementation",
    "proxy",
    "weth",
    "wsteth",
    "steth",
)
CONSTRUCTOR_DURATION_HINTS = (
    "duration",
    "period",
    "delay",
    "interval",
    "cliff",
    "lock",
    "vesting",
    "timelock",
)
CONSTRUCTOR_TIMESTAMP_START_HINTS = (
    "start",
    "begin",
    "opening",
    "launch",
)
CONSTRUCTOR_TIMESTAMP_END_HINTS = (
    "end",
    "close",
    "closing",
    "expiry",
    "expire",
    "deadline",
    "until",
)
CONSTRUCTOR_AMOUNT_HINTS = (
    "amount",
    "supply",
    "cap",
    "rate",
    "price",
    "fee",
    "limit",
    "threshold",
    "reserve",
    "liquidity",
    "share",
)
COVERAGE_ROLE_NAMES = (
    "DEFAULT_ADMIN_ROLE",
    "ADMIN_ROLE",
    "OWNER_ROLE",
    "PAUSER_ROLE",
    "MINTER_ROLE",
    "BURNER_ROLE",
    "UPGRADER_ROLE",
    "OPERATOR_ROLE",
    "MANAGER_ROLE",
    "KEEPER_ROLE",
    "MAINTAINER_ROLE",
    "GOVERNOR_ROLE",
    "SETTER_ROLE",
    "CONTROLLER_ROLE",
    "GUARDIAN_ROLE",
)
SENDER_BALANCE_WEI = 340282366920938463463374607431768211455
COVERAGE_TIMESTAMP = 4102444800
COVERAGE_BLOCK_NUMBER = 100000000
COVERAGE_EOA_SENDERS = (
    "0x1000000000000000000000000000000000000001",
    "0x2000000000000000000000000000000000000002",
    "0x3000000000000000000000000000000000000003",
    "0x4000000000000000000000000000000000000004",
    "0x5000000000000000000000000000000000000005",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Foundry-only semantic testcase generation and replay for SCDBench.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate ground-truth testcase suites with forge test.")
    gen.add_argument("--dataset", action="append", type=Path, default=None)
    gen.add_argument("--output", type=Path, required=True)
    add_common_filters(gen)
    add_common_runtime(gen)
    gen.add_argument("--runs", type=int, default=10)
    gen.add_argument("--compile-timeout", type=int, default=60)
    gen.add_argument("--max-dynamic-array-len", type=int, default=3)
    gen.add_argument("--max-fixed-array-len", type=int, default=16)
    gen.add_argument(
        "--branch-cases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add deterministic boundary-value cases on top of --runs random cases.",
    )
    gen.add_argument(
        "--static-hint-cases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add cases from constants recovered by simple source/bytecode static analysis.",
    )
    gen.add_argument("--max-static-values-per-type", type=int, default=12)
    gen.add_argument("--max-static-hint-cases-per-function", type=int, default=16)
    gen.add_argument("--max-static-int-abs", type=int, default=10**24)
    gen.add_argument(
        "--stateful-setup-cases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add cases that run a deterministic setup call before the measured target call.",
    )
    gen.add_argument("--max-setup-functions", type=int, default=3)
    gen.add_argument("--max-stateful-cases-per-function", type=int, default=2)

    rep = sub.add_parser("replay", help="Replay stored testcase suites against model fuzz targets with forge test.")
    rep.add_argument("--targets", type=Path, required=True)
    rep.add_argument("--testcases", type=Path, required=True, help="Checkpoint JSON or complete benchmark JSONL.")
    rep.add_argument("--output", type=Path, required=True)
    rep.add_argument(
        "--matched-functions-only",
        action="store_true",
        help="Replay only function signatures present in matched_signatures; exclude fallback and receive cases.",
    )
    add_common_filters(rep)
    add_common_runtime(rep)

    cov = sub.add_parser("coverage", help="Recompute source coverage for existing ground-truth checkpoint suites.")
    cov.add_argument("--checkpoints", type=Path, required=True)
    cov.add_argument("--output", type=Path, required=True)
    cov.add_argument("--dataset", action="append", type=Path, default=None)
    cov.add_argument("--compile-timeout", type=int, default=60)
    cov.add_argument("--save-every", type=int, default=10)
    cov.add_argument(
        "--recompute-successes",
        action="store_true",
        help="Recompute coverage even for records that already have successful current-schema coverage.",
    )
    cov.add_argument("--max-dynamic-array-len", type=int, default=3)
    cov.add_argument("--max-fixed-array-len", type=int, default=16)
    cov.add_argument("--max-static-values-per-type", type=int, default=12)
    cov.add_argument("--max-static-int-abs", type=int, default=10**24)
    add_common_filters(cov)
    add_common_runtime(cov)

    args = parser.parse_args()
    if getattr(args, "dataset", None):
        args.datasets = [p.expanduser().resolve() for p in args.dataset]
    else:
        args.datasets = [DEFAULT_DATASET.resolve()]
    for name in ("output", "targets", "testcases", "checkpoints", "ids_file"):
        value = getattr(args, name, None)
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    return args


def add_common_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--difficulty", action="append", default=None)
    parser.add_argument("--id", action="append", default=None)
    parser.add_argument("--ids-file", type=Path, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-errors", action="store_true")


def add_common_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fork-url", default=None)
    parser.add_argument("--fork-block", type=int, default=DEFAULT_BLOCK_NUMBER)
    parser.add_argument("--cast-bin", default="cast")
    parser.add_argument("--forge-bin", default="forge")
    parser.add_argument("--run-coverage", action="store_true")
    parser.add_argument("--coverage-report", default="bytecode")
    parser.add_argument(
        "--coverage-synthetic-constructor-args",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For source coverage only, deploy instrumented source with deterministic synthetic "
            "constructor arguments when exact source creation is unavailable but runtime logic matches."
        ),
    )
    parser.add_argument("--coverage-batch-size", type=int, default=25)
    parser.add_argument("--coverage-call-gas-limit", type=int, default=5_000_000)
    parser.add_argument(
        "--coverage-sender-variants",
        type=int,
        default=2,
        help="Maximum sender variants per coverage call: address(this), then address arguments when present.",
    )
    parser.add_argument(
        "--coverage-auth-storage-slots",
        type=int,
        default=16,
        help=(
            "Coverage-only boost: also replay calls against a second deployment whose first N storage "
            "slots are set to address(this), which helps exercise owner/admin-gated paths."
        ),
    )
    parser.add_argument(
        "--coverage-legacy-solidity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attempt Foundry coverage for Solidity 0.4.x contracts instead of skipping them.",
    )
    parser.add_argument(
        "--coverage-ir-minimum-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry forge coverage with --ir-minimum when the first run hits stack-too-deep compilation failures.",
    )
    parser.add_argument(
        "--coverage-allow-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow replay tests to fail during coverage collection and parse LCOV if Foundry emits it.",
    )
    parser.add_argument(
        "--coverage-trace-timeout",
        type=int,
        default=45,
        help="Per-test timeout for Foundry debug-dump fallback coverage after LCOV parser failures.",
    )
    parser.add_argument(
        "--coverage-max-trace-tests",
        type=int,
        default=8,
        help="Maximum replay test functions to execute under debug-dump fallback for one contract.",
    )
    parser.add_argument(
        "--coverage-augment-calls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add coverage-only ABI-valid branch/static/stateful calls beyond stored checkpoint cases.",
    )
    parser.add_argument(
        "--coverage-augment-fallback-calls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add coverage-only fallback calldata variants, including large payloads for fallback dispatchers.",
    )
    parser.add_argument("--coverage-augment-runs", type=int, default=24)
    parser.add_argument("--coverage-augment-static-cases-per-function", type=int, default=32)
    parser.add_argument("--coverage-augment-setup-functions", type=int, default=6)
    parser.add_argument(
        "--coverage-synthetic-setup-functions",
        type=int,
        default=8,
        help="Coverage-only setup-like ABI functions called with owner/mock-friendly arguments before replay.",
    )
    parser.add_argument("--coverage-augment-stateful-cases-per-function", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Maximum semantic calls per Foundry test transaction. Chunking prevents aggregate harness OOG.",
    )
    parser.add_argument(
        "--value-wei",
        type=int,
        default=0,
        help="If positive, use this fixed msg.value for all payable testcases. Default: generate varied values.",
    )
    parser.add_argument("--max-msg-value-wei", type=int, default=10**17)


def parse_dotenv_line(raw: str) -> Optional[tuple[str, str]]:
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_env_file(path: Path) -> int:
    if not path.exists():
        return 0
    loaded = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            parsed = parse_dotenv_line(raw)
            if parsed is None:
                continue
            key, value = parsed
            if key not in os.environ:
                os.environ[key] = value
                loaded += 1
    return loaded


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                yield json.loads(raw)


def load_id_filter(args: argparse.Namespace) -> set[str] | None:
    ids = set(args.id or [])
    if args.ids_file:
        with args.ids_file.open("r", encoding="utf-8") as f:
            for raw in f:
                value = raw.strip()
                if value and not value.startswith("#"):
                    ids.add(value)
    return ids or None


def load_dataset(args: argparse.Namespace) -> list[Dict[str, Any]]:
    split_filter = set(args.split or [])
    difficulty_filter = set(args.difficulty or [])
    id_filter = load_id_filter(args)
    rows: list[Dict[str, Any]] = []
    for path in args.datasets:
        for line_no, row in enumerate(read_jsonl(path), start=1):
            if split_filter and row.get("split") not in split_filter:
                continue
            if difficulty_filter and row.get("difficulty_bin") not in difficulty_filter:
                continue
            if id_filter is not None and row.get("benchmark_id") not in id_filter:
                continue
            row = dict(row)
            row["_dataset_path"] = str(path)
            row["_dataset_line"] = line_no
            rows.append(row)
    return slice_rows(rows, args)


def slice_rows(rows: list[Dict[str, Any]], args: argparse.Namespace) -> list[Dict[str, Any]]:
    rows.sort(key=lambda r: str(r.get("benchmark_id") or ""))
    if args.offset:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def load_targets(path: Path, args: argparse.Namespace) -> list[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    difficulty_filter = set(args.difficulty or [])
    id_filter = load_id_filter(args)
    out: list[Dict[str, Any]] = []
    for diff in ("easy", "medium", "hard"):
        if difficulty_filter and diff not in difficulty_filter:
            continue
        for row in payload.get(diff, []) or []:
            if id_filter is not None and row.get("benchmark_id") not in id_filter:
                continue
            row = dict(row)
            row["difficulty"] = row.get("difficulty") or diff
            out.append(row)
    return slice_rows(out, args)


def load_existing_results(path: Path, retry_errors: bool) -> tuple[list[Dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [], set()
    raw = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return [], set()
    results = [item for item in raw if isinstance(item, dict)]
    completed: set[str] = set()
    for item in results:
        bid = item.get("benchmark_id")
        if not bid:
            continue
        if retry_errors and item.get("error"):
            continue
        completed.add(str(bid))
    if retry_errors:
        results = [item for item in results if str(item.get("benchmark_id") or "") in completed]
    return results, completed


def write_results(path: Path, args: argparse.Namespace, results: Sequence[Dict[str, Any]]) -> None:
    payload = {
        "schema": SCHEMA,
        "command": args.command,
        "generation_settings": {
            "datasets": [str(path) for path in getattr(args, "datasets", [])],
            "split": args.split,
            "difficulty": args.difficulty,
            "offset": args.offset,
            "limit": args.limit,
            "retry_errors": args.retry_errors,
            "fork_block": args.fork_block,
            "run_coverage": args.run_coverage,
            "coverage_report": args.coverage_report if args.run_coverage else None,
            "coverage_schema": COVERAGE_SCHEMA,
            "coverage_synthetic_constructor_args": getattr(args, "coverage_synthetic_constructor_args", None),
            "coverage_batch_size": getattr(args, "coverage_batch_size", None),
            "coverage_call_gas_limit": getattr(args, "coverage_call_gas_limit", None),
            "coverage_sender_variants": getattr(args, "coverage_sender_variants", None),
            "coverage_auth_storage_slots": getattr(args, "coverage_auth_storage_slots", None),
            "coverage_legacy_solidity": getattr(args, "coverage_legacy_solidity", None),
            "coverage_recompute_successes": getattr(args, "recompute_successes", None),
            "coverage_ir_minimum_retry": getattr(args, "coverage_ir_minimum_retry", None),
            "coverage_allow_failure": getattr(args, "coverage_allow_failure", None),
            "coverage_trace_timeout": getattr(args, "coverage_trace_timeout", None),
            "coverage_max_trace_tests": getattr(args, "coverage_max_trace_tests", None),
            "coverage_augment_calls": getattr(args, "coverage_augment_calls", None),
            "coverage_augment_runs": getattr(args, "coverage_augment_runs", None),
            "coverage_augment_static_cases_per_function": getattr(args, "coverage_augment_static_cases_per_function", None),
            "coverage_augment_setup_functions": getattr(args, "coverage_augment_setup_functions", None),
            "coverage_augment_stateful_cases_per_function": getattr(args, "coverage_augment_stateful_cases_per_function", None),
            "timeout": args.timeout,
            "seed": args.seed,
            "chunk_size": args.chunk_size,
            "value_wei": args.value_wei,
            "max_msg_value_wei": args.max_msg_value_wei,
            "runs": getattr(args, "runs", None),
            "compile_timeout": getattr(args, "compile_timeout", None),
            "max_dynamic_array_len": getattr(args, "max_dynamic_array_len", None),
            "max_fixed_array_len": getattr(args, "max_fixed_array_len", None),
            "branch_cases": getattr(args, "branch_cases", None),
            "static_hint_cases": getattr(args, "static_hint_cases", None),
            "max_static_values_per_type": getattr(args, "max_static_values_per_type", None),
            "max_static_hint_cases_per_function": getattr(args, "max_static_hint_cases_per_function", None),
            "max_static_int_abs": getattr(args, "max_static_int_abs", None),
            "stateful_setup_cases": getattr(args, "stateful_setup_cases", None),
            "max_setup_functions": getattr(args, "max_setup_functions", None),
            "max_stateful_cases_per_function": getattr(args, "max_stateful_cases_per_function", None),
        },
        "fork_block": args.fork_block,
        "coverage_report": args.coverage_report if args.run_coverage else None,
        "summary": summarize(results),
        "results": list(results),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in results if not r.get("error")]
    return {
        "contracts": len(results),
        "valid_contracts": len(valid),
        "calls_total": sum(int(r.get("calls_total", count_test_cases(r))) for r in valid),
        "calls_matched": sum(int(r.get("calls_matched", 0)) for r in valid),
        "functions_total": sum(int(r.get("functions_total", 0)) for r in valid),
        "functions_equivalent": sum(int(r.get("functions_equivalent", 0)) for r in valid),
    }


def count_test_cases(record: Dict[str, Any]) -> int:
    return len(flatten_test_case_groups(record.get("test_cases") or []))


def strip_solc_metadata(bytecode_hex: str) -> str:
    body = str(bytecode_hex or "").strip().removeprefix("0x").lower()
    if len(body) < 4:
        return body
    try:
        meta_len = int(body[-4:], 16)
    except ValueError:
        return body
    total_meta_hex = meta_len * 2 + 4
    if total_meta_hex >= len(body):
        return body
    start = len(body) - total_meta_hex
    try:
        header_byte = int(body[start : start + 2], 16)
    except ValueError:
        return body
    return body[:start] if 0xA0 <= header_byte <= 0xBF else body


def make_init_code_from_runtime(runtime_hex: str) -> str:
    body = runtime_hex.removeprefix("0x")
    code_bytes = bytes.fromhex(body)
    n = len(code_bytes)

    def push_bytes(value: int) -> bytes:
        raw = value.to_bytes((value.bit_length() + 7) // 8 or 1, "big")
        return bytes([0x5F + len(raw)]) + raw

    prefix = b""
    for _ in range(4):
        prefix = push_bytes(n) + push_bytes(len(prefix)) + push_bytes(0) + bytes([0x39]) + push_bytes(n) + push_bytes(0) + bytes([0xF3])
        new_prefix = push_bytes(n) + push_bytes(len(prefix)) + push_bytes(0) + bytes([0x39]) + push_bytes(n) + push_bytes(0) + bytes([0xF3])
        if len(new_prefix) == len(prefix):
            prefix = new_prefix
            break
        prefix = new_prefix
    return "0x" + (prefix + code_bytes).hex()


def constructor_input_count(abi: Sequence[Dict[str, Any]]) -> int:
    for item in abi:
        if isinstance(item, dict) and item.get("type") == "constructor":
            inputs = item.get("inputs") or []
            return len(inputs) if isinstance(inputs, list) else 0
    return 0


def source_bytecode_matches_input(row: Dict[str, Any], deployed_bytecode: str) -> bool:
    return strip_solc_metadata(str(row.get("bytecode") or "")) == strip_solc_metadata(deployed_bytecode)


def split_array_type(typ: str) -> tuple[str, list[Optional[int]]]:
    dims_raw = re.findall(r"\[[0-9]*\]", typ)
    if not dims_raw:
        return typ, []
    base = typ[: -sum(len(dim) for dim in dims_raw)]
    dims: list[Optional[int]] = []
    for dim in dims_raw:
        body = dim[1:-1]
        dims.append(int(body) if body else None)
    return base, dims


def array_suffix(dims: Sequence[Optional[int]]) -> str:
    return "".join(f"[{'' if dim is None else dim}]" for dim in dims)


def is_contract_abi_type(entry: Dict[str, Any], base_type: str) -> bool:
    internal_type = str(entry.get("internalType") or "")
    if not IDENTIFIER_RE.fullmatch(base_type):
        return False
    return internal_type.startswith(("contract ", "interface ", "library "))


def canonical_abi_type(entry: Dict[str, Any]) -> str:
    raw_type = str(entry.get("type") or "")
    if raw_type.startswith("tuple"):
        suffix = raw_type[len("tuple") :]
        components = entry.get("components") or []
        inner = ",".join(canonical_abi_type(c) for c in components if isinstance(c, dict))
        return f"({inner}){suffix}"
    base, dims = split_array_type(raw_type)
    if is_contract_abi_type(entry, base):
        return "address" + array_suffix(dims)
    return raw_type


def function_signature_from_abi(entry: Dict[str, Any]) -> str:
    inputs = entry.get("inputs") or []
    types = [canonical_abi_type(inp) for inp in inputs if isinstance(inp, dict)]
    return f"{entry.get('name')}({','.join(types)})"


def deterministic_bytes(seed: int, size: int) -> bytes:
    import hashlib

    data = bytearray()
    cur = hashlib.sha256(str(seed).encode()).digest()
    while len(data) < size:
        data.extend(cur)
        cur = hashlib.sha256(cur).digest()
    return bytes(data[:size])


def stable_word(seed: int) -> int:
    return int.from_bytes(deterministic_bytes(seed, 8), "big")


def gen_scalar_arg(typ: str, seed: int) -> Any:
    typ = (typ or "").strip()
    x = stable_word(seed)
    if typ.startswith("uint"):
        bits = int(typ[4:] or 256)
        return x % min(1000, 2**bits)
    if typ.startswith("int"):
        bits = int(typ[3:] or 256)
        lo = -(2 ** (bits - 1))
        hi = 2 ** (bits - 1) - 1
        value = (x % 2001) - 1000
        return max(lo, min(hi, value))
    if typ == "bool":
        return bool(x % 2)
    if typ == "address":
        addr = bytearray(deterministic_bytes(seed, 20))
        if all(b == 0 for b in addr):
            addr[-1] = 1
        return "0x" + bytes(addr).hex()
    if typ.startswith("bytes") and typ != "bytes":
        size = int(typ[5:])
        return deterministic_bytes(seed, size)
    if typ == "bytes":
        size = stable_word(seed + 17) % 33
        return deterministic_bytes(seed, size)
    if typ == "string":
        variants = ["", "scdbench", f"case-{seed % 1000}", "Solidity"]
        return variants[stable_word(seed + 29) % len(variants)]
    raise ValueError(f"unsupported_scalar_type:{typ}")


def max_uint_for_bits(bits: int) -> int:
    return (1 << bits) - 1


def int_bounds(bits: int) -> tuple[int, int]:
    return (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)


def branch_scalar_arg(typ: str, variant: int, seed: int) -> Any:
    typ = (typ or "").strip()
    if typ.startswith("uint"):
        bits = int(typ[4:] or 256)
        maxv = max_uint_for_bits(bits)
        values = [0, 1, min(maxv, 2), min(maxv, 16)]
        return values[variant % len(values)]
    if typ.startswith("int"):
        bits = int(typ[3:] or 256)
        lo, hi = int_bounds(bits)
        values = [0, 1, min(hi, 2), max(lo, -2)]
        return values[variant % len(values)]
    if typ == "bool":
        return bool(variant % 2)
    if typ == "address":
        values = [
            "0x" + "00" * 20,
            "0x" + "00" * 19 + "01",
            "0x" + "ff" * 20,
            "0x" + deterministic_bytes(seed, 20).hex(),
        ]
        return values[variant % len(values)]
    if typ.startswith("bytes") and typ != "bytes":
        size = int(typ[5:])
        values = [
            bytes(size),
            (b"\x01" + bytes(max(0, size - 1)))[:size],
            (b"\x02" + bytes(max(0, size - 1)))[:size],
            deterministic_bytes(seed, size),
        ]
        return values[variant % len(values)]
    if typ == "bytes":
        max_len = 16
        values = [
            b"",
            b"\x00",
            b"\x02" * 2,
            deterministic_bytes(seed, max_len),
        ]
        return values[variant % len(values)]
    if typ == "string":
        values = ["", "a", "aa", f"case-{seed % 1000}"]
        return values[variant % len(values)]
    raise ValueError(f"unsupported_scalar_type:{typ}")


def gen_abi_value(entry: Dict[str, Any], seed: int, run_index: int, args: argparse.Namespace) -> Any:
    typ = str(entry.get("type") or "")
    base, dims = split_array_type(typ)
    if dims:
        base_entry = dict(entry)
        base_entry["type"] = "address" if is_contract_abi_type(entry, base) else base
        return gen_array_value(base_entry, dims, seed, run_index, args)
    if is_contract_abi_type(entry, base):
        return gen_scalar_arg("address", seed)
    if base == "tuple":
        components = entry.get("components") or []
        if not isinstance(components, list):
            raise ValueError("tuple_missing_components")
        return tuple(
            gen_abi_value(component, seed + (idx + 1) * 101, run_index, args)
            for idx, component in enumerate(components)
            if isinstance(component, dict)
        )
    return gen_scalar_arg(base, seed)


def gen_branch_abi_value(entry: Dict[str, Any], seed: int, variant: int, args: argparse.Namespace) -> Any:
    typ = str(entry.get("type") or "")
    base, dims = split_array_type(typ)
    if dims:
        base_entry = dict(entry)
        base_entry["type"] = "address" if is_contract_abi_type(entry, base) else base
        return gen_branch_array_value(base_entry, dims, seed, variant, args)
    if is_contract_abi_type(entry, base):
        return branch_scalar_arg("address", variant, seed)
    if base == "tuple":
        components = entry.get("components") or []
        if not isinstance(components, list):
            raise ValueError("tuple_missing_components")
        return tuple(
            gen_branch_abi_value(component, seed + (idx + 1) * 101, (variant + idx) % len(BRANCH_CASE_VARIANTS), args)
            for idx, component in enumerate(components)
            if isinstance(component, dict)
        )
    return branch_scalar_arg(base, variant, seed)


def gen_array_value(
    base_entry: Dict[str, Any],
    dims: Sequence[Optional[int]],
    seed: int,
    run_index: int,
    args: argparse.Namespace,
) -> list[Any]:
    outer_dim = dims[-1]
    inner_dims = list(dims[:-1])
    if outer_dim is None:
        length = run_index % (max(0, args.max_dynamic_array_len) + 1)
    else:
        if outer_dim > args.max_fixed_array_len:
            raise ValueError(f"fixed_array_too_large:{outer_dim}")
        length = outer_dim
    values = []
    for idx in range(length):
        item_seed = seed + (idx + 1) * 1009 + len(dims) * 53
        if inner_dims:
            values.append(gen_array_value(base_entry, inner_dims, item_seed, run_index, args))
        else:
            values.append(gen_abi_value(base_entry, item_seed, run_index, args))
    return values


def gen_branch_array_value(
    base_entry: Dict[str, Any],
    dims: Sequence[Optional[int]],
    seed: int,
    variant: int,
    args: argparse.Namespace,
) -> list[Any]:
    outer_dim = dims[-1]
    inner_dims = list(dims[:-1])
    if outer_dim is None:
        max_len = max(0, args.max_dynamic_array_len)
        length_values = [0, 1, max_len, min(2, max_len)]
        length = length_values[variant % len(length_values)]
    else:
        if outer_dim > args.max_fixed_array_len:
            raise ValueError(f"fixed_array_too_large:{outer_dim}")
        length = outer_dim
    values = []
    for idx in range(length):
        item_seed = seed + (idx + 1) * 1009 + len(dims) * 53
        item_variant = (variant + idx) % len(BRANCH_CASE_VARIANTS)
        if inner_dims:
            values.append(gen_branch_array_value(base_entry, inner_dims, item_seed, item_variant, args))
        else:
            values.append(gen_branch_abi_value(base_entry, item_seed, item_variant, args))
    return values


def jsonable_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return "0x" + value.hex()
    if isinstance(value, tuple):
        return [jsonable_value(item) for item in value]
    if isinstance(value, list):
        return [jsonable_value(item) for item in value]
    return value


def function_selector(signature: str) -> bytes:
    return keccak(text=signature)[:4]


def abi_state_mutability(entry: Dict[str, Any]) -> str:
    state = str(entry.get("stateMutability") or "").lower()
    if not state and entry.get("payable") is True:
        return "payable"
    if not state and entry.get("constant") is True:
        return "view"
    return state or "nonpayable"


def encode_function_calldata(signature: str, input_types: Sequence[str], values: Sequence[Any]) -> str:
    return "0x" + (function_selector(signature) + abi_encode(list(input_types), list(values))).hex()


def unknown_fallback_calldata(seed: int, payload_bytes: int = 32) -> str:
    selector = deterministic_bytes(seed, 4)
    if selector in {b"\x00\x00\x00\x00", b"\xff\xff\xff\xff"}:
        selector = b"\xde\xad\xbe\xef"
    return "0x" + (selector + deterministic_bytes(seed + 1, max(0, payload_bytes))).hex()


def msg_value_for_case(state: str, run_index: int, seed: int, args: argparse.Namespace) -> int:
    if state != "payable":
        return 0
    if args.value_wei > 0:
        return args.value_wei
    max_value = max(1, int(args.max_msg_value_wei))
    if run_index == 0:
        return 0
    if run_index == 1:
        return 1
    return 1 + (stable_word(seed + run_index * 7919) % max_value)


def source_string_literals(source: str, limit: int = 64) -> list[str]:
    out: list[str] = []
    for match in re.finditer(r"""(?P<quote>["'])(?P<body>(?:\\.|(?!\1).){0,128})(?P=quote)""", source):
        body = match.group("body")
        if "\\" in body:
            continue
        if len(body) <= limit:
            out.append(body)
    return out


def dedupe_preserve_order(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        key = value if isinstance(value, (str, int, bool, bytes)) else json.dumps(jsonable_value(value), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def parse_assembly_ops(assembly: str) -> list[tuple[str, Optional[str]]]:
    ops: list[tuple[str, Optional[str]]] = []
    tokens = assembly.split()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.endswith(":"):
            index += 1
            continue
        mnemonic = token.upper()
        operand: Optional[str] = None
        if mnemonic.startswith("PUSH") and index + 1 < len(tokens) and re.fullmatch(r"0x[0-9a-fA-F]+", tokens[index + 1]):
            operand = tokens[index + 1]
            index += 2
        else:
            index += 1
        ops.append((mnemonic, operand))
    return ops


def bytecode_comparison_literals(assembly: str, max_abs: int) -> list[int]:
    values: list[int] = []
    ops = parse_assembly_ops(assembly)
    comparisons = {"EQ", "LT", "GT", "SLT", "SGT"}
    for index, (mnemonic, _) in enumerate(ops):
        if mnemonic not in comparisons:
            continue
        for prev_mnemonic, operand in ops[max(0, index - 8) : index]:
            if not prev_mnemonic.startswith("PUSH") or not operand:
                continue
            body = operand[2:]
            if len(body) > 64:
                continue
            value = int(body, 16)
            if abs(value) <= max_abs:
                values.append(value)
    return dedupe_preserve_order(values)


def extract_static_hints(row: Optional[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    hints: Dict[str, Any] = {
        "ints": [],
        "addresses": [],
        "bytes_by_size": {},
        "bytes_dynamic": [],
        "strings": [],
    }
    if row is None:
        return hints

    max_abs = max(0, int(getattr(args, "max_static_int_abs", 10**24)))
    int_values: list[int] = [0, 1, 2, 10, 16, 100, 1000, 10**18]
    comparison_int_values: list[int] = []
    addresses: list[str] = []
    byte_values: list[bytes] = []
    strings: list[str] = []

    source = str(row.get("source_code") or "")
    assembly = str(row.get("assembly") or "")
    comparison_int_values.extend(bytecode_comparison_literals(assembly, max_abs))

    for raw in re.findall(r"\b\d[\d_]*\b", source):
        try:
            value = int(raw.replace("_", ""), 10)
        except ValueError:
            continue
        if abs(value) <= max_abs:
            int_values.append(value)

    hex_literals = re.findall(r"0x[0-9a-fA-F]+", source)
    hex_literals += [f"0x{m}" for m in re.findall(r"\bPUSH\d+\s+0x([0-9a-fA-F]+)", assembly)]
    for raw in hex_literals:
        body = raw[2:]
        if not body:
            continue
        if len(body) == 40:
            addresses.append("0x" + body.lower())
        if len(body) % 2 == 0 and len(body) <= 64:
            try:
                byte_values.append(bytes.fromhex(body))
            except ValueError:
                pass
        if len(body) <= 64:
            try:
                value = int(body, 16)
            except ValueError:
                continue
            if abs(value) <= max_abs:
                int_values.append(value)

    functions = row.get("functions")
    if isinstance(functions, dict):
        for selector in functions:
            if isinstance(selector, str) and re.fullmatch(r"0x[0-9a-fA-F]{8}", selector):
                byte_values.append(bytes.fromhex(selector[2:]))
                int_values.append(int(selector, 16))
    events = row.get("events")
    if isinstance(events, dict):
        for topic in events:
            if isinstance(topic, str) and re.fullmatch(r"0x[0-9a-fA-F]{64}", topic):
                byte_values.append(bytes.fromhex(topic[2:]))

    strings.extend(source_string_literals(source))
    strings.extend(str(row.get("contract_name") or "").split())

    expanded_ints: list[int] = []
    for value in comparison_int_values + int_values:
        for candidate in (value - 1, value, value + 1):
            if abs(candidate) <= max_abs:
                expanded_ints.append(candidate)
    prioritized_ints = dedupe_preserve_order(expanded_ints)
    ints = prioritized_ints + [value for value in sorted(set(expanded_ints), key=lambda x: (abs(x), x)) if value not in set(prioritized_ints)]

    by_size: Dict[int, list[bytes]] = {}
    for raw in dedupe_preserve_order(byte_values):
        if not isinstance(raw, bytes):
            continue
        by_size.setdefault(len(raw), []).append(raw)

    hints["ints"] = ints[: max(0, int(getattr(args, "max_static_values_per_type", 12))) * 4]
    hints["addresses"] = dedupe_preserve_order(addresses)[: max(0, int(getattr(args, "max_static_values_per_type", 12)))]
    hints["bytes_by_size"] = {
        size: values[: max(0, int(getattr(args, "max_static_values_per_type", 12)))]
        for size, values in by_size.items()
    }
    hints["bytes_dynamic"] = dedupe_preserve_order(byte_values)[: max(0, int(getattr(args, "max_static_values_per_type", 12)))]
    hints["strings"] = [
        s
        for s in dedupe_preserve_order(strings)
        if isinstance(s, str) and len(s) <= 64 and not s.endswith(".sol")
    ][: max(0, int(getattr(args, "max_static_values_per_type", 12)))]
    return hints


def int_hint_values(typ: str, hints: Dict[str, Any], args: argparse.Namespace) -> list[int]:
    values: list[int] = []
    if typ.startswith("uint"):
        bits = int(typ[4:] or 256)
        hi = max_uint_for_bits(bits)
        for value in hints.get("ints") or []:
            if isinstance(value, int) and 0 <= value <= hi:
                values.append(value)
    elif typ.startswith("int"):
        bits = int(typ[3:] or 256)
        lo, hi = int_bounds(bits)
        for value in hints.get("ints") or []:
            if isinstance(value, int) and lo <= value <= hi:
                values.append(value)
    return dedupe_preserve_order(values)[: max(0, int(getattr(args, "max_static_values_per_type", 12)))]


def static_hint_values_for_entry(entry: Dict[str, Any], hints: Dict[str, Any], args: argparse.Namespace) -> list[Any]:
    typ = str(entry.get("type") or "")
    base, dims = split_array_type(typ)
    if dims:
        base_entry = dict(entry)
        base_entry["type"] = "address" if is_contract_abi_type(entry, base) else base
        base_values = static_hint_values_for_entry(base_entry, hints, args)
        if not base_values:
            return []
        outer_dim = dims[-1]
        if outer_dim is None:
            return [[value] for value in base_values[: max(1, int(getattr(args, "max_static_values_per_type", 12)) // 2)]]
        if outer_dim <= args.max_fixed_array_len:
            return [[value for _ in range(outer_dim)] for value in base_values[:2]]
        return []
    if is_contract_abi_type(entry, base):
        base = "address"
    if base.startswith(("uint", "int")):
        return int_hint_values(base, hints, args)
    if base == "address":
        return list(hints.get("addresses") or [])[: max(0, int(getattr(args, "max_static_values_per_type", 12)))]
    if base.startswith("bytes") and base != "bytes":
        size = int(base[5:])
        values = list((hints.get("bytes_by_size") or {}).get(size) or [])
        if size == 4:
            values.extend(value[:4] for value in (hints.get("bytes_by_size") or {}).get(20, [])[:1])
        return dedupe_preserve_order(values)[: max(0, int(getattr(args, "max_static_values_per_type", 12)))]
    if base == "bytes":
        return list(hints.get("bytes_dynamic") or [])[: max(0, int(getattr(args, "max_static_values_per_type", 12)))]
    if base == "string":
        return list(hints.get("strings") or [])[: max(0, int(getattr(args, "max_static_values_per_type", 12)))]
    return []


def build_case(
    *,
    signature: str,
    entry_type: str,
    state: str,
    calldata: str,
    values: Sequence[Any],
    value_wei: int,
    func_index: int,
    run_index: int,
    case_kind: str,
    setup: Optional[list[Dict[str, Any]]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    case: Dict[str, Any] = {
        "case_id": "",
        "signature": signature,
        "entry_type": entry_type,
        "state": state,
        "calldata": calldata,
        "args": jsonable_value(list(values)),
        "value": value_wei,
        "func_index": func_index,
        "run_index": run_index,
        "case_kind": case_kind,
    }
    if setup:
        case["setup"] = deepcopy_setup(setup)
    if extra:
        case.update(extra)
    return case


def deepcopy_setup(setup: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [dict(item) for item in setup]


def setup_score(entry: Dict[str, Any]) -> tuple[int, str]:
    name = str(entry.get("name") or "").lower()
    for index, hint in enumerate(SETUP_NAME_HINTS):
        if name.startswith(hint):
            return index, name
    return len(SETUP_NAME_HINTS), name


def setup_candidates(
    function_entries: Sequence[Dict[str, Any]],
    line_seed: int,
    hints: Dict[str, Any],
    args: argparse.Namespace,
) -> list[Dict[str, Any]]:
    candidates: list[Dict[str, Any]] = []
    if not getattr(args, "stateful_setup_cases", False):
        return candidates
    scored = []
    for func_index, item in enumerate(function_entries):
        state = abi_state_mutability(item)
        if state in {"view", "pure"}:
            continue
        score = setup_score(item)
        if score[0] >= len(SETUP_NAME_HINTS) and item.get("inputs"):
            continue
        scored.append((score, func_index, item))
    scored.sort(key=lambda x: (x[0], function_signature_from_abi(x[2])))

    for setup_index, (_, func_index, item) in enumerate(scored[: max(0, int(args.max_setup_functions))]):
        sig = function_signature_from_abi(item)
        state = abi_state_mutability(item)
        inputs = item.get("inputs") or []
        input_types = [canonical_abi_type(inp) for inp in inputs if isinstance(inp, dict)]
        try:
            values = []
            for arg_index, inp in enumerate(inputs):
                if not isinstance(inp, dict):
                    continue
                hint_values = static_hint_values_for_entry(inp, hints, args)
                if hint_values:
                    values.append(hint_values[min(setup_index, len(hint_values) - 1)])
                else:
                    values.append(
                        gen_branch_abi_value(
                            inp,
                            args.seed + line_seed * 10000 + func_index * 1000 + setup_index * 100 + arg_index,
                            (setup_index + arg_index + 1) % len(BRANCH_CASE_VARIANTS),
                            args,
                        )
                    )
            calldata = encode_function_calldata(sig, input_types, values)
        except Exception:
            continue
        candidates.append(
            {
                "signature": sig,
                "calldata": calldata,
                "args": jsonable_value(values),
                "value": msg_value_for_case(state, setup_index, args.seed + line_seed * 10000 + func_index * 1000, args),
                "case_kind": "stateful_setup",
            }
        )
    return candidates


def generate_calls(
    abi: Sequence[Dict[str, Any]],
    line_seed: int,
    args: argparse.Namespace,
    row: Optional[Dict[str, Any]] = None,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    calls: list[Dict[str, Any]] = []
    unsupported: list[Dict[str, Any]] = []
    hints = extract_static_hints(row, args) if getattr(args, "static_hint_cases", False) else extract_static_hints(None, args)
    function_entries = [
        item for item in abi if isinstance(item, dict) and item.get("type") == "function" and item.get("name")
    ]
    function_entries.sort(key=function_signature_from_abi)
    setup_pool = setup_candidates(function_entries, line_seed, hints, args)
    for func_index, item in enumerate(function_entries):
        sig = function_signature_from_abi(item)
        state = abi_state_mutability(item)
        inputs = item.get("inputs") or []
        input_types = [canonical_abi_type(inp) for inp in inputs if isinstance(inp, dict)]
        repeat = args.runs if inputs or state == "payable" else 1
        made_for_function = 0
        for run_index in range(repeat):
            try:
                values = [
                    gen_abi_value(inp, args.seed + line_seed * 10000 + func_index * 1000 + run_index * 10 + arg_index, run_index, args)
                    for arg_index, inp in enumerate(inputs)
                    if isinstance(inp, dict)
                ]
                calldata = encode_function_calldata(sig, input_types, values)
            except Exception as exc:
                if not any(u.get("signature") == sig for u in unsupported):
                    unsupported.append(
                        {
                            "type": "function",
                            "signature": sig,
                            "reason": str(exc),
                            "abi": item,
                        }
                    )
                continue
            value_wei = msg_value_for_case(
                state,
                run_index,
                args.seed + line_seed * 10000 + func_index * 1000,
                args,
            )
            calls.append(
                build_case(
                    signature=sig,
                    entry_type="function",
                    state=state,
                    calldata=calldata,
                    values=values,
                    value_wei=value_wei,
                    func_index=func_index,
                    run_index=run_index,
                    case_kind="deterministic_random",
                )
            )
            made_for_function += 1
        existing_payloads = {
            (str(call.get("calldata") or "").lower(), int(call.get("value") or 0))
            for call in calls
            if call.get("signature") == sig
        }
        if args.branch_cases and inputs:
            for variant_index, variant_name in enumerate(BRANCH_CASE_VARIANTS):
                try:
                    values = [
                        gen_branch_abi_value(
                            inp,
                            args.seed + line_seed * 10000 + func_index * 1000 + variant_index * 100 + arg_index,
                            (variant_index + arg_index) % len(BRANCH_CASE_VARIANTS),
                            args,
                        )
                        for arg_index, inp in enumerate(inputs)
                        if isinstance(inp, dict)
                    ]
                    calldata = encode_function_calldata(sig, input_types, values)
                except Exception as exc:
                    if not any(u.get("signature") == sig and u.get("case_kind") == "branch_boundary" for u in unsupported):
                        unsupported.append(
                            {
                                "type": "function",
                                "signature": sig,
                                "case_kind": "branch_boundary",
                                "reason": str(exc),
                                "abi": item,
                            }
                        )
                    continue
                value_wei = msg_value_for_case(
                    state,
                    variant_index,
                    args.seed + line_seed * 10000 + func_index * 1000 + 555,
                    args,
                )
                payload_key = (calldata.lower(), value_wei)
                if payload_key in existing_payloads:
                    continue
                existing_payloads.add(payload_key)
                calls.append(
                    build_case(
                        signature=sig,
                        entry_type="function",
                        state=state,
                        calldata=calldata,
                        values=values,
                        value_wei=value_wei,
                        func_index=func_index,
                        run_index=repeat + variant_index,
                        case_kind=f"branch_boundary:{variant_name}",
                    )
                )
                made_for_function += 1
        if getattr(args, "static_hint_cases", False) and inputs:
            added_static = 0
            for arg_index, inp in enumerate(inputs):
                if not isinstance(inp, dict):
                    continue
                for hint_index, hint_value in enumerate(static_hint_values_for_entry(inp, hints, args)):
                    if added_static >= max(0, int(args.max_static_hint_cases_per_function)):
                        break
                    try:
                        values = [
                            hint_value
                            if other_index == arg_index
                            else gen_branch_abi_value(
                                other,
                                args.seed + line_seed * 10000 + func_index * 1000 + hint_index * 100 + other_index,
                                (hint_index + other_index + 1) % len(BRANCH_CASE_VARIANTS),
                                args,
                            )
                            for other_index, other in enumerate(inputs)
                            if isinstance(other, dict)
                        ]
                        calldata = encode_function_calldata(sig, input_types, values)
                    except Exception as exc:
                        if not any(u.get("signature") == sig and u.get("case_kind") == "static_hint" for u in unsupported):
                            unsupported.append(
                                {
                                    "type": "function",
                                    "signature": sig,
                                    "case_kind": "static_hint",
                                    "reason": str(exc),
                                    "abi": item,
                                }
                            )
                        continue
                    value_wei = msg_value_for_case(
                        state,
                        repeat + len(BRANCH_CASE_VARIANTS) + added_static,
                        args.seed + line_seed * 10000 + func_index * 1000 + 777,
                        args,
                    )
                    payload_key = (calldata.lower(), value_wei)
                    if payload_key in existing_payloads:
                        continue
                    existing_payloads.add(payload_key)
                    calls.append(
                        build_case(
                            signature=sig,
                            entry_type="function",
                            state=state,
                            calldata=calldata,
                            values=values,
                            value_wei=value_wei,
                            func_index=func_index,
                            run_index=repeat + len(BRANCH_CASE_VARIANTS) + added_static,
                            case_kind="static_hint:literal",
                            extra={"hint_arg_index": arg_index, "hint_value": jsonable_value(hint_value)},
                        )
                    )
                    added_static += 1
                if added_static >= max(0, int(args.max_static_hint_cases_per_function)):
                    break
            made_for_function += added_static
        if setup_pool:
            added_stateful = 0
            base_targets = [
                call
                for call in calls
                if call.get("signature") == sig and call.get("entry_type") == "function" and not call.get("setup")
            ]
            for target_call in base_targets[:1]:
                for setup_call in setup_pool:
                    if added_stateful >= max(0, int(args.max_stateful_cases_per_function)):
                        break
                    if setup_call.get("signature") == sig:
                        continue
                    calls.append(
                        build_case(
                            signature=sig,
                            entry_type="function",
                            state=state,
                            calldata=str(target_call.get("calldata") or "0x"),
                            values=target_call.get("args") or [],
                            value_wei=int(target_call.get("value") or 0),
                            func_index=func_index,
                            run_index=100000 + added_stateful,
                            case_kind="stateful_setup",
                            setup=[setup_call],
                            extra={"setup_signature": setup_call.get("signature")},
                        )
                    )
                    added_stateful += 1
                if added_stateful >= max(0, int(args.max_stateful_cases_per_function)):
                    break
            made_for_function += added_stateful
        if made_for_function == 0 and not any(item is u.get("abi") for u in unsupported):
            unsupported.append({"type": "function", "signature": sig, "reason": "no_testcase_generated", "abi": item})

    for item in abi:
        if not isinstance(item, dict) or item.get("type") not in {"fallback", "receive"}:
            continue
        entry_type = str(item.get("type"))
        state = abi_state_mutability(item)
        repeat = args.runs if entry_type == "fallback" or state == "payable" else 1
        for run_index in range(repeat):
            if entry_type == "receive":
                calldata = "0x"
                signature = "<receive>"
            else:
                payload_size = 32
                if getattr(args, "large_fallback_calldata", False):
                    fallback_sizes = (32, 128, 512, 2048, 8192, 15520)
                    payload_size = fallback_sizes[run_index % len(fallback_sizes)]
                calldata = unknown_fallback_calldata(args.seed + line_seed * 10000 + 777 + run_index, payload_size)
                signature = "<fallback>"
            value_wei = msg_value_for_case(
                state,
                run_index,
                args.seed + line_seed * 10000 + len(function_entries) * 1000,
                args,
            )
            calls.append(
                {
                    "case_id": "",
                    "signature": signature,
                    "entry_type": entry_type,
                    "state": state,
                    "calldata": calldata,
                    "args": [],
                    "value": value_wei,
                    "func_index": len(function_entries),
                    "run_index": run_index,
                    "case_kind": "deterministic_random",
                }
            )
    return calls, unsupported


def group_test_cases(calls: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    groups: list[Dict[str, Any]] = []
    index: dict[tuple[str, str], Dict[str, Any]] = {}
    for call in calls:
        entry_type = str(call.get("entry_type") or "function")
        signature = str(call.get("signature") or "")
        key = (entry_type, signature)
        group = index.get(key)
        if group is None:
            group = {
                "entry_type": entry_type,
                "signature": signature,
                "state": call.get("state"),
                "func_index": call.get("func_index"),
                "cases": [],
            }
            index[key] = group
            groups.append(group)
        case = {
            k: v
            for k, v in call.items()
            if k not in {"entry_type", "signature", "state", "func_index"}
        }
        group["cases"].append(case)
    for group in groups:
        group["case_count"] = len(group["cases"])
    return groups


def flatten_test_case_groups(raw_test_cases: Any) -> list[Dict[str, Any]]:
    if isinstance(raw_test_cases, dict):
        grouped_items: list[Dict[str, Any]] = []
        for signature, item in raw_test_cases.items():
            if not isinstance(item, dict):
                continue
            group = dict(item)
            group.setdefault("signature", signature)
            grouped_items.append(group)
    elif isinstance(raw_test_cases, list):
        grouped_items = raw_test_cases
    else:
        return []
    calls: list[Dict[str, Any]] = []
    for item in grouped_items:
        if not isinstance(item, dict):
            continue
        nested = item.get("cases")
        if isinstance(nested, list):
            for case in nested:
                if not isinstance(case, dict):
                    continue
                call = dict(case)
                for key in ("entry_type", "signature", "state", "func_index"):
                    call.setdefault(key, item.get(key))
                calls.append(call)
        else:
            calls.append(dict(item))
    return calls


def solidity_version_tuple(version: str) -> tuple[int, int, int]:
    parts = [int(p) for p in str(version or "0.0.0").split(".")[:3]]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def compiler_version_from_row(row: Dict[str, Any]) -> str:
    compiler = row.get("compiler")
    if isinstance(compiler, dict):
        return str(compiler.get("compiler_version") or "")
    return str(compiler or "")


def append_coverage_low_level_call(
    lines: list[str],
    target_expr: str,
    calldata: str,
    value: int,
    version: tuple[int, int, int],
    var_name: str,
    gas_limit: int,
    sender_expr: str,
) -> None:
    body = ensure0x(calldata).removeprefix("0x")
    lines.append(f'        _scdbenchCall({sender_expr}, address({target_expr}), hex"{body}", {value});')


def is_library_source(source_code: str, contract_name: str) -> bool:
    if not contract_name:
        return False
    return re.search(r"\blibrary\s+" + re.escape(contract_name) + r"\b", source_code) is not None


def abi_entries_by_signature(abi: Sequence[Dict[str, Any]] | None) -> dict[str, Dict[str, Any]]:
    out: dict[str, Dict[str, Any]] = {}
    for item in abi or []:
        if not isinstance(item, dict) or item.get("type") != "function" or not item.get("name"):
            continue
        out[function_signature_from_abi(item)] = item
    return out


def signature_input_entries(signature: str, abi_by_signature: dict[str, Dict[str, Any]]) -> list[Dict[str, Any]]:
    item = abi_by_signature.get(signature)
    if isinstance(item, dict) and isinstance(item.get("inputs"), list):
        return [inp for inp in item["inputs"] if isinstance(inp, dict)]
    match = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\((.*)\)", signature)
    if match is None:
        return []
    raw_types = split_top_level_commas(match.group(1))
    return [{"type": typ} for typ in raw_types]


def append_coverage_library_call(
    lines: list[str],
    library_name: str,
    call: Dict[str, Any],
    solc_version: str,
    abi_by_signature: dict[str, Dict[str, Any]],
) -> bool:
    signature = str(call.get("signature") or "")
    if "(" not in signature:
        return False
    function_name = signature.split("(", 1)[0]
    if not IDENTIFIER_RE.fullmatch(function_name):
        return False
    values = call.get("args")
    if not isinstance(values, list):
        values = []
    entries = signature_input_entries(signature, abi_by_signature)
    if len(entries) != len(values):
        return False
    arg_exprs: list[str] = []
    for value, entry in zip(values, entries):
        expr = solidity_literal_for_abi_value(value, entry, solc_version)
        if expr is None:
            return False
        arg_exprs.append(expr)
    lines.append(f"        {library_name}.{function_name}({', '.join(arg_exprs)});")
    return True


def coverage_call_helper_source(version: tuple[int, int, int], gas_limit: int) -> list[str]:
    if version < (0, 5, 0):
        return [
            "    function _scdbenchCall(address sender, address target, bytes data, uint256 value) internal {",
            "        if (sender == address(0)) { sender = address(this); }",
            "        SCD_VM.deal(sender, SCD_BALANCE);",
            "        if (sender != address(this)) { SCD_VM.prank(sender); }",
            f"        if (value != 0) {{ bool okValue = target.call.gas({gas_limit}).value(value)(data); okValue; }}",
            f"        else {{ bool okNoValue = target.call.gas({gas_limit})(data); okNoValue; }}",
            "    }",
        ]
    if version < (0, 6, 0):
        return [
            "    function _scdbenchCall(address sender, address target, bytes memory data, uint256 value) internal {",
            "        if (sender == address(0)) { sender = address(this); }",
            "        SCD_VM.deal(sender, SCD_BALANCE);",
            "        if (sender != address(this)) { SCD_VM.prank(sender); }",
            f"        if (value != 0) {{ (bool ok, bytes memory out) = target.call.gas({gas_limit}).value(value)(data); ok; out; }}",
            f"        else {{ (bool ok, bytes memory out) = target.call.gas({gas_limit})(data); ok; out; }}",
            "    }",
        ]
    return [
        "    function _scdbenchCall(address sender, address target, bytes memory data, uint256 value) internal {",
        "        if (sender == address(0)) { sender = address(this); }",
        "        SCD_VM.deal(sender, SCD_BALANCE);",
        "        if (sender != address(this)) { SCD_VM.prank(sender); }",
        f"        (bool ok, bytes memory out) = target.call{{gas: {gas_limit}, value: value}}(data);",
        "        ok; out;",
        "    }",
    ]


def coverage_vm_interface_source(version: tuple[int, int, int]) -> list[str]:
    etch_arg = "bytes calldata newRuntimeBytecode" if version >= (0, 5, 0) else "bytes newRuntimeBytecode"
    return [
        "interface SCDbenchVm {",
        "    function prank(address sender) external;",
        "    function deal(address who, uint256 newBalance) external;",
        "    function store(address target, bytes32 slot, bytes32 value) external;",
        f"    function etch(address target, {etch_arg}) external;",
        "    function warp(uint256 newTimestamp) external;",
        "    function roll(uint256 newHeight) external;",
        "}",
        "",
    ]


def coverage_mock_contract_source(version: tuple[int, int, int]) -> list[str]:
    if version < (0, 5, 0):
        return []
    max_uint = "115792089237316195423570985008687907853269984665640564039457584007913129639935"
    lines = [
        "contract CoverageMock {",
    ]
    if version >= (0, 6, 0):
        lines.extend(
            [
            "    receive() external payable {}",
            "    fallback() external payable {",
            "        bytes memory out = new bytes(4096);",
            "        assembly {",
            "            mstore(add(out, 32), 32)",
            "            return(add(out, 32), 4096)",
            "        }",
            "    }",
            ]
        )
    lines.extend(
        [
        f"    function balanceOf(address) external pure returns (uint256) {{ return {max_uint}; }}",
        f"    function allowance(address, address) external pure returns (uint256) {{ return {max_uint}; }}",
        "    function approve(address, uint256) external pure returns (bool) { return true; }",
        "    function transfer(address, uint256) external pure returns (bool) { return true; }",
        "    function transferFrom(address, address, uint256) external pure returns (bool) { return true; }",
        "    function ownerOf(uint256) external view returns (address) { return address(this); }",
        "    function getApproved(uint256) external view returns (address) { return address(this); }",
        "    function isApprovedForAll(address, address) external pure returns (bool) { return true; }",
        "    function decimals() external pure returns (uint8) { return 18; }",
        "    function stEthPerToken() external pure returns (uint256) { return 1000000000000000000; }",
        "    function totalSupply() external pure returns (uint256) { return 1000000000000000000000000000; }",
        "    function WETH() external view returns (address) { return address(this); }",
        "    function token0() external view returns (address) { return address(this); }",
        "    function token1() external view returns (address) { return address(this); }",
        "    function factory() external view returns (address) { return address(this); }",
        "    function getPair(address, address) external view returns (address) { return address(this); }",
        "    function allPairs(uint256) external view returns (address) { return address(this); }",
        "    function allPairsLength() external pure returns (uint256) { return 1; }",
        "    function allowedPairs() external view returns (address[] memory out) { out = new address[](1); out[0] = address(this); }",
        "    function getReserves() external view returns (uint112, uint112, uint32) { return (1000000, 1000000, uint32(block.timestamp)); }",
        "    function latestAnswer() external pure returns (int256) { return 100000000; }",
        "    function latestRoundData() external view returns (uint80, int256, uint256, uint256, uint80) { return (1, 100000000, block.timestamp, block.timestamp, 1); }",
        "    function policyGroup() external view returns (address) { return address(this); }",
        "    function policy() external pure returns (address) { return address(0); }",
        "    function setPolicy(address) external { }",
        "    function addGroup(address) external { }",
        "    function isGroup(address) external pure returns (bool) { return true; }",
        "    function isDuringVotingPeriod(address) external pure returns (bool) { return true; }",
        "    function isKeeper(address) external pure returns (bool) { return true; }",
        "    function worked(address) external { }",
        "    function getUserReserveData(address, address) external pure returns (uint256, uint256, uint256, uint256, uint256, uint256, uint256, uint256, uint256, bool) {",
        "        return (0, 1000000000000000000, 0, 0, 0, 0, 0, 0, 0, true);",
        "    }",
        "    function liquidationCall(address, address, address, uint256, bool) external { }",
        "    function getAmountsOut(uint256 amountIn, address[] calldata path) external pure returns (uint256[] memory amounts) {",
        "        amounts = new uint256[](path.length == 0 ? 1 : path.length);",
        "        for (uint256 i = 0; i < amounts.length; i++) { amounts[i] = amountIn == 0 ? 1 : amountIn; }",
        "    }",
        "    function swapExactTokensForTokens(uint256 amountIn, uint256, address[] calldata path, address, uint256) external pure returns (uint256[] memory amounts) {",
        "        amounts = new uint256[](path.length == 0 ? 1 : path.length);",
        "        for (uint256 i = 0; i < amounts.length; i++) { amounts[i] = amountIn == 0 ? 1 : amountIn; }",
        "    }",
        "    function swapExactETHForTokens(uint256, address[] calldata path, address, uint256) external payable returns (uint256[] memory amounts) {",
        "        amounts = new uint256[](path.length == 0 ? 1 : path.length);",
        "        for (uint256 i = 0; i < amounts.length; i++) { amounts[i] = msg.value == 0 ? 1 : msg.value; }",
        "    }",
        "    function mint(uint256, uint256, uint256, uint256, uint256) external pure returns (uint256) { return 1; }",
        "    function burn(uint256, uint256, uint256 amount0, uint256 amount1) external pure returns (uint256, uint256) { return (amount0, amount1); }",
        "}",
        "",
        ]
    )
    return lines


def coverage_authority_helper_source(version: tuple[int, int, int], slots: int) -> list[str]:
    if version < (0, 5, 0) or slots <= 0:
        return [
            "    function _scdbenchPrepareAuthority(address target, address actor) internal {",
            "        target;",
            "        actor;",
            "    }",
        ]
    role_literals = ["0x" + ("00" * 32)] + ["0x" + keccak(text=name).hex() for name in COVERAGE_ROLE_NAMES]
    roles_type = f"bytes32[{len(role_literals)}] memory"
    roles_value = ", ".join(f"bytes32({literal})" for literal in role_literals)
    lines = [
        "    function _scdbenchPrepareAuthority(address target, address actor) internal {",
        "        if (actor == address(0)) { actor = address(this); }",
        "        bytes32 actorWord = bytes32(uint256(uint160(actor)));",
        "        bytes32 trueWord = bytes32(uint256(1));",
        "        bytes32 maxWord = bytes32(~uint256(0));",
        f"        {roles_type} roles = [{roles_value}];",
        f"        for (uint256 slot = 0; slot < {slots}; slot++) {{",
        "            SCD_VM.store(target, bytes32(slot), actorWord);",
        "            SCD_VM.store(target, keccak256(abi.encode(actor, slot)), maxWord);",
        "            bytes32 nestedOwnerSlot = keccak256(abi.encode(actor, slot));",
        "            SCD_VM.store(target, keccak256(abi.encode(address(this), nestedOwnerSlot)), maxWord);",
        f"            for (uint256 roleIndex = 0; roleIndex < {len(role_literals)}; roleIndex++) {{",
        "                bytes32 roleSlot = keccak256(abi.encode(roles[roleIndex], slot));",
        "                SCD_VM.store(target, keccak256(abi.encode(actor, roleSlot)), trueWord);",
        "            }",
        "        }",
        "    }",
    ]
    return lines


def coverage_environment_helper_source(version: tuple[int, int, int]) -> list[str]:
    return [
        "    function _scdbenchPrepareEnvironment() internal {",
        "        SCD_VM.deal(address(this), SCD_BALANCE);",
        f"        SCD_VM.warp({COVERAGE_TIMESTAMP});",
        f"        SCD_VM.roll({COVERAGE_BLOCK_NUMBER});",
        "    }",
    ]


def coverage_mock_helper_source(version: tuple[int, int, int]) -> list[str]:
    if version < (0, 5, 0):
        return [
            "    function _scdbenchMockAddress(address target) internal {",
            "        target;",
            "    }",
        ]
    if version < (0, 6, 0):
        return [
            "    function _scdbenchRuntimeCode(address source) internal view returns (bytes memory code) {",
            "        assembly {",
            "            let size := extcodesize(source)",
            "            code := mload(0x40)",
            "            let rounded := and(add(add(size, 0x20), 0x1f), not(0x1f))",
            "            mstore(0x40, add(code, rounded))",
            "            mstore(code, size)",
            "            extcodecopy(source, add(code, 0x20), 0, size)",
            "        }",
            "    }",
            "    function _scdbenchMockAddress(address target) internal {",
            "        if (target == address(0) || target == address(this)) { return; }",
            "        CoverageMock mock = new CoverageMock();",
            "        SCD_VM.etch(target, _scdbenchRuntimeCode(address(mock)));",
            "        SCD_VM.deal(target, SCD_BALANCE);",
            "    }",
        ]
    return [
        "    function _scdbenchMockAddress(address target) internal {",
        "        if (target == address(0) || target == address(this)) { return; }",
        "        SCD_VM.etch(target, type(CoverageMock).runtimeCode);",
        "        SCD_VM.deal(target, SCD_BALANCE);",
        "    }",
    ]


def address_literals_from_value(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str) and re.fullmatch(r"0x[0-9a-fA-F]{40}", value):
        if value.lower() != "0x" + "00" * 20:
            found.append(value)
    elif isinstance(value, list):
        for item in value:
            found.extend(address_literals_from_value(item))
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(address_literals_from_value(item))
    return dedupe_preserve_order(found)


def source_address_literals(source_code: str, limit: int = 64) -> list[str]:
    found: list[str] = []
    for raw in re.findall(r"0x[0-9a-fA-F]{40}", source_code):
        if raw.lower() == "0x" + "00" * 20:
            continue
        if raw not in found:
            found.append(raw)
        if len(found) >= limit:
            break
    return found


def coverage_storage_slot_literals(source_code: str, limit: int = 24) -> list[str]:
    slots: list[str] = []
    pattern = re.compile(
        r"\b(?P<decl>(?:bytes32|uint256)\s+(?:private\s+|internal\s+|public\s+)?(?:constant|immutable)?\s*"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)[^=;]*=\s*(?P<value>0x[0-9a-fA-F]{64}))"
    )
    for match in pattern.finditer(source_code):
        name = match.group("name").lower()
        if not any(hint in name for hint in ("slot", "storage", "lock", "status", "flag", "role")):
            continue
        value = match.group("value")
        if value not in slots:
            slots.append(value)
        if len(slots) >= limit:
            break
    return slots


def append_coverage_storage_slot_preparation(lines: list[str], target_expr: str, slots: Sequence[str]) -> None:
    for slot in slots:
        if re.fullmatch(r"0x[0-9a-fA-F]{64}", slot):
            lines.append(f"        SCD_VM.store(address({target_expr}), bytes32({slot}), bytes32(uint256(1)));")


def coverage_sender_expressions(call: Dict[str, Any], args: argparse.Namespace) -> list[str]:
    limit = max(1, int(getattr(args, "coverage_sender_variants", 2)))
    exprs = ["address(this)"]
    for addr in address_literals_from_value(call.get("args") or []):
        expr = address_uint160_expr(addr)
        if expr not in exprs:
            exprs.append(expr)
        if len(exprs) >= limit:
            break
    for addr in COVERAGE_EOA_SENDERS:
        if len(exprs) >= limit:
            break
        expr = address_uint160_expr(addr)
        if expr not in exprs:
            exprs.append(expr)
    return exprs[:limit]


def coverage_setup_score(call: Dict[str, Any]) -> tuple[int, str]:
    signature = str(call.get("signature") or "")
    name = signature.split("(", 1)[0].lower()
    for index, hint in enumerate(SETUP_NAME_HINTS):
        if name.startswith(hint):
            return index, signature
    return len(SETUP_NAME_HINTS), signature


def coverage_global_setup_calls(calls: Sequence[Dict[str, Any]], args: argparse.Namespace) -> list[Dict[str, Any]]:
    limit = max(0, int(getattr(args, "coverage_augment_setup_functions", 6)))
    if limit <= 0:
        return []
    candidates: list[tuple[tuple[int, str], int, Dict[str, Any]]] = []
    seen: set[str] = set()
    for index, call in enumerate(calls):
        if not isinstance(call, dict) or call.get("entry_type") != "function":
            continue
        state = str(call.get("state") or "").lower()
        if state in {"view", "pure"}:
            continue
        signature = str(call.get("signature") or "")
        if signature in seen:
            continue
        score = coverage_setup_score(call)
        if score[0] >= len(SETUP_NAME_HINTS):
            continue
        seen.add(signature)
        candidates.append((score, index, call))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [call for _, _, call in candidates[:limit]]


def coverage_synthetic_setup_entries(
    abi: Sequence[Dict[str, Any]] | None,
    args: argparse.Namespace,
) -> list[Dict[str, Any]]:
    limit = max(0, int(getattr(args, "coverage_synthetic_setup_functions", 8)))
    if limit <= 0:
        return []
    candidates: list[tuple[tuple[int, str], Dict[str, Any]]] = []
    seen: set[str] = set()
    for item in abi or []:
        if not isinstance(item, dict) or item.get("type") != "function" or not item.get("name"):
            continue
        state = abi_state_mutability(item)
        if state in {"view", "pure"}:
            continue
        signature = function_signature_from_abi(item)
        if signature in seen:
            continue
        score = coverage_setup_score({"signature": signature})
        if score[0] >= len(SETUP_NAME_HINTS):
            continue
        seen.add(signature)
        candidates.append((score, item))
    candidates.sort(key=lambda item: item[0])
    return [item for _, item in candidates[:limit]]


def solidity_synthetic_setup_expr(entry: Dict[str, Any], solc_version: str) -> str | None:
    version = solidity_version_tuple(solc_version)
    raw_type = str(entry.get("type") or "")
    base, dims = split_array_type(raw_type)
    text = f"{entry.get('name') or ''} {entry.get('internalType') or ''}".lower()
    if dims:
        if dims[-1] is None:
            new_type = solidity_array_new_type(entry)
            if new_type is None:
                return None
            return f"new {new_type}(0)"
        return None
    if is_contract_abi_type(entry, base) or base == "address":
        if is_authority_address_entry(entry):
            return solidity_test_contract_address_expr(entry, solc_version)
        if any(hint in text for hint in CONSTRUCTOR_CODE_ADDRESS_HINTS):
            mock_expr = solidity_mock_contract_address_expr(entry, solc_version)
            if mock_expr is not None:
                return mock_expr
        return solidity_test_contract_address_expr(entry, solc_version)
    if base == "bool":
        return "true" if any(hint in text for hint in ("enable", "enabled", "active")) else "false"
    if base.startswith("uint"):
        bits = int(base[4:] or 256)
        hi = max_uint_for_bits(bits)
        value = 1
        if any(hint in text for hint in CONSTRUCTOR_DURATION_HINTS):
            value = 3600
        elif any(hint in text for hint in CONSTRUCTOR_TIMESTAMP_START_HINTS):
            value = COVERAGE_TIMESTAMP - 3600
        elif any(hint in text for hint in CONSTRUCTOR_TIMESTAMP_END_HINTS):
            value = COVERAGE_TIMESTAMP + 31536000
        elif any(hint in text for hint in CONSTRUCTOR_AMOUNT_HINTS):
            value = 10**18
        return f"{base}({min(value, hi)})"
    if base.startswith("int"):
        return f"{base}(1)"
    if base.startswith("bytes") and base != "bytes":
        return f"{base}(bytes32(uint256(1)))"
    if base == "bytes":
        return 'hex"01"'
    if base == "string":
        return '"scdbench"'
    return None


def append_coverage_synthetic_setup_call(
    lines: list[str],
    target_expr: str,
    item: Dict[str, Any],
    solc_version: str,
    call_id: str,
    gas_limit: int,
) -> bool:
    if solidity_version_tuple(solc_version) < (0, 5, 0):
        return False
    inputs = item.get("inputs") or []
    if not isinstance(inputs, list):
        return False
    arg_exprs: list[str] = []
    for inp in inputs:
        if not isinstance(inp, dict):
            return False
        expr = solidity_synthetic_setup_expr(inp, solc_version)
        if expr is None:
            return False
        arg_exprs.append(expr)
    signature = function_signature_from_abi(item)
    selector = "0x" + function_selector(signature).hex()
    value = 1 if abi_state_mutability(item) == "payable" else 0
    selector_args = f"bytes4({selector})" + (", " + ", ".join(arg_exprs) if arg_exprs else "")
    lines.append(f"        bytes memory scdbenchSetup_{call_id} = abi.encodeWithSelector({selector_args});")
    lines.append(
        f"        _scdbenchCall(address(this), address({target_expr}), scdbenchSetup_{call_id}, {value});"
    )
    return True


def append_coverage_authority_preparation(
    lines: list[str],
    target_expr: str,
    calls: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines.append(f"        _scdbenchPrepareAuthority(address({target_expr}), address(this));")
    sender_exprs: list[str] = []
    for call in calls:
        for sender_expr in coverage_sender_expressions(call, args):
            if sender_expr not in sender_exprs:
                sender_exprs.append(sender_expr)
    for sender_expr in sender_exprs:
        if sender_expr != "address(this)":
            lines.append(f"        _scdbenchPrepareAuthority(address({target_expr}), {sender_expr});")


def generate_coverage_test_source(
    contract_name: str,
    solc_version: str,
    calls: Sequence[Dict[str, Any]],
    constructor_exprs: Sequence[str],
    args: argparse.Namespace,
    source_code: str = "",
    abi: Sequence[Dict[str, Any]] | None = None,
) -> str:
    version = solidity_version_tuple(solc_version)
    has_vm = True
    batch_size = max(1, int(getattr(args, "coverage_batch_size", 25)))
    gas_limit = max(21_000, int(getattr(args, "coverage_call_gas_limit", 5_000_000)))
    auth_slots = max(0, int(getattr(args, "coverage_auth_storage_slots", 4)))
    ctor = ", ".join(constructor_exprs)
    library_mode = is_library_source(source_code, contract_name)
    abi_by_sig = abi_entries_by_signature(abi)
    global_setup_calls = coverage_global_setup_calls(calls, args)
    synthetic_setup_entries = coverage_synthetic_setup_entries(abi, args)
    storage_slot_literals = coverage_storage_slot_literals(source_code)
    lines = [
        f"pragma solidity {solc_version};",
    ]
    lines.extend(coverage_vm_interface_source(version))
    lines += [
        'import "../src/Target.sol";',
        "",
    ]
    lines.extend(coverage_mock_contract_source(version))
    lines += [
        "contract CoverageReplayTest {",
        f"    uint256 constant SCD_BALANCE = {SENDER_BALANCE_WEI};",
    ]
    if has_vm:
        lines.append(
            '    SCDbenchVm constant SCD_VM = SCDbenchVm(address(uint160(uint256(keccak256("hevm cheat code")))));'
        )
    lines.extend(coverage_call_helper_source(version, gas_limit))
    lines.extend(coverage_authority_helper_source(version, auth_slots))
    lines.extend(coverage_environment_helper_source(version))
    lines.extend(coverage_mock_helper_source(version))
    lines.append("")

    for batch_index, start in enumerate(range(0, len(calls), batch_size)):
        batch = calls[start : start + batch_size]
        lines.extend(
            [
                f"    function testCoverageReplay_{batch_index:04d}() public payable {{",
                "        _scdbenchPrepareEnvironment();",
            ]
        )
        if library_mode:
            for call in batch:
                for setup_call in call.get("setup") or []:
                    if isinstance(setup_call, dict):
                        append_coverage_library_call(lines, contract_name, setup_call, solc_version, abi_by_sig)
                append_coverage_library_call(lines, contract_name, call, solc_version, abi_by_sig)
            lines.append("    }")
            continue
        lines.append(f"        {contract_name} target = new {contract_name}({ctor});")
        environment_calls = list(batch) + list(global_setup_calls)
        if version >= (0, 5, 0):
            mock_addresses: list[str] = source_address_literals(source_code)
            for call in environment_calls:
                for addr in address_literals_from_value(call.get("args") or []):
                    if addr not in mock_addresses:
                        mock_addresses.append(addr)
                for setup_call in call.get("setup") or []:
                    if not isinstance(setup_call, dict):
                        continue
                    for addr in address_literals_from_value(setup_call.get("args") or []):
                        if addr not in mock_addresses:
                            mock_addresses.append(addr)
            for addr in mock_addresses[:64]:
                lines.append(f"        _scdbenchMockAddress({address_uint160_expr(addr)});")
        if has_vm:
            lines.append("        SCD_VM.deal(address(target), SCD_BALANCE);")
        if auth_slots > 0 and has_vm:
            lines.extend(
                [
                    f"        {contract_name} authTarget = new {contract_name}({ctor});",
                    "        SCD_VM.deal(address(authTarget), SCD_BALANCE);",
                ]
            )
            append_coverage_authority_preparation(lines, "authTarget", environment_calls, args)
            append_coverage_storage_slot_preparation(lines, "authTarget", storage_slot_literals)
        for setup_index, setup_item in enumerate(synthetic_setup_entries):
            append_coverage_synthetic_setup_call(
                lines,
                "target",
                setup_item,
                solc_version,
                f"{batch_index}_{setup_index}",
                gas_limit,
            )
            if auth_slots > 0 and has_vm:
                append_coverage_synthetic_setup_call(
                    lines,
                    "authTarget",
                    setup_item,
                    solc_version,
                    f"auth_{batch_index}_{setup_index}",
                    gas_limit,
                )
        if auth_slots > 0 and has_vm and synthetic_setup_entries:
            lines.append("        SCD_VM.deal(address(authTarget), SCD_BALANCE);")
            append_coverage_authority_preparation(lines, "authTarget", environment_calls, args)
            append_coverage_storage_slot_preparation(lines, "authTarget", storage_slot_literals)
        for setup_index, setup_call in enumerate(global_setup_calls):
            append_coverage_low_level_call(
                lines,
                "target",
                str(setup_call.get("calldata") or "0x"),
                int(setup_call.get("value") or 0),
                version,
                f"global_setup_ok{batch_index}_{setup_index}",
                gas_limit,
                "address(this)",
            )
            if auth_slots > 0 and has_vm:
                append_coverage_low_level_call(
                    lines,
                    "authTarget",
                    str(setup_call.get("calldata") or "0x"),
                    int(setup_call.get("value") or 0),
                    version,
                    f"auth_global_setup_ok{batch_index}_{setup_index}",
                    gas_limit,
                    "address(this)",
                )
        if auth_slots > 0 and has_vm and global_setup_calls:
            lines.append("        SCD_VM.deal(address(authTarget), SCD_BALANCE);")
            append_coverage_authority_preparation(lines, "authTarget", environment_calls, args)
            append_coverage_storage_slot_preparation(lines, "authTarget", storage_slot_literals)
        for local_index, call in enumerate(batch):
            call_index = start + local_index
            for setup_index, setup_call in enumerate(call.get("setup") or []):
                if not isinstance(setup_call, dict):
                    continue
                append_coverage_low_level_call(
                    lines,
                    "target",
                    str(setup_call.get("calldata") or "0x"),
                    int(setup_call.get("value") or 0),
                    version,
                    f"setup_ok{call_index}_{setup_index}",
                    gas_limit,
                    "address(this)",
                )
                if auth_slots > 0 and has_vm:
                    append_coverage_low_level_call(
                        lines,
                        "authTarget",
                        str(setup_call.get("calldata") or "0x"),
                        int(setup_call.get("value") or 0),
                        version,
                        f"auth_setup_ok{call_index}_{setup_index}",
                        gas_limit,
                        "address(this)",
                    )
            for sender_index, sender_expr in enumerate(coverage_sender_expressions(call, args)):
                append_coverage_low_level_call(
                    lines,
                    "target",
                    str(call.get("calldata") or "0x"),
                    int(call.get("value") or 0),
                    version,
                    f"ok{call_index}_{sender_index}",
                    gas_limit,
                    sender_expr,
                )
                if auth_slots > 0 and has_vm:
                    append_coverage_low_level_call(
                        lines,
                        "authTarget",
                        str(call.get("calldata") or "0x"),
                        int(call.get("value") or 0),
                        version,
                        f"auth_ok{call_index}_{sender_index}",
                        gas_limit,
                        sender_expr,
                    )
        lines.append("    }")
    lines += [
        "}",
        "",
    ]
    return "\n".join(lines)


def find_contract_line_range(source_code: str, contract_name: str) -> tuple[int, int] | None:
    if not contract_name:
        return None
    match = re.search(r"\b(?:contract|library|interface)\s+" + re.escape(contract_name) + r"\b", source_code)
    if match is None:
        return None
    open_brace = source_code.find("{", match.end())
    if open_brace < 0:
        return None

    depth = 0
    in_string: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    index = open_brace
    while index < len(source_code):
        ch = source_code[index]
        pair = source_code[index : index + 2]
        if line_comment:
            if ch == "\n":
                line_comment = False
        elif block_comment:
            if pair == "*/":
                block_comment = False
                index += 1
        elif in_string is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_string:
                in_string = None
        else:
            if pair == "//":
                line_comment = True
                index += 1
            elif pair == "/*":
                block_comment = True
                index += 1
            elif ch in {"'", '"'}:
                in_string = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    start_line = source_code.count("\n", 0, match.start()) + 1
                    end_line = source_code.count("\n", 0, index) + 1
                    return start_line, end_line
        index += 1
    return None


def coverage_metrics_from_hits(
    line_hits: Dict[int, int],
    function_lines: Dict[str, Any],
    function_hits: Dict[str, int],
    branches: Sequence[Dict[str, int]],
) -> Dict[str, Any]:
    covered_lines = sorted(line_no for line_no, hits in line_hits.items() if hits > 0)
    uncovered_lines = sorted(line_no for line_no, hits in line_hits.items() if hits <= 0)
    function_names = sorted(function_lines)
    covered_functions = sorted(name for name in function_names if function_hits.get(name, 0) > 0)
    uncovered_functions = sorted(name for name in function_names if function_hits.get(name, 0) <= 0)
    covered_branches = sum(1 for branch in branches if int(branch.get("hits") or 0) > 0)

    def ratio(covered: int, total: int) -> float | None:
        return (covered / total) if total else None

    return {
        "lines": {
            "covered": len(covered_lines),
            "total": len(line_hits),
            "ratio": ratio(len(covered_lines), len(line_hits)),
            "covered_line_numbers": covered_lines,
            "uncovered_line_numbers": uncovered_lines,
        },
        "functions": {
            "covered": len(covered_functions),
            "total": len(function_names),
            "ratio": ratio(len(covered_functions), len(function_names)),
            "covered_names": covered_functions,
            "uncovered_names": uncovered_functions,
        },
        "branches": {
            "covered": covered_branches,
            "total": len(branches),
            "ratio": ratio(covered_branches, len(branches)),
        },
    }


def byte_line_starts(source: str) -> list[int]:
    raw = source.encode("utf-8", errors="replace")
    starts = [0]
    for index, byte in enumerate(raw):
        if byte == 10:
            starts.append(index + 1)
    return starts


def line_for_byte_offset(starts: Sequence[int], offset: int) -> int:
    lo = 0
    hi = len(starts)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if starts[mid] <= offset:
            lo = mid
        else:
            hi = mid
    return lo + 1


def lines_for_source_span(starts: Sequence[int], offset: int, length: int) -> set[int]:
    if offset < 0:
        return set()
    start_line = line_for_byte_offset(starts, offset)
    end_offset = offset + max(0, length - 1)
    end_line = line_for_byte_offset(starts, end_offset)
    return set(range(start_line, end_line + 1))


def source_function_line_ranges(source_code: str, contract_name: str) -> dict[str, tuple[int, int]]:
    target_range = find_contract_line_range(source_code, contract_name)
    if target_range is None:
        return {}
    start_line, end_line = target_range
    starts = byte_line_starts(source_code)
    matches: list[tuple[str, int]] = []
    pattern = re.compile(r"\b(function\s+[A-Za-z_][A-Za-z0-9_]*|constructor|fallback|receive)\s*\(")
    for match in pattern.finditer(source_code):
        line_no = line_for_byte_offset(starts, len(source_code[: match.start()].encode("utf-8", errors="replace")))
        if not (start_line <= line_no <= end_line):
            continue
        raw_name = match.group(1)
        name = raw_name.split()[-1] if raw_name.startswith("function ") else raw_name
        matches.append((name, line_no))
    ranges: dict[str, tuple[int, int]] = {}
    for index, (name, line_no) in enumerate(matches):
        next_line = matches[index + 1][1] - 1 if index + 1 < len(matches) else end_line
        ranges[f"{contract_name}.{name}.{index}"] = (line_no, max(line_no, next_line))
    return ranges


def trace_dump_command(args: argparse.Namespace, test_name: str, dump_path: Path) -> list[str]:
    return [
        args.forge_bin,
        "test",
        "--debug",
        "--match-contract",
        "CoverageReplayTest",
        "--match-test",
        test_name,
        "--allow-failure",
        "--dump",
        str(dump_path),
        "--fork-url",
        args.fork_url,
        "--fork-block-number",
        str(args.fork_block),
        "--initial-balance",
        str(SENDER_BALANCE_WEI),
    ]


def parse_trace_dump_line_coverage(
    dump_payload: Dict[str, Any],
    source_code: str,
    contract_name: str,
) -> tuple[set[int], set[int]]:
    contracts = dump_payload.get("contracts")
    if not isinstance(contracts, dict):
        return set(), set()
    sources_block = contracts.get("sources")
    if not isinstance(sources_block, dict):
        return set(), set()
    artifacts = sources_block.get("artifacts_by_name")
    sources_by_id = sources_block.get("sources_by_id")
    identified = contracts.get("identified_contracts")
    if not isinstance(artifacts, dict) or not isinstance(sources_by_id, dict) or not isinstance(identified, dict):
        return set(), set()

    artifact_list = artifacts.get(contract_name)
    if not isinstance(artifact_list, list) or not artifact_list or not isinstance(artifact_list[0], dict):
        return set(), set()
    artifact = artifact_list[0]
    build_id = str(artifact.get("build_id") or "")
    file_id = str(artifact.get("file_id"))
    source_group = sources_by_id.get(build_id)
    if not isinstance(source_group, dict):
        return set(), set()
    source_entry = source_group.get(file_id)
    if not isinstance(source_entry, dict) or not str(source_entry.get("path") or "").endswith("src/Target.sol"):
        return set(), set()

    target_range = find_contract_line_range(source_code, contract_name)
    if target_range is None:
        return set(), set()
    start_line, end_line = target_range
    line_starts = byte_line_starts(source_code)

    def line_set_from_source_map(source_map: Any) -> set[int]:
        lines: set[int] = set()
        if not isinstance(source_map, list):
            return lines
        for entry in source_map:
            if not isinstance(entry, dict):
                continue
            source_index = entry.get("index")
            if source_index is None or int(source_index) != int(file_id):
                continue
            offset_raw = entry.get("offset")
            offset = int(offset_raw) if offset_raw is not None else -1
            length_raw = entry.get("length")
            length = int(length_raw) if length_raw is not None else 0
            for line_no in lines_for_source_span(line_starts, offset, length):
                if start_line <= line_no <= end_line:
                    lines.add(line_no)
        return lines

    executable_lines = line_set_from_source_map(artifact.get("source_map"))
    executable_lines.update(line_set_from_source_map(artifact.get("source_map_runtime")))

    covered_lines: set[int] = set()
    debug_arena = dump_payload.get("debug_arena")
    if not isinstance(debug_arena, list):
        return executable_lines, covered_lines
    for arena in debug_arena:
        if not isinstance(arena, dict):
            continue
        address = str(arena.get("address") or "").lower()
        if identified.get(address) != contract_name:
            continue
        is_create = str(arena.get("kind") or "").upper() == "CREATE"
        source_map = artifact.get("source_map") if is_create else artifact.get("source_map_runtime")
        pc_map = artifact.get("pc_ic_map") if is_create else artifact.get("pc_ic_map_runtime")
        if not isinstance(source_map, list) or not isinstance(pc_map, dict):
            continue
        steps = arena.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            pc = str(step.get("pc"))
            instruction_index = pc_map.get(pc)
            if instruction_index is None:
                continue
            try:
                source_entry_for_pc = source_map[int(instruction_index)]
            except Exception:
                continue
            if not isinstance(source_entry_for_pc, dict):
                continue
            source_index = source_entry_for_pc.get("index")
            if source_index is None or int(source_index) != int(file_id):
                continue
            offset_raw = source_entry_for_pc.get("offset")
            offset = int(offset_raw) if offset_raw is not None else -1
            length_raw = source_entry_for_pc.get("length")
            length = int(length_raw) if length_raw is not None else 0
            for line_no in lines_for_source_span(line_starts, offset, length):
                if start_line <= line_no <= end_line:
                    covered_lines.add(line_no)
    return executable_lines, covered_lines


def coverage_from_trace_dumps(
    root: Path,
    source_code: str,
    contract_name: str,
    solc_version: str,
    calls: Sequence[Dict[str, Any]],
    constructor_exprs: Sequence[str],
    batch_count: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    trace_args = SimpleNamespace(**vars(args))
    trace_args.coverage_sender_variants = 1
    trace_args.coverage_auth_storage_slots = 0
    (root / "test" / "CoverageReplay.t.sol").write_text(
        generate_coverage_test_source(
            contract_name,
            solc_version,
            calls,
            constructor_exprs,
            trace_args,
            source_code=source_code,
        ),
        encoding="utf-8",
    )
    batch_count = (len(calls) + max(1, int(getattr(trace_args, "coverage_batch_size", 25))) - 1) // max(
        1, int(getattr(trace_args, "coverage_batch_size", 25))
    )
    executable_lines: set[int] = set()
    covered_lines: set[int] = set()
    trace_runs: list[Dict[str, Any]] = []
    trace_count = min(batch_count, max(0, int(getattr(args, "coverage_max_trace_tests", 8))))
    trace_timeout = max(1, int(getattr(args, "coverage_trace_timeout", args.timeout)))
    for batch_index in range(trace_count):
        test_name = f"testCoverageReplay_{batch_index:04d}"
        dump_path = root / f"trace_dump_{batch_index:04d}.json"
        cmd = trace_dump_command(args, test_name, dump_path)
        try:
            proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=trace_timeout)
        except subprocess.TimeoutExpired:
            trace_runs.append({"test": test_name, "cmd": cmd, "timed_out": True})
            continue
        except Exception as exc:
            trace_runs.append({"test": test_name, "cmd": cmd, "error": truncate(repr(exc), 1000)})
            continue
        trace_runs.append(
            {
                "test": test_name,
                "cmd": cmd,
                "exit_code": proc.returncode,
                "stdout": truncate(proc.stdout, 2000),
                "stderr": truncate(proc.stderr, 2000),
                "dump_exists": dump_path.exists(),
            }
        )
        if not dump_path.exists():
            continue
        try:
            payload = json.loads(dump_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        total_for_dump, covered_for_dump = parse_trace_dump_line_coverage(payload, source_code, contract_name)
        executable_lines.update(total_for_dump)
        covered_lines.update(covered_for_dump)

    if not executable_lines:
        return {
            "available": False,
            "reason": "trace_dump_source_map_unavailable",
            "trace_runs": trace_runs,
        }
    if not covered_lines:
        return {
            "available": False,
            "reason": "trace_dump_no_target_execution",
            "trace_runs": trace_runs,
            "executable_line_count": len(executable_lines),
        }
    function_ranges = source_function_line_ranges(source_code, contract_name)
    function_hits = {
        name: 1 if any(start <= line_no <= end for line_no in covered_lines) else 0
        for name, (start, end) in function_ranges.items()
    }
    function_lines = {name: start for name, (start, _) in function_ranges.items()}
    line_hits = {line_no: (1 if line_no in covered_lines else 0) for line_no in executable_lines}
    metrics = coverage_metrics_from_hits(line_hits, function_lines, function_hits, [])
    metrics["contract_name"] = contract_name
    target_range = find_contract_line_range(source_code, contract_name)
    if target_range is not None:
        metrics["line_range"] = list(target_range)
    return {
        "kind": "foundry_trace_source_map",
        "instrumentation_schema": COVERAGE_SCHEMA,
        "available": True,
        "coverage_scope": "target_contract",
        "source_file": "src/Target.sol",
        "source_coverage_method": "foundry_debug_dump_source_map",
        "lines": metrics["lines"],
        "functions": metrics["functions"],
        "branches": metrics["branches"],
        "target_contract": metrics,
        "full_source": metrics,
        "trace_runs": trace_runs,
    }


def parse_lcov_report(
    text: str,
    source_suffix: str = "src/Target.sol",
    source_code: str = "",
    target_contract_name: str = "",
) -> Dict[str, Any]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line == "TN:":
            current = {
                "source_file": "",
                "line_hits": {},
                "functions": {},
                "function_hits": {},
                "branches": [],
            }
            continue
        if line == "end_of_record":
            if current is not None:
                records.append(current)
            current = None
            continue
        if current is None:
            continue
        if line.startswith("SF:"):
            current["source_file"] = line[3:]
        elif line.startswith("DA:"):
            parts = line[3:].split(",", 1)
            if len(parts) == 2:
                try:
                    current["line_hits"][int(parts[0])] = int(parts[1])
                except ValueError:
                    pass
        elif line.startswith("FN:"):
            parts = line[3:].split(",", 1)
            if len(parts) == 2:
                try:
                    current["functions"][parts[1]] = int(parts[0])
                except ValueError:
                    current["functions"][parts[1]] = None
        elif line.startswith("FNDA:"):
            parts = line[5:].split(",", 1)
            if len(parts) == 2:
                try:
                    current["function_hits"][parts[1]] = int(parts[0])
                except ValueError:
                    pass
        elif line.startswith("BRDA:"):
            parts = line[5:].split(",")
            if len(parts) == 4:
                taken = parts[3]
                try:
                    line_no = int(parts[0])
                except ValueError:
                    line_no = 0
                current["branches"].append({"line": line_no, "hits": 0 if taken == "-" else int(taken)})

    target_records = [r for r in records if str(r.get("source_file") or "").endswith(source_suffix)]
    if not target_records:
        return {"available": False, "reason": "target_source_missing_from_lcov"}

    line_hits: dict[int, int] = {}
    function_hits: dict[str, int] = {}
    function_lines: dict[str, Any] = {}
    branches: list[int] = []
    for record in target_records:
        for line_no, hits in record.get("line_hits", {}).items():
            line_hits[line_no] = line_hits.get(line_no, 0) + hits
        for name, line_no in record.get("functions", {}).items():
            function_lines[name] = line_no
        for name, hits in record.get("function_hits", {}).items():
            function_hits[name] = function_hits.get(name, 0) + hits
        branches.extend(record.get("branches", []))

    full_source = coverage_metrics_from_hits(line_hits, function_lines, function_hits, branches)
    result = {
        "available": True,
        "source_file": source_suffix,
        "coverage_scope": "target_contract" if target_contract_name else "full_source",
        "full_source": full_source,
        "lines": full_source["lines"],
        "functions": full_source["functions"],
        "branches": full_source["branches"],
    }
    target_range = find_contract_line_range(source_code, target_contract_name)
    if target_range is not None:
        start_line, end_line = target_range
        target_line_hits = {
            line_no: hits
            for line_no, hits in line_hits.items()
            if start_line <= line_no <= end_line
        }
        prefix = f"{target_contract_name}."
        target_function_lines = {
            name: line_no
            for name, line_no in function_lines.items()
            if name.startswith(prefix)
        }
        target_function_hits = {
            name: hits
            for name, hits in function_hits.items()
            if name in target_function_lines
        }
        target_branches = [
            branch
            for branch in branches
            if start_line <= int(branch.get("line") or 0) <= end_line
        ]
        target_metrics = coverage_metrics_from_hits(
            target_line_hits,
            target_function_lines,
            target_function_hits,
            target_branches,
        )
        target_metrics["contract_name"] = target_contract_name
        target_metrics["line_range"] = [start_line, end_line]
        if int(target_metrics.get("lines", {}).get("total") or 0) == 0:
            return {
                "available": False,
                "reason": "target_contract_no_executable_lines",
                "source_file": source_suffix,
                "coverage_scope": "target_contract",
                "full_source": full_source,
                "target_contract": target_metrics,
            }
        result["target_contract"] = target_metrics
        result["coverage_scope"] = "target_contract"
        result["lines"] = target_metrics["lines"]
        result["functions"] = target_metrics["functions"]
        result["branches"] = target_metrics["branches"]
    return result


def constructor_abi_entry(abi: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for item in abi:
        if isinstance(item, dict) and item.get("type") == "constructor":
            return item
    return None


def split_top_level_commas(raw: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_string: str | None = None
    escaped = False
    for index, ch in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_string:
                in_string = None
            continue
        if ch in {"'", '"'}:
            in_string = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(raw[start:index].strip())
            start = index + 1
    tail = raw[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def constructor_source_param_types(source_code: str, contract_name: str) -> list[str]:
    patterns = [r"\bconstructor\s*\((?P<params>.*?)\)"]
    if contract_name:
        patterns.append(r"\bfunction\s+" + re.escape(contract_name) + r"\s*\((?P<params>.*?)\)")
    for pattern in patterns:
        match = re.search(pattern, source_code, re.DOTALL)
        if match is None:
            continue
        types: list[str] = []
        for raw_param in split_top_level_commas(match.group("params")):
            clean = re.sub(r"/\*.*?\*/", " ", raw_param, flags=re.DOTALL)
            clean = re.sub(r"//.*", " ", clean).strip()
            if not clean:
                continue
            tokens = clean.split()
            if len(tokens) == 1:
                types.append(tokens[0])
                continue
            param_type = " ".join(tokens[:-1])
            param_type = re.sub(r"\s+(memory|storage|calldata)$", "", param_type).strip()
            types.append(param_type)
        return types
    return []


def solidity_identifier_from_internal_type(internal_type: str) -> str | None:
    value = str(internal_type or "").strip()
    for prefix in ("contract ", "interface ", "library "):
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
            break
    value = re.sub(r"\s+(storage|memory|calldata)$", "", value)
    base, dims = split_array_type(value)
    if "." in base:
        base = base.split(".")[-1]
    if IDENTIFIER_RE.fullmatch(base):
        return base + array_suffix(dims)
    return None


def solidity_array_new_type(entry: Dict[str, Any]) -> str | None:
    raw_type = str(entry.get("type") or "")
    base, dims = split_array_type(raw_type)
    if not dims or dims[-1] is not None:
        return None
    internal = str(entry.get("internalType") or "")
    if internal.startswith(("contract ", "interface ", "library ")):
        return solidity_identifier_from_internal_type(internal)
    if base == "tuple":
        return None
    return raw_type


def solidity_string_literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def solidity_bytes_literal(value: Any) -> str:
    if isinstance(value, bytes):
        body = value.hex()
    elif isinstance(value, str) and value.startswith("0x") and re.fullmatch(r"0x[0-9a-fA-F]*", value):
        body = value[2:]
    else:
        body = bytes(str(value), "utf-8").hex()
    return f'hex"{body}"'


def address_uint160_expr(raw: str) -> str:
    if not re.fullmatch(r"0x[0-9a-fA-F]{40}", raw):
        raw = "0x" + "00" * 19 + "01"
    return f"address(uint160({int(raw, 16)}))"


def solidity_address_expr(value: Any, entry: Dict[str, Any], version: tuple[int, int, int]) -> str:
    raw = str(value or "0x" + "00" * 19 + "01")
    address_expr = address_uint160_expr(raw)
    internal = str(entry.get("internalType") or "")
    contract_type = solidity_identifier_from_internal_type(internal)
    if contract_type and not contract_type.startswith("address"):
        return f"{contract_type}({address_expr})"
    if "payable" in internal and version >= (0, 6, 0):
        return f"payable({address_expr})"
    return address_expr


def solidity_test_contract_address_expr(entry: Dict[str, Any], solc_version: str) -> str:
    version = solidity_version_tuple(solc_version)
    internal = str(entry.get("internalType") or "")
    contract_type = solidity_identifier_from_internal_type(internal)
    if contract_type and not contract_type.startswith("address"):
        return f"{contract_type}(address(this))"
    if "payable" in internal and version >= (0, 6, 0):
        return "payable(address(this))"
    return "address(this)"


def solidity_mock_contract_address_expr(entry: Dict[str, Any], solc_version: str) -> str | None:
    version = solidity_version_tuple(solc_version)
    if version < (0, 5, 0):
        return None
    internal = str(entry.get("internalType") or "")
    contract_type = solidity_identifier_from_internal_type(internal)
    base_expr = "address(new CoverageMock())"
    if contract_type and not contract_type.startswith("address"):
        return f"{contract_type}({base_expr})"
    if "payable" in internal and version >= (0, 6, 0):
        return f"payable({base_expr})"
    return base_expr


def is_authority_address_entry(entry: Dict[str, Any]) -> bool:
    raw_type = str(entry.get("type") or "")
    base, dims = split_array_type(raw_type)
    if dims:
        return False
    if base != "address" and not is_contract_abi_type(entry, base):
        return False
    text = f"{entry.get('name') or ''} {entry.get('internalType') or ''}".lower()
    return any(hint in text for hint in AUTHORITY_PARAM_HINTS)


def preferred_constructor_scalar_value(entry: Dict[str, Any]) -> Any:
    raw_type = str(entry.get("type") or "")
    base, dims = split_array_type(raw_type)
    if dims:
        return None
    text = f"{entry.get('name') or ''} {entry.get('internalType') or ''}".lower()
    if is_contract_abi_type(entry, base) or (
        base == "address" and any(hint in text for hint in CONSTRUCTOR_CODE_ADDRESS_HINTS)
    ):
        return "<coverage_mock_contract>"
    if base.startswith("uint"):
        bits = int(base[4:] or 256)
        hi = max_uint_for_bits(bits)
        if "decimal" in text:
            return min(18, hi)
        if any(hint in text for hint in CONSTRUCTOR_TIMESTAMP_START_HINTS):
            return min(max(1, COVERAGE_TIMESTAMP - 86400), hi)
        if any(hint in text for hint in CONSTRUCTOR_TIMESTAMP_END_HINTS):
            return min(COVERAGE_TIMESTAMP + 31536000, hi)
        if any(hint in text for hint in CONSTRUCTOR_DURATION_HINTS):
            return min(7 * 24 * 60 * 60, hi)
        if any(hint in text for hint in CONSTRUCTOR_AMOUNT_HINTS):
            return min(10**18, hi)
        if any(hint in text for hint in ("id", "index", "count", "number", "num")):
            return min(1, hi)
    if base.startswith("int"):
        bits = int(base[3:] or 256)
        lo, hi = int_bounds(bits)
        if any(hint in text for hint in CONSTRUCTOR_TIMESTAMP_START_HINTS):
            return max(lo, min(hi, COVERAGE_TIMESTAMP - 86400))
        if any(hint in text for hint in CONSTRUCTOR_TIMESTAMP_END_HINTS):
            return max(lo, min(hi, COVERAGE_TIMESTAMP + 31536000))
        if any(hint in text for hint in CONSTRUCTOR_DURATION_HINTS):
            return max(lo, min(hi, 31536000))
        if any(hint in text for hint in CONSTRUCTOR_AMOUNT_HINTS):
            return max(lo, min(hi, 10**18))
    if base == "bool":
        if "paused" in text:
            return False
        if any(hint in text for hint in ("enable", "enabled", "active", "revocable")):
            return True
    return None


def solidity_literal_for_abi_value(value: Any, entry: Dict[str, Any], solc_version: str) -> str | None:
    version = solidity_version_tuple(solc_version)
    typ = str(entry.get("type") or "")
    base, dims = split_array_type(typ)
    if dims:
        if dims[-1] is None:
            new_type = solidity_array_new_type(entry)
            if new_type is None:
                return None
            return f"new {new_type}(0)"
        if not isinstance(value, list):
            return None
        element_entry = dict(entry)
        element_entry["type"] = ("address" if is_contract_abi_type(entry, base) else base) + array_suffix(dims[:-1])
        items = [
            solidity_literal_for_abi_value(item, element_entry, solc_version)
            for item in value
        ]
        if any(item is None for item in items):
            return None
        return "[" + ", ".join(str(item) for item in items) + "]"
    if is_contract_abi_type(entry, base):
        return solidity_address_expr(value, entry, version)
    if base == "address":
        return solidity_address_expr(value, entry, version)
    if base == "bool":
        return "true" if bool(value) else "false"
    if base.startswith("uint"):
        return f"{base}({int(value)})"
    if base.startswith("int"):
        return f"{base}({int(value)})"
    if base.startswith("bytes") and base != "bytes":
        return f"{base}({solidity_bytes_literal(value)})"
    if base == "bytes":
        return solidity_bytes_literal(value)
    if base == "string":
        return solidity_string_literal(str(value))
    return None


def coverage_constructor_expressions(
    row: Dict[str, Any],
    abi: Sequence[Dict[str, Any]] | None,
    solc_version: str,
    args: argparse.Namespace,
) -> tuple[list[str] | None, list[Any], str]:
    constructor = constructor_abi_entry(abi or [])
    inputs = constructor.get("inputs") if isinstance(constructor, dict) else []
    if not isinstance(inputs, list) or not inputs:
        return [], [], ""
    if not getattr(args, "coverage_synthetic_constructor_args", True):
        return None, [], "constructor_args_required"

    hints = extract_static_hints(row, args)
    source_param_types = constructor_source_param_types(
        str(row.get("source_code") or ""),
        str(row.get("contract_name") or ""),
    )
    line_seed = int(row.get("_dataset_line") or row.get("line_no") or 0)
    exprs: list[str] = []
    values: list[Any] = []
    for arg_index, inp in enumerate(inputs):
        if not isinstance(inp, dict):
            return None, [], "constructor_abi_malformed"
        typed_inp = dict(inp)
        if arg_index < len(source_param_types) and source_param_types[arg_index]:
            typed_inp.setdefault("internalType", source_param_types[arg_index])
        if is_authority_address_entry(typed_inp):
            exprs.append(solidity_test_contract_address_expr(typed_inp, solc_version))
            values.append("<coverage_test_contract>")
            continue
        preferred_value = preferred_constructor_scalar_value(typed_inp)
        if preferred_value == "<coverage_mock_contract>":
            mock_expr = solidity_mock_contract_address_expr(typed_inp, solc_version)
            if mock_expr is not None:
                exprs.append(mock_expr)
                values.append("<coverage_mock_contract>")
                continue
            preferred_value = "<coverage_test_contract>"
        if preferred_value == "<coverage_test_contract>":
            exprs.append(solidity_test_contract_address_expr(typed_inp, solc_version))
            values.append("<coverage_test_contract>")
            continue
        if preferred_value is not None:
            value = preferred_value
            expr = solidity_literal_for_abi_value(value, typed_inp, solc_version)
            if expr is None:
                return None, [], f"synthetic_constructor_arg_literal_unsupported:{typed_inp.get('type')}"
            exprs.append(expr)
            values.append(jsonable_value(value))
            continue
        hint_values = static_hint_values_for_entry(typed_inp, hints, args)
        if hint_values:
            value = hint_values[0]
        else:
            try:
                value = gen_branch_abi_value(
                    typed_inp,
                    args.seed + line_seed * 10000 + arg_index * 101,
                    (arg_index + 1) % len(BRANCH_CASE_VARIANTS),
                    args,
                )
            except Exception as exc:
                return None, [], f"synthetic_constructor_arg_unsupported:{exc}"
        expr = solidity_literal_for_abi_value(value, typed_inp, solc_version)
        if expr is None:
            return None, [], f"synthetic_constructor_arg_literal_unsupported:{typed_inp.get('type')}"
        exprs.append(expr)
        values.append(jsonable_value(value))
    return exprs, values, ""


def coverage_command(args: argparse.Namespace, solc_version: str, ir_minimum: bool = False) -> list[str]:
    cmd = [
        args.forge_bin,
        "coverage",
        "--report",
        "lcov",
        "--report-file",
        "lcov.info",
        "--exclude-tests",
        "--fork-url",
        args.fork_url,
        "--fork-block-number",
        str(args.fork_block),
        "--initial-balance",
        str((1 << 128) - 1),
    ]
    if ir_minimum and solidity_version_tuple(solc_version) >= (0, 8, 13):
        cmd.append("--ir-minimum")
    if getattr(args, "coverage_allow_failure", True):
        cmd.append("--allow-failure")
    return cmd


def coverage_failure_text(proc: subprocess.CompletedProcess[str]) -> str:
    return f"{proc.stderr or ''}\n{proc.stdout or ''}"


def should_retry_coverage_with_ir(proc: subprocess.CompletedProcess[str], solc_version: str, args: argparse.Namespace) -> bool:
    if not getattr(args, "coverage_ir_minimum_retry", True):
        return False
    if solidity_version_tuple(solc_version) < (0, 8, 13):
        return False
    text = coverage_failure_text(proc).lower()
    return "stack too deep" in text


def coverage_augmented_calls(
    row: Dict[str, Any],
    abi: Sequence[Dict[str, Any]] | None,
    base_calls: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[Dict[str, Any]], int]:
    calls = [dict(call) for call in base_calls if isinstance(call, dict)]
    if not abi:
        return calls, 0
    aug_args = SimpleNamespace(**vars(args))
    aug_args.runs = int(getattr(args, "coverage_augment_runs", 24))
    aug_args.branch_cases = True
    aug_args.static_hint_cases = True
    aug_args.stateful_setup_cases = True
    aug_args.max_static_hint_cases_per_function = int(
        getattr(args, "coverage_augment_static_cases_per_function", 32)
    )
    aug_args.max_setup_functions = int(getattr(args, "coverage_augment_setup_functions", 6))
    aug_args.max_stateful_cases_per_function = int(
        getattr(args, "coverage_augment_stateful_cases_per_function", 4)
    )
    aug_args.large_fallback_calldata = True
    line_seed = int(row.get("_dataset_line") or row.get("line_no") or 0)
    try:
        supplemental, _ = generate_calls(abi, line_seed, aug_args, row)
    except Exception:
        return calls, 0
    if not getattr(args, "coverage_augment_calls", True):
        if getattr(args, "coverage_augment_fallback_calls", True):
            supplemental = [call for call in supplemental if call.get("entry_type") == "fallback"]
        else:
            supplemental = []
    elif not getattr(args, "coverage_augment_fallback_calls", True):
        supplemental = [call for call in supplemental if call.get("entry_type") != "fallback"]

    existing: set[str] = set()
    for call in calls:
        setup_key = json.dumps(call.get("setup") or [], sort_keys=True)
        existing.add(f"{str(call.get('calldata') or '').lower()}:{int(call.get('value') or 0)}:{setup_key}")
    added = 0
    for call in supplemental:
        setup_key = json.dumps(call.get("setup") or [], sort_keys=True)
        key = f"{str(call.get('calldata') or '').lower()}:{int(call.get('value') or 0)}:{setup_key}"
        if key in existing:
            continue
        existing.add(key)
        new_call = dict(call)
        new_call["coverage_supplemental"] = True
        calls.append(new_call)
        added += 1
    return calls, added


def run_source_coverage(
    row: Dict[str, Any],
    calls: Sequence[Dict[str, Any]],
    deployment_mode: str,
    args: argparse.Namespace,
    abi: Sequence[Dict[str, Any]] | None = None,
    source_matches: bool | None = None,
) -> Dict[str, Any]:
    coverage: Dict[str, Any] = {
        "kind": "foundry_source_lcov",
        "instrumentation_schema": COVERAGE_SCHEMA,
        "available": False,
    }
    exact_source_creation = deployment_mode == "ground_truth_source_creation"
    if not exact_source_creation and not (source_matches and getattr(args, "coverage_synthetic_constructor_args", True)):
        coverage["reason"] = "source_creation_not_recoverable"
        return coverage
    contract_name = str(row.get("contract_name") or "")
    if not IDENTIFIER_RE.fullmatch(contract_name):
        coverage["reason"] = "invalid_contract_name"
        return coverage
    solc_version = extract_solc_version(compiler_version_from_row(row))
    if not solc_version:
        coverage["reason"] = "missing_solc_version"
        return coverage
    coverage["solc_version"] = solc_version
    if solidity_version_tuple(solc_version) < (0, 5, 0) and not getattr(args, "coverage_legacy_solidity", True):
        coverage["reason"] = "legacy_solidity_0_4_coverage_skipped"
        return coverage
    source_code = str(row.get("source_code") or "")
    if not source_code.strip():
        coverage["reason"] = "missing_source_code"
        return coverage

    constructor_exprs, constructor_values, constructor_error = coverage_constructor_expressions(
        row,
        abi,
        solc_version,
        args,
    )
    if constructor_exprs is None:
        coverage["reason"] = constructor_error or "constructor_args_not_available"
        return coverage
    coverage["source_deployment_mode"] = (
        "exact_source_creation" if exact_source_creation and not constructor_exprs else "synthetic_constructor_source_creation"
    )
    coverage["constructor_arg_count"] = len(constructor_exprs)
    if constructor_values:
        coverage["synthetic_constructor_args"] = constructor_values
    coverage_calls, supplemental_call_count = coverage_augmented_calls(row, abi, calls, args)
    coverage["checkpoint_call_count"] = len(calls)
    coverage["supplemental_call_count"] = supplemental_call_count

    with tempfile.TemporaryDirectory(prefix="scdbench-source-coverage-", dir="/tmp") as tmp:
        root = Path(tmp)
        batch_count = (len(coverage_calls) + max(1, int(getattr(args, "coverage_batch_size", 25))) - 1) // max(
            1, int(getattr(args, "coverage_batch_size", 25))
        )
        (root / "src").mkdir(parents=True)
        (root / "test").mkdir()
        (root / "src" / "Target.sol").write_text(source_code, encoding="utf-8")
        (root / "test" / "CoverageReplay.t.sol").write_text(
            generate_coverage_test_source(
                contract_name,
                solc_version,
                coverage_calls,
                constructor_exprs,
                args,
                source_code=source_code,
                abi=abi,
            ),
            encoding="utf-8",
        )
        (root / "foundry.toml").write_text(
            "\n".join(
                [
                    "[profile.default]",
                    'src = "src"',
                    'test = "test"',
                    'out = "out"',
                    'cache_path = "cache"',
                    "auto_detect_solc = true",
                    "optimizer = false",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        cmd = coverage_command(args, solc_version, ir_minimum=False)
        proc: subprocess.CompletedProcess[str]
        try:
            proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            coverage["reason"] = f"forge_coverage_timed_out_after_{args.timeout}s"
            return coverage
        except Exception as exc:
            coverage["reason"] = f"forge_coverage_error:{exc}"
            return coverage
        retried_with_ir = False
        if proc.returncode != 0 and should_retry_coverage_with_ir(proc, solc_version, args):
            retried_with_ir = True
            cmd = coverage_command(args, solc_version, ir_minimum=True)
            try:
                proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                coverage["reason"] = f"forge_coverage_ir_minimum_timed_out_after_{args.timeout}s"
                return coverage
            except Exception as exc:
                coverage["reason"] = f"forge_coverage_ir_minimum_error:{exc}"
                return coverage
        coverage["forge"] = {
            "cmd": cmd,
            "exit_code": proc.returncode,
            "retried_with_ir_minimum": retried_with_ir,
            "stdout": truncate(proc.stdout, 8000),
            "stderr": truncate(proc.stderr, 8000),
        }
        lcov_path = root / "lcov.info"
        if proc.returncode != 0:
            stderr = proc.stderr or ""
            if "solar run failed" in stderr:
                coverage["reason"] = "foundry_coverage_parser_failed"
            else:
                coverage["reason"] = "forge_coverage_failed"
            if batch_count > 0:
                fallback = coverage_from_trace_dumps(
                    root,
                    source_code,
                    contract_name,
                    solc_version,
                    coverage_calls,
                    constructor_exprs,
                    batch_count,
                    args,
                )
                if fallback.get("available"):
                    fallback["lcov_failure_reason"] = coverage["reason"]
                    fallback["forge_lcov"] = coverage.get("forge")
                    return fallback
                coverage["trace_fallback"] = fallback
            return coverage
        if not lcov_path.exists():
            coverage["reason"] = "lcov_report_missing"
            if batch_count > 0:
                fallback = coverage_from_trace_dumps(
                    root,
                    source_code,
                    contract_name,
                    solc_version,
                    coverage_calls,
                    constructor_exprs,
                    batch_count,
                    args,
                )
                if fallback.get("available"):
                    fallback["lcov_failure_reason"] = coverage["reason"]
                    fallback["forge_lcov"] = coverage.get("forge")
                    return fallback
                coverage["trace_fallback"] = fallback
            return coverage
        parsed = parse_lcov_report(
            lcov_path.read_text(encoding="utf-8"),
            source_code=source_code,
            target_contract_name=contract_name,
        )
        if not parsed.get("available"):
            coverage["reason"] = parsed.get("reason") or "coverage_unavailable"
            return coverage
        coverage.update(parsed)
        return coverage


def foundry_env(benchmark_id: str, original_init: str, calls: Sequence[Dict[str, Any]], predicted_init: str | None = None) -> Dict[str, str]:
    env = os.environ.copy()
    env["SCDBENCH_ID"] = benchmark_id
    env["SCDBENCH_ORIGINAL_INIT"] = ensure0x(original_init)
    env["SCDBENCH_HAS_PREDICTED"] = "1" if predicted_init else "0"
    if predicted_init:
        env["SCDBENCH_PREDICTED_INIT"] = ensure0x(predicted_init)
    env["SCDBENCH_CALLS"] = str(len(calls))
    for i, call in enumerate(calls):
        env[f"SCDBENCH_CASE_ID_{i}"] = str(call.get("case_id") or f"{benchmark_id}-case-{i:04d}")
        env[f"SCDBENCH_SIG_{i}"] = str(call.get("signature") or "")
        env[f"SCDBENCH_CALLDATA_{i}"] = ensure0x(str(call.get("calldata") or "0x"))
        env[f"SCDBENCH_VALUE_{i}"] = str(int(call.get("value") or 0))
        setup = call.get("setup") or []
        if not isinstance(setup, list):
            setup = []
        env[f"SCDBENCH_SETUP_COUNT_{i}"] = str(len(setup))
        for setup_index, setup_call in enumerate(setup):
            if not isinstance(setup_call, dict):
                continue
            env[f"SCDBENCH_SETUP_CALLDATA_{i}_{setup_index}"] = ensure0x(str(setup_call.get("calldata") or "0x"))
            env[f"SCDBENCH_SETUP_VALUE_{i}_{setup_index}"] = str(int(setup_call.get("value") or 0))
    return env


def ensure0x(value: str) -> str:
    return value if value.startswith("0x") else "0x" + value


def run_forge(args: argparse.Namespace, env: Dict[str, str], coverage: bool = False) -> Dict[str, Any]:
    forge = shutil.which(args.forge_bin) or args.forge_bin
    base = [
        forge,
        "coverage" if coverage else "test",
        "--match-contract",
        "SemanticReplayTest",
        "--match-test",
        "testSemanticReplay",
        "--fork-url",
        args.fork_url,
        "--fork-block-number",
        str(args.fork_block),
    ]
    if coverage:
        base += ["--report", args.coverage_report]
    else:
        base += ["-vv"]
    started = time.time()
    proc = subprocess.run(base, cwd=FOUNDRY_ROOT, env=env, capture_output=True, text=True, timeout=args.timeout)
    return {
        "cmd": base,
        "exit_code": proc.returncode,
        "latency_sec": time.time() - started,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def merge_observations(
    target: Dict[str, Dict[str, Dict[str, Any]]],
    source: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    for case_id, by_target in source.items():
        dst_case = target.setdefault(case_id, {})
        for target_name, observation in by_target.items():
            dst_case[target_name] = observation


def classify_forge_failure(run: Dict[str, Any]) -> str:
    text = f"{run.get('stdout') or ''}\n{run.get('stderr') or ''}"
    if "OutOfGas" in text:
        return "harness_out_of_gas"
    if "MemoryOOG" in text:
        return "harness_memory_oog"
    if "SCDBench deploy failed" in text:
        return "deploy_failed"
    if "timed out" in text.lower():
        return "forge_timeout"
    return "forge_test_failed"


def run_forge_observations(
    args: argparse.Namespace,
    benchmark_id: str,
    original_init: str,
    calls: Sequence[Dict[str, Any]],
    predicted_init: str | None = None,
) -> tuple[
    Dict[str, Dict[str, Dict[str, Any]]],
    list[Dict[str, Any]],
    Optional[Dict[str, Any]],
    list[Dict[str, Any]],
]:
    observations: Dict[str, Dict[str, Dict[str, Any]]] = {}
    runs: list[Dict[str, Any]] = []
    skipped_cases: list[Dict[str, Any]] = []
    chunk_size = max(1, int(args.chunk_size))

    def run_chunk(chunk: list[Dict[str, Any]], start: int, chunk_index: int, depth: int) -> Optional[Dict[str, Any]]:
        env = foundry_env(benchmark_id, original_init, chunk, predicted_init)
        run = run_forge(args, env, coverage=False)
        compact = compact_forge_run(run)
        if compact is not None:
            compact["chunk_index"] = chunk_index
            compact["chunk_start"] = start
            compact["chunk_calls"] = len(chunk)
            compact["split_depth"] = depth
            compact["case_ids"] = [str(call.get("case_id") or "") for call in chunk]
        runs.append(compact or {})
        if run["exit_code"] != 0:
            if len(chunk) > 1:
                mid = len(chunk) // 2
                failed_left = run_chunk(chunk[:mid], start, chunk_index, depth + 1)
                if failed_left is not None:
                    return failed_left
                return run_chunk(chunk[mid:], start + mid, chunk_index, depth + 1)
            if "SCDBENCH_DEPLOY" in str(run.get("stdout") or ""):
                call = chunk[0]
                skipped_cases.append(
                    {
                        "case_id": call.get("case_id"),
                        "signature": call.get("signature"),
                        "entry_type": call.get("entry_type"),
                        "state": call.get("state"),
                        "case_kind": call.get("case_kind"),
                        "reason": classify_forge_failure(run),
                        "forge": compact,
                    }
                )
                return None
            return compact
        merge_observations(observations, parse_foundry_observations(run["stdout"]))
        return None

    for chunk_index, start in enumerate(range(0, len(calls), chunk_size)):
        chunk = list(calls[start : start + chunk_size])
        failed = run_chunk(chunk, start, chunk_index, 0)
        if failed is not None:
            return observations, runs, failed, skipped_cases
    return observations, runs, None, skipped_cases


def parse_key_values(line: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key] = value
    return out


def parse_foundry_observations(stdout: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for raw in stdout.splitlines():
        if "SCDBENCH_CALL " in raw:
            payload = raw.split("SCDBENCH_CALL ", 1)[1]
            fields = parse_key_values(payload)
            case_id = fields.get("case_id", "")
            target = fields.get("target", "")
            case = out.setdefault(case_id, {})
            case[target] = {
                "call": {
                    "ok": fields.get("success") == "1",
                    "result": fields.get("returndata", "0x").lower(),
                },
                "logs": [],
                "storage_changes": {},
                "storage_touched": {},
            }
        elif "SCDBENCH_STORAGE " in raw:
            payload = raw.split("SCDBENCH_STORAGE ", 1)[1]
            fields = parse_key_values(payload)
            case_id = fields.get("case_id", "")
            target = fields.get("target", "")
            slot = fields.get("slot", "")
            case = out.setdefault(case_id, {})
            obs = case.setdefault(
                target,
                {"call": {"ok": False, "result": "0x"}, "logs": [], "storage_changes": {}, "storage_touched": {}},
            )
            if slot:
                before = fields.get("before", "0x")
                after = fields.get("after", "0x")
                obs.setdefault("storage_touched", {})[slot.lower()] = {
                    "before": before.lower(),
                    "after": after.lower(),
                }
                if before.lower() != after.lower():
                    obs.setdefault("storage_changes", {})[slot.lower()] = {
                        "before": before.lower(),
                        "after": after.lower(),
                    }
        elif "SCDBENCH_LOG " in raw:
            payload = raw.split("SCDBENCH_LOG ", 1)[1]
            fields = parse_key_values(payload)
            case_id = fields.get("case_id", "")
            target = fields.get("target", "")
            topics_raw = fields.get("topics", "")
            topics = [part.lower() for part in topics_raw.split(",") if part]
            case = out.setdefault(case_id, {})
            obs = case.setdefault(
                target,
                {"call": {"ok": False, "result": "0x"}, "logs": [], "storage_changes": {}, "storage_touched": {}},
            )
            obs.setdefault("logs", []).append(
                {
                    "emitter": fields.get("emitter", "").lower(),
                    "topics": topics,
                    "data": fields.get("data", "0x").lower(),
                }
            )
    return out


def call_equal(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    exp = expected.get("call") or {}
    got = actual.get("call") or {}
    if bool(exp.get("ok")) != bool(got.get("ok")):
        return False
    return str(exp.get("result") or "0x").lower() == str(got.get("result") or "0x").lower()


def observation_equal(expected: Dict[str, Any], actual: Dict[str, Any], state: str) -> bool:
    if not call_equal(expected, actual):
        return False
    if "logs" in expected and (expected.get("logs") or []) != (actual.get("logs") or []):
        return False
    if state in {"view", "pure"}:
        return True
    return (expected.get("storage_changes") or {}) == (actual.get("storage_changes") or {})


def build_ground_truth_record(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    bid = str(row.get("benchmark_id") or "")
    gt_compile = compile_ground_truth(row, args.compile_timeout)
    if not gt_compile.success:
        return {"benchmark_id": bid, "difficulty": row.get("difficulty_bin"), "error": "ground_truth_compile_failed", "compile_error": gt_compile.error}
    constructor_inputs = constructor_input_count(gt_compile.abi)
    source_matches = source_bytecode_matches_input(row, gt_compile.deployed_bytecode)
    runtime_wrapper_init = make_init_code_from_runtime(ensure0x(str(row.get("bytecode") or "")))
    source_creation_fallback: Optional[Dict[str, Any]] = None
    if constructor_inputs == 0 and source_matches and gt_compile.bytecode:
        original_init = ensure0x(gt_compile.bytecode)
        deployment_mode = "ground_truth_source_creation"
    else:
        original_init = runtime_wrapper_init
        deployment_mode = "input_runtime_wrapper"
    calls, unsupported_entries = generate_calls(
        gt_compile.abi,
        int(row.get("_dataset_line") or row.get("line_no") or 0),
        args,
        row,
    )
    for i, call in enumerate(calls):
        call["case_id"] = f"{bid}-case-{i:04d}"
    if not calls:
        return {
            "benchmark_id": bid,
            "difficulty": row.get("difficulty_bin"),
            "error": "no_supported_abi_calls",
            "unsupported_entries": unsupported_entries,
        }
    observations, forge_runs, failed_run, skipped_cases = run_forge_observations(args, bid, original_init, calls)
    if (
        failed_run is not None
        and deployment_mode == "ground_truth_source_creation"
        and classify_forge_failure(failed_run) == "deploy_failed"
    ):
        source_creation_fallback = {
            "reason": "source_creation_deploy_failed",
            "forge": failed_run,
            "forge_runs": forge_runs,
        }
        deployment_mode = "input_runtime_wrapper"
        observations, forge_runs, failed_run, skipped_cases = run_forge_observations(
            args,
            bid,
            runtime_wrapper_init,
            calls,
        )
    if failed_run is not None:
        record = {
            "benchmark_id": bid,
            "difficulty": row.get("difficulty_bin"),
            "error": "forge_test_failed",
            "forge": failed_run,
            "forge_runs": forge_runs,
            "skipped_test_cases": skipped_cases,
        }
        if source_creation_fallback is not None:
            record["source_creation_fallback"] = source_creation_fallback
        return record
    kept_calls: list[Dict[str, Any]] = []
    skipped_case_ids = {str(item.get("case_id") or "") for item in skipped_cases}
    for call in calls:
        if str(call.get("case_id") or "") in skipped_case_ids:
            continue
        baseline = (observations.get(call["case_id"]) or {}).get("original") or {}
        if not baseline:
            skipped_cases.append(
                {
                    "case_id": call.get("case_id"),
                    "signature": call.get("signature"),
                    "entry_type": call.get("entry_type"),
                    "state": call.get("state"),
                    "case_kind": call.get("case_kind"),
                    "reason": "missing_foundry_observation",
                }
            )
            continue
        call["baseline"] = baseline
        kept_calls.append(call)
    if not kept_calls:
        record = {
            "benchmark_id": bid,
            "difficulty": row.get("difficulty_bin"),
            "error": "no_executable_test_cases_after_skips",
            "unsupported_entries": unsupported_entries,
            "skipped_test_cases": skipped_cases,
            "forge_runs": forge_runs,
        }
        if source_creation_fallback is not None:
            record["source_creation_fallback"] = source_creation_fallback
        return record
    coverage = (
        run_source_coverage(
            row,
            kept_calls,
            deployment_mode,
            args,
            abi=gt_compile.abi,
            source_matches=source_matches,
        )
        if args.run_coverage
        else None
    )
    record = {
        "benchmark_id": bid,
        "difficulty": row.get("difficulty_bin"),
        "contract_name": row.get("contract_name"),
        "deployment_mode": deployment_mode,
        "constructor_input_count": constructor_inputs,
        "source_deployed_bytecode_matches_input_logic": source_matches,
        "abi_function_count": len(function_abi_map(gt_compile.abi)),
        "abi_executable_entry_count": len(
            [item for item in gt_compile.abi if isinstance(item, dict) and item.get("type") in {"function", "fallback", "receive"}]
        ),
        "functions_with_testcases": len({c["signature"] for c in kept_calls if c.get("entry_type") == "function"}),
        "entries_with_testcases": len({c["signature"] for c in kept_calls}),
        "unsupported_entries": unsupported_entries,
        "calls_generated": len(calls),
        "calls_total": len(kept_calls),
        "skipped_test_cases": skipped_cases,
        "test_cases": group_test_cases(kept_calls),
        "function_signatures": sorted(function_abi_map(gt_compile.abi)),
        "forge": forge_runs[-1] if forge_runs else None,
        "forge_runs": forge_runs,
        "coverage": coverage,
    }
    if source_creation_fallback is not None:
        record["source_creation_fallback"] = source_creation_fallback
    return record


def compact_forge_run(run: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if run is None:
        return None
    return {
        "cmd": run.get("cmd"),
        "exit_code": run.get("exit_code"),
        "latency_sec": run.get("latency_sec"),
        "stdout": truncate(str(run.get("stdout") or ""), 20000),
        "stderr": truncate(str(run.get("stderr") or ""), 8000),
    }


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n...<truncated {len(text) - limit}>..."


def load_testcase_suites(path: Path) -> Dict[str, Dict[str, Any]]:
    suites: Dict[str, Dict[str, Any]] = {}

    def add_item(item: Any) -> None:
        if isinstance(item, dict) and item.get("benchmark_id") and not item.get("error"):
            suites[str(item["benchmark_id"])] = item

    if path.suffix == ".jsonl":
        for item in read_jsonl(path):
            add_item(item)
        return suites

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for item in payload.get("results") or []:
            add_item(item)
    elif isinstance(payload, list):
        for item in payload:
            add_item(item)
    return suites


def replay_record(target: Dict[str, Any], suite: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    bid = str(target.get("benchmark_id") or "")
    matched = set(str(sig) for sig in target.get("matched_signatures") or [])
    include_special_entries = not getattr(args, "matched_functions_only", False)
    calls = [
        dict(c)
        for c in flatten_test_case_groups(suite.get("test_cases") or [])
        if str(c.get("signature") or "") in matched
        or (include_special_entries and c.get("entry_type") in {"fallback", "receive"})
    ]
    if not calls:
        return {"benchmark_id": bid, "difficulty": target.get("difficulty"), "error": "no_replayable_cases"}
    original_init = make_init_code_from_runtime(ensure0x(str(target.get("original_bytecode") or "")))
    predicted_init = ensure0x(str(target.get("compiled_bytecode") or ""))
    observations, forge_runs, failed_run, skipped_cases = run_forge_observations(args, bid, original_init, calls, predicted_init)
    if failed_run is not None:
        return {
            "benchmark_id": bid,
            "difficulty": target.get("difficulty"),
            "error": "forge_test_failed",
            "forge": failed_run,
            "forge_runs": forge_runs,
            "skipped_test_cases": skipped_cases,
        }
    results = []
    by_sig: Dict[str, list[bool]] = {}
    for call in calls:
        expected = call.get("baseline") or {}
        actual = (observations.get(call["case_id"]) or {}).get("predicted") or {}
        ok = observation_equal(expected, actual, str(call.get("state") or ""))
        by_sig.setdefault(str(call.get("signature") or ""), []).append(ok)
        results.append(
            {
                "case_id": call.get("case_id"),
                "signature": call.get("signature"),
                "state": call.get("state"),
                "matched": ok,
                "expected": expected,
                "actual": actual,
            }
        )
    functions = [{"signature": sig, "equivalent": bool(vals) and all(vals), "cases": len(vals)} for sig, vals in sorted(by_sig.items())]
    return {
        "benchmark_id": bid,
        "difficulty": target.get("difficulty"),
        "contract_name": target.get("contract_name"),
        "predicted_contract_name": target.get("predicted_contract_name"),
        "mode": "foundry_replay",
        "test_cases_replayed": len(results),
        "calls_total": len(results),
        "calls_matched": sum(1 for item in results if item["matched"]),
        "functions_total": len(functions),
        "functions_equivalent": sum(1 for item in functions if item["equivalent"]),
        "function_results": functions,
        "call_results": results,
        "skipped_test_cases": skipped_cases,
        "forge": forge_runs[-1] if forge_runs else None,
        "forge_runs": forge_runs,
        "coverage": (
            {
                "kind": "model_replay_coverage",
                "available": False,
                "reason": "coverage_is_generated_for_ground_truth_checkpoint_suites",
            }
            if args.run_coverage
            else None
        ),
    }


def run_generate(args: argparse.Namespace) -> None:
    rows = load_dataset(args)
    existing, completed = load_existing_results(args.output, args.retry_errors)
    rows = [r for r in rows if str(r.get("benchmark_id") or "") not in completed]
    results = list(existing)
    for index, row in enumerate(rows, start=1):
        try:
            record = build_ground_truth_record(row, args)
        except Exception as exc:
            record = {"benchmark_id": row.get("benchmark_id"), "difficulty": row.get("difficulty_bin"), "error": truncate(repr(exc), 2000)}
        results.append(record)
        write_results(args.output, args, results)
        print(f"[foundry-generate] {index}/{len(rows)} {row.get('benchmark_id')} error={record.get('error')}")
    write_results(args.output, args, results)
    print(json.dumps({"results_file": str(args.output)}))


def run_replay(args: argparse.Namespace) -> None:
    targets = load_targets(args.targets, args)
    suites = load_testcase_suites(args.testcases)
    existing, completed = load_existing_results(args.output, args.retry_errors)
    targets = [t for t in targets if str(t.get("benchmark_id") or "") not in completed]
    results = list(existing)
    for index, target in enumerate(targets, start=1):
        suite = suites.get(str(target.get("benchmark_id") or ""))
        if suite is None:
            record = {"benchmark_id": target.get("benchmark_id"), "difficulty": target.get("difficulty"), "error": "missing_testcase_suite"}
        else:
            try:
                record = replay_record(target, suite, args)
            except Exception as exc:
                record = {"benchmark_id": target.get("benchmark_id"), "difficulty": target.get("difficulty"), "error": truncate(repr(exc), 2000)}
        results.append(record)
        write_results(args.output, args, results)
        print(f"[foundry-replay] {index}/{len(targets)} {target.get('benchmark_id')} error={record.get('error')}")
    write_results(args.output, args, results)
    print(json.dumps({"results_file": str(args.output)}))


def load_checkpoint_results(path: Path) -> list[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"{path} is not a checkpoint JSON with a results list")
    return [item for item in payload["results"] if isinstance(item, dict)]


def coverage_is_current(record: Dict[str, Any]) -> bool:
    coverage = record.get("coverage")
    return isinstance(coverage, dict) and coverage.get("instrumentation_schema") == COVERAGE_SCHEMA


def coverage_is_success(record: Dict[str, Any]) -> bool:
    coverage = record.get("coverage")
    return isinstance(coverage, dict) and coverage.get("instrumentation_schema") == COVERAGE_SCHEMA and bool(coverage.get("available"))


def run_coverage_update(args: argparse.Namespace) -> None:
    rows = {str(row.get("benchmark_id") or ""): row for row in load_dataset(args)}
    source_results = load_checkpoint_results(args.output if args.output.exists() else args.checkpoints)
    results: list[Dict[str, Any]] = []
    selected_total = 0
    updated = 0
    save_every = max(1, int(getattr(args, "save_every", 10)))

    for index, record in enumerate(source_results, start=1):
        bid = str(record.get("benchmark_id") or "")
        if bid not in rows or record.get("error"):
            results.append(record)
            continue
        selected_total += 1
        if coverage_is_current(record):
            if coverage_is_success(record) and not getattr(args, "recompute_successes", False):
                results.append(record)
                continue
            if not coverage_is_success(record) and not args.retry_errors:
                results.append(record)
                continue

        row = rows[bid]
        new_record = dict(record)
        try:
            gt_compile = compile_ground_truth(row, args.compile_timeout)
            if not gt_compile.success:
                coverage = {
                    "kind": "foundry_source_lcov",
                    "instrumentation_schema": COVERAGE_SCHEMA,
                    "available": False,
                    "reason": "ground_truth_compile_failed",
                    "compile_error": gt_compile.error,
                }
            else:
                calls = flatten_test_case_groups(record.get("test_cases") or [])
                coverage = run_source_coverage(
                    row,
                    calls,
                    str(record.get("deployment_mode") or ""),
                    args,
                    abi=gt_compile.abi,
                    source_matches=bool(record.get("source_deployed_bytecode_matches_input_logic")),
                )
            new_record["coverage"] = coverage
        except Exception as exc:
            new_record["coverage"] = {
                "kind": "foundry_source_lcov",
                "instrumentation_schema": COVERAGE_SCHEMA,
                "available": False,
                "reason": "coverage_update_exception",
                "error": truncate(repr(exc), 2000),
            }

        results.append(new_record)
        updated += 1
        print(
            "[foundry-coverage] "
            f"{updated}/{selected_total} {bid} available={(new_record.get('coverage') or {}).get('available')} "
            f"reason={(new_record.get('coverage') or {}).get('reason')}"
        )
        if updated % save_every == 0:
            remainder = source_results[index:]
            write_results(args.output, args, results + remainder)

    write_results(args.output, args, results)
    print(json.dumps({"results_file": str(args.output)}))


def main() -> None:
    args = parse_args()
    loaded = load_env_file(Path(".env"))
    if loaded:
        print(f"[env] loaded {loaded} variable(s) from .env")
    if args.fork_url is None:
        args.fork_url = os.environ.get("ARCHIVE_RPC_URL")
    if not args.fork_url:
        raise SystemExit("missing fork URL: set ARCHIVE_RPC_URL in .env or pass --fork-url")
    args.cast_bin = shutil.which(args.cast_bin) or args.cast_bin
    args.forge_bin = shutil.which(args.forge_bin) or args.forge_bin
    if args.command == "generate":
        run_generate(args)
    elif args.command == "replay":
        run_replay(args)
    elif args.command == "coverage":
        run_coverage_update(args)


if __name__ == "__main__":
    main()
