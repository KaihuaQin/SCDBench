#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from deduplicate_contract_dataset import (  # noqa: E402
    ACCESS_CONTROL_FNS,
    AMM_PAIR_SIGS,
    BRIDGE_FNS,
    ERC1155_SIGS,
    ERC20_SIGS,
    ERC4626_SIGS,
    ERC721_SIGS,
    FACTORY_FNS,
    GOVERNANCE_SIGS,
    MARKET_FNS,
    OPCODES,
    ORACLE_FNS,
    PAUSABLE_FNS,
    ROUTER_FNS,
    SALE_FNS,
    STAKING_FNS,
    infer_family_label,
    source_signatures,
    strip_solc_metadata,
)


VERSION_RE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")
CONTRACT_RE = re.compile(r"\b(?:abstract\s+)?contract\s+[A-Za-z_][A-Za-z0-9_]*")
INTERFACE_RE = re.compile(r"\binterface\s+[A-Za-z_][A-Za-z0-9_]*")
LIBRARY_RE = re.compile(r"\blibrary\s+[A-Za-z_][A-Za-z0-9_]*")
FUNCTION_NAME_RE = re.compile(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
EVENT_RE = re.compile(r"\bevent\s+[A-Za-z_][A-Za-z0-9_]*\s*\(")
ERROR_RE = re.compile(r"\berror\s+[A-Za-z_][A-Za-z0-9_]*\s*\(")
MODIFIER_RE = re.compile(r"\bmodifier\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:\(|\{)")
STRUCT_RE = re.compile(r"\bstruct\s+[A-Za-z_][A-Za-z0-9_]*")
ENUM_RE = re.compile(r"\benum\s+[A-Za-z_][A-Za-z0-9_]*")
MAPPING_RE = re.compile(r"\bmapping\s*\(")
ASSEMBLY_RE = re.compile(r"\bassembly\b", re.IGNORECASE)
FALLBACK_RE = re.compile(r"\bfallback\s*\(", re.IGNORECASE)
RECEIVE_RE = re.compile(r"\breceive\s*\(", re.IGNORECASE)
DELEGATECALL_RE = re.compile(r"\bdelegatecall\b", re.IGNORECASE)
STATICCALL_RE = re.compile(r"\bstaticcall\b", re.IGNORECASE)
LOW_LEVEL_CALL_RE = re.compile(r"\.\s*call(?:\s*\{|\.|\s*\()", re.IGNORECASE)
CREATE2_RE = re.compile(r"\bcreate2\b", re.IGNORECASE)
SELFDESTRUCT_RE = re.compile(r"\bselfdestruct\b", re.IGNORECASE)
MINIMAL_PROXY_RE = re.compile(
    r"363d3d373d3d3d363d73[0-9a-f]{40}5af43d82803e903d9160[0-9a-f]{2}57fd5bf3"
)

ARITHMETIC_OPS = set(range(0x01, 0x0C))
COMPARISON_BITWISE_OPS = set(range(0x10, 0x1E))
ENVIRONMENT_OPS = set(range(0x30, 0x49))
MEMORY_OPS = {0x51, 0x52, 0x53, 0x59}
STORAGE_OPS = {0x54, 0x55}
CONTROL_OPS = {0x56, 0x57, 0x5B}
TERMINATOR_OPS = {0x00, 0xF3, 0xFD, 0xFE, 0xFF}
CALL_OPS = {0xF1, 0xF2, 0xF4, 0xFA}
CREATE_OPS = {0xF0, 0xF5}
LOG_OPS = set(range(0xA0, 0xA5))
CALLDATA_OPS = {0x35, 0x36, 0x37}
RETURNDATA_OPS = {0x3D, 0x3E}


class NumericSeries:
    def __init__(self) -> None:
        self.values = array("d")

    def add(self, value: float | int) -> None:
        self.values.append(float(value))

    def summary(self) -> dict:
        if not self.values:
            return {"count": 0}
        vals = sorted(self.values)
        count = len(vals)
        total = math.fsum(vals)
        mean = total / count
        if count > 1:
            variance = math.fsum((value - mean) ** 2 for value in vals) / (count - 1)
            stdev = math.sqrt(max(0.0, variance))
        else:
            stdev = 0.0
        return {
            "count": count,
            "min": vals[0],
            "p01": percentile_sorted(vals, 0.01),
            "p05": percentile_sorted(vals, 0.05),
            "p10": percentile_sorted(vals, 0.10),
            "p25": percentile_sorted(vals, 0.25),
            "median": percentile_sorted(vals, 0.50),
            "p75": percentile_sorted(vals, 0.75),
            "p90": percentile_sorted(vals, 0.90),
            "p95": percentile_sorted(vals, 0.95),
            "p99": percentile_sorted(vals, 0.99),
            "max": vals[-1],
            "mean": mean,
            "stdev": stdev,
        }


def percentile_sorted(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return vals[lower]
    weight = pos - lower
    return vals[lower] * (1.0 - weight) + vals[upper] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect detailed descriptive statistics for the exact-runtime-bytecode-deduped "
            "Solidity contract corpus. The per-contract output is designed to feed later "
            "benchmark sampling steps."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/contracts_simplified_solc_v1.runtime_bytecode_clean.exact_bytecode_deduped_compiler.jsonl"
        ),
    )
    parser.add_argument("--stats-output", type=Path, default=Path("data/scdbench_corpus_stats.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/scdbench_corpus_stats_summary.json"))
    parser.add_argument(
        "--extract-source-signatures",
        action="store_true",
        help=(
            "Extract full source-declaration signatures with argument types. This is slower and "
            "is not needed for descriptive corpus statistics."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50000)
    return parser.parse_args()


def compiler_version(row: dict) -> str:
    compiler = row.get("compiler")
    if not isinstance(compiler, dict):
        return ""
    value = compiler.get("compiler_version")
    return value if isinstance(value, str) else ""


def compiler_minor(version: str) -> str:
    match = VERSION_RE.search(version)
    if not match:
        return "unknown"
    return f"v{int(match.group(1))}.{int(match.group(2))}"


def compiler_patch(version: str) -> str:
    match = VERSION_RE.search(version)
    if not match:
        return "unknown"
    return f"v{int(match.group(1))}.{int(match.group(2))}.{int(match.group(3))}"


def compiler_settings(row: dict) -> tuple[Optional[bool], Optional[int], str, int]:
    compiler = row.get("compiler")
    if not isinstance(compiler, dict):
        return None, None, "", 0
    settings = compiler.get("settings")
    if not isinstance(settings, dict):
        return None, None, "", 0
    optimizer = settings.get("optimizer")
    optimizer_enabled = None
    optimizer_runs = None
    if isinstance(optimizer, dict):
        enabled = optimizer.get("enabled")
        runs = optimizer.get("runs")
        optimizer_enabled = bool(enabled) if isinstance(enabled, bool) else None
        optimizer_runs = int(runs) if isinstance(runs, int) else None
    evm_version = settings.get("evmVersion")
    libraries = settings.get("libraries")
    library_count = 0
    if isinstance(libraries, dict):
        for value in libraries.values():
            if isinstance(value, dict):
                library_count += len(value)
    return optimizer_enabled, optimizer_runs, evm_version if isinstance(evm_version, str) else "", library_count


def log2_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    return f"2^{int(math.floor(math.log2(value)))}"


def function_names(source: str) -> list[str]:
    return FUNCTION_NAME_RE.findall(source)


def signature_function_names(signatures: Iterable[str]) -> set[str]:
    names = set()
    for sig in signatures:
        if not sig.startswith("function:"):
            continue
        body = sig[len("function:") :]
        name, _, _ = body.partition("(")
        if name:
            names.add(name)
    return names


def signature_names(signature_set: set[str]) -> set[str]:
    return {sig.split(":", 1)[1].partition("(")[0] for sig in signature_set if ":" in sig}


ERC20_NAMES = signature_names(ERC20_SIGS)
ERC721_NAMES = signature_names(ERC721_SIGS)
ERC1155_NAMES = signature_names(ERC1155_SIGS)
ERC4626_NAMES = signature_names(ERC4626_SIGS)
AMM_PAIR_NAMES = signature_names(AMM_PAIR_SIGS)
GOVERNANCE_NAMES = signature_names(GOVERNANCE_SIGS)
VAULT_FNS = {"deposit", "withdraw", "mint", "redeem", "asset", "totalAssets"}


def has_any(names: set[str], expected: set[str]) -> bool:
    return bool(names & expected)


def has_at_least(names: set[str], expected: set[str], threshold: int) -> bool:
    return len(names & expected) >= threshold


def infer_family_label_fast(names: set[str], wrapper_type: str, contract_name: str) -> str:
    if wrapper_type and wrapper_type != "none":
        return wrapper_type
    name = contract_name.lower()
    if ERC20_NAMES <= names:
        return "erc20_like"
    if len(ERC721_NAMES & names) >= 5:
        return "erc721_like"
    if len(ERC1155_NAMES & names) >= 5:
        return "erc1155_like"
    if len(ERC4626_NAMES & names) >= 4:
        return "erc4626_like"
    if len(AMM_PAIR_NAMES & names) >= 3:
        return "amm_pair_like"
    if len(GOVERNANCE_NAMES & names) >= 2:
        return "governance_like"
    if has_at_least(names, ACCESS_CONTROL_FNS, 3):
        return "access_control_like"
    if "implementation" in names or "upgradeTo" in names:
        return "upgradeable_proxy"
    if has_at_least(names, STAKING_FNS, 3) or (("staking" in name or "chef" in name) and has_at_least(names, STAKING_FNS, 2)):
        return "staking_like"
    if has_at_least(names, ROUTER_FNS, 2) or ("router" in name and has_any(names, ROUTER_FNS)):
        return "router_like"
    if has_at_least(names, FACTORY_FNS, 2) or (("factory" in name or "deployer" in name) and has_any(names, FACTORY_FNS)):
        return "factory_like"
    if has_at_least(names, BRIDGE_FNS, 2) or (("bridge" in name or "oft" in name) and has_any(names, BRIDGE_FNS)):
        return "bridge_like"
    if has_at_least(names, ORACLE_FNS, 2) or ("oracle" in name and has_any(names, ORACLE_FNS)):
        return "oracle_like"
    if len(VAULT_FNS & names) >= 3 or ("vault" in name and len(VAULT_FNS & names) >= 2):
        return "vault_like"
    if has_at_least(names, SALE_FNS, 2) or (any(token in name for token in ("sale", "presale", "crowdsale")) and has_any(names, SALE_FNS)):
        return "sale_like"
    if has_at_least(names, MARKET_FNS, 2) or (
        any(token in name for token in ("market", "exchange", "orderbook")) and has_any(names, MARKET_FNS)
    ):
        return "market_like"
    if has_at_least(names, PAUSABLE_FNS, 2):
        return "pausable_like"
    if "owner" in names and "transferOwnership" in names:
        return "ownable_misc"
    return "misc"


def source_stats(source: str, names: list[str]) -> dict:
    lower = source.lower()
    lines = source.splitlines()
    nonblank = 0
    max_line_length = 0
    for line in lines:
        if line.strip():
            nonblank += 1
        if len(line) > max_line_length:
            max_line_length = len(line)
    return {
        "source_char_length": len(source),
        "source_line_count": len(lines),
        "source_nonblank_line_count": nonblank,
        "source_max_line_length": max_line_length,
        "contract_decl_count": source.count("contract "),
        "interface_decl_count": source.count("interface "),
        "library_decl_count": source.count("library "),
        "function_decl_count": len(names),
        "event_decl_count": source.count("event "),
        "error_decl_count": source.count("error "),
        "modifier_decl_count": source.count("modifier "),
        "struct_decl_count": source.count("struct "),
        "enum_decl_count": source.count("enum "),
        "mapping_count": source.count("mapping(") + source.count("mapping ("),
        "inheritance_parent_count": source.count(" is "),
        "using_for_count": source.count("using "),
        "require_count": source.count("require(") + source.count("require ("),
        "assert_count": source.count("assert(") + source.count("assert ("),
        "source_revert_count": source.count("revert(") + source.count("revert ("),
        "emit_count": source.count("emit "),
        "try_count": source.count("try "),
        "catch_count": source.count("catch "),
        "inline_assembly_count": lower.count("assembly"),
        "fallback_decl_count": len(FALLBACK_RE.findall(source)),
        "receive_decl_count": len(RECEIVE_RE.findall(source)),
        "source_delegatecall_count": lower.count("delegatecall"),
        "source_staticcall_count": lower.count("staticcall"),
        "source_low_level_call_count": len(LOW_LEVEL_CALL_RE.findall(source)),
        "source_create2_count": lower.count("create2"),
        "source_selfdestruct_count": lower.count("selfdestruct"),
    }


def fast_clean_hex_bytecode(bytecode: str) -> str:
    text = bytecode.strip()
    if text.startswith("0x"):
        text = text[2:]
    if "__" in text or len(text) % 2 == 1:
        text = re.sub(r"__[^_]{1,64}__", "00" * 20, text)
        text = re.sub(r"[^0-9a-fA-F]", "", text)
        if len(text) % 2 == 1:
            text = text[:-1]
    return text.lower()


def opcode_name(opcode: int) -> str:
    if opcode == 0x5F:
        return "PUSH0"
    if 0x60 <= opcode <= 0x7F:
        return f"PUSH{opcode - 0x5F}"
    if 0x80 <= opcode <= 0x8F:
        return f"DUP{opcode - 0x7F}"
    if 0x90 <= opcode <= 0x9F:
        return f"SWAP{opcode - 0x8F}"
    return OPCODES.get(opcode, f"OP_{opcode:02x}")


def wrapper_type_fast(source: str, raw_hex: str) -> str:
    lower = source.lower()
    if MINIMAL_PROXY_RE.search(raw_hex):
        return "minimal_proxy"
    if "delegatecall" in lower and "fallback" in lower:
        return "proxy"
    if "fallback" in lower and ".call" in lower:
        return "forwarder"
    return "none"


def scan_bytecode(
    bytecode: str,
    global_opcode_counts: list[int],
    *,
    strip_metadata: bool = True,
) -> dict:
    raw_hex = fast_clean_hex_bytecode(bytecode)
    logic_hex = strip_solc_metadata(raw_hex) if strip_metadata else raw_hex
    try:
        code = bytes.fromhex(logic_hex)
    except ValueError:
        code = b""
    unique_mask = 0
    opcode_count = 0
    push_count = 0
    push_data_bytes = 0
    unknown_opcode_count = 0
    jumpdest_count = 0
    jump_count = 0
    jumpi_count = 0
    terminal_count = 0
    control_flow_op_count = 0
    arithmetic_op_count = 0
    comparison_bitwise_op_count = 0
    environment_op_count = 0
    memory_op_count = 0
    storage_op_count = 0
    sload_count = 0
    sstore_count = 0
    calldata_op_count = 0
    returndata_op_count = 0
    external_call_count = 0
    call_count = 0
    callcode_count = 0
    delegatecall_count = 0
    staticcall_count = 0
    create_op_count = 0
    create_count = 0
    create2_count = 0
    log_op_count = 0
    sha3_count = 0
    return_count = 0
    bytecode_revert_count = 0
    invalid_count = 0
    selfdestruct_count = 0
    i = 0
    n = len(code)
    while i < n:
        opcode = code[i]
        i += 1
        opcode_count += 1
        global_opcode_counts[opcode] += 1
        unique_mask |= 1 << opcode
        if opcode == 0x5F:
            push_count += 1
            continue
        if 0x60 <= opcode <= 0x7F:
            push_len = opcode - 0x5F
            push_count += 1
            push_data_bytes += push_len
            i += push_len
            continue
        if opcode not in OPCODES and not (0x80 <= opcode <= 0x9F):
            unknown_opcode_count += 1
        if opcode == 0x5B:
            jumpdest_count += 1
            control_flow_op_count += 1
        elif opcode == 0x56:
            jump_count += 1
            control_flow_op_count += 1
        elif opcode == 0x57:
            jumpi_count += 1
            control_flow_op_count += 1
        elif opcode in TERMINATOR_OPS:
            terminal_count += 1
            if opcode == 0xF3:
                return_count += 1
            elif opcode == 0xFD:
                bytecode_revert_count += 1
            elif opcode == 0xFE:
                invalid_count += 1
            elif opcode == 0xFF:
                selfdestruct_count += 1
        if opcode in ARITHMETIC_OPS:
            arithmetic_op_count += 1
        elif opcode in COMPARISON_BITWISE_OPS:
            comparison_bitwise_op_count += 1
        elif opcode in ENVIRONMENT_OPS:
            environment_op_count += 1
        elif opcode in MEMORY_OPS:
            memory_op_count += 1
        elif opcode == 0x54:
            storage_op_count += 1
            sload_count += 1
        elif opcode == 0x55:
            storage_op_count += 1
            sstore_count += 1
        if opcode in CALLDATA_OPS:
            calldata_op_count += 1
        elif opcode in RETURNDATA_OPS:
            returndata_op_count += 1
        elif opcode == 0xF1:
            external_call_count += 1
            call_count += 1
        elif opcode == 0xF2:
            external_call_count += 1
            callcode_count += 1
        elif opcode == 0xF4:
            external_call_count += 1
            delegatecall_count += 1
        elif opcode == 0xFA:
            external_call_count += 1
            staticcall_count += 1
        elif opcode == 0xF0:
            create_op_count += 1
            create_count += 1
        elif opcode == 0xF5:
            create_op_count += 1
            create2_count += 1
        elif opcode in LOG_OPS:
            log_op_count += 1
        elif opcode == 0x20:
            sha3_count += 1

    block_count_proxy = jumpdest_count + 1 if opcode_count else 0
    cyclomatic_proxy = jumpi_count + 1 if opcode_count else 0

    return {
        "bytecode_raw_byte_length": len(raw_hex) // 2,
        "bytecode_logic_byte_length": len(logic_hex) // 2,
        "metadata_byte_length": max(0, (len(raw_hex) - len(logic_hex)) // 2),
        "opcode_count": opcode_count,
        "unique_opcode_count": unique_mask.bit_count(),
        "unknown_opcode_count": unknown_opcode_count,
        "push_count": push_count,
        "push_data_bytes": push_data_bytes,
        "jumpdest_count": jumpdest_count,
        "jump_count": jump_count,
        "jumpi_count": jumpi_count,
        "terminal_count": terminal_count,
        "block_count_proxy": block_count_proxy,
        "cyclomatic_complexity_proxy": cyclomatic_proxy,
        "control_flow_op_count": control_flow_op_count,
        "arithmetic_op_count": arithmetic_op_count,
        "comparison_bitwise_op_count": comparison_bitwise_op_count,
        "environment_op_count": environment_op_count,
        "memory_op_count": memory_op_count,
        "storage_op_count": storage_op_count,
        "sload_count": sload_count,
        "sstore_count": sstore_count,
        "calldata_op_count": calldata_op_count,
        "returndata_op_count": returndata_op_count,
        "external_call_count": external_call_count,
        "call_count": call_count,
        "callcode_count": callcode_count,
        "delegatecall_count": delegatecall_count,
        "staticcall_count": staticcall_count,
        "create_op_count": create_op_count,
        "create_count": create_count,
        "create2_count": create2_count,
        "log_op_count": log_op_count,
        "sha3_count": sha3_count,
        "return_count": return_count,
        "bytecode_revert_count": bytecode_revert_count,
        "invalid_count": invalid_count,
        "selfdestruct_count": selfdestruct_count,
    }


def low_level_flags(source_info: dict, bytecode_info: dict) -> list[str]:
    checks = [
        ("inline_assembly", source_info["inline_assembly_count"] > 0),
        ("fallback", source_info["fallback_decl_count"] > 0),
        ("receive", source_info["receive_decl_count"] > 0),
        ("delegatecall", bytecode_info["delegatecall_count"] > 0 or source_info["source_delegatecall_count"] > 0),
        ("call", bytecode_info["call_count"] > 0 or source_info["source_low_level_call_count"] > 0),
        ("staticcall", bytecode_info["staticcall_count"] > 0 or source_info["source_staticcall_count"] > 0),
        ("create", bytecode_info["create_count"] > 0),
        ("create2", bytecode_info["create2_count"] > 0 or source_info["source_create2_count"] > 0),
        ("selfdestruct", bytecode_info["selfdestruct_count"] > 0 or source_info["source_selfdestruct_count"] > 0),
    ]
    return [name for name, enabled in checks if enabled]


def valid_candidate(row: dict, version: str, bytecode_info: dict) -> bool:
    return (
        isinstance(row.get("bytecode"), str)
        and bool(row.get("bytecode"))
        and isinstance(row.get("source_code"), str)
        and bool(row.get("source_code"))
        and isinstance(row.get("contract_name"), str)
        and bool(row.get("contract_name"))
        and bool(version)
        and compiler_minor(version) != "unknown"
        and isinstance(row.get("compiler"), dict)
        and isinstance(row["compiler"].get("settings"), dict)
        and bytecode_info["opcode_count"] > 0
    )


def add_numeric(numeric: dict[str, NumericSeries], stats: dict, keys: Iterable[str]) -> None:
    for key in keys:
        numeric[key].add(stats.get(key, 0))


def group_add(grouped: dict[str, dict[str, float]], key: str, row: dict) -> None:
    entry = grouped[key]
    entry["count"] += 1
    entry["bytecode_logic_byte_length_sum"] += row["bytecode_logic_byte_length"]
    entry["source_char_length_sum"] += row["source_char_length"]
    entry["function_decl_count_sum"] += row["function_decl_count"]
    entry["cyclomatic_complexity_proxy_sum"] += row["cyclomatic_complexity_proxy"]
    entry["storage_op_count_sum"] += row["storage_op_count"]
    entry["external_call_count_sum"] += row["external_call_count"]


def finalize_grouped(grouped: dict[str, dict[str, float]]) -> dict:
    result = {}
    for key, values in sorted(grouped.items()):
        count = values["count"]
        result[key] = {"count": int(count)}
        for metric, value in values.items():
            if metric == "count" or not metric.endswith("_sum"):
                continue
            result[key][metric.removesuffix("_sum") + "_mean"] = value / count if count else 0.0
    return result


def pearson(xs: array, ys: array) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    mean_x = math.fsum(xs) / n
    mean_y = math.fsum(ys) / n
    cov = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = math.fsum((x - mean_x) ** 2 for x in xs)
    var_y = math.fsum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    return cov / denom if denom else None


def main() -> int:
    args = parse_args()
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    rows_seen = 0
    valid_rows = 0
    compiler_minor_counts: Counter[str] = Counter()
    compiler_patch_counts: Counter[str] = Counter()
    optimizer_counts: Counter[str] = Counter()
    optimizer_runs_counts: Counter[str] = Counter()
    evm_version_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    wrapper_counts: Counter[str] = Counter()
    low_level_flag_counts: Counter[str] = Counter()
    bytecode_length_buckets: Counter[str] = Counter()
    source_length_buckets: Counter[str] = Counter()
    signature_count_buckets: Counter[str] = Counter()
    opcode_counts = [0] * 256
    invalid_reasons: Counter[str] = Counter()
    grouped_by_compiler_minor: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    grouped_by_family: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    numeric_keys = [
        "bytecode_raw_byte_length",
        "bytecode_logic_byte_length",
        "metadata_byte_length",
        "opcode_count",
        "unique_opcode_count",
        "unknown_opcode_count",
        "push_count",
        "push_data_bytes",
        "source_char_length",
        "source_line_count",
        "source_nonblank_line_count",
        "source_max_line_length",
        "contract_decl_count",
        "interface_decl_count",
        "library_decl_count",
        "function_decl_count",
        "event_decl_count",
        "error_decl_count",
        "modifier_decl_count",
        "struct_decl_count",
        "enum_decl_count",
        "mapping_count",
        "inheritance_parent_count",
        "using_for_count",
        "signature_count",
        "jumpdest_count",
        "jump_count",
        "jumpi_count",
        "terminal_count",
        "block_count_proxy",
        "cyclomatic_complexity_proxy",
        "control_flow_op_count",
        "arithmetic_op_count",
        "comparison_bitwise_op_count",
        "environment_op_count",
        "memory_op_count",
        "storage_op_count",
        "sload_count",
        "sstore_count",
        "calldata_op_count",
        "returndata_op_count",
        "external_call_count",
        "call_count",
        "callcode_count",
        "delegatecall_count",
        "staticcall_count",
        "create_op_count",
        "create_count",
        "create2_count",
        "log_op_count",
        "sha3_count",
        "return_count",
        "bytecode_revert_count",
        "invalid_count",
        "selfdestruct_count",
        "require_count",
        "assert_count",
        "source_revert_count",
        "emit_count",
        "inline_assembly_count",
        "fallback_decl_count",
        "receive_decl_count",
        "source_delegatecall_count",
        "source_staticcall_count",
        "source_low_level_call_count",
        "source_create2_count",
        "source_selfdestruct_count",
    ]
    numeric = {key: NumericSeries() for key in numeric_keys}

    with args.input.open("r", encoding="utf-8", errors="replace") as in_f, args.stats_output.open(
        "w", encoding="utf-8"
    ) as out_f:
        for line_no, raw in enumerate(in_f, start=1):
            if args.limit is not None and rows_seen >= args.limit:
                break
            rows_seen += 1
            row = json.loads(raw)
            source = row.get("source_code") if isinstance(row.get("source_code"), str) else ""
            bytecode = row.get("bytecode") if isinstance(row.get("bytecode"), str) else ""
            version = compiler_version(row)
            opt_enabled, opt_runs, evm_version, library_count = compiler_settings(row)
            bytecode_info = scan_bytecode(bytecode, opcode_counts)
            names = function_names(source) if source else []
            source_info = source_stats(source, names)
            signatures = source_signatures(source) if args.extract_source_signatures and source else frozenset()
            wrapper = wrapper_type_fast(source, fast_clean_hex_bytecode(bytecode)) if bytecode else "none"
            if args.extract_source_signatures:
                family = infer_family_label(signatures, None if wrapper == "none" else wrapper, str(row.get("contract_name") or ""))
                signature_count = len(signatures)
                signature_source = "source_declarations_with_arg_types"
            else:
                family = infer_family_label_fast(set(names), wrapper, str(row.get("contract_name") or ""))
                signature_count = source_info["function_decl_count"] + source_info["event_decl_count"] + source_info["error_decl_count"]
                signature_source = "source_declaration_counts"
            flags = low_level_flags(source_info, bytecode_info)
            is_valid = valid_candidate(row, version, bytecode_info)

            if not is_valid:
                if not bytecode:
                    invalid_reasons["missing_bytecode"] += 1
                if not source:
                    invalid_reasons["missing_source"] += 1
                if not row.get("contract_name"):
                    invalid_reasons["missing_contract_name"] += 1
                if not version:
                    invalid_reasons["missing_compiler_version"] += 1
                elif compiler_minor(version) == "unknown":
                    invalid_reasons["unparseable_compiler_version"] += 1
                if not isinstance(row.get("compiler"), dict) or not isinstance(row["compiler"].get("settings"), dict):
                    invalid_reasons["missing_compiler_settings"] += 1
                if bytecode_info["opcode_count"] == 0:
                    invalid_reasons["empty_normalized_bytecode"] += 1

            row_stats = {
                "line_no": line_no,
                "candidate_valid": is_valid,
                "contract_name": row.get("contract_name", ""),
                "compiler_version": version,
                "compiler_minor": compiler_minor(version),
                "compiler_patch": compiler_patch(version),
                "optimizer_enabled": opt_enabled,
                "optimizer_runs": opt_runs,
                "evm_version": evm_version,
                "library_count": library_count,
                "wrapper_type": wrapper,
                "family_label": family,
                "signature_source": signature_source,
                "signature_count": signature_count,
                "low_level_flags": flags,
                **source_info,
                **bytecode_info,
            }
            out_f.write(json.dumps(row_stats, ensure_ascii=False) + "\n")

            if is_valid:
                valid_rows += 1
                compiler_minor_counts[row_stats["compiler_minor"]] += 1
                compiler_patch_counts[row_stats["compiler_patch"]] += 1
                optimizer_counts["unknown" if opt_enabled is None else str(opt_enabled).lower()] += 1
                optimizer_runs_counts["unknown" if opt_runs is None else str(opt_runs)] += 1
                evm_version_counts[evm_version or "unknown"] += 1
                family_counts[family] += 1
                wrapper_counts[wrapper] += 1
                bytecode_length_buckets[log2_bucket(row_stats["bytecode_logic_byte_length"])] += 1
                source_length_buckets[log2_bucket(row_stats["source_char_length"])] += 1
                signature_count_buckets[log2_bucket(row_stats["signature_count"])] += 1
                for flag in flags:
                    low_level_flag_counts[flag] += 1
                add_numeric(numeric, row_stats, numeric_keys)
                group_add(grouped_by_compiler_minor, row_stats["compiler_minor"], row_stats)
                group_add(grouped_by_family, family, row_stats)

            if args.log_every > 0 and rows_seen % args.log_every == 0:
                print(f"[stats] rows_seen={rows_seen} valid={valid_rows}", flush=True)

    correlations = {
        "bytecode_logic_byte_length_vs_source_char_length": pearson(
            numeric["bytecode_logic_byte_length"].values, numeric["source_char_length"].values
        ),
        "bytecode_logic_byte_length_vs_function_decl_count": pearson(
            numeric["bytecode_logic_byte_length"].values, numeric["function_decl_count"].values
        ),
        "bytecode_logic_byte_length_vs_cyclomatic_complexity_proxy": pearson(
            numeric["bytecode_logic_byte_length"].values, numeric["cyclomatic_complexity_proxy"].values
        ),
        "source_char_length_vs_function_decl_count": pearson(
            numeric["source_char_length"].values, numeric["function_decl_count"].values
        ),
        "function_decl_count_vs_cyclomatic_complexity_proxy": pearson(
            numeric["function_decl_count"].values, numeric["cyclomatic_complexity_proxy"].values
        ),
    }

    summary = {
        "input": str(args.input),
        "stats_output": str(args.stats_output),
        "rows_seen": rows_seen,
        "candidate_valid_rows": valid_rows,
        "candidate_invalid_rows": rows_seen - valid_rows,
        "invalid_reasons": dict(invalid_reasons.most_common()),
        "numeric_summaries_valid": {key: series.summary() for key, series in numeric.items()},
        "compiler_minor_counts_valid": dict(compiler_minor_counts.most_common()),
        "compiler_patch_counts_valid_top50": dict(compiler_patch_counts.most_common(50)),
        "optimizer_enabled_counts_valid": dict(optimizer_counts.most_common()),
        "optimizer_runs_counts_valid_top50": dict(optimizer_runs_counts.most_common(50)),
        "evm_version_counts_valid": dict(evm_version_counts.most_common()),
        "family_label_counts_valid": dict(family_counts.most_common()),
        "wrapper_type_counts_valid": dict(wrapper_counts.most_common()),
        "low_level_flag_counts_valid": dict(low_level_flag_counts.most_common()),
        "bytecode_logic_byte_length_log2_buckets_valid": dict(sorted(bytecode_length_buckets.items())),
        "source_char_length_log2_buckets_valid": dict(sorted(source_length_buckets.items())),
        "signature_count_log2_buckets_valid": dict(sorted(signature_count_buckets.items())),
        "opcode_counts_top80_all_rows": {
            opcode_name(opcode): count
            for opcode, count in sorted(enumerate(opcode_counts), key=lambda item: item[1], reverse=True)[:80]
            if count
        },
        "grouped_by_compiler_minor_valid": finalize_grouped(grouped_by_compiler_minor),
        "grouped_by_family_label_valid": finalize_grouped(grouped_by_family),
        "pearson_correlations_valid": correlations,
        "cfg_complexity_note": (
            "CFG metrics are bytecode-level proxies. block_count_proxy is JUMPDEST+1; "
            "cyclomatic_complexity_proxy is JUMPI+1. They are intended for stratification "
            "and corpus characterization, not as a fully resolved EVM CFG."
        ),
        "signature_note": "Signatures are extracted from source declarations, not ABI JSON.",
    }
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[write] stats -> {args.stats_output}", flush=True)
    print(f"[write] summary -> {args.summary_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
