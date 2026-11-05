#!/usr/bin/env python3
"""Compare SCDBench with an exact-million deployed-contract table sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import struct
import sys
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent

from collect_corpus_statistics import (  # noqa: E402
    opcode_name,
    scan_bytecode,
    strip_solc_metadata,
)


DEFAULT_BASE_DB = EXPERIMENT_DIR / "raw/deployed_sample.sqlite3"
DEFAULT_EXTENSION_DB = EXPERIMENT_DIR / "raw/deployed_sample_extension.sqlite3"
DEFAULT_BENCHMARK = EXPERIMENT_DIR / "raw/scdbench_sample.jsonl"
DEFAULT_VERIFIED_SUMMARY = EXPERIMENT_DIR / "raw/scdbench_corpus_stats_summary.json"
DEFAULT_VERIFIED_STATS = EXPERIMENT_DIR / "raw/scdbench_corpus_stats.jsonl"
DEFAULT_VERIFIED_EIP1167 = EXPERIMENT_DIR / "raw/full_verified_eip1167.json"
DEFAULT_VERIFIED_RESULTS = EXPERIMENT_DIR / "raw/verified_results.json"
TARGET_SIZE = 1_000_000

FEATURES = (
    "bytecode_raw_byte_length",
    "bytecode_logic_byte_length",
    "metadata_byte_length",
    "opcode_count",
    "cyclomatic_complexity_proxy",
    "control_flow_op_count",
    "storage_op_count",
    "external_call_count",
    "delegatecall_count",
    "create_op_count",
    "log_op_count",
    "control_flow_per_1k_opcodes",
    "storage_per_1k_opcodes",
    "external_calls_per_1k_opcodes",
)

FEATURE_LABELS = {
    "bytecode_raw_byte_length": "Raw runtime bytecode (bytes)",
    "bytecode_logic_byte_length": "Metadata-stripped bytecode (bytes)",
    "metadata_byte_length": "Validated metadata trailer (bytes)",
    "opcode_count": "Opcode count",
    "cyclomatic_complexity_proxy": "Cyclomatic proxy (JUMPI + 1)",
    "control_flow_op_count": "Control-flow opcodes",
    "storage_op_count": "Storage opcodes",
    "external_call_count": "External-call opcodes",
    "delegatecall_count": "DELEGATECALL opcodes",
    "create_op_count": "CREATE/CREATE2 opcodes",
    "log_op_count": "LOG opcodes",
    "control_flow_per_1k_opcodes": "Control-flow opcodes / 1k",
    "storage_per_1k_opcodes": "Storage opcodes / 1k",
    "external_calls_per_1k_opcodes": "External calls / 1k",
}

REPORT_FEATURES = (
    "bytecode_raw_byte_length",
    "bytecode_logic_byte_length",
    "opcode_count",
    "cyclomatic_complexity_proxy",
    "storage_op_count",
    "external_call_count",
)

LOW_LEVEL_FLAGS = (
    ("sload", "Contains SLOAD"),
    ("sstore", "Contains SSTORE"),
    ("call", "Contains CALL"),
    ("callcode", "Contains CALLCODE"),
    ("delegatecall", "Contains DELEGATECALL"),
    ("staticcall", "Contains STATICCALL"),
    ("create", "Contains CREATE"),
    ("create2", "Contains CREATE2"),
    ("log", "Contains LOG"),
    ("selfdestruct", "Contains SELFDESTRUCT"),
    ("tload", "Contains TLOAD"),
    ("tstore", "Contains TSTORE"),
    ("mcopy", "Contains MCOPY"),
)

VERIFIED_LOW_LEVEL_COUNT_FIELDS = {
    "sload": "sload_count",
    "sstore": "sstore_count",
    "call": "call_count",
    "callcode": "callcode_count",
    "delegatecall": "delegatecall_count",
    "staticcall": "staticcall_count",
    "create": "create_count",
    "create2": "create2_count",
    "log": "log_op_count",
    "selfdestruct": "selfdestruct_count",
}

MODERN_OPCODE_NAMES = {
    0x49: "BLOBHASH",
    0x4A: "BLOBBASEFEE",
    0x5C: "TLOAD",
    0x5D: "TSTORE",
    0x5E: "MCOPY",
}

EIP1967_IMPLEMENTATION_SLOT = bytes.fromhex(
    "360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
)
EIP1967_BEACON_SLOT = bytes.fromhex(
    "a3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50"
)
EIP1167_RUNTIME_RE = re.compile(
    rb"^"
    rb"\x36\x3d\x3d\x37\x3d\x3d\x3d\x36\x3d\x73"
    rb".{20}"
    rb"\x5a\xf4\x3d\x82\x80\x3e\x90\x3d\x91\x60\x2b\x57\xfd\x5b\xf3"
    rb"$",
    re.DOTALL,
)
VERSION_RE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")
RECOGNIZED_METADATA_KEYS = {
    "ipfs",
    "bzzr0",
    "bzzr1",
    "solc",
    "vyper",
    "experimental",
}


@dataclass(frozen=True)
class SelectedContract:
    source: str
    sample_index: int
    address: bytes
    bytecode: bytes
    code_keccak: bytes


@dataclass
class Program:
    digest: bytes
    bytecode: bytes
    multiplicity: int = 0
    features: dict[str, float] = field(default_factory=dict)
    opcode_counts: list[int] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    compiler_engine: str = "unknown"
    compiler_version: str = "unknown"
    compiler_minor: str = "unknown"
    metadata_valid: bool = False
    scanner_metadata_mismatch: bool = False


@dataclass
class PopulationView:
    key: str
    total: int = 0
    numeric: dict[str, list[tuple[float, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    flag_counts: Counter[str] = field(default_factory=Counter)
    feature_sums: Counter[str] = field(default_factory=Counter)
    opcode_counts: list[int] = field(default_factory=lambda: [0] * 256)
    compiler_counts: Counter[str] = field(default_factory=Counter)
    metadata_valid_count: int = 0
    compiler_detected_count: int = 0
    scanner_metadata_mismatch_count: int = 0

    def add(self, program: Program, weight: int) -> None:
        if weight <= 0:
            raise ValueError("population weights must be positive")
        self.total += weight
        for feature in FEATURES:
            value = float(program.features[feature])
            self.numeric[feature].append((value, weight))
            self.feature_sums[feature] += value * weight
        for flag in program.flags:
            self.flag_counts[flag] += weight
        for opcode, count in enumerate(program.opcode_counts):
            self.opcode_counts[opcode] += count * weight
        self.compiler_counts[program.compiler_minor] += weight
        if program.metadata_valid:
            self.metadata_valid_count += weight
        if program.compiler_version != "unknown":
            self.compiler_detected_count += weight
        if program.scanner_metadata_mismatch:
            self.scanner_metadata_mismatch_count += weight


class CBORDecodeError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-db", type=Path, default=DEFAULT_BASE_DB)
    parser.add_argument(
        "--extension-db",
        type=Path,
        default=DEFAULT_EXTENSION_DB,
    )
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--verified-summary",
        type=Path,
        default=DEFAULT_VERIFIED_SUMMARY,
    )
    parser.add_argument(
        "--verified-stats",
        type=Path,
        default=DEFAULT_VERIFIED_STATS,
    )
    parser.add_argument(
        "--verified-eip1167",
        type=Path,
        default=DEFAULT_VERIFIED_EIP1167,
    )
    parser.add_argument(
        "--verified-results",
        type=Path,
        default=DEFAULT_VERIFIED_RESULTS,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=EXPERIMENT_DIR / "results.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=EXPERIMENT_DIR / "report.md",
    )
    parser.add_argument("--log-every", type=int, default=5000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def collector_manifest_provenance(database_path: Path) -> dict[str, Any]:
    manifest_path = database_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"collector manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sampling = manifest.get("sampling") or {}
    return {
        "name": manifest_path.name,
        "sha256": sha256_file(manifest_path),
        "schema": manifest.get("schema"),
        "sampling": sampling,
    }


def select_exact_nonempty(
    base_db: Path,
    extension_db: Path,
    *,
    target_size: int,
) -> Iterator[SelectedContract]:
    if target_size < 1:
        raise ValueError("target_size must be positive")
    seen: set[bytes] = set()
    selected = 0
    for source, path in (("base", base_db), ("extension", extension_db)):
        connection = sqlite3.connect(
            f"file:{path.resolve()}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """
                SELECT sample_index, address, bytecode, code_keccak
                FROM contracts
                WHERE rpc_status = 'ok'
                ORDER BY sample_index
                """
            )
            for row in rows:
                if selected >= target_size:
                    break
                address = bytes(row["address"])
                if address in seen:
                    raise ValueError(
                        f"address overlap in exact selection: 0x{address.hex()}"
                    )
                bytecode = bytes(row["bytecode"] or b"")
                digest = bytes(row["code_keccak"] or b"")
                if not bytecode or not digest:
                    raise ValueError(
                        f"{source}:{row['sample_index']} has incomplete bytecode"
                    )
                seen.add(address)
                selected += 1
                yield SelectedContract(
                    source=source,
                    sample_index=int(row["sample_index"]),
                    address=address,
                    bytecode=bytecode,
                    code_keccak=digest,
                )
        finally:
            connection.close()
        if selected >= target_size:
            break
    if selected != target_size:
        raise ValueError(
            f"only {selected} non-empty unique rows are available, "
            f"expected {target_size}"
        )


def _merged_weighted(
    pairs: Iterable[tuple[float, int]],
) -> list[tuple[float, int]]:
    counts: Counter[float] = Counter()
    for value, weight in pairs:
        if weight < 0:
            raise ValueError("weights must be non-negative")
        if weight:
            counts[float(value)] += int(weight)
    if not counts:
        raise ValueError("weighted statistic requires positive total weight")
    return sorted(counts.items())


def _weighted_value_at_rank(
    merged: Sequence[tuple[float, int]],
    rank: int,
) -> float:
    if rank < 0:
        raise ValueError("rank must be non-negative")
    cumulative = 0
    for value, weight in merged:
        cumulative += weight
        if rank < cumulative:
            return float(value)
    raise ValueError("rank exceeds total weight")


def weighted_quantile(
    pairs: Iterable[tuple[float, int]],
    q: float,
) -> float:
    if q < 0 or q > 1:
        raise ValueError("q must be between zero and one")
    merged = _merged_weighted(pairs)
    total = sum(weight for _, weight in merged)
    position = q * (total - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    lower_value = _weighted_value_at_rank(merged, lower)
    if lower == upper:
        return lower_value
    upper_value = _weighted_value_at_rank(merged, upper)
    fraction = position - lower
    return lower_value * (1.0 - fraction) + upper_value * fraction


def weighted_summary(
    pairs: Iterable[tuple[float, int]],
) -> dict[str, float | int]:
    merged = _merged_weighted(pairs)
    total = sum(weight for _, weight in merged)
    mean = math.fsum(value * weight for value, weight in merged) / total
    variance = (
        math.fsum(weight * (value - mean) ** 2 for value, weight in merged)
        / max(1, total - 1)
    )
    return {
        "n": total,
        "mean": mean,
        "stdev": math.sqrt(max(0.0, variance)),
        "p10": weighted_quantile(merged, 0.10),
        "p25": weighted_quantile(merged, 0.25),
        "median": weighted_quantile(merged, 0.50),
        "p75": weighted_quantile(merged, 0.75),
        "p90": weighted_quantile(merged, 0.90),
        "p95": weighted_quantile(merged, 0.95),
        "p99": weighted_quantile(merged, 0.99),
    }


def weighted_ks_distance(
    reference: Iterable[tuple[float, int]],
    comparison: Iterable[tuple[float, int]],
) -> float:
    ref = _merged_weighted(reference)
    comp = _merged_weighted(comparison)
    ref_total = sum(weight for _, weight in ref)
    comp_total = sum(weight for _, weight in comp)
    i = 0
    j = 0
    ref_cumulative = 0
    comp_cumulative = 0
    distance = 0.0
    while i < len(ref) or j < len(comp):
        if j >= len(comp) or (
            i < len(ref) and ref[i][0] <= comp[j][0]
        ):
            value = ref[i][0]
        else:
            value = comp[j][0]
        while i < len(ref) and ref[i][0] <= value:
            ref_cumulative += ref[i][1]
            i += 1
        while j < len(comp) and comp[j][0] <= value:
            comp_cumulative += comp[j][1]
            j += 1
        distance = max(
            distance,
            abs(
                ref_cumulative / ref_total
                - comp_cumulative / comp_total
            ),
        )
    return distance


def _weighted_moments(
    pairs: Iterable[tuple[float, int]],
    *,
    transform=None,
) -> tuple[float, float]:
    merged = _merged_weighted(pairs)
    total = sum(weight for _, weight in merged)
    converted = [
        (transform(value) if transform else value, weight)
        for value, weight in merged
    ]
    mean = math.fsum(value * weight for value, weight in converted) / total
    variance = (
        math.fsum(weight * (value - mean) ** 2 for value, weight in converted)
        / max(1, total - 1)
    )
    return mean, variance


def log1p_standardized_mean_difference(
    reference: Iterable[tuple[float, int]],
    comparison: Iterable[tuple[float, int]],
) -> float:
    ref_mean, ref_variance = _weighted_moments(
        reference,
        transform=math.log1p,
    )
    comp_mean, comp_variance = _weighted_moments(
        comparison,
        transform=math.log1p,
    )
    pooled = math.sqrt(max(0.0, (ref_variance + comp_variance) / 2.0))
    return 0.0 if pooled == 0 else (comp_mean - ref_mean) / pooled


def _decode_cbor_length(
    data: bytes,
    position: int,
    additional: int,
) -> tuple[int, int]:
    if additional < 24:
        return additional, position
    byte_lengths = {24: 1, 25: 2, 26: 4, 27: 8}
    length = byte_lengths.get(additional)
    if length is None or position + length > len(data):
        raise CBORDecodeError("unsupported or truncated CBOR length")
    value = int.from_bytes(data[position : position + length], "big")
    return value, position + length


def _decode_cbor(
    data: bytes,
    position: int = 0,
    *,
    depth: int = 0,
) -> tuple[Any, int]:
    if depth > 16 or position >= len(data):
        raise CBORDecodeError("invalid CBOR nesting or truncation")
    initial = data[position]
    position += 1
    major = initial >> 5
    additional = initial & 0x1F
    if additional == 31:
        raise CBORDecodeError("indefinite-length CBOR is unsupported")
    value, position = _decode_cbor_length(data, position, additional)
    if major == 0:
        return value, position
    if major == 1:
        return -1 - value, position
    if major in {2, 3}:
        end = position + value
        if end > len(data):
            raise CBORDecodeError("truncated CBOR string")
        raw = data[position:end]
        if major == 2:
            return raw, end
        try:
            return raw.decode("utf-8"), end
        except UnicodeDecodeError as exc:
            raise CBORDecodeError("invalid CBOR UTF-8") from exc
    if major == 4:
        items = []
        for _ in range(value):
            item, position = _decode_cbor(
                data,
                position,
                depth=depth + 1,
            )
            items.append(item)
        return items, position
    if major == 5:
        result: dict[Any, Any] = {}
        for _ in range(value):
            key, position = _decode_cbor(
                data,
                position,
                depth=depth + 1,
            )
            item, position = _decode_cbor(
                data,
                position,
                depth=depth + 1,
            )
            try:
                if key in result:
                    raise CBORDecodeError("duplicate CBOR map key")
                result[key] = item
            except TypeError as exc:
                raise CBORDecodeError("unhashable CBOR map key") from exc
        return result, position
    if major == 6:
        tagged, position = _decode_cbor(
            data,
            position,
            depth=depth + 1,
        )
        return ("tag", value, tagged), position
    if major == 7:
        if additional == 20:
            return False, position
        if additional == 21:
            return True, position
        if additional in {22, 23}:
            return None, position
        float_lengths = {25: (2, ">e"), 26: (4, ">f"), 27: (8, ">d")}
        descriptor = float_lengths.get(additional)
        if descriptor is None:
            return value, position
        length, format_string = descriptor
        start = position - length
        if start < 0:
            raise CBORDecodeError("invalid CBOR float")
        return struct.unpack(format_string, data[start:position])[0], position
    raise CBORDecodeError("unsupported CBOR major type")


def _version_from_value(engine: str, value: Any) -> str:
    if engine == "solc" and isinstance(value, bytes) and len(value) == 3:
        return f"v{value[0]}.{value[1]}.{value[2]}"
    if engine == "solc" and isinstance(value, str):
        match = VERSION_RE.search(value)
        if match:
            return (
                f"v{int(match.group(1))}."
                f"{int(match.group(2))}.{int(match.group(3))}"
            )
    if (
        engine == "vyper"
        and isinstance(value, list)
        and len(value) == 3
        and all(
            isinstance(item, int) and 0 <= item <= 255
            for item in value
        )
    ):
        return f"v{value[0]}.{value[1]}.{value[2]}"
    return "unknown"


def _valid_metadata_value(key: str, value: Any) -> bool:
    if key == "ipfs":
        return isinstance(value, bytes) and len(value) == 34
    if key in {"bzzr0", "bzzr1"}:
        return isinstance(value, bytes) and len(value) == 32
    if key == "experimental":
        return isinstance(value, bool)
    if key in {"solc", "vyper"}:
        return _version_from_value(key, value) != "unknown"
    return True


def decode_terminal_metadata(
    bytecode: bytes,
) -> dict[str, Any]:
    result = {
        "valid": False,
        "logic": bytecode,
        "metadata": b"",
        "keys": [],
        "compiler_engine": "unknown",
        "compiler_version": "unknown",
    }
    if len(bytecode) < 3:
        return result
    metadata_length = int.from_bytes(bytecode[-2:], "big")
    start = len(bytecode) - metadata_length - 2
    if metadata_length <= 0 or start < 0:
        return result
    payload = bytecode[start:-2]
    try:
        decoded, position = _decode_cbor(payload)
    except CBORDecodeError:
        return result
    if position != len(payload) or not isinstance(decoded, dict):
        return result
    normalized: dict[str, Any] = {}
    for key, value in decoded.items():
        if isinstance(key, bytes):
            try:
                key = key.decode("utf-8")
            except UnicodeDecodeError:
                continue
        if isinstance(key, str):
            normalized[key] = value
    recognized = sorted(RECOGNIZED_METADATA_KEYS & set(normalized))
    if not recognized:
        return result
    if any(
        not _valid_metadata_value(key, normalized[key])
        for key in recognized
    ):
        return result
    if "solc" in normalized and "vyper" in normalized:
        return result
    engine = "unknown"
    version = "unknown"
    for candidate in ("solc", "vyper"):
        if candidate in normalized:
            engine = candidate
            version = _version_from_value(candidate, normalized[candidate])
            break
    result.update(
        {
            "valid": True,
            "logic": bytecode[:start],
            "metadata": bytecode[start:],
            "keys": recognized,
            "compiler_engine": engine,
            "compiler_version": version,
        }
    )
    return result


def compiler_minor(version: str) -> str:
    match = VERSION_RE.search(version)
    if not match:
        return "unknown"
    return f"v{int(match.group(1))}.{int(match.group(2))}"


def scan_program(
    digest: bytes,
    bytecode: bytes,
    multiplicity: int,
) -> Program:
    metadata = decode_terminal_metadata(bytecode)
    local_opcode_counts = [0] * 256
    strict_logic = bytes(metadata["logic"])
    # Match the submitted corpus scanner for structural comparisons.
    scan_logic = bytes.fromhex(strip_solc_metadata(bytecode.hex()))
    features = scan_bytecode(
        scan_logic.hex(),
        local_opcode_counts,
        strip_metadata=False,
    )
    features["bytecode_raw_byte_length"] = len(bytecode)
    features["bytecode_logic_byte_length"] = len(scan_logic)
    features["metadata_byte_length"] = len(bytecode) - len(scan_logic)
    opcode_count = int(features["opcode_count"])
    divisor = opcode_count / 1000.0
    features["control_flow_per_1k_opcodes"] = (
        float(features["control_flow_op_count"]) / divisor
        if divisor > 0
        else 0.0
    )
    features["storage_per_1k_opcodes"] = (
        float(features["storage_op_count"]) / divisor
        if divisor > 0
        else 0.0
    )
    features["external_calls_per_1k_opcodes"] = (
        float(features["external_call_count"]) / divisor
        if divisor > 0
        else 0.0
    )
    flags = set()
    flag_sources = {
        "sload": int(features["sload_count"]) > 0,
        "sstore": int(features["sstore_count"]) > 0,
        "call": int(features["call_count"]) > 0,
        "callcode": int(features["callcode_count"]) > 0,
        "delegatecall": int(features["delegatecall_count"]) > 0,
        "staticcall": int(features["staticcall_count"]) > 0,
        "create": int(features["create_count"]) > 0,
        "create2": int(features["create2_count"]) > 0,
        "log": int(features["log_op_count"]) > 0,
        "selfdestruct": int(features["selfdestruct_count"]) > 0,
        "tload": local_opcode_counts[0x5C] > 0,
        "tstore": local_opcode_counts[0x5D] > 0,
        "mcopy": local_opcode_counts[0x5E] > 0,
    }
    flags.update(name for name, enabled in flag_sources.items() if enabled)
    if EIP1167_RUNTIME_RE.fullmatch(strict_logic):
        flags.add("canonical_eip1167")
    has_delegatecall = "delegatecall" in flags
    if has_delegatecall and EIP1967_IMPLEMENTATION_SLOT in strict_logic:
        flags.add("eip1967_implementation_pattern")
    if has_delegatecall and EIP1967_BEACON_SLOT in strict_logic:
        flags.add("eip1967_beacon_pattern")
    if {
        "canonical_eip1167",
        "eip1967_implementation_pattern",
        "eip1967_beacon_pattern",
    } & flags:
        flags.add("proxy_related_pattern_union")
    version = str(metadata["compiler_version"])
    return Program(
        digest=digest,
        bytecode=bytecode,
        multiplicity=multiplicity,
        features={name: float(features[name]) for name in FEATURES},
        opcode_counts=local_opcode_counts,
        flags=flags,
        compiler_engine=str(metadata["compiler_engine"]),
        compiler_version=version,
        compiler_minor=compiler_minor(version),
        metadata_valid=bool(metadata["valid"]),
        scanner_metadata_mismatch=(
            len(scan_logic) != len(strict_logic)
        ),
    )


def load_deployed_programs(
    base_db: Path,
    extension_db: Path,
    *,
    target_size: int,
) -> tuple[list[Program], dict[str, Any]]:
    aggregates: dict[bytes, list[Any]] = {}
    membership_digest = hashlib.sha256()
    source_counts: Counter[str] = Counter()
    total_bytecode_bytes = 0
    for contract in select_exact_nonempty(
        base_db,
        extension_db,
        target_size=target_size,
    ):
        membership_digest.update(
            contract.source.encode("ascii")
            + b":"
            + str(contract.sample_index).encode("ascii")
            + b":"
            + contract.address.hex().encode("ascii")
            + b":"
            + contract.code_keccak.hex().encode("ascii")
            + b"\n"
        )
        source_counts[contract.source] += 1
        total_bytecode_bytes += len(contract.bytecode)
        existing = aggregates.get(contract.code_keccak)
        if existing is None:
            aggregates[contract.code_keccak] = [contract.bytecode, 1]
        else:
            if existing[0] != contract.bytecode:
                raise ValueError(
                    "one code hash maps to multiple runtime bytecodes"
                )
            existing[1] += 1
    programs = [
        scan_program(digest, value[0], int(value[1]))
        for digest, value in aggregates.items()
    ]
    multiplicities = sorted(
        (program.multiplicity for program in programs),
        reverse=True,
    )
    deployments_in_groups_ge_100 = sum(
        value for value in multiplicities if value >= 100
    )
    validation = {
        "target_size": target_size,
        "selected": sum(source_counts.values()),
        "source_counts": dict(source_counts),
        "membership_sha256": membership_digest.hexdigest(),
        "total_bytecode_bytes": total_bytecode_bytes,
        "unique_runtime_bytecodes": len(programs),
        "singleton_runtime_bytecodes": sum(
            value == 1 for value in multiplicities
        ),
        "largest_runtime_multiplicity": multiplicities[0],
        "deployments_in_runtime_groups_ge_100": (
            deployments_in_groups_ge_100
        ),
    }
    return programs, validation


def build_view(
    key: str,
    programs: Iterable[Program],
    *,
    deployment_weighted: bool,
) -> PopulationView:
    view = PopulationView(key)
    for program in programs:
        view.add(
            program,
            program.multiplicity if deployment_weighted else 1,
        )
    return view


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def load_benchmark(
    path: Path,
) -> tuple[list[Program], Counter[str], dict[str, Any]]:
    programs = []
    known_compilers: Counter[str] = Counter()
    validation = {
        "rows": 0,
        "metadata_compiler_detected": 0,
        "metadata_compiler_exact_matches": 0,
    }
    detected_by_known_minor: Counter[str] = Counter()
    matched_by_known_minor: Counter[str] = Counter()
    for index, row in enumerate(iter_jsonl(path), start=1):
        bytecode_text = str(row.get("bytecode") or "")
        if bytecode_text.startswith("0x"):
            bytecode_text = bytecode_text[2:]
        bytecode = bytes.fromhex(bytecode_text)
        digest = hashlib.sha256(bytecode).digest()
        program = scan_program(digest, bytecode, 1)
        programs.append(program)
        known_version = str(
            row.get("compiler_version")
            or (row.get("compiler") or {}).get("compiler_version")
            or "unknown"
        )
        known_minor = compiler_minor(known_version)
        known_compilers[known_minor] += 1
        validation["rows"] += 1
        if program.compiler_version != "unknown":
            validation["metadata_compiler_detected"] += 1
            detected_by_known_minor[known_minor] += 1
            known_match = VERSION_RE.search(known_version)
            inferred_match = VERSION_RE.search(program.compiler_version)
            if (
                known_match is not None
                and inferred_match is not None
                and known_match.groups() == inferred_match.groups()
            ):
                validation["metadata_compiler_exact_matches"] += 1
                matched_by_known_minor[known_minor] += 1
    validation["metadata_recovery_by_known_minor"] = {
        name: {
            "known": count,
            "detected": detected_by_known_minor.get(name, 0),
            "matched": matched_by_known_minor.get(name, 0),
            "detection_rate": detected_by_known_minor.get(name, 0) / count,
        }
        for name, count in sorted(known_compilers.items())
    }
    return programs, known_compilers, validation


def load_verified_numeric(
    path: Path,
) -> tuple[dict[str, array], Counter[str], int]:
    values = {feature: array("d") for feature in REPORT_FEATURES}
    low_level_counts: Counter[str] = Counter()
    rows = 0
    for row in iter_jsonl(path):
        for feature in REPORT_FEATURES:
            value = row.get(feature)
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"verified statistics row {rows + 1} lacks {feature}"
                )
            values[feature].append(float(value))
        for flag, field_name in VERIFIED_LOW_LEVEL_COUNT_FIELDS.items():
            count = row.get(field_name)
            if not isinstance(count, int) or count < 0:
                raise ValueError(
                    f"verified statistics row {rows + 1} has invalid "
                    f"{field_name}"
                )
            if count > 0:
                low_level_counts[flag] += 1
        rows += 1
    return values, low_level_counts, rows


def compare_view_to_verified(
    view: PopulationView,
    verified_values: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    result = {}
    for feature in REPORT_FEATURES:
        verified_pairs = [
            (float(value), 1) for value in verified_values[feature]
        ]
        result[feature] = {
            "ks_distance": weighted_ks_distance(
                verified_pairs,
                view.numeric[feature],
            ),
            "log1p_standardized_mean_difference": (
                log1p_standardized_mean_difference(
                    verified_pairs,
                    view.numeric[feature],
                )
            ),
        }
    return result


def compare_numeric(
    reference: PopulationView,
    comparison: PopulationView,
) -> dict[str, Any]:
    result = {}
    for feature in FEATURES:
        result[feature] = {
            "reference": weighted_summary(reference.numeric[feature]),
            "comparison": weighted_summary(comparison.numeric[feature]),
            "ks_distance": weighted_ks_distance(
                reference.numeric[feature],
                comparison.numeric[feature],
            ),
            "log1p_standardized_mean_difference": (
                log1p_standardized_mean_difference(
                    reference.numeric[feature],
                    comparison.numeric[feature],
                )
            ),
        }
    return result


def normalized_opcode_distribution(
    view: PopulationView,
) -> list[float]:
    total = sum(view.opcode_counts)
    if total <= 0:
        raise ValueError(f"{view.key} has no decoded opcodes")
    return [value / total for value in view.opcode_counts]


def opcode_display_name(opcode: int) -> str:
    return MODERN_OPCODE_NAMES.get(opcode, opcode_name(opcode))


def opcode_distribution_comparison(
    reference: PopulationView,
    comparison: PopulationView,
) -> dict[str, Any]:
    ref = normalized_opcode_distribution(reference)
    comp = normalized_opcode_distribution(comparison)
    total_variation = 0.5 * sum(
        abs(left - right) for left, right in zip(ref, comp)
    )
    midpoint = [(left + right) / 2.0 for left, right in zip(ref, comp)]

    def kl(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(
            value * math.log2(value / target)
            for value, target in zip(left, right)
            if value > 0 and target > 0
        )

    js_divergence = 0.5 * kl(ref, midpoint) + 0.5 * kl(comp, midpoint)
    differences = []
    for opcode, (left, right) in enumerate(zip(ref, comp)):
        name = opcode_display_name(opcode)
        differences.append(
            {
                "opcode": opcode,
                "name": name,
                "reference_share": left,
                "comparison_share": right,
                "difference": right - left,
            }
        )
    differences.sort(key=lambda row: abs(row["difference"]), reverse=True)
    return {
        "reference_opcode_total": sum(reference.opcode_counts),
        "comparison_opcode_total": sum(comparison.opcode_counts),
        "total_variation": total_variation,
        "jensen_shannon_divergence_bits": js_divergence,
        "largest_absolute_differences": differences[:20],
    }


def opcode_distribution_vs_verified(
    view: PopulationView,
    verified_top_counts: Mapping[str, int],
    verified_opcode_total: int,
) -> dict[str, Any]:
    top_names = set(verified_top_counts)
    view_by_name: Counter[str] = Counter()
    for opcode, count in enumerate(view.opcode_counts):
        view_by_name[opcode_display_name(opcode)] += count
    view_total = sum(view.opcode_counts)
    verified_other = verified_opcode_total - sum(
        int(value) for value in verified_top_counts.values()
    )
    view_other = view_total - sum(
        view_by_name.get(name, 0) for name in top_names
    )
    categories = sorted(top_names) + ["OTHER"]
    verified_distribution = {}
    view_distribution = {}
    for name in categories:
        if name == "OTHER":
            verified_count = verified_other
            view_count = view_other
        else:
            verified_count = int(verified_top_counts[name])
            view_count = int(view_by_name.get(name, 0))
        verified_distribution[name] = verified_count / verified_opcode_total
        view_distribution[name] = view_count / view_total
    total_variation = 0.5 * sum(
        abs(verified_distribution[name] - view_distribution[name])
        for name in categories
    )
    midpoint = {
        name: (
            verified_distribution[name] + view_distribution[name]
        )
        / 2.0
        for name in categories
    }

    def kl(left: Mapping[str, float]) -> float:
        return sum(
            left[name] * math.log2(left[name] / midpoint[name])
            for name in categories
            if left[name] > 0 and midpoint[name] > 0
        )

    differences = [
        {
            "name": name,
            "verified_share": verified_distribution[name],
            "deployed_share": view_distribution[name],
            "difference": (
                view_distribution[name] - verified_distribution[name]
            ),
        }
        for name in categories
    ]
    differences.sort(key=lambda row: abs(row["difference"]), reverse=True)
    return {
        "verified_opcode_total": verified_opcode_total,
        "deployed_opcode_total": view_total,
        "partition": "verified_top80_plus_other",
        "total_variation": total_variation,
        "jensen_shannon_divergence_bits": (
            0.5 * kl(verified_distribution)
            + 0.5 * kl(view_distribution)
        ),
        "largest_absolute_differences": differences[:20],
    }


def prevalence(view: PopulationView, flag: str) -> float:
    return view.flag_counts.get(flag, 0) / view.total


def view_summary(view: PopulationView) -> dict[str, Any]:
    return {
        "total": view.total,
        "numeric": {
            feature: weighted_summary(view.numeric[feature])
            for feature in FEATURES
        },
        "flag_counts": dict(view.flag_counts),
        "flag_prevalence": {
            flag: count / view.total
            for flag, count in sorted(view.flag_counts.items())
        },
        "compiler_counts": dict(view.compiler_counts),
        "metadata_valid_count": view.metadata_valid_count,
        "metadata_valid_rate": view.metadata_valid_count / view.total,
        "compiler_detected_count": view.compiler_detected_count,
        "compiler_detected_rate": view.compiler_detected_count / view.total,
        "scanner_metadata_mismatch_count": (
            view.scanner_metadata_mismatch_count
        ),
        "opcode_total": sum(view.opcode_counts),
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def format_number(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.1f}"
    return f"{value:,.2f}"


def conditional_compiler_shares(
    counts: Mapping[str, int],
) -> dict[str, float]:
    total = sum(
        count for name, count in counts.items() if name != "unknown"
    )
    if total == 0:
        return {}
    return {
        name: count / total
        for name, count in counts.items()
        if name != "unknown"
    }


def compiler_minor_sort_key(name: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"v(\d+)\.(\d+)", name)
    if match is None:
        return (sys.maxsize, sys.maxsize, name)
    return (int(match.group(1)), int(match.group(2)), name)


def generate_report(results: Mapping[str, Any]) -> str:
    populations = results["populations"]
    weighted = populations["deployed_weighted"]
    dedup = populations["deployed_unique_bytecode"]
    benchmark = populations["scdbench"]
    validation = results["validation"]["deployed_selection"]
    numeric_weighted = results["comparisons"][
        "scdbench_vs_deployed_weighted"
    ]["numeric"]
    numeric_dedup = results["comparisons"][
        "scdbench_vs_deployed_unique_bytecode"
    ]["numeric"]
    weighted_vs_full = results["comparisons"][
        "deployed_weighted_vs_full_verified"
    ]
    unique_vs_full = results["comparisons"][
        "deployed_unique_bytecode_vs_full_verified"
    ]
    benchmark_vs_full = results["comparisons"][
        "scdbench_vs_full_verified_source_free"
    ]
    benchmark_vs_full_opcode = results["comparisons"][
        "scdbench_vs_full_verified_opcode_distribution"
    ]
    verified_numeric = results["verified_reference"][
        "numeric_summaries"
    ]
    verified_low_level = results["verified_reference"][
        "low_level_flag_prevalence"
    ]
    verified_eip1167 = results["verified_reference"][
        "canonical_eip1167"
    ]
    base_sampling = results["inputs"]["base_db"]["collector_manifest"][
        "sampling"
    ]
    extension_sampling = results["inputs"]["extension_db"][
        "collector_manifest"
    ]["sampling"]
    sampled_min_block = min(
        int(base_sampling["min_block"]),
        int(extension_sampling["min_block"]),
    )
    sampled_max_block = max(
        int(base_sampling["max_block"]),
        int(extension_sampling["max_block"]),
    )
    lines = [
        "# Deployed-Population External-Validity Analysis",
        "",
        "## Scope",
        "",
        (
            "We compare SCDBench with a deterministic one-million-address "
            "sample of non-empty historical Ethereum runtime bytecode. This is "
            "a sample from the deployed-address table, not an address-labeled "
            "unverified-only population. It directly measures bytecode "
            "covariate shift, but it cannot establish model accuracy on "
            "contracts without source ground truth."
        ),
        "",
        (
            "The PostgreSQL collection used reproducible random page sampling "
            "followed by hash ranking. Results are descriptive effect sizes; "
            "we do not report IID p-values or confidence intervals."
        ),
        "",
            (
                f"Historical bytecode was queried at each row's recorded "
                f"`max_block_number`; sampled rows span blocks "
                f"{sampled_min_block:,}--{sampled_max_block:,}."
            ),
        "",
        "## Populations",
        "",
        "| Population | Weighting | Contracts/programs |",
        "| --- | --- | ---: |",
        (
            f"| Deployed-address sample | Deployment weighted | "
            f"{weighted['total']:,} |"
        ),
        (
            f"| Deployed-address sample | Unique runtime bytecode | "
            f"{dedup['total']:,} |"
        ),
        (
            f"| Full source-verified corpus | Unique runtime bytecode | "
            f"{results['verified_reference']['rows']:,} |"
        ),
        f"| SCDBench | Contract | {benchmark['total']:,} |",
        "",
        (
            f"The exact-million selection contains {validation['unique_runtime_bytecodes']:,} "
            f"distinct runtime bytecodes. The largest bytecode occurs "
            f"{validation['largest_runtime_multiplicity']:,} times, and "
            f"{validation['deployments_in_runtime_groups_ge_100']:,} deployments "
            "belong to bytecode groups with at least 100 sampled addresses. We "
            "therefore report both deployment-weighted and deduplicated views."
        ),
        "",
        "## Bytecode and Structural Features",
        "",
        (
            "KS is the maximum empirical-CDF distance. The two KS columns "
            "compare SCDBench with each deployed-population view."
        ),
        "",
        (
            "| Feature | Deployed median | Deployed unique median | "
            "Full verified median | SCDBench median | KS weighted | KS unique |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for feature in REPORT_FEATURES:
        full_summary = verified_numeric.get(feature) or {}
        full_median = full_summary.get("median")
        full_text = (
            format_number(float(full_median))
            if isinstance(full_median, (int, float))
            else "n/a"
        )
        lines.append(
            f"| {FEATURE_LABELS[feature]} | "
            f"{format_number(weighted['numeric'][feature]['median'])} | "
            f"{format_number(dedup['numeric'][feature]['median'])} | "
            f"{full_text} | "
            f"{format_number(benchmark['numeric'][feature]['median'])} | "
            f"{numeric_weighted[feature]['ks_distance']:.3f} | "
            f"{numeric_dedup[feature]['ks_distance']:.3f} |"
        )

    lines.extend(
        [
            "",
            "### Full-Corpus and Benchmark Comparisons",
            "",
            (
                "The full source-verified corpus provides a much larger "
                "reference than SCDBench. Reporting it beside both deployed "
                "views separates deployment clone weighting from SCDBench's "
                "additional difficulty and diversity balancing."
            ),
            "",
            (
                "| Comparison | Raw-byte KS | Logic-byte KS | Opcode-count KS | "
                "Opcode-mix TV |"
            ),
            "| --- | ---: | ---: | ---: | ---: |",
            (
                "| Deployed weighted vs. full verified | "
                f"{weighted_vs_full['numeric']['bytecode_raw_byte_length']['ks_distance']:.3f} | "
                f"{weighted_vs_full['numeric']['bytecode_logic_byte_length']['ks_distance']:.3f} | "
                f"{weighted_vs_full['numeric']['opcode_count']['ks_distance']:.3f} | "
                f"{weighted_vs_full['opcode_distribution']['total_variation']:.3f} |"
            ),
            (
                "| Deployed unique vs. full verified | "
                f"{unique_vs_full['numeric']['bytecode_raw_byte_length']['ks_distance']:.3f} | "
                f"{unique_vs_full['numeric']['bytecode_logic_byte_length']['ks_distance']:.3f} | "
                f"{unique_vs_full['numeric']['opcode_count']['ks_distance']:.3f} | "
                f"{unique_vs_full['opcode_distribution']['total_variation']:.3f} |"
            ),
            (
                "| SCDBench vs. full verified | "
                f"{benchmark_vs_full['numeric']['bytecode_raw_byte_length']['ks_distance']:.3f} | "
                f"{benchmark_vs_full['numeric']['bytecode_logic_byte_length']['ks_distance']:.3f} | "
                f"{benchmark_vs_full['numeric']['opcode_count']['ks_distance']:.3f} | "
                f"{benchmark_vs_full_opcode['total_variation']:.3f} |"
            ),
            (
                "| SCDBench vs. deployed unique | "
                f"{numeric_dedup['bytecode_raw_byte_length']['ks_distance']:.3f} | "
                f"{numeric_dedup['bytecode_logic_byte_length']['ks_distance']:.3f} | "
                f"{numeric_dedup['opcode_count']['ks_distance']:.3f} | "
                f"{results['comparisons']['scdbench_vs_deployed_unique_bytecode']['opcode_distribution']['total_variation']:.3f} |"
            ),
        ]
    )

    proxy_rows = (
        ("canonical_eip1167", "Canonical EIP-1167 runtime"),
        (
            "eip1967_implementation_pattern",
            "DELEGATECALL + EIP-1967 implementation slot",
        ),
        (
            "eip1967_beacon_pattern",
            "DELEGATECALL + EIP-1967 beacon slot",
        ),
        ("proxy_related_pattern_union", "Proxy-related pattern union"),
        ("delegatecall", "Any DELEGATECALL"),
    )
    lines.extend(
        [
            "",
            "## Bytecode-Detectable Proxy Patterns",
            "",
            (
                "These are source-free indicators. The EIP-1167 row is an "
                "exact runtime match applied to all populations. The EIP-1967 "
                "rows are heuristics applied to the deployed sample and "
                "SCDBench; they can also match programs that embed "
                "proxy-deployment bytecode. Any DELEGATECALL is reported "
                "separately and is not treated as proof of a proxy."
            ),
            "",
            (
                "| Indicator | Deployed | Deployed unique | Full verified "
                "unique | SCDBench |"
            ),
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for flag, label in proxy_rows:
        if flag == "canonical_eip1167":
            full_verified_text = pct(verified_eip1167["rate"])
        elif flag == "delegatecall":
            full_verified_text = pct(verified_low_level["delegatecall"])
        else:
            full_verified_text = "n/a"
        lines.append(
            f"| {label} | {pct(weighted['flag_prevalence'].get(flag, 0.0))} | "
            f"{pct(dedup['flag_prevalence'].get(flag, 0.0))} | "
            f"{full_verified_text} | "
            f"{pct(benchmark['flag_prevalence'].get(flag, 0.0))} |"
        )

    lines.extend(
        [
            "",
            "## Low-Level Bytecode Features",
            "",
            (
                "| Feature | Deployed | Deployed unique | Full verified "
                "unique | SCDBench |"
            ),
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for flag, label in LOW_LEVEL_FLAGS:
        full_verified_value = verified_low_level.get(flag)
        full_verified_text = (
            pct(full_verified_value)
            if isinstance(full_verified_value, (int, float))
            else "n/a"
        )
        lines.append(
            f"| {label} | {pct(weighted['flag_prevalence'].get(flag, 0.0))} | "
            f"{pct(dedup['flag_prevalence'].get(flag, 0.0))} | "
            f"{full_verified_text} | "
            f"{pct(benchmark['flag_prevalence'].get(flag, 0.0))} |"
        )

    compiler_validation = results["validation"]["benchmark_metadata"]
    weighted_compiler = conditional_compiler_shares(
        weighted["compiler_counts"]
    )
    dedup_compiler = conditional_compiler_shares(dedup["compiler_counts"])
    benchmark_inferred = conditional_compiler_shares(
        benchmark["compiler_counts"]
    )
    benchmark_known = conditional_compiler_shares(
        results["validation"]["benchmark_known_compilers"]
    )
    full_compiler = conditional_compiler_shares(
        results["verified_reference"]["compiler_counts"]
    )
    compiler_names = sorted(
        set(weighted_compiler)
        | set(dedup_compiler)
        | set(benchmark_inferred)
        | set(benchmark_known)
        | set(full_compiler),
        key=compiler_minor_sort_key,
    )
    recovery_by_minor = compiler_validation[
        "metadata_recovery_by_known_minor"
    ]
    detected = compiler_validation["metadata_compiler_detected"]
    matched = compiler_validation["metadata_compiler_exact_matches"]
    accuracy = matched / detected if detected else 0.0
    lines.extend(
        [
            "",
            "## Compiler Metadata",
            "",
            (
                "Exact versions are reported only when a terminal, "
                "length-delimited CBOR map decodes completely and contains a "
                "supported `solc`/`vyper` version encoding. Missing metadata is "
                "kept as unknown."
            ),
            "",
            "| Population | Valid metadata | Compiler version recovered |",
            "| --- | ---: | ---: |",
            (
                f"| Deployed | {pct(weighted['metadata_valid_rate'])} | "
                f"{pct(weighted['compiler_detected_rate'])} |"
            ),
            (
                f"| Deployed unique | {pct(dedup['metadata_valid_rate'])} | "
                f"{pct(dedup['compiler_detected_rate'])} |"
            ),
            (
                f"| SCDBench | {pct(benchmark['metadata_valid_rate'])} | "
                f"{pct(benchmark['compiler_detected_rate'])} |"
            ),
            "",
            (
                f"On SCDBench, metadata recovered {detected}/{benchmark['total']} "
                f"versions and matched source-provided compiler versions for "
                f"{matched}/{detected} ({pct(accuracy)}) of recovered cases."
            ),
            "",
            "| Known SCDBench compiler | Recovered | Exact matches |",
            "| --- | ---: | ---: |",
        ]
    )
    for name in sorted(recovery_by_minor, key=compiler_minor_sort_key):
        recovery = recovery_by_minor[name]
        lines.append(
            f"| {name} | {recovery['detected']}/{recovery['known']} "
            f"({pct(recovery['detection_rate'])}) | "
            f"{recovery['matched']}/{recovery['detected']} |"
        )
    lines.extend(
        [
            "",
            (
                "The distribution below is conditional on a recovered version "
                "for deployed bytecode; the full verified and SCDBench-known "
                "columns use source-provided compiler metadata."
            ),
            (
                "Recovery depends strongly on compiler version, so the "
                "metadata-inferred deployed mix cannot estimate the "
                "unconditional compiler-version distribution."
            ),
            "",
            (
                "| Compiler | Deployed | Deployed unique | SCDBench inferred | "
                "SCDBench known | Full verified |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in compiler_names:
        lines.append(
            f"| {name} | {pct(weighted_compiler.get(name, 0.0))} | "
            f"{pct(dedup_compiler.get(name, 0.0))} | "
            f"{pct(benchmark_inferred.get(name, 0.0))} | "
            f"{pct(benchmark_known.get(name, 0.0))} | "
            f"{pct(full_compiler.get(name, 0.0))} |"
        )

    opcode_weighted = results["comparisons"][
        "scdbench_vs_deployed_weighted"
    ]["opcode_distribution"]
    opcode_dedup = results["comparisons"][
        "scdbench_vs_deployed_unique_bytecode"
    ]["opcode_distribution"]
    lines.extend(
        [
            "",
            "## Opcode Identity Distribution",
            "",
            (
                f"Aggregate instruction-distribution TV distance is "
                f"**{opcode_weighted['total_variation']:.3f}** against the "
                f"deployment-weighted population and "
                f"**{opcode_dedup['total_variation']:.3f}** against unique "
                "deployed bytecodes. Jensen-Shannon divergence is "
                f"{opcode_weighted['jensen_shannon_divergence_bits']:.3f} and "
                f"{opcode_dedup['jensen_shannon_divergence_bits']:.3f} bits, "
                "respectively."
            ),
            "",
            (
                "| Opcode | Deployed unique | SCDBench | Difference |"
            ),
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in opcode_dedup["largest_absolute_differences"][:12]:
        lines.append(
            f"| {row['name']} | {pct(row['reference_share'])} | "
            f"{pct(row['comparison_share'])} | "
            f"{100.0 * row['difference']:+.2f} pp |"
        )

    weighted_raw = weighted["numeric"]["bytecode_raw_byte_length"]
    dedup_raw = dedup["numeric"]["bytecode_raw_byte_length"]
    benchmark_raw = benchmark["numeric"]["bytecode_raw_byte_length"]
    full_raw = verified_numeric["bytecode_raw_byte_length"]
    weighted_proxy = weighted["flag_prevalence"].get(
        "canonical_eip1167",
        0.0,
    )
    dedup_proxy = dedup["flag_prevalence"].get(
        "canonical_eip1167",
        0.0,
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"- The deployment-weighted population is dominated by short "
                f"repeated programs: median runtime size is "
                f"{weighted_raw['median']:,.0f} bytes, versus "
                f"{float(full_raw['median']):,.0f} in the full verified corpus "
                f"and {benchmark_raw['median']:,.0f} in SCDBench. Canonical "
                f"EIP-1167 runtimes account for {pct(weighted_proxy)} of "
                "sampled deployments."
            ),
            (
                f"- Deduplicating sampled runtime bytecodes raises the median "
                f"to {dedup_raw['median']:,.0f} bytes and reduces canonical "
                f"EIP-1167 prevalence to {pct(dedup_proxy)}, but the structural "
                "gap with SCDBench remains."
            ),
            (
                "- This does not make SCDBench invalid: a source-recovery "
                "benchmark necessarily targets contracts with recoverable "
                "logic and source/ABI ground truth. It does mean that SCDBench "
                "should be described as diversity-balanced over source-verified "
                "logic contracts, not population-weighted over all deployed "
                "addresses."
            ),
            (
                "- Source lines, source-level inline assembly, and exact "
                "compiler settings are unavailable for arbitrary deployed "
                "bytecode. We therefore compare only source-free features and "
                "report compiler versions conditional on validated metadata."
            ),
            "",
            "## Interpretation",
            "",
            (
                "To assess the benchmark's external validity, we sampled "
                "1,000,000 addresses with non-empty "
                "historical runtime bytecode from the deployed-address table "
                "without using verification labels. Because this sample is "
                "not explicitly labeled as unverified, we use it as a "
                "deployed-versus-verified covariate-shift audit, not as a "
                "direct accuracy evaluation on unverified contracts."
            ),
            "",
            (
                "The sample contains "
                f"{validation['unique_runtime_bytecodes']:,} distinct "
                f"runtime bytecodes. Its deployment-weighted median runtime is "
                f"{weighted_raw['median']:,.0f} bytes, and "
                f"{pct(weighted_proxy)} are exact canonical EIP-1167 runtimes. "
                "After exact-bytecode deduplication, these values become "
                f"{dedup_raw['median']:,.0f} bytes and {pct(dedup_proxy)}, "
                f"compared with median sizes of "
                f"{float(full_raw['median']):,.0f} bytes in the full verified "
                f"corpus and {benchmark_raw['median']:,.0f} bytes in SCDBench. "
                "On comparable unique-runtime views, deployed-versus-full-"
                "verified raw-size KS distance is "
                f"{unique_vs_full['numeric']['bytecode_raw_byte_length']['ks_distance']:.3f}, "
                "while opcode-mix TV distance is "
                f"{unique_vs_full['opcode_distribution']['total_variation']:.3f}. "
                "Low-level prevalence also differs: SLOAD is "
                f"{pct(dedup['flag_prevalence']['sload'])} versus "
                f"{pct(verified_low_level['sload'])}, SSTORE "
                f"{pct(dedup['flag_prevalence']['sstore'])} versus "
                f"{pct(verified_low_level['sstore'])}, and DELEGATECALL "
                f"{pct(dedup['flag_prevalence']['delegatecall'])} versus "
                f"{pct(verified_low_level['delegatecall'])}. SCDBench is "
                "closer to the full verified corpus in runtime size "
                f"(KS={benchmark_vs_full['numeric']['bytecode_raw_byte_length']['ks_distance']:.3f}), "
                "supporting its intended coverage within the source-verified "
                "frame."
            ),
            "",
            (
                "Compiler versions are recoverable from validated metadata for "
                f"only {pct(dedup['compiler_detected_rate'])} of unique "
                "deployed runtimes, with strongly version-dependent missingness. "
                "We therefore report conditional counts without claiming an "
                "unconditional compiler-version match. We will clarify that "
                "SCDBench is diversity-balanced over source-verified logic "
                "contracts, rather than population-representative of all "
                "deployed addresses."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    for path in (
        args.base_db,
        args.extension_db,
        args.benchmark,
        args.verified_stats,
        args.verified_eip1167,
        args.verified_summary,
        args.verified_results,
    ):
        if not path.exists():
            raise SystemExit(f"required input is missing: {path}")
    print("[deployed-comparison] loading exact deployed selection", flush=True)
    programs, deployed_validation = load_deployed_programs(
        args.base_db,
        args.extension_db,
        target_size=args.target_size,
    )
    if args.log_every > 0:
        print(
            "[deployed-comparison] "
            f"scanned_unique_programs={len(programs):,}",
            flush=True,
        )
    deployed_weighted = build_view(
        "deployed_weighted",
        programs,
        deployment_weighted=True,
    )
    deployed_unique = build_view(
        "deployed_unique_bytecode",
        programs,
        deployment_weighted=False,
    )
    benchmark_programs, known_compilers, benchmark_validation = (
        load_benchmark(args.benchmark)
    )
    if len(benchmark_programs) != 600:
        raise ValueError(
            f"expected 600 SCDBench rows, got {len(benchmark_programs)}"
        )
    benchmark = build_view(
        "scdbench",
        benchmark_programs,
        deployment_weighted=False,
    )
    verified_summary = json.loads(
        args.verified_summary.read_text(encoding="utf-8")
    )
    verified_results = json.loads(
        args.verified_results.read_text(encoding="utf-8")
    )
    verified_eip1167 = json.loads(
        args.verified_eip1167.read_text(encoding="utf-8")
    )
    full_rows = int(verified_summary["candidate_valid_rows"])
    expected_full_rows = int(
        verified_results["populations"][
            "verified_exact_bytecode_deduped"
        ]["rows"]
    )
    if full_rows != expected_full_rows:
        raise ValueError("verified summary row counts disagree")
    eip1167_rows = int(verified_eip1167["input"]["rows"])
    eip1167_count = int(
        verified_eip1167["result"]["canonical_eip1167_count"]
    )
    eip1167_rate = float(verified_eip1167["result"]["rate"])
    if eip1167_rows != full_rows:
        raise ValueError("verified EIP-1167 row count disagrees")
    if not math.isclose(
        eip1167_rate,
        eip1167_count / eip1167_rows,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("verified EIP-1167 count and rate disagree")
    print(
        "[deployed-comparison] loading verified numeric reference",
        flush=True,
    )
    (
        verified_values,
        verified_low_level_counts,
        verified_stats_rows,
    ) = load_verified_numeric(args.verified_stats)
    if verified_stats_rows != full_rows:
        raise ValueError(
            f"verified stats contain {verified_stats_rows} rows, "
            f"expected {full_rows}"
        )

    weighted_comparison = {
        "numeric": compare_numeric(deployed_weighted, benchmark),
        "opcode_distribution": opcode_distribution_comparison(
            deployed_weighted,
            benchmark,
        ),
    }
    unique_comparison = {
        "numeric": compare_numeric(deployed_unique, benchmark),
        "opcode_distribution": opcode_distribution_comparison(
            deployed_unique,
            benchmark,
        ),
    }
    verified_opcode_total = int(
        verified_results["populations"][
            "verified_exact_bytecode_deduped"
        ]["opcode_total"]
    )
    verified_top_opcode_counts = verified_summary[
        "opcode_counts_top80_all_rows"
    ]
    weighted_vs_verified = {
        "numeric": compare_view_to_verified(
            deployed_weighted,
            verified_values,
        ),
        "opcode_distribution": opcode_distribution_vs_verified(
            deployed_weighted,
            verified_top_opcode_counts,
            verified_opcode_total,
        ),
    }
    unique_vs_verified = {
        "numeric": compare_view_to_verified(
            deployed_unique,
            verified_values,
        ),
        "opcode_distribution": opcode_distribution_vs_verified(
            deployed_unique,
            verified_top_opcode_counts,
            verified_opcode_total,
        ),
    }
    benchmark_vs_verified = {
        "numeric": compare_view_to_verified(
            benchmark,
            verified_values,
        ),
        "opcode_distribution": verified_results[
            "opcode_identity_mix"
        ],
    }
    base_manifest = collector_manifest_provenance(args.base_db)
    extension_manifest = collector_manifest_provenance(args.extension_db)
    results = {
        "schema": "scdbench-deployed-comparison-v1",
        "scope": {
            "primary": "deployed-address table sample versus SCDBench",
            "not_claimed": [
                "address-labeled verified-versus-unverified comparison",
                "model accuracy on contracts without source ground truth",
                "IID row-sampling confidence intervals",
            ],
            "weighting_views": [
                "deployment_weighted",
                "exact_runtime_bytecode_deduplicated",
            ],
        },
        "inputs": {
            "base_db": {
                "path": str(args.base_db.resolve()),
                "bytes": args.base_db.stat().st_size,
                "collector_manifest": base_manifest,
            },
            "extension_db": {
                "path": str(args.extension_db.resolve()),
                "bytes": args.extension_db.stat().st_size,
                "collector_manifest": extension_manifest,
            },
            "benchmark": {
                "path": str(args.benchmark.resolve()),
                "sha256": sha256_file(args.benchmark),
            },
            "verified_summary": {
                "path": str(args.verified_summary.resolve()),
                "sha256": sha256_file(args.verified_summary),
            },
            "verified_stats": {
                "path": str(args.verified_stats.resolve()),
                "sha256": sha256_file(args.verified_stats),
            },
            "verified_eip1167": {
                "path": str(args.verified_eip1167.resolve()),
                "sha256": sha256_file(args.verified_eip1167),
                "source_input": verified_eip1167["input"],
            },
        },
        "validation": {
            "deployed_selection": deployed_validation,
            "benchmark_metadata": benchmark_validation,
            "benchmark_known_compilers": dict(known_compilers),
        },
        "populations": {
            "deployed_weighted": view_summary(deployed_weighted),
            "deployed_unique_bytecode": view_summary(deployed_unique),
            "scdbench": view_summary(benchmark),
        },
        "verified_reference": {
            "rows": full_rows,
            "numeric_summaries": verified_summary[
                "numeric_summaries_valid"
            ],
            "low_level_flag_counts": dict(verified_low_level_counts),
            "low_level_flag_prevalence": {
                flag: count / full_rows
                for flag, count in sorted(
                    verified_low_level_counts.items()
                )
            },
            "canonical_eip1167": {
                "count": eip1167_count,
                "rate": eip1167_rate,
            },
            "compiler_counts": verified_summary[
                "compiler_minor_counts_valid"
            ],
            "wrapper_type_counts": verified_summary[
                "wrapper_type_counts_valid"
            ],
        },
        "comparisons": {
            "scdbench_vs_deployed_weighted": weighted_comparison,
            "scdbench_vs_deployed_unique_bytecode": unique_comparison,
            "deployed_weighted_vs_full_verified": weighted_vs_verified,
            "deployed_unique_bytecode_vs_full_verified": (
                unique_vs_verified
            ),
            "scdbench_vs_full_verified": verified_results["comparisons"][
                "benchmark_vs_full_verified"
            ],
            "scdbench_vs_full_verified_source_free": (
                benchmark_vs_verified
            ),
            "scdbench_vs_full_verified_opcode_distribution": (
                verified_results["opcode_identity_mix"]
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = generate_report(results)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(report, encoding="utf-8")
    print(f"[deployed-comparison] wrote {args.output_json}", flush=True)
    print(f"[deployed-comparison] wrote {args.output_markdown}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
