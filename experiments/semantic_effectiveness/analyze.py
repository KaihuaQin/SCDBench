#!/usr/bin/env python3
"""Analyze the effectiveness of SCDBench's submitted semantic checkpoints.

The bundled inputs preserve case identities, test families, archived coverage,
and observed mismatch modalities from the submitted benchmark and repaired runs.
See export_inputs.py for the projection from full replay observations.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCHEMA = "scdbench-rebuttal-semantic-effectiveness-v1"
EXPECTED_CONTRACTS = 600
EXPECTED_FUNCTION_GROUPS = 14_553
EXPECTED_CASES = 227_383
CATEGORIES = ("random", "boundary", "static", "stateful")
MODALITIES = ("status", "returndata", "logs", "storage")
MODEL_FILES = (
    (
        "opus_repair",
        "Claude Opus 4.7 + repair",
        "inputs/semantic_replay_claude_opus47_reasoning_high_after_repair.json.gz",
    ),
    (
        "gpt_repair",
        "GPT-5.3-Codex + repair",
        "inputs/semantic_replay_gpt53_codex_reasoning_high_after_repair.json.gz",
    ),
    (
        "glm_high_repair",
        "GLM-5 high + repair",
        "inputs/semantic_replay_glm5_reasoning_high_after_repair.json.gz",
    ),
    (
        "glm_instruct_repair",
        "GLM-5 instruct + repair",
        "inputs/semantic_replay_glm5_reasoning_none_after_repair.json.gz",
    ),
)
COVERAGE_SUBSETS = (
    ("all_contracts", None),
    ("coverage_measurable", "measurable"),
    ("coverage_ge_25pct", 0.25),
    ("coverage_ge_50pct", 0.50),
    ("coverage_ge_75pct", 0.75),
    ("coverage_ge_80pct", 0.80),
)
CHUNK_SIZE = 4 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--benchmark", type=Path, default=Path("inputs/benchmark.jsonl.gz"))
    parser.add_argument("--output-json", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--output-md", type=Path, default=Path(__file__).with_name("results.md"))
    return parser.parse_args()


def resolve_under(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def pct(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * float(numerator) / float(denominator)


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "p05": quantile(values, 0.05),
        "p10": quantile(values, 0.10),
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def category_for_kind(case_kind: str) -> str:
    if case_kind == "deterministic_random":
        return "random"
    if case_kind.startswith("branch_boundary"):
        return "boundary"
    if case_kind.startswith("static_hint"):
        return "static"
    if case_kind == "stateful_setup":
        return "stateful"
    raise ValueError(f"unknown semantic testcase kind: {case_kind!r}")


def iter_json_array(path: Path, key: str = "results") -> Iterator[dict[str, Any]]:
    """Stream objects from a named top-level JSON array.

    This avoids json.load() on replay artifacts as large as 726 MiB.
    """

    decoder = json.JSONDecoder()
    key_pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*\[')
    with open_text(path) as handle:
        buffer = ""
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                raise ValueError(f"{path}: top-level array {key!r} not found")
            buffer += chunk
            match = key_pattern.search(buffer)
            if match:
                buffer = buffer[match.end() :]
                break
            buffer = buffer[-max(1024, len(key) + 32) :]

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            while True:
                try:
                    item, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError as exc:
                    chunk = handle.read(CHUNK_SIZE)
                    if not chunk:
                        raise ValueError(f"{path}: incomplete {key!r} array item") from exc
                    buffer += chunk
            if not isinstance(item, dict):
                raise ValueError(f"{path}: expected object in {key!r}, got {type(item).__name__}")
            buffer = buffer[end:]
            yield item


def load_tail_summary(path: Path) -> dict[str, Any]:
    """Read the small top-level summary object at the end of a replay file."""

    if path.suffix == ".gz":
        with open_text(path) as handle:
            return json.load(handle)["summary"]
    size = path.stat().st_size
    tail_bytes = min(size, 256 * 1024)
    with path.open("rb") as handle:
        handle.seek(size - tail_bytes)
        tail = handle.read().decode("utf-8")
    marker = '\n  "summary": '
    index = tail.rfind(marker)
    if index < 0:
        raise ValueError(f"{path}: top-level summary not found in final {tail_bytes} bytes")
    summary, _ = json.JSONDecoder().raw_decode(tail[index + len(marker) :])
    if not isinstance(summary, dict):
        raise ValueError(f"{path}: top-level summary is not an object")
    return summary


def compact_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def metric_coverage_summary(records: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    measurable = []
    covered_total = 0
    executable_total = 0
    for coverage in records:
        values = coverage.get(metric) or {}
        value = values.get("ratio")
        if value is None:
            continue
        measurable.append(float(value))
        covered_total += int(values.get("covered") or 0)
        executable_total += int(values.get("total") or 0)
    return {
        "measurable_contracts": len(measurable),
        "covered": covered_total,
        "total": executable_total,
        "pooled_ratio": ratio(covered_total, executable_total),
        "pooled_percent": pct(covered_total, executable_total),
        "per_contract_ratio": distribution(measurable),
    }


def analyze_benchmark(
    path: Path,
) -> tuple[dict[str, Any], dict[str, tuple[str, str, str, str, str]], dict[str, dict[str, Any]]]:
    case_index: dict[str, tuple[str, str, str, str, str]] = {}
    contract_meta: dict[str, dict[str, Any]] = {}
    category_cases: Counter[str] = Counter()
    raw_kind_cases: Counter[str] = Counter()
    entry_types: Counter[str] = Counter()
    category_functions: dict[str, set[tuple[str, str]]] = {category: set() for category in CATEGORIES}
    category_contracts: dict[str, set[str]] = {category: set() for category in CATEGORIES}
    coverage_records: list[dict[str, Any]] = []
    coverage_by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    coverage_available = 0
    coverage_unavailable_reasons: Counter[str] = Counter()
    available_without_line_ratio = 0
    contract_count = 0
    function_group_count = 0

    with open_text(path) as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            contract_count += 1
            benchmark_id = str(row.get("benchmark_id") or "")
            if not benchmark_id or benchmark_id in contract_meta:
                raise ValueError(f"{path}:{line_number}: missing or duplicate benchmark_id {benchmark_id!r}")
            difficulty = str(row.get("difficulty_bin") or "")
            coverage = row.get("semantic_coverage") or {}
            if coverage.get("available") is True:
                coverage_available += 1
                coverage_records.append(coverage)
                coverage_by_difficulty[difficulty].append(coverage)
                if (coverage.get("lines") or {}).get("ratio") is None:
                    available_without_line_ratio += 1
            else:
                coverage_unavailable_reasons[str(coverage.get("reason") or "coverage_missing")] += 1

            groups = row.get("test_cases") or {}
            if not isinstance(groups, dict):
                raise ValueError(f"{path}:{line_number}: test_cases is not an object")
            line_ratio = (coverage.get("lines") or {}).get("ratio") if coverage.get("available") is True else None
            contract_meta[benchmark_id] = {
                "difficulty": difficulty,
                "function_groups": len(groups),
                "coverage_available": coverage.get("available") is True,
                "line_coverage_ratio": float(line_ratio) if line_ratio is not None else None,
            }
            function_group_count += len(groups)

            for signature, group in groups.items():
                signature = str(signature)
                entry_types[str(group.get("entry_type") or "unknown")] += 1
                cases = group.get("cases") or []
                declared_count = int(group.get("case_count", len(cases)))
                if declared_count != len(cases):
                    raise ValueError(
                        f"{benchmark_id}/{signature}: declared case_count={declared_count}, actual={len(cases)}"
                    )
                for case in cases:
                    case_id = str(case.get("case_id") or "")
                    if not case_id or case_id in case_index:
                        raise ValueError(f"{benchmark_id}/{signature}: missing or duplicate case_id {case_id!r}")
                    raw_kind = str(case.get("case_kind") or "")
                    category = category_for_kind(raw_kind)
                    case_signature = str(case.get("signature") or signature)
                    if case_signature != signature:
                        raise ValueError(
                            f"{case_id}: group signature {signature!r} != case signature {case_signature!r}"
                        )
                    state = str(case.get("state") or group.get("state") or "")
                    case_index[case_id] = (benchmark_id, signature, category, raw_kind, state)
                    category_cases[category] += 1
                    raw_kind_cases[raw_kind] += 1
                    category_functions[category].add((benchmark_id, signature))
                    category_contracts[category].add(benchmark_id)

    if contract_count != EXPECTED_CONTRACTS:
        raise ValueError(f"expected {EXPECTED_CONTRACTS} contracts, found {contract_count}")
    if function_group_count != EXPECTED_FUNCTION_GROUPS:
        raise ValueError(f"expected {EXPECTED_FUNCTION_GROUPS} function groups, found {function_group_count}")
    if len(case_index) != EXPECTED_CASES:
        raise ValueError(f"expected {EXPECTED_CASES} testcases, found {len(case_index)}")
    if sum(category_cases.values()) != EXPECTED_CASES:
        raise ValueError("case category counts do not sum to the submitted testcase total")
    if set(category_cases) != set(CATEGORIES):
        raise ValueError(f"unexpected category set: {set(category_cases)}")

    measurable_lines = [
        coverage
        for coverage in coverage_records
        if (coverage.get("lines") or {}).get("ratio") is not None
    ]
    by_difficulty = {}
    for difficulty in sorted({meta["difficulty"] for meta in contract_meta.values()}):
        all_contracts = [meta for meta in contract_meta.values() if meta["difficulty"] == difficulty]
        records = coverage_by_difficulty.get(difficulty, [])
        by_difficulty[difficulty] = {
            "contracts": len(all_contracts),
            "coverage_available": sum(1 for meta in all_contracts if meta["coverage_available"]),
            "line_coverage_measurable": sum(1 for meta in all_contracts if meta["line_coverage_ratio"] is not None),
            "lines": metric_coverage_summary(records, "lines"),
            "functions": metric_coverage_summary(records, "functions"),
            "branches": metric_coverage_summary(records, "branches"),
        }

    category_summary = {}
    for category in CATEGORIES:
        count = category_cases[category]
        category_summary[category] = {
            "cases": count,
            "case_share": ratio(count, EXPECTED_CASES),
            "case_share_percent": pct(count, EXPECTED_CASES),
            "function_groups_with_category": len(category_functions[category]),
            "contracts_with_category": len(category_contracts[category]),
        }

    benchmark = {
        "contracts": contract_count,
        "function_groups": function_group_count,
        "test_cases": len(case_index),
        "entry_types": compact_counter(entry_types),
        "case_composition": {
            "by_category": category_summary,
            "by_case_kind": compact_counter(raw_kind_cases),
        },
        "coverage": {
            "available_contracts": coverage_available,
            "available_without_line_ratio": available_without_line_ratio,
            "line_coverage_measurable_contracts": len(measurable_lines),
            "unavailable_contracts": contract_count - coverage_available,
            "unavailable_reasons": compact_counter(coverage_unavailable_reasons),
            "overall": {
                "lines": metric_coverage_summary(coverage_records, "lines"),
                "functions": metric_coverage_summary(coverage_records, "functions"),
                "branches": metric_coverage_summary(coverage_records, "branches"),
            },
            "by_difficulty": by_difficulty,
        },
    }
    return benchmark, case_index, contract_meta


def call_modalities(call: dict[str, Any], state: str) -> tuple[str, ...]:
    if "observed_mismatch_modalities" in call:
        values = call["observed_mismatch_modalities"]
        if not isinstance(values, list) or values != [key for key in MODALITIES if key in values]:
            raise ValueError("invalid projected mismatch modalities")
        if state in {"view", "pure"} and "storage" in values:
            raise ValueError("view/pure calls cannot have a storage-comparison mismatch")
        return tuple(values)
    expected = call.get("expected") or {}
    actual = call.get("actual") or {}
    expected_call = expected.get("call") or {}
    actual_call = actual.get("call") or {}
    modalities = []
    if bool(expected_call.get("ok")) != bool(actual_call.get("ok")):
        modalities.append("status")
    if str(expected_call.get("result") or "0x").lower() != str(actual_call.get("result") or "0x").lower():
        modalities.append("returndata")
    if (expected.get("logs") or []) != (actual.get("logs") or []):
        modalities.append("logs")
    if state not in {"view", "pure"} and (expected.get("storage_changes") or {}) != (
        actual.get("storage_changes") or {}
    ):
        modalities.append("storage")
    return tuple(modality for modality in MODALITIES if modality in modalities)


def analyze_model(
    path: Path,
    case_index: dict[str, tuple[str, str, str, str, str]],
) -> tuple[dict[str, Any], dict[str, int]]:
    by_category = {category: Counter() for category in CATEGORIES}
    by_case_kind: dict[str, Counter[str]] = defaultdict(Counter)
    modality_cases: Counter[str] = Counter()
    modality_case_combinations: Counter[str] = Counter()
    returndata_contexts: Counter[str] = Counter()
    functions: dict[tuple[str, str], dict[str, Any]] = {}
    contract_equivalent: dict[str, int] = {}
    calls_total = 0
    calls_matched = 0
    result_contracts = 0
    result_errors: Counter[str] = Counter()

    for result in iter_json_array(path):
        benchmark_id = str(result.get("benchmark_id") or "")
        if result.get("error"):
            result_errors[str(result.get("error"))] += 1
            continue
        result_contracts += 1
        per_result_functions: dict[tuple[str, str], bool] = {}
        for call in result.get("call_results") or []:
            calls_total += 1
            case_id = str(call.get("case_id") or "")
            if case_id not in case_index:
                raise ValueError(f"{path}: replayed unknown case_id {case_id!r}")
            case_benchmark, case_signature, category, raw_kind, indexed_state = case_index[case_id]
            signature = str(call.get("signature") or "")
            if case_benchmark != benchmark_id or case_signature != signature:
                raise ValueError(
                    f"{path}: {case_id} maps to {case_benchmark}/{case_signature}, "
                    f"replay says {benchmark_id}/{signature}"
                )
            state = str(call.get("state") or indexed_state)
            matched = bool(call.get("matched"))
            modalities = call_modalities(call, state)
            if matched != (len(modalities) == 0):
                raise ValueError(
                    f"{path}: {case_id} matched={matched}, reconstructed modalities={modalities}"
                )
            calls_matched += int(matched)
            by_category[category]["cases"] += 1
            by_category[category]["matched"] += int(matched)
            by_category[category]["mismatched"] += int(not matched)
            by_case_kind[raw_kind]["cases"] += 1
            by_case_kind[raw_kind]["matched"] += int(matched)
            by_case_kind[raw_kind]["mismatched"] += int(not matched)

            function_key = (benchmark_id, signature)
            observation = functions.setdefault(
                function_key,
                {
                    "categories_present": set(),
                    "mismatch_categories": set(),
                    "mismatch_modalities": set(),
                    "cases": 0,
                    "mismatches": 0,
                },
            )
            observation["categories_present"].add(category)
            observation["cases"] += 1
            per_result_functions.setdefault(function_key, False)
            if not matched:
                observation["mismatch_categories"].add(category)
                observation["mismatch_modalities"].update(modalities)
                observation["mismatches"] += 1
                per_result_functions[function_key] = True
                for modality in modalities:
                    modality_cases[modality] += 1
                combination = "+".join(modalities)
                modality_case_combinations[combination] += 1
                if "returndata" in modalities:
                    expected_ok = bool((call.get("expected") or {}).get("call", {}).get("ok"))
                    actual_ok = bool((call.get("actual") or {}).get("call", {}).get("ok"))
                    if expected_ok and actual_ok:
                        returndata_contexts["successful_return_bytes"] += 1
                    elif not expected_ok and not actual_ok:
                        returndata_contexts["revert_bytes"] += 1
                    else:
                        returndata_contexts["status_disagrees"] += 1

        derived_total = len(per_result_functions)
        derived_equivalent = sum(1 for failed in per_result_functions.values() if not failed)
        if derived_total != int(result.get("functions_total") or 0):
            raise ValueError(
                f"{path}: {benchmark_id} derived functions_total={derived_total}, "
                f"stored={result.get('functions_total')}"
            )
        if derived_equivalent != int(result.get("functions_equivalent") or 0):
            raise ValueError(
                f"{path}: {benchmark_id} derived functions_equivalent={derived_equivalent}, "
                f"stored={result.get('functions_equivalent')}"
            )
        contract_equivalent[benchmark_id] = derived_equivalent

    stored_summary = load_tail_summary(path)
    expected_summary = {
        "calls_total": calls_total,
        "calls_matched": calls_matched,
        "contracts": result_contracts,
        "valid_contracts": result_contracts,
        "functions_total": len(functions),
        "functions_equivalent": sum(1 for item in functions.values() if not item["mismatch_categories"]),
    }
    for key, value in expected_summary.items():
        if int(stored_summary.get(key) or 0) != value:
            raise ValueError(f"{path}: summary {key}={stored_summary.get(key)}, derived={value}")

    failing_functions = [item for item in functions.values() if item["mismatch_categories"]]
    category_function_summary = {}
    for category in CATEGORIES:
        functions_with_category = sum(category in item["categories_present"] for item in functions.values())
        detected = sum(category in item["mismatch_categories"] for item in failing_functions)
        exclusive = sum(item["mismatch_categories"] == {category} for item in failing_functions)
        additional_beyond_random = (
            None
            if category == "random"
            else sum(
                category in item["mismatch_categories"] and "random" not in item["mismatch_categories"]
                for item in failing_functions
            )
        )
        shapley_credit = sum(
            1.0 / len(item["mismatch_categories"])
            for item in failing_functions
            if category in item["mismatch_categories"]
        )
        category_function_summary[category] = {
            "replayed_function_evaluations_with_category": functions_with_category,
            "failing_function_evaluations_detected": detected,
            "detection_rate_among_category_bearing_functions": ratio(detected, functions_with_category),
            "detection_rate_among_category_bearing_functions_percent": pct(detected, functions_with_category),
            "exclusive_detection_removal_loss": exclusive,
            "additional_detections_beyond_random": additional_beyond_random,
            "order_independent_shapley_credit": shapley_credit,
        }

    shapley_total = sum(
        category_function_summary[category]["order_independent_shapley_credit"]
        for category in CATEGORIES
    )
    if not math.isclose(shapley_total, len(failing_functions), rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"{path}: category Shapley credit {shapley_total} != failures {len(failing_functions)}")

    function_modality_counts = Counter()
    function_modality_combinations = Counter()
    for item in failing_functions:
        modalities = tuple(modality for modality in MODALITIES if modality in item["mismatch_modalities"])
        if not modalities:
            raise ValueError(f"{path}: failing function has no reconstructed mismatch modality")
        for modality in modalities:
            function_modality_counts[modality] += 1
        function_modality_combinations["+".join(modalities)] += 1

    random_detected = sum("random" in item["mismatch_categories"] for item in failing_functions)
    augmentation_detected = sum(
        bool(item["mismatch_categories"] & {"boundary", "static", "stateful"})
        for item in failing_functions
    )
    augmentation_only = sum(
        "random" not in item["mismatch_categories"]
        and bool(item["mismatch_categories"] & {"boundary", "static", "stateful"})
        for item in failing_functions
    )
    random_only = sum(item["mismatch_categories"] == {"random"} for item in failing_functions)

    case_category_summary = {}
    for category in CATEGORIES:
        counts = by_category[category]
        case_category_summary[category] = {
            "replayed_cases": counts["cases"],
            "matched_cases": counts["matched"],
            "mismatched_cases": counts["mismatched"],
            "mismatch_rate": ratio(counts["mismatched"], counts["cases"]),
            "mismatch_rate_percent": pct(counts["mismatched"], counts["cases"]),
        }

    case_kind_summary = {}
    for case_kind in sorted(by_case_kind):
        counts = by_case_kind[case_kind]
        case_kind_summary[case_kind] = {
            "replayed_cases": counts["cases"],
            "matched_cases": counts["matched"],
            "mismatched_cases": counts["mismatched"],
            "mismatch_rate": ratio(counts["mismatched"], counts["cases"]),
            "mismatch_rate_percent": pct(counts["mismatched"], counts["cases"]),
        }

    model_summary = {
        "stored_replay_summary": stored_summary,
        "result_errors": compact_counter(result_errors),
        "denominators": {
            "replayed_contracts": result_contracts,
            "replayed_cases": calls_total,
            "replayed_model_function_evaluations": len(functions),
            "failing_model_function_evaluations": len(failing_functions),
            "equivalent_model_function_evaluations": len(functions) - len(failing_functions),
            "scope_note": (
                "Mismatch effectiveness is evaluated separately for this model over unique "
                "(benchmark_id, signature) pairs that were actually replayed. Missing or "
                "non-compiling functions have no case-level observations and are excluded here."
            ),
        },
        "case_mismatch_detection": {
            "by_category": case_category_summary,
            "by_case_kind": case_kind_summary,
        },
        "function_detection_contribution": {
            "by_category": category_function_summary,
            "random_vs_augmented": {
                "any_failing_function_evaluations": len(failing_functions),
                "detected_by_random": random_detected,
                "detected_by_any_augmented_category": augmentation_detected,
                "detected_by_augmentation_but_not_random": augmentation_only,
                "detected_only_by_random": random_only,
                "detected_by_both_random_and_augmentation": sum(
                    "random" in item["mismatch_categories"]
                    and bool(item["mismatch_categories"] & {"boundary", "static", "stateful"})
                    for item in failing_functions
                ),
            },
            "definitions": {
                "exclusive_detection_removal_loss": (
                    "Failing model-function evaluations detected by this category and no other; "
                    "these failures would be missed if the category were removed."
                ),
                "additional_detections_beyond_random": (
                    "Failing model-function evaluations detected by this category for which random "
                    "cases did not detect a mismatch. Augmented categories may overlap each other."
                ),
                "order_independent_shapley_credit": (
                    "For each failing model-function evaluation detected by k categories, each "
                    "detecting category receives 1/k credit. Credits sum to all detected failures."
                ),
            },
        },
        "mismatch_modalities": {
            "case_level": {
                "mismatched_cases": calls_total - calls_matched,
                "cases_by_modality_with_overlaps": compact_counter(modality_cases),
                "exclusive_and_overlap_combinations": compact_counter(modality_case_combinations),
                "returndata_mismatch_contexts": compact_counter(returndata_contexts),
            },
            "function_level": {
                "failing_model_function_evaluations": len(failing_functions),
                "functions_by_modality_with_overlaps": compact_counter(function_modality_counts),
                "exclusive_and_overlap_combinations": compact_counter(function_modality_combinations),
            },
            "comparison_note": (
                "Modalities reconstruct the evaluator exactly: success/revert status, returned or "
                "revert bytes, ordered logs, and storage_changes for non-view/non-pure calls."
            ),
        },
    }
    return model_summary, contract_equivalent


def coverage_subset(
    contract_meta: dict[str, dict[str, Any]],
    selector: str | float | None,
) -> list[str]:
    if selector is None:
        return sorted(contract_meta)
    if selector == "measurable":
        return sorted(
            benchmark_id
            for benchmark_id, meta in contract_meta.items()
            if meta["line_coverage_ratio"] is not None
        )
    threshold = float(selector)
    return sorted(
        benchmark_id
        for benchmark_id, meta in contract_meta.items()
        if meta["line_coverage_ratio"] is not None and meta["line_coverage_ratio"] >= threshold
    )


def ranking_sensitivity(
    contract_meta: dict[str, dict[str, Any]],
    model_equivalent: dict[str, dict[str, int]],
) -> dict[str, Any]:
    subsets = {}
    for subset_name, selector in COVERAGE_SUBSETS:
        benchmark_ids = coverage_subset(contract_meta, selector)
        denominator = sum(contract_meta[benchmark_id]["function_groups"] for benchmark_id in benchmark_ids)
        model_scores = {}
        for model_key, equivalent_by_contract in model_equivalent.items():
            equivalent = sum(equivalent_by_contract.get(benchmark_id, 0) for benchmark_id in benchmark_ids)
            model_scores[model_key] = {
                "equivalent_function_groups": equivalent,
                "function_group_denominator": denominator,
                "cumulative_semantic_consistency": ratio(equivalent, denominator),
                "cumulative_semantic_consistency_percent": pct(equivalent, denominator),
            }
        ranking = sorted(
            model_scores,
            key=lambda model_key: (
                -(model_scores[model_key]["cumulative_semantic_consistency"] or 0.0),
                model_key,
            ),
        )
        subsets[subset_name] = {
            "coverage_selector": selector,
            "contracts": len(benchmark_ids),
            "function_group_denominator": denominator,
            "model_scores": model_scores,
            "ranking": ranking,
        }
    return {
        "subsets": subsets,
        "denominator_note": (
            "Unlike testcase-effectiveness results, ranking sensitivity is cumulative over every "
            "submitted function group in selected contracts. Missing, non-compiling, or unrecovered "
            "functions contribute zero equivalent functions."
        ),
    }


def fmt_int(value: int | float | None) -> str:
    return "n/a" if value is None else f"{int(value):,}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}%"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def render_markdown(results: dict[str, Any]) -> str:
    benchmark = results["benchmark"]
    coverage = benchmark["coverage"]
    lines = [
        "# Semantic-Test Effectiveness",
        "",
        "This analysis uses compact inputs exported from the submitted benchmark snapshot and repaired "
        "semantic replay artifacts. Testcase-effectiveness denominators include only functions actually "
        "replayed for each model. Coverage-threshold rankings instead use the cumulative benchmark "
        "denominator, so missing or unrecovered functions count as failures.",
        "",
        "## Snapshot Checks",
        "",
        f"- Contracts: **{fmt_int(benchmark['contracts'])}**",
        f"- Executable ABI entry groups: **{fmt_int(benchmark['function_groups'])}**",
        f"- Concrete semantic cases: **{fmt_int(benchmark['test_cases'])}**",
        f"- Benchmark SHA-256: `{results['provenance']['benchmark']['sha256']}`",
        "",
        "## Case Composition",
        "",
    ]
    category_rows = []
    for category in CATEGORIES:
        item = benchmark["case_composition"]["by_category"][category]
        category_rows.append(
            (
                category,
                fmt_int(item["cases"]),
                fmt_pct(item["case_share_percent"]),
                fmt_int(item["function_groups_with_category"]),
                fmt_int(item["contracts_with_category"]),
            )
        )
    lines.extend(
        markdown_table(
            ("Category", "Cases", "Share", "ABI entry groups", "Contracts"),
            category_rows,
        )
    )
    lines.extend(
        [
            "",
            "The four category counts sum to 227,383. `boundary` combines all `branch_boundary:*` "
            "variants, and `static` combines all `static_hint:*` variants.",
            "",
            "## Coverage",
            "",
            f"Coverage is marked available for **{coverage['available_contracts']}/600** contracts. "
            f"Line coverage is numerically measurable for **{coverage['line_coverage_measurable_contracts']}**; "
            f"one available record has no line ratio. Coverage is diagnostic and is not used to decide "
            f"semantic equivalence.",
            "",
        ]
    )
    coverage_rows = []
    for metric in ("lines", "functions", "branches"):
        item = coverage["overall"][metric]
        dist = item["per_contract_ratio"]
        coverage_rows.append(
            (
                metric,
                fmt_int(item["measurable_contracts"]),
                f"{fmt_int(item['covered'])}/{fmt_int(item['total'])}",
                fmt_pct(item["pooled_percent"]),
                fmt_pct(pct(dist.get("mean", 0.0), 1.0) if dist.get("count") else None),
                fmt_pct(pct(dist.get("median", 0.0), 1.0) if dist.get("count") else None),
                fmt_pct(pct(dist.get("p10", 0.0), 1.0) if dist.get("count") else None),
            )
        )
    lines.extend(
        markdown_table(
            ("Metric", "Measurable", "Covered/total", "Pooled", "Macro mean", "Median", "P10"),
            coverage_rows,
        )
    )
    reason_text = ", ".join(
        f"{reason}: {count}" for reason, count in coverage["unavailable_reasons"].items()
    )
    lines.extend(
        [
            "",
            "The paper's 77.8% figure is pooled target-contract line coverage "
            "(42,107/54,122), not the per-contract mean (69.6%).",
            "",
            f"Unavailable reasons: {reason_text}.",
            "",
            "## Mismatch Detection by Case Category",
            "",
        ]
    )
    for model_key, model in results["models"].items():
        denominators = model["denominators"]
        lines.extend(
            [
                f"### {model['label']}",
                "",
                f"Denominator: **{fmt_int(denominators['replayed_model_function_evaluations'])}** "
                f"model-function evaluations and **{fmt_int(denominators['replayed_cases'])}** replayed cases; "
                f"**{fmt_int(denominators['failing_model_function_evaluations'])}** functions have at least "
                f"one detected mismatch.",
                "",
            ]
        )
        case_rows = []
        function_rows = []
        for category in CATEGORIES:
            case_item = model["case_mismatch_detection"]["by_category"][category]
            function_item = model["function_detection_contribution"]["by_category"][category]
            case_rows.append(
                (
                    category,
                    fmt_int(case_item["replayed_cases"]),
                    fmt_int(case_item["mismatched_cases"]),
                    fmt_pct(case_item["mismatch_rate_percent"]),
                )
            )
            function_rows.append(
                (
                    category,
                    fmt_int(function_item["replayed_function_evaluations_with_category"]),
                    fmt_int(function_item["failing_function_evaluations_detected"]),
                    fmt_int(function_item["exclusive_detection_removal_loss"]),
                    fmt_int(function_item["additional_detections_beyond_random"]),
                    f"{function_item['order_independent_shapley_credit']:.1f}",
                )
            )
        lines.extend(markdown_table(("Category", "Cases", "Mismatches", "Mismatch rate"), case_rows))
        lines.extend(
            [
                "",
                "Function-level contribution:",
                "",
            ]
        )
        lines.extend(
            markdown_table(
                (
                    "Category",
                    "Functions with cases",
                    "Detected failures",
                    "Exclusive/removal loss",
                    "Beyond random",
                    "1/k credit",
                ),
                function_rows,
            )
        )
        modality = model["mismatch_modalities"]["case_level"]
        modality_counts = modality["cases_by_modality_with_overlaps"]
        lines.extend(
            [
                "",
                "Mismatch modalities overlap. Of "
                f"**{fmt_int(modality['mismatched_cases'])}** mismatched calls: "
                + ", ".join(
                    f"{name}={fmt_int(modality_counts.get(name, 0))}" for name in MODALITIES
                )
                + ".",
                "",
            ]
        )

    lines.extend(
        [
            "## Exact Case-Kind Mismatch Rates",
            "",
        ]
    )
    exact_kind_rows = []
    for model_key, _, _ in MODEL_FILES:
        model = results["models"][model_key]
        for case_kind, item in model["case_mismatch_detection"]["by_case_kind"].items():
            exact_kind_rows.append(
                (
                    model["label"],
                    case_kind,
                    fmt_int(item["replayed_cases"]),
                    fmt_int(item["mismatched_cases"]),
                    fmt_pct(item["mismatch_rate_percent"]),
                )
            )
    lines.extend(
        markdown_table(
            ("Model", "Exact case_kind", "Cases", "Mismatches", "Mismatch rate"),
            exact_kind_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Coverage-Threshold Ranking Sensitivity",
            "",
            "Scores below use all submitted function groups in the selected contracts, including zero "
            "credit for missing or unrecovered functions.",
            "",
        ]
    )
    ranking = results["coverage_threshold_ranking"]["subsets"]
    ranking_rows = []
    for subset_name, _ in COVERAGE_SUBSETS:
        subset = ranking[subset_name]
        scores = subset["model_scores"]
        ranking_rows.append(
            (
                subset_name,
                fmt_int(subset["contracts"]),
                fmt_int(subset["function_group_denominator"]),
                *(
                    fmt_pct(scores[model_key]["cumulative_semantic_consistency_percent"])
                    for model_key, _, _ in MODEL_FILES
                ),
            )
        )
    lines.extend(
        markdown_table(
            (
                "Subset",
                "Contracts",
                "Functions",
                "Opus + repair",
                "GPT + repair",
                "GLM high + repair",
                "GLM instruct + repair",
            ),
            ranking_rows,
        )
    )
    lines.extend(
        [
            "",
            "The ordering is unchanged at every threshold, although Opus and GPT become nearly tied at "
            "the 80% threshold. This sensitivity analysis conditions on measurable source coverage and "
            "does not turn finite testing into a proof of semantic equivalence.",
            "",
            "## Contribution Definitions",
            "",
            "- **Detected failures:** replayed model-function evaluations with at least one mismatching case "
            "from the category.",
            "- **Exclusive/removal loss:** failures detected by that category and no other; removing the "
            "category would make those functions pass all remaining cases.",
            "- **Beyond random:** failures detected by the category where random cases found no mismatch; "
            "augmented categories can overlap.",
            "- **1/k credit:** order-independent attribution. A failing function detected by `k` categories "
            "gives each detecting category `1/k` credit; credits sum to all detected failing functions.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    benchmark_path = resolve_under(root, args.benchmark)
    output_json = resolve_under(root, args.output_json)
    output_md = resolve_under(root, args.output_md)
    replay_paths = {
        model_key: resolve_under(root, Path(relative_path))
        for model_key, _, relative_path in MODEL_FILES
    }

    print(f"[benchmark] {benchmark_path}", flush=True)
    benchmark, case_index, contract_meta = analyze_benchmark(benchmark_path)
    print(
        f"[benchmark] contracts={benchmark['contracts']} groups={benchmark['function_groups']} "
        f"cases={benchmark['test_cases']}",
        flush=True,
    )

    models = {}
    model_equivalent = {}
    for model_key, label, relative_path in MODEL_FILES:
        replay_path = replay_paths[model_key]
        print(f"[model] {label}: {replay_path}", flush=True)
        summary, equivalent = analyze_model(replay_path, case_index)
        summary["label"] = label
        summary["source_path"] = relative_path
        models[model_key] = summary
        model_equivalent[model_key] = equivalent
        print(
            f"[model] {label}: cases={summary['denominators']['replayed_cases']} "
            f"functions={summary['denominators']['replayed_model_function_evaluations']} "
            f"failures={summary['denominators']['failing_model_function_evaluations']}",
            flush=True,
        )

    results = {
        "schema": SCHEMA,
        "provenance": {
            "benchmark": {
                "path": str(args.benchmark),
                "bytes": benchmark_path.stat().st_size,
                "sha256": sha256_file(benchmark_path),
            },
            "replays": {
                model_key: {
                    "path": relative_path,
                    "bytes": replay_paths[model_key].stat().st_size,
                    "sha256": sha256_file(replay_paths[model_key]),
                }
                for model_key, _, relative_path in MODEL_FILES
            },
        },
        "benchmark": benchmark,
        "models": models,
        "coverage_threshold_ranking": ranking_sensitivity(contract_meta, model_equivalent),
        "analysis_scope": {
            "testcase_effectiveness": (
                "Per-model unique (benchmark_id, signature) evaluations actually replayed. "
                "No cross-model pooling or deduplication is performed."
            ),
            "coverage_threshold_ranking": (
                "Cumulative submitted function-group denominator within each contract subset."
            ),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(results), encoding="utf-8")
    print(f"[write] {output_json}", flush=True)
    print(f"[write] {output_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
