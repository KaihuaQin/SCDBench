"""Metric aggregation copied unchanged from the sampling-stability analysis."""
from __future__ import annotations
import collections
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

DIFFICULTIES = ("easy", "medium", "hard")

def load_targets(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a grouped semantic-target JSON")
    return [
        row
        for difficulty_rows in payload.values()
        if isinstance(difficulty_rows, list)
        for row in difficulty_rows
        if isinstance(row, dict)
    ]


def by_id(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        benchmark_id = str(row.get("benchmark_id") or "")
        if not benchmark_id:
            raise ValueError("record is missing benchmark_id")
        if benchmark_id in result:
            raise ValueError(f"duplicate benchmark_id: {benchmark_id}")
        result[benchmark_id] = row
    return result


def test_groups(row: dict[str, Any]) -> list[dict[str, Any]]:
    groups = row.get("test_cases") or []
    if isinstance(groups, dict):
        return [group for group in groups.values() if isinstance(group, dict)]
    if isinstance(groups, list):
        return [group for group in groups if isinstance(group, dict)]
    return []


def difficulty(row: dict[str, Any]) -> str:
    return str(row.get("difficulty") or row.get("difficulty_bin") or "")


def abi_function_count(row: dict[str, Any]) -> int:
    direct = row.get("abi_function_count")
    if direct is not None:
        return int(direct)
    return int((row.get("semantic_metadata") or {}).get("abi_function_count") or 0)


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def harmonic_mean(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def normalized_counter(values: Iterable[Any]) -> dict[str, int]:
    counts = collections.Counter("null" if value is None else str(value) for value in values)
    return dict(sorted(counts.items()))


def perfect_contracts(
    checkpoint: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    replay: dict[str, dict[str, Any]],
) -> list[str]:
    perfect: list[str] = []
    for benchmark_id, checkpoint_row in checkpoint.items():
        ground_truth = {
            str(group.get("signature") or "")
            for group in test_groups(checkpoint_row)
            if group.get("entry_type") == "function"
        }
        evaluation = evaluations.get(benchmark_id)
        replay_row = replay.get(benchmark_id)
        if not ground_truth or evaluation is None or replay_row is None or replay_row.get("error"):
            continue
        abi = evaluation.get("abi") or {}
        if not abi.get("evaluated") or int(abi.get("false_positive") or 0) != 0:
            continue
        if not ground_truth.issubset(set(abi.get("matched_signatures") or [])):
            continue
        function_results = {
            str(item.get("signature") or ""): bool(item.get("equivalent"))
            for item in replay_row.get("function_results") or []
            if isinstance(item, dict)
        }
        if all(function_results.get(signature, False) for signature in ground_truth):
            perfect.append(benchmark_id)
    return sorted(perfect)


def metric_group(
    checkpoint_rows: Sequence[dict[str, Any]],
    evaluation_by_id: dict[str, dict[str, Any]],
    replay_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    benchmark_ids = [str(row["benchmark_id"]) for row in checkpoint_rows]
    evaluations = [evaluation_by_id[benchmark_id] for benchmark_id in benchmark_ids]
    replay_rows = [
        replay_by_id[benchmark_id]
        for benchmark_id in benchmark_ids
        if benchmark_id in replay_by_id and not replay_by_id[benchmark_id].get("error")
    ]
    total = len(checkpoint_rows)
    format_ok = sum(bool((row.get("format") or {}).get("ok")) for row in evaluations)
    compile_ok = sum(bool((row.get("compilation") or {}).get("success")) for row in evaluations)
    abi_tp = sum(int((row.get("abi") or {}).get("true_positive") or 0) for row in evaluations)
    abi_fp = sum(int((row.get("abi") or {}).get("false_positive") or 0) for row in evaluations)
    abi_fn_reached = sum(int((row.get("abi") or {}).get("false_negative") or 0) for row in evaluations)
    abi_ground_truth = sum(abi_function_count(row) for row in checkpoint_rows)
    abi_precision = ratio(abi_tp, abi_tp + abi_fp)
    abi_recall = ratio(abi_tp, abi_ground_truth)
    conditional_abi_recall = ratio(abi_tp, abi_tp + abi_fn_reached)
    semantic_ground_truth = sum(len(test_groups(row)) for row in checkpoint_rows)
    semantic_equivalent = sum(int(row.get("functions_equivalent") or 0) for row in replay_rows)
    semantic_reached = sum(int(row.get("functions_total") or 0) for row in replay_rows)
    return {
        "contracts": total,
        "format_ok": format_ok,
        "format_rate": ratio(format_ok, total),
        "compile_ok": compile_ok,
        "compile_rate": ratio(compile_ok, total),
        "semantic_eligible_contracts": sum(
            bool((row.get("semantic") or {}).get("eligible")) for row in evaluations
        ),
        "replayed_contracts": len(replay_rows),
        "abi": {
            "true_positive": abi_tp,
            "false_positive": abi_fp,
            "ground_truth": abi_ground_truth,
            "precision": abi_precision,
            "recall": abi_recall,
            "f1": harmonic_mean(abi_precision, abi_recall),
            "conditional_recall": conditional_abi_recall,
            "conditional_f1": harmonic_mean(abi_precision, conditional_abi_recall),
        },
        "semantic": {
            "equivalent": semantic_equivalent,
            "ground_truth": semantic_ground_truth,
            "rate": ratio(semantic_equivalent, semantic_ground_truth),
            "reached": semantic_reached,
            "conditional_rate": ratio(semantic_equivalent, semantic_reached),
        },
    }
