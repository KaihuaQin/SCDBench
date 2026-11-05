"""Metric functions from the original second-sample analysis, unchanged."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

EVALUATION_FIELDS = ("format", "compilation", "abi", "semantic")

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


def repair_candidates(
    evaluations: dict[str, dict[str, Any]],
) -> set[str]:
    return {
        benchmark_id
        for benchmark_id, row in evaluations.items()
        if bool((row.get("format") or {}).get("ok"))
        and bool((row.get("compilation") or {}).get("attempted"))
        and not bool((row.get("compilation") or {}).get("success"))
    }


def _evaluation_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in EVALUATION_FIELDS}


def combine_protocol_evaluations(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    repair_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if set(after) != set(before):
        missing = sorted(set(before) - set(after))
        extra = sorted(set(after) - set(before))
        raise ValueError(
            "before/after evaluation ID mismatch: "
            f"missing_after={missing[:5]}, extra_after={extra[:5]}"
        )

    expected_repair_ids = repair_candidates(before)
    if repair_ids != expected_repair_ids:
        missing = sorted(expected_repair_ids - repair_ids)
        extra = sorted(repair_ids - expected_repair_ids)
        raise ValueError(
            "repair response IDs do not exactly match initial format-pass, "
            "compilation-attempted failures: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    unchanged_ids = set(before) - repair_ids
    changed_without_repair = [
        benchmark_id
        for benchmark_id in sorted(unchanged_ids)
        if _evaluation_projection(before[benchmark_id])
        != _evaluation_projection(after[benchmark_id])
    ]
    if changed_without_repair:
        raise ValueError(
            "after-repair evaluation changed contracts without a repair attempt: "
            f"{changed_without_repair[:5]}"
        )

    combined: dict[str, dict[str, Any]] = {}
    repaired_stage1_status_changed = 0
    for benchmark_id, before_row in before.items():
        if benchmark_id not in repair_ids:
            combined[benchmark_id] = before_row
            continue
        combined_row = dict(after[benchmark_id])
        if bool((combined_row.get("format") or {}).get("ok")) != bool(
            (before_row.get("format") or {}).get("ok")
        ):
            repaired_stage1_status_changed += 1
        # Stage 1 evaluates the original decompilation response. A compilation
        # repair may alter later stages but cannot revoke that eligibility.
        combined_row["format"] = before_row.get("format")
        combined[benchmark_id] = combined_row

    return combined, {
        "contracts": len(combined),
        "repair_candidates": len(repair_ids),
        "unrepaired_contracts_verified_unchanged": len(unchanged_ids),
        "repair_rows_whose_re_evaluated_format_status_differed": (
            repaired_stage1_status_changed
        ),
        "format_stage_policy": (
            "Original stage-1 format result is preserved for every repair "
            "candidate; stages 2-4 use the after-repair artifact."
        ),
    }
