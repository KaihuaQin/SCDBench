#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable

import metrics


HERE = Path(__file__).resolve().parent
DATASET = HERE / "data/subset_metadata.jsonl"
RUNS = HERE / "runs"

NO_HINT_PATHS = {
    "initial": {
        "evaluation": HERE / "data/no_hints_eval.jsonl",
        "targets": HERE / "data/no_hints_targets.json",
        "replay": HERE / "data/no_hints_semantic.json",
    },
    "repaired": {
        "evaluation": HERE / "data/no_hints_after_repair_eval.jsonl",
        "targets": HERE / "data/no_hints_after_repair_targets.json",
        "replay": HERE / "data/no_hints_after_repair_semantic.json",
    },
}

WITH_HINT_PATHS = {
    "initial": {
        "evaluation": HERE / "data/with_hints_eval.jsonl",
        "replay": HERE / "data/with_hints_semantic.json",
    },
    "repaired": {
        "evaluation": HERE / "data/with_hints_after_repair_eval.jsonl",
        "replay": HERE / "data/with_hints_after_repair_semantic.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and summarize the 60-contract signature-hint ablation."
    )
    parser.add_argument("--input-dir", type=Path, default=HERE,
                        help="Directory containing data/ and runs/ for this experiment.")
    parser.add_argument("--output-json", type=Path, default=HERE / "results/results.json")
    parser.add_argument("--output-md", type=Path, default=HERE / "results/RESULTS.md")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                yield json.loads(raw)


def load_replay(path: Path, selected_ids: set[str]) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("results") or []
        if isinstance(row, dict) and str(row.get("benchmark_id") or "") in selected_ids
    ]
    return metrics.by_id(rows)


def load_evaluations(
    path: Path, selected_ids: set[str]
) -> dict[str, dict[str, Any]]:
    return metrics.by_id(
        row
        for row in read_jsonl(path)
        if str(row.get("benchmark_id") or "") in selected_ids
    )


def repair_candidates(evaluations: dict[str, dict[str, Any]]) -> set[str]:
    return {
        benchmark_id
        for benchmark_id, row in evaluations.items()
        if bool((row.get("format") or {}).get("ok"))
        and bool((row.get("compilation") or {}).get("attempted"))
        and not bool((row.get("compilation") or {}).get("success"))
    }


def preserve_initial_format(
    initial: dict[str, dict[str, Any]],
    repaired: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if set(initial) != set(repaired):
        raise ValueError("initial and repaired evaluation ID sets differ")
    result: dict[str, dict[str, Any]] = {}
    for benchmark_id, repaired_row in repaired.items():
        row = dict(repaired_row)
        row["format"] = initial[benchmark_id].get("format")
        result[benchmark_id] = row
    return result


def load_targets(path: Path) -> dict[str, dict[str, Any]]:
    return metrics.by_id(metrics.load_targets(path))


def validate_stage(
    label: str,
    selected_ids: set[str],
    evaluations: dict[str, dict[str, Any]],
    replay: dict[str, dict[str, Any]],
    targets: dict[str, dict[str, Any]] | None,
) -> None:
    if set(evaluations) != selected_ids:
        raise ValueError(f"{label}: evaluation IDs do not match the selected subset")
    if any(row.get("evaluation_error") for row in evaluations.values()):
        raise ValueError(f"{label}: evaluation contains errors")
    eligible_ids = {
        benchmark_id
        for benchmark_id, row in evaluations.items()
        if bool((row.get("semantic") or {}).get("eligible"))
    }
    if set(replay) != eligible_ids:
        raise ValueError(f"{label}: replay IDs do not match semantic-eligible IDs")
    if any(row.get("error") for row in replay.values()):
        raise ValueError(f"{label}: semantic replay contains errors")
    if targets is not None and set(targets) != eligible_ids:
        raise ValueError(f"{label}: target IDs do not match semantic-eligible IDs")


def stage_summary(
    dataset_rows: list[dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    replay: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checkpoint = metrics.by_id(dataset_rows)
    overall = metrics.metric_group(dataset_rows, evaluations, replay)
    by_difficulty = {
        difficulty: metrics.metric_group(
            [
                row
                for row in dataset_rows
                if str(row.get("difficulty_bin") or "") == difficulty
            ],
            evaluations,
            replay,
        )
        for difficulty in metrics.DIFFICULTIES
    }
    perfect_ids = metrics.perfect_contracts(checkpoint, evaluations, replay)
    perfect_by_difficulty = collections.Counter(
        str(checkpoint[benchmark_id].get("difficulty_bin") or "")
        for benchmark_id in perfect_ids
    )
    return {
        "overall": overall,
        "by_difficulty": by_difficulty,
        "perfect": {
            "contracts": len(perfect_ids),
            "by_difficulty": {
                difficulty: perfect_by_difficulty[difficulty]
                for difficulty in metrics.DIFFICULTIES
            },
            "benchmark_ids": perfect_ids,
        },
    }


def cost_summary(path: Path, selected_ids: set[str] | None = None) -> dict[str, Any]:
    rows = [
        row
        for row in read_jsonl(path)
        if selected_ids is None
        or str(row.get("benchmark_id") or "") in selected_ids
    ]
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[str(row.get("benchmark_id") or "")] = row
    successful_rows = [row for row in rows if row.get("ok") is True]
    return {
        "raw_records": len(rows),
        "unique_contracts": len(latest),
        "latest_successes": sum(row.get("ok") is True for row in latest.values()),
        "transport_failure_records": sum(row.get("ok") is not True for row in rows),
        "response_reported_cost_usd": sum(
            float((row.get("usage") or {}).get("cost") or 0.0)
            for row in successful_rows
        ),
        "prompt_tokens": sum(
            int((row.get("usage") or {}).get("prompt_tokens") or 0)
            for row in successful_rows
        ),
        "completion_tokens": sum(
            int((row.get("usage") or {}).get("completion_tokens") or 0)
            for row in successful_rows
        ),
        "response_models": metrics.normalized_counter(
            row.get("response_model") for row in latest.values()
        ),
    }


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def stage_row(label: str, stage: dict[str, Any]) -> str:
    overall = stage["overall"]
    abi = overall["abi"]
    semantic = overall["semantic"]
    return (
        f"| {label} | "
        f"{overall['format_ok']}/60 ({pct(overall['format_rate'])}) | "
        f"{overall['compile_ok']}/60 ({pct(overall['compile_rate'])}) | "
        f"{abi['precision']:.3f}/{abi['recall']:.3f}/{abi['f1']:.3f} | "
        f"{semantic['equivalent']}/{semantic['ground_truth']} "
        f"({pct(semantic['rate'])}) | "
        f"{stage['perfect']['contracts']}/60 |"
    )


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# No-External-Signature-Hints Ablation",
        "",
        "The same 60 submitted-benchmark contracts are used in every row. The no-hint "
        "intervention removes only external selector/topic-to-text mappings. Selectors, "
        "event topics, assembly, checkpoints, and all other inputs remain unchanged.",
        "",
        "| Setting | Format | Compile | ABI P/R/F1 | Semantic | Perfect |",
        "|---|---:|---:|---:|---:|---:|",
        stage_row("With hints, initial", result["with_hints"]["initial"]),
        stage_row("Without hints, initial", result["without_hints"]["initial"]),
        stage_row("With hints, repaired", result["with_hints"]["repaired"]),
        stage_row("Without hints, repaired", result["without_hints"]["repaired"]),
        "",
        "All metrics are cumulative over the original 60 contracts and their callable "
        "entries. Repaired rows preserve the initial format-stage result, matching the "
        "paper protocol.",
        "",
        "## No-Hint Results by Complexity Stratum",
        "",
        "| Stage | Difficulty | Format | Compile | ABI F1 | Semantic |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for stage_name, stage_label in (("initial", "Initial"), ("repaired", "Repaired")):
        stage = result["without_hints"][stage_name]
        for difficulty in metrics.DIFFICULTIES:
            item = stage["by_difficulty"][difficulty]
            lines.append(
                f"| {stage_label} | {difficulty.title()} | "
                f"{item['format_ok']}/20 | {item['compile_ok']}/20 | "
                f"{item['abi']['f1']:.3f} | "
                f"{item['semantic']['equivalent']}/{item['semantic']['ground_truth']} "
                f"({pct(item['semantic']['rate'])}) |"
            )

    initial_delta = result["without_hints_minus_with_hints"]["initial"]
    repaired_delta = result["without_hints_minus_with_hints"]["repaired"]
    cost = result["without_hints"]["cost"]
    lines.extend(
        [
            "",
            "## Main Result",
            "",
            f"Removing external mappings changes cumulative ABI F1 by "
            f"{initial_delta['abi_f1']:+.3f} before repair and "
            f"{repaired_delta['abi_f1']:+.3f} after repair. Semantic consistency "
            f"changes by {initial_delta['semantic_rate_pp']:+.1f} and "
            f"{repaired_delta['semantic_rate_pp']:+.1f} percentage points, respectively.",
            "",
            f"The no-hint arm cost ${cost['initial']['response_reported_cost_usd']:.4f} "
            f"for 60 initial responses and ${cost['repair']['response_reported_cost_usd']:.4f} "
            f"for {result['without_hints']['repair']['candidates']} repairs.",
            "",
            "## Interpretation Limit",
            "",
            "The no-hint calls were run later against the undated model alias, while the "
            "stored with-hint calls report a dated snapshot and include two selected "
            "contracts without successful API responses. The large ABI difference is useful "
            "sensitivity evidence, but this historical-control comparison is not a clean "
            "contemporaneous causal estimate.",
            "",
        ]
    )
    return "\n".join(lines)


def delta(without: dict[str, Any], with_hints: dict[str, Any]) -> dict[str, float]:
    return {
        "format_rate_pp": 100
        * (without["overall"]["format_rate"] - with_hints["overall"]["format_rate"]),
        "compile_rate_pp": 100
        * (without["overall"]["compile_rate"] - with_hints["overall"]["compile_rate"]),
        "abi_precision": without["overall"]["abi"]["precision"]
        - with_hints["overall"]["abi"]["precision"],
        "abi_recall": without["overall"]["abi"]["recall"]
        - with_hints["overall"]["abi"]["recall"],
        "abi_f1": without["overall"]["abi"]["f1"]
        - with_hints["overall"]["abi"]["f1"],
        "semantic_rate_pp": 100
        * (
            without["overall"]["semantic"]["rate"]
            - with_hints["overall"]["semantic"]["rate"]
        ),
    }


def main() -> int:
    args = parse_args()
    global RUNS
    RUNS = args.input_dir / "runs"
    def input_path(path: Path) -> Path:
        return args.input_dir / path.relative_to(HERE)
    dataset_rows = list(read_jsonl(input_path(DATASET)))
    selected_ids = {str(row["benchmark_id"]) for row in dataset_rows}
    if len(dataset_rows) != 60 or len(selected_ids) != 60:
        raise ValueError("the ablation dataset must contain 60 unique contracts")

    without_evaluations: dict[str, dict[str, dict[str, Any]]] = {}
    without_replays: dict[str, dict[str, dict[str, Any]]] = {}
    for stage_name, paths in NO_HINT_PATHS.items():
        paths = {name: input_path(path) for name, path in paths.items()}
        evaluations = load_evaluations(paths["evaluation"], selected_ids)
        replay = load_replay(paths["replay"], selected_ids)
        targets = load_targets(paths["targets"])
        validate_stage(
            f"without_hints.{stage_name}",
            selected_ids,
            evaluations,
            replay,
            targets,
        )
        without_evaluations[stage_name] = evaluations
        without_replays[stage_name] = replay

    with_evaluations: dict[str, dict[str, dict[str, Any]]] = {}
    with_replays: dict[str, dict[str, dict[str, Any]]] = {}
    for stage_name, paths in WITH_HINT_PATHS.items():
        paths = {name: input_path(path) for name, path in paths.items()}
        evaluations = load_evaluations(paths["evaluation"], selected_ids)
        replay = load_replay(paths["replay"], selected_ids)
        validate_stage(
            f"with_hints.{stage_name}",
            selected_ids,
            evaluations,
            replay,
            None,
        )
        with_evaluations[stage_name] = evaluations
        with_replays[stage_name] = replay

    without_protocol_repaired = preserve_initial_format(
        without_evaluations["initial"], without_evaluations["repaired"]
    )
    with_protocol_repaired = preserve_initial_format(
        with_evaluations["initial"], with_evaluations["repaired"]
    )

    without_initial = stage_summary(
        dataset_rows, without_evaluations["initial"], without_replays["initial"]
    )
    without_repaired = stage_summary(
        dataset_rows, without_protocol_repaired, without_replays["repaired"]
    )
    with_initial = stage_summary(
        dataset_rows, with_evaluations["initial"], with_replays["initial"]
    )
    with_repaired = stage_summary(
        dataset_rows, with_protocol_repaired, with_replays["repaired"]
    )

    without_repair_ids = repair_candidates(without_evaluations["initial"])
    with_repair_ids = repair_candidates(with_evaluations["initial"])
    result = {
        "schema": "scdbench-no-external-signature-hints-ablation-v1",
        "design": {
            "contracts": 60,
            "difficulty_bins": {"easy": 20, "medium": 20, "hard": 20},
            "intervention": (
                "Remove top-level external function-selector and event-topic "
                "candidate mappings only."
            ),
            "unchanged_inputs": (
                "EVM assembly, intrinsic selectors/topics, system prompt, decoding "
                "settings, ground truth, and semantic checkpoints."
            ),
            "model": "openai/gpt-5.3-codex",
            "reasoning_effort": "high",
        },
        "without_hints": {
            "initial": without_initial,
            "repaired": without_repaired,
            "repair": {
                "candidates": len(without_repair_ids),
                "compilation_fixed": sum(
                    bool(
                        (
                            without_evaluations["repaired"][benchmark_id].get(
                                "compilation"
                            )
                            or {}
                        ).get("success")
                    )
                    for benchmark_id in without_repair_ids
                ),
                "candidate_ids": sorted(without_repair_ids),
            },
            "cost": {
                "initial": cost_summary(RUNS / "no_hints_predictions.jsonl"),
                "repair": cost_summary(
                    RUNS / "no_hints_repair_predictions.jsonl"
                ),
            },
        },
        "with_hints": {
            "initial": with_initial,
            "repaired": with_repaired,
            "repair": {
                "candidates": len(with_repair_ids),
                "compilation_fixed": sum(
                    bool(
                        (
                            with_evaluations["repaired"][benchmark_id].get(
                                "compilation"
                            )
                            or {}
                        ).get("success")
                    )
                    for benchmark_id in with_repair_ids
                ),
            },
            "provenance": {
                "initial_responses": cost_summary(
                    RUNS / "with_hints_predictions.jsonl",
                    selected_ids,
                )
            },
        },
        "without_hints_minus_with_hints": {
            "initial": delta(without_initial, with_initial),
            "repaired": delta(without_repaired, with_repaired),
        },
        "validation": {
            "all_evaluation_id_sets_exact": True,
            "all_target_id_sets_exact": True,
            "all_replay_id_sets_exact": True,
            "all_semantic_replays_error_free": True,
            "paper_cumulative_denominators": {
                "contracts": 60,
                "abi_functions": without_initial["overall"]["abi"]["ground_truth"],
                "semantic_functions": without_initial["overall"]["semantic"][
                    "ground_truth"
                ],
            },
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown(result), encoding="utf-8")
    print(f"[results] wrote {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
