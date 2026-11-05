#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent / "dataset/scdbench.jsonl"

RUNS = [
    (
        "Opus 4.7",
        ROOT / "runs/semantic_prep/eval_semanticprep_claude_opus47_reasoning_high_before.jsonl",
        ROOT / "runs/semantic/semantic_replay_claude_opus47_reasoning_high_before.json",
    ),
    (
        "Opus 4.7$^\\dagger$",
        ROOT / "runs/semantic_prep/eval_semanticprep_claude_opus47_reasoning_high_after_repair.jsonl",
        ROOT / "runs/semantic/semantic_replay_claude_opus47_reasoning_high_after_repair.json",
    ),
    (
        "GPT-5.3-Codex",
        ROOT / "runs/semantic_prep/eval_semanticprep_gpt53_codex_reasoning_high_before.jsonl",
        ROOT / "runs/semantic/semantic_replay_gpt53_codex_reasoning_high_before.json",
    ),
    (
        "GPT-5.3-Codex$^\\dagger$",
        ROOT / "runs/semantic_prep/eval_semanticprep_gpt53_codex_reasoning_high_after_repair.jsonl",
        ROOT / "runs/semantic/semantic_replay_gpt53_codex_reasoning_high_after_repair.json",
    ),
    (
        "GLM-5",
        ROOT / "runs/semantic_prep/eval_semanticprep_glm5_reasoning_high_before.jsonl",
        ROOT / "runs/semantic/semantic_replay_glm5_reasoning_high_before.json",
    ),
    (
        "GLM-5$^\\dagger$",
        ROOT / "runs/semantic_prep/eval_semanticprep_glm5_reasoning_high_after_repair.jsonl",
        ROOT / "runs/semantic/semantic_replay_glm5_reasoning_high_after_repair.json",
    ),
    (
        "GLM-5 (instruct)",
        ROOT / "runs/semantic_prep/eval_semanticprep_glm5_reasoning_none_before.jsonl",
        ROOT / "runs/semantic/semantic_replay_glm5_reasoning_none_before.json",
    ),
    (
        "GLM-5 (instruct)$^\\dagger$",
        ROOT / "runs/semantic_prep/eval_semanticprep_glm5_reasoning_none_after_repair.jsonl",
        ROOT / "runs/semantic/semantic_replay_glm5_reasoning_none_after_repair.json",
    ),
]

DIFFICULTIES = ("easy", "medium", "hard")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                yield json.loads(raw)


def load_dataset_functions(dataset_path: Path = DATASET) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(dataset_path):
        functions = {
            str(entry.get("signature") or signature)
            for signature, entry in (row.get("test_cases") or {}).items()
            if isinstance(entry, dict) and entry.get("entry_type") == "function"
        }
        out[str(row["benchmark_id"])] = {
            "difficulty": str(row.get("difficulty_bin") or ""),
            "functions": functions,
        }
    return out


def load_prep(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["benchmark_id"]): row for row in read_jsonl(path)}


def load_replay(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["benchmark_id"]): row for row in payload.get("results", [])}


def is_perfect_contract(
    benchmark_id: str,
    dataset_info: dict[str, Any],
    prep_row: dict[str, Any] | None,
    replay_row: dict[str, Any] | None,
) -> bool:
    gt_functions = set(dataset_info["functions"])
    if not gt_functions:
        return False
    if not prep_row or not replay_row:
        return False

    abi = prep_row.get("abi") or {}
    if not abi.get("evaluated"):
        return False
    if int(abi.get("false_positive") or 0) != 0:
        return False

    matched_signatures = set(abi.get("matched_signatures") or [])
    if not gt_functions.issubset(matched_signatures):
        return False

    function_results = {
        str(item.get("signature")): bool(item.get("equivalent"))
        for item in replay_row.get("function_results") or []
        if isinstance(item, dict)
    }
    return all(function_results.get(signature, False) for signature in gt_functions)


def summarize(dataset_path: Path = DATASET, runs_dir: Path = ROOT / "runs") -> dict[str, Any]:
    dataset = load_dataset_functions(dataset_path)
    out: dict[str, Any] = {}
    for label, prep_path, replay_path in RUNS:
        prep_path = runs_dir / prep_path.relative_to(ROOT / "runs")
        replay_path = runs_dir / replay_path.relative_to(ROOT / "runs")
        prep = load_prep(prep_path)
        replay = load_replay(replay_path)
        counts = {difficulty: 0 for difficulty in DIFFICULTIES}
        perfect_ids = []
        for benchmark_id, info in dataset.items():
            if is_perfect_contract(benchmark_id, info, prep.get(benchmark_id), replay.get(benchmark_id)):
                counts[info["difficulty"]] += 1
                perfect_ids.append(benchmark_id)
        out[label] = {
            "easy": counts["easy"],
            "medium": counts["medium"],
            "hard": counts["hard"],
            "total": sum(counts.values()),
            "perfect_ids": perfect_ids,
        }
    return out


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Perfect Semantic Contract Recovery",
        "",
        "A contract is counted as perfect when all benchmarked public functions are recovered, every recovered function passes all semantic replay cases, and no spurious function is recovered.",
        "",
        "| Model | Easy | Medium | Hard | Total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model, counts in summary.items():
        lines.append(
            f"| {model} | {counts['easy']}/200 | {counts['medium']}/200 | {counts['hard']}/200 | {counts['total']}/600 |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize perfect recovery for either 600-contract sample.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    summary = summarize(args.dataset, args.runs_dir)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.runs_dir / "perfect_semantic_contracts_summary.json"
    md_path = args.runs_dir / "perfect_semantic_contracts_summary.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, md_path)
    print(json.dumps({"json": str(json_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
