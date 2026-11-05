#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASET = Path("../dataset/scdbench.jsonl")

MODEL_RUNS = [
    {
        "slug": "claude_opus47_reasoning_high",
        "prediction": Path("runs/openrouter_claude_opus47_reasoning_high.jsonl"),
        "repair_prediction": Path("runs/openrouter_repair_claude_opus47_reasoning_high.jsonl"),
    },
    {
        "slug": "gpt53_codex_reasoning_high",
        "prediction": Path("runs/openrouter_gpt53_codex_reasoning_high.jsonl"),
        "repair_prediction": Path("runs/openrouter_repair_gpt53_codex_reasoning_high.jsonl"),
    },
    {
        "slug": "glm5_reasoning_high",
        "prediction": Path("runs/openrouter_glm5_reasoning_high.jsonl"),
        "repair_prediction": Path("runs/openrouter_repair_glm5_reasoning_high.jsonl"),
    },
    {
        "slug": "glm5_reasoning_none",
        "prediction": Path("runs/openrouter_glm5_reasoning_none.jsonl"),
        "repair_prediction": Path("runs/openrouter_repair_glm5_reasoning_none.jsonl"),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full format, compilation, and ABI evaluations for SCDBench model outputs."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--compile-timeout", type=int, default=60)
    parser.add_argument("--model", action="append", choices=[run["slug"] for run in MODEL_RUNS])
    parser.add_argument("--original-only", action="store_true")
    parser.add_argument("--repair-only", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args()


def run_eval(
    dataset: Path,
    predictions: Path,
    output: Path,
    summary: Path,
    concurrency: int,
    compile_timeout: int,
    retry_errors: bool,
) -> None:
    cmd = [
        sys.executable,
        "scripts/evaluate_model_outputs.py",
        "--dataset",
        str(dataset),
        "--predictions",
        str(predictions),
        "--output",
        str(output),
        "--summary",
        str(summary),
        "--concurrency",
        str(concurrency),
        "--compile-timeout",
        str(compile_timeout),
    ]
    if retry_errors:
        cmd.append("--retry-errors")
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def selected_runs(args: argparse.Namespace) -> list[dict[str, Path | str]]:
    wanted = set(args.model or [])
    return [run for run in MODEL_RUNS if not wanted or str(run["slug"]) in wanted]


def main() -> None:
    args = parse_args()
    if args.original_only and args.repair_only:
        raise SystemExit("--original-only and --repair-only cannot both be set")

    for run in selected_runs(args):
        slug = str(run["slug"])
        if not args.repair_only:
            output = args.runs_dir / f"eval_first3_{slug}.jsonl"
            summary = args.runs_dir / f"eval_first3_{slug}.summary.json"
            run_eval(
                dataset=args.dataset,
                predictions=Path(run["prediction"]),
                output=output,
                summary=summary,
                concurrency=args.concurrency,
                compile_timeout=args.compile_timeout,
                retry_errors=args.retry_errors,
            )
        if not args.original_only:
            output = args.runs_dir / f"eval_first3_repair_{slug}.jsonl"
            summary = args.runs_dir / f"eval_first3_repair_{slug}.summary.json"
            run_eval(
                dataset=args.dataset,
                predictions=Path(run["repair_prediction"]),
                output=output,
                summary=summary,
                concurrency=args.concurrency,
                compile_timeout=args.compile_timeout,
                retry_errors=args.retry_errors,
            )

    if args.summarize:
        cmd = [
            sys.executable,
            "scripts/summarize_first_three_metrics.py",
            "--dataset",
            str(args.dataset),
            "--runs-dir",
            str(args.runs_dir),
        ]
        print("[run]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
