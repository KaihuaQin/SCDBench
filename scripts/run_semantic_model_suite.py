#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("../dataset/scdbench.jsonl")
DEFAULT_TESTCASES = Path("../dataset/scdbench.jsonl")
DEFAULT_RUNS_DIR = Path("runs")

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                item = json.loads(raw)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def dedupe_latest(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        bid = str(row.get("benchmark_id") or "")
        if bid:
            out[bid] = row
    return out


def make_after_repair_predictions(original_path: Path, repair_path: Path, output_path: Path) -> Path:
    original_rows = read_jsonl(original_path)
    repair_by_id = dedupe_latest(read_jsonl(repair_path))
    combined = [repair_by_id.get(str(row.get("benchmark_id") or ""), row) for row in original_rows]
    write_jsonl(output_path, combined)
    return output_path


def count_targets(path: Path) -> int:
    if not path.exists():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return 0
    return sum(len(items) for items in payload.values() if isinstance(items, list))


def run_prepare(
    dataset: Path,
    predictions: Path,
    eval_output: Path,
    summary_output: Path,
    targets_output: Path,
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
        str(eval_output),
        "--summary",
        str(summary_output),
        "--fuzz-targets",
        str(targets_output),
        "--concurrency",
        str(concurrency),
        "--compile-timeout",
        str(compile_timeout),
    ]
    if retry_errors:
        cmd.append("--retry-errors")
    print("[prepare]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"[prepare] targets={count_targets(targets_output)} path={targets_output}", flush=True)


def run_replay(
    targets: Path,
    testcases: Path,
    output: Path,
    retry_errors: bool,
    matched_functions_only: bool,
    job_id: str,
    log_path: Path | None = None,
) -> None:
    cmd = [
        sys.executable,
        "scripts/run_foundry_semantic.py",
        "replay",
        "--targets",
        str(targets),
        "--testcases",
        str(testcases),
        "--output",
        str(output),
        "--fork-block",
        "24136053",
    ]
    if retry_errors:
        cmd.append("--retry-errors")
    if matched_functions_only:
        cmd.append("--matched-functions-only")
    env = os.environ.copy()
    safe_job_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in job_id)
    env["FOUNDRY_OUT"] = f"out_{safe_job_id}"
    env["FOUNDRY_CACHE_PATH"] = f"cache_{safe_job_id}"
    print("[replay]", job_id, " ".join(cmd), flush=True)
    if log_path is None:
        subprocess.run(cmd, check=True, env=env)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[replay] {job_id} {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def selected_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    wanted = set(args.model or [])
    return [run for run in MODEL_RUNS if not wanted or run["slug"] in wanted]


def run_parallel(label: str, jobs: list[tuple[str, Any]], workers: int) -> None:
    if not jobs:
        return
    if workers <= 1:
        for job_id, fn in jobs:
            print(f"[{label}] start {job_id}", flush=True)
            fn()
            print(f"[{label}] done {job_id}", flush=True)
        return
    print(f"[{label}] launching {len(jobs)} job(s) with workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {executor.submit(fn): job_id for job_id, fn in jobs}
        for future in as_completed(future_to_id):
            job_id = future_to_id[future]
            future.result()
            print(f"[{label}] done {job_id}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ABI-matched semantic targets and replay them for SCDBench model outputs."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--testcases", type=Path, default=DEFAULT_TESTCASES)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--model", action="append", choices=[run["slug"] for run in MODEL_RUNS])
    parser.add_argument("--before-only", action="store_true")
    parser.add_argument("--after-only", action="store_true")
    parser.add_argument("--prepare-concurrency", type=int, default=10)
    parser.add_argument("--prepare-jobs", type=int, default=1)
    parser.add_argument("--replay-jobs", type=int, default=1)
    parser.add_argument("--compile-timeout", type=int, default=60)
    parser.add_argument("--retry-prepare-errors", action="store_true")
    parser.add_argument("--retry-replay-errors", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--matched-functions-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.before_only and args.after_only:
        raise SystemExit("--before-only and --after-only cannot both be set")

    inputs_dir = args.runs_dir / "semantic_inputs"
    prep_dir = args.runs_dir / "semantic_prep"
    targets_dir = args.runs_dir / "semantic_targets"
    output_dir = args.runs_dir / "semantic"
    log_dir = args.runs_dir / "semantic_logs"
    for directory in (inputs_dir, prep_dir, targets_dir, output_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    variant_jobs: list[dict[str, Any]] = []
    for run in selected_runs(args):
        slug = str(run["slug"])
        prediction = args.runs_dir / Path(run["prediction"]).name
        repair_prediction = args.runs_dir / Path(run["repair_prediction"]).name
        variants: list[tuple[str, Path]] = []
        if not args.after_only:
            variants.append(("before", prediction))
        if not args.before_only:
            after_path = make_after_repair_predictions(
                prediction,
                repair_prediction,
                inputs_dir / f"openrouter_after_repair_{slug}.jsonl",
            )
            variants.append(("after_repair", after_path))

        for variant, prediction_path in variants:
            eval_output = prep_dir / f"eval_semanticprep_{slug}_{variant}.jsonl"
            summary_output = prep_dir / f"eval_semanticprep_{slug}_{variant}.summary.json"
            targets_output = targets_dir / f"semantic_targets_{slug}_{variant}.json"
            replay_output = output_dir / f"semantic_replay_{slug}_{variant}.json"
            variant_jobs.append(
                {
                    "job_id": f"{slug}_{variant}",
                    "prediction_path": prediction_path,
                    "eval_output": eval_output,
                    "summary_output": summary_output,
                    "targets_output": targets_output,
                    "replay_output": replay_output,
                }
            )

    if not args.skip_prepare:
        prepare_jobs = []
        for job in variant_jobs:
            prepare_jobs.append(
                (
                    str(job["job_id"]),
                    lambda job=job: run_prepare(
                        dataset=args.dataset,
                        predictions=job["prediction_path"],
                        eval_output=job["eval_output"],
                        summary_output=job["summary_output"],
                        targets_output=job["targets_output"],
                        concurrency=args.prepare_concurrency,
                        compile_timeout=args.compile_timeout,
                        retry_errors=args.retry_prepare_errors,
                    ),
                )
            )
        run_parallel("prepare", prepare_jobs, args.prepare_jobs)

    if not args.skip_replay:
        replay_jobs = []
        for job in variant_jobs:
            replay_jobs.append(
                (
                    str(job["job_id"]),
                    lambda job=job: run_replay(
                        targets=job["targets_output"],
                        testcases=args.testcases,
                        output=job["replay_output"],
                        retry_errors=args.retry_replay_errors,
                        matched_functions_only=args.matched_functions_only,
                        job_id=job["job_id"],
                        log_path=log_dir / f"{job['job_id']}.log",
                    ),
                )
            )
        run_parallel("replay", replay_jobs, args.replay_jobs)


if __name__ == "__main__":
    main()
