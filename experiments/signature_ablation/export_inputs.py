#!/usr/bin/env python3
"""Project archived experiment records into the small public analysis inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREDICTION_FIELDS = ("benchmark_id", "content", "difficulty_bin", "finish_reason", "model", "ok", "request_parameters", "response_model", "usage")


def jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_evaluation(row):
    return {
        "benchmark_id": row["benchmark_id"],
        "format": {"ok": bool((row.get("format") or {}).get("ok"))},
        "compilation": {key: (row.get("compilation") or {}).get(key) for key in ("attempted", "success")},
        "abi": {key: (row.get("abi") or {}).get(key) for key in ("evaluated", "true_positive", "false_positive", "false_negative", "matched_signatures")},
        "semantic": {"eligible": bool((row.get("semantic") or {}).get("eligible"))},
        "evaluation_error": bool(row.get("evaluation_error")),
    }


def project_replay(row):
    return {
        "benchmark_id": row["benchmark_id"],
        "error": bool(row.get("error")),
        "functions_total": row.get("functions_total", 0),
        "functions_equivalent": row.get("functions_equivalent", 0),
        "function_results": [{"signature": item["signature"], "equivalent": bool(item.get("equivalent"))} for item in row.get("function_results", [])],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--original-runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    data = args.output_dir / "data"
    runs = args.output_dir / "runs"
    data.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    source_dataset = args.experiment_dir / "generated/scdbench_no_signature_hints_60.jsonl"
    source_rows = list(jsonl(source_dataset))
    ids = {row["benchmark_id"] for row in source_rows}
    assert len(source_rows) == len(ids) == 60
    metadata = []
    for row in source_rows:
        groups = row["test_cases"]
        if isinstance(groups, dict):
            groups = groups.values()
        metadata.append({
            "benchmark_id": row["benchmark_id"],
            "difficulty_bin": row["difficulty_bin"],
            "semantic_metadata": {"abi_function_count": row["semantic_metadata"]["abi_function_count"]},
            "test_cases": [{"signature": group["signature"], "entry_type": group["entry_type"]} for group in groups],
        })
    write_jsonl(data / "subset_metadata.jsonl", metadata)
    sources = {"subset_dataset": source_dataset}
    for stage, suffix in (("initial", ""), ("repaired", "_after_repair")):
        for hints in (False, True):
            arm = "with_hints" if hints else "no_hints"
            label = arm + suffix
            if hints:
                original_suffix = "before" if stage == "initial" else "after_repair"
                evaluation = args.original_runs_dir / f"semantic_prep/eval_semanticprep_gpt53_codex_reasoning_high_{original_suffix}.jsonl"
                replay = args.original_runs_dir / f"semantic/semantic_replay_gpt53_codex_reasoning_high_{original_suffix}.json"
            else:
                evaluation = args.experiment_dir / "runs" / f"{label}_eval.jsonl"
                replay = args.experiment_dir / "runs" / f"{label}_semantic.json"
            write_jsonl(data / f"{label}_eval.jsonl", (project_evaluation(row) for row in jsonl(evaluation) if row["benchmark_id"] in ids))
            payload = json.loads(replay.read_text())
            replay_rows = [project_replay(row) for row in payload["results"] if row["benchmark_id"] in ids]
            (data / f"{label}_semantic.json").write_text(json.dumps({"results": replay_rows}, separators=(",", ":")) + "\n")
            sources[f"{label}_evaluation"] = evaluation
            sources[f"{label}_replay"] = replay
            if not hints:
                target_path = args.experiment_dir / "runs" / f"{label}_targets.json"
                target_payload = json.loads(target_path.read_text())
                targets = {key: [{"benchmark_id": row["benchmark_id"]} for row in rows] for key, rows in target_payload.items() if isinstance(rows, list)}
                (data / f"{label}_targets.json").write_text(json.dumps(targets, separators=(",", ":")) + "\n")
                sources[f"{label}_targets"] = target_path
    for name, source in (
        ("no_hints_predictions.jsonl", args.experiment_dir / "runs/no_hints_predictions.canonical.jsonl"),
        ("no_hints_repair_predictions.jsonl", args.experiment_dir / "runs/no_hints_repair_predictions.jsonl"),
        ("with_hints_predictions.jsonl", args.original_runs_dir / "openrouter_gpt53_codex_reasoning_high.jsonl"),
        ("with_hints_repair_predictions.jsonl", args.original_runs_dir / "openrouter_repair_gpt53_codex_reasoning_high.jsonl"),
    ):
        write_jsonl(runs / name, ({key: row[key] for key in PREDICTION_FIELDS if key in row} for row in jsonl(source) if row["benchmark_id"] in ids))
        sources[name] = source
    manifest = {"schema": "scdbench-signature-ablation-public-inputs-v1", "source_files": {key: {"name": path.name, "sha256": sha256(path)} for key, path in sources.items()}}
    (data / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported paired inputs for {len(ids)} contracts to {args.output_dir}")


if __name__ == "__main__":
    main()
