"""Package stored second-sample artifacts without running models or replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIGS = {
    "claude_opus47_high": "claude_opus47_reasoning_high",
    "gpt53_codex_high": "gpt53_codex_reasoning_high",
    "glm5_high": "glm5_reasoning_high",
    "glm5_instruct": "glm5_reasoning_none",
}
RESPONSE_FIELDS = (
    "benchmark_id", "content", "difficulty_bin", "finish_reason", "model", "ok",
    "request_parameters", "response_model", "usage", "compilation_error",
    "source_compiler_json", "source_content_sha256", "source_contract_name",
    "source_model", "source_response_model",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def project_large_json(path, expression):
    result = subprocess.run(["jq", "-c", expression, str(path)], check=True,
                            capture_output=True, text=True)
    return json.loads(result.stdout)


def scrub_paths(value):
    if isinstance(value, dict):
        return {key: scrub_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_paths(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"/(?:Users|home|tmp|private|workspace|root)/[^\s:\"'<>]+",
                      "<local-path>", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Original experiment workspace, needed only to recreate this release.")
    args = parser.parse_args()
    source = args.source_root.resolve()
    artifacts = source / "neurips_rebuttal/experiments/sample_stability/artifacts"
    llm = artifacts / "llm_rerun"
    output = HERE / "data"
    manifest = {"inputs": {}, "outputs": {}}

    def record(path):
        manifest["inputs"][str(path.relative_to(source))] = sha256(path)

    checkpoints = artifacts / "holdout_checkpoints.generated.json"
    metadata = project_large_json(checkpoints, '''{
      fork_block, results: [.results[] | {
        benchmark_id, difficulty, abi_function_count, calls_total,
        test_cases: [.test_cases[] | {signature,entry_type,state,case_count}]
      }]
    }''')
    record(checkpoints)
    write_json(output / "contracts.json", metadata)

    for config, slug in CONFIGS.items():
        for repair in (False, True):
            filename = f"openrouter_{'repair_' if repair else ''}{slug}.jsonl"
            path = llm / ("raw" if repair else "canonical") / filename
            latest = {}
            for row in read_jsonl(path):
                latest[row["benchmark_id"]] = row
            if not repair and len(latest) != 600:
                raise ValueError(f"Incomplete initial responses: {config}")
            if any(row.get("ok") is not True for row in latest.values()):
                raise ValueError(f"Unresolved transport failure: {filename}")
            rows = []
            for row in latest.values():
                clean = {key: row[key] for key in RESPONSE_FIELDS if key in row}
                if "compilation_error" in clean:
                    clean["compilation_error"] = scrub_paths(clean["compilation_error"])
                rows.append(clean)
            write_jsonl(ROOT / "runs/second_sample" / filename, rows)
            record(path)

        for variant, suffix in (("initial", Path()), ("repaired", Path("after_repair"))):
            directory = llm / "evaluation" / config / suffix
            evaluation_path = directory / "eval.jsonl"
            evaluations = []
            for row in read_jsonl(evaluation_path):
                if row.get("evaluation_error"):
                    raise ValueError(f"Evaluation error: {row['benchmark_id']}")
                evaluations.append(scrub_paths({key: row[key] for key in (
                    "benchmark_id", "format", "compilation", "abi", "semantic"
                ) if key in row}))
            write_jsonl(output / config / f"{variant}.evaluation.jsonl", evaluations)
            record(evaluation_path)
            replay_path = directory / "semantic_replay.json"
            replay = project_large_json(replay_path, '''{results: [.results[] | {
              benchmark_id, difficulty, functions_equivalent, functions_total,
              calls_total, calls_matched, function_results, error
            }]}''')
            if any(row.get("error") for row in replay["results"]):
                raise ValueError(f"Replay error: {config}/{variant}")
            write_json(output / config / f"{variant}.replay.json", replay)
            record(replay_path)
        print(f"Packaged {config}", flush=True)

    initial_path = llm / "evaluation/comparison_to_submission.json"
    repaired_path = llm / "evaluation/repaired_comparison_to_submission.json"
    initial = json.loads(initial_path.read_text())["configurations"]
    repaired = json.loads(repaired_path.read_text())["configurations"]
    reference = {name: {
        "initial": {key: initial[name]["holdout"][key]
                    for key in ("overall", "by_difficulty", "perfect")},
        "repaired": repaired[name]["holdout_after_repair"],
    } for name in CONFIGS}
    write_json(output / "reference_metrics.json", reference)
    record(initial_path)
    record(repaired_path)
    for directory in (output, ROOT / "runs/second_sample"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                manifest["outputs"][str(path.relative_to(ROOT))] = sha256(path)
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
