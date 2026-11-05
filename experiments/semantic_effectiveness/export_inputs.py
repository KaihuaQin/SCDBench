#!/usr/bin/env python3
"""Export compact, auditable inputs for semantic-effectiveness analysis."""

from __future__ import annotations

import argparse
import gzip
import io
import json
from contextlib import contextmanager
from pathlib import Path

import analyze


@contextmanager
def gzip_text(path: Path):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                yield handle


def project_benchmark(source: Path, destination: Path) -> None:
    with source.open(encoding="utf-8") as rows, gzip_text(destination) as output:
        for raw in rows:
            row = json.loads(raw)
            coverage = row.get("semantic_coverage") or {}
            groups = {}
            for signature, group in row["test_cases"].items():
                cases = [{key: case[key] for key in ("case_id", "case_kind", "signature", "state") if key in case}
                         for case in group["cases"]]
                groups[signature] = {key: group[key] for key in ("entry_type", "state", "case_count") if key in group}
                groups[signature]["cases"] = cases
            record = {
                "benchmark_id": row["benchmark_id"],
                "difficulty_bin": row["difficulty_bin"],
                "semantic_coverage": {key: coverage[key] for key in
                    ("available", "kind", "reason", "lines", "functions", "branches") if key in coverage},
                "test_cases": groups,
            }
            output.write(json.dumps(record, separators=(",", ":")) + "\n")


def project_replay(source: Path, destination: Path, case_index: dict) -> None:
    with gzip_text(destination) as output:
        output.write('{"results":[\n')
        first = True
        for result in analyze.iter_json_array(source):
            projected = {key: result[key] for key in
                ("benchmark_id", "error", "functions_total", "functions_equivalent") if key in result}
            calls = []
            for call in result.get("call_results") or []:
                state = str(call.get("state") or case_index[call["case_id"]][4])
                modalities = analyze.call_modalities(call, state)
                if bool(call.get("matched")) != (not modalities):
                    raise ValueError(f"{call['case_id']}: mismatch flags disagree with replay")
                item = {key: call[key] for key in ("case_id", "signature", "matched")}
                item.update({
                    "state": state,
                    "observed_mismatch_modalities": list(modalities),
                    "expected": {"call": {"ok": bool((call.get("expected") or {}).get("call", {}).get("ok"))}},
                    "actual": {"call": {"ok": bool((call.get("actual") or {}).get("call", {}).get("ok"))}},
                })
                calls.append(item)
            projected["call_results"] = calls
            output.write(("" if first else ",\n") + json.dumps(projected, separators=(",", ":")))
            first = False
        output.write('],\n  "summary": ' + json.dumps(analyze.load_tail_summary(source)) + "\n}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True,
        help="Construction snapshot containing archived semantic_coverage metadata.")
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("inputs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_output = args.output_dir / "benchmark.jsonl.gz"
    project_benchmark(args.benchmark, benchmark_output)
    _, case_index, _ = analyze.analyze_benchmark(benchmark_output)
    sources = [(args.benchmark, benchmark_output)]
    for _, _, relative in analyze.MODEL_FILES:
        output = args.output_dir / Path(relative).name
        source = args.replay_dir / output.stem
        print(f"Projecting {source.name}", flush=True)
        project_replay(source, output, case_index)
        sources.append((source, output))
    manifest = {
        "schema": "scdbench-semantic-analysis-input-v1",
        "projection": "Case identity, test family, original archived coverage, comparison modality flags, and call success flags. Full returned bytes, logs, storage values, source, and execution logs are not duplicated.",
        "files": [{"source_name": source.name, "source_sha256": analyze.sha256_file(source),
                   "name": output.name, "sha256": analyze.sha256_file(output)} for source, output in sources],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
