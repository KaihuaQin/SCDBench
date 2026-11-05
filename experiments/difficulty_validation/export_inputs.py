#!/usr/bin/env python3
"""Project benchmark metadata and stored evaluations into offline analysis rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import analyze


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True,
        help="Construction snapshot containing complexity scores and semantic entries.")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
        default=Path(__file__).with_name("inputs") / "observations.json")
    args = parser.parse_args()
    specs = []
    input_files = [args.benchmark]
    for spec in analyze.MODEL_SPECS:
        paths = {name: args.runs_dir / getattr(spec, name).relative_to(analyze.ROOT / "runs")
                 for name in ("prompt_path", "original_evaluation_path",
                              "repaired_evaluation_path", "semantic_path")}
        specs.append(replace(spec, **paths))
        input_files.extend(paths.values())
    benchmark = analyze.load_benchmark(args.benchmark)
    observations, integrity = analyze.join_rows(benchmark, specs)
    for row in observations:
        if not math.isfinite(row["prompt_tokens"]):
            row["prompt_tokens"] = None
    contracts = [{key: value for key, value in row.items() if key != "function_signatures"}
                 for row in benchmark.values()]
    payload = {
        "schema": "scdbench-complexity-analysis-input-v1",
        "contracts": contracts,
        "observations": observations,
        "model_integrity": integrity,
        "provenance": [{"name": path.name, "sha256": sha256(path)} for path in input_files],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(observations)} observations to {args.output}")


if __name__ == "__main__":
    main()
