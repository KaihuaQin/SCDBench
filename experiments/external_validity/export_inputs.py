#!/usr/bin/env python3
"""Export non-identifying population counts from the archived bytecode analysis inputs."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import from_bytecode as core


def histogram(pairs):
    counts = collections.Counter()
    for value, weight in pairs:
        counts[float(value)] += int(weight)
    return [[value, count] for value, count in sorted(counts.items())]


def export_view(view):
    return {
        "total": view.total,
        "numeric": {key: histogram(values) for key, values in view.numeric.items()},
        "flag_counts": dict(view.flag_counts),
        "opcode_counts": view.opcode_counts,
        "compiler_counts": dict(view.compiler_counts),
        "compiler_detected_count": view.compiler_detected_count,
        "metadata_valid_count": view.metadata_valid_count,
        "scanner_metadata_mismatch_count": view.scanner_metadata_mismatch_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-db", "extension-db", "benchmark", "verified-stats", "verified-summary", "verified-results", "verified-eip1167"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data/population_counts.json")
    args = parser.parse_args()
    programs, selection = core.load_deployed_programs(args.base_db, args.extension_db, target_size=1_000_000)
    benchmark, known_compilers, benchmark_validation = core.load_benchmark(args.benchmark)
    print(f"Scanned {len(programs)} unique deployed and {len(benchmark)} benchmark runtimes", flush=True)
    views = {
        "deployed_weighted": core.build_view("deployed_weighted", programs, deployment_weighted=True),
        "deployed_unique_bytecode": core.build_view("deployed_unique_bytecode", programs, deployment_weighted=False),
        "scdbench": core.build_view("scdbench", benchmark, deployment_weighted=False),
    }
    verified_values, verified_flags, rows = core.load_verified_numeric(args.verified_stats)
    summary = json.loads(args.verified_summary.read_text())
    results = json.loads(args.verified_results.read_text())
    proxies = json.loads(args.verified_eip1167.read_text())
    assert rows == summary["candidate_valid_rows"] == proxies["input"]["rows"]
    payload = {
        "schema": "scdbench-external-validity-population-counts-v1",
        "populations": {key: export_view(view) for key, view in views.items()},
        "verified": {
            "total": rows,
            "numeric": {key: histogram((value, 1) for value in values) for key, values in verified_values.items()},
            "flag_counts": dict(verified_flags),
            "canonical_eip1167_count": proxies["result"]["canonical_eip1167_count"],
            "compiler_counts": summary["compiler_minor_counts_valid"],
            "opcode_counts_top80": summary["opcode_counts_top80_all_rows"],
            "opcode_total": results["populations"]["verified_exact_bytecode_deduped"]["opcode_total"],
        },
        "selection": {key: value for key, value in selection.items() if key not in {"source_databases", "database_paths"}},
        "source_files": {},
    }
    # Only names and checksums are retained, never endpoints or filesystem paths.
    for name in ("base_db", "extension_db", "benchmark", "verified_stats", "verified_summary", "verified_results", "verified_eip1167"):
        path = getattr(args, name)
        payload["source_files"][name] = {"name": path.name, "sha256": core.sha256_file(path)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
