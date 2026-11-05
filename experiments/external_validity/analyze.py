#!/usr/bin/env python3
"""Recompute external-validity statistics from archived population histograms."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import from_bytecode as core

HERE = Path(__file__).resolve().parent


def load_view(key, data):
    total = data["total"]
    if not isinstance(total, int) or total <= 0:
        raise ValueError(f"{key}: invalid population total")
    for name, pairs in data["numeric"].items():
        if sum(weight for _, weight in pairs) != total or any(not isinstance(weight, int) or weight <= 0 for _, weight in pairs):
            raise ValueError(f"{key}.{name}: invalid histogram weights")
    if any(not isinstance(count, int) or not 0 <= count <= total for count in data["flag_counts"].values()):
        raise ValueError(f"{key}: invalid flag counts")
    if sum(data["compiler_counts"].values()) != total:
        raise ValueError(f"{key}: compiler counts do not sum to population total")
    if len(data["opcode_counts"]) != 256 or any(not isinstance(count, int) or count < 0 for count in data["opcode_counts"]):
        raise ValueError(f"{key}: invalid opcode counts")
    return core.PopulationView(
        key=key, total=total, numeric=data["numeric"],
        flag_counts=collections.Counter(data["flag_counts"]),
        opcode_counts=data["opcode_counts"],
        compiler_counts=collections.Counter(data["compiler_counts"]),
        compiler_detected_count=data["compiler_detected_count"],
        metadata_valid_count=data["metadata_valid_count"],
        scanner_metadata_mismatch_count=data["scanner_metadata_mismatch_count"],
    )


def compute(payload):
    if payload.get("schema") != "scdbench-external-validity-population-counts-v1":
        raise ValueError("unknown population-count schema")
    views = {key: load_view(key, data) for key, data in payload["populations"].items()}
    verified = payload["verified"]
    if any(sum(weight for _, weight in pairs) != verified["total"] for pairs in verified["numeric"].values()):
        raise ValueError("verified histogram counts disagree")
    summary = {key: core.view_summary(view) for key, view in views.items()}
    summary["verified"] = {
        "total": verified["total"],
        "numeric": {key: core.weighted_summary(pairs) for key, pairs in verified["numeric"].items()},
        "flag_counts": {**verified["flag_counts"], "canonical_eip1167": verified["canonical_eip1167_count"]},
        "compiler_counts": verified["compiler_counts"],
        "opcode_total": verified["opcode_total"],
    }
    comparisons = {}
    for key, view in views.items():
        comparisons[key + "_vs_verified"] = {
            "numeric": {feature: {
                "ks_distance": core.weighted_ks_distance(verified["numeric"][feature], view.numeric[feature]),
                "log1p_standardized_mean_difference": core.log1p_standardized_mean_difference(verified["numeric"][feature], view.numeric[feature]),
            } for feature in core.REPORT_FEATURES},
            "opcode_distribution": core.opcode_distribution_vs_verified(view, verified["opcode_counts_top80"], verified["opcode_total"]),
        }
    unique = views["deployed_unique_bytecode"]
    return {
        "schema": "scdbench-public-external-validity-v1",
        "populations": summary,
        "comparisons": comparisons,
        "compiler_comparison": {
            "unique_deployed_metadata_recovery_rate": unique.compiler_detected_count / unique.total,
            "unique_deployed_v08_share_among_recovered": unique.compiler_counts.get("v0.8", 0) / unique.compiler_detected_count,
            "verified_v08_share": verified["compiler_counts"].get("v0.8", 0) / verified["total"],
        },
        "scope": "Deployed-population comparison, not a verification-labeled unverified-contract accuracy set.",
    }


def markdown(results):
    order = ("deployed_weighted", "deployed_unique_bytecode", "verified", "scdbench")
    populations = results["populations"]
    lines = [
        "# External Validity", "",
        "| Statistic | All sampled deployments | Unique deployed runtimes | Unique verified runtimes | SCDBench |",
        "|---|---:|---:|---:|---:|",
        "| Contracts | " + " | ".join(f"{populations[key]['total']:,}" for key in order) + " |",
        "| Median runtime bytes | " + " | ".join(f"{populations[key]['numeric']['bytecode_raw_byte_length']['median']:,.0f}" for key in order) + " |",
    ]
    for flag, label in (("canonical_eip1167", "Canonical EIP-1167 minimal proxy"), ("sload", "Contains SLOAD"), ("sstore", "Contains SSTORE"), ("delegatecall", "Contains DELEGATECALL")):
        lines.append("| " + label + " | " + " | ".join(f"{100 * populations[key]['flag_counts'].get(flag, 0) / populations[key]['total']:.2f}%" for key in order) + " |")
    opcode = results["comparisons"]["deployed_unique_bytecode_vs_verified"]["opcode_distribution"]
    compiler = results["compiler_comparison"]
    lines += ["", f"Unique deployed versus verified opcode total-variation distance: {opcode['total_variation']:.3f}.", "",
              f"Compiler metadata is recoverable for {100 * compiler['unique_deployed_metadata_recovery_rate']:.2f}% of unique deployed runtimes. Solidity 0.8 accounts for {100 * compiler['unique_deployed_v08_share_among_recovered']:.2f}% of recovered versions and {100 * compiler['verified_v08_share']:.2f}% of the verified corpus.", "", results["scope"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "data/population_counts.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    results = compute(json.loads(args.input.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "REPORT.md").write_text(markdown(results))
    print(markdown(results))


if __name__ == "__main__":
    main()
