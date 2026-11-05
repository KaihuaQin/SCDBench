"""Recompute second-sample tables from stored per-contract evaluations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from metrics import by_id, combine_protocol_evaluations, difficulty, metric_group, perfect_contracts

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = {
    "claude_opus47_high": ("Opus 4.7", "claude_opus47_reasoning_high"),
    "gpt53_codex_high": ("GPT-5.3-Codex", "gpt53_codex_reasoning_high"),
    "glm5_high": ("GLM-5", "glm5_reasoning_high"),
    "glm5_instruct": ("GLM-5 (instruct)", "glm5_reasoning_none"),
}
STRATA = ("easy", "medium", "hard", "all")


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(rows, evaluations, replay):
    output = {}
    for stratum in STRATA:
        subset = rows if stratum == "all" else [row for row in rows if difficulty(row) == stratum]
        result = metric_group(subset, evaluations, replay)
        ids = perfect_contracts(by_id(subset), evaluations, replay)
        result["perfect"] = {"contracts": len(ids), "benchmark_ids": ids}
        output[stratum] = result
    return output


def validate_reference(result, reference):
    for name in MODELS:
        for variant in ("initial", "repaired"):
            expected = reference[name][variant]
            for stratum in STRATA:
                old = expected["overall"] if stratum == "all" else expected["by_difficulty"][stratum]
                new = result[name][variant][stratum]
                for key in ("contracts", "format_ok", "compile_ok"):
                    if new[key] != old[key]:
                        raise ValueError(f"Reference mismatch: {name}/{variant}/{stratum}/{key}")
                for group, keys in {
                    "abi": ("true_positive", "false_positive", "ground_truth", "precision", "recall", "f1"),
                    "semantic": ("equivalent", "ground_truth", "rate"),
                }.items():
                    for key in keys:
                        if not math.isclose(new[group][key], old[group][key], abs_tol=1e-12):
                            raise ValueError(f"Reference mismatch: {name}/{variant}/{stratum}/{group}/{key}")
                if variant == "initial":
                    perfect = expected["perfect"]["contracts"] if stratum == "all" else expected["perfect"]["by_difficulty"][stratum]
                else:
                    perfect = old["perfect"]["contracts"]
                if new["perfect"]["contracts"] != perfect:
                    raise ValueError(f"Perfect-recovery mismatch: {name}/{variant}/{stratum}")


def markdown(result):
    lines = ["# Second-Sample Results", "", "Recomputed from stored per-contract evaluations and function-level replay outcomes.", ""]
    metrics = (
        ("Format completeness", lambda x: f"{x['format_ok']}/{x['contracts']} ({100*x['format_rate']:.1f}%)"),
        ("Compilability", lambda x: f"{x['compile_ok']}/{x['contracts']} ({100*x['compile_rate']:.1f}%)"),
        ("ABI recovery (precision / recall / F1)", lambda x: " / ".join(f"{x['abi'][key]:.3f}" for key in ("precision", "recall", "f1"))),
        ("Semantic consistency", lambda x: f"{x['semantic']['equivalent']}/{x['semantic']['ground_truth']} ({100*x['semantic']['rate']:.1f}%)"),
        ("Perfect contract recovery", lambda x: f"{x['perfect']['contracts']}/{x['contracts']}"),
    )
    for title, cell in metrics:
        lines.extend([f"## {title}", "", "| Model | Setting | Easy | Medium | Hard | All |", "|---|---|---:|---:|---:|---:|"])
        for name, (label, _) in MODELS.items():
            for variant in ("initial", "repaired"):
                cells = " | ".join(cell(result[name][variant][stratum]) for stratum in STRATA)
                lines.append(f"| {label} | {variant} | {cells} |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=HERE / "data")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs/second_sample")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--check", action="store_true", help="Compare every table cell against the archived experiment summaries.")
    args = parser.parse_args()
    rows = json.loads((args.input_dir / "contracts.json").read_text())["results"]
    contracts = by_id(rows)
    if len(contracts) != 600 or any(sum(difficulty(row) == s for row in rows) != 200 for s in STRATA[:3]):
        raise ValueError("Expected 600 contracts with 200 per complexity stratum")
    result = {}
    for name, (_, slug) in MODELS.items():
        initial_predictions = by_id(read_jsonl(args.runs_dir / f"openrouter_{slug}.jsonl"))
        repair_predictions = by_id(read_jsonl(args.runs_dir / f"openrouter_repair_{slug}.jsonl"))
        before = by_id(read_jsonl(args.input_dir / name / "initial.evaluation.jsonl"))
        after = by_id(read_jsonl(args.input_dir / name / "repaired.evaluation.jsonl"))
        if set(initial_predictions) != set(contracts) or set(before) != set(contracts):
            raise ValueError(f"Incomplete input IDs for {name}")
        repaired, validation = combine_protocol_evaluations(before, after, set(repair_predictions))
        result[name] = {"repair_validation": validation}
        for variant, evaluations in (("initial", before), ("repaired", repaired)):
            replay = by_id(json.loads((args.input_dir / name / f"{variant}.replay.json").read_text())["results"])
            eligible = {bid for bid, row in evaluations.items() if (row.get("semantic") or {}).get("eligible")}
            if set(replay) != eligible or any(row.get("error") for row in replay.values()):
                raise ValueError(f"Incomplete/error replay set for {name}/{variant}")
            for bid, row in replay.items():
                functions = row.get("function_results") or []
                signatures = {item["signature"] for item in functions}
                equivalent = sum(bool(item.get("equivalent")) for item in functions)
                if (len(signatures) != len(functions)
                        or len(functions) != row["functions_total"]
                        or equivalent != row["functions_equivalent"]):
                    raise ValueError(f"Inconsistent function-level replay counts: {name}/{variant}/{bid}")
            result[name][variant] = summarize(rows, evaluations, replay)
    if args.check:
        validate_reference(result, json.loads((args.input_dir / "reference_metrics.json").read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "tables.md").write_text(markdown(result))
    print(f"Recomputed five tables for all four models. Reference check: {'passed' if args.check else 'not requested'}.")


if __name__ == "__main__":
    main()
