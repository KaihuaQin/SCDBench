#!/usr/bin/env python3
"""Recreate the released paired ablation inputs from the original public dataset."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=HERE / "data/subset_metadata.jsonl")
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated")
    args = parser.parse_args()
    metadata = [json.loads(line) for line in args.metadata.read_text().splitlines() if line.strip()]
    selected = {row["benchmark_id"] for row in metadata}
    if len(metadata) != 60 or len(selected) != 60:
        raise ValueError("expected 60 unique selected contract IDs")
    rows = []
    with args.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["benchmark_id"] in selected:
                    rows.append(row)
    if len(rows) != 60 or {row["benchmark_id"] for row in rows} != selected:
        raise ValueError("dataset does not contain every selected contract exactly once")
    if collections.Counter(row["difficulty_bin"] for row in rows) != {"easy": 20, "medium": 20, "hard": 20}:
        raise ValueError("expected 20 contracts per complexity stratum")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, no_hints in (("with_hints.jsonl", False), ("without_hints.jsonl", True)):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                output = dict(row)
                if no_hints:
                    output.pop("functions", None)
                    output.pop("events", None)
                handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote two paired 60-contract datasets to {args.output_dir}")


if __name__ == "__main__":
    main()
