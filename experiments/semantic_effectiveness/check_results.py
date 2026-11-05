#!/usr/bin/env python3
"""Check scientific outputs without comparing release-specific file metadata."""

import json
import math
from pathlib import Path


def compare(actual, expected, path="root"):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{path}[{index}]")
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    actual = json.loads((root / "results.json").read_text())
    expected = json.loads((root / "expected_results.json").read_text())
    for model in actual["models"]:
        actual["models"][model].pop("source_path", None)
        expected["models"][model].pop("source_path", None)
    for key in ("schema", "benchmark", "models", "coverage_threshold_ranking", "analysis_scope"):
        compare(actual[key], expected[key], key)
    print("All counts match exactly and floating-point results match within rounding tolerance.")
