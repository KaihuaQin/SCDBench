#!/usr/bin/env python3
"""Check regenerated scientific results against the archived analysis."""

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
        assert math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    actual = json.loads((root / "results.json").read_text())
    expected = json.loads((root / "expected_results.json").read_text())
    for key in ("analysis", "outcome_definitions", "model_specs", "integrity", "descriptive",
                "spearman", "difficulty_component_associations", "regression", "predictor_diagnostics"):
        compare(actual[key], expected[key], key)
    print("All scientific result fields match the archived analysis.")
