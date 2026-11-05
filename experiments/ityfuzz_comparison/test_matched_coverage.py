#!/usr/bin/env python3

from __future__ import annotations

import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from summarize_matched_coverage import coverage_line_sets


class MatchedCoverageTests(unittest.TestCase):
    def test_recorded_table_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "coverage.json"
            subprocess.run(
                [sys.executable, str(Path(__file__).with_name("summarize_matched_coverage.py")),
                 "--output", str(output)], check=True, capture_output=True,
            )
            result = json.loads(output.read_text())
        self.assertEqual(result["selected_contracts"], 60)
        self.assertEqual(result["comparable_contracts"], 51)
        self.assertEqual(result["summary"]["all"]["scdbench"]["covered"], 3370)
        self.assertEqual(result["summary"]["all"]["ityfuzz"]["covered"], 1192)
        self.assertEqual(result["summary"]["all"]["scdbench"]["total"], 4960)
        by_id = {case["benchmark_id"]: case for case in result["cases"]}
        self.assertFalse(by_id["scdbench-medium-163"]["scdbench_budget_capped"])
        recorded = json.loads((Path(__file__).parent / "recorded/ityfuzz_results.json").read_text())
        self.assertEqual(max(case["sequence_export"].get("max_sequence_length", 0)
                             for case in recorded["cases"]), 21)

    def test_coverage_line_sets(self) -> None:
        result = coverage_line_sets(
            {
                "available": True,
                "lines": {
                    "covered_line_numbers": [2, 4],
                    "uncovered_line_numbers": [3],
                    "total": 3,
                },
            }
        )
        self.assertEqual(result, ({2, 3, 4}, {2, 4}))

    def test_unavailable_coverage(self) -> None:
        self.assertIsNone(
            coverage_line_sets({"available": False, "reason": "unsupported"})
        )


if __name__ == "__main__":
    unittest.main()
