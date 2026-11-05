"""Regression checks for the paired signature-hint experiment."""
from __future__ import annotations

import json
import importlib.util
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class AnalysisTests(unittest.TestCase):
    def assert_equivalent(self, expected, actual, path="result"):
        if isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual), path)
            for key in expected:
                self.assert_equivalent(expected[key], actual[key], path + "." + key)
        elif isinstance(expected, list):
            self.assertEqual(len(expected), len(actual), path)
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.assert_equivalent(left, right, f"{path}[{index}]")
        elif isinstance(expected, float):
            self.assertTrue(math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12), path)
        else:
            self.assertEqual(expected, actual, path)

    def test_all_archived_results(self):
        expected = json.loads((HERE / "data/expected_results.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            subprocess.run([sys.executable, str(HERE / "analyze_results.py"), "--output-json", str(result), "--output-md", str(Path(directory) / "report.md")], check=True)
            self.assert_equivalent(expected, json.loads(result.read_text()))

    def test_projection_has_no_private_prediction_fields(self):
        allowed = {"benchmark_id", "content", "difficulty_bin", "finish_reason", "model", "ok", "request_parameters", "response_model", "usage"}
        for path in (HERE / "runs").glob("*.jsonl"):
            with path.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    self.assertTrue(set(row) <= allowed)

    def test_duplicate_ids_are_rejected(self):
        spec = importlib.util.spec_from_file_location("signature_ablation_public_metrics", HERE / "metrics.py")
        metrics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics)
        with self.assertRaises(ValueError):
            metrics.by_id([{"benchmark_id": "duplicate"}, {"benchmark_id": "duplicate"}])


if __name__ == "__main__":
    unittest.main()
