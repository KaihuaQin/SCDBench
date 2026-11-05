"""Regression checks for the external-validity population summaries."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("external_validity_public_analysis", HERE / "analyze.py")
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads((HERE / "data/population_counts.json").read_text())
        cls.results = analyze.compute(cls.payload)

    def test_reported_table(self):
        keys = ("deployed_weighted", "deployed_unique_bytecode", "verified", "scdbench")
        populations = self.results["populations"]
        self.assertEqual([populations[key]["total"] for key in keys], [1000000, 37653, 772736, 600])
        self.assertEqual([populations[key]["numeric"]["bytecode_raw_byte_length"]["median"] for key in keys], [45, 2100, 6831, 7830])
        expected = {
            "canonical_eip1167": [44.63, 16.94, 0, 0],
            "sload": [19.54, 64.45, 96.41, 93.50],
            "sstore": [8.44, 54.85, 87.52, 89.83],
            "delegatecall": [56.39, 36.96, 11.40, 9.83],
        }
        for flag, values in expected.items():
            actual = [round(100 * populations[key]["flag_counts"].get(flag, 0) / populations[key]["total"], 2) for key in keys]
            self.assertEqual(values, actual)

    def test_compiler_and_opcode_comparison(self):
        compiler = self.results["compiler_comparison"]
        self.assertEqual(round(100 * compiler["unique_deployed_metadata_recovery_rate"], 2), 54.56)
        self.assertEqual(round(100 * compiler["unique_deployed_v08_share_among_recovered"], 2), 75.46)
        self.assertEqual(round(100 * compiler["verified_v08_share"], 2), 77.45)
        distance = self.results["comparisons"]["deployed_unique_bytecode_vs_verified"]["opcode_distribution"]["total_variation"]
        self.assertAlmostEqual(distance, 0.035049781033005545, places=14)

    def test_inconsistent_histogram_is_rejected(self):
        bad = copy.deepcopy(self.payload["populations"]["scdbench"])
        bad["numeric"]["bytecode_raw_byte_length"][0][1] += 1
        with self.assertRaises(ValueError):
            analyze.load_view("scdbench", bad)


if __name__ == "__main__":
    unittest.main()
