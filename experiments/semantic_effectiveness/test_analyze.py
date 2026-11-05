#!/usr/bin/env python3

import unittest

import analyze


class ProjectionTests(unittest.TestCase):
    def test_storage_comparison_uses_original_mutability(self):
        call = {"expected": {"call": {"ok": True, "result": "0x"}, "storage_changes": {}},
                "actual": {"call": {"ok": True, "result": "0x"}, "storage_changes": {"0": "1"}}}
        self.assertEqual(analyze.call_modalities(call, "view"), ())
        self.assertEqual(analyze.call_modalities(call, "pure"), ())
        self.assertEqual(analyze.call_modalities(call, "nonpayable"), ("storage",))

    def test_projected_modalities_match_raw_observations(self):
        call = {"expected": {"call": {"ok": True, "result": "0x01"}, "logs": []},
                "actual": {"call": {"ok": False, "result": "0x02"}, "logs": [1]}}
        modalities = analyze.call_modalities(call, "view")
        self.assertEqual(modalities, ("status", "returndata", "logs"))
        projected = {"observed_mismatch_modalities": list(modalities)}
        self.assertEqual(analyze.call_modalities(projected, "view"), modalities)

    def test_invalid_projected_modalities_are_rejected(self):
        for values in (["unknown"], ["logs", "logs"], ["logs", "status"], "logs"):
            with self.subTest(values=values), self.assertRaises(ValueError):
                analyze.call_modalities({"observed_mismatch_modalities": values}, "nonpayable")
        with self.assertRaises(ValueError):
            analyze.call_modalities({"observed_mismatch_modalities": ["storage"]}, "view")

    def test_coverage_threshold_keeps_boundary_and_excludes_unavailable(self):
        records = {"a": {"line_coverage_ratio": 0.8},
                   "b": {"line_coverage_ratio": 0.79},
                   "c": {"line_coverage_ratio": None}}
        self.assertEqual(analyze.coverage_subset(records, 0.8), ["a"])
        self.assertEqual(analyze.coverage_subset(records, None), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
