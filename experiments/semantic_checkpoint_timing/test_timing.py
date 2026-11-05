from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from run_sequential import DEFAULT_RECORDED_RESULTS, aggregate_group, write_summary


class TimingTests(unittest.TestCase):
    def test_recorded_measurements(self) -> None:
        payload = json.loads(DEFAULT_RECORDED_RESULTS.read_text())
        records = payload["results"]
        self.assertEqual(len({row["benchmark_id"] for row in records}), 60)
        self.assertEqual(Counter(row["difficulty"] for row in records),
                         {"easy": 20, "medium": 20, "hard": 20})
        self.assertTrue(all(row["checkpoint_validation"]["valid"] for row in records))
        summary = aggregate_group(records)
        self.assertEqual(summary["successful_contracts"], 60)
        self.assertEqual(summary["checkpoints"], 20389)
        self.assertEqual(summary["max_transaction_sequence_depth"], 2)
        outlier = next(row for row in records if row["benchmark_id"] == "scdbench-medium-163")
        self.assertGreater(outlier["timing_seconds"]["checkpoint_execution"], 123)
        with tempfile.TemporaryDirectory() as directory:
            write_summary(Path(directory), records, 60)
            rebuilt = json.loads((Path(directory) / "results.json").read_text())
            self.assertEqual(rebuilt["by_difficulty"]["all"], summary)


if __name__ == "__main__":
    unittest.main()
