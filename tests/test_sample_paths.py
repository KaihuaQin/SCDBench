import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


suite = load("semantic_suite", "scripts/run_semantic_model_suite.py")
perfect = load("perfect_summary", "scripts/summarize_perfect_semantic_contracts.py")
metrics = load("second_sample_metrics", "experiments/second_sample/metrics.py")


class SamplePathTests(unittest.TestCase):
    def test_semantic_inputs_use_selected_runs_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            args = SimpleNamespace(
                before_only=False, after_only=False, runs_dir=directory,
                skip_prepare=False, skip_replay=True, dataset=directory / "sample.jsonl",
                prepare_concurrency=1, compile_timeout=60, retry_prepare_errors=False,
                prepare_jobs=1,
            )
            run = suite.MODEL_RUNS[0]
            merged = directory / "merged.jsonl"
            def execute(_label, jobs, _count):
                for _, callback in jobs:
                    callback()
            with patch.object(suite, "parse_args", return_value=args), \
                 patch.object(suite, "selected_runs", return_value=[run]), \
                 patch.object(suite, "make_after_repair_predictions", return_value=merged) as repair, \
                 patch.object(suite, "run_parallel", side_effect=execute), \
                 patch.object(suite, "run_prepare") as prepare:
                suite.main()
            self.assertEqual(repair.call_args.args[0], directory / run["prediction"].name)
            self.assertEqual(repair.call_args.args[1], directory / run["repair_prediction"].name)
            self.assertEqual(prepare.call_args_list[0].kwargs["predictions"], directory / run["prediction"].name)
            self.assertEqual(prepare.call_args_list[1].kwargs["predictions"], merged)

    def test_perfect_summary_reads_selected_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.jsonl"
            path.write_text(json.dumps({
                "benchmark_id": "second-easy-1", "difficulty_bin": "easy",
                "test_cases": {"f()": {"entry_type": "function", "signature": "f()"}},
            }) + "\n")
            self.assertEqual(perfect.load_dataset_functions(path), {
                "second-easy-1": {"difficulty": "easy", "functions": {"f()"}},
            })

    def test_repair_preserves_initial_format_eligibility(self):
        before = {"x": {"format": {"ok": True}, "compilation": {"attempted": True, "success": False}}}
        after = {"x": {"format": {"ok": False}, "compilation": {"attempted": False, "success": False}}}
        combined, _ = metrics.combine_protocol_evaluations(before, after, {"x"})
        self.assertTrue(combined["x"]["format"]["ok"])
        with self.assertRaises(ValueError):
            metrics.combine_protocol_evaluations(before, after, set())


if __name__ == "__main__":
    unittest.main()
