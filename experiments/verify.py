"""Run the paper's offline analyses and regression tests. No network calls."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSES = [
    ["experiments/second_sample/analyze.py", "--check"],
    ["experiments/difficulty_validation/analyze.py"],
    ["experiments/difficulty_validation/check_results.py"],
    ["experiments/signature_ablation/analyze_results.py"],
    ["experiments/semantic_effectiveness/analyze.py"],
    ["experiments/semantic_effectiveness/check_results.py"],
    ["experiments/external_validity/analyze.py"],
    ["experiments/ityfuzz_comparison/summarize_matched_coverage.py"],
    ["experiments/semantic_checkpoint_timing/run_sequential.py", "--summarize-only",
     "--output-dir", "experiments/semantic_checkpoint_timing/results"],
]
TEST_DIRS = [
    "tests", "experiments/difficulty_validation", "experiments/semantic_effectiveness",
    "experiments/signature_ablation", "experiments/external_validity",
    "experiments/ityfuzz_comparison", "experiments/semantic_checkpoint_timing",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-only", action="store_true")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required.")
    commands = [] if args.tests_only else list(ANALYSES)
    commands.extend(["-m", "unittest", "discover", "-s", directory, "-p", "test_*.py", "-v"]
                    for directory in TEST_DIRS)
    for command in commands:
        print("\nRunning:", sys.executable, *command, flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
    print("\nAll requested offline analyses and tests passed.")


if __name__ == "__main__":
    main()
