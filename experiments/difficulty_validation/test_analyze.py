#!/usr/bin/env python3

from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import analyze  # noqa: E402


def score(components: dict[str, float]) -> float:
    return 100.0 * math.fsum(
        analyze.DIFFICULTY_COMPONENT_WEIGHTS[key] * components[key]
        for key in analyze.DIFFICULTY_COMPONENTS
    )


def component_values(value: float) -> dict[str, float]:
    return {
        "bytecode_scale": value,
        "control_flow": 1.0 - value,
        "interface_source": min(1.0, value * 0.8 + 0.1),
        "state_interaction": min(1.0, value * value + 0.05),
        "low_level": min(1.0, value * 0.6 + 0.2),
    }


def synthetic_rows(
    contracts: int = 12,
    models: int = 3,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    adjustments = [
        (index - (models - 1) / 2.0) * 0.001
        for index in range(models)
    ]
    for contract_index in range(contracts):
        value = (contract_index + 1) / (contracts + 1)
        components = component_values(value)
        base = 1.0 - value
        for model_index, adjustment in enumerate(adjustments):
            rows.append(
                {
                    "benchmark_id": f"contract-{contract_index:02d}",
                    "model_key": f"model-{model_index}",
                    "difficulty_score": score(components),
                    "difficulty_components": dict(components),
                    "format_pass": base + adjustment,
                    "compile_pass": base**2 + adjustment,
                    "abi_f1_cumulative": 0.1 + 0.8 * base + adjustment,
                    "semantic_rate_cumulative": 0.05 + 0.7 * base + adjustment,
                }
            )
    return rows


class DifficultyComponentValidationTests(unittest.TestCase):
    def test_accepts_exact_finite_components_consistent_with_score(self) -> None:
        components = component_values(0.4)
        validated = analyze.validate_difficulty_components(
            components,
            benchmark_id="example",
            difficulty_score=score(components),
        )
        self.assertEqual(validated, components)
        self.assertEqual(
            list(validated),
            list(analyze.DIFFICULTY_COMPONENTS),
        )

    def test_rejects_bad_keys_values_and_weighted_score(self) -> None:
        base = component_values(0.4)
        cases: list[tuple[str, dict[str, object], float, str]] = []

        missing = dict(base)
        missing.pop("low_level")
        cases.append(("missing", missing, score(base), "keys are not exact"))

        extra = dict(base)
        extra["other"] = 0.1
        cases.append(("extra", extra, score(base), "keys are not exact"))

        boolean = dict(base)
        boolean["low_level"] = True
        cases.append(("boolean", boolean, score(base), "is not numeric"))

        infinite = dict(base)
        infinite["control_flow"] = math.inf
        cases.append(("infinite", infinite, score(base), "within \\[0, 1\\]"))

        out_of_range = dict(base)
        out_of_range["state_interaction"] = 1.1
        cases.append(("range", out_of_range, score(base), "within \\[0, 1\\]"))

        cases.append(("score", dict(base), score(base) + 1.0, "reconstruct score"))

        for name, components, difficulty_score, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    analyze.validate_difficulty_components(
                        components,
                        benchmark_id="bad",
                        difficulty_score=difficulty_score,
                    )


class DifficultyComponentAssociationTests(unittest.TestCase):
    def test_component_associations_are_complete_and_deterministic(self) -> None:
        rows = synthetic_rows()
        first = analyze.run_component_spearman_analysis(
            rows,
            replicates=100,
            seed=20260724,
        )
        second = analyze.run_component_spearman_analysis(
            rows,
            replicates=100,
            seed=20260724,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["contracts"], 12)
        self.assertEqual(
            first["model_keys_averaged"],
            ["model-0", "model-1", "model-2"],
        )
        self.assertEqual(len(first["components"]), 5)
        self.assertEqual(len(first["records"]), 20)
        self.assertEqual(
            {
                (record["component"], record["outcome"])
                for record in first["records"]
            },
            {
                (component, outcome)
                for component in analyze.DIFFICULTY_COMPONENTS
                for outcome in analyze.PRIMARY_OUTCOMES
            },
        )
        for record in first["records"]:
            self.assertIn(
                "bh_q_value_across_20_component_outcome_tests",
                record,
            )
            self.assertEqual(
                record["bootstrap_successes"]
                + record["bootstrap_failures"],
                100,
            )
        self.assertEqual(
            [
                record["bh_q_value_across_20_component_outcome_tests"]
                for record in first["records"]
            ],
            analyze.benjamini_hochberg(
                [
                    record["analytic_p_value"]
                    for record in first["records"]
                ]
            ),
        )

        by_key = {
            (record["component"], record["outcome"]): record
            for record in first["records"]
        }
        self.assertAlmostEqual(
            by_key[("bytecode_scale", "format_pass")]["rho"],
            -1.0,
        )
        self.assertAlmostEqual(
            by_key[("control_flow", "format_pass")]["rho"],
            1.0,
        )

    def test_rejects_components_that_differ_across_model_rows(self) -> None:
        rows = synthetic_rows(contracts=4, models=2)
        changed = copy.deepcopy(rows)
        components = dict(changed[1]["difficulty_components"])
        components["bytecode_scale"] += 0.01
        changed[1]["difficulty_components"] = components
        changed[1]["difficulty_score"] = score(components)

        with self.assertRaisesRegex(
            ValueError,
            "components differ across model rows",
        ):
            analyze.run_component_spearman_analysis(
                changed,
                replicates=10,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
