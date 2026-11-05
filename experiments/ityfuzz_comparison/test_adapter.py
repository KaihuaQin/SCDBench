#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_ityfuzz import (
    canonical_sequence,
    decompress_source_map,
    export_sequences,
    has_ityfuzz_target,
    instruction_pcs,
    ityfuzz_failure_zero_reason,
    lines_for_range,
    parse_difficulty_budgets,
    source_map_coverable_lines,
    zero_coverage,
)


class AdapterTests(unittest.TestCase):
    def test_instruction_pcs_skips_push_payload(self) -> None:
        self.assertEqual(
            instruction_pcs("60015b61020300"),
            [(0, 0x60), (2, 0x5B), (3, 0x61), (6, 0x00)],
        )

    def test_source_map_inherits_omitted_fields(self) -> None:
        self.assertEqual(
            decompress_source_map("0:4:0:-;:2::;8::0:"),
            [(0, 4, 0), (0, 2, 0), (8, 2, 0)],
        )

    def test_parse_difficulty_budgets(self) -> None:
        self.assertEqual(
            parse_difficulty_budgets("easy=4,medium=9,hard=16"),
            {"easy": 4.0, "medium": 9.0, "hard": 16.0},
        )

    def test_ityfuzz_target_detection(self) -> None:
        self.assertFalse(
            has_ityfuzz_target(
                {
                    "ityfuzz_abi": [
                        {"type": "function", "stateMutability": "view"},
                    ]
                }
            )
        )
        self.assertTrue(
            has_ityfuzz_target(
                {
                    "ityfuzz_abi": [
                        {"type": "function", "stateMutability": "nonpayable"}
                    ]
                }
            )
        )
        self.assertTrue(
            has_ityfuzz_target(
                {
                    "ityfuzz_abi": [
                        {"type": "function", "stateMutability": "pure"},
                    ]
                }
            )
        )
        self.assertTrue(
            has_ityfuzz_target(
                {
                    "ityfuzz_abi": [
                        {"type": "fallback", "stateMutability": "nonpayable"},
                    ]
                }
            )
        )

    def test_view_only_contract_has_zero_coverage_with_denominator(self) -> None:
        metadata = {
            "runtime_bytecode": "6001005b6002",
            "runtime_source_map": "0:3:0:-;5:3:0:-;10:3:0:-;10:3:0:-",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "contract.sol"
            source_path.write_text("a();\nb();\nc();\n", encoding="utf-8")
            coverage = zero_coverage(metadata, source_path, "view-only")
        self.assertTrue(coverage["available"])
        self.assertEqual(coverage["source_line_covered"], 0)
        self.assertEqual(coverage["source_line_total"], 2)
        self.assertEqual(coverage["source_line_coverage"], 0.0)

    def test_known_ityfuzz_failure_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "campaign.log"
            log_path.write_text(
                "thread 'main' has overflowed its stack\n",
                encoding="utf-8",
            )
            reason = ityfuzz_failure_zero_reason(
                {"timed_out": False, "log": str(log_path)}
            )
        self.assertIn("stack overflow", reason or "")

    def test_lines_for_byte_range(self) -> None:
        source = b"one\ntwo\nthree\n"
        self.assertEqual(lines_for_range(source, 4, 5), {2, 3})

    def test_coverable_lines_mirror_opcode_exclusions(self) -> None:
        metadata = {
            "runtime_bytecode": "6001005b6002",
            "runtime_source_map": "0:3:0:-;5:3:0:-;10:3:0:-;10:3:0:-",
        }
        source = "a();\nb();\nc();\n"
        self.assertEqual(source_map_coverable_lines(metadata, source), {1, 3})

    def test_export_deduplicates_and_rejects_partial_files(self) -> None:
        transaction = {"caller": "00", "contract": "01"}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus"
            output = root / "sequences"
            corpus.mkdir()
            line = json.dumps(transaction)
            (corpus / "1_replayable").write_text(line + "\n", encoding="utf-8")
            (corpus / "2_replayable").write_text(line + "\n", encoding="utf-8")
            (corpus / "3_replayable").write_text("{partial", encoding="utf-8")
            result = export_sequences(corpus, output)
            self.assertEqual(result["candidate_replayable_files"], 3)
            self.assertEqual(result["unique_sequences"], 1)
            self.assertEqual(result["duplicate_sequences"], 1)
            self.assertEqual(len(result["rejected_sequences"]), 1)
            self.assertEqual(result["max_sequence_length"], 1)

    def test_canonical_sequence_requires_json_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace"
            path.write_text("[]\n", encoding="utf-8")
            sequence, error = canonical_sequence(path)
            self.assertIsNone(sequence)
            self.assertIn("not a JSON object", error or "")


if __name__ == "__main__":
    unittest.main()
