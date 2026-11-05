#!/usr/bin/env python3
"""Focused tests for the deployed-population comparison helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collect_corpus_statistics import scan_bytecode  # noqa: E402


def load_analyzer() -> ModuleType | None:
    path = SCRIPT_DIR / "from_bytecode.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("deployed_comparison_analyze", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load analyzer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYZER = load_analyzer()


def row_field(row: Any, name: str) -> Any:
    """Allow selection records to be mappings, dataclasses, or sqlite Rows."""

    if isinstance(row, Mapping):
        return row[name]
    try:
        return row[name]
    except (KeyError, TypeError):
        return getattr(row, name)


def create_sample_database(
    path: Path,
    rows: list[tuple[int, bytes, str, bytes | None]],
) -> None:
    """Create the subset of the collector schema needed by selection tests."""

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE contracts (
                sample_index INTEGER PRIMARY KEY,
                address BLOB NOT NULL UNIQUE,
                rpc_status TEXT NOT NULL,
                bytecode BLOB,
                bytecode_bytes INTEGER,
                code_keccak BLOB
            )
            """
        )
        for sample_index, address, status, bytecode in rows:
            code_hash = hashlib.sha256(bytecode).digest() if bytecode else None
            connection.execute(
                """
                INSERT INTO contracts(
                    sample_index, address, rpc_status, bytecode,
                    bytecode_bytes, code_keccak
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_index,
                    address,
                    status,
                    bytecode,
                    len(bytecode) if bytecode is not None else 0,
                    code_hash,
                ),
            )
        connection.commit()
    finally:
        connection.close()


@unittest.skipIf(ANALYZER is None, "deployed_comparison/analyze.py not created yet")
class ExactNonemptySelectionTests(unittest.TestCase):
    def test_uses_all_base_successes_then_ranked_extension_successes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.sqlite3"
            extension = Path(directory) / "extension.sqlite3"
            create_sample_database(
                base,
                [
                    (1, b"\x01" * 20, "ok", b"\x00"),
                    (2, b"\x02" * 20, "empty", b""),
                    (3, b"\x03" * 20, "ok", b"\x01"),
                    (4, b"\x04" * 20, "ok", b"\x02"),
                ],
            )
            create_sample_database(
                extension,
                [
                    (1, b"\x11" * 20, "ok", b"\x11"),
                    (2, b"\x12" * 20, "ok", b"\x12"),
                    (3, b"\x13" * 20, "empty", b""),
                    (4, b"\x14" * 20, "ok", b"\x14"),
                ],
            )

            selected = list(
                ANALYZER.select_exact_nonempty(base, extension, target_size=5)
            )

            self.assertEqual(len(selected), 5)
            self.assertEqual(
                [row_field(row, "address") for row in selected],
                [
                    b"\x01" * 20,
                    b"\x03" * 20,
                    b"\x04" * 20,
                    b"\x11" * 20,
                    b"\x12" * 20,
                ],
            )

    def test_rejects_an_insufficient_nonempty_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.sqlite3"
            extension = Path(directory) / "extension.sqlite3"
            create_sample_database(
                base,
                [(1, b"\x01" * 20, "ok", b"\x01")],
            )
            create_sample_database(
                extension,
                [(1, b"\x11" * 20, "empty", b"")],
            )

            with self.assertRaises(ValueError):
                list(
                    ANALYZER.select_exact_nonempty(
                        base, extension, target_size=2
                    )
                )

    def test_rejects_address_overlap_between_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.sqlite3"
            extension = Path(directory) / "extension.sqlite3"
            repeated = b"\x01" * 20
            create_sample_database(base, [(1, repeated, "ok", b"\x01")])
            create_sample_database(extension, [(1, repeated, "ok", b"\x02")])

            with self.assertRaises(ValueError):
                list(
                    ANALYZER.select_exact_nonempty(
                        base, extension, target_size=2
                    )
                )


@unittest.skipIf(ANALYZER is None, "deployed_comparison/analyze.py not created yet")
class WeightedStatisticsTests(unittest.TestCase):
    def test_weighted_quantile_matches_expanded_linear_interpolation(self) -> None:
        weighted = [(1.0, 2), (5.0, 1), (9.0, 2)]

        self.assertEqual(ANALYZER.weighted_quantile(weighted, 0.0), 1.0)
        self.assertEqual(ANALYZER.weighted_quantile(weighted, 0.5), 5.0)
        self.assertEqual(ANALYZER.weighted_quantile(weighted, 1.0), 9.0)
        self.assertEqual(ANALYZER.weighted_quantile(weighted, 0.375), 3.0)

    def test_weighted_ks_matches_expanded_empirical_cdfs(self) -> None:
        reference = [(0.0, 2), (2.0, 2)]
        comparison = [(0.0, 1), (1.0, 2), (2.0, 1)]

        self.assertAlmostEqual(
            ANALYZER.weighted_ks_distance(reference, comparison),
            0.25,
        )

    def test_weighted_ks_is_invariant_to_common_weight_scaling(self) -> None:
        reference = [(0.0, 1), (1.0, 3), (2.0, 2)]
        comparison = [(0.0, 10), (1.0, 30), (2.0, 20)]

        self.assertEqual(
            ANALYZER.weighted_ks_distance(reference, comparison),
            0.0,
        )


class PushAwareScanningTests(unittest.TestCase):
    def test_opcode_bytes_inside_push_data_are_not_counted(self) -> None:
        opcode_counts = [0] * 256

        result = scan_bytecode("60f4600155f4", opcode_counts)

        self.assertEqual(result["opcode_count"], 4)
        self.assertEqual(result["push_count"], 2)
        self.assertEqual(result["storage_op_count"], 1)
        self.assertEqual(result["delegatecall_count"], 1)
        self.assertEqual(opcode_counts[0xF4], 1)

    def test_push32_payload_is_not_interpreted_as_code(self) -> None:
        opcode_counts = [0] * 256
        bytecode = "7f" + ("f4" * 32) + "55"

        result = scan_bytecode(bytecode, opcode_counts)

        self.assertEqual(result["opcode_count"], 2)
        self.assertEqual(result["push_count"], 1)
        self.assertEqual(result["delegatecall_count"], 0)
        self.assertEqual(result["storage_op_count"], 1)
        self.assertEqual(opcode_counts[0xF4], 0)


@unittest.skipIf(ANALYZER is None, "deployed_comparison/analyze.py not created yet")
class TerminalMetadataValidationTests(unittest.TestCase):
    @staticmethod
    def bytecode_with_terminal_metadata(payload_hex: str) -> bytes:
        payload = bytes.fromhex(payload_hex)
        return b"\x60\x00" + payload + len(payload).to_bytes(2, "big")

    def test_accepts_solc_bytes3_version(self) -> None:
        bytecode = self.bytecode_with_terminal_metadata(
            "a164736f6c6343000817"
        )

        metadata = ANALYZER.decode_terminal_metadata(bytecode)

        self.assertTrue(metadata["valid"])
        self.assertEqual(metadata["logic"], b"\x60\x00")
        self.assertEqual(metadata["keys"], ["solc"])
        self.assertEqual(metadata["compiler_engine"], "solc")
        self.assertEqual(metadata["compiler_version"], "v0.8.23")

    def test_accepts_vyper_three_integer_version(self) -> None:
        bytecode = self.bytecode_with_terminal_metadata(
            "a165767970657283000309"
        )

        metadata = ANALYZER.decode_terminal_metadata(bytecode)

        self.assertTrue(metadata["valid"])
        self.assertEqual(metadata["logic"], b"\x60\x00")
        self.assertEqual(metadata["keys"], ["vyper"])
        self.assertEqual(metadata["compiler_engine"], "vyper")
        self.assertEqual(metadata["compiler_version"], "v0.3.9")

    def test_rejects_recognized_fields_with_invalid_schemas(self) -> None:
        invalid_payloads = {
            "solc array instead of bytes3": "a164736f6c6383000817",
            "ipfs integer instead of bytes34": "a1646970667300",
        }

        for case, payload_hex in invalid_payloads.items():
            with self.subTest(case=case):
                bytecode = self.bytecode_with_terminal_metadata(payload_hex)

                metadata = ANALYZER.decode_terminal_metadata(bytecode)

                self.assertFalse(metadata["valid"])
                self.assertEqual(metadata["logic"], bytecode)
                self.assertEqual(metadata["compiler_engine"], "unknown")
                self.assertEqual(metadata["compiler_version"], "unknown")


@unittest.skipIf(ANALYZER is None, "deployed_comparison/analyze.py not created yet")
class ProxyPatternTests(unittest.TestCase):
    def test_canonical_eip1167_sets_detection_and_union_flags(self) -> None:
        implementation = bytes.fromhex("11" * 20)
        runtime = (
            bytes.fromhex("363d3d373d3d3d363d73")
            + implementation
            + bytes.fromhex("5af43d82803e903d91602b57fd5bf3")
        )

        program = ANALYZER.scan_program(
            hashlib.sha256(runtime).digest(),
            runtime,
            multiplicity=1,
        )

        self.assertIn("delegatecall", program.flags)
        self.assertIn("canonical_eip1167", program.flags)
        self.assertIn("proxy_related_pattern_union", program.flags)
        self.assertNotIn("eip1967_implementation_pattern", program.flags)
        self.assertNotIn("eip1967_beacon_pattern", program.flags)


if __name__ == "__main__":
    unittest.main()
