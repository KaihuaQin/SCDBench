#!/usr/bin/env python3
"""Compile selected submitted SCDBench rows into ItyFuzz glob artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.foundry_compile import (  # noqa: E402
    build_solc_standard_json_settings,
    ensure_solc_version,
    extract_solc_version,
)


HEX_RE = re.compile(r"^[0-9a-fA-F]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a selected subset of scdbench.jsonl into the "
            ".bin/.abi/combined.json layout consumed by ItyFuzz's local glob loader."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT.parent / "dataset/scdbench.jsonl",
        help="Submitted SCDBench JSONL snapshot.",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        required=True,
        help="Text file containing one benchmark_id per line.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).resolve().parent / "recorded/contract_metadata.jsonl",
        help="Recorded deployment modes omitted from the compact public dataset.",
    )
    parser.add_argument("--compile-timeout", type=float, default=120.0)
    parser.add_argument(
        "--allow-runtime-mismatch",
        action="store_true",
        help="Keep an artifact even if recompilation does not reproduce the submitted runtime bytecode.",
    )
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_hex(value: object) -> str:
    text = str(value or "").strip()
    if text.startswith("0x"):
        text = text[2:]
    if len(text) % 2 or not HEX_RE.fullmatch(text):
        raise ValueError("invalid hex bytecode")
    return text.lower()


def load_selected_rows(dataset: Path, ids_file: Path) -> list[dict[str, Any]]:
    selected_ids = [
        line.strip()
        for line in ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not selected_ids:
        raise ValueError(f"no benchmark IDs found in {ids_file}")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError(f"duplicate benchmark IDs in {ids_file}")

    wanted = set(selected_ids)
    rows: dict[str, dict[str, Any]] = {}
    with dataset.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            benchmark_id = str(row.get("benchmark_id", ""))
            if benchmark_id in wanted:
                row["_dataset_line_no"] = line_no
                rows[benchmark_id] = row

    missing = [benchmark_id for benchmark_id in selected_ids if benchmark_id not in rows]
    if missing:
        raise ValueError(f"{len(missing)} selected IDs are absent from {dataset}: {missing[:10]}")
    return [rows[benchmark_id] for benchmark_id in selected_ids]


def compiler_inputs(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    compiler = row.get("compiler")
    if not isinstance(compiler, dict):
        compiler = {}
    compiler_version = str(
        compiler.get("compiler_version") or row.get("compiler_version") or ""
    )
    settings = build_solc_standard_json_settings(
        compiler_version=compiler_version,
        optimization_used=(compiler.get("settings") or {}).get("optimizer", {}).get(
            "enabled", False
        ),
        runs=(compiler.get("settings") or {}).get("optimizer", {}).get("runs", 200),
        evm_version=(compiler.get("settings") or {}).get("evmVersion", ""),
        multi_file_settings=compiler.get("settings"),
        include_contract_outputs=True,
    )
    settings["outputSelection"] = {
        "*": {
            "": ["ast"],
            "*": [
                "abi",
                "evm.bytecode.object",
                "evm.deployedBytecode.object",
                "evm.deployedBytecode.sourceMap",
                "evm.bytecode.linkReferences",
                "evm.deployedBytecode.linkReferences",
            ],
        }
    }
    return compiler_version, settings


def run_solc(
    source_code: str,
    compiler_version: str,
    settings: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any], str]:
    version = extract_solc_version(compiler_version)
    if not version:
        raise ValueError(f"cannot extract solc version from {compiler_version!r}")
    solc_path = ensure_solc_version(version)
    standard_input = {
        "language": "Solidity",
        "sources": {"contract.sol": {"content": source_code}},
        "settings": settings,
    }
    completed = subprocess.run(
        [solc_path, "--standard-json"],
        input=json.dumps(standard_input),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    try:
        output = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"solc {version} returned invalid JSON: {completed.stderr[-2000:]}"
        ) from exc

    errors = [
        entry
        for entry in output.get("errors", [])
        if isinstance(entry, dict) and entry.get("severity") == "error"
    ]
    if completed.returncode != 0 or errors:
        formatted = "\n\n".join(
            str(entry.get("formattedMessage") or entry.get("message") or entry)
            for entry in errors[:10]
        )
        raise RuntimeError(
            f"solc {version} failed with exit {completed.returncode}:\n{formatted}"
        )
    return output, solc_path


def artifact_runtime(artifact: dict[str, Any]) -> str:
    return normalize_hex(
        ((artifact.get("evm") or {}).get("deployedBytecode") or {}).get("object", "")
    )


def select_contract_artifact(
    row: dict[str, Any], contracts: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    requested_name = str(row.get("contract_name") or "")
    if requested_name in contracts:
        return requested_name, contracts[requested_name]

    submitted_runtime = normalize_hex(row.get("bytecode"))
    exact_matches: list[tuple[str, dict[str, Any]]] = []
    deployable: list[tuple[str, dict[str, Any]]] = []
    for name, artifact in contracts.items():
        if not isinstance(artifact, dict):
            continue
        runtime = artifact_runtime(artifact)
        if runtime:
            deployable.append((name, artifact))
        if runtime == submitted_runtime:
            exact_matches.append((name, artifact))
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(deployable) == 1:
        return deployable[0]
    names = sorted(name for name, _ in deployable)
    raise ValueError(
        f"cannot uniquely select target {requested_name!r}; deployable artifacts: {names}"
    )


def write_case(
    row: dict[str, Any],
    output_dir: Path,
    compile_timeout: float,
    allow_runtime_mismatch: bool,
) -> dict[str, Any]:
    benchmark_id = str(row["benchmark_id"])
    source_code = str(row.get("source_code") or "")
    if not source_code:
        raise ValueError("source_code is empty")

    compiler_version, settings = compiler_inputs(row)
    solc_output, solc_path = run_solc(
        source_code, compiler_version, settings, compile_timeout
    )
    contracts = (solc_output.get("contracts") or {}).get("contract.sol") or {}
    target_name, artifact = select_contract_artifact(row, contracts)

    evm = artifact.get("evm") or {}
    creation = normalize_hex((evm.get("bytecode") or {}).get("object", ""))
    runtime = normalize_hex((evm.get("deployedBytecode") or {}).get("object", ""))
    runtime_source_map = str(
        (evm.get("deployedBytecode") or {}).get("sourceMap") or ""
    )
    abi = artifact.get("abi")
    if not creation or not runtime or not isinstance(abi, list):
        raise ValueError("selected artifact is missing creation bytecode, runtime bytecode, or ABI")

    submitted_runtime = normalize_hex(row.get("bytecode"))
    runtime_matches_submission = runtime == submitted_runtime
    if not runtime_matches_submission and not allow_runtime_mismatch:
        raise ValueError(
            "recompiled runtime bytecode differs from the submitted benchmark row"
        )

    semantic_deployment_mode = str(
        (row.get("semantic_metadata") or {}).get("deployment_mode") or ""
    )
    if semantic_deployment_mode == "ground_truth_source_creation":
        ityfuzz_init = creation
        ityfuzz_abi = abi
    elif semantic_deployment_mode == "input_runtime_wrapper":
        from scripts.run_foundry_semantic import make_init_code_from_runtime
        ityfuzz_init = normalize_hex(make_init_code_from_runtime(runtime))
        ityfuzz_abi = [
            entry
            for entry in abi
            if not isinstance(entry, dict) or entry.get("type") != "constructor"
        ]
    else:
        raise ValueError(
            f"unsupported semantic deployment mode: {semantic_deployment_mode!r}"
        )

    artifact_dir = output_dir / benchmark_id / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", target_name).strip("._") or "Target"
    (artifact_dir / "contract.sol").write_text(source_code, encoding="utf-8")
    (artifact_dir / f"{prefix}.bin").write_text(ityfuzz_init + "\n", encoding="ascii")
    (artifact_dir / f"{prefix}.abi").write_text(
        json.dumps(ityfuzz_abi, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    combined = {
        "contracts": {
            f"contract.sol:{target_name}": {
                "srcmap-runtime": runtime_source_map,
            }
        },
        "sourceList": ["contract.sol"],
    }
    (artifact_dir / "combined.json").write_text(
        json.dumps(combined, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "difficulty_bin": row.get("difficulty_bin"),
        "dataset_line_no": row["_dataset_line_no"],
        "requested_contract_name": row.get("contract_name"),
        "compiled_contract_name": target_name,
        "artifact_prefix": prefix,
        "compiler_version": compiler_version,
        "compiler_settings": settings,
        "solc_path": solc_path,
        "source_sha256": sha256_text(source_code),
        "submitted_runtime_sha256": sha256_text(submitted_runtime),
        "compiled_runtime_sha256": sha256_text(runtime),
        "runtime_matches_submission": runtime_matches_submission,
        "creation_bytecode": creation,
        "ityfuzz_init_bytecode": ityfuzz_init,
        "ityfuzz_deployment_mode": semantic_deployment_mode,
        "runtime_bytecode": runtime,
        "runtime_source_map": runtime_source_map,
        "abi": abi,
        "ityfuzz_abi": ityfuzz_abi,
        "artifact_dir": str(artifact_dir.resolve()),
    }
    metadata_path = output_dir / benchmark_id / "artifact_meta.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "benchmark_id": benchmark_id,
        "difficulty_bin": row.get("difficulty_bin"),
        "status": "prepared",
        "artifact_meta": str(metadata_path.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "runtime_matches_submission": runtime_matches_submission,
    }


def main() -> int:
    args = parse_args()
    rows = load_selected_rows(args.dataset, args.ids_file)
    metadata = {
        row["benchmark_id"]: row
        for row in (
            json.loads(line) for line in args.metadata.read_text().splitlines() if line.strip()
        )
    }
    for row in rows:
        if row["benchmark_id"] not in metadata:
            raise ValueError(f"missing deployment metadata for {row['benchmark_id']}")
        row["semantic_metadata"] = metadata[row["benchmark_id"]]["semantic_metadata"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    for row in rows:
        benchmark_id = str(row["benchmark_id"])
        try:
            result = write_case(
                row,
                args.output_dir,
                args.compile_timeout,
                args.allow_runtime_mismatch,
            )
            print(f"prepared {benchmark_id}", flush=True)
        except Exception as exc:
            result = {
                "benchmark_id": benchmark_id,
                "difficulty_bin": row.get("difficulty_bin"),
                "status": "prepare_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"failed {benchmark_id}: {result['error']}", file=sys.stderr, flush=True)
        cases.append(result)

    manifest = {
        "schema_version": 1,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "ids_file": str(args.ids_file.resolve()),
        "requested_cases": len(rows),
        "prepared_cases": sum(case["status"] == "prepared" for case in cases),
        "failed_cases": sum(case["status"] != "prepared" for case in cases),
        "cases": cases,
    }
    manifest_path = args.output_dir / "prepared_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {manifest_path}")
    return 0 if manifest["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
