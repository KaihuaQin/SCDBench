#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import solcx
from solcx import install as solcx_install


SOLCX_TMPDIR = Path(__file__).resolve().parents[1] / ".solcx-tmp"
STRICT_HEX_RE = re.compile(r"^(?:0x)?[0-9a-fA-F]*$")


@dataclass
class CompileResult:
    attempted: bool
    success: bool
    timed_out: bool
    abi_found: bool
    target_artifact: str
    error: str
    bytecode: str = ""
    deployed_bytecode: str = ""
    abi: List[Dict[str, Any]] = field(default_factory=list)
    solc_version: str = ""
    solc_path: str = ""


@dataclass
class SolcAstResult:
    attempted: bool
    success: bool
    timed_out: bool
    source_units: Dict[str, Dict[str, Any]]
    error: str


def configure_solcx_tempdir() -> Path:
    SOLCX_TMPDIR.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(SOLCX_TMPDIR)
    tempfile.tempdir = str(SOLCX_TMPDIR)
    return SOLCX_TMPDIR


configure_solcx_tempdir()


def truncate_for_meta(text: object, limit: int = 12000) -> str:
    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...<truncated {len(value) - limit} chars>..."


def normalize_path(path: str) -> str:
    return str(Path(path).as_posix()).lstrip("./")


def extract_solc_version(compiler_version: object) -> str:
    if not isinstance(compiler_version, str):
        return ""
    match = re.search(r"v?(\d+\.\d+\.\d+)", compiler_version)
    return match.group(1) if match else ""


def pragma_from_compiler_version(compiler_version: object) -> str:
    version = extract_solc_version(compiler_version)
    return f"pragma solidity {version};" if version else "pragma solidity >=0.4.11;"


def to_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def to_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip(), 0)
        except Exception:
            return default
    return default


def normalize_evm_version(evm_version: object) -> str:
    if evm_version is None:
        return ""
    value = str(evm_version).strip()
    if not value or value.lower() in {"none", "null", "default"}:
        return ""
    return value


def build_solc_standard_json_settings(
    compiler_version: str,
    optimization_used: object,
    runs: object,
    evm_version: object,
    multi_file_settings: Optional[Dict[str, object]],
    include_contract_outputs: bool = False,
) -> Dict[str, Any]:
    optimizer_enabled = to_bool(optimization_used, default=False)
    optimizer_runs = to_int(runs, default=200)
    evm = normalize_evm_version(evm_version)

    raw_settings = multi_file_settings if isinstance(multi_file_settings, dict) else {}
    settings: Dict[str, Any] = {}

    optimizer_cfg = raw_settings.get("optimizer")
    if isinstance(optimizer_cfg, dict):
        settings["optimizer"] = dict(optimizer_cfg)
        if "enabled" not in settings["optimizer"]:
            settings["optimizer"]["enabled"] = optimizer_enabled
        if "runs" not in settings["optimizer"]:
            settings["optimizer"]["runs"] = optimizer_runs
    else:
        settings["optimizer"] = {"enabled": optimizer_enabled, "runs": optimizer_runs}

    if "evmVersion" in raw_settings:
        evm = normalize_evm_version(raw_settings.get("evmVersion"))
    if evm:
        settings["evmVersion"] = evm

    if "viaIR" in raw_settings:
        settings["viaIR"] = bool(raw_settings.get("viaIR"))

    metadata = raw_settings.get("metadata")
    if isinstance(metadata, dict):
        settings["metadata"] = dict(metadata)

    if "remappings" in raw_settings and isinstance(raw_settings["remappings"], list):
        settings["remappings"] = [
            str(entry).strip()
            for entry in raw_settings["remappings"]
            if isinstance(entry, str) and str(entry).strip()
        ]

    if "libraries" in raw_settings and isinstance(raw_settings["libraries"], dict):
        raw_libraries = raw_settings["libraries"]
        settings["libraries"] = dict(raw_libraries)
        if include_contract_outputs:
            flattened: Dict[str, str] = {}
            ambiguous: set[str] = set()
            for _, source_libraries in raw_libraries.items():
                if not isinstance(source_libraries, dict):
                    continue
                for library_name, library_address in source_libraries.items():
                    if not isinstance(library_name, str) or not isinstance(library_address, str):
                        continue
                    existing = flattened.get(library_name)
                    if existing is None:
                        flattened[library_name] = library_address
                    elif existing != library_address:
                        ambiguous.add(library_name)
            for library_name in ambiguous:
                flattened.pop(library_name, None)
            if flattened:
                existing_contract_sol = settings["libraries"].get("contract.sol")
                merged_contract_sol = dict(existing_contract_sol) if isinstance(existing_contract_sol, dict) else {}
                merged_contract_sol.update(flattened)
                settings["libraries"]["contract.sol"] = merged_contract_sol

    output_selection: Dict[str, Dict[str, List[str]]] = {"*": {"": ["ast"]}}
    if include_contract_outputs:
        output_selection["*"]["*"] = ["abi", "evm.bytecode.object", "evm.deployedBytecode.object"]
    settings["outputSelection"] = output_selection

    return settings


def ensure_solc_version(version: str) -> str:
    installed_versions = {str(v) for v in solcx.get_installed_solc_versions()}
    if version not in installed_versions:
        solcx.install_solc(version)
    return str(solcx_install.get_executable(version))


def format_solc_errors(errors: Sequence[Dict[str, Any]], max_errors: int = 10) -> str:
    messages: List[str] = []
    for entry in errors[:max_errors]:
        if not isinstance(entry, dict):
            continue
        formatted = entry.get("formattedMessage")
        if isinstance(formatted, str) and formatted.strip():
            messages.append(formatted.strip())
            continue
        message = entry.get("message")
        if isinstance(message, str) and message.strip():
            messages.append(message.strip())
    if not messages:
        return "solc compile failed with no structured errors"
    return truncate_for_meta("\n\n".join(messages))


def classify_bytecode_string(raw_bytecode: str) -> Tuple[bool, str]:
    if not raw_bytecode:
        return False, "bytecode is empty"
    if "__$" in raw_bytecode or "$__" in raw_bytecode:
        return False, "bytecode contains unresolved library link placeholders"
    if "__" in raw_bytecode:
        return False, "bytecode contains unresolved library name placeholders"
    if not STRICT_HEX_RE.fullmatch(raw_bytecode):
        return False, "bytecode is not valid hex"
    hex_body = raw_bytecode[2:] if raw_bytecode.startswith("0x") else raw_bytecode
    if len(hex_body) % 2 != 0:
        return False, "bytecode hex length is odd"
    return True, ""


def run_solc_ast_compile(
    source_map: Dict[str, str],
    compiler_version: str,
    optimization_used: object,
    runs: object,
    evm_version: object,
    multi_file_settings: Optional[Dict[str, object]],
    timeout_sec: int,
) -> SolcAstResult:
    solc_version = extract_solc_version(compiler_version)
    if not solc_version:
        return SolcAstResult(False, False, False, {}, "unable to extract solc version")

    try:
        solc_path = ensure_solc_version(solc_version)
    except Exception as exc:
        return SolcAstResult(False, False, False, {}, f"failed to install solc {solc_version}: {exc}")

    settings = build_solc_standard_json_settings(
        compiler_version=compiler_version,
        optimization_used=optimization_used,
        runs=runs,
        evm_version=evm_version,
        multi_file_settings=multi_file_settings,
    )
    standard_json = {
        "language": "Solidity",
        "sources": {path: {"content": code} for path, code in source_map.items()},
        "settings": settings,
    }

    try:
        proc = subprocess.run(
            [solc_path, "--standard-json"],
            input=json.dumps(standard_json),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return SolcAstResult(True, False, True, {}, f"solc --standard-json timed out after {timeout_sec}s")
    except Exception as exc:
        return SolcAstResult(True, False, False, {}, f"solc --standard-json error: {exc}")

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if not stdout:
        return SolcAstResult(True, False, False, {}, truncate_for_meta(stderr or "solc produced no output"))

    try:
        output = json.loads(stdout)
    except Exception:
        return SolcAstResult(True, False, False, {}, truncate_for_meta(stderr or stdout))

    errors = output.get("errors")
    if not isinstance(errors, list):
        errors = []
    severe_errors = [
        err for err in errors if isinstance(err, dict) and str(err.get("severity") or "").lower() == "error"
    ]
    if proc.returncode != 0 or severe_errors:
        return SolcAstResult(True, False, False, {}, format_solc_errors(severe_errors or errors))

    source_units: Dict[str, Dict[str, Any]] = {}
    sources_out = output.get("sources")
    if isinstance(sources_out, dict):
        for source_name, payload in sources_out.items():
            if not isinstance(source_name, str) or not isinstance(payload, dict):
                continue
            ast = payload.get("ast")
            if isinstance(ast, dict):
                source_units[normalize_path(source_name)] = ast

    if not source_units:
        return SolcAstResult(True, False, False, {}, "solc succeeded but no SourceUnit ASTs were found")

    return SolcAstResult(True, True, False, source_units, "")


def run_solc_compile(
    simplified_source: str,
    contract_name: str,
    compiler_version: str,
    optimization_used: object,
    runs: object,
    evm_version: object,
    multi_file_settings: Optional[Dict[str, object]],
    timeout_sec: int,
) -> CompileResult:
    solc_version = extract_solc_version(compiler_version)
    if not solc_version:
        return CompileResult(False, False, False, False, "", "unable to extract solc version")

    try:
        solc_path = ensure_solc_version(solc_version)
    except Exception as exc:
        return CompileResult(False, False, False, False, "", f"failed to install solc {solc_version}: {exc}")

    settings = build_solc_standard_json_settings(
        compiler_version=compiler_version,
        optimization_used=optimization_used,
        runs=runs,
        evm_version=evm_version,
        multi_file_settings=multi_file_settings,
        include_contract_outputs=True,
    )
    standard_json = {
        "language": "Solidity",
        "sources": {"contract.sol": {"content": simplified_source}},
        "settings": settings,
    }

    try:
        proc = subprocess.run(
            [solc_path, "--standard-json"],
            input=json.dumps(standard_json),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return CompileResult(True, False, True, False, "", f"solc --standard-json timed out after {timeout_sec}s")
    except Exception as exc:
        return CompileResult(True, False, False, False, "", f"solc --standard-json error: {exc}")

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if not stdout:
        return CompileResult(True, False, False, False, "", truncate_for_meta(stderr or "solc produced no output"))

    try:
        output = json.loads(stdout)
    except Exception:
        return CompileResult(True, False, False, False, "", truncate_for_meta(stderr or stdout))

    errors = output.get("errors")
    if not isinstance(errors, list):
        errors = []
    severe_errors = [
        err for err in errors if isinstance(err, dict) and str(err.get("severity") or "").lower() == "error"
    ]
    if proc.returncode != 0 or severe_errors:
        return CompileResult(True, False, False, False, "", format_solc_errors(severe_errors or errors))

    contract_outputs = output.get("contracts")
    if not isinstance(contract_outputs, dict):
        return CompileResult(True, False, False, False, "", "solc succeeded but returned no contract outputs")

    file_contracts = contract_outputs.get("contract.sol")
    if not isinstance(file_contracts, dict):
        return CompileResult(True, False, False, False, "", "solc succeeded but did not emit outputs for contract.sol")

    artifact = file_contracts.get(contract_name)
    if not isinstance(artifact, dict):
        available = ", ".join(sorted(k for k in file_contracts if isinstance(k, str)))
        return CompileResult(
            True,
            False,
            False,
            False,
            "",
            truncate_for_meta(
                f"solc compiled successfully but target contract '{contract_name}' was not found"
                + (f"; available artifacts: {available}" if available else "")
            ),
        )

    abi = artifact.get("abi")
    evm = artifact.get("evm")
    creation_bytecode = ""
    deployed_bytecode = ""
    if isinstance(evm, dict):
        bytecode_obj = evm.get("bytecode")
        if isinstance(bytecode_obj, dict) and isinstance(bytecode_obj.get("object"), str):
            creation_bytecode = bytecode_obj["object"]
        deployed_obj = evm.get("deployedBytecode")
        if isinstance(deployed_obj, dict) and isinstance(deployed_obj.get("object"), str):
            deployed_bytecode = deployed_obj["object"]

    bytecode_ok, bytecode_error = classify_bytecode_string(deployed_bytecode)
    if not bytecode_ok:
        return CompileResult(
            True,
            False,
            False,
            isinstance(abi, list),
            f"contract.sol:{contract_name}",
            f"solc compiled successfully but runtime {bytecode_error}",
            bytecode=creation_bytecode,
            deployed_bytecode="",
            abi=abi if isinstance(abi, list) else [],
            solc_version=solc_version,
            solc_path=solc_path,
        )

    return CompileResult(
        True,
        True,
        False,
        isinstance(abi, list),
        f"contract.sol:{contract_name}",
        "",
        bytecode=creation_bytecode,
        deployed_bytecode=deployed_bytecode,
        abi=abi if isinstance(abi, list) else [],
        solc_version=solc_version,
        solc_path=solc_path,
    )
