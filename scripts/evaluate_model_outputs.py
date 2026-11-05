#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from foundry_compile import CompileResult, run_solc_compile, to_bool, to_int
except ImportError:
    from scripts.foundry_compile import CompileResult, run_solc_compile, to_bool, to_int


XML_OUTPUT_RE = re.compile(
    r"^\s*<compiler>\s*(?P<compiler>.*?)\s*</compiler>\s*"
    r"<code>\s*(?P<code>.*?)\s*</code>\s*"
    r"<contract_name>\s*(?P<contract_name>.*?)\s*</contract_name>\s*$",
    re.DOTALL,
)

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SOLIDITY_DECL_RE = re.compile(r"\b(contract|interface|library)\s+[A-Za-z_][A-Za-z0-9_]*\b")
PLACEHOLDER_RE = re.compile(
    r"(\.\.\.|TODO|FIXME|placeholder|omitted|same as above|rest of|implementation omitted)",
    re.IGNORECASE,
)

DEFAULT_DATASET = Path("../dataset/scdbench.jsonl")


@dataclass
class ParsedModelOutput:
    format_ok: bool
    compiler_raw: str
    compiler_json: Optional[Dict[str, Any]]
    code: str
    contract_name: str
    errors: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SCDBench model outputs with format, compile, ABI, and fuzz-target metrics."
    )
    parser.add_argument("--dataset", action="append", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--fuzz-targets", type=Path, default=None)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--id", action="append", default=None)
    parser.add_argument("--ids-file", type=Path, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--compile-timeout", type=int, default=60)
    parser.add_argument(
        "--format-compile-only",
        action="store_true",
        help="Only evaluate response format and prediction compilation; skip ground-truth ABI/semantic target construction.",
    )
    args = parser.parse_args()

    args.datasets = [p.expanduser().resolve() for p in (args.dataset or [DEFAULT_DATASET])]
    args.predictions = args.predictions.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.summary = (args.summary.expanduser().resolve() if args.summary else args.output.with_suffix(".summary.json"))
    args.fuzz_targets = args.fuzz_targets.expanduser().resolve() if args.fuzz_targets else None
    args.ids_file = args.ids_file.expanduser().resolve() if args.ids_file else None
    return args


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                yield json.loads(raw)


def load_id_filter(args: argparse.Namespace) -> Optional[set[str]]:
    ids = set(args.id or [])
    if args.ids_file:
        with args.ids_file.open("r", encoding="utf-8") as f:
            for raw in f:
                value = raw.strip()
                if value and not value.startswith("#"):
                    ids.add(value)
    return ids or None


def load_dataset(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    split_filter = set(args.split or [])
    id_filter = load_id_filter(args)
    rows: Dict[str, Dict[str, Any]] = {}
    for path in args.datasets:
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                row = json.loads(raw)
                bid = row["benchmark_id"]
                if split_filter and row.get("split") not in split_filter:
                    continue
                if id_filter is not None and bid not in id_filter:
                    continue
                row["_dataset_path"] = str(path)
                row["_dataset_line"] = line_no
                rows[bid] = row
    return rows


def load_predictions(args: argparse.Namespace, dataset: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = [row for row in read_jsonl(args.predictions) if row.get("benchmark_id") in dataset]
    if args.offset:
        out = out[args.offset :]
    if args.limit is not None:
        out = out[: args.limit]
    return out


def load_completed(path: Path, retry_errors: bool) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for row in read_jsonl(path):
        bid = row.get("benchmark_id")
        if not bid:
            continue
        if retry_errors and row.get("evaluation_error"):
            continue
        completed.add(str(bid))
    return completed


def parse_model_output(prediction: Dict[str, Any]) -> ParsedModelOutput:
    content = str(prediction.get("content") or "")
    match = XML_OUTPUT_RE.fullmatch(content)
    if match is None:
        return ParsedModelOutput(False, "", None, "", "", ["missing_or_misordered_xml_blocks"])

    compiler_raw = match.group("compiler").strip()
    code = match.group("code").strip()
    contract_name = match.group("contract_name").strip()
    compiler_json: Optional[Dict[str, Any]] = None
    errors: List[str] = []

    try:
        parsed = json.loads(compiler_raw)
        if isinstance(parsed, dict):
            compiler_json = parsed
        else:
            errors.append("compiler_block_not_json_object")
    except json.JSONDecodeError:
        errors.append("compiler_block_invalid_json")

    validate_format(compiler_json, code, contract_name, errors)
    return ParsedModelOutput(not errors, compiler_raw, compiler_json, code, contract_name, errors)


def validate_format(
    compiler_json: Optional[Dict[str, Any]],
    code: str,
    contract_name: str,
    errors: List[str],
) -> None:
    if compiler_json is None:
        errors.append("missing_compiler_json")
    else:
        if not isinstance(compiler_json.get("version"), str) or not compiler_json.get("version"):
            errors.append("missing_compiler_version")
        if not isinstance(compiler_json.get("settings"), dict):
            errors.append("missing_compiler_settings")
    if not code:
        errors.append("empty_code")
    if re.search(r"^\s*import\s+", code, flags=re.MULTILINE):
        errors.append("contains_import")
    if PLACEHOLDER_RE.search(code):
        errors.append("contains_placeholder_text")
    if "pragma solidity" not in code:
        errors.append("missing_pragma")
    if not SOLIDITY_DECL_RE.search(strip_solidity_comments(code)):
        errors.append("missing_contract_interface_or_library_declaration")
    if not IDENTIFIER_RE.fullmatch(contract_name or ""):
        errors.append("invalid_contract_name")


def strip_solidity_comments(code: str) -> str:
    no_block = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    return re.sub(r"//.*", "", no_block)


def compiler_parts(compiler_json: Dict[str, Any]) -> tuple[str, object, object, object, Dict[str, object]]:
    version = str(compiler_json.get("version") or "")
    settings = compiler_json.get("settings")
    settings_dict: Dict[str, object] = dict(settings) if isinstance(settings, dict) else {}
    optimizer = settings_dict.get("optimizer")
    optimizer_dict = optimizer if isinstance(optimizer, dict) else {}
    optimization_used = optimizer_dict.get("enabled")
    runs = optimizer_dict.get("runs")
    evm_version = settings_dict.get("evmVersion")
    return version, optimization_used, runs, evm_version, settings_dict


def dataset_compiler_parts(row: Dict[str, Any]) -> tuple[str, object, object, object, Dict[str, object]]:
    compiler = row.get("compiler") if isinstance(row.get("compiler"), dict) else {}
    settings = compiler.get("settings") if isinstance(compiler.get("settings"), dict) else {}
    optimizer = settings.get("optimizer") if isinstance(settings.get("optimizer"), dict) else {}
    version = str(compiler.get("compiler_version") or row.get("compiler_version") or "")
    return version, optimizer.get("enabled"), optimizer.get("runs"), settings.get("evmVersion"), dict(settings)


def compile_prediction(parsed: ParsedModelOutput, timeout: int) -> CompileResult:
    assert parsed.compiler_json is not None
    version, optimization_used, runs, evm_version, settings = compiler_parts(parsed.compiler_json)
    return run_solc_compile(
        simplified_source=parsed.code,
        contract_name=parsed.contract_name,
        compiler_version=version,
        optimization_used=optimization_used,
        runs=runs,
        evm_version=evm_version,
        multi_file_settings=settings,
        timeout_sec=timeout,
    )


def compile_ground_truth(row: Dict[str, Any], timeout: int) -> CompileResult:
    version, optimization_used, runs, evm_version, settings = dataset_compiler_parts(row)
    return run_solc_compile(
        simplified_source=row["source_code"],
        contract_name=row["contract_name"],
        compiler_version=version,
        optimization_used=optimization_used,
        runs=runs,
        evm_version=evm_version,
        multi_file_settings=settings,
        timeout_sec=timeout,
    )


def canonical_abi_type(entry: Dict[str, Any]) -> str:
    raw_type = str(entry.get("type") or "")
    if raw_type.startswith("tuple"):
        suffix = raw_type[len("tuple") :]
        components = entry.get("components") or []
        inner = ",".join(canonical_abi_type(c) for c in components if isinstance(c, dict))
        return f"({inner}){suffix}"
    return raw_type


def function_signature(entry: Dict[str, Any]) -> str:
    inputs = entry.get("inputs") or []
    types = [canonical_abi_type(i) for i in inputs if isinstance(i, dict)]
    return f"{entry.get('name')}({','.join(types)})"


def function_abi_map(abi: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for entry in abi:
        if isinstance(entry, dict) and entry.get("type") == "function" and entry.get("name"):
            out[function_signature(entry)] = entry
    return out


def abi_metrics(gt_abi: Sequence[Dict[str, Any]], pred_abi: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    gt_map = function_abi_map(gt_abi)
    pred_map = function_abi_map(pred_abi)
    gt = set(gt_map)
    pred = set(pred_map)
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not gt else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "evaluated": True,
        "ground_truth_count": len(gt),
        "predicted_count": len(pred),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_signatures": sorted(gt & pred),
        "missing_signatures": sorted(gt - pred),
        "spurious_signatures": sorted(pred - gt),
    }


def compile_result_record(result: CompileResult) -> Dict[str, Any]:
    rec = asdict(result)
    rec["abi_function_count"] = len(function_abi_map(result.abi))
    rec["bytecode_bytes"] = len(result.bytecode.removeprefix("0x")) // 2 if result.bytecode else 0
    rec["deployed_bytecode_bytes"] = len(result.deployed_bytecode.removeprefix("0x")) // 2 if result.deployed_bytecode else 0
    rec.pop("abi", None)
    rec.pop("bytecode", None)
    rec.pop("deployed_bytecode", None)
    return rec


def evaluate_one(
    prediction: Dict[str, Any],
    dataset_row: Dict[str, Any],
    args: argparse.Namespace,
    gt_cache: Dict[str, CompileResult],
    gt_lock: threading.Lock,
) -> Dict[str, Any]:
    bid = prediction["benchmark_id"]
    parsed = parse_model_output(prediction)
    record: Dict[str, Any] = {
        "benchmark_id": bid,
        "split": dataset_row.get("split"),
        "difficulty_bin": dataset_row.get("difficulty_bin"),
        "family_label": dataset_row.get("family_label"),
        "model": prediction.get("model"),
        "prompt_version": prediction.get("prompt_version"),
        "finish_reason": prediction.get("finish_reason"),
        "format": {
            "ok": parsed.format_ok,
            "errors": parsed.errors,
            "contract_name": parsed.contract_name,
            "code_chars": len(parsed.code),
            "compiler_json": parsed.compiler_json,
        },
        "compilation": {"attempted": False, "success": False, "error": "format_not_ok"},
        "ground_truth_compilation": {"attempted": False, "success": False, "error": "not_needed"},
        "abi": {"evaluated": False, "error": "not_compiled"},
        "semantic": {"eligible": False, "reason": "not_compiled"},
        "evaluation_mode": "format_compile_only" if args.format_compile_only else "full",
    }
    if not parsed.format_ok:
        return record

    pred_compile = compile_prediction(parsed, args.compile_timeout)
    record["compilation"] = compile_result_record(pred_compile)
    if not pred_compile.success:
        record["abi"] = {"evaluated": False, "error": "prediction_compile_failed"}
        return record
    if args.format_compile_only:
        record["ground_truth_compilation"] = {
            "attempted": False,
            "success": False,
            "error": "skipped_format_compile_only",
        }
        record["abi"] = {"evaluated": False, "error": "skipped_format_compile_only"}
        record["semantic"] = {"eligible": False, "reason": "skipped_format_compile_only"}
        return record

    with gt_lock:
        gt_compile = gt_cache.get(bid)
    if gt_compile is None:
        gt_compile = compile_ground_truth(dataset_row, args.compile_timeout)
        with gt_lock:
            gt_cache[bid] = gt_compile
    record["ground_truth_compilation"] = compile_result_record(gt_compile)
    if not gt_compile.success:
        record["abi"] = {"evaluated": False, "error": "ground_truth_compile_failed"}
        return record

    abi = abi_metrics(gt_compile.abi, pred_compile.abi)
    record["abi"] = abi
    matched_pred = function_abi_map(pred_compile.abi)
    matched_gt = function_abi_map(gt_compile.abi)
    matched_signatures = abi["matched_signatures"]
    record["semantic"] = {
        "eligible": bool(matched_signatures and pred_compile.bytecode and pred_compile.deployed_bytecode),
        "matched_function_count": len(matched_signatures),
        "reason": "" if matched_signatures else "no_matched_abi_functions",
    }
    record["_fuzz_target"] = build_fuzz_target(
        record,
        dataset_row,
        gt_compile,
        pred_compile,
        matched_gt,
        matched_pred,
        matched_signatures,
    )
    return record


def build_fuzz_target(
    record: Dict[str, Any],
    dataset_row: Dict[str, Any],
    gt_compile: CompileResult,
    pred_compile: CompileResult,
    matched_gt: Dict[str, Dict[str, Any]],
    matched_pred: Dict[str, Dict[str, Any]],
    matched_signatures: Sequence[str],
) -> Optional[Dict[str, Any]]:
    if not record["semantic"]["eligible"]:
        return None
    return {
        "benchmark_id": record["benchmark_id"],
        "line": dataset_row.get("_dataset_line"),
        "difficulty": dataset_row.get("difficulty_bin"),
        "contract_name": dataset_row.get("contract_name"),
        "predicted_contract_name": record["format"].get("contract_name"),
        "original_bytecode": dataset_row.get("bytecode"),
        "original_bytecode_logic_byte_length": dataset_row.get("bytecode_logic_byte_length"),
        "compiled_bytecode": pred_compile.bytecode,
        "compiled_deployedBytecode": pred_compile.deployed_bytecode,
        "original_abi_matched": [matched_gt[sig] for sig in matched_signatures],
        "compiled_abi_matched": [matched_pred[sig] for sig in matched_signatures],
        "matched_signatures": list(matched_signatures),
        "abi_precision": record["abi"].get("precision"),
        "abi_recall": record["abi"].get("recall"),
        "abi_f1": record["abi"].get("f1"),
    }


def append_record(path: Path, record: Dict[str, Any], lock: threading.Lock) -> None:
    serializable = {k: v for k, v in record.items() if not k.startswith("_")}
    line = json.dumps(serializable, sort_keys=True, ensure_ascii=False)
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def grouped_fuzz_targets(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for record in records:
        target = record.get("_fuzz_target")
        if not target:
            continue
        diff = target.get("difficulty") or "unknown"
        out.setdefault(str(diff), []).append(target)
    for targets in out.values():
        targets.sort(key=lambda r: str(r.get("benchmark_id") or ""))
    return out


def read_grouped_fuzz_targets(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.exists():
        return {"easy": [], "medium": [], "hard": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"easy": [], "medium": [], "hard": []}
    out: Dict[str, List[Dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    if not isinstance(data, dict):
        return out
    for diff, items in data.items():
        if not isinstance(items, list):
            continue
        out.setdefault(str(diff), [])
        out[str(diff)].extend(item for item in items if isinstance(item, dict))
    return out


def merge_grouped_fuzz_targets(
    existing: Dict[str, List[Dict[str, Any]]],
    updates: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for grouped in (existing, updates):
        for diff, items in grouped.items():
            bucket = out.setdefault(str(diff), {})
            for item in items:
                bid = item.get("benchmark_id")
                if bid:
                    bucket[str(bid)] = item
    merged: Dict[str, List[Dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for diff, by_id in out.items():
        merged[str(diff)] = sorted(by_id.values(), key=lambda r: str(r.get("benchmark_id") or ""))
    return merged


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total": len(records),
        "overall": summarize_group(records),
        "by_difficulty_bin": {
            diff: summarize_group(group)
            for diff, group in group_by(records, lambda r: str(r.get("difficulty_bin") or "unknown")).items()
        },
    }


def group_by(records: Sequence[Dict[str, Any]], key_fn) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        out[key_fn(record)].append(record)
    return dict(out)


def summarize_group(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    format_ok = sum(1 for r in records if r.get("format", {}).get("ok"))
    compile_ok = sum(1 for r in records if r.get("compilation", {}).get("success"))
    abi_rows = [r.get("abi", {}) for r in records if r.get("abi", {}).get("evaluated")]
    semantic_eligible = sum(1 for r in records if r.get("semantic", {}).get("eligible"))
    tp = sum(int(r.get("true_positive", 0)) for r in abi_rows)
    fp = sum(int(r.get("false_positive", 0)) for r in abi_rows)
    fn = sum(int(r.get("false_negative", 0)) for r in abi_rows)
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0
    return {
        "contracts": total,
        "format_ok": format_ok,
        "format_rate": format_ok / total if total else 0.0,
        "compile_ok": compile_ok,
        "compile_rate_all": compile_ok / total if total else 0.0,
        "compile_rate_format_eligible": compile_ok / format_ok if format_ok else 0.0,
        "abi_evaluated": len(abi_rows),
        "abi_macro_precision": mean(float(r.get("precision", 0.0)) for r in abi_rows),
        "abi_macro_recall": mean(float(r.get("recall", 0.0)) for r in abi_rows),
        "abi_macro_f1": mean(float(r.get("f1", 0.0)) for r in abi_rows),
        "abi_micro_precision": micro_precision,
        "abi_micro_recall": micro_recall,
        "abi_micro_f1": micro_f1,
        "semantic_eligible": semantic_eligible,
    }


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def write_outputs(
    args: argparse.Namespace,
    summary_records: Sequence[Dict[str, Any]],
    target_records: Sequence[Dict[str, Any]],
) -> None:
    summary = summarize(summary_records)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.fuzz_targets:
        if target_records:
            existing_targets = read_grouped_fuzz_targets(args.fuzz_targets)
            targets = merge_grouped_fuzz_targets(existing_targets, grouped_fuzz_targets(target_records))
            args.fuzz_targets.parent.mkdir(parents=True, exist_ok=True)
            args.fuzz_targets.write_text(json.dumps(targets, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif args.fuzz_targets.exists():
            print(f"[fuzz-targets] kept existing {args.fuzz_targets}")
        else:
            targets = {"easy": [], "medium": [], "hard": []}
            args.fuzz_targets.parent.mkdir(parents=True, exist_ok=True)
            args.fuzz_targets.write_text(json.dumps(targets, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[summary] wrote {args.summary}")
    if args.fuzz_targets:
        if target_records or not args.fuzz_targets.exists():
            print(f"[fuzz-targets] wrote {args.fuzz_targets}")


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args)
    predictions = load_predictions(args, dataset)
    completed = load_completed(args.output, args.retry_errors)
    todo = [row for row in predictions if row.get("benchmark_id") not in completed]
    print(f"[load] predictions={len(predictions)} existing={len(completed)} todo={len(todo)}")

    append_lock = threading.Lock()
    gt_lock = threading.Lock()
    gt_cache: Dict[str, CompileResult] = {}
    emitted: List[Dict[str, Any]] = []

    if todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = {
                executor.submit(evaluate_one, prediction, dataset[prediction["benchmark_id"]], args, gt_cache, gt_lock): prediction[
                    "benchmark_id"
                ]
                for prediction in todo
            }
            done = 0
            for future in concurrent.futures.as_completed(futures):
                bid = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {"benchmark_id": bid, "evaluation_error": repr(exc)}
                append_record(args.output, record, append_lock)
                emitted.append(record)
                done += 1
                if done == 1 or done % 10 == 0 or done == len(todo):
                    print(f"[eval] {done}/{len(todo)} last={bid}")

    all_records = list(read_jsonl(args.output)) if args.output.exists() else []
    # Reconstruct fuzz targets for this process from emitted records; existing output JSONL
    # intentionally omits bytecode-sized target payloads, so a fresh target export should
    # be produced in the same run that evaluates the predictions.
    records_for_targets = emitted if emitted else all_records
    write_outputs(args, all_records, records_for_targets)


if __name__ == "__main__":
    main()
