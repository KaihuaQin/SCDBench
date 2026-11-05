#!/usr/bin/env python3
"""Validate how SCDBench difficulty relates to cumulative model outcomes.

The unit of analysis is a benchmark contract evaluated by one model. Primary
outcomes are cumulative, matching the paper's staged protocol:

* original decompilation format pass;
* compilation pass after the allowed repair;
* per-contract ABI F1, set to zero when a prerequisite fails; and
* semantically equivalent entries divided by all ground-truth replay entries.

The adjusted analysis is an OLS score/linear-probability model with model fixed
effects. Inference is clustered by benchmark contract because every contract is
evaluated by all four models. Percentile confidence intervals use deterministic
contract-cluster bootstrap resampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

DEFAULT_SEED = 20260724
DEFAULT_BOOTSTRAP_REPLICATES = 2000

PRIMARY_OUTCOMES = (
    "format_pass",
    "compile_pass",
    "abi_f1_cumulative",
    "semantic_rate_cumulative",
)
ROBUSTNESS_OUTCOMES = (
    "abi_exact_cumulative",
    "semantic_perfect_cumulative",
)

DIFFICULTY_COMPONENTS = (
    "bytecode_scale",
    "control_flow",
    "interface_source",
    "state_interaction",
    "low_level",
)
DIFFICULTY_COMPONENT_LABELS = {
    "bytecode_scale": "Bytecode scale",
    "control_flow": "Control flow",
    "interface_source": "Interface/source",
    "state_interaction": "State interaction",
    "low_level": "Low-level Solidity",
}
DIFFICULTY_COMPONENT_DEFINITIONS = {
    "bytecode_scale": (
        "Mean of robust-normalized metadata-stripped bytecode bytes and opcode count."
    ),
    "control_flow": (
        "Mean of robust-normalized cyclomatic proxy, block-count proxy, and "
        "control-flow opcode count."
    ),
    "interface_source": (
        "Mean of robust-normalized signature count, source declaration count, "
        "and nonblank source lines."
    ),
    "state_interaction": (
        "Mean of robust-normalized storage operations, external calls, create "
        "operations, and log operations."
    ),
    "low_level": (
        "Weighted indicators for inline assembly, delegatecall, fallback, "
        "receive, create2, and selfdestruct."
    ),
}
DIFFICULTY_COMPONENT_WEIGHTS = {
    "bytecode_scale": 0.30,
    "control_flow": 0.20,
    "interface_source": 0.20,
    "state_interaction": 0.15,
    "low_level": 0.15,
}

OUTCOME_LABELS = {
    "format_pass": "Format pass",
    "compile_pass": "Compilation pass",
    "abi_f1_cumulative": "ABI F1 (cumulative)",
    "semantic_rate_cumulative": "Semantic rate (cumulative)",
    "abi_exact_cumulative": "Exact ABI recovery",
    "semantic_perfect_cumulative": "Perfect semantic recovery",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    prompt_path: Path
    original_evaluation_path: Path
    repaired_evaluation_path: Path
    semantic_path: Path


MODEL_SPECS = (
    ModelSpec(
        key="opus47",
        label="Opus 4.7",
        prompt_path=ROOT / "runs/openrouter_claude_opus47_reasoning_high.jsonl",
        original_evaluation_path=ROOT
        / "runs/eval_first3_claude_opus47_reasoning_high.jsonl",
        repaired_evaluation_path=ROOT
        / "runs/semantic_prep/eval_semanticprep_claude_opus47_reasoning_high_after_repair.jsonl",
        semantic_path=ROOT
        / "runs/semantic/semantic_replay_claude_opus47_reasoning_high_after_repair.json",
    ),
    ModelSpec(
        key="gpt53_codex",
        label="GPT-5.3-Codex",
        prompt_path=ROOT / "runs/openrouter_gpt53_codex_reasoning_high.jsonl",
        original_evaluation_path=ROOT
        / "runs/eval_first3_gpt53_codex_reasoning_high.jsonl",
        repaired_evaluation_path=ROOT
        / "runs/semantic_prep/eval_semanticprep_gpt53_codex_reasoning_high_after_repair.jsonl",
        semantic_path=ROOT
        / "runs/semantic/semantic_replay_gpt53_codex_reasoning_high_after_repair.json",
    ),
    ModelSpec(
        key="glm5_reasoning",
        label="GLM-5",
        prompt_path=ROOT / "runs/openrouter_glm5_reasoning_high.jsonl",
        original_evaluation_path=ROOT
        / "runs/eval_first3_glm5_reasoning_high.jsonl",
        repaired_evaluation_path=ROOT
        / "runs/semantic_prep/eval_semanticprep_glm5_reasoning_high_after_repair.jsonl",
        semantic_path=ROOT
        / "runs/semantic/semantic_replay_glm5_reasoning_high_after_repair.json",
    ),
    ModelSpec(
        key="glm5_instruct",
        label="GLM-5 (instruct)",
        prompt_path=ROOT / "runs/openrouter_glm5_reasoning_none.jsonl",
        original_evaluation_path=ROOT
        / "runs/eval_first3_glm5_reasoning_none.jsonl",
        repaired_evaluation_path=ROOT
        / "runs/semantic_prep/eval_semanticprep_glm5_reasoning_none_after_repair.jsonl",
        semantic_path=ROOT
        / "runs/semantic/semantic_replay_glm5_reasoning_none_after_repair.json",
    ),
)

EXPECTED_STAGE_TOTALS = {
    "opus47": {
        "format_passes": 435,
        "compilation_passes": 415,
        "semantic_equivalent": 4284,
        "semantic_perfect": 42,
    },
    "gpt53_codex": {
        "format_passes": 564,
        "compilation_passes": 542,
        "semantic_equivalent": 4078,
        "semantic_perfect": 41,
    },
    "glm5_reasoning": {
        "format_passes": 520,
        "compilation_passes": 403,
        "semantic_equivalent": 3015,
        "semantic_perfect": 18,
    },
    "glm5_instruct": {
        "format_passes": 540,
        "compilation_passes": 386,
        "semantic_equivalent": 2570,
        "semantic_perfect": 6,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=EXPERIMENT_DIR / "inputs/observations.json")
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
        help="Contract-cluster bootstrap replicates (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=EXPERIMENT_DIR / "results.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=EXPERIMENT_DIR / "report.md",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def stream_top_level_array(
    path: Path,
    key: str,
    chunk_size: int = 1024 * 1024,
) -> Iterator[dict[str, Any]]:
    """Stream objects from a named top-level JSON array without loading the file."""

    decoder = json.JSONDecoder()
    marker = json.dumps(key)
    buffer = ""
    found_array = False
    eof = False

    with path.open(encoding="utf-8") as handle:
        while not found_array:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(f"{path}: top-level key {key!r} was not found")
            buffer += chunk
            marker_index = buffer.find(marker)
            if marker_index < 0:
                buffer = buffer[-(len(marker) + 32) :]
                continue
            array_index = buffer.find("[", marker_index + len(marker))
            while array_index < 0:
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError(f"{path}: key {key!r} is not followed by an array")
                buffer += chunk
                array_index = buffer.find("[", marker_index + len(marker))
            buffer = buffer[array_index + 1 :]
            found_array = True

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return

            while True:
                try:
                    value, consumed = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"{path}: truncated or invalid {key!r} array")
                    chunk = handle.read(chunk_size)
                    if chunk:
                        buffer += chunk
                    else:
                        eof = True

            if not isinstance(value, dict):
                raise ValueError(f"{path}: expected objects in top-level {key!r} array")
            yield value
            buffer = buffer[consumed:]


def read_trailing_summary(path: Path, tail_bytes: int = 256 * 1024) -> dict[str, Any]:
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        handle.seek(max(0, size - tail_bytes))
        text = handle.read().decode("utf-8")
    marker = '\n  "summary": '
    index = text.rfind(marker)
    if index < 0:
        raise ValueError(f"{path}: trailing top-level summary was not found")
    payload = text[index + len(marker) :]
    value, _ = json.JSONDecoder().raw_decode(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level summary is not an object")
    return value


def stable_seed(base_seed: int, *labels: str) -> int:
    digest = hashlib.sha256(
        (str(base_seed) + "\0" + "\0".join(labels)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def percentile_interval(values: Sequence[float], level: float = 0.95) -> list[float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return [math.nan, math.nan]
    alpha = (1.0 - level) / 2.0
    return [
        float(np.quantile(finite, alpha)),
        float(np.quantile(finite, 1.0 - alpha)),
    ]


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(p))
    if not finite_indices.size:
        return out.tolist()
    ordered = finite_indices[np.argsort(p[finite_indices])]
    adjusted = np.empty(ordered.size, dtype=float)
    running = 1.0
    total = ordered.size
    for reverse_rank, index in enumerate(reversed(ordered), start=1):
        rank = total - reverse_rank + 1
        running = min(running, float(p[index]) * total / rank)
        adjusted[rank - 1] = running
    for rank, index in enumerate(ordered):
        out[index] = min(1.0, adjusted[rank])
    return out.tolist()


def validate_difficulty_components(
    value: Any,
    *,
    benchmark_id: str,
    difficulty_score: float,
) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{benchmark_id}: difficulty_components must be a JSON object"
        )
    expected = set(DIFFICULTY_COMPONENTS)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{benchmark_id}: difficulty_components keys are not exact "
            f"(missing={missing}, extra={extra})"
        )

    components: dict[str, float] = {}
    for key in DIFFICULTY_COMPONENTS:
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(
                f"{benchmark_id}: difficulty component {key} is not numeric"
            )
        component = float(raw)
        if not math.isfinite(component) or not 0.0 <= component <= 1.0:
            raise ValueError(
                f"{benchmark_id}: difficulty component {key}={component!r} "
                "is not finite and within [0, 1]"
            )
        components[key] = component

    reconstructed = 100.0 * math.fsum(
        DIFFICULTY_COMPONENT_WEIGHTS[key] * components[key]
        for key in DIFFICULTY_COMPONENTS
    )
    if not math.isclose(
        reconstructed,
        difficulty_score,
        rel_tol=0.0,
        abs_tol=1e-4,
    ):
        raise ValueError(
            f"{benchmark_id}: difficulty components reconstruct score "
            f"{reconstructed:.6f}, not stored score {difficulty_score:.6f}"
        )
    return components


def load_benchmark(path: Path) -> dict[str, dict[str, Any]]:
    benchmark: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        benchmark_id = str(row["benchmark_id"])
        test_cases = row.get("test_cases") or {}
        function_signatures = {
            str(entry.get("signature") or signature)
            for signature, entry in test_cases.items()
            if isinstance(entry, dict) and entry.get("entry_type") == "function"
        }
        semantic_entry_count = len(test_cases)
        if semantic_entry_count <= 0:
            raise ValueError(f"{benchmark_id}: no semantic entries")
        difficulty_score = float(row["difficulty_score"])
        if not math.isfinite(difficulty_score) or not 0.0 <= difficulty_score <= 100.0:
            raise ValueError(
                f"{benchmark_id}: difficulty_score={difficulty_score!r} "
                "is not finite and within [0, 100]"
            )
        difficulty_components = validate_difficulty_components(
            row.get("difficulty_components"),
            benchmark_id=benchmark_id,
            difficulty_score=difficulty_score,
        )
        benchmark[benchmark_id] = {
            "benchmark_id": benchmark_id,
            "difficulty_bin": str(row["difficulty_bin"]),
            "difficulty_score": difficulty_score,
            "difficulty_components": difficulty_components,
            "n_public_functions": len(function_signatures),
            "n_semantic_entries": semantic_entry_count,
            "function_signatures": function_signatures,
        }
    if len(benchmark) != 600:
        raise ValueError(f"expected 600 benchmark contracts, found {len(benchmark)}")
    return benchmark


def load_prompt_lengths(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in read_jsonl(path):
        benchmark_id = str(row["benchmark_id"])
        prompt_chars = row.get("prompt_char_length")
        if not isinstance(prompt_chars, (int, float)) or prompt_chars <= 0:
            raise ValueError(f"{path}: {benchmark_id} has invalid prompt_char_length")
        prompt_tokens = (row.get("usage") or {}).get("prompt_tokens")
        out[benchmark_id] = {
            "prompt_char_length": float(prompt_chars),
            "prompt_tokens": (
                float(prompt_tokens)
                if isinstance(prompt_tokens, (int, float)) and prompt_tokens > 0
                else math.nan
            ),
        }
    return out


def load_original_format(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in read_jsonl(path):
        benchmark_id = str(row["benchmark_id"])
        out[benchmark_id] = float(bool((row.get("format") or {}).get("ok")))
    return out


def load_repaired_evaluation(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in read_jsonl(path):
        benchmark_id = str(row["benchmark_id"])
        compile_pass = float(bool((row.get("compilation") or {}).get("success")))
        abi = row.get("abi") or {}
        abi_evaluated = bool(abi.get("evaluated"))
        abi_f1 = float(abi.get("f1") or 0.0) if abi_evaluated else 0.0
        abi_exact = float(
            abi_evaluated
            and int(abi.get("false_negative") or 0) == 0
            and int(abi.get("false_positive") or 0) == 0
        )
        if not 0.0 <= abi_f1 <= 1.0:
            raise ValueError(f"{path}: {benchmark_id} has ABI F1 outside [0, 1]")
        out[benchmark_id] = {
            "compile_pass": compile_pass,
            "abi_f1_cumulative": abi_f1,
            "abi_exact_cumulative": abi_exact,
            "abi_false_positive": float(abi.get("false_positive") or 0),
            "abi_false_negative": float(abi.get("false_negative") or 0),
        }
    return out


def load_semantic_results(path: Path) -> tuple[dict[str, set[str]], dict[str, Any]]:
    equivalent: dict[str, set[str]] = {}
    summed_equivalent = 0
    for row in stream_top_level_array(path, "results"):
        benchmark_id = str(row["benchmark_id"])
        signatures = {
            str(item["signature"])
            for item in (row.get("function_results") or [])
            if isinstance(item, dict) and item.get("equivalent") and item.get("signature")
        }
        row_total = int(row.get("functions_equivalent") or 0)
        if row_total != len(signatures):
            raise ValueError(
                f"{path}: {benchmark_id} reports {row_total} equivalent functions "
                f"but exposes {len(signatures)} unique equivalent signatures"
            )
        equivalent[benchmark_id] = signatures
        summed_equivalent += row_total

    summary = read_trailing_summary(path)
    summary_total = int(summary.get("functions_equivalent") or 0)
    if summed_equivalent != summary_total:
        raise ValueError(
            f"{path}: streamed equivalent total {summed_equivalent} != summary {summary_total}"
        )
    return equivalent, summary


def join_rows(
    benchmark: Mapping[str, Mapping[str, Any]],
    model_specs: Sequence[ModelSpec],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_integrity: dict[str, Any] = {}
    benchmark_ids = set(benchmark)

    for spec in model_specs:
        prompt = load_prompt_lengths(spec.prompt_path)
        original_format = load_original_format(spec.original_evaluation_path)
        evaluation = load_repaired_evaluation(spec.repaired_evaluation_path)
        semantic, semantic_summary = load_semantic_results(spec.semantic_path)

        for name, values in (
            ("prompt", prompt),
            ("original format evaluation", original_format),
            ("repaired evaluation", evaluation),
        ):
            ids = set(values)
            if ids != benchmark_ids:
                raise ValueError(
                    f"{spec.key}: {name} IDs differ from benchmark "
                    f"(missing={len(benchmark_ids - ids)}, extra={len(ids - benchmark_ids)})"
                )
        if not set(semantic).issubset(benchmark_ids):
            raise ValueError(f"{spec.key}: semantic results contain unknown benchmark IDs")

        semantic_equivalent_sum = 0
        semantic_perfect_sum = 0
        for benchmark_id in sorted(benchmark):
            base = benchmark[benchmark_id]
            eval_row = evaluation[benchmark_id]
            equivalent = semantic.get(benchmark_id, set())
            unexpected = equivalent - set(base["function_signatures"])
            if unexpected:
                raise ValueError(
                    f"{spec.key}: {benchmark_id} has unexpected equivalent signatures: "
                    f"{sorted(unexpected)}"
                )
            semantic_equivalent = len(equivalent)
            semantic_equivalent_sum += semantic_equivalent
            semantic_rate = semantic_equivalent / int(base["n_semantic_entries"])
            semantic_perfect = float(
                int(base["n_public_functions"]) > 0
                and eval_row["abi_exact_cumulative"] == 1.0
                and set(base["function_signatures"]).issubset(equivalent)
            )
            semantic_perfect_sum += int(semantic_perfect)

            joined = {
                "benchmark_id": benchmark_id,
                "model_key": spec.key,
                "model_label": spec.label,
                "difficulty_bin": base["difficulty_bin"],
                "difficulty_score": float(base["difficulty_score"]),
                "difficulty_components": dict(base["difficulty_components"]),
                "n_public_functions": int(base["n_public_functions"]),
                "n_semantic_entries": int(base["n_semantic_entries"]),
                "prompt_char_length": prompt[benchmark_id]["prompt_char_length"],
                "prompt_tokens": prompt[benchmark_id]["prompt_tokens"],
                "semantic_equivalent": semantic_equivalent,
                "semantic_rate_cumulative": semantic_rate,
                "semantic_perfect_cumulative": semantic_perfect,
                "format_pass": original_format[benchmark_id],
                **eval_row,
            }
            for outcome in PRIMARY_OUTCOMES + ROBUSTNESS_OUTCOMES:
                value = float(joined[outcome])
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"{spec.key}: {benchmark_id}: {outcome}={value} outside [0, 1]"
                    )
            rows.append(joined)

        summary_equivalent = int(semantic_summary.get("functions_equivalent") or 0)
        if semantic_equivalent_sum != summary_equivalent:
            raise ValueError(
                f"{spec.key}: joined semantic total {semantic_equivalent_sum} "
                f"!= replay summary {summary_equivalent}"
            )
        observed_totals = {
            "format_passes": int(sum(original_format.values())),
            "compilation_passes": int(
                sum(row["compile_pass"] for row in evaluation.values())
            ),
            "semantic_equivalent": semantic_equivalent_sum,
            "semantic_perfect": semantic_perfect_sum,
        }
        expected_totals = EXPECTED_STAGE_TOTALS[spec.key]
        if observed_totals != expected_totals:
            raise ValueError(
                f"{spec.key}: stage totals differ from published repaired results: "
                f"observed={observed_totals}, expected={expected_totals}"
            )
        model_integrity[spec.key] = {
            "contracts": len(evaluation),
            "stage_totals": observed_totals,
            "semantic_result_contracts": len(semantic),
            "semantic_equivalent": semantic_equivalent_sum,
            "semantic_summary": {
                key: semantic_summary.get(key)
                for key in (
                    "contracts",
                    "valid_contracts",
                    "functions_equivalent",
                    "functions_total",
                    "calls_matched",
                    "calls_total",
                )
            },
        }

    expected_rows = len(benchmark) * len(model_specs)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} joined rows, found {len(rows)}")
    return rows, model_integrity


def zscore(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=0))
    if standard_deviation <= 0:
        raise ValueError("cannot standardize a constant predictor")
    return (values - mean) / standard_deviation, {
        "mean": mean,
        "standard_deviation": standard_deviation,
    }


def prepare_design(
    rows: Sequence[Mapping[str, Any]],
    model_specs: Sequence[ModelSpec],
) -> tuple[np.ndarray, list[str], dict[str, Any], np.ndarray]:
    difficulty_raw = np.asarray([row["difficulty_score"] for row in rows], dtype=float)
    log_prompt_raw = np.log1p(
        np.asarray([row["prompt_char_length"] for row in rows], dtype=float)
    )
    log_functions_raw = np.log1p(
        np.asarray([row["n_public_functions"] for row in rows], dtype=float)
    )

    difficulty, difficulty_stats = zscore(difficulty_raw)
    log_prompt, prompt_stats = zscore(log_prompt_raw)
    log_functions, function_stats = zscore(log_functions_raw)

    columns = [np.ones(len(rows)), difficulty, log_prompt, log_functions]
    names = [
        "intercept",
        "difficulty_score_z",
        "log_prompt_char_length_z",
        "log_public_function_count_z",
    ]
    reference_model = model_specs[0].key
    for spec in model_specs[1:]:
        columns.append(
            np.asarray([float(row["model_key"] == spec.key) for row in rows], dtype=float)
        )
        names.append(f"model_fe[{spec.key}]")

    design = np.column_stack(columns)
    clusters = np.asarray([row["benchmark_id"] for row in rows], dtype=object)
    scaling = {
        "difficulty_score": difficulty_stats,
        "log1p_prompt_char_length": prompt_stats,
        "log1p_public_function_count": function_stats,
        "reference_model": reference_model,
    }
    return design, names, scaling, clusters


def fit_clustered_ols(
    design: np.ndarray,
    outcome: np.ndarray,
    clusters: np.ndarray,
    coefficient_names: Sequence[str],
) -> dict[str, Any]:
    beta, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
    if rank != design.shape[1]:
        raise ValueError(
            f"regression design is rank deficient ({rank} < {design.shape[1]})"
        )
    fitted = design @ beta
    residual = outcome - fitted
    inverse_xtx = np.linalg.inv(design.T @ design)

    unique_clusters = np.unique(clusters)
    meat = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    for cluster in unique_clusters:
        mask = clusters == cluster
        score = design[mask].T @ residual[mask]
        meat += np.outer(score, score)

    n = design.shape[0]
    p = design.shape[1]
    g = unique_clusters.size
    correction = (g / (g - 1)) * ((n - 1) / (n - p))
    covariance = correction * inverse_xtx @ meat @ inverse_xtx
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    z_values = np.divide(
        beta,
        standard_errors,
        out=np.full_like(beta, np.nan),
        where=standard_errors > 0,
    )
    p_values = 2.0 * stats.norm.sf(np.abs(z_values))
    confidence_low = beta - 1.96 * standard_errors
    confidence_high = beta + 1.96 * standard_errors

    centered = outcome - np.mean(outcome)
    denominator = float(centered @ centered)
    r_squared = (
        1.0 - float(residual @ residual) / denominator if denominator > 0 else math.nan
    )

    coefficients = {}
    for index, name in enumerate(coefficient_names):
        coefficients[name] = {
            "estimate": float(beta[index]),
            "cluster_robust_standard_error": float(standard_errors[index]),
            "normal_p_value": float(p_values[index]),
            "cluster_robust_95_ci": [
                float(confidence_low[index]),
                float(confidence_high[index]),
            ],
        }
    return {
        "n_observations": n,
        "n_contract_clusters": int(g),
        "rank": int(rank),
        "r_squared": r_squared,
        "coefficients": coefficients,
    }


def bootstrap_regression_coefficient(
    design: np.ndarray,
    outcome: np.ndarray,
    clusters: np.ndarray,
    coefficient_index: int,
    replicates: int,
    seed: int,
) -> tuple[list[float], int]:
    unique_clusters = np.unique(clusters)
    cluster_rows = [np.flatnonzero(clusters == cluster) for cluster in unique_clusters]
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    failures = 0

    for _ in range(replicates):
        sampled = rng.integers(0, len(cluster_rows), size=len(cluster_rows))
        indices = np.concatenate([cluster_rows[index] for index in sampled])
        beta, _, rank, _ = np.linalg.lstsq(
            design[indices], outcome[indices], rcond=None
        )
        if rank != design.shape[1] or not math.isfinite(beta[coefficient_index]):
            failures += 1
            continue
        estimates.append(float(beta[coefficient_index]))
    return estimates, failures


def spearman_with_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    result = stats.spearmanr(x, y)
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    rng = np.random.default_rng(seed)
    bootstrapped: list[float] = []
    failures = 0
    for _ in range(replicates):
        indices = rng.integers(0, len(x), size=len(x))
        sampled = stats.spearmanr(x[indices], y[indices]).statistic
        if math.isfinite(sampled):
            bootstrapped.append(float(sampled))
        else:
            failures += 1
    return {
        "n": len(x),
        "rho": rho,
        "analytic_p_value": p_value,
        "bootstrap_95_ci": percentile_interval(bootstrapped),
        "bootstrap_successes": len(bootstrapped),
        "bootstrap_failures": failures,
    }


def predictor_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        unique.setdefault(str(row["benchmark_id"]), row)
    ordered = [unique[key] for key in sorted(unique)]
    raw = np.column_stack(
        [
            np.asarray([row["difficulty_score"] for row in ordered], dtype=float),
            np.log1p(
                np.asarray([row["prompt_char_length"] for row in ordered], dtype=float)
            ),
            np.log1p(
                np.asarray([row["n_public_functions"] for row in ordered], dtype=float)
            ),
        ]
    )
    names = (
        "difficulty_score",
        "log1p_prompt_char_length",
        "log1p_public_function_count",
    )
    correlation = np.corrcoef(raw, rowvar=False)

    vif: dict[str, float] = {}
    for index, name in enumerate(names):
        y = raw[:, index]
        other = np.delete(raw, index, axis=1)
        design = np.column_stack([np.ones(len(y)), other])
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        total = y - np.mean(y)
        r_squared = 1.0 - float(residual @ residual) / float(total @ total)
        vif[name] = float(1.0 / (1.0 - r_squared))

    return {
        "n_contracts": len(ordered),
        "predictor_order": list(names),
        "pearson_correlation_matrix": correlation.tolist(),
        "variance_inflation_factor": vif,
    }


def descriptive_results(
    rows: Sequence[Mapping[str, Any]],
    model_specs: Sequence[ModelSpec],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for spec in model_specs:
        selected = [row for row in rows if row["model_key"] == spec.key]
        semantic_numerator = int(sum(row["semantic_equivalent"] for row in selected))
        semantic_denominator = int(sum(row["n_semantic_entries"] for row in selected))
        out[spec.key] = {
            "label": spec.label,
            "contracts": len(selected),
            "format_passes": int(sum(row["format_pass"] for row in selected)),
            "compilation_passes": int(sum(row["compile_pass"] for row in selected)),
            "abi_exact_contracts": int(
                sum(row["abi_exact_cumulative"] for row in selected)
            ),
            "semantic_perfect_contracts": int(
                sum(row["semantic_perfect_cumulative"] for row in selected)
            ),
            "mean_contract_outcomes": {
                outcome: float(np.mean([row[outcome] for row in selected]))
                for outcome in PRIMARY_OUTCOMES + ROBUSTNESS_OUTCOMES
            },
            "semantic_micro": {
                "equivalent": semantic_numerator,
                "entries": semantic_denominator,
                "rate": semantic_numerator / semantic_denominator,
            },
            "prompt_char_length": {
                "median": float(
                    np.median([row["prompt_char_length"] for row in selected])
                ),
                "maximum": float(
                    np.max([row["prompt_char_length"] for row in selected])
                ),
            },
        }
    return out


def run_spearman_analysis(
    rows: Sequence[Mapping[str, Any]],
    model_specs: Sequence[ModelSpec],
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for outcome in PRIMARY_OUTCOMES:
        for spec in model_specs:
            selected = [row for row in rows if row["model_key"] == spec.key]
            x = np.asarray([row["difficulty_score"] for row in selected], dtype=float)
            y = np.asarray([row[outcome] for row in selected], dtype=float)
            result = spearman_with_bootstrap(
                x,
                y,
                replicates,
                stable_seed(seed, "spearman", outcome, spec.key),
            )
            records.append(
                {
                    "outcome": outcome,
                    "model_key": spec.key,
                    "model_label": spec.label,
                    **result,
                }
            )

        by_contract: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            by_contract.setdefault(str(row["benchmark_id"]), []).append(row)
        contract_rows = [by_contract[key] for key in sorted(by_contract)]
        x = np.asarray(
            [contract_group[0]["difficulty_score"] for contract_group in contract_rows],
            dtype=float,
        )
        y = np.asarray(
            [
                np.mean([row[outcome] for row in contract_group])
                for contract_group in contract_rows
            ],
            dtype=float,
        )
        result = spearman_with_bootstrap(
            x,
            y,
            replicates,
            stable_seed(seed, "spearman", outcome, "across_model_mean"),
        )
        records.append(
            {
                "outcome": outcome,
                "model_key": "across_model_mean",
                "model_label": "Across-model mean",
                **result,
            }
        )

    adjusted = benjamini_hochberg(
        [record["analytic_p_value"] for record in records]
    )
    for record, q_value in zip(records, adjusted):
        record["bh_q_value_across_primary_spearman_tests"] = q_value
    return records


def collapse_across_model_outcomes(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not rows:
        raise ValueError("cannot analyze difficulty components without joined rows")

    expected_models = sorted({str(row["model_key"]) for row in rows})
    by_contract: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_contract.setdefault(str(row["benchmark_id"]), []).append(row)

    collapsed: list[dict[str, Any]] = []
    for benchmark_id in sorted(by_contract):
        contract_rows = by_contract[benchmark_id]
        observed_models = [str(row["model_key"]) for row in contract_rows]
        if len(observed_models) != len(set(observed_models)):
            raise ValueError(
                f"{benchmark_id}: duplicate model rows in component analysis"
            )
        if sorted(observed_models) != expected_models:
            raise ValueError(
                f"{benchmark_id}: component analysis model rows are not exact "
                f"(expected={expected_models}, observed={sorted(observed_models)})"
            )

        first = contract_rows[0]
        first_components = validate_difficulty_components(
            first.get("difficulty_components"),
            benchmark_id=benchmark_id,
            difficulty_score=float(first["difficulty_score"]),
        )
        for row in contract_rows[1:]:
            components = validate_difficulty_components(
                row.get("difficulty_components"),
                benchmark_id=benchmark_id,
                difficulty_score=float(row["difficulty_score"]),
            )
            if components != first_components:
                raise ValueError(
                    f"{benchmark_id}: difficulty components differ across model rows"
                )

        mean_outcomes: dict[str, float] = {}
        for outcome in PRIMARY_OUTCOMES:
            values = np.asarray(
                [float(row[outcome]) for row in contract_rows],
                dtype=float,
            )
            if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(
                values > 1.0
            ):
                raise ValueError(
                    f"{benchmark_id}: {outcome} contains a non-finite value "
                    "or a value outside [0, 1]"
                )
            mean_outcomes[outcome] = float(np.mean(values))

        collapsed.append(
            {
                "benchmark_id": benchmark_id,
                "difficulty_components": first_components,
                **mean_outcomes,
            }
        )
    return collapsed, expected_models


def run_component_spearman_analysis(
    rows: Sequence[Mapping[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    contracts, model_keys = collapse_across_model_outcomes(rows)
    records: list[dict[str, Any]] = []

    for component in DIFFICULTY_COMPONENTS:
        x = np.asarray(
            [
                row["difficulty_components"][component]
                for row in contracts
            ],
            dtype=float,
        )
        for outcome in PRIMARY_OUTCOMES:
            y = np.asarray([row[outcome] for row in contracts], dtype=float)
            result = spearman_with_bootstrap(
                x,
                y,
                replicates,
                stable_seed(
                    seed,
                    "component_spearman",
                    component,
                    outcome,
                    "across_model_mean",
                ),
            )
            records.append(
                {
                    "component": component,
                    "component_label": DIFFICULTY_COMPONENT_LABELS[component],
                    "outcome": outcome,
                    "outcome_label": OUTCOME_LABELS[outcome],
                    **result,
                }
            )

    adjusted = benjamini_hochberg(
        [record["analytic_p_value"] for record in records]
    )
    for record, q_value in zip(records, adjusted):
        record[
            "bh_q_value_across_20_component_outcome_tests"
        ] = q_value

    components = []
    for component in DIFFICULTY_COMPONENTS:
        values = np.asarray(
            [
                row["difficulty_components"][component]
                for row in contracts
            ],
            dtype=float,
        )
        components.append(
            {
                "key": component,
                "label": DIFFICULTY_COMPONENT_LABELS[component],
                "definition": DIFFICULTY_COMPONENT_DEFINITIONS[component],
                "weight_in_composite": DIFFICULTY_COMPONENT_WEIGHTS[component],
                "minimum": float(np.min(values)),
                "median": float(np.median(values)),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values)),
                "population_standard_deviation": float(np.std(values, ddof=0)),
                "unique_values": int(np.unique(values).size),
                "zero_values": int(np.count_nonzero(values == 0.0)),
            }
        )

    return {
        "label": "Descriptive difficulty-component associations",
        "analysis_type": "unadjusted Spearman rank association",
        "interpretation": (
            "Descriptive associations with across-model mean cumulative success "
            "outcomes; not adjusted, causal, or independent component importance."
        ),
        "unit_of_analysis": "benchmark contract",
        "contracts": len(contracts),
        "model_keys_averaged": model_keys,
        "outcome_aggregation": (
            "Arithmetic mean of each cumulative outcome across all model rows "
            "for the same contract."
        ),
        "bootstrap": (
            "Deterministic percentile bootstrap resampling complete contracts."
        ),
        "multiple_testing": (
            "Benjamini-Hochberg correction jointly across the five components "
            "and four primary outcomes (20 tests)."
        ),
        "components": components,
        "records": records,
    }


def run_regression_analysis(
    rows: Sequence[Mapping[str, Any]],
    model_specs: Sequence[ModelSpec],
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    design, coefficient_names, scaling, clusters = prepare_design(rows, model_specs)
    outcomes: dict[str, Any] = {}
    difficulty_p_values: list[float] = []

    for outcome_name in PRIMARY_OUTCOMES + ROBUSTNESS_OUTCOMES:
        outcome = np.asarray([row[outcome_name] for row in rows], dtype=float)
        fit = fit_clustered_ols(design, outcome, clusters, coefficient_names)
        bootstrap, failures = bootstrap_regression_coefficient(
            design,
            outcome,
            clusters,
            coefficient_index=coefficient_names.index("difficulty_score_z"),
            replicates=replicates,
            seed=stable_seed(seed, "regression", outcome_name),
        )
        difficulty = fit["coefficients"]["difficulty_score_z"]
        difficulty["cluster_bootstrap_95_ci"] = percentile_interval(bootstrap)
        difficulty["bootstrap_successes"] = len(bootstrap)
        difficulty["bootstrap_failures"] = failures
        difficulty_p_values.append(difficulty["normal_p_value"])
        outcomes[outcome_name] = fit

    q_values = benjamini_hochberg(difficulty_p_values)
    for outcome_name, q_value in zip(
        PRIMARY_OUTCOMES + ROBUSTNESS_OUTCOMES, q_values
    ):
        outcomes[outcome_name]["coefficients"]["difficulty_score_z"][
            "bh_q_value_across_adjusted_outcomes"
        ] = q_value

    specification = {
        "formula": (
            "outcome ~ difficulty_score_z + log1p(prompt_char_length)_z + "
            "log1p(n_public_functions)_z + model fixed effects"
        ),
        "estimator": "ordinary least squares score/linear-probability model",
        "inference": (
            "contract-clustered sandwich standard errors and deterministic "
            "contract-cluster percentile bootstrap"
        ),
        "cluster": "benchmark_id",
        "difficulty_effect_unit": "one population standard deviation",
        "coefficient_order": coefficient_names,
        "predictor_scaling": scaling,
    }
    return outcomes, specification


def as_markdown_number(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "NA"
    if abs(value) < 0.5 * 10 ** (-digits):
        return "0"
    return f"{value:.{digits}f}"


def as_markdown_probability(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def format_interval(interval: Sequence[float], scale: float = 1.0) -> str:
    return (
        f"[{as_markdown_number(float(interval[0]) * scale)}, "
        f"{as_markdown_number(float(interval[1]) * scale)}]"
    )


def render_markdown(results: Mapping[str, Any]) -> str:
    lines = [
        "# Difficulty Validation",
        "",
        "## Scope",
        "",
        (
            "This analysis tests whether the continuous SCDBench difficulty score is "
            "associated with cumulative model outcomes. Format is measured on the "
            "original output; later stages use the artifact after allowed repair. It "
            "is an association analysis, "
            "not evidence that the score is a causal or complete measure of LLM "
            "decompilation hardness."
        ),
        "",
        (
            "All primary outcomes are cumulative. An earlier-stage failure receives "
            "zero at later stages, matching the paper's fixed-denominator protocol."
        ),
        "",
        "## Data And Outcome Checks",
        "",
        "| Model | Format | Compile | Exact ABI | Semantic equivalent | Semantic perfect |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for model_key, item in results["descriptive"].items():
        semantic = item["semantic_micro"]
        lines.append(
            f"| {item['label']} | {item['format_passes']}/600 | "
            f"{item['compilation_passes']}/600 | {item['abi_exact_contracts']}/600 | "
            f"{semantic['equivalent']}/{semantic['entries']} "
            f"({100 * semantic['rate']:.1f}%) | "
            f"{item['semantic_perfect_contracts']}/600 |"
        )

    lines.extend(
        [
            "",
            "Outcome definitions:",
            "",
            "- `format_pass`: original decompilation output passes the format protocol.",
            "- `compile_pass`: artifact after the allowed repair compiles.",
            (
                "- `abi_f1_cumulative`: evaluator per-contract ABI F1; zero if ABI "
                "evaluation is unavailable because a prerequisite failed."
            ),
            (
                "- `semantic_rate_cumulative`: equivalent ground-truth functions "
                "divided by all benchmark semantic entries for that contract, including "
                "fallback/receive entries in the denominator; missing entries score zero."
            ),
            (
                "- Binary robustness outcomes are exact ABI recovery and perfect "
                "contract-level semantic recovery."
            ),
            "",
            "## Unadjusted Spearman Associations",
            "",
            (
                "Negative rho means that higher difficulty is associated with worse "
                "performance. Confidence intervals resample contracts."
            ),
            "",
            "| Outcome | Model | rho | Bootstrap 95% CI | BH q |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )

    for record in results["spearman"]:
        lines.append(
            f"| {OUTCOME_LABELS[record['outcome']]} | {record['model_label']} | "
            f"{as_markdown_number(record['rho'])} | "
            f"{format_interval(record['bootstrap_95_ci'])} | "
            f"{as_markdown_probability(record['bh_q_value_across_primary_spearman_tests'])} |"
        )

    component_analysis = results["difficulty_component_associations"]
    lines.extend(
        [
            "",
            "## Descriptive Difficulty-Component Associations",
            "",
            (
                "The five component values are the frozen benchmark-construction "
                "scores in `[0, 1]`. These results are descriptive, unadjusted "
                "associations, not causal effects or independent component importance."
            ),
            "",
            (
                "Each contract contributes one observation: its cumulative outcome "
                "averaged across the four models. Negative rho means a larger component "
                "is associated with worse performance. Confidence intervals resample "
                "complete contracts, and BH q-values use one family of 20 "
                "component-by-outcome tests."
            ),
            "",
            "| Component | Min | Median | Max | Unique | Zeros |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for component in component_analysis["components"]:
        lines.append(
            f"| {component['label']} | "
            f"{as_markdown_number(component['minimum'])} | "
            f"{as_markdown_number(component['median'])} | "
            f"{as_markdown_number(component['maximum'])} | "
            f"{component['unique_values']} | {component['zero_values']} |"
        )

    lines.extend(
        [
            "",
            "| Component | Outcome | rho | Contract-bootstrap 95% CI | BH q (20 tests) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for record in component_analysis["records"]:
        lines.append(
            f"| {record['component_label']} | {record['outcome_label']} | "
            f"{as_markdown_number(record['rho'])} | "
            f"{format_interval(record['bootstrap_95_ci'])} | "
            f"{as_markdown_probability(record['bh_q_value_across_20_component_outcome_tests'])} |"
        )

    lines.extend(
        [
            "",
            "## Adjusted Model-Fixed-Effect Results",
            "",
            (
                "The table reports the change in outcome percentage points for a "
                "one-standard-deviation increase in difficulty, controlling for "
                "`log1p` prompt characters, `log1p` public-function count, and model "
                "fixed effects. Standard errors are clustered by contract."
            ),
            "",
            "| Outcome | Difficulty effect (pp) | Cluster bootstrap 95% CI | BH q | R2 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for outcome in PRIMARY_OUTCOMES:
        fit = results["regression"]["outcomes"][outcome]
        effect = fit["coefficients"]["difficulty_score_z"]
        lines.append(
            f"| {OUTCOME_LABELS[outcome]} | "
            f"{as_markdown_number(100 * effect['estimate'])} | "
            f"{format_interval(effect['cluster_bootstrap_95_ci'], scale=100)} | "
            f"{as_markdown_probability(effect['bh_q_value_across_adjusted_outcomes'])} | "
            f"{as_markdown_number(fit['r_squared'])} |"
        )

    lines.extend(
        [
            "",
            "## Binary Robustness Outcomes",
            "",
            "| Outcome | Difficulty effect (pp) | Cluster bootstrap 95% CI | BH q | R2 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for outcome in ROBUSTNESS_OUTCOMES:
        fit = results["regression"]["outcomes"][outcome]
        effect = fit["coefficients"]["difficulty_score_z"]
        lines.append(
            f"| {OUTCOME_LABELS[outcome]} | "
            f"{as_markdown_number(100 * effect['estimate'])} | "
            f"{format_interval(effect['cluster_bootstrap_95_ci'], scale=100)} | "
            f"{as_markdown_probability(effect['bh_q_value_across_adjusted_outcomes'])} | "
            f"{as_markdown_number(fit['r_squared'])} |"
        )

    diagnostics = results["predictor_diagnostics"]
    names = diagnostics["predictor_order"]
    matrix = diagnostics["pearson_correlation_matrix"]
    lines.extend(
        [
            "",
            "## Predictor Diagnostics",
            "",
            "| Predictor | VIF |",
            "| --- | ---: |",
        ]
    )
    for name in names:
        lines.append(
            f"| `{name}` | "
            f"{as_markdown_number(diagnostics['variance_inflation_factor'][name])} |"
        )

    lines.extend(
        [
            "",
            "Pearson predictor correlation matrix:",
            "",
            "|  | Difficulty | Log prompt chars | Log public functions |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    short_names = ("Difficulty", "Log prompt chars", "Log public functions")
    for index, name in enumerate(short_names):
        values = " | ".join(as_markdown_number(float(value)) for value in matrix[index])
        lines.append(f"| {name} | {values} |")

    primary_effects = [
        results["regression"]["outcomes"][outcome]["coefficients"][
            "difficulty_score_z"
        ]
        for outcome in PRIMARY_OUTCOMES
    ]
    negative_count = sum(effect["estimate"] < 0 for effect in primary_effects)
    excluded_zero_count = sum(
        effect["cluster_bootstrap_95_ci"][1] < 0 for effect in primary_effects
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"Higher difficulty has a negative adjusted coefficient for "
                f"{negative_count}/{len(PRIMARY_OUTCOMES)} primary outcomes; "
                f"{excluded_zero_count}/{len(PRIMARY_OUTCOMES)} contract-bootstrap "
                "intervals lie fully below zero."
            ),
            "",
            (
                "The adjusted coefficient is the association with the residual part of "
                "the composite difficulty score after controlling for two components "
                "closely related to that score. Predictor correlations and VIFs should "
                "therefore be considered when interpreting coefficient magnitude."
            ),
            "",
            (
                "OLS is used deliberately to estimate an average score/probability "
                "difference with contract-clustered inference. It does not assert that "
                "predictions outside [0, 1] are meaningful, and no causal interpretation "
                "is intended."
            ),
            "",
            "## Reproduction",
            "",
            "```bash",
            (
                "python experiments/difficulty_validation/analyze.py "
                f"--bootstrap-replicates {results['analysis']['bootstrap_replicates']} "
                f"--seed {results['analysis']['seed']}"
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def file_metadata(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        out.append(
            {
                "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "bytes": path.stat().st_size,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise SystemExit("--bootstrap-replicates must be at least 100")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    benchmark = {row["benchmark_id"]: row for row in payload["contracts"]}
    rows = payload["observations"]
    for row in rows:
        if row["prompt_tokens"] is None:
            row["prompt_tokens"] = math.nan
    model_integrity = payload["model_integrity"]
    if len(benchmark) != 600 or len(rows) != 2400:
        raise ValueError("expected 600 contracts and 2400 contract-model observations")
    pairs = {(row["benchmark_id"], row["model_key"]) for row in rows}
    expected_pairs = {(bid, spec.key) for bid in benchmark for spec in MODEL_SPECS}
    if pairs != expected_pairs or len(pairs) != len(rows):
        raise ValueError("contract-model observations contain missing or duplicate pairs")
    for row in rows:
        validate_difficulty_components(row["difficulty_components"],
            benchmark_id=row["benchmark_id"], difficulty_score=row["difficulty_score"])
    descriptive = descriptive_results(rows, MODEL_SPECS)
    spearman = run_spearman_analysis(
        rows,
        MODEL_SPECS,
        args.bootstrap_replicates,
        args.seed,
    )
    component_associations = run_component_spearman_analysis(
        rows,
        args.bootstrap_replicates,
        args.seed,
    )
    regression_outcomes, regression_specification = run_regression_analysis(
        rows,
        MODEL_SPECS,
        args.bootstrap_replicates,
        args.seed,
    )
    diagnostics = predictor_diagnostics(rows)

    input_paths = [args.input]

    results = {
        "schema": "scdbench-rebuttal-difficulty-validation-v1",
        "analysis": {
            "seed": args.seed,
            "bootstrap_replicates": args.bootstrap_replicates,
            "contracts": len(benchmark),
            "models": len(MODEL_SPECS),
            "joined_contract_model_rows": len(rows),
            "primary_outcomes": list(PRIMARY_OUTCOMES),
            "robustness_outcomes": list(ROBUSTNESS_OUTCOMES),
            "difficulty_components": list(DIFFICULTY_COMPONENTS),
        },
        "outcome_definitions": {
            "format_pass": (
                "One when the original decompilation output passes the output format "
                "protocol; zero otherwise. Repair is downstream and cannot erase an "
                "original format success."
            ),
            "compile_pass": (
                "One when the final artifact after repair compiles; zero otherwise."
            ),
            "abi_f1_cumulative": (
                "Per-contract ABI F1 from the evaluator, with zero assigned when "
                "format or compilation failure prevents ABI evaluation."
            ),
            "semantic_rate_cumulative": (
                "Number of ground-truth public-function signatures that pass every "
                "stored semantic case divided by all benchmark semantic entries for "
                "the contract, including fallback and receive entries; missing or "
                "ineligible entries contribute zero."
            ),
            "abi_exact_cumulative": (
                "One when ABI evaluation has no false-positive or false-negative "
                "signatures; zero when recovery is inexact or a prerequisite fails."
            ),
            "semantic_perfect_cumulative": (
                "One when the contract has at least one ABI function, ABI recovery "
                "has no false positives or false negatives, and every ground-truth "
                "ABI function passes all semantic cases; zero otherwise."
            ),
        },
        "model_specs": [
            {"key": spec.key, "label": spec.label} for spec in MODEL_SPECS
        ],
        "input_files": file_metadata(input_paths),
        "integrity": {
            "semantic_entries": sum(
                int(row["n_semantic_entries"]) for row in benchmark.values()
            ),
            "public_functions": sum(
                int(row["n_public_functions"]) for row in benchmark.values()
            ),
            "model_results": model_integrity,
        },
        "descriptive": descriptive,
        "spearman": spearman,
        "difficulty_component_associations": component_associations,
        "regression": {
            "specification": regression_specification,
            "outcomes": regression_outcomes,
        },
        "predictor_diagnostics": diagnostics,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(render_markdown(results), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(args.output_json),
                "markdown": str(args.output_markdown),
                "contracts": len(benchmark),
                "joined_rows": len(rows),
                "bootstrap_replicates": args.bootstrap_replicates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
