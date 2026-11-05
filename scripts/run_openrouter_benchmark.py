#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence


OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_VERSION = "scdbench-openrouter-v2-selfcontained"

SYSTEM_PROMPT = """You are a specialized engine for decompiling EVM assembly into Solidity.

Your entire response must consist of exactly three XML-style blocks in this exact order:
<compiler></compiler>
<code></code>
<contract_name></contract_name>

The <compiler> block must contain a single JSON object. Example:
<compiler>
{"version":"v0.8.20","settings":{"optimizer":{"enabled":true,"runs":200},"evmVersion":"shanghai"}}
</compiler>

Rules:
- Do not output Markdown fences, explanations, notes, or any text outside the three required blocks.
- The three blocks must be top-level blocks. Do not place <compiler>, <code>, or <contract_name> tags inside another block.
- Close the <code> block before starting the <contract_name> block.
- Put complete, self-contained Solidity source in the <code> block, including the pragma and the main contract.
- Do not use import statements. Do not reference external packages such as OpenZeppelin, Solmate, Uniswap, Chainlink, or project-local files.
- If a library, interface, base contract, or error type is needed, define the minimal required version directly inside the <code> block.
- The <code> block must be compilable as a single Solidity file using only built-in Solidity/EVM features and definitions present in that same block.
- Infer a plausible compiler version and settings from the assembly when possible. If a field cannot be inferred, use null for that field in the compiler JSON.
- Use the function and event signature mappings only as uncertain hints. They may be incomplete, ambiguous, or non-unique.
- Prefer readable, compiling Solidity over literal opcode-by-opcode comments."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SCDBench contracts against OpenRouter chat-completion models."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=Path,
        default=None,
        help="Benchmark JSONL path. May be repeated. Defaults to the SCDBench sample.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="OpenRouter model id. May be repeated or comma-separated, e.g. openai/gpt-4.1,anthropic/claude-sonnet-4.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/openrouter_predictions.jsonl"),
        help="JSONL output for model predictions.",
    )
    parser.add_argument("--api-url", default=OPENROUTER_CHAT_COMPLETIONS_URL)
    parser.add_argument("--split", action="append", default=None, help="Optional split filter, e.g. benchmark.")
    parser.add_argument("--id", action="append", default=None, help="Benchmark id to run. May be repeated.")
    parser.add_argument("--ids-file", type=Path, default=None, help="Optional text file with one benchmark id per line.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many filtered benchmark rows before running.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of filtered benchmark rows to run.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Request reasoning from OpenRouter using {'reasoning': {'enabled': true}}.",
    )
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=None,
        help="Skip rows whose rendered user prompt exceeds this many characters. Default: no limit.",
    )
    parser.add_argument(
        "--max-signatures-per-selector",
        type=int,
        default=None,
        help="Cap possible signatures shown per selector/topic. Default: include all.",
    )
    parser.add_argument(
        "--provider-json",
        default=None,
        help="Optional OpenRouter provider routing JSON, e.g. '{\"order\":[\"Anthropic\"]}'.",
    )
    parser.add_argument(
        "--extra-request-json",
        default=None,
        help="Optional JSON object merged into every OpenRouter request body.",
    )
    parser.add_argument("--site-url", default=None, help="Optional HTTP-Referer header for OpenRouter rankings.")
    parser.add_argument("--app-title", default="SCDBench", help="Optional X-Title header.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="When resuming, retry existing failed records instead of skipping them.",
    )
    parser.add_argument(
        "--store-prompts",
        action="store_true",
        help="Store rendered messages in the prediction JSONL. This substantially increases output size.",
    )
    parser.add_argument(
        "--store-raw-response",
        action="store_true",
        help="Store the full OpenRouter JSON response in addition to extracted content and usage.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm an actual paid API run. Required for more than --require-yes-above requests.",
    )
    parser.add_argument("--require-yes-above", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Render prompts and print a summary without API calls.")
    parser.add_argument("--dry-run-output", type=Path, default=None, help="Optional JSONL file for dry-run prompts.")
    parser.add_argument("--preview-chars", type=int, default=4000, help="Prompt preview characters for dry runs.")
    return parser.parse_args()


def parse_json_arg(raw: str | None, name: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{name} must be a JSON object")
    return value


def parse_models(values: Sequence[str] | None) -> list[str]:
    models: list[str] = []
    for value in values or []:
        models.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(models))


def parse_dotenv_line(raw: str) -> tuple[str, str] | None:
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_env_file(path: Path) -> int:
    if not path.exists():
        return 0
    loaded = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            parsed = parse_dotenv_line(raw)
            if parsed is None:
                continue
            key, value = parsed
            if key not in os.environ:
                os.environ[key] = value
                loaded += 1
    return loaded


def default_datasets() -> list[Path]:
    return [Path("../dataset/scdbench.jsonl")]


def load_id_filter(args: argparse.Namespace) -> set[str] | None:
    ids = set(args.id or [])
    if args.ids_file is not None:
        with args.ids_file.open("r", encoding="utf-8") as f:
            for raw in f:
                value = raw.strip()
                if value and not value.startswith("#"):
                    ids.add(value)
    return ids or None


def load_benchmark_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = args.dataset or default_datasets()
    split_filter = set(args.split or [])
    id_filter = load_id_filter(args)
    rows: list[dict[str, Any]] = []
    for path in datasets:
        with path.open("r", encoding="utf-8") as f:
            for line_index, raw in enumerate(f, start=1):
                row = json.loads(raw)
                if split_filter and row.get("split") not in split_filter:
                    continue
                if id_filter is not None and row.get("benchmark_id") not in id_filter:
                    continue
                item = dict(row)
                item["_dataset_path"] = str(path)
                item["_dataset_line"] = line_index
                rows.append(item)
    if args.offset:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def format_mapping_lines(mapping: dict[str, list[str]], max_signatures: int | None) -> list[str]:
    lines = []
    for key in sorted(mapping):
        values = mapping.get(key) or []
        if max_signatures is not None:
            values = values[:max_signatures]
        joined = " | ".join(values)
        lines.append(f"{key}: {joined}")
    return lines


def render_user_prompt(row: dict[str, Any], max_signatures: int | None) -> str:
    assembly = row.get("assembly") or ""
    prompt = (
        "Decompile the following EVM assembly into Solidity.\n\n"
        "Assembly:\n"
        "<assembly>\n"
        f"{assembly}\n"
        "</assembly>"
    )
    functions = row.get("functions") or {}
    events = row.get("events") or {}
    if functions or events:
        prompt += "\n\nPossible non-unique, uncertain function/event signature mappings:"
        if functions:
            prompt += "\nFunctions:\n" + "\n".join(format_mapping_lines(functions, max_signatures))
        if events:
            prompt += "\nEvents:\n" + "\n".join(format_mapping_lines(events, max_signatures))
    return prompt


def build_messages(row: dict[str, Any], max_signatures: int | None) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(row, max_signatures)},
    ]


def prompt_sha256(messages: Sequence[dict[str, str]]) -> str:
    serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def request_variant_payload(row: dict[str, Any], model: str, args: argparse.Namespace) -> dict[str, Any]:
    provider = parse_json_arg(args.provider_json, "--provider-json")
    extra_request = parse_json_arg(args.extra_request_json, "--extra-request-json")
    return {
        "benchmark_id": row.get("benchmark_id"),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "reasoning": args.reasoning,
        "max_completion_tokens": args.max_completion_tokens,
        "max_signatures_per_selector": args.max_signatures_per_selector,
        "provider": provider,
        "extra_request": extra_request,
    }


def make_request_key(row: dict[str, Any], model: str, args: argparse.Namespace) -> str:
    payload = request_variant_payload(row, model, args)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def legacy_existing_key(row: dict[str, Any]) -> str | None:
    model = row.get("model")
    benchmark_id = row.get("benchmark_id")
    if not model or not benchmark_id:
        return None
    request_parameters = row.get("request_parameters") or {}
    payload = {
        "benchmark_id": benchmark_id,
        "model": model,
        "prompt_version": row.get("prompt_version", PROMPT_VERSION),
        "temperature": request_parameters.get("temperature"),
        "top_p": request_parameters.get("top_p"),
        "seed": request_parameters.get("seed"),
        "reasoning": request_parameters.get("reasoning", False),
        "max_completion_tokens": request_parameters.get("max_completion_tokens"),
        "max_signatures_per_selector": row.get("max_signatures_per_selector"),
        "provider": request_parameters.get("provider"),
        "extra_request": request_parameters.get("extra_request"),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def existing_keys(path: Path, retry_errors: bool) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if retry_errors and not row.get("ok"):
                continue
            request_key = row.get("request_key") or legacy_existing_key(row)
            if request_key:
                keys.add(request_key)
    return keys


def strip_json_fence(value: str) -> str:
    value = value.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return value


def parse_response_blocks(content: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "format_ok": False,
        "compiler": None,
        "compiler_json": None,
        "code": None,
        "contract_name": None,
        "code_has_import": False,
        "import_lines": [],
        "errors": [],
    }
    strict = re.fullmatch(
        r"\s*<compiler>\s*(.*?)\s*</compiler>\s*"
        r"<code>\s*(.*?)\s*</code>\s*"
        r"<contract_name>\s*(.*?)\s*</contract_name>\s*",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if strict:
        parsed["compiler"] = strict.group(1).strip()
        parsed["code"] = strict.group(2).strip()
        parsed["contract_name"] = strict.group(3).strip()
        parsed["format_ok"] = True
        compiler_raw = strip_json_fence(parsed["compiler"] or "")
        try:
            parsed["compiler_json"] = json.loads(compiler_raw)
        except json.JSONDecodeError as exc:
            parsed["format_ok"] = False
            parsed["errors"].append(f"compiler block is not valid JSON: {exc}")
    else:
        names = ["compiler", "code", "contract_name"]
        for name in names:
            matches = list(re.finditer(rf"<{name}>\s*(.*?)\s*</{name}>", content, flags=re.DOTALL | re.IGNORECASE))
            if len(matches) != 1:
                parsed["errors"].append(f"expected exactly one top-level <{name}> block, found {len(matches)}")
            elif parsed.get(name) is None:
                parsed[name] = matches[0].group(1).strip()
        parsed["errors"].append(
            "response must exactly match <compiler>...</compiler><code>...</code><contract_name>...</contract_name>"
        )
    code = parsed.get("code") or ""
    import_lines = [
        line.strip()
        for line in code.splitlines()
        if re.match(r"^\s*import(?:\s|['\"])", line)
    ]
    parsed["import_lines"] = import_lines
    parsed["code_has_import"] = bool(import_lines)
    return parsed


def make_request_body(
    model: str,
    messages: Sequence[dict[str, str]],
    row: dict[str, Any],
    args: argparse.Namespace,
    provider: dict[str, Any] | None,
    extra_request: dict[str, Any] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": args.temperature,
        "max_completion_tokens": args.max_completion_tokens,
        "stream": False,
        "metadata": {
            "benchmark_id": str(row.get("benchmark_id")),
            "split": str(row.get("split")),
            "prompt_version": PROMPT_VERSION,
        },
    }
    if args.top_p is not None:
        body["top_p"] = args.top_p
    if args.seed is not None:
        body["seed"] = args.seed
    if args.reasoning:
        body["reasoning"] = {"enabled": True}
    if provider is not None:
        body["provider"] = provider
    if extra_request is not None:
        body.update(extra_request)
    return body


def make_headers(args: argparse.Namespace, api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if args.site_url:
        headers["HTTP-Referer"] = args.site_url
    if args.app_title:
        headers["X-Title"] = args.app_title
    return headers


def should_retry_status(status: int) -> bool:
    return status in {408, 409, 425, 429, 500, 502, 503, 504}


def post_json_with_retries(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float,
    retries: int,
    base_delay: float,
) -> tuple[dict[str, Any], int]:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    attempt = 0
    while True:
        attempt += 1
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw), attempt
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            if attempt <= retries and should_retry_status(exc.code):
                sleep_before_retry(attempt, base_delay)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {raw[:2000]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt <= retries:
                sleep_before_retry(attempt, base_delay)
                continue
            raise RuntimeError(f"request failed after {attempt} attempts: {exc}") from exc


def sleep_before_retry(attempt: int, base_delay: float) -> None:
    delay = base_delay * (2 ** max(0, attempt - 1))
    delay = min(delay, 60.0)
    delay += random.uniform(0.0, min(1.0, delay * 0.1))
    time.sleep(delay)


def base_result(row: dict[str, Any], model: str, messages: Sequence[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    user_prompt = messages[1]["content"]
    result = {
        "benchmark_id": row.get("benchmark_id"),
        "model": model,
        "request_key": make_request_key(row, model, args),
        "split": row.get("split"),
        "difficulty_bin": row.get("difficulty_bin"),
        "family_label": row.get("family_label"),
        "compiler_version": row.get("compiler_version"),
        "near_duplicate_cluster_id": row.get("near_duplicate_cluster_id"),
        "dataset_path": row.get("_dataset_path"),
        "dataset_line": row.get("_dataset_line"),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(messages),
        "prompt_char_length": sum(len(message["content"]) for message in messages),
        "user_prompt_char_length": len(user_prompt),
        "assembly_char_length": len(row.get("assembly") or ""),
        "function_mapping_count": len(row.get("functions") or {}),
        "event_mapping_count": len(row.get("events") or {}),
        "max_signatures_per_selector": args.max_signatures_per_selector,
        "request_parameters": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "reasoning": args.reasoning,
            "max_completion_tokens": args.max_completion_tokens,
            "provider": parse_json_arg(args.provider_json, "--provider-json"),
            "extra_request": parse_json_arg(args.extra_request_json, "--extra-request-json"),
        },
    }
    if args.store_prompts:
        result["messages"] = list(messages)
    return result


def run_one(
    row: dict[str, Any],
    model: str,
    args: argparse.Namespace,
    api_key: str,
    provider: dict[str, Any] | None,
    extra_request: dict[str, Any] | None,
) -> dict[str, Any]:
    messages = build_messages(row, args.max_signatures_per_selector)
    result = base_result(row, model, messages, args)
    if args.max_input_chars is not None and result["user_prompt_char_length"] > args.max_input_chars:
        result.update(
            {
                "ok": False,
                "skipped": True,
                "error_type": "input_too_long",
                "error": f"user prompt has {result['user_prompt_char_length']} chars > {args.max_input_chars}",
                "created_at": now_iso(),
            }
        )
        return result

    headers = make_headers(args, api_key)
    body = make_request_body(model, messages, row, args, provider, extra_request)
    start = time.monotonic()
    try:
        response, attempts = post_json_with_retries(
            args.api_url,
            headers,
            body,
            timeout=args.timeout,
            retries=args.retries,
            base_delay=args.retry_base_delay,
        )
        latency = time.monotonic() - start
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning_fields = {}
        for key in ("reasoning", "reasoning_content", "reasoning_details"):
            if message.get(key) is not None:
                reasoning_fields[key] = message.get(key)
        result.update(
            {
                "ok": True,
                "skipped": False,
                "created_at": now_iso(),
                "latency_seconds": round(latency, 3),
                "attempts": attempts,
                "response_id": response.get("id"),
                "response_model": response.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"),
                "content": content,
                "parsed": parse_response_blocks(content),
                **reasoning_fields,
            }
        )
        if args.store_raw_response:
            result["raw_response"] = response
    except Exception as exc:  # noqa: BLE001 - keep batch runs moving.
        latency = time.monotonic() - start
        result.update(
            {
                "ok": False,
                "skipped": False,
                "created_at": now_iso(),
                "latency_seconds": round(latency, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return result


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def make_tasks(rows: Sequence[dict[str, Any]], models: Sequence[str]) -> list[tuple[dict[str, Any], str]]:
    return [(row, model) for row in rows for model in models]


def write_jsonl_line(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def dry_run(tasks: Sequence[tuple[dict[str, Any], str]], args: argparse.Namespace) -> int:
    print(f"[dry-run] benchmark_rows={len({row['benchmark_id'] for row, _ in tasks})} requests={len(tasks)}")
    total_prompt_chars = 0
    max_prompt_chars = 0
    preview_printed = False
    out_f = None
    if args.dry_run_output is not None:
        args.dry_run_output.parent.mkdir(parents=True, exist_ok=True)
        out_f = args.dry_run_output.open("w", encoding="utf-8")
    try:
        for row, model in tasks:
            messages = build_messages(row, args.max_signatures_per_selector)
            result = base_result(row, model, messages, args)
            result["dry_run"] = True
            result["messages"] = messages
            total_prompt_chars += result["prompt_char_length"]
            max_prompt_chars = max(max_prompt_chars, result["prompt_char_length"])
            if out_f is not None:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            if not preview_printed:
                preview_printed = True
                print(f"[dry-run] first benchmark_id={row.get('benchmark_id')} model={model}")
                preview = messages[1]["content"][: args.preview_chars]
                if len(messages[1]["content"]) > args.preview_chars:
                    preview += "\n...[preview truncated]..."
                print(preview)
        if tasks:
            print(f"[dry-run] avg_prompt_chars={total_prompt_chars / len(tasks):.1f} max_prompt_chars={max_prompt_chars}")
    finally:
        if out_f is not None:
            out_f.close()
    return 0


def main() -> int:
    args = parse_args()
    loaded_env = load_env_file(Path(".env"))
    if loaded_env:
        print(f"[env] loaded {loaded_env} variable(s) from .env", flush=True)
    models = parse_models(args.model)
    if not models:
        raise SystemExit("at least one --model is required")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    provider = parse_json_arg(args.provider_json, "--provider-json")
    extra_request = parse_json_arg(args.extra_request_json, "--extra-request-json")

    rows = load_benchmark_rows(args)
    tasks = make_tasks(rows, models)
    if args.dry_run:
        return dry_run(tasks, args)

    completed = existing_keys(args.output, retry_errors=args.retry_errors)
    tasks = [(row, model) for row, model in tasks if make_request_key(row, model, args) not in completed]
    if not tasks:
        print("[run] no pending requests")
        return 0
    if len(tasks) > args.require_yes_above and not args.yes:
        raise SystemExit(
            f"refusing to start {len(tasks)} API requests without --yes "
            f"(adjust with --require-yes-above if needed)"
        )
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("missing API key: set OPENROUTER_API_KEY in .env")

    print(
        f"[run] rows={len(rows)} models={len(models)} pending_requests={len(tasks)} "
        f"output={args.output} concurrency={args.concurrency}",
        flush=True,
    )
    lock = threading.Lock()
    done = 0
    ok = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(run_one, row, model, args, api_key, provider, extra_request)
            for row, model in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            write_jsonl_line(args.output, result, lock)
            done += 1
            if result.get("ok"):
                ok += 1
            else:
                failed += 1
            print(
                f"[run] {done}/{len(tasks)} ok={ok} failed={failed} "
                f"model={result.get('model')} id={result.get('benchmark_id')}",
                flush=True,
            )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
