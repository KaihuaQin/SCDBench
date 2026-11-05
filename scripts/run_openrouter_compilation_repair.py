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
from typing import Any, Iterable, Sequence

try:
    from run_openrouter_benchmark import (
        OPENROUTER_CHAT_COMPLETIONS_URL,
        load_env_file,
        parse_response_blocks,
    )
except ImportError:
    from scripts.run_openrouter_benchmark import (
        OPENROUTER_CHAT_COMPLETIONS_URL,
        load_env_file,
        parse_response_blocks,
    )


PROMPT_VERSION = "scdbench-openrouter-compilation-repair-v2"

SYSTEM_PROMPT = """You are a specialized Solidity compilation repair engine.

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
- Repair the provided decompilation so it compiles as a single self-contained Solidity file.
- Do not use import statements or external packages.
- If a library, interface, base contract, or error type is needed, define a minimal version directly in the <code> block.
- Fix compiler JSON mistakes when the compiler error indicates invalid settings.
- Preserve the intended ABI and behavior of the initial decompilation as much as possible.
- The <contract_name> block must name the contract artifact that should be compiled and evaluated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair compilation-failing SCDBench model outputs through OpenRouter.")
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-url", default=OPENROUTER_CHAT_COMPLETIONS_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--id", action="append", default=None)
    parser.add_argument("--ids-file", type=Path, default=None)
    parser.add_argument("--extra-request-json", default=None)
    parser.add_argument("--provider-json", default=None)
    parser.add_argument("--site-url", default=None)
    parser.add_argument("--app-title", default="SCDBench")
    parser.add_argument("--store-prompts", action="store_true")
    parser.add_argument("--store-raw-response", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--require-yes-above", type=int, default=20)
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


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                yield json.loads(raw)


def load_id_filter(args: argparse.Namespace) -> set[str] | None:
    ids = set(args.id or [])
    if args.ids_file is not None:
        with args.ids_file.open("r", encoding="utf-8") as handle:
            for raw in handle:
                value = raw.strip()
                if value and not value.startswith("#"):
                    ids.add(value)
    return ids or None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def request_payload(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "benchmark_id": task.get("benchmark_id"),
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "source_content_sha256": task.get("source_content_sha256"),
        "compilation_error_sha256": task.get("compilation_error_sha256"),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_completion_tokens": args.max_completion_tokens,
        "provider": parse_json_arg(args.provider_json, "--provider-json"),
        "extra_request": parse_json_arg(args.extra_request_json, "--extra-request-json"),
    }


def request_key(task: dict[str, Any], args: argparse.Namespace) -> str:
    return sha256_text(json.dumps(request_payload(task, args), sort_keys=True, separators=(",", ":")))


def existing_request_keys(path: Path, retry_errors: bool) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    for row in read_jsonl(path):
        if retry_errors and not row.get("ok"):
            continue
        key = row.get("request_key")
        if key:
            keys.add(str(key))
    return keys


def load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    predictions = {
        str(row.get("benchmark_id")): row
        for row in read_jsonl(args.predictions)
        if row.get("benchmark_id")
    }
    id_filter = load_id_filter(args)
    tasks: list[dict[str, Any]] = []
    for eval_row in read_jsonl(args.evaluations):
        bid = str(eval_row.get("benchmark_id") or "")
        if not bid or (id_filter is not None and bid not in id_filter):
            continue
        fmt = eval_row.get("format") or {}
        comp = eval_row.get("compilation") or {}
        if not fmt.get("ok"):
            continue
        if comp.get("success"):
            continue
        if not comp.get("attempted"):
            continue
        source = predictions.get(bid)
        if source is None:
            continue
        content = str(source.get("content") or "")
        error = str(comp.get("error") or "")
        tasks.append(
            {
                "benchmark_id": bid,
                "difficulty_bin": eval_row.get("difficulty_bin"),
                "family_label": eval_row.get("family_label"),
                "source_model": source.get("model") or eval_row.get("model"),
                "source_response_model": source.get("response_model"),
                "source_finish_reason": source.get("finish_reason"),
                "source_content": content,
                "source_content_sha256": sha256_text(content),
                "source_usage": source.get("usage"),
                "compiler_json": fmt.get("compiler_json"),
                "contract_name": fmt.get("contract_name"),
                "compilation_error": error,
                "compilation_error_sha256": sha256_text(error),
                "compilation_timed_out": bool(comp.get("timed_out")),
                "target_artifact": comp.get("target_artifact"),
            }
        )
    if args.offset:
        tasks = tasks[args.offset :]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    return tasks


def render_user_prompt(task: dict[str, Any]) -> str:
    return (
        "A previous EVM-to-Solidity decompilation failed to compile. "
        "Repair it using the compiler diagnostics below.\n\n"
        "Initial decompilation output:\n"
        "<initial_output>\n"
        f"{task.get('source_content') or ''}\n"
        "</initial_output>\n\n"
        "Compilation error from solc:\n"
        "<compilation_error>\n"
        f"{task.get('compilation_error') or ''}\n"
        "</compilation_error>\n\n"
        "Return only the repaired <compiler>, <code>, and <contract_name> blocks. "
        "Do not explain the changes."
    )


def build_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(task)},
    ]


def prompt_sha256(messages: Sequence[dict[str, str]]) -> str:
    return sha256_text(json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


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


def sleep_before_retry(attempt: int, base_delay: float) -> None:
    delay = min(base_delay * (2 ** max(0, attempt - 1)), 60.0)
    delay += random.uniform(0.0, min(1.0, delay * 0.1))
    time.sleep(delay)


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
                return json.loads(response.read().decode("utf-8")), attempt
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


def make_request_body(
    task: dict[str, Any],
    model: str,
    messages: Sequence[dict[str, str]],
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
            "benchmark_id": str(task.get("benchmark_id")),
            "prompt_version": PROMPT_VERSION,
            "repair_of": str(task.get("source_content_sha256")),
        },
    }
    if args.top_p is not None:
        body["top_p"] = args.top_p
    if args.seed is not None:
        body["seed"] = args.seed
    if provider is not None:
        body["provider"] = provider
    if extra_request is not None:
        body.update(extra_request)
    return body


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def base_result(task: dict[str, Any], messages: Sequence[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "benchmark_id": task.get("benchmark_id"),
        "difficulty_bin": task.get("difficulty_bin"),
        "family_label": task.get("family_label"),
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "request_key": request_key(task, args),
        "prompt_sha256": prompt_sha256(messages),
        "prompt_char_length": sum(len(message["content"]) for message in messages),
        "source_model": task.get("source_model"),
        "source_response_model": task.get("source_response_model"),
        "source_finish_reason": task.get("source_finish_reason"),
        "source_content_sha256": task.get("source_content_sha256"),
        "compilation_error_sha256": task.get("compilation_error_sha256"),
        "compilation_error": task.get("compilation_error"),
        "source_compiler_json": task.get("compiler_json"),
        "source_contract_name": task.get("contract_name"),
        "request_parameters": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_completion_tokens": args.max_completion_tokens,
            "provider": parse_json_arg(args.provider_json, "--provider-json"),
            "extra_request": parse_json_arg(args.extra_request_json, "--extra-request-json"),
        },
    }


def run_one(
    task: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
    provider: dict[str, Any] | None,
    extra_request: dict[str, Any] | None,
) -> dict[str, Any]:
    messages = build_messages(task)
    result = base_result(task, messages, args)
    if args.store_prompts:
        result["messages"] = list(messages)

    headers = make_headers(args, api_key)
    body = make_request_body(task, args.model, messages, args, provider, extra_request)
    started = time.monotonic()
    try:
        response, attempts = post_json_with_retries(
            args.api_url,
            headers,
            body,
            timeout=args.timeout,
            retries=args.retries,
            base_delay=args.retry_base_delay,
        )
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning_fields: dict[str, Any] = {}
        for key in ("reasoning", "reasoning_content", "reasoning_details"):
            if message.get(key) is not None:
                reasoning_fields[key] = message.get(key)
        result.update(
            {
                "ok": True,
                "created_at": now_iso(),
                "latency_seconds": round(time.monotonic() - started, 3),
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
        result.update(
            {
                "ok": False,
                "created_at": now_iso(),
                "latency_seconds": round(time.monotonic() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return result


def write_jsonl_line(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def main() -> None:
    args = parse_args()
    loaded = load_env_file(Path(".env"))
    if loaded:
        print(f"[env] loaded {loaded} variable(s) from .env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set after loading .env")

    provider = parse_json_arg(args.provider_json, "--provider-json")
    extra_request = parse_json_arg(args.extra_request_json, "--extra-request-json")
    tasks = load_tasks(args)
    completed = existing_request_keys(args.output, args.retry_errors)
    tasks = [task for task in tasks if request_key(task, args) not in completed]
    print(f"[load] repair_tasks={len(tasks)} existing={len(completed)} output={args.output}")

    if args.dry_run:
        if tasks:
            messages = build_messages(tasks[0])
            print(json.dumps({"first_benchmark_id": tasks[0]["benchmark_id"], "messages": messages}, indent=2)[:5000])
        return
    if len(tasks) > args.require_yes_above and not args.yes:
        raise SystemExit(f"Refusing to run {len(tasks)} paid requests without --yes")

    lock = threading.Lock()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {executor.submit(run_one, task, args, api_key, provider, extra_request): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                row = {
                    "benchmark_id": task.get("benchmark_id"),
                    "model": args.model,
                    "request_key": request_key(task, args),
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            write_jsonl_line(args.output, row, lock)
            done += 1
            if done == 1 or done % 10 == 0 or done == len(tasks):
                print(f"[repair] {done}/{len(tasks)} last={task.get('benchmark_id')} ok={row.get('ok')}")


if __name__ == "__main__":
    main()
