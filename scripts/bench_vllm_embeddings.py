#!/usr/bin/env python3
"""
Simple benchmark for a vLLM embeddings endpoint.

The script sends OpenAI-compatible POST requests to `/v1/embeddings` while
sweeping:
1. Approximate input length
2. Target request rate

It reports:
- achieved RPS
- average latency
- p95 latency

By default it uses a synthetic random-text dataset so it can run without any
external files. A plain-text or JSONL dataset can be provided if desired.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import string
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


DEFAULT_RATES = [1.0, 2.0, 4.0, 8.0]
DEFAULT_INPUT_LENGTHS = [32, 128, 512, 1024]
DEFAULT_DATASET_SIZE = 512


@dataclass
class RequestResult:
    ok: bool
    latency_s: float
    status_code: int | None
    error: str | None


@dataclass
class BenchmarkResult:
    input_length: int
    target_rps: float
    requests_sent: int
    requests_ok: int
    requests_failed: int
    elapsed_s: float
    achieved_rps: float
    avg_latency_ms: float
    p95_latency_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a vLLM embeddings endpoint across input lengths and request rates."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the vLLM server.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name exposed by the vLLM server.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional bearer token for OpenAI-compatible auth.",
    )
    parser.add_argument(
        "--input-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_INPUT_LENGTHS,
        help="Approximate token-like input lengths to benchmark.",
    )
    parser.add_argument(
        "--request-rates",
        type=float,
        nargs="+",
        default=DEFAULT_RATES,
        help="Target request rates in requests per second.",
    )
    parser.add_argument(
        "--requests-per-run",
        type=int,
        default=100,
        help="Number of requests to send for each length/rate combination.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=64,
        help="Maximum number of worker threads used to issue requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--dataset",
        choices=["random", "text", "jsonl"],
        default="random",
        help="Input dataset source.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to a text or JSONL dataset file.",
    )
    parser.add_argument(
        "--jsonl-field",
        default="text",
        help="Field to read from each JSONL record when --dataset jsonl is used.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=DEFAULT_DATASET_SIZE,
        help="Number of base samples to keep in memory for the random dataset.",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=5,
        help="Warmup requests sent before the measured runs start.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save full benchmark results as JSON.",
    )
    return parser.parse_args()


def build_random_dataset(dataset_size: int) -> list[str]:
    rng = random.Random(7)
    vocab = [
        "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 10)))
        for _ in range(2048)
    ]
    samples: list[str] = []
    for _ in range(dataset_size):
        length = rng.randint(32, 256)
        samples.append(" ".join(rng.choice(vocab) for _ in range(length)))
    return samples


def load_dataset(args: argparse.Namespace) -> list[str]:
    if args.dataset == "random":
        return build_random_dataset(args.dataset_size)

    if args.dataset_path is None:
        raise ValueError("--dataset-path is required when --dataset is not random.")

    if args.dataset == "text":
        lines = [
            line.strip()
            for line in args.dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            raise ValueError("Text dataset is empty.")
        return lines

    records: list[str] = []
    with args.dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            value = payload.get(args.jsonl_field)
            if isinstance(value, str) and value.strip():
                records.append(value.strip())
            else:
                raise ValueError(
                    f"Missing string field '{args.jsonl_field}' at JSONL line {line_number}."
                )
    if not records:
        raise ValueError("JSONL dataset is empty.")
    return records


def resize_text(text: str, target_length: int) -> str:
    words = text.split()
    if not words:
        words = ["empty"]

    if len(words) >= target_length:
        return " ".join(words[:target_length])

    repeats = (target_length + len(words) - 1) // len(words)
    expanded = (words * repeats)[:target_length]
    return " ".join(expanded)


def build_inputs(dataset: list[str], target_length: int, total_requests: int) -> list[str]:
    return [
        resize_text(dataset[i % len(dataset)], target_length)
        for i in range(total_requests)
    ]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def post_embedding_request(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    text: str,
    timeout: float,
) -> RequestResult:
    body = json.dumps({"model": model, "input": text}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/v1/embeddings",
        data=body,
        headers=headers,
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            _ = response.read()
            latency = time.perf_counter() - start
            return RequestResult(
                ok=True,
                latency_s=latency,
                status_code=response.status,
                error=None,
            )
    except urllib.error.HTTPError as exc:
        latency = time.perf_counter() - start
        details = exc.read().decode("utf-8", errors="replace")
        return RequestResult(
            ok=False,
            latency_s=latency,
            status_code=exc.code,
            error=details,
        )
    except Exception as exc:  # noqa: BLE001
        latency = time.perf_counter() - start
        return RequestResult(
            ok=False,
            latency_s=latency,
            status_code=None,
            error=str(exc),
        )


def run_single_benchmark(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    texts: Iterable[str],
    target_rps: float,
    max_workers: int,
    timeout: float,
) -> tuple[BenchmarkResult, list[RequestResult]]:
    texts = list(texts)
    results: list[RequestResult] = []
    result_lock = threading.Lock()

    start_time = time.perf_counter()

    def task(text: str) -> None:
        result = post_embedding_request(
            base_url=base_url,
            model=model,
            api_key=api_key,
            text=text,
            timeout=timeout,
        )
        with result_lock:
            results.append(result)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for index, text in enumerate(texts):
            scheduled_at = start_time + (index / target_rps)
            delay = scheduled_at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            futures.append(executor.submit(task, text))

        for future in futures:
            future.result()

    elapsed_s = time.perf_counter() - start_time
    latencies_ms = [item.latency_s * 1000.0 for item in results if item.ok]
    requests_ok = sum(1 for item in results if item.ok)
    requests_failed = len(results) - requests_ok

    benchmark = BenchmarkResult(
        input_length=len(texts[0].split()) if texts else 0,
        target_rps=target_rps,
        requests_sent=len(results),
        requests_ok=requests_ok,
        requests_failed=requests_failed,
        elapsed_s=elapsed_s,
        achieved_rps=(requests_ok / elapsed_s) if elapsed_s > 0 else 0.0,
        avg_latency_ms=statistics.fmean(latencies_ms) if latencies_ms else 0.0,
        p95_latency_ms=percentile(latencies_ms, 0.95) if latencies_ms else 0.0,
    )
    return benchmark, results


def warmup(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    texts: list[str],
    timeout: float,
    warmup_requests: int,
) -> None:
    for text in texts[:warmup_requests]:
        post_embedding_request(
            base_url=base_url,
            model=model,
            api_key=api_key,
            text=text,
            timeout=timeout,
        )


def print_result(result: BenchmarkResult) -> None:
    print(
        f"input_len={result.input_length:>5} | "
        f"target_rps={result.target_rps:>6.2f} | "
        f"achieved_rps={result.achieved_rps:>6.2f} | "
        f"avg_latency_ms={result.avg_latency_ms:>8.2f} | "
        f"p95_latency_ms={result.p95_latency_ms:>8.2f} | "
        f"ok={result.requests_ok:>4} | failed={result.requests_failed:>4}"
    )


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args)

    all_results: list[BenchmarkResult] = []

    print("Warming up endpoint...")
    warmup_texts = build_inputs(
        dataset=dataset,
        target_length=min(args.input_lengths),
        total_requests=max(args.warmup_requests, 1),
    )
    warmup(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        texts=warmup_texts,
        timeout=args.timeout,
        warmup_requests=args.warmup_requests,
    )

    print("Starting benchmark sweep...")
    for input_length in args.input_lengths:
        run_inputs = build_inputs(
            dataset=dataset,
            target_length=input_length,
            total_requests=args.requests_per_run,
        )
        for request_rate in args.request_rates:
            result, request_results = run_single_benchmark(
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
                texts=run_inputs,
                target_rps=request_rate,
                max_workers=args.max_workers,
                timeout=args.timeout,
            )
            all_results.append(result)
            print_result(result)

            failed = [item for item in request_results if not item.ok]
            if failed:
                first_failure = failed[0]
                print(
                    "  first_error="
                    f"{first_failure.status_code or 'request_error'} {first_failure.error}"
                )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps([asdict(result) for result in all_results], indent=2),
            encoding="utf-8",
        )
        print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
