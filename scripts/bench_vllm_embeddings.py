#!/usr/bin/env python3
"""Run embedding benchmark sweeps with `vllm bench serve`."""

from __future__ import annotations

import os
import sys
import json
import argparse
import subprocess
import urllib.error
import urllib.request
from typing import Any
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import safe_filename

DEFAULT_INPUT_LENGTHS = [32, 128, 512, 1024]
DEFAULT_REQUEST_RATES = [1, 2, 4, 8]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark sweep."""
    parser = argparse.ArgumentParser(
        description="Benchmark a vLLM embedding endpoint with vllm bench serve."
    )
    parser.add_argument("--model", required=True, help="Served model name.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running vLLM server.",
    )
    parser.add_argument(
        "--input-lengths",
        nargs="+",
        type=int,
        default=DEFAULT_INPUT_LENGTHS,
        help="Input lengths to benchmark.",
    )
    parser.add_argument(
        "--request-rates",
        nargs="+",
        type=int,
        default=DEFAULT_REQUEST_RATES,
        help="Request rates to benchmark.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of requests per benchmark run.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. Defaults to request-rate.",
    )
    parser.add_argument(
        "--dataset-name",
        default="random",
        help="Dataset name passed to vllm bench serve.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional dataset path for non-random datasets.",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=3,
        help="Number of warmup requests for each run.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key for the OpenAI-compatible server.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/vllm_embeddings"),
        help="Directory where vLLM result JSON files are written.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the compact benchmark summary.",
    )
    return parser.parse_args()


def append_arg(command: list[str], flag: str, value: str | int | float | None) -> None:
    """Append a CLI flag and value when the value is not None."""
    if value is None:
        return
    command.extend([flag, str(value)])


def check_server_ready(base_url: str) -> None:
    """Raise a clear error if the configured vLLM server is not reachable."""
    health_url = f"{base_url.rstrip('/')}/health"
    request = urllib.request.Request(health_url, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status >= 400:
                raise RuntimeError(
                    f"vLLM server check failed at {health_url} with status {response.status}."
                )
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"No vLLM server found at {base_url}. Start the container before running the benchmark."
        ) from exc


def build_command(
    args: argparse.Namespace,
    input_length: int,
    request_rate: int,
    result_filename: str,
) -> list[str]:
    """Build one `vllm bench serve` command for embeddings."""
    max_concurrency = (
        args.max_concurrency
        if args.max_concurrency is not None
        else max(1, request_rate)
    )

    command = [
        "vllm",
        "bench",
        "serve",
        "--model",
        args.model,
        "--backend",
        "openai-embeddings",
        "--endpoint",
        "/v1/embeddings",
        "--base-url",
        args.base_url,
        "--dataset-name",
        args.dataset_name,
        "--num-prompts",
        str(args.num_prompts),
        "--request-rate",
        str(request_rate),
        "--max-concurrency",
        str(max_concurrency),
        "--input-len",
        str(input_length),
        "--num-warmups",
        str(args.num_warmups),
        "--percentile-metrics",
        "e2el",
        "--metric-percentiles",
        "95",
        "--save-result",
        "--result-dir",
        str(args.result_dir),
        "--result-filename",
        result_filename,
        "--disable-tqdm",
        "--save-detailed",
    ]

    append_arg(command, "--dataset-path", args.dataset_path)

    if args.api_key:
        command.extend(["--header", f"Authorization: Bearer {args.api_key}"])

    return command


def clean_benchmark_json_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load a benchmark JSON, remove `input_lens`, keep only unique non-empty
    error messages, and optionally save the cleaned result.
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    data.pop("input_lens", None)

    errors = data.get("errors", [])
    data["errors"] = sorted(
        {err.strip() for err in errors if isinstance(err, str) and err.strip()}
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_single_benchmark(
    args: argparse.Namespace,
    input_length: int,
    request_rate: int,
):
    """Run one benchmark command and return the extracted metrics."""
    model_name = safe_filename(args.model)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_filename = f"{model_name}--{args.num_prompts}prompts--{input_length}tk--{request_rate}rps--{timestamp}.json"
    command = build_command(args, input_length, request_rate, result_filename)

    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)

    clean_benchmark_json_file(os.path.join(args.result_dir, result_filename))


def main() -> None:
    """Run the full benchmark sweep."""
    args = parse_args()
    check_server_ready(args.base_url)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    for input_length in args.input_lengths:
        for request_rate in args.request_rates:
            run_single_benchmark(args, input_length, request_rate)


if __name__ == "__main__":
    main()
