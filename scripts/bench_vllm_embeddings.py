#!/usr/bin/env python3
"""Run embedding benchmark sweeps with `vllm bench serve`."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


DEFAULT_INPUT_LENGTHS = [32, 128, 512, 1024]
DEFAULT_REQUEST_RATES = [1.0, 2.0, 4.0, 8.0]


@dataclass
class BenchmarkSummary:
    """Store the metrics extracted from one benchmark run."""

    input_length: int
    target_rps: float
    achieved_rps: float
    latency_ms: float
    p95_latency_ms: float
    result_file: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark sweep."""
    parser = argparse.ArgumentParser(
        description="Benchmark a vLLM embedding endpoint with vllm bench serve."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Optional YAML config path or config name under configs/embedders.",
    )
    parser.add_argument("--model", default=None, help="Served model name.")
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
        type=float,
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
        default=32,
        help="Maximum number of concurrent requests.",
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


def load_config(config_path: Path | None) -> dict[str, Any]:
    """Load benchmark defaults from a YAML config file."""

    assert config_path.exists(), FileNotFoundError(
        f"Config file not found: {config_path}"
    )

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def apply_config_defaults(
    args: argparse.Namespace, config: dict[str, Any]
) -> argparse.Namespace:
    """Apply YAML config values only when the CLI did not override them."""
    if args.model is None:
        args.model = config.get("model")

    if args.base_url == "http://localhost:8000":
        port = config.get("port")
        if port is not None:
            args.base_url = f"http://localhost:{port}"

    if args.api_key is None:
        args.api_key = config.get("api_key")

    if (
        args.input_lengths == DEFAULT_INPUT_LENGTHS
        and config.get("input_lengths") is not None
    ):
        args.input_lengths = config["input_lengths"]

    if (
        args.request_rates == DEFAULT_REQUEST_RATES
        and config.get("request_rates") is not None
    ):
        args.request_rates = config["request_rates"]

    if args.num_prompts == 100 and config.get("num_prompts") is not None:
        args.num_prompts = config["num_prompts"]

    if args.max_concurrency == 32 and config.get("max_concurrency") is not None:
        args.max_concurrency = config["max_concurrency"]

    if args.dataset_name == "random" and config.get("dataset_name") is not None:
        args.dataset_name = config["dataset_name"]

    if args.dataset_path is None and config.get("dataset_path") is not None:
        args.dataset_path = Path(config["dataset_path"])

    if args.num_warmups == 3 and config.get("num_warmups") is not None:
        args.num_warmups = config["num_warmups"]

    if (
        args.result_dir == Path("results/vllm_embeddings")
        and config.get("result_dir") is not None
    ):
        args.result_dir = Path(config["result_dir"])

    if args.output_json is None and config.get("output_json") is not None:
        args.output_json = Path(config["output_json"])

    if not args.model:
        raise ValueError("A model must be set with --model or in the config file.")

    return args


def append_arg(command: list[str], flag: str, value: str | int | float | None) -> None:
    """Append a CLI flag and value when the value is not None."""
    if value is None:
        return
    command.extend([flag, str(value)])


def build_command(
    args: argparse.Namespace,
    input_length: int,
    request_rate: float,
    result_filename: str,
) -> list[str]:
    """Build one `vllm bench serve` command for embeddings."""
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
        str(args.max_concurrency),
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
    ]

    append_arg(command, "--dataset-path", args.dataset_path)

    if args.api_key:
        command.extend(["--header", f"Authorization: Bearer {args.api_key}"])

    return command


def run_single_benchmark(
    args: argparse.Namespace,
    input_length: int,
    request_rate: float,
):
    """Run one benchmark command and return the extracted metrics."""
    result_filename = f"embeddings_in{input_length}_rps{request_rate}.json"
    command = build_command(args, input_length, request_rate, result_filename)

    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    """Run the full benchmark sweep."""
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    args = apply_config_defaults(args, config)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    for input_length in args.input_lengths:
        for request_rate in args.request_rates:
            run_single_benchmark(args, input_length, request_rate)


if __name__ == "__main__":
    main()
