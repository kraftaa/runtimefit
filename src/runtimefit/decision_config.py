from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from runtimefit.config import ConfigError

SUPPORTED_RUNTIMES = {"sglang", "vllm"}


def load_decision_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
    except FileNotFoundError as exc:
        raise ConfigError(f"Decision config not found: {config_path}") from exc
    suffix = config_path.suffix.casefold()
    try:
        if suffix == ".toml":
            data = tomllib.loads(raw.decode("utf-8"))
        elif suffix == ".json":
            data = json.loads(raw)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ConfigError("YAML support requires: pip install 'runtimefit[yaml]'") from exc
            try:
                data = yaml.safe_load(raw)
            except yaml.YAMLError as exc:
                raise ConfigError(f"Invalid YAML decision config: {exc}") from exc
        else:
            raise ConfigError("Decision config must use .toml, .json, .yaml, or .yml")
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Invalid {suffix[1:].upper()} decision config: {exc}") from exc
    validate_decision_config(data)
    data["_config_dir"] = str(config_path.parent.resolve())
    return data


def validate_decision_config(data: Any) -> None:
    if not isinstance(data, dict):
        raise ConfigError("Decision config root must be an object")
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ConfigError("Decision config requires a non-empty candidates list")
    identifiers: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ConfigError(f"candidates[{index}] must be an object")
        for key in ("id", "runtime"):
            if not isinstance(candidate.get(key), str) or not candidate[key].strip():
                raise ConfigError(f"candidates[{index}].{key} must be a non-empty string")
        if candidate["runtime"] not in SUPPORTED_RUNTIMES:
            supported = ", ".join(sorted(SUPPORTED_RUNTIMES))
            raise ConfigError(
                f"Unsupported decision runtime {candidate['runtime']!r}; v0.1 supports: {supported}"
            )
        if candidate["id"] in identifiers:
            raise ConfigError(f"Duplicate candidate id: {candidate['id']}")
        identifiers.add(candidate["id"])
        concurrency = candidate.get("concurrency")
        if concurrency is not None and (
            not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency <= 0
        ):
            raise ConfigError(f"candidates[{index}].concurrency must be a positive integer")
        quantization = candidate.get("quantization")
        if quantization is not None and (
            not isinstance(quantization, str) or not quantization.strip()
        ):
            raise ConfigError(f"candidates[{index}].quantization must be a non-empty string")
        evidence = candidate.get("evidence")
        if not isinstance(evidence, dict):
            raise ConfigError(f"candidates[{index}].evidence must be an object")
        provider = evidence.get("provider", "inline")
        if provider not in {"inline", "runtimefit", "guidellm"}:
            raise ConfigError(f"Unsupported evidence provider: {provider}")
        if provider == "inline" and not isinstance(evidence.get("metrics"), dict):
            raise ConfigError(f"candidates[{index}].evidence.metrics must be an object")
        if provider == "runtimefit":
            if not isinstance(evidence.get("path"), str) or not isinstance(evidence.get("target"), str):
                raise ConfigError(f"candidates[{index}] runtimefit evidence requires path and target")
        if provider == "guidellm":
            benchmark_index = evidence.get("benchmark_index")
            path = evidence.get("path")
            paths = evidence.get("paths")
            has_path = isinstance(path, str) and bool(path.strip())
            has_paths = (
                isinstance(paths, list)
                and bool(paths)
                and all(isinstance(item, str) and item.strip() for item in paths)
            )
            if has_path == has_paths:
                raise ConfigError(
                    f"candidates[{index}] guidellm evidence requires exactly one of path or non-empty paths"
                )
            if not isinstance(benchmark_index, int) or benchmark_index < 0:
                raise ConfigError(
                    f"candidates[{index}] guidellm evidence requires a non-negative benchmark_index"
                )
            aggregation = evidence.get("aggregation", "median")
            if aggregation != "median":
                raise ConfigError("GuideLLM repeated evidence currently supports aggregation = 'median'")
        for cost_key in ("monthly_cost_usd", "cost_per_hour_usd"):
            cost_value = candidate.get(cost_key)
            if cost_value is not None and (
                not isinstance(cost_value, (int, float)) or isinstance(cost_value, bool) or cost_value < 0
            ):
                raise ConfigError(f"candidates[{index}].{cost_key} must be a non-negative number")
    requirements = data.get("requirements", {})
    if not isinstance(requirements, dict):
        raise ConfigError("requirements must be an object")
    supported = {
        "p95_ttft_ms",
        "p99_latency_ms",
        "max_monthly_cost_usd",
        "max_error_rate",
        "min_request_throughput_rps",
        "min_output_token_throughput_tps",
        "max_gpu_memory_gb",
        "min_throughput_headroom_fraction",
    }
    unknown = set(requirements) - supported
    if unknown:
        raise ConfigError(f"Unsupported requirements: {', '.join(sorted(unknown))}")
    for key, value in requirements.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"requirements.{key} must be a non-negative number")
    objective = data.get("objective", "lowest_monthly_cost")
    supported_objectives = {
        "lowest_monthly_cost",
        "lowest_p95_ttft",
        "lowest_p99_latency",
        "highest_request_throughput",
        "highest_output_token_throughput",
    }
    if objective not in supported_objectives:
        raise ConfigError(f"Unsupported objective: {objective}")
    cost = data.get("cost", {})
    if not isinstance(cost, dict):
        raise ConfigError("cost must be an object")
    hours = cost.get("hours_per_month", 730)
    if not isinstance(hours, (int, float)) or isinstance(hours, bool) or hours <= 0:
        raise ConfigError("cost.hours_per_month must be a positive number")
    workload = data.get("workload", {})
    if not isinstance(workload, dict):
        raise ConfigError("workload must be an object")
    requests_per_second = workload.get("requests_per_second")
    if requests_per_second is not None and (
        not isinstance(requests_per_second, (int, float))
        or isinstance(requests_per_second, bool)
        or requests_per_second <= 0
    ):
        raise ConfigError("workload.requests_per_second must be a positive number")
    if "min_throughput_headroom_fraction" in requirements and requests_per_second is None:
        raise ConfigError(
            "workload.requests_per_second is required when min_throughput_headroom_fraction is set"
        )
