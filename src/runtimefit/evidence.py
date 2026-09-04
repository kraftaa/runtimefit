from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from runtimefit.config import ConfigError


CANONICAL_METRICS = {
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "tpot_p99_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "request_throughput_rps",
    "output_token_throughput_tps",
    "error_rate",
    "gpu_memory_gb",
    "monthly_cost_usd",
}


RUNTIMEFIT_METRIC_MAP = {
    "ttft_p50_ms": "ttft_p50_ms",
    "ttft_p95_ms": "ttft_p95_ms",
    "latency_p50_ms": "latency_p50_ms",
    "latency_p95_ms": "latency_p95_ms",
    "latency_p99_ms": "latency_p99_ms",
    "request_throughput_rps": "requests_per_second",
    "output_token_throughput_tps": "output_tokens_per_second",
    "error_rate": "error_rate",
}


def load_candidate_evidence(candidate: dict[str, Any], base_dir: Path, hours_per_month: float) -> dict[str, Any]:
    evidence = candidate["evidence"]
    provider = evidence.get("provider", "inline")
    source: dict[str, Any]
    if provider == "inline":
        metrics = _numeric_metrics(evidence["metrics"])
        source = {"provider": "inline"}
    elif provider == "runtimefit":
        path = base_dir / evidence["path"]
        try:
            raw = path.read_bytes()
            result = json.loads(raw)
        except FileNotFoundError as exc:
            raise ConfigError(f"Evidence file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid RuntimeFit evidence JSON: {path}: {exc}") from exc
        matching = [target for target in result.get("targets", []) if target.get("name") == evidence["target"]]
        if not matching:
            raise ConfigError(f"Target '{evidence['target']}' not found in evidence file {path}")
        source_metrics = matching[0].get("metrics", {})
        metrics = {
            canonical: float(source_metrics[source])
            for canonical, source in RUNTIMEFIT_METRIC_MAP.items()
            if source_metrics.get(source) is not None
        }
        source = {
            "provider": "runtimefit",
            "path": evidence["path"],
            "target": evidence["target"],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "definition_fingerprint": result.get("definition_fingerprint"),
        }
    else:
        metrics, source = _load_guidellm_evidence(evidence, base_dir)
    if candidate.get("monthly_cost_usd") is not None:
        metrics["monthly_cost_usd"] = float(candidate["monthly_cost_usd"])
    elif candidate.get("cost_per_hour_usd") is not None:
        metrics["monthly_cost_usd"] = float(candidate["cost_per_hour_usd"]) * hours_per_month
    return {"metrics": metrics, "source": source}


def _load_guidellm_evidence(
    evidence: dict[str, Any], base_dir: Path
) -> tuple[dict[str, float], dict[str, Any]]:
    if "paths" not in evidence:
        return _load_guidellm(evidence, base_dir)
    loaded = [
        _load_guidellm({**evidence, "path": path}, base_dir)
        for path in evidence["paths"]
    ]
    shared_metrics = set.intersection(*(set(metrics) for metrics, _ in loaded))
    metrics: dict[str, float] = {}
    variability: dict[str, dict[str, float]] = {}
    for name in sorted(shared_metrics):
        values = [run_metrics[name] for run_metrics, _ in loaded]
        median = float(statistics.median(values))
        metrics[name] = median
        variability[name] = {
            "min": min(values),
            "median": median,
            "max": max(values),
            "relative_range": (max(values) - min(values)) / median if median else 0.0,
        }
    sources = [source for _, source in loaded]
    return metrics, {
        "provider": "guidellm",
        "aggregation": "median",
        "run_count": len(loaded),
        "runs": sources,
        "variability": variability,
    }


def _load_guidellm(evidence: dict[str, Any], base_dir: Path) -> tuple[dict[str, float], dict[str, Any]]:
    path = base_dir / evidence["path"]
    try:
        raw = path.read_bytes()
        report = json.loads(raw)
    except FileNotFoundError as exc:
        raise ConfigError(f"GuideLLM evidence file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid GuideLLM evidence JSON: {path}: {exc}") from exc
    schema_version = (report.get("metadata") or {}).get("version")
    if schema_version != 2:
        raise ConfigError(f"Unsupported GuideLLM report schema version {schema_version!r}; expected 2")
    index = evidence["benchmark_index"]
    benchmarks = report.get("benchmarks")
    if not isinstance(benchmarks, list) or index >= len(benchmarks):
        raise ConfigError(f"GuideLLM benchmark_index {index} does not exist in {path}")
    benchmark = benchmarks[index]
    root = benchmark.get("metrics") or {}

    def distribution(name: str) -> dict[str, Any]:
        value = root.get(name) or {}
        return value.get("successful") or {}

    def percentile(name: str, key: str, scale: float = 1.0) -> float | None:
        value = (distribution(name).get("percentiles") or {}).get(key)
        return float(value) * scale if value is not None else None

    def mean(name: str) -> float | None:
        value = distribution(name).get("mean")
        return float(value) if value is not None else None

    extracted: dict[str, float | None] = {
        "ttft_p50_ms": percentile("time_to_first_token_ms", "p50"),
        "ttft_p95_ms": percentile("time_to_first_token_ms", "p95"),
        "ttft_p99_ms": percentile("time_to_first_token_ms", "p99"),
        "tpot_p50_ms": percentile("time_per_output_token_ms", "p50"),
        "tpot_p95_ms": percentile("time_per_output_token_ms", "p95"),
        "tpot_p99_ms": percentile("time_per_output_token_ms", "p99"),
        "latency_p50_ms": percentile("request_latency", "p50", 1000.0),
        "latency_p95_ms": percentile("request_latency", "p95", 1000.0),
        "latency_p99_ms": percentile("request_latency", "p99", 1000.0),
        "request_throughput_rps": mean("requests_per_second"),
        "output_token_throughput_tps": mean("output_tokens_per_second"),
    }
    totals = root.get("request_totals") or {}
    total_requests = totals.get("total")
    errored_requests = totals.get("errored")
    if total_requests:
        extracted["error_rate"] = (float(errored_requests or 0) / float(total_requests))
    metrics = {name: value for name, value in extracted.items() if value is not None}
    config = benchmark.get("config") or {}
    source = {
        "provider": "guidellm",
        "path": evidence["path"],
        "benchmark_index": index,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "guidellm_version": (report.get("metadata") or {}).get("guidellm_version"),
        "report_schema_version": schema_version,
        "benchmark_id": config.get("id_"),
        "run_index": config.get("run_index"),
        "strategy": config.get("strategy"),
    }
    return metrics, source


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    unknown = set(metrics) - CANONICAL_METRICS
    if unknown:
        raise ConfigError(f"Unsupported evidence metrics: {', '.join(sorted(unknown))}")
    normalized: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"Evidence metric {name} must be a non-negative number")
        normalized[name] = float(value)
    return normalized
