from __future__ import annotations

import math
from statistics import mean

from runtimefit.models import Sample


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(samples: list[Sample], wall_seconds: float) -> dict[str, float | int | None]:
    successful = [sample for sample in samples if sample.error is None]
    latencies = [sample.latency_ms for sample in successful]
    ttfts = [sample.ttft_ms for sample in successful]
    qualities = [sample.quality for sample in successful if sample.quality is not None]
    tokens = sum(sample.output_tokens for sample in successful)
    total = len(samples)
    return {
        "requests": total,
        "successful_requests": len(successful),
        "error_rate": (total - len(successful)) / total if total else 0.0,
        "quality": mean(qualities) if qualities else None,
        "latency_p50_ms": percentile(latencies, 0.50) if latencies else None,
        "latency_p95_ms": percentile(latencies, 0.95) if latencies else None,
        "latency_p99_ms": percentile(latencies, 0.99) if latencies else None,
        "ttft_p50_ms": percentile(ttfts, 0.50) if ttfts else None,
        "ttft_p95_ms": percentile(ttfts, 0.95) if ttfts else None,
        "output_tokens": tokens,
        "output_tokens_per_second": tokens / wall_seconds if wall_seconds > 0 else 0.0,
        "requests_per_second": len(successful) / wall_seconds if wall_seconds > 0 else 0.0,
        "wall_seconds": wall_seconds,
    }

