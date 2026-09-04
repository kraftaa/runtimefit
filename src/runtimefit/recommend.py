from __future__ import annotations

import math
from typing import Any

from runtimefit.models import TargetResult


def recommend(
    results: list[TargetResult],
    objective: str,
    constraints: dict[str, Any],
) -> tuple[str | None, dict[str, str]]:
    reasons: dict[str, str] = {}
    qualities = [
        float(result.metrics["quality"])
        for result in results
        if result.metrics.get("quality") is not None
    ]
    best_quality = max(qualities) if qualities else None
    max_quality_loss = float(constraints.get("max_quality_loss", 1.0))
    max_error_rate = float(constraints.get("max_error_rate", 0.0))
    eligible: list[TargetResult] = []
    for result in results:
        error_rate = float(result.metrics.get("error_rate") or 0.0)
        if error_rate > max_error_rate:
            reasons[result.name] = f"error rate {error_rate:.1%} exceeds {max_error_rate:.1%}"
            continue
        quality = result.metrics.get("quality")
        if best_quality is not None and quality is not None and best_quality - float(quality) > max_quality_loss:
            reasons[result.name] = "quality loss exceeds configured limit"
            continue
        checks = (
            ("latency_p95_ms", "max_p95_latency_ms", "p95 latency"),
            ("ttft_p95_ms", "max_p95_ttft_ms", "p95 TTFT"),
            ("estimated_usd_per_million_output_tokens", "max_usd_per_million_output_tokens", "estimated cost"),
        )
        rejected = False
        for metric_key, constraint_key, label in checks:
            limit = constraints.get(constraint_key)
            metric = result.metrics.get(metric_key)
            if limit is not None and (metric is None or float(metric) > float(limit)):
                reasons[result.name] = f"{label} exceeds configured limit"
                rejected = True
                break
        if rejected:
            continue
        minimum_throughput = constraints.get("min_output_tokens_per_second")
        throughput = float(result.metrics.get("output_tokens_per_second") or 0.0)
        if minimum_throughput is not None and throughput < float(minimum_throughput):
            reasons[result.name] = "output-token throughput is below configured minimum"
            continue
        eligible.append(result)
    if not eligible:
        return None, reasons
    if objective == "throughput":
        selected = max(eligible, key=lambda result: float(result.metrics["output_tokens_per_second"] or 0))
    elif objective == "latency":
        selected = min(eligible, key=lambda result: float(result.metrics["latency_p95_ms"] or math.inf))
    elif objective == "ttft":
        selected = min(eligible, key=lambda result: float(result.metrics["ttft_p95_ms"] or math.inf))
    else:
        priced = [
            result for result in eligible
            if result.metrics.get("estimated_usd_per_million_output_tokens") is not None
        ]
        if not priced:
            for result in eligible:
                reasons[result.name] = "cost objective requires target.cost_per_hour_usd"
            return None, reasons
        selected = min(
            priced,
            key=lambda result: float(result.metrics["estimated_usd_per_million_output_tokens"]),
        )
    for result in eligible:
        reasons.setdefault(result.name, "eligible but not optimal for the configured objective")
    reasons[selected.name] = f"best eligible target for objective '{objective}'"
    return selected.name, reasons


def pareto_frontier(results: list[TargetResult]) -> list[str]:
    """Return candidates not dominated on throughput, p95 latency, quality, and errors."""

    def dominates(left: TargetResult, right: TargetResult) -> bool:
        left_quality = left.metrics.get("quality")
        right_quality = right.metrics.get("quality")
        comparable = [
            (float(left.metrics.get("output_tokens_per_second") or 0), float(right.metrics.get("output_tokens_per_second") or 0), 1),
            (float(left.metrics.get("latency_p95_ms") or math.inf), float(right.metrics.get("latency_p95_ms") or math.inf), -1),
            (float(left.metrics.get("error_rate") or 0), float(right.metrics.get("error_rate") or 0), -1),
        ]
        if left_quality is not None and right_quality is not None:
            comparable.append((float(left_quality), float(right_quality), 1))
        no_worse = all(a >= b if direction == 1 else a <= b for a, b, direction in comparable)
        strictly_better = any(a > b if direction == 1 else a < b for a, b, direction in comparable)
        return no_worse and strictly_better

    return [
        candidate.name
        for candidate in results
        if not any(other is not candidate and dominates(other, candidate) for other in results)
    ]
