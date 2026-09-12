from __future__ import annotations

from typing import Any

from runtimefit.models import TargetResult
from runtimefit.selection import best_candidate, pareto_ids, satisfies

OBJECTIVES = {
    "throughput": ("output_tokens_per_second", "max"),
    "latency": ("latency_p95_ms", "min"),
    "ttft": ("ttft_p95_ms", "min"),
    "cost": ("estimated_usd_per_million_output_tokens", "min"),
}


def recommend(
    results: list[TargetResult],
    objective: str,
    constraints: dict[str, Any],
) -> tuple[str | None, dict[str, str]]:
    """Adapt development-runner results to the canonical selection primitives."""

    eligible, reasons = _eligible_results(results, constraints)
    metric, direction = OBJECTIVES[objective]
    selected = best_candidate(
        eligible, metric, direction, lambda result: result.metrics
    )
    if selected is None:
        for result in eligible:
            reasons[result.name] = f"objective requires measured {metric}"
        return None, reasons
    for result in eligible:
        reasons.setdefault(
            result.name, "eligible but not optimal for the configured objective"
        )
    reasons[selected.name] = f"best eligible target for objective '{objective}'"
    return selected.name, reasons


def pareto_frontier(
    results: list[TargetResult], constraints: dict[str, Any] | None = None
) -> list[str]:
    eligible, _ = _eligible_results(results, constraints or {})
    dimensions = {
        "output_tokens_per_second": "max",
        "latency_p95_ms": "min",
        "error_rate": "min",
    }
    if eligible and all(
        result.metrics.get("quality") is not None for result in eligible
    ):
        dimensions["quality"] = "max"
    return pareto_ids(
        eligible,
        dimensions,
        lambda result: result.metrics,
        lambda result: result.name,
    )


def _eligible_results(
    results: list[TargetResult], constraints: dict[str, Any]
) -> tuple[list[TargetResult], dict[str, str]]:
    reasons: dict[str, str] = {}
    qualities = [
        float(value)
        for result in results
        if (value := result.metrics.get("quality")) is not None
    ]
    best_quality = max(qualities) if qualities else None
    max_quality_loss = float(constraints.get("max_quality_loss", 1.0))
    eligible: list[TargetResult] = []
    checks = (
        ("error_rate", "max_error_rate", "error rate", "max", 0.0),
        ("latency_p95_ms", "max_p95_latency_ms", "p95 latency", "max", None),
        ("ttft_p95_ms", "max_p95_ttft_ms", "p95 TTFT", "max", None),
        (
            "estimated_usd_per_million_output_tokens",
            "max_usd_per_million_output_tokens",
            "estimated cost",
            "max",
            None,
        ),
        (
            "output_tokens_per_second",
            "min_output_tokens_per_second",
            "output-token throughput",
            "min",
            None,
        ),
    )
    for result in results:
        rejected = False
        for metric_name, constraint_name, label, direction, default in checks:
            limit = constraints.get(constraint_name, default)
            if limit is None:
                continue
            value = result.metrics.get(metric_name)
            if not satisfies(value, float(limit), direction):
                reasons[result.name] = (
                    f"missing {label}"
                    if value is None
                    else f"{label} violates configured limit"
                )
                rejected = True
                break
        if rejected:
            continue
        quality = result.metrics.get("quality")
        if (
            best_quality is not None
            and quality is not None
            and best_quality - float(quality) > max_quality_loss
        ):
            reasons[result.name] = "quality loss exceeds configured limit"
            continue
        eligible.append(result)
    return eligible, reasons
