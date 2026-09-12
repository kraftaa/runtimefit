from __future__ import annotations

from typing import Any, Callable, Iterable


def satisfies(value: float | int | None, limit: float, direction: str) -> bool:
    """Return False for missing evidence and otherwise apply a min/max constraint."""

    if value is None:
        return False
    return float(value) <= limit if direction == "max" else float(value) >= limit


def best_candidate(
    candidates: Iterable[Any],
    metric: str,
    direction: str,
    metrics: Callable[[Any], dict[str, Any]],
) -> Any | None:
    """Select by one declared objective without manufacturing values for missing data."""

    rankable = [
        candidate
        for candidate in candidates
        if metrics(candidate).get(metric) is not None
    ]
    if not rankable:
        return None
    chooser = min if direction == "min" else max
    return chooser(rankable, key=lambda candidate: float(metrics(candidate)[metric]))


def pareto_ids(
    candidates: Iterable[Any],
    dimensions: dict[str, str],
    metrics: Callable[[Any], dict[str, Any]],
    identifier: Callable[[Any], str],
) -> list[str]:
    """Return the complete-evidence candidates not dominated on declared dimensions."""

    comparable = [
        candidate
        for candidate in candidates
        if all(metrics(candidate).get(name) is not None for name in dimensions)
    ]

    def dominates(left: Any, right: Any) -> bool:
        if not dimensions:
            return False
        no_worse = all(
            float(metrics(left)[name]) <= float(metrics(right)[name])
            if direction == "min"
            else float(metrics(left)[name]) >= float(metrics(right)[name])
            for name, direction in dimensions.items()
        )
        better = any(
            float(metrics(left)[name]) < float(metrics(right)[name])
            if direction == "min"
            else float(metrics(left)[name]) > float(metrics(right)[name])
            for name, direction in dimensions.items()
        )
        return no_worse and better

    return [
        identifier(candidate)
        for candidate in comparable
        if not any(
            other is not candidate and dominates(other, candidate)
            for other in comparable
        )
    ]
