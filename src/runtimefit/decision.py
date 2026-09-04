from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtimefit.evidence import load_candidate_evidence


REQUIREMENTS = {
    "p95_ttft_ms": ("ttft_p95_ms", "max"),
    "p99_latency_ms": ("latency_p99_ms", "max"),
    "max_monthly_cost_usd": ("monthly_cost_usd", "max"),
    "max_error_rate": ("error_rate", "max"),
    "min_request_throughput_rps": ("request_throughput_rps", "min"),
    "min_output_token_throughput_tps": ("output_token_throughput_tps", "min"),
    "max_gpu_memory_gb": ("gpu_memory_gb", "max"),
}

OBJECTIVES = {
    "lowest_monthly_cost": ("monthly_cost_usd", "min"),
    "lowest_p95_ttft": ("ttft_p95_ms", "min"),
    "lowest_p99_latency": ("latency_p99_ms", "min"),
    "highest_request_throughput": ("request_throughput_rps", "max"),
    "highest_output_token_throughput": ("output_token_throughput_tps", "max"),
}

OBJECTIVE_LABELS = {
    "lowest_monthly_cost": "lowest estimated monthly cost",
    "lowest_p95_ttft": "lowest p95 TTFT",
    "lowest_p99_latency": "lowest p99 end-to-end latency",
    "highest_request_throughput": "highest request throughput",
    "highest_output_token_throughput": "highest output-token throughput",
}


def choose(config: dict[str, Any]) -> dict[str, Any]:
    base_dir = Path(config["_config_dir"])
    requirements = config.get("requirements", {})
    objective = config.get("objective", "lowest_monthly_cost")
    if objective not in OBJECTIVES:
        raise ValueError(f"Unsupported objective: {objective}")
    hours_per_month = float(config.get("cost", {}).get("hours_per_month", 730))
    evaluated: list[dict[str, Any]] = []
    for candidate in config["candidates"]:
        loaded = load_candidate_evidence(candidate, base_dir, hours_per_month)
        metrics = loaded["metrics"]
        failures: list[str] = []
        checks: list[dict[str, Any]] = []
        for requirement, limit in requirements.items():
            metric_name, direction = REQUIREMENTS[requirement]
            actual = metrics.get(metric_name)
            passed = actual is not None and (actual <= limit if direction == "max" else actual >= limit)
            checks.append({
                "requirement": requirement,
                "metric": metric_name,
                "actual": actual,
                "limit": limit,
                "passed": passed,
            })
            if actual is None:
                failures.append(f"missing metric required for {requirement}")
            elif not passed:
                operator = "exceeds" if direction == "max" else "is below"
                failures.append(f"{metric_name} {actual:g} {operator} requirement {limit:g}")
        evaluated.append({
            "id": candidate["id"],
            "runtime": candidate["runtime"],
            "quantization": candidate.get("quantization"),
            "concurrency": candidate.get("concurrency"),
            "metrics": metrics,
            "source": loaded["source"],
            "eligible": not failures,
            "checks": checks,
            "rejection_reasons": failures,
        })
    objective_metric, objective_direction = OBJECTIVES[objective]
    eligible = [candidate for candidate in evaluated if candidate["eligible"]]
    rankable = [candidate for candidate in eligible if candidate["metrics"].get(objective_metric) is not None]
    if rankable:
        selected = (min if objective_direction == "min" else max)(
            rankable, key=lambda candidate: candidate["metrics"][objective_metric]
        )
        selected_id: str | None = selected["id"]
    else:
        selected_id = None
    frontier = _pareto_frontier(evaluated)
    for candidate in evaluated:
        candidate["comparison_to_selected"] = _comparison(candidate, selected if rankable else None)
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    config_fingerprint = hashlib.sha256(
        json.dumps(public_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    evidence_identity = [
        {
            "candidate": candidate["id"],
            "source": candidate["source"],
            "metrics": candidate["metrics"] if candidate["source"]["provider"] == "inline" else None,
        }
        for candidate in evaluated
    ]
    decision_fingerprint = hashlib.sha256(
        json.dumps(
            {"config_fingerprint": config_fingerprint, "evidence": evidence_identity},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    reason = None
    if selected_id:
        reason = (
            f"{OBJECTIVE_LABELS[objective]} among candidates satisfying every requirement"
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": config.get("name", "unnamed"),
        "objective": objective,
        "objective_metric": objective_metric,
        "requirements": requirements,
        "config_fingerprint": config_fingerprint,
        "decision_fingerprint": decision_fingerprint,
        "summary": {
            "candidate_count": len(evaluated),
            "eligible_count": len(eligible),
            "selected": selected_id,
            "reason": reason,
        },
        "pareto_frontier": frontier,
        "candidates": evaluated,
    }


def _pareto_frontier(candidates: list[dict[str, Any]]) -> list[str]:
    dimensions = {
        "monthly_cost_usd": "min",
        "ttft_p95_ms": "min",
        "latency_p99_ms": "min",
        "request_throughput_rps": "max",
        "output_token_throughput_tps": "max",
        "error_rate": "min",
        "gpu_memory_gb": "min",
    }

    shared = [
        name for name in dimensions
        if candidates and all(name in candidate["metrics"] for candidate in candidates)
    ]

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        if not shared:
            return False
        no_worse = all(
            left["metrics"][name] <= right["metrics"][name]
            if dimensions[name] == "min"
            else left["metrics"][name] >= right["metrics"][name]
            for name in shared
        )
        better = any(
            left["metrics"][name] < right["metrics"][name]
            if dimensions[name] == "min"
            else left["metrics"][name] > right["metrics"][name]
            for name in shared
        )
        return no_worse and better

    return [
        candidate["id"] for candidate in candidates
        if not any(other is not candidate and dominates(other, candidate) for other in candidates)
    ]


def _comparison(candidate: dict[str, Any], selected: dict[str, Any] | None) -> list[str]:
    if selected is None or candidate is selected:
        return []
    comparisons: list[str] = []
    labels = {
        "monthly_cost_usd": "monthly cost",
        "ttft_p95_ms": "p95 TTFT",
        "latency_p99_ms": "p99 latency",
        "request_throughput_rps": "request throughput",
        "output_token_throughput_tps": "output-token throughput",
    }
    for metric, label in labels.items():
        candidate_value = candidate["metrics"].get(metric)
        selected_value = selected["metrics"].get(metric)
        if candidate_value is None or selected_value in (None, 0) or candidate_value == selected_value:
            continue
        difference = (candidate_value - selected_value) / selected_value * 100
        comparisons.append(f"{label} is {abs(difference):.1f}% {'higher' if difference > 0 else 'lower'}")
    return comparisons
