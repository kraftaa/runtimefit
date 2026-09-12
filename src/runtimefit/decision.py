from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtimefit.evidence import load_candidate_evidence
from runtimefit.selection import best_candidate, pareto_ids, satisfies

REQUIREMENTS = {
    "p95_ttft_ms": ("ttft_p95_ms", "max"),
    "p99_latency_ms": ("latency_p99_ms", "max"),
    "max_monthly_cost_usd": ("monthly_cost_usd", "max"),
    "max_error_rate": ("error_rate", "max"),
    "min_request_throughput_rps": ("request_throughput_rps", "min"),
    "min_output_token_throughput_tps": ("output_token_throughput_tps", "min"),
    "max_gpu_memory_gb": ("gpu_memory_gb", "max"),
    "min_throughput_headroom_fraction": ("throughput_headroom_fraction", "min"),
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
    objective_metric, objective_direction = OBJECTIVES[objective]
    evidence_policy = {
        "minimum_runs": 3,
        "minimum_samples_p95": 100,
        "minimum_samples_p99": 1000,
        **config.get("evidence_policy", {}),
    }
    relevant_metrics = {
        REQUIREMENTS[requirement][0] for requirement in requirements
    } | {objective_metric}
    hours_per_month = float(config.get("cost", {}).get("hours_per_month", 730))
    evaluated: list[dict[str, Any]] = []
    for candidate in config["candidates"]:
        loaded = load_candidate_evidence(candidate, base_dir, hours_per_month)
        metrics = loaded["metrics"]
        _add_capacity_metrics(
            metrics, candidate, config, hours_per_month, loaded["source"]
        )
        warnings = _evidence_warnings(
            loaded["source"], evidence_policy, relevant_metrics
        )
        failures: list[str] = []
        checks: list[dict[str, Any]] = []
        for requirement, limit in requirements.items():
            metric_name, direction = REQUIREMENTS[requirement]
            estimate = metrics.get(metric_name)
            actual, observed_range = _conservative_value(
                metric_name, estimate, direction, loaded["source"]
            )
            passed = satisfies(actual, float(limit), direction)
            checks.append(
                {
                    "requirement": requirement,
                    "metric": metric_name,
                    "actual": actual,
                    "estimate": estimate,
                    "observed_range": observed_range,
                    "limit": limit,
                    "passed": passed,
                }
            )
            if actual is None:
                failures.append(f"missing metric required for {requirement}")
            elif not passed:
                operator = "exceeds" if direction == "max" else "is below"
                failures.append(
                    f"{metric_name} {actual:g} {operator} requirement {limit:g}"
                )
        evaluated.append(
            {
                "id": candidate["id"],
                "runtime": candidate["runtime"],
                "quantization": candidate.get("quantization"),
                "concurrency": candidate.get("concurrency"),
                "metrics": metrics,
                "source": loaded["source"],
                "eligible": not failures,
                "checks": checks,
                "rejection_reasons": failures,
                "warnings": warnings,
                "evidence_status": "limited" if warnings else "supported",
            }
        )
    eligible = [candidate for candidate in evaluated if candidate["eligible"]]
    rankable = [
        candidate
        for candidate in eligible
        if candidate["metrics"].get(objective_metric) is not None
    ]
    selected = best_candidate(
        eligible,
        objective_metric,
        objective_direction,
        lambda candidate: candidate["metrics"],
    )
    if selected is not None:
        selected_id: str | None = selected["id"]
    else:
        selected_id = None
    frontier_dimensions = {
        REQUIREMENTS[requirement][0]: (
            "min" if REQUIREMENTS[requirement][1] == "max" else "max"
        )
        for requirement in requirements
    }
    frontier_dimensions[objective_metric] = objective_direction
    frontier = pareto_ids(
        rankable,
        frontier_dimensions,
        lambda candidate: candidate["metrics"],
        lambda candidate: candidate["id"],
    )
    fastest = max(
        (
            candidate
            for candidate in evaluated
            if candidate["metrics"].get("request_throughput_rps") is not None
        ),
        key=lambda candidate: candidate["metrics"]["request_throughput_rps"],
        default=None,
    )
    for candidate in evaluated:
        candidate["comparison_to_selected"] = _comparison(
            candidate, selected if rankable else None
        )
    public_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    config_fingerprint = hashlib.sha256(
        json.dumps(public_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    evidence_identity = [
        {
            "candidate": candidate["id"],
            "source": candidate["source"],
            "metrics": candidate["metrics"]
            if candidate["source"]["provider"] == "inline"
            else None,
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
    objective_ties: list[str] = []
    if selected is not None:
        reason = f"{OBJECTIVE_LABELS[objective]} among candidates satisfying every requirement"
        objective_ties = _objective_ties(selected, rankable, objective_metric)
        if objective_ties:
            reason = (
                f"provisional {reason}; observed {objective_metric} ranges overlap with "
                + ", ".join(objective_ties)
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
            "selection_status": "indistinguishable"
            if objective_ties
            else ("selected" if selected_id else "none"),
            "indistinguishable_candidates": objective_ties,
            "evidence_status": (
                selected["evidence_status"] if selected is not None else "insufficient"
            ),
            "fastest_candidate": fastest["id"] if fastest else None,
            "fastest_candidate_eligible": fastest["eligible"] if fastest else None,
            "fastest_candidate_rejection_reasons": fastest["rejection_reasons"]
            if fastest
            else [],
        },
        "pareto_frontier": frontier,
        "candidates": evaluated,
    }


def _variability_warnings(source: dict[str, Any], threshold: float = 0.15) -> list[str]:
    warnings: list[str] = []
    for metric, values in (source.get("variability") or {}).items():
        relative_range = values.get("relative_range")
        if relative_range is not None and relative_range > threshold:
            warnings.append(
                f"{metric} varied by {relative_range:.1%} across repeated evidence runs"
            )
    return warnings


def _evidence_warnings(
    source: dict[str, Any], policy: dict[str, int], relevant_metrics: set[str]
) -> list[str]:
    warnings = _variability_warnings(source)
    provider = source.get("provider")
    run_count = int(source.get("run_count") or 0)
    if provider == "inline":
        warnings.append(
            "inline evidence has no independently verifiable sample provenance"
        )
    elif run_count < policy["minimum_runs"]:
        warnings.append(
            f"only {run_count} evidence run(s); policy requires {policy['minimum_runs']}"
        )
    sample_counts = source.get("sample_counts") or {}
    for metric in sorted(relevant_metrics):
        if "_p95_" in metric or metric.endswith("_p95_ms"):
            required = policy["minimum_samples_p95"]
        elif "_p99_" in metric or metric.endswith("_p99_ms"):
            required = policy["minimum_samples_p99"]
        else:
            continue
        count = sample_counts.get(metric)
        if count is None:
            warnings.append(f"sample count unavailable for {metric}")
        elif count < required:
            warnings.append(
                f"{metric} has {count} samples per run; policy requires {required}"
            )
    return warnings


def _conservative_value(
    metric: str,
    estimate: float | None,
    direction: str,
    source: dict[str, Any],
) -> tuple[float | None, dict[str, float] | None]:
    variability = (source.get("variability") or {}).get(metric)
    if not variability:
        return estimate, None
    observed_range = {
        "min": float(variability["min"]),
        "max": float(variability["max"]),
    }
    value = observed_range["max"] if direction == "max" else observed_range["min"]
    return value, observed_range


def _add_capacity_metrics(
    metrics: dict[str, float],
    candidate: dict[str, Any],
    config: dict[str, Any],
    hours_per_month: float,
    source: dict[str, Any],
) -> None:
    required_rps = (config.get("workload") or {}).get("requests_per_second")
    per_replica_rps = metrics.get("request_throughput_rps")
    required_headroom = (config.get("requirements") or {}).get(
        "min_throughput_headroom_fraction", 0.0
    )
    replicas = 1
    if required_rps is not None and per_replica_rps:
        variability = (source.get("variability") or {}).get("request_throughput_rps")
        planning_rps = float(variability["min"]) if variability else per_replica_rps
        required_capacity = float(required_rps) * (1 + float(required_headroom))
        ratio = required_capacity / planning_rps
        tolerance = 1e-12 * max(1.0, abs(ratio))
        replicas = max(1, math.ceil(ratio - tolerance))
        deployment_rps = planning_rps * replicas
        metrics["capacity_basis_request_throughput_rps"] = planning_rps
        metrics["required_replicas"] = float(replicas)
        metrics["deployment_throughput_rps"] = deployment_rps
        metrics["throughput_headroom_fraction"] = (
            deployment_rps / float(required_rps) - 1
        )
    if candidate.get("cost_per_hour_usd") is not None:
        metrics["monthly_cost_usd"] = (
            float(candidate["cost_per_hour_usd"]) * hours_per_month * replicas
        )


def _objective_ties(
    selected: dict[str, Any], candidates: list[dict[str, Any]], metric: str
) -> list[str]:
    selected_range = _metric_range(selected, metric)
    ties: list[str] = []
    for candidate in candidates:
        if candidate is selected:
            continue
        candidate_range = _metric_range(candidate, metric)
        overlaps = max(selected_range[0], candidate_range[0]) <= min(
            selected_range[1], candidate_range[1]
        )
        has_measured_range = (
            selected_range[0] != selected_range[1]
            or candidate_range[0] != candidate_range[1]
        )
        exact_tie = math.isclose(
            selected["metrics"][metric], candidate["metrics"][metric], rel_tol=1e-12
        )
        if exact_tie or (overlaps and has_measured_range):
            ties.append(candidate["id"])
    return ties


def _metric_range(candidate: dict[str, Any], metric: str) -> tuple[float, float]:
    variability = (candidate["source"].get("variability") or {}).get(metric)
    if variability:
        return float(variability["min"]), float(variability["max"])
    value = float(candidate["metrics"][metric])
    return value, value


def _comparison(
    candidate: dict[str, Any], selected: dict[str, Any] | None
) -> list[str]:
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
        if (
            candidate_value is None
            or selected_value in (None, 0)
            or candidate_value == selected_value
        ):
            continue
        difference = (candidate_value - selected_value) / selected_value * 100
        comparisons.append(
            f"{label} is {abs(difference):.1f}% {'higher' if difference > 0 else 'lower'}"
        )
    return comparisons
