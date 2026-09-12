from __future__ import annotations

from typing import Any


def decision_markdown(decision: dict[str, Any]) -> str:
    summary = decision["summary"]
    selected_id = summary["selected"]
    selected = next(
        (
            candidate
            for candidate in decision["candidates"]
            if candidate["id"] == selected_id
        ),
        None,
    )
    lines = [
        f"# RuntimeFit decision: {decision['decision']}",
        "",
        f"**{summary['eligible_count']}/{summary['candidate_count']} configurations satisfy the requirements.**",
        "",
    ]
    if selected:
        provisional = summary.get("selection_status") == "indistinguishable"
        identity = " / ".join(
            str(value)
            for value in (
                selected["runtime"],
                selected.get("quantization"),
                f"concurrency={selected['concurrency']}"
                if selected.get("concurrency") is not None
                else None,
            )
            if value is not None
        )
        lines.extend(
            [
                "## Provisional recommendation" if provisional else "## Recommended",
                "",
                f"**{identity}** (`{selected['id']}`)",
                "",
                (
                    f"Provisional choice for `{decision['objective']}`; its observed objective range overlaps "
                    f"with {', '.join(f'`{name}`' for name in summary['indistinguishable_candidates'])}."
                    if provisional
                    else f"Selected for `{decision['objective']}` among configurations satisfying every requirement."
                ),
                "",
                f"Evidence status: **{summary.get('evidence_status', 'unknown')}**.",
                "",
                "### Requirement checks",
                "",
                "| Requirement | Observed | Required | Result |",
                "|---|---:|---:|:---:|",
            ]
        )
        for check in selected["checks"]:
            lines.append(
                f"| `{check['requirement']}` | {_requirement_value(check, check['actual'])} | "
                f"{_requirement_value(check, check['limit'])} | {'✓' if check['passed'] else '✗'} |"
            )
        lines.append("")
        ranged_checks = [
            check for check in selected["checks"] if check.get("observed_range")
        ]
        if ranged_checks:
            lines.append(
                "Requirement checks use the conservative end of the observed repeated-run range."
            )
            lines.append("")
    else:
        lines.extend(
            [
                "## No recommendation",
                "",
                "No configuration is both eligible and rankable for the objective.",
                "",
            ]
        )
    fastest_id = summary.get("fastest_candidate")
    if fastest_id and not summary.get("fastest_candidate_eligible"):
        fastest = next(
            candidate
            for candidate in decision["candidates"]
            if candidate["id"] == fastest_id
        )
        reasons = "; ".join(format_failed_checks(fastest))
        lines.extend(
            [
                "## Fastest candidate rejected",
                "",
                f"`{fastest_id}` had the highest measured request throughput but was rejected: {reasons}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Evidence",
            "",
            "| Candidate | Eligible | P95 TTFT | P99 latency | Req/s | Replicas | Headroom | Error | Monthly cost |",
            "|---|:---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in decision["candidates"]:
        metrics = candidate["metrics"]
        lines.append(
            f"| {candidate['id']} | {'✓' if candidate['eligible'] else '✗'} | "
            f"{_metric(metrics, 'ttft_p95_ms', ' ms')} | {_metric(metrics, 'latency_p99_ms', ' ms')} | "
            f"{_metric(metrics, 'request_throughput_rps')} | {_integer(metrics.get('required_replicas'))} | "
            f"{_percent(metrics.get('throughput_headroom_fraction'))} | "
            f"{_percent(metrics.get('error_rate'))} | {_money(metrics.get('monthly_cost_usd'))} |"
        )
    lines.extend(["", "## Why not the alternatives?", ""])
    for candidate in decision["candidates"]:
        if candidate["id"] == selected_id:
            continue
        alternative_reasons = (
            format_failed_checks(candidate) or candidate["comparison_to_selected"]
        )
        lines.append(
            f"- **{candidate['id']}**: "
            f"{'; '.join(alternative_reasons) if alternative_reasons else 'eligible, but not optimal'}"
        )
    warned = [
        candidate for candidate in decision["candidates"] if candidate.get("warnings")
    ]
    if warned:
        lines.extend(["", "## Evidence limitations", ""])
        for candidate in warned:
            lines.append(f"- **{candidate['id']}**: {'; '.join(candidate['warnings'])}")
    lines.extend(
        [
            "",
            "## Feasible Pareto frontier",
            "",
            ", ".join(f"`{name}`" for name in decision["pareto_frontier"]) or "None",
            "",
            f"Decision fingerprint: `{decision['decision_fingerprint']}`",
            "",
            f"Configuration fingerprint: `{decision['config_fingerprint']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _metric(metrics: dict[str, float], name: str, suffix: str = "") -> str:
    value = metrics.get(name)
    return "—" if value is None else f"{value:,.2f}{suffix}"


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def _money(value: float | None) -> str:
    return "—" if value is None else f"${value:,.0f}"


def _integer(value: float | None) -> str:
    return "—" if value is None else str(int(value))


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:,.4g}"


def format_failed_checks(candidate: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for check in candidate["checks"]:
        if check["passed"]:
            continue
        if check["actual"] is None:
            failures.append(f"missing {_requirement_label(check['requirement'])}")
            continue
        operator = "<" if check["requirement"].startswith("min_") else ">"
        failures.append(
            f"{_requirement_label(check['requirement'])} "
            f"{_requirement_value(check, check['actual'])} {operator} "
            f"{_requirement_value(check, check['limit'])}"
        )
    return failures


def _requirement_label(requirement: str) -> str:
    return {
        "p95_ttft_ms": "p95 TTFT",
        "p99_latency_ms": "p99 latency",
        "max_monthly_cost_usd": "monthly cost",
        "max_error_rate": "error rate",
        "min_request_throughput_rps": "request throughput",
        "min_output_token_throughput_tps": "output-token throughput",
        "max_gpu_memory_gb": "GPU memory",
        "min_throughput_headroom_fraction": "throughput headroom",
    }.get(requirement, requirement)


def _requirement_value(check: dict[str, Any], value: float | None) -> str:
    if value is None:
        return "—"
    requirement = check["requirement"]
    if requirement == "p95_ttft_ms":
        return f"{value:,.0f} ms"
    if requirement == "p99_latency_ms":
        return f"{value / 1000:,.2f} s"
    if requirement == "max_monthly_cost_usd":
        return f"${value:,.0f}/mo"
    if requirement in {"max_error_rate", "min_throughput_headroom_fraction"}:
        return f"{value:.1%}"
    if requirement == "min_request_throughput_rps":
        return f"{value:,.2f} req/s"
    if requirement == "min_output_token_throughput_tps":
        return f"{value:,.2f} tok/s"
    if requirement == "max_gpu_memory_gb":
        return f"{value:,.2f} GiB"
    return _number(value)
