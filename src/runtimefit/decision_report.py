from __future__ import annotations

from typing import Any


def decision_markdown(decision: dict[str, Any]) -> str:
    summary = decision["summary"]
    selected_id = summary["selected"]
    selected = next((candidate for candidate in decision["candidates"] if candidate["id"] == selected_id), None)
    lines = [
        f"# RuntimeFit decision: {decision['decision']}",
        "",
        f"**{summary['eligible_count']}/{summary['candidate_count']} configurations satisfy the requirements.**",
        "",
    ]
    if selected:
        identity = " / ".join(
            str(value) for value in (
                selected["runtime"], selected.get("quantization"),
                f"concurrency={selected['concurrency']}" if selected.get("concurrency") is not None else None,
            ) if value is not None
        )
        lines.extend([
            "## Recommended",
            "",
            f"**{identity}** (`{selected['id']}`)",
            "",
            f"Selected for `{decision['objective']}` among configurations satisfying every requirement.",
            "",
            "### Requirement checks",
            "",
            "| Requirement | Observed | Required | Result |",
            "|---|---:|---:|:---:|",
        ])
        for check in selected["checks"]:
            lines.append(
                f"| `{check['requirement']}` | {_number(check['actual'])} | "
                f"{_number(check['limit'])} | {'✓' if check['passed'] else '✗'} |"
            )
        lines.append("")
    else:
        lines.extend(["## No recommendation", "", "No configuration is both eligible and rankable for the objective.", ""])
    lines.extend([
        "## Evidence",
        "",
        "| Candidate | Eligible | P95 TTFT | P99 latency | Req/s | Out tok/s | Error | Monthly cost |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|",
    ])
    for candidate in decision["candidates"]:
        metrics = candidate["metrics"]
        lines.append(
            f"| {candidate['id']} | {'✓' if candidate['eligible'] else '✗'} | "
            f"{_metric(metrics, 'ttft_p95_ms', ' ms')} | {_metric(metrics, 'latency_p99_ms', ' ms')} | "
            f"{_metric(metrics, 'request_throughput_rps')} | {_metric(metrics, 'output_token_throughput_tps')} | "
            f"{_percent(metrics.get('error_rate'))} | {_money(metrics.get('monthly_cost_usd'))} |"
        )
    lines.extend(["", "## Why not the alternatives?", ""])
    for candidate in decision["candidates"]:
        if candidate["id"] == selected_id:
            continue
        reasons = candidate["rejection_reasons"] or candidate["comparison_to_selected"]
        lines.append(f"- **{candidate['id']}**: {'; '.join(reasons) if reasons else 'eligible, but not optimal'}")
    lines.extend([
        "",
        "## Pareto frontier",
        "",
        ", ".join(f"`{name}`" for name in decision["pareto_frontier"]) or "None",
        "",
        f"Decision fingerprint: `{decision['decision_fingerprint']}`",
        "",
        f"Configuration fingerprint: `{decision['config_fingerprint']}`",
        "",
    ])
    return "\n".join(lines)


def _metric(metrics: dict[str, float], name: str, suffix: str = "") -> str:
    value = metrics.get(name)
    return "—" if value is None else f"{value:,.2f}{suffix}"


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def _money(value: float | None) -> str:
    return "—" if value is None else f"${value:,.0f}"


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:,.4g}"
