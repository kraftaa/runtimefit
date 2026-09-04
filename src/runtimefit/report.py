from __future__ import annotations

from typing import Any


def _number(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def markdown_report(result: dict[str, Any]) -> str:
    chosen = result["recommendation"]["target"] or "None (no target met the constraints)"
    lines = [
        f"# RuntimeFit report: {result['benchmark']}",
        "",
        f"**Recommendation:** `{chosen}`",
        "",
        f"Objective: `{result['objective']}`  ",
        f"Created: {result['created_at']}",
        "",
        "| Target | C | Quality | Error rate | p50 latency | p95 latency | p95 TTFT | Output tok/s | $/1M out | Startup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in result["targets"]:
        metrics = target["metrics"]
        quality = _number(metrics["quality"], 3)
        error_rate = f"{float(metrics['error_rate']):.1%}"
        startup = metrics.get("server_startup_seconds")
        startup_display = "—" if startup is None else f"{_number(startup)} s"
        lines.append(
            f"| {target['name']} | {metrics.get('concurrency', 1)} | {quality} | {error_rate} | "
            f"{_number(metrics['latency_p50_ms'])} ms | {_number(metrics['latency_p95_ms'])} ms | "
            f"{_number(metrics['ttft_p95_ms'])} ms | {_number(metrics['output_tokens_per_second'])} | "
            f"{_number(metrics.get('estimated_usd_per_million_output_tokens'), 4)} | "
            f"{startup_display} |"
        )
    lines.extend([
        "",
        "## Decision details",
        "",
        "Pareto frontier: " + ", ".join(f"`{name}`" for name in result.get("pareto_frontier", [])),
        "",
    ])
    for name, reason in result["recommendation"].get("reasons", {}).items():
        lines.append(f"- `{name}`: {reason}")
    estimated_tokens = any(
        sample.get("token_count_source") == "estimated"
        for target in result["targets"] for sample in target.get("samples", [])
    )
    non_streaming = any(
        not target.get("options", {}).get("streaming", True)
        for target in result.get("definition", {}).get("config", {}).get("targets", [])
        if target.get("adapter") == "openai"
    )
    lines.extend([
        "",
        "## Reproducibility metadata",
        "",
        f"- Definition SHA-256: `{result['definition_fingerprint']}`",
        f"- RuntimeFit: `{result['environment']['runtimefit']}`",
        f"- Python: `{result['environment']['python']}`",
        f"- Platform: `{result['environment']['platform']}`",
        f"- Machine: `{result['environment']['machine']}`",
        f"- Processor: `{result['environment']['processor']}`",
        f"- Logical CPUs: `{result['environment']['logical_cpu_count']}`",
        f"- Physical memory: `{result['environment']['physical_memory_bytes']}` bytes",
    ])
    if non_streaming or estimated_tokens:
        lines.append("")
    if non_streaming:
        lines.append("> TTFT for non-streaming HTTP endpoints means time to response headers/first response byte, not first generated token.")
    if estimated_tokens:
        lines.append("> Token counts marked `estimated` in the JSON result use a character-based approximation.")
    lines.append("")
    return "\n".join(lines)
