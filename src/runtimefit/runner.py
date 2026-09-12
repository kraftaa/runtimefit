from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtimefit.adapters import Adapter, create_adapter
from runtimefit.config import ConfigError
from runtimefit.dataset import load_dataset, score_output
from runtimefit.environment import collect_environment
from runtimefit.metrics import summarize
from runtimefit.models import Sample, TargetResult, WorkItem
from runtimefit.process import ManagedProcess
from runtimefit.recommend import pareto_frontier, recommend


def _execute(adapter: Adapter, item: WorkItem, repetition: int) -> Sample:
    try:
        generation = adapter.generate(item.prompt)
        return Sample(
            item_id=item.id,
            repetition=repetition,
            latency_ms=generation.latency_ms,
            ttft_ms=generation.ttft_ms,
            output_tokens=generation.output_tokens,
            token_count_source=generation.token_count_source,
            quality=score_output(item, generation.text),
        )
    except Exception as exc:  # benchmark errors belong in results, not a crashed run
        return Sample(
            item_id=item.id,
            repetition=repetition,
            latency_ms=0.0,
            ttft_ms=0.0,
            output_tokens=0,
            token_count_source="none",
            quality=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_benchmark(
    config: dict[str, Any], allow_processes: bool = False
) -> dict[str, Any]:
    base_dir = Path(config["_config_dir"])
    dataset_path = base_dir / config["dataset"]
    items = load_dataset(dataset_path)
    run = config.get("run", {})
    warmup = int(run.get("warmup", 1))
    repetitions = int(run.get("repetitions", 1))
    configured_concurrency = run.get("concurrency", 1)
    concurrencies = (
        configured_concurrency
        if isinstance(configured_concurrency, list)
        else [configured_concurrency]
    )
    timeout_s = float(run.get("timeout_s", 60))
    results: list[TargetResult] = []
    for target in config["targets"]:
        launch = target.get("launch")
        if launch and not allow_processes:
            raise ConfigError(
                f"Target '{target['name']}' launches a local process; rerun with --allow-processes after reviewing its command"
            )
        manager = ManagedProcess(launch) if launch else nullcontext()
        with manager as managed:
            adapter = create_adapter(
                target["adapter"], target.get("options", {}), timeout_s
            )
            for index in range(warmup):
                _execute(adapter, items[index % len(items)], -1)
            startup_seconds = (
                managed.startup_seconds if isinstance(managed, ManagedProcess) else None
            )
            for concurrency in concurrencies:
                jobs = [
                    (item, repetition)
                    for repetition in range(repetitions)
                    for item in items
                ]
                samples: list[Sample] = []
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [
                        pool.submit(_execute, adapter, item, repetition)
                        for item, repetition in jobs
                    ]
                    for future in as_completed(futures):
                        samples.append(future.result())
                wall_seconds = time.perf_counter() - started
                samples.sort(key=lambda sample: (sample.repetition, sample.item_id))
                metrics = summarize(samples, wall_seconds)
                metrics["concurrency"] = concurrency
                metrics["server_startup_seconds"] = startup_seconds
                hourly_cost = target.get("cost_per_hour_usd")
                metrics["cost_per_hour_usd"] = hourly_cost
                throughput = float(metrics["output_tokens_per_second"] or 0.0)
                metrics["estimated_usd_per_million_output_tokens"] = (
                    float(hourly_cost) * 1_000_000 / (3600 * throughput)
                    if hourly_cost is not None and throughput > 0
                    else None
                )
                candidate_name = (
                    target["name"]
                    if len(concurrencies) == 1
                    else f"{target['name']}@c{concurrency}"
                )
                results.append(
                    TargetResult(
                        name=candidate_name,
                        adapter=target["adapter"],
                        metrics=metrics,
                        samples=samples,
                    )
                )
    objective = config.get("objective", "throughput")
    selected, reasons = recommend(results, objective, config.get("constraints", {}))
    public_config = _public_config(config)
    definition = {
        "config": public_config,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(definition, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": config.get("name", "unnamed"),
        "objective": objective,
        "constraints": config.get("constraints", {}),
        "definition_fingerprint": fingerprint,
        "definition": definition,
        "environment": collect_environment(),
        "recommendation": {"target": selected, "reasons": reasons},
        "pareto_frontier": pareto_frontier(results, config.get("constraints", {})),
        "targets": [result.to_dict() for result in results],
    }


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    def sensitive_key(key: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
        if normalized.endswith("_env"):
            return False
        parts = set(normalized.split("_"))
        exact = {
            "api_key",
            "apikey",
            "authorization",
            "bearer",
            "credential",
            "credentials",
            "password",
            "passwd",
            "private_key",
            "secret",
            "token",
            "access_token",
            "client_secret",
        }
        return (
            normalized in exact
            or bool(
                parts
                & {
                    "authorization",
                    "bearer",
                    "credential",
                    "credentials",
                    "password",
                    "passwd",
                    "secret",
                    "token",
                }
            )
            or normalized == "key"
            or normalized.endswith("_key")
        )

    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for key, child in value.items():
                if key.startswith("_"):
                    continue
                cleaned[key] = "<redacted>" if sensitive_key(key) else sanitize(child)
            return cleaned
        if isinstance(value, list):
            cleaned_items: list[Any] = []
            redact_next = False
            for item in value:
                if redact_next:
                    cleaned_items.append("<redacted>")
                    redact_next = False
                    continue
                if isinstance(item, str) and item.startswith("-"):
                    option, separator, _option_value = item.partition("=")
                    if sensitive_key(option.lstrip("-")):
                        if separator:
                            cleaned_items.append(f"{option}=<redacted>")
                        else:
                            cleaned_items.append(item)
                            redact_next = True
                        continue
                cleaned_items.append(sanitize(item))
            return cleaned_items
        return value

    return sanitize(config)
