from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    validate_config(data, config_path.parent)
    data["_config_dir"] = str(config_path.parent.resolve())
    return data


def validate_config(data: Any, base_dir: Path = Path(".")) -> None:
    if not isinstance(data, dict):
        raise ConfigError("Config root must be a JSON object")
    targets = data.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ConfigError("'targets' must be a non-empty list")
    names: set[str] = set()
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise ConfigError(f"targets[{index}] must be an object")
        for key in ("name", "adapter"):
            if not isinstance(target.get(key), str) or not target[key].strip():
                raise ConfigError(f"targets[{index}].{key} must be a non-empty string")
        if target["name"] in names:
            raise ConfigError(f"Duplicate target name: {target['name']}")
        names.add(target["name"])
        if target["adapter"] not in {"mock", "openai"}:
            raise ConfigError(f"Unsupported adapter: {target['adapter']}")
        if not isinstance(target.get("options", {}), dict):
            raise ConfigError(f"targets[{index}].options must be an object")
        launch = target.get("launch")
        if launch is not None:
            if not isinstance(launch, dict):
                raise ConfigError(f"targets[{index}].launch must be an object")
            command = launch.get("command")
            if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
                raise ConfigError(f"targets[{index}].launch.command must be a non-empty list of strings")
            health_url = launch.get("health_url")
            if not isinstance(health_url, str) or not health_url.startswith(("http://127.0.0.1", "http://localhost")):
                raise ConfigError(f"targets[{index}].launch.health_url must use localhost")
    dataset = data.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ConfigError("'dataset' must be a JSONL path")
    if not (base_dir / dataset).is_file():
        raise ConfigError(f"Dataset not found: {base_dir / dataset}")
    run = data.get("run", {})
    for key in ("warmup", "repetitions"):
        value = run.get(key, 1)
        if not isinstance(value, int) or value < (0 if key == "warmup" else 1):
            raise ConfigError(f"run.{key} must be a valid integer")
    concurrency = run.get("concurrency", 1)
    if isinstance(concurrency, int):
        valid_concurrency = concurrency >= 1
    else:
        valid_concurrency = (
            isinstance(concurrency, list)
            and bool(concurrency)
            and all(isinstance(value, int) and value >= 1 for value in concurrency)
            and len(set(concurrency)) == len(concurrency)
        )
    if not valid_concurrency:
        raise ConfigError("run.concurrency must be a positive integer or a non-empty list of unique positive integers")
    objective = data.get("objective", "throughput")
    if objective not in {"throughput", "latency", "ttft", "cost"}:
        raise ConfigError("objective must be throughput, latency, ttft, or cost")
    constraints = data.get("constraints", {})
    if not isinstance(constraints, dict):
        raise ConfigError("constraints must be an object")
    allowed_constraints = {
        "max_quality_loss",
        "max_error_rate",
        "max_p95_latency_ms",
        "max_p95_ttft_ms",
        "min_output_tokens_per_second",
        "max_usd_per_million_output_tokens",
    }
    unknown_constraints = set(constraints) - allowed_constraints
    if unknown_constraints:
        raise ConfigError(f"Unsupported constraints: {', '.join(sorted(unknown_constraints))}")
    for key, value in constraints.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"constraints.{key} must be a non-negative number")
    for index, target in enumerate(targets):
        cost = target.get("cost_per_hour_usd")
        if cost is not None and (
            not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0
        ):
            raise ConfigError(f"targets[{index}].cost_per_hour_usd must be a non-negative number")
