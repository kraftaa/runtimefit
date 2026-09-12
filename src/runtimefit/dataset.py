from __future__ import annotations

import json
from pathlib import Path

from runtimefit.config import ConfigError
from runtimefit.models import WorkItem


def load_dataset(path: str | Path) -> list[WorkItem]:
    items: list[WorkItem] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(
                    f"Invalid dataset JSON on line {line_number}: {exc.msg}"
                ) from exc
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ConfigError(
                    f"Dataset line {line_number} needs a non-empty prompt"
                )
            match = row.get("match", "exact")
            if match not in {"exact", "contains"}:
                raise ConfigError(
                    f"Dataset line {line_number}: match must be exact or contains"
                )
            items.append(
                WorkItem(
                    id=str(row.get("id", line_number)),
                    prompt=prompt,
                    expected=row.get("expected"),
                    match=match,
                )
            )
    if not items:
        raise ConfigError("Dataset contains no work items")
    return items


def score_output(item: WorkItem, output: str) -> float | None:
    if item.expected is None:
        return None
    actual = " ".join(output.casefold().split())
    expected = " ".join(str(item.expected).casefold().split())
    if item.match == "contains":
        return float(expected in actual)
    return float(actual == expected)
