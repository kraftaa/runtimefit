from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkItem:
    id: str
    prompt: str
    expected: str | None = None
    match: str = "exact"


@dataclass(frozen=True)
class Generation:
    text: str
    latency_ms: float
    ttft_ms: float | None
    output_tokens: int
    token_count_source: str


@dataclass(frozen=True)
class Sample:
    item_id: str
    repetition: int
    latency_ms: float
    ttft_ms: float | None
    output_tokens: int
    token_count_source: str
    quality: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TargetResult:
    name: str
    adapter: str
    metrics: dict[str, float | int | None]
    samples: list[Sample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "adapter": self.adapter,
            "metrics": self.metrics,
            "samples": [sample.to_dict() for sample in self.samples],
        }
