from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from runtimefit.models import Generation


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


class Adapter(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> Generation:
        raise NotImplementedError


class MockAdapter(Adapter):
    """Deterministic test adapter; useful for verifying the benchmark harness."""

    def __init__(self, options: dict[str, Any]):
        self.latency_ms = float(options.get("latency_ms", 20))
        self.ttft_ms = float(options.get("ttft_ms", self.latency_ms / 2))
        self.response = str(options.get("response", "ok"))
        self.responses = {str(key): str(value) for key, value in options.get("responses", {}).items()}
        self.fail_on = {str(value) for value in options.get("fail_on", [])}

    def generate(self, prompt: str) -> Generation:
        started = time.perf_counter()
        time.sleep(self.latency_ms / 1000)
        if prompt in self.fail_on:
            raise RuntimeError("configured mock failure")
        latency_ms = (time.perf_counter() - started) * 1000
        text = self.responses.get(prompt, self.response)
        return Generation(
            text=text,
            latency_ms=latency_ms,
            ttft_ms=min(self.ttft_ms, latency_ms),
            output_tokens=estimate_tokens(text),
            token_count_source="estimated",
        )


class OpenAIAdapter(Adapter):
    """Adapter for OpenAI-compatible chat-completions endpoints."""

    def __init__(self, options: dict[str, Any], timeout_s: float):
        base_url = str(options.get("base_url", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.url = f"{base_url}/chat/completions"
        self.model = str(options.get("model", "default"))
        self.timeout_s = timeout_s
        self.generation = dict(options.get("generation", {}))
        self.streaming = bool(options.get("streaming", True))
        api_key_env = str(options.get("api_key_env", ""))
        self.api_key = os.environ.get(api_key_env) if api_key_env else None

    def generate(self, prompt: str) -> Generation:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": self.streaming,
            **self.generation,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                if self.streaming:
                    text, output_tokens, first_token = self._read_stream(response, started)
                    finished = time.perf_counter()
                    return Generation(
                        text=text,
                        latency_ms=(finished - started) * 1000,
                        ttft_ms=(first_token - started) * 1000,
                        output_tokens=output_tokens if output_tokens is not None else estimate_tokens(text),
                        token_count_source="reported" if output_tokens is not None else "estimated",
                    )
                response_started = time.perf_counter()
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Connection failed: {exc.reason}") from exc
        finished = time.perf_counter()
        try:
            parsed = json.loads(body)
            text = parsed["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Endpoint returned an invalid chat-completions response") from exc
        usage_tokens = parsed.get("usage", {}).get("completion_tokens")
        return Generation(
            text=str(text),
            latency_ms=(finished - started) * 1000,
            ttft_ms=(response_started - started) * 1000,
            output_tokens=int(usage_tokens) if usage_tokens is not None else estimate_tokens(str(text)),
            token_count_source="reported" if usage_tokens is not None else "estimated",
        )

    @staticmethod
    def _read_stream(response: Any, started: float) -> tuple[str, int | None, float]:
        chunks: list[str] = []
        output_tokens: int | None = None
        first_token: float | None = None
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Endpoint returned invalid JSON in its SSE stream") from exc
            usage = event.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                output_tokens = int(usage["completion_tokens"])
            choices = event.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if content:
                if first_token is None:
                    first_token = time.perf_counter()
                chunks.append(str(content))
        if first_token is None:
            first_token = time.perf_counter()
        return "".join(chunks), output_tokens, first_token


def create_adapter(kind: str, options: dict[str, Any], timeout_s: float) -> Adapter:
    if kind == "mock":
        return MockAdapter(options)
    if kind == "openai":
        return OpenAIAdapter(options, timeout_s)
    raise ValueError(f"Unsupported adapter: {kind}")
