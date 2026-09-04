from __future__ import annotations

import json
import io
import tempfile
import tomllib
import unittest
from pathlib import Path

from runtimefit.config import ConfigError, load_config
from runtimefit.decision import choose
from runtimefit.decision_config import load_decision_config
from runtimefit.decision_report import decision_markdown
from runtimefit.evidence import load_candidate_evidence
from runtimefit import __version__
from runtimefit.adapters import OpenAIAdapter
from runtimefit.dataset import score_output
from runtimefit.metrics import percentile
from runtimefit.models import WorkItem
from runtimefit.runner import run_benchmark
from runtimefit.runner import _public_config
from runtimefit.models import TargetResult
from runtimefit.recommend import pareto_frontier, recommend
from scripts.render_homebrew_formula import render_formula


class RuntimeFitTests(unittest.TestCase):
    def test_package_version_matches_project_metadata(self) -> None:
        root = Path(__file__).parents[1]
        with (root / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)
        self.assertEqual(__version__, project["project"]["version"])

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)

    def test_decision_example_selects_cheapest_eligible_candidate(self) -> None:
        root = Path(__file__).parents[1]
        config = load_decision_config(root / "examples" / "decision" / "runtimefit.toml")
        result = choose(config)
        self.assertEqual(result["summary"]["eligible_count"], 2)
        self.assertEqual(result["summary"]["selected"], "sglang-fp16-c8")
        self.assertNotEqual(result["decision_fingerprint"], result["config_fingerprint"])
        report = decision_markdown(result)
        self.assertIn("2/4 configurations", report)
        self.assertIn("Requirement checks", report)
        self.assertIn("latency_p99_ms 4250 exceeds requirement 4000", report)

    def test_guidellm_v2_evidence_is_normalized(self) -> None:
        def distribution(p50: float, p95: float, p99: float, mean: float = 0.0) -> dict:
            return {"successful": {"mean": mean, "percentiles": {
                "p50": p50, "p95": p95, "p99": p99,
            }}}

        report = {
            "metadata": {"version": 2, "guidellm_version": "0.7.0"},
            "benchmarks": [{
                "config": {"id_": "bench-1", "run_index": 2, "strategy": {"kind": "concurrent"}},
                "metrics": {
                    "time_to_first_token_ms": distribution(100, 200, 300),
                    "time_per_output_token_ms": distribution(10, 20, 30),
                    "request_latency": distribution(1.0, 2.0, 3.0),
                    "requests_per_second": distribution(0, 0, 0, 8.5),
                    "output_tokens_per_second": distribution(0, 0, 0, 900),
                    "request_totals": {"successful": 99, "errored": 1, "incomplete": 0, "total": 100},
                },
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "benchmarks.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            loaded = load_candidate_evidence({
                "id": "candidate",
                "runtime": "vllm",
                "cost_per_hour_usd": 2,
                "evidence": {"provider": "guidellm", "path": "benchmarks.json", "benchmark_index": 0},
            }, root, 730)
        self.assertEqual(loaded["metrics"]["ttft_p95_ms"], 200)
        self.assertEqual(loaded["metrics"]["latency_p99_ms"], 3000)
        self.assertEqual(loaded["metrics"]["request_throughput_rps"], 8.5)
        self.assertEqual(loaded["metrics"]["error_rate"], 0.01)
        self.assertEqual(loaded["metrics"]["monthly_cost_usd"], 1460)
        self.assertEqual(loaded["source"]["guidellm_version"], "0.7.0")

    def test_decision_config_rejects_runtime_outside_v01_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.toml"
            path.write_text('''
                [[candidates]]
                id = "outside-scope"
                runtime = "llama.cpp"
                [candidates.evidence.metrics]
                latency_p99_ms = 100
            ''', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "v0.1 supports"):
                load_decision_config(path)

    def test_quality_matching(self) -> None:
        self.assertEqual(score_output(WorkItem("1", "p", "Hello World"), " hello   world "), 1.0)
        self.assertEqual(score_output(WorkItem("1", "p", "world", "contains"), "Hello WORLD!"), 1.0)

    def test_recommends_faster_eligible_target(self) -> None:
        root = Path(__file__).parents[1]
        config = load_config(root / "examples" / "demo.json")
        result = run_benchmark(config)
        self.assertEqual(result["recommendation"]["target"], "optimized-runtime")
        self.assertEqual(len(result["targets"]), 2)

    def test_stream_parser_collects_content_and_usage(self) -> None:
        stream = io.BytesIO(
            b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n'
            b'data: [DONE]\n\n'
        )
        text, tokens, first_token = OpenAIAdapter._read_stream(stream, 0.0)
        self.assertEqual(text, "Hello")
        self.assertEqual(tokens, 2)
        self.assertGreater(first_token, 0)

    def test_quality_constraint_rejects_fast_bad_target(self) -> None:
        slow_good = TargetResult("slow-good", "mock", {
            "quality": 1.0, "error_rate": 0.0, "output_tokens_per_second": 10.0,
            "latency_p95_ms": 100.0, "ttft_p95_ms": 50.0,
        })
        fast_bad = TargetResult("fast-bad", "mock", {
            "quality": 0.5, "error_rate": 0.0, "output_tokens_per_second": 100.0,
            "latency_p95_ms": 10.0, "ttft_p95_ms": 5.0,
        })
        selected, reasons = recommend(
            [slow_good, fast_bad], "throughput", {"max_quality_loss": 0.01}
        )
        self.assertEqual(selected, "slow-good")
        self.assertIn("quality loss", reasons["fast-bad"])

    def test_latency_slo_rejects_throughput_winner(self) -> None:
        low_latency = TargetResult("low-latency", "mock", {
            "quality": 1.0, "error_rate": 0.0, "output_tokens_per_second": 20.0,
            "latency_p95_ms": 100.0, "ttft_p95_ms": 20.0,
        })
        high_throughput = TargetResult("high-throughput", "mock", {
            "quality": 1.0, "error_rate": 0.0, "output_tokens_per_second": 100.0,
            "latency_p95_ms": 500.0, "ttft_p95_ms": 30.0,
        })
        selected, reasons = recommend(
            [low_latency, high_throughput],
            "throughput",
            {"max_p95_latency_ms": 200},
        )
        self.assertEqual(selected, "low-latency")
        self.assertIn("p95 latency", reasons["high-throughput"])

    def test_cost_objective_and_pareto_frontier(self) -> None:
        cheap = TargetResult("cheap", "mock", {
            "quality": 1.0, "error_rate": 0.0, "output_tokens_per_second": 50.0,
            "latency_p95_ms": 200.0, "ttft_p95_ms": 20.0,
            "estimated_usd_per_million_output_tokens": 0.2,
        })
        expensive = TargetResult("expensive", "mock", {
            "quality": 1.0, "error_rate": 0.0, "output_tokens_per_second": 40.0,
            "latency_p95_ms": 300.0, "ttft_p95_ms": 30.0,
            "estimated_usd_per_million_output_tokens": 0.5,
        })
        selected, _ = recommend([cheap, expensive], "cost", {})
        self.assertEqual(selected, "cheap")
        self.assertEqual(pareto_frontier([cheap, expensive]), ["cheap"])

    def test_public_config_redacts_inline_secrets_but_keeps_env_name(self) -> None:
        public = _public_config({
            "token": "do-not-leak",
            "api_key_env": "MY_API_KEY",
            "max_tokens": 64,
            "nested": {"password": "also-secret"},
            "_config_dir": "/private/path",
        })
        self.assertEqual(public["token"], "<redacted>")
        self.assertEqual(public["api_key_env"], "MY_API_KEY")
        self.assertEqual(public["max_tokens"], 64)
        self.assertEqual(public["nested"]["password"], "<redacted>")
        self.assertNotIn("_config_dir", public)

    def test_rejects_duplicate_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.jsonl").write_text('{"prompt":"hello"}\n', encoding="utf-8")
            config = {
                "dataset": "data.jsonl",
                "targets": [
                    {"name": "same", "adapter": "mock"},
                    {"name": "same", "adapter": "mock"},
                ],
            }
            path = root / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_managed_process_requires_explicit_permission(self) -> None:
        root = Path(__file__).parents[1]
        config = load_config(root / "examples" / "demo.json")
        config["targets"][0]["launch"] = {
            "command": ["never-run-this"],
            "health_url": "http://127.0.0.1:9999/health",
        }
        with self.assertRaises(ConfigError):
            run_benchmark(config)

    def test_concurrency_sweep_creates_distinct_candidates(self) -> None:
        root = Path(__file__).parents[1]
        config = load_config(root / "examples" / "demo.json")
        config["run"]["concurrency"] = [1, 2]
        result = run_benchmark(config)
        names = [target["name"] for target in result["targets"]]
        self.assertEqual(names, [
            "reference-runtime@c1", "reference-runtime@c2",
            "optimized-runtime@c1", "optimized-runtime@c2",
        ])

    def test_homebrew_formula_contains_release_identity(self) -> None:
        formula = render_formula("1.2.3", "a" * 64, "example")
        self.assertIn("runtimefit-1.2.3.tar.gz", formula)
        self.assertIn("releases/download/v1.2.3", formula)
        self.assertIn('sha256 "' + "a" * 64 + '"', formula)
        self.assertIn("https://github.com/example/runtimefit", formula)
        self.assertIn('#{bin}/runtimefit --version', formula)


if __name__ == "__main__":
    unittest.main()
