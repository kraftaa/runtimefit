from __future__ import annotations

import io
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from runtimefit import __version__
from runtimefit.adapters import OpenAIAdapter
from runtimefit.config import ConfigError, load_config
from runtimefit.dataset import score_output
from runtimefit.decision import choose
from runtimefit.decision_config import load_decision_config
from runtimefit.decision_report import decision_markdown
from runtimefit.evidence import load_candidate_evidence
from runtimefit.metrics import percentile
from runtimefit.models import TargetResult, WorkItem
from runtimefit.process import ManagedProcess
from runtimefit.recommend import pareto_frontier, recommend
from runtimefit.runner import _public_config, run_benchmark
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
        config = load_decision_config(
            root / "examples" / "decision" / "runtimefit.toml"
        )
        result = choose(config)
        self.assertEqual(result["summary"]["eligible_count"], 2)
        self.assertEqual(result["summary"]["selected"], "sglang-fp16-c8")
        self.assertEqual(result["summary"]["fastest_candidate"], "vllm-awq-c8")
        self.assertFalse(result["summary"]["fastest_candidate_eligible"])
        self.assertEqual(result["pareto_frontier"], ["sglang-fp16-c8", "vllm-fp16-c8"])
        self.assertNotEqual(
            result["decision_fingerprint"], result["config_fingerprint"]
        )
        selected = next(
            candidate
            for candidate in result["candidates"]
            if candidate["id"] == "sglang-fp16-c8"
        )
        self.assertEqual(selected["metrics"]["required_replicas"], 1)
        self.assertAlmostEqual(selected["metrics"]["throughput_headroom_fraction"], 0.3)
        self.assertAlmostEqual(selected["metrics"]["monthly_cost_usd"], 1430.8)
        scaled = next(
            candidate
            for candidate in result["candidates"]
            if candidate["id"] == "sglang-awq-c8"
        )
        self.assertEqual(scaled["metrics"]["required_replicas"], 2)
        report = decision_markdown(result)
        self.assertIn("2/4 configurations", report)
        self.assertIn("Requirement checks", report)
        self.assertIn("Fastest candidate rejected", report)
        self.assertIn("p99 latency 4.25 s > 4.00 s", report)

    def test_guidellm_v2_evidence_is_normalized(self) -> None:
        def distribution(p50: float, p95: float, p99: float, mean: float = 0.0) -> dict:
            return {
                "successful": {
                    "mean": mean,
                    "percentiles": {
                        "p50": p50,
                        "p95": p95,
                        "p99": p99,
                    },
                }
            }

        report = {
            "metadata": {"version": 2, "guidellm_version": "0.7.0"},
            "benchmarks": [
                {
                    "config": {
                        "id_": "bench-1",
                        "run_index": 2,
                        "strategy": {"kind": "concurrent"},
                    },
                    "metrics": {
                        "time_to_first_token_ms": distribution(100, 200, 300),
                        "time_per_output_token_ms": distribution(10, 20, 30),
                        "request_latency": distribution(1.0, 2.0, 3.0),
                        "requests_per_second": distribution(0, 0, 0, 8.5),
                        "output_tokens_per_second": distribution(0, 0, 0, 900),
                        "request_totals": {
                            "successful": 99,
                            "errored": 1,
                            "incomplete": 0,
                            "total": 100,
                        },
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "benchmarks.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            loaded = load_candidate_evidence(
                {
                    "id": "candidate",
                    "runtime": "vllm",
                    "cost_per_hour_usd": 2,
                    "evidence": {
                        "provider": "guidellm",
                        "path": "benchmarks.json",
                        "benchmark_index": 0,
                    },
                },
                root,
                730,
            )
        self.assertEqual(loaded["metrics"]["ttft_p95_ms"], 200)
        self.assertEqual(loaded["metrics"]["latency_p99_ms"], 3000)
        self.assertEqual(loaded["metrics"]["request_throughput_rps"], 8.5)
        self.assertEqual(loaded["metrics"]["error_rate"], 0.01)
        self.assertEqual(loaded["metrics"]["monthly_cost_usd"], 1460)
        self.assertEqual(loaded["source"]["guidellm_version"], "0.7.0")
        self.assertEqual(loaded["source"]["sample_counts"]["latency_p99_ms"], 99)

    def test_guidellm_repeated_evidence_uses_median_and_records_variability(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, p99_seconds in enumerate((2.0, 5.0, 3.0), start=1):
                report = {
                    "metadata": {"version": 2, "guidellm_version": "0.7.2"},
                    "benchmarks": [
                        {
                            "config": {"id_": f"run-{index}", "run_index": 0},
                            "metrics": {
                                "request_latency": {
                                    "successful": {
                                        "percentiles": {
                                            "p50": 1.0,
                                            "p95": 1.5,
                                            "p99": p99_seconds,
                                        }
                                    }
                                },
                                "requests_per_second": {
                                    "successful": {"mean": 10 + index}
                                },
                                "request_totals": {
                                    "successful": 100,
                                    "errored": 0,
                                    "total": 100,
                                },
                            },
                        }
                    ],
                }
                path = root / f"run-{index}.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                paths.append(path.name)
            loaded = load_candidate_evidence(
                {
                    "id": "candidate",
                    "runtime": "vllm",
                    "evidence": {
                        "provider": "guidellm",
                        "paths": paths,
                        "benchmark_index": 0,
                    },
                },
                root,
                730,
            )
        self.assertEqual(loaded["metrics"]["latency_p99_ms"], 3000)
        self.assertEqual(loaded["metrics"]["request_throughput_rps"], 12)
        self.assertEqual(loaded["source"]["run_count"], 3)
        self.assertEqual(len(loaded["source"]["runs"]), 3)
        self.assertEqual(
            loaded["source"]["variability"]["latency_p99_ms"]["relative_range"], 1.0
        )
        self.assertEqual(loaded["source"]["sample_counts"]["latency_p99_ms"], 100)

    def test_decision_config_rejects_runtime_outside_v01_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.toml"
            path.write_text(
                """
                [[candidates]]
                id = "outside-scope"
                runtime = "llama.cpp"
                [candidates.evidence.metrics]
                latency_p99_ms = 100
            """,
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "currently supports"):
                load_decision_config(path)

    def test_quality_matching(self) -> None:
        self.assertEqual(
            score_output(WorkItem("1", "p", "Hello World"), " hello   world "), 1.0
        )
        self.assertEqual(
            score_output(WorkItem("1", "p", "world", "contains"), "Hello WORLD!"), 1.0
        )

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
            b"data: [DONE]\n\n"
        )
        text, tokens, first_token = OpenAIAdapter._read_stream(stream, 0.0)
        self.assertEqual(text, "Hello")
        self.assertEqual(tokens, 2)
        self.assertGreater(first_token, 0)

    def test_stream_without_content_has_no_ttft(self) -> None:
        stream = io.BytesIO(
            b'data: {"choices":[],"usage":{"completion_tokens":0}}\n\ndata: [DONE]\n\n'
        )
        text, tokens, first_token = OpenAIAdapter._read_stream(stream, 0.0)
        self.assertEqual(text, "")
        self.assertEqual(tokens, 0)
        self.assertIsNone(first_token)

    def test_stream_requires_done_marker(self) -> None:
        stream = io.BytesIO(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
        with self.assertRaisesRegex(RuntimeError, r"before \[DONE\]"):
            OpenAIAdapter._read_stream(stream, 0.0)

    def test_stream_enforces_total_deadline(self) -> None:
        stream = io.BytesIO(b"data: [DONE]\n\n")
        with self.assertRaisesRegex(TimeoutError, "deadline exceeded"):
            OpenAIAdapter._read_stream(stream, 0.0, deadline=0.0)

    def test_non_streaming_adapter_parses_chat_completion(self) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"completion_tokens": 1},
            }
        ).encode()
        response.__enter__.return_value = response
        adapter = OpenAIAdapter(
            {"base_url": "http://127.0.0.1:8000/v1", "streaming": False}, 1
        )
        with patch("runtimefit.adapters.urllib.request.urlopen", return_value=response):
            generation = adapter.generate("test")
        self.assertEqual(generation.text, "hello")
        self.assertEqual(generation.output_tokens, 1)
        self.assertEqual(generation.token_count_source, "reported")

    def test_quality_constraint_rejects_fast_bad_target(self) -> None:
        slow_good = TargetResult(
            "slow-good",
            "mock",
            {
                "quality": 1.0,
                "error_rate": 0.0,
                "output_tokens_per_second": 10.0,
                "latency_p95_ms": 100.0,
                "ttft_p95_ms": 50.0,
            },
        )
        fast_bad = TargetResult(
            "fast-bad",
            "mock",
            {
                "quality": 0.5,
                "error_rate": 0.0,
                "output_tokens_per_second": 100.0,
                "latency_p95_ms": 10.0,
                "ttft_p95_ms": 5.0,
            },
        )
        selected, reasons = recommend(
            [slow_good, fast_bad], "throughput", {"max_quality_loss": 0.01}
        )
        self.assertEqual(selected, "slow-good")
        self.assertIn("quality loss", reasons["fast-bad"])

    def test_latency_slo_rejects_throughput_winner(self) -> None:
        low_latency = TargetResult(
            "low-latency",
            "mock",
            {
                "quality": 1.0,
                "error_rate": 0.0,
                "output_tokens_per_second": 20.0,
                "latency_p95_ms": 100.0,
                "ttft_p95_ms": 20.0,
            },
        )
        high_throughput = TargetResult(
            "high-throughput",
            "mock",
            {
                "quality": 1.0,
                "error_rate": 0.0,
                "output_tokens_per_second": 100.0,
                "latency_p95_ms": 500.0,
                "ttft_p95_ms": 30.0,
            },
        )
        selected, reasons = recommend(
            [low_latency, high_throughput],
            "throughput",
            {"max_p95_latency_ms": 200},
        )
        self.assertEqual(selected, "low-latency")
        self.assertIn("p95 latency", reasons["high-throughput"])

    def test_cost_objective_and_pareto_frontier(self) -> None:
        cheap = TargetResult(
            "cheap",
            "mock",
            {
                "quality": 1.0,
                "error_rate": 0.0,
                "output_tokens_per_second": 50.0,
                "latency_p95_ms": 200.0,
                "ttft_p95_ms": 20.0,
                "estimated_usd_per_million_output_tokens": 0.2,
            },
        )
        expensive = TargetResult(
            "expensive",
            "mock",
            {
                "quality": 1.0,
                "error_rate": 0.0,
                "output_tokens_per_second": 40.0,
                "latency_p95_ms": 300.0,
                "ttft_p95_ms": 30.0,
                "estimated_usd_per_million_output_tokens": 0.5,
            },
        )
        selected, _ = recommend([cheap, expensive], "cost", {})
        self.assertEqual(selected, "cheap")
        self.assertEqual(pareto_frontier([cheap, expensive]), ["cheap"])

    def test_runner_selection_rejects_missing_required_evidence(self) -> None:
        incomplete = TargetResult(
            "incomplete",
            "mock",
            {
                "output_tokens_per_second": 100.0,
                "latency_p95_ms": 10.0,
            },
        )
        selected, reasons = recommend([incomplete], "throughput", {})
        self.assertIsNone(selected)
        self.assertEqual(reasons["incomplete"], "missing error rate")

    def test_public_config_redacts_inline_secrets_but_keeps_env_name(self) -> None:
        public = _public_config(
            {
                "token": "do-not-leak",
                "api_key_env": "MY_API_KEY",
                "max_tokens": 64,
                "nested": {"password": "also-secret"},
                "_config_dir": "/private/path",
            }
        )
        self.assertEqual(public["token"], "<redacted>")
        self.assertEqual(public["api_key_env"], "MY_API_KEY")
        self.assertEqual(public["max_tokens"], 64)
        self.assertEqual(public["nested"]["password"], "<redacted>")
        self.assertNotIn("_config_dir", public)

    def test_public_config_redacts_common_secret_key_variants(self) -> None:
        public = _public_config(
            {
                "apikey": "one",
                "api-key": "two",
                "bearer": "three",
                "credential": "four",
                "signing_key": "five",
                "headers": {"Authorization": "six"},
                "command": ["server", "--api-key", "seven", "--token=eight"],
                "api-key-env": "SAFE_ENV_NAME",
                "max_tokens": 64,
            }
        )
        for key in ("apikey", "api-key", "bearer", "credential", "signing_key"):
            self.assertEqual(public[key], "<redacted>")
        self.assertEqual(public["headers"]["Authorization"], "<redacted>")
        self.assertEqual(
            public["command"],
            ["server", "--api-key", "<redacted>", "--token=<redacted>"],
        )
        self.assertEqual(public["api-key-env"], "SAFE_ENV_NAME")
        self.assertEqual(public["max_tokens"], 64)

    def test_health_check_requires_success_status(self) -> None:
        process = MagicMock()
        process.poll.return_value = None
        process.wait.return_value = 0
        response = MagicMock()
        response.status = 404
        response.__enter__.return_value = response
        specification = {
            "command": ["server"],
            "health_url": "http://127.0.0.1:9999/health",
            "startup_timeout_s": 0.001,
        }
        with (
            patch("runtimefit.process.subprocess.Popen", return_value=process),
            patch("runtimefit.process.urllib.request.urlopen", return_value=response),
            patch("runtimefit.process.time.sleep", return_value=None),
        ):
            with self.assertRaisesRegex(ConfigError, "did not become healthy"):
                with ManagedProcess(specification):
                    pass

    def test_bare_unset_environment_variable_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unset environment variable"):
            ManagedProcess(
                {
                    "command": ["server", "$RUNTIMEFIT_MISSING_TEST_VARIABLE"],
                    "health_url": "http://127.0.0.1:9999/health",
                }
            )

    def test_exact_capacity_does_not_round_up_replica_count(self) -> None:
        config = {
            "_config_dir": ".",
            "objective": "lowest_monthly_cost",
            "workload": {"requests_per_second": 8},
            "requirements": {"min_throughput_headroom_fraction": 0.1},
            "candidates": [
                {
                    "id": "exact",
                    "runtime": "vllm",
                    "cost_per_hour_usd": 1,
                    "evidence": {
                        "provider": "inline",
                        "metrics": {
                            "request_throughput_rps": 8.8,
                        },
                    },
                }
            ],
        }
        result = choose(config)
        self.assertEqual(result["candidates"][0]["metrics"]["required_replicas"], 1)

    def test_repeated_evidence_uses_worst_observed_value_for_slo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_guidellm_runs(root, "candidate", (2.0, 5.0, 3.0))
            config = {
                "_config_dir": str(root),
                "objective": "lowest_p99_latency",
                "requirements": {"p99_latency_ms": 4000},
                "candidates": [
                    {
                        "id": "candidate",
                        "runtime": "vllm",
                        "evidence": {
                            "provider": "guidellm",
                            "paths": paths,
                            "benchmark_index": 0,
                        },
                    }
                ],
            }
            result = choose(config)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["metrics"]["latency_p99_ms"], 3000)
        self.assertEqual(candidate["checks"][0]["actual"], 5000)
        self.assertFalse(candidate["eligible"])

    def test_overlapping_objective_ranges_are_reported_as_indistinguishable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._write_guidellm_runs(root, "first", (0.100, 0.120, 0.110))
            second = self._write_guidellm_runs(root, "second", (0.105, 0.115, 0.112))
            config = {
                "_config_dir": str(root),
                "objective": "lowest_p99_latency",
                "requirements": {},
                "candidates": [
                    {
                        "id": "first",
                        "runtime": "vllm",
                        "evidence": {
                            "provider": "guidellm",
                            "paths": first,
                            "benchmark_index": 0,
                        },
                    },
                    {
                        "id": "second",
                        "runtime": "sglang",
                        "evidence": {
                            "provider": "guidellm",
                            "paths": second,
                            "benchmark_index": 0,
                        },
                    },
                ],
            }
            result = choose(config)
        self.assertEqual(result["summary"]["selection_status"], "indistinguishable")
        self.assertEqual(result["summary"]["selected"], "first")
        self.assertEqual(result["summary"]["indistinguishable_candidates"], ["second"])

    @staticmethod
    def _write_guidellm_runs(
        root: Path, prefix: str, p99_seconds: tuple[float, float, float]
    ) -> list[str]:
        paths: list[str] = []
        for index, p99 in enumerate(p99_seconds, start=1):
            report = {
                "metadata": {"version": 2, "guidellm_version": "0.7.2"},
                "benchmarks": [
                    {
                        "config": {"id_": f"{prefix}-{index}", "run_index": index},
                        "metrics": {
                            "request_latency": {
                                "successful": {
                                    "percentiles": {
                                        "p50": p99 / 2,
                                        "p95": p99 * 0.9,
                                        "p99": p99,
                                    }
                                }
                            },
                            "requests_per_second": {"successful": {"mean": 10}},
                            "request_totals": {
                                "successful": 1000,
                                "errored": 0,
                                "total": 1000,
                            },
                        },
                    }
                ],
            }
            path = root / f"{prefix}-{index}.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            paths.append(path.name)
        return paths

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
        self.assertEqual(
            names,
            [
                "reference-runtime@c1",
                "reference-runtime@c2",
                "optimized-runtime@c1",
                "optimized-runtime@c2",
            ],
        )

    def test_homebrew_formula_contains_release_identity(self) -> None:
        formula = render_formula("1.2.3", "a" * 64, "example")
        self.assertIn("runtimefit-1.2.3.tar.gz", formula)
        self.assertIn("releases/download/v1.2.3", formula)
        self.assertIn('sha256 "' + "a" * 64 + '"', formula)
        self.assertIn("https://github.com/example/runtimefit", formula)
        self.assertIn("#{bin}/runtimefit --version", formula)


if __name__ == "__main__":
    unittest.main()
