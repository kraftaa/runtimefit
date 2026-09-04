# RuntimeFit

**Benchmarks measure. RuntimeFit decides.**

RuntimeFit is an open-source decision engine that turns LLM inference benchmarks into
deployment recommendations.

Give it your workload evidence, deployment constraints, and candidate configurations.
RuntimeFit rejects configurations that violate your SLOs, exposes the Pareto frontier,
and explains which configuration to deploy and why.

RuntimeFit answers: **Given my workload and SLOs, what should I actually deploy?**

## Installation

After the first public release:

```bash
pip install runtimefit
```

Or, from the project's Homebrew tap:

```bash
brew install kraftaa/tap/runtimefit
```
Source-checkout commands below remain available during development.

## Why this exists

Tools such as GuideLLM, vLLM benchmarks, and NVIDIA performance tooling already measure
inference well. But a benchmark chart does not decide whether a speed improvement is
worth its cost or whether the fastest configuration violates a latency SLO. RuntimeFit
turns independently collected measurements into an explicit, reproducible decision.

## Choose a deployment

The dependency-free configuration format is TOML or JSON. YAML is available through
the optional `runtimefit[yaml]` installation extra.

```bash
runtimefit validate examples/decision/runtimefit.toml
runtimefit choose examples/decision/runtimefit.toml \
  --output results/decision.json \
  --report results/decision.md
```

Example output:

```text
2/4 configurations satisfy requirements.
Winner: sglang-fp16-c8
Reason: lowest estimated monthly cost among candidates satisfying every requirement.
Fastest rejected: vllm-awq-c8
  - p99 latency 4.25 s > 4.00 s
Evidence: results/decision.json
```

The result records every requirement check, rejection reason, alternative comparison,
source-evidence hash, selected candidate, and Pareto frontier.

The metrics in the checked-in decision example are synthetic and demonstrate the
selection contract; they are not performance claims about either runtime.

Evidence providers currently include inline metrics, RuntimeFit's development runner,
and GuideLLM report schema v2. See the [GuideLLM provider documentation](https://github.com/kraftaa/runtimefit/blob/main/docs/providers/guidellm.md).

The v0.1 decision contract supports vLLM and SGLang candidates across concurrency and
quantization settings. Constraints cover TTFT, end-to-end latency, throughput, error
rate, capacity headroom, GPU memory, and monthly cost. When hourly instance cost and
required request rate are supplied, RuntimeFit derives the replica count and monthly
deployment cost. RuntimeFit does not blend unlike measurements into an opaque score:
it filters by hard requirements, ranks feasible candidates by one declared objective,
and keeps the Pareto tradeoffs visible.

Quality is deliberately outside the decision MVP. Comparing it responsibly requires a
separate evaluator protocol, particularly when model weights or quantization change.

## Built-in development provider

RuntimeFit retains a small OpenAI-compatible runner for development, smoke tests, and
producing native evidence. It is not intended to replace dedicated load generators.

### Try the deterministic runner demo

No third-party packages or GPU are required:

```bash
PYTHONPATH=src python -m runtimefit validate-benchmark examples/demo.json
PYTHONPATH=src python -m runtimefit doctor
PYTHONPATH=src python -m runtimefit run examples/demo.json \
  --output results/demo.json \
  --report results/demo.md
```

The demo uses simulated runtimes solely to test the harness. Do not publish its numbers
as performance evidence.

## Benchmark real servers

RuntimeFit currently supports OpenAI-compatible `/v1/chat/completions` endpoints, so
the same workload can be sent to servers such as vLLM or SGLang:

```bash
cp examples/openai-compatible.json my-benchmark.json
# Edit endpoint URLs, model names, prompts, and constraints.
PYTHONPATH=src python -m runtimefit run my-benchmark.json \
  -o results/my-benchmark.json \
  --report results/my-benchmark.md
```

Targets may optionally contain a `launch` block with a command list and localhost
health URL. RuntimeFit starts each managed server, waits until it is healthy, records
startup time, and stops it after the run. Because this executes local programs, it is
disabled unless you pass `--allow-processes` after reviewing the configuration.

Environment variables can keep machine-specific model paths out of configurations:

```bash
export RUNTIMEFIT_MODEL=/absolute/path/to/model.gguf
runtimefit run benchmark.json -o result.json --allow-processes
```

Dataset files use OpenAI-compatible JSON Lines:

```json
{"id":"q1","prompt":"Summarize this support request: ..."}
{"id":"q2","prompt":"Extract the requested fields from: ..."}
```

Use `api_key_env` in target options when authentication is required. RuntimeFit reads
the named environment variable and never places its value in result files.

## What the development runner measures

- p50, p95, and p99 end-to-end latency
- true time to first generated token for streaming endpoints
- requests and output tokens per second
- request error rate
- Python and platform metadata
- installed inference-tool versions and non-identifying hardware metadata
- a SHA-256 fingerprint covering the sanitized configuration and dataset contents
- individual samples for independent analysis

See [the decision methodology](https://github.com/kraftaa/runtimefit/blob/main/docs/methodology.md) for evidence normalization,
eligibility, ranking, Pareto analysis, and known limitations.

## Important limitations

- Non-streaming targets measure time to response headers rather than true first-token latency.
- Character-based token estimates are used when a server omits usage data.
- The built-in runner does not yet measure GPU memory or provision configurations;
  those values must come from external evidence or explicit cost inputs.
- RuntimeFit compares declared candidates; it does not yet search runtime parameters.
- Comparable results require identical hardware, model weights, prompts, generation
  parameters, warm-up policy, and concurrency.

## Near-term roadmap

1. Stable vLLM and SGLang evidence manifests.
2. NVIDIA GenAI-Perf evidence import.
3. GPU-memory and cloud-price evidence providers.
4. Search over runtime parameters and quantization candidates.
5. A separately versioned quality-evaluation protocol.

## Development

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

RuntimeFit is licensed under the MIT License.

Maintainers should follow [the release guide](https://github.com/kraftaa/runtimefit/blob/main/docs/releasing.md) for secure PyPI
Trusted Publishing and Homebrew tap updates.

## Real smoke result

`benchmarks/apple-m1-pro-llama2-7b-q4/` contains a small, clearly labelled end-to-end
smoke run against llama.cpp on Apple Silicon. It compares concurrency levels and two
Flash Attention settings. It proves that managed processes and streaming measurement
work; with one old model and a tiny unlabelled workload, it is not evidence of general
superiority or model quality.

## Real vLLM vs SGLang case study

The six-candidate, three-repetition protocol is checked in under
[`case-studies/vllm-vs-sglang/`](https://github.com/kraftaa/runtimefit/tree/main/case-studies/vllm-vs-sglang). It is clearly
marked as awaiting a Linux NVIDIA GPU, sanitized workload, and real measurements. The
repository will not present placeholder or synthetic metrics as that case study.
