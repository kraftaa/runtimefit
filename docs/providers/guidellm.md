# GuideLLM evidence provider

GuideLLM is a dedicated inference benchmarking platform. RuntimeFit consumes its
authoritative `benchmarks.json` output and makes a deployment decision; it does not
reimplement GuideLLM's traffic generation or statistical engine.

Configure one RuntimeFit candidate for each GuideLLM strategy you want to consider:

```toml
[[candidates]]
id = "sglang-fp16-c8"
runtime = "sglang"
quantization = "fp16"
concurrency = 8
cost_per_hour_usd = 1.96

[candidates.evidence]
provider = "guidellm"
path = "results/sglang-fp16/benchmarks.json"
benchmark_index = 2
```

`benchmark_index` selects an entry in GuideLLM's `benchmarks` array. RuntimeFit records
the file SHA-256, GuideLLM version, report-schema version, benchmark ID, run index, and
scheduling strategy alongside the normalized metrics.

## Imported metrics

- TTFT p50/p95/p99
- TPOT p50/p95/p99
- end-to-end latency p50/p95/p99
- request throughput
- output-token throughput
- request error rate

RuntimeFit currently supports GuideLLM report schema version 2 and fails explicitly on
other schema versions. GuideLLM records request latency in seconds; RuntimeFit converts
it to milliseconds during normalization. Other imported latency metrics are already in
milliseconds.

GPU memory and cost do not come from this importer. Supply hourly or monthly cost on
the candidate, and provide GPU-memory evidence separately when that requirement is
enabled.

