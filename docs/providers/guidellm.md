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

For a serious decision, provide repeated runs. RuntimeFit takes the median of metrics
present in every run, preserves each file hash, and reports the relative range:

```toml
[candidates.evidence]
provider = "guidellm"
paths = [
  "results/sglang-c8/run-1.json",
  "results/sglang-c8/run-2.json",
  "results/sglang-c8/run-3.json",
]
benchmark_index = 0
aggregation = "median"
```

Metrics whose maximum-to-minimum range exceeds 15% of the median are surfaced as
evidence-stability warnings. Constraints use the conservative end of the observed
range rather than allowing the median to hide a failed run. RuntimeFit also records the
minimum per-run sample count for each imported metric.

The default disclosure policy expects three runs, at least 100 samples per run for a
p95 statistic, and at least 1,000 for p99. It can be made explicit in the decision file:

```toml
[evidence_policy]
minimum_runs = 3
minimum_samples_p95 = 100
minimum_samples_p99 = 1000
```

These thresholds are evidence-quality warnings, not confidence intervals. If observed
objective ranges overlap, the recommendation is labelled provisional.

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

GPU memory and cost do not come from this importer. Supply per-replica hourly cost or
a total pre-sized monthly cost on the candidate, and provide GPU-memory evidence
separately when that requirement is enabled.
