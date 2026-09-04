# Decision methodology

RuntimeFit is a decision layer over benchmark evidence. It does not claim that one
inference engine is universally fastest, and its primary `choose` workflow does not
run a load test. It answers a narrower question: which declared candidate best meets
this workload's requirements, according to a stated objective?

The v0.1 decision contract intentionally accepts only vLLM and SGLang candidates.

## Decision pipeline

1. Load each candidate's evidence from a supported provider.
2. Normalize provider-specific measurements into canonical units and names.
3. Calculate monthly cost from an explicit monthly value or hourly price assumption.
4. Evaluate every declared requirement independently.
5. Reject a candidate when it violates a limit or lacks a required measurement.
6. Rank the remaining candidates by the one declared objective.
7. Report the selection, every check and rejection, alternative comparisons, source
   hashes, and the Pareto frontier.

RuntimeFit never converts latency, throughput, reliability, and cost into a hidden
weighted score. Requirements are hard filters; the objective ranks feasible choices.

## Canonical evidence

The decision contract currently understands:

- TTFT p50/p95/p99 in milliseconds;
- time per output token p50/p95/p99 in milliseconds;
- end-to-end latency p50/p95/p99 in milliseconds;
- request throughput in requests per second;
- output-token throughput in tokens per second;
- request error rate as a fraction;
- peak GPU memory in GiB; and
- estimated monthly cost in US dollars.

Provider adapters preserve provenance. File-backed evidence records the source path
and SHA-256 digest, along with provider metadata such as the GuideLLM schema version
and benchmark index. Inline evidence is useful for demonstrations and manual imports,
but it is not independently verifiable benchmark data.

The configuration fingerprint identifies the normalized decision configuration. The
decision fingerprint covers that configuration fingerprint plus every imported source
identity and inline metric set, so changing a benchmark file changes the final decision
identity.

## Eligibility and missing data

A candidate is eligible only when every configured requirement passes. A missing
measurement is a failed requirement, not an implicit pass. If no candidate is both
eligible and has the objective metric, RuntimeFit returns no recommendation and exits
with a non-zero status.

Monthly cost is either supplied directly or calculated as:

```text
cost_per_hour_usd × hours_per_month
```

The default is 730 hours per month. This is an explicit capacity assumption, not a
cloud bill prediction; networking, storage, autoscaling, idle policy, and discounts
remain outside the current model.

## Pareto frontier

The frontier uses only decision metrics present for every candidate, so a sparse
record cannot dominate a fully measured record merely because inconvenient metrics
are absent. Lower latency, error rate, memory, and cost are better; higher throughput
is better. The frontier includes feasible and infeasible candidates because a rejected
candidate can still reveal a real tradeoff. Eligibility remains visible beside it.

## Why quality is not in v0.1

Quality needs its own controlled protocol. Quantization, sampling parameters, model
revision, output nondeterminism, task data, and evaluator choice can all change the
result. RuntimeFit v0.1 therefore does not accept a quality requirement in the
decision contract. A future quality provider must expose the evaluator, dataset
revision, sampling configuration, repetitions, and uncertainty before quality can
participate in deployment selection.

The small built-in runner still has simple exact/contains checks for harness testing.
Those checks are not imported into the `choose` decision and must not be presented as
a general model-quality evaluation.

## Requirements for a public decision

Publish the decision config, generated decision JSON, all source evidence files when
licensing permits, runtime and model revisions, launch parameters, workload identity,
hardware and driver details, warm-up and caching policy, and cost assumptions. Include
losing and failed candidates. Use multiple measured repetitions for performance claims;
a tiny run on one machine is a smoke test, not evidence of universal superiority.

## Known limitations

- Source benchmark methodology still determines whether imported evidence is valid.
- RuntimeFit cannot yet prove that every imported file used the declared workload;
  reviewers must inspect the source benchmark metadata.
- Prompt-cache reuse can materially distort repeated workloads.
- Small samples make tail percentiles unstable.
- Output-length differences can distort latency comparisons.
- Cloud prices and available instance types change independently of saved evidence.
- A recommendation only applies to the candidates and constraints that were declared.
