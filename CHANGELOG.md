# Changelog

## 0.2.0

- Carry run and sample counts through evidence provenance.
- Disclose insufficient evidence using configurable run, p95, and p99 thresholds.
- Apply hard constraints to the conservative end of repeated-run ranges.
- Use lowest observed throughput for capacity planning and tolerate floating-point
  noise at exact replica boundaries.
- Mark objective choices provisional when repeated-run ranges overlap.
- Restrict the Pareto frontier to eligible candidates with complete decision evidence.
- Share missing-data, ranking, and Pareto semantics between both execution paths.
- Require strict 2xx health checks and complete SSE streams.
- Report no TTFT when a stream contains no generated content.
- Enforce a total streaming deadline on best-effort supported HTTP transports.
- Expand result redaction across common secret-key spellings and command arguments.
- Add formatting, linting, and type checking to CI.

## 0.1.0

- Initial constraint-based deployment decision engine.
- RuntimeFit, inline, and GuideLLM v2 evidence providers.
- Capacity-aware monthly cost, repeated-evidence aggregation, decision reports, and
  vLLM/SGLang case-study protocol.
