# Real vLLM vs SGLang decision case study

Status: **protocol ready; measurements not yet run**.

This directory is intentionally missing benchmark results. RuntimeFit's maintainers do
not currently have a Linux NVIDIA GPU in the development environment, and synthetic or
invented measurements will not be committed as a real case study.

## Question

Given the same model, GPU, workload, generation settings, and hourly infrastructure
price, which vLLM or SGLang concurrency configuration should serve a workload requiring
8 requests per second while meeting these requirements?

- p95 TTFT below 400 ms;
- p99 end-to-end latency below 3 seconds;
- error rate below 0.5%;
- measured per-replica throughput of at least 7 requests per second;
- at least 10% deployed throughput headroom; and
- monthly cost below $2,000.

The candidates are vLLM and SGLang at concurrency 4, 8, and 16. Each candidate is run
three times. RuntimeFit selects using the median measurement and warns when the range
across repetitions exceeds 15% of the median.

## Required machine

- Linux with one dedicated NVIDIA GPU;
- enough VRAM for `Qwen/Qwen2.5-7B-Instruct` in FP16;
- Python 3.11 or 3.12;
- vLLM, SGLang, GuideLLM 0.7.2, and RuntimeFit installed; and
- no other GPU workload during measurement.

The commands follow the official [GuideLLM output interface](https://github.com/vllm-project/guidellm/blob/main/docs/guides/outputs.md),
[vLLM serving interface](https://docs.vllm.ai/en/stable/getting_started/quickstart/),
and [SGLang server interface](https://docs.sglang.ai/developer_guide/bench_serving).

Pin and record the exact vLLM, SGLang, CUDA, driver, PyTorch, model-revision, and
container or Python-package versions used. Do not compare runs from different hardware.

## Workload

Place a sanitized production-style JSONL workload at `workload.jsonl`. Each line must
contain a `prompt` field. Use enough unique requests to avoid prompt-cache reuse
dominating the result; do not commit private prompts.

```json
{"prompt":"A real, sanitized request from the target application..."}
```

Hash the final workload and keep it unchanged for both runtimes:

```bash
sha256sum workload.jsonl
```

## Run protocol

Run the following commands from this directory.

1. Start vLLM on port 8000 with the pinned model and FP16 dtype:

   ```bash
   vllm serve Qwen/Qwen2.5-7B-Instruct \
     --host 127.0.0.1 --port 8000 \
     --dtype float16 --generation-config vllm
   ```

2. In another shell, benchmark it:

   ```bash
   ./benchmark-endpoint.sh vllm http://127.0.0.1:8000 \
     Qwen/Qwen2.5-7B-Instruct workload.jsonl
   ```

3. Stop vLLM, allow the GPU to return to idle, then start SGLang:

   ```bash
   python -m sglang.launch_server \
     --model-path Qwen/Qwen2.5-7B-Instruct \
     --host 127.0.0.1 --port 8000 --dtype float16
   ```

4. Run the same benchmark against SGLang:

   ```bash
   ./benchmark-endpoint.sh sglang http://127.0.0.1:8000 \
     Qwen/Qwen2.5-7B-Instruct workload.jsonl
   ```

5. Put the actual on-demand hourly price of the GPU host in each candidate's
   `cost_per_hour_usd` field in `runtimefit.toml`.

6. Generate the decision:

   ```bash
   runtimefit validate runtimefit.toml
   runtimefit choose runtimefit.toml \
     --output results/decision.json \
     --report results/decision.md
   ```

7. Rerun the full protocol with runtime order reversed before publication. If the
   selection changes, investigate rather than choosing the more attractive result.

## Publication checklist

Commit the sanitized workload or its provenance and hash, all 18 GuideLLM JSON files,
the RuntimeFit decision JSON and report, `environment.txt`, server logs, exact launch
commands, GPU hourly price source and date, and a short limitations section. Report
failures and losing candidates. The desired story is not predetermined: if the fastest
candidate is also selected, publish that result rather than tuning SLOs after the fact.
