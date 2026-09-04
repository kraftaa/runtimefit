# RuntimeFit report: apple-m1-pro-llama2-7b-q4-smoke

**Recommendation:** `llama.cpp-8640-metal-fa-auto@c2`

Objective: `throughput`  
Created: 2026-09-04T18:49:58.807471+00:00

| Target | C | Quality | Error rate | p50 latency | p95 latency | p95 TTFT | Output tok/s | $/1M out | Startup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama.cpp-8640-metal-fa-auto@c1 | 1 | — | 0.0% | 1923.96 ms | 2088.87 ms | 249.00 ms | 32.10 | — | 4.10 s |
| llama.cpp-8640-metal-fa-auto@c2 | 2 | — | 0.0% | 2852.11 ms | 3325.56 ms | 242.04 ms | 38.32 | — | 4.10 s |
| llama.cpp-8640-metal-fa-auto@c4 | 4 | — | 0.0% | 3254.06 ms | 4277.23 ms | 355.75 ms | 56.43 | — | 4.10 s |
| llama.cpp-8640-metal-fa-off@c1 | 1 | — | 0.0% | 1947.74 ms | 2193.42 ms | 263.81 ms | 31.04 | — | 1.27 s |
| llama.cpp-8640-metal-fa-off@c2 | 2 | — | 0.0% | 3207.94 ms | 3864.35 ms | 373.93 ms | 33.84 | — | 1.27 s |
| llama.cpp-8640-metal-fa-off@c4 | 4 | — | 0.0% | 4012.28 ms | 5452.01 ms | 1074.60 ms | 47.59 | — | 1.27 s |

## Decision details

Pareto frontier: `llama.cpp-8640-metal-fa-auto@c1`, `llama.cpp-8640-metal-fa-auto@c2`, `llama.cpp-8640-metal-fa-auto@c4`

- `llama.cpp-8640-metal-fa-auto@c4`: p95 latency exceeds configured limit
- `llama.cpp-8640-metal-fa-off@c4`: p95 latency exceeds configured limit
- `llama.cpp-8640-metal-fa-auto@c1`: eligible but not optimal for the configured objective
- `llama.cpp-8640-metal-fa-auto@c2`: best eligible target for objective 'throughput'
- `llama.cpp-8640-metal-fa-off@c1`: eligible but not optimal for the configured objective
- `llama.cpp-8640-metal-fa-off@c2`: eligible but not optimal for the configured objective

## Reproducibility metadata

- Definition SHA-256: `414dc5b70d35c29fe876481510023ad65e257bf842623d67559b607831dfbfd7`
- RuntimeFit: `0.1.0`
- Python: `3.14.7`
- Platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`
- Machine: `arm64`
- Processor: `arm`
- Logical CPUs: `10`
- Physical memory: `34359738368` bytes
