# Gemma 4 vLLM Resource Sweep Report

This report is generated from `results/sweep-results.jsonl`. It covers a standalone vLLM sweep for `nvidia/Gemma-4-26B-A4B-NVFP4`; it does not use LocalPager or OpenClaw runtime paths.

Rows recorded: 100.
Context windows covered: 4096, 8192, 16384, 32768, 65536, 98304, 100000, 131072, 196608, 262144.
Highest requested concurrency covered: 32.

The load probe uses concurrent short prompts with 64 requested output tokens. High-risk rows are still started and measured at idle, but load is skipped when vLLM reports insufficient capacity or when the harness risk guard blocks it.

## Status Counts

| status | count |
| --- | ---: |
| load_complete | 52 |
| startup_only | 43 |
| startup_service_exit | 5 |

## Main Findings

- Best measured output throughput was `360.694` completion tok/s at `ctx8192-seq16-small` with `8192` context and `16` requested concurrency.
- Best measured total token throughput was `10533.390` total tok/s at `ctx8192-seq16-small`.
- The `small` batching policy (`max_num_batched_tokens` capped at 8192) is the stable high-context path in this sweep. It starts successfully at 100k, 131k, 196k, and 262k context, though high-risk load was intentionally skipped.
- The `match_context` batching policy often reduces reported concurrency at high context and hit a CUTLASS FP4 MoE kernel/config boundary at 196k and 262k contexts.
- No run was classified as a machine OOM. The startup exits were vLLM/kernel assertions, and the guard stopped load when capacity or risk was unsafe.

## Startup Boundaries

- 5 rows exited during startup.
- 5 of those hit the CUTLASS FP4 MoE MAX_TOKENS_PER_EXPERT assertion, which is a kernel/config boundary rather than an observed machine OOM: ctx196608-seq1-match_context, ctx262144-seq1-match_context, ctx196608-seq2-match_context, ctx262144-seq2-match_context, ctx196608-seq4-match_context.

## Simplified Capacity Model

The capacity view deliberately simplifies the parameter sweep. For capacity planning, context length is the main input. The batch policy is the main choice. The other capacity parameters can be derived from those two choices.

| symbol | meaning | simplified role |
| --- | --- | --- |
| $c$ | context window / `--max-model-len` | primary input |
| $p$ | batch policy | user chooses `small` or `match_context` |
| $b$ | `--max-num-batched-tokens` | derived from $c,p$ |
| $q$ | vLLM-reported max concurrency | measured/estimated capacity curve $Q_p(c)$ |
| $s$ | requested max sequences / `--max-num-seqs` | recommended from $q$ with a safety margin |

Policy definitions used in this sweep:

$$b_{small}(c)=\min(c,8192)$$

$$b_{match\_context}(c)=c$$

Recommended requested concurrency:

$$s_{recommended}(c,p,m)=\left\lfloor m\,Q_p(c)\right\rfloor$$

where $m$ is the safety margin. The interactive dashboard defaults to $m=0.8$.

For a fixed memory budget, the capacity question is not `what is memory as a smooth function of every knob?` The edge-case question is `what requested session count can survive for this context window?` In this report that frontier is:

$$s \le \left\lfloor m\,Q_p(c)\right\rfloor$$

The planner therefore answers high-context questions by reading the measured capacity frontier. For example, at a 150k context target, choose the context slider, choose the safety margin, and compare the `small` and `match_context` rows. The table gives the derived batch-token value and the recommended session count.

This simplification is valid for capacity planning: deciding whether a context window can fit and choosing a conservative `--max-num-seqs`. It is not sufficient for throughput or latency, because throughput and latency still depend on the actual requested concurrency, prompt/load shape, and whether the load phase was run.

The capacity curve in the dashboard is policy-specific and empirical. It uses the measured vLLM-reported concurrency points from this sweep and connects them continuously on a log-context scale. That avoids forcing a bad global linear model onto a clearly non-linear capacity curve.

## Abstract System

The sweep is a system with startup parameters as inputs and resource/performance measurements as outputs. The symbols below are single-letter quantities used by the formulas.

| symbol | quantity |
| --- | --- |
| $c$ | context window, measured in 1024-token units |
| $s$ | requested maximum concurrent sequences |
| $b$ | maximum batched tokens |
| $u$ | vLLM GPU memory utilization target |
| $p$ | batching policy indicator, where $p=0$ for `small` and $p=1$ for `match_context` |
| $v$ | requested token budget |
| $k$ | vLLM-reported KV cache token capacity |
| $q$ | vLLM-reported maximum concurrency for the configured context |
| $m$ | idle service memory |
| $h$ | idle service peak memory |
| $a$ | loaded service peak memory |
| $y$ | completion token throughput |
| $z$ | total token throughput |
| $l$ | p95 request latency |
| $w$ | free swap memory |
| $r$ | risk guard indicator, where $r=1$ means load is blocked by policy |
| $o$ | startup outcome |
| $d$ | load decision, where $d=1$ means load was run |

Derived budget:

$$v = cs$$

Startup and capacity:

$$o = O(c,s,b,u,p)$$

$$k = K(c,s,b,u,p)$$

$$q = Q(c,s,b,u,p)$$

Load is run only when the configured concurrency is inside reported capacity and safety bounds:

$$d = \mathbb{1}\{q \ge s \;\land\; m < m_{\max} \;\land\; w > w_{\min} \;\land\; r = 0\}$$

Memory model family:

$$m = M(c,s,b,u,p,cs) + \epsilon_m$$

$$h = H(c,s,b,u,p,cs) + \epsilon_h$$

$$a = A(c,s,b,u,p,cs) + \epsilon_a, \quad d=1$$

Performance model family:

$$y = Y(c,s,b,u,p,cs) + \epsilon_y, \quad d=1$$

$$z = Z(c,s,b,u,p,cs) + \epsilon_z, \quad d=1$$

$$l = L(c,s,b,u,p,cs) + \epsilon_l, \quad d=1$$

In this run, the fitted instances of $M,H,A,Q,Y,L$ are simple continuous response models over the observed rows. Most are linear interaction models; the interactive capacity surface uses a policy-specific piecewise log-log empirical model for $Q$ so it follows the measured capacity curve. The abstract system is the important part; the numeric coefficients are only run-specific estimates.

## Fit Quality

The coefficients for these model families are stored in `models/linear-models.json`. They are intentionally not printed in the report formulas; the report formulas define the system, and this table only says how well each family fit the observed rows.

| target | rows | R2 | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| idle_memory_gib | 95 | 0.437 | 0.193 | 0.265 |
| idle_memory_peak_gib | 95 | 0.409 | 0.201 | 0.259 |
| load_memory_peak_gib | 52 | 0.225 | 0.139 | 0.218 |
| reported_concurrency | 95 | 0.522 | 23.164 | 28.921 |
| reported_concurrency_log | 95 | 0.272 | 23.180 | 35.701 |
| reported_concurrency_empirical | 95 | 1.000 | 0.087 | 0.194 |
| completion_tok_s | 52 | 0.911 | 17.494 | 27.832 |
| latency_p95 | 52 | 0.099 | 0.423 | 0.535 |

## Context Summary

| context | rows | load complete | startup only | startup exits | max loaded seqs | best completion tok/s | max reported concurrency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 14 | 10 | 4 | 0 | 16 | 358.957 | 137.570 |
| 8192 | 14 | 10 | 4 | 0 | 16 | 360.694 | 67.770 |
| 16384 | 14 | 10 | 4 | 0 | 16 | 288.419 | 56.220 |
| 32768 | 14 | 9 | 5 | 0 | 16 | 216.743 | 48.710 |
| 65536 | 10 | 7 | 3 | 0 | 8 | 151.082 | 38.610 |
| 98304 | 8 | 6 | 2 | 0 | 8 | 141.812 | 31.950 |
| 100000 | 8 | 0 | 8 | 0 | 0 |  | 31.690 |
| 131072 | 8 | 0 | 8 | 0 | 0 |  | 27.260 |
| 196608 | 6 | 0 | 3 | 3 | 0 |  | 21.100 |
| 262144 | 4 | 0 | 2 | 2 | 0 |  | 17.180 |

## Top Output Throughput Rows

| rank | candidate | context | seqs | policy | completion tok/s | total tok/s | p95 latency | idle GiB | load peak GiB |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | ctx8192-seq16-small | 8192 | 16 | small | 360.694 | 10533.390 | 2.835 | 3.739 | 10.837 |
| 2 | ctx4096-seq16-small | 4096 | 16 | small | 358.957 | 10482.670 | 2.848 | 3.695 | 10.789 |
| 3 | ctx8192-seq16-match_context | 8192 | 16 | match_context | 354.195 | 10343.592 | 2.887 | 3.817 | 10.821 |
| 4 | ctx16384-seq16-match_context | 16384 | 16 | match_context | 288.419 | 8422.739 | 3.547 | 3.800 | 10.884 |
| 5 | ctx16384-seq16-small | 16384 | 16 | small | 218.784 | 6389.167 | 4.678 | 3.952 | 10.837 |
| 6 | ctx4096-seq16-match_context | 4096 | 16 | match_context | 217.576 | 6353.886 | 4.703 | 3.905 | 10.790 |
| 7 | ctx32768-seq16-small | 32768 | 16 | small | 216.743 | 6329.567 | 4.721 | 3.975 | 10.845 |
| 8 | ctx4096-seq8-small | 4096 | 8 | small | 196.096 | 5726.602 | 2.610 | 3.672 | 10.790 |
| 9 | ctx8192-seq8-match_context | 8192 | 8 | match_context | 194.430 | 5677.964 | 2.632 | 3.755 | 10.840 |
| 10 | ctx8192-seq8-small | 8192 | 8 | small | 193.529 | 5651.643 | 2.644 | 3.645 | 10.821 |

## 100k Context Rows

| candidate | status | seqs | policy | reported concurrency | idle GiB | completion tok/s | notes |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| ctx100000-seq1-match_context | startup_only | 1 | match_context | 3.300 | 5.081 |  | load skipped by capacity or risk guard |
| ctx100000-seq1-small | startup_only | 1 | small | 31.690 | 3.982 |  | load skipped by capacity or risk guard |
| ctx100000-seq2-match_context | startup_only | 2 | match_context | 3.280 | 4.812 |  | load skipped by capacity or risk guard |
| ctx100000-seq2-small | startup_only | 2 | small | 31.450 | 4.018 |  | load skipped by capacity or risk guard |
| ctx100000-seq4-match_context | startup_only | 4 | match_context | 3.200 | 4.894 |  | load skipped by capacity or risk guard |
| ctx100000-seq4-small | startup_only | 4 | small | 31.630 | 4.054 |  | load skipped by capacity or risk guard |
| ctx100000-seq8-match_context | startup_only | 8 | match_context | 3.310 | 4.710 |  | load skipped by capacity or risk guard |
| ctx100000-seq8-small | startup_only | 8 | small | 31.640 | 3.933 |  | load skipped by capacity or risk guard |

## Latest Measurements

| candidate | status | context | seqs | idle GiB | tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| ctx65536-seq8-match_context | startup_only | 65536 | 8 | 4.251 |  |
| ctx65536-seq8-small | load_complete | 65536 | 8 | 3.912 | 151.082 |
| ctx131072-seq4-match_context | startup_only | 131072 | 4 | 4.915 |  |
| ctx131072-seq4-small | startup_only | 131072 | 4 | 4.070 |  |
| ctx262144-seq2-match_context | startup_service_exit | 262144 | 2 |  |  |
| ctx262144-seq2-small | startup_only | 262144 | 2 | 4.046 |  |
| ctx32768-seq24-match_context | startup_only | 32768 | 24 | 4.089 |  |
| ctx32768-seq24-small | startup_only | 32768 | 24 | 3.991 |  |
| ctx98304-seq8-match_context | startup_only | 98304 | 8 | 4.828 |  |
| ctx98304-seq8-small | load_complete | 98304 | 8 | 4.078 | 141.812 |
| ctx196608-seq4-match_context | startup_service_exit | 196608 | 4 |  |  |
| ctx196608-seq4-small | startup_only | 196608 | 4 | 4.075 |  |
| ctx100000-seq8-match_context | startup_only | 100000 | 8 | 4.710 |  |
| ctx100000-seq8-small | startup_only | 100000 | 8 | 3.933 |  |
| ctx32768-seq32-match_context | startup_only | 32768 | 32 | 4.124 |  |
| ctx32768-seq32-small | startup_only | 32768 | 32 | 4.020 |  |
| ctx65536-seq16-match_context | startup_only | 65536 | 16 | 4.287 |  |
| ctx65536-seq16-small | startup_only | 65536 | 16 | 3.962 |  |
| ctx131072-seq8-match_context | startup_only | 131072 | 8 | 4.758 |  |
| ctx131072-seq8-small | startup_only | 131072 | 8 | 3.867 |  |

## Runtime And Safety

- vLLM was started directly as transient user systemd services from the standalone harness.
- Model: `nvidia/Gemma-4-26B-A4B-NVFP4`.
- Fixed vLLM flags included `--gpu-memory-utilization 0.65`, `--kv-cache-dtype fp8`, `--moe-backend cutlass`, `--language-model-only`, and `--no-enable-flashinfer-autotune`.
- Each candidate used `MemoryMax=95G`, `min_available_gib=12`, and `min_swap_free_gib=4` guardrails.
- Conflicting local LLM services were stopped before each candidate to isolate the measurement; the harness itself is independent from LocalPager.
- `nvidia-smi` memory fields are not useful on this GB10 setup, so the report uses vLLM logs, cgroup memory, and system memory snapshots.

## Limitations

- The load phase is a short-decode probe, not a full long-prefill benchmark.
- High-risk rows are intentionally startup/capacity measurements unless explicitly allowed for risky load.
- The fitted formulas summarize this run's measured relationships and can be distorted by guardrails, cold-start compile behavior, and skipped load rows.
- cgroup memory is the best available process-level memory signal here; it is not a direct GPU-memory counter.

## Artifacts

- `results/sweep-results.jsonl`: machine-readable per-candidate measurements.
- `results/sweep-summary.json`: machine-readable aggregate summary.
- `tables/measurements.csv`: flattened table for spreadsheets.
- `plots/idle-memory-by-context.svg`
- `plots/load-peak-memory-by-context.svg`
- `plots/reported-concurrency-by-context.svg`
- `plots/throughput-by-concurrency.svg`
- `plots/latency-p95-by-concurrency.svg`
- `models/linear-models.json`: fitted coefficients and error metrics.
