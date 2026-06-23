# Gemma 4 vLLM Resource Sweep Report

This report is generated from `results/tegrastats-sweep-20260623T075153Z-results.jsonl`. It covers a standalone vLLM sweep for `nvidia/Gemma-4-26B-A4B-NVFP4`; it does not use LocalPager or OpenClaw runtime paths.

Rows recorded: 140.
Context windows covered: 4096, 8192, 16384, 32768, 65536, 98304, 100000, 131072, 196608, 262144.
Highest requested concurrency covered: 32.

The load probe uses concurrent short prompts with 64 requested output tokens. High-risk rows are still started and measured at idle, but load is skipped when vLLM reports insufficient capacity or when the harness risk guard blocks it.

## Status Counts

| status | count |
| --- | ---: |
| load_complete | 52 |
| startup_only | 74 |
| startup_service_exit | 14 |

## Main Findings

- Best measured output throughput was `361.593` completion tok/s at `ctx8192-seq16-small` with `8192` context and `16` requested concurrency.
- Best measured total token throughput was `10559.650` total tok/s at `ctx8192-seq16-small`.
- Highest measured total RAM pressure by `tegrastats` was `82.372` GiB at `ctx4096-seq8-small`.
- Memory reporting now separates total machine pressure (`tegrastats` RAM delta and system `MemAvailable` drop), process accounting (cgroup), and vLLM capacity.
- The `small` batching policy (`max_num_batched_tokens` capped at 8192) is the stable high-context path in this sweep. It starts successfully at 100k and above when capacity guards allow startup.
- The `match_context` batching policy can reduce reported concurrency at high context and may hit CUTLASS FP4 MoE kernel/config boundaries.
- Startup exits were vLLM/kernel exits rather than observed machine OOMs; load guards still block unsafe rows.

## Startup Boundaries

- 14 rows exited during startup.
- 14 of those hit the CUTLASS FP4 MoE MAX_TOKENS_PER_EXPERT assertion, which is a kernel/config boundary rather than an observed machine OOM: ctx196608-seq1-match_context, ctx262144-seq1-match_context, ctx196608-seq2-match_context, ctx262144-seq2-match_context, ctx196608-seq4-match_context, ctx262144-seq4-match_context, ctx196608-seq8-match_context, ctx262144-seq8-match_context, ctx196608-seq16-match_context, ctx262144-seq16-match_context, ctx196608-seq24-match_context, ctx196608-seq32-match_context, ctx262144-seq24-match_context, ctx262144-seq32-match_context.

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
| idle_memory_gib | 126 | 0.204 | 0.270 | 0.345 |
| idle_memory_peak_gib | 126 | 0.256 | 0.261 | 0.334 |
| system_memory_drop_gib | 140 | 0.569 | 5.154 | 6.688 |
| tegrastats_ram_delta_gib | 140 | 0.634 | 4.723 | 5.690 |
| tegrastats_max_temp_c | 140 | 0.111 | 2.579 | 3.127 |
| load_memory_peak_gib | 52 | 0.103 | 0.225 | 0.315 |
| reported_concurrency | 126 | 0.443 | 22.433 | 29.286 |
| reported_concurrency_log | 126 | 0.018 | 22.422 | 38.896 |
| reported_concurrency_empirical | 126 | 1.000 | 0.100 | 0.251 |
| completion_tok_s | 52 | 0.917 | 14.864 | 23.239 |
| latency_p95 | 52 | 0.312 | 0.386 | 0.473 |

## Context Summary

| context | rows | load complete | startup only | startup exits | max loaded seqs | best completion tok/s | max reported concurrency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 14 | 10 | 4 | 0 | 16 | 283.262 | 137.810 |
| 8192 | 14 | 10 | 4 | 0 | 16 | 361.593 | 67.730 |
| 16384 | 14 | 10 | 4 | 0 | 16 | 228.173 | 56.250 |
| 32768 | 14 | 9 | 5 | 0 | 16 | 222.045 | 48.870 |
| 65536 | 14 | 7 | 7 | 0 | 8 | 147.757 | 38.650 |
| 98304 | 14 | 6 | 8 | 0 | 8 | 147.571 | 31.970 |
| 100000 | 14 | 0 | 14 | 0 | 0 |  | 31.740 |
| 131072 | 14 | 0 | 14 | 0 | 0 |  | 27.300 |
| 196608 | 14 | 0 | 7 | 7 | 0 |  | 21.120 |
| 262144 | 14 | 0 | 7 | 7 | 0 |  | 17.300 |

## Top Output Throughput Rows

| rank | candidate | context | seqs | policy | completion tok/s | total tok/s | p95 latency | idle GiB | load peak GiB |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | ctx8192-seq16-small | 8192 | 16 | small | 361.593 | 10559.650 | 2.830 | 3.723 | 10.821 |
| 2 | ctx4096-seq16-small | 4096 | 16 | small | 283.262 | 8272.122 | 3.613 | 4.019 | 11.005 |
| 3 | ctx16384-seq16-small | 16384 | 16 | small | 228.173 | 6663.368 | 4.486 | 3.842 | 10.820 |
| 4 | ctx8192-seq16-match_context | 8192 | 16 | match_context | 227.390 | 6640.503 | 4.501 | 4.015 | 10.882 |
| 5 | ctx16384-seq16-match_context | 16384 | 16 | match_context | 224.513 | 6556.476 | 4.558 | 3.842 | 10.900 |
| 6 | ctx32768-seq16-small | 32768 | 16 | small | 222.045 | 6484.400 | 4.608 | 3.845 | 10.822 |
| 7 | ctx4096-seq16-match_context | 4096 | 16 | match_context | 221.812 | 6477.595 | 4.613 | 4.144 | 10.991 |
| 8 | ctx8192-seq8-small | 8192 | 8 | small | 195.998 | 5723.746 | 2.609 | 3.700 | 10.846 |
| 9 | ctx4096-seq8-small | 4096 | 8 | small | 195.364 | 5705.240 | 2.618 | 4.069 | 10.899 |
| 10 | ctx32768-seq8-match_context | 32768 | 8 | match_context | 159.781 | 4666.109 | 3.203 | 3.872 | 11.010 |

## 100k Context Rows

| candidate | status | seqs | policy | reported concurrency | idle GiB | completion tok/s | notes |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| ctx100000-seq1-match_context | startup_only | 1 | match_context | 3.260 | 4.845 |  | risk_guard(high) |
| ctx100000-seq1-small | startup_only | 1 | small | 31.740 | 3.906 |  | risk_guard(high) |
| ctx100000-seq2-match_context | startup_only | 2 | match_context | 3.280 | 4.743 |  | risk_guard(high) |
| ctx100000-seq2-small | startup_only | 2 | small | 31.550 | 3.919 |  | risk_guard(high) |
| ctx100000-seq4-match_context | startup_only | 4 | match_context | 3.280 | 4.816 |  | risk_guard(high)+capacity_guard |
| ctx100000-seq4-small | startup_only | 4 | small | 31.470 | 3.974 |  | risk_guard(high) |
| ctx100000-seq8-match_context | startup_only | 8 | match_context | 3.280 | 4.631 |  | risk_guard(high)+capacity_guard |
| ctx100000-seq8-small | startup_only | 8 | small | 31.690 | 3.838 |  | risk_guard(high) |
| ctx100000-seq16-match_context | startup_only | 16 | match_context | 3.300 | 4.789 |  | risk_guard(extreme)+capacity_guard |
| ctx100000-seq16-small | startup_only | 16 | small | 31.640 | 3.949 |  | risk_guard(extreme) |
| ctx100000-seq24-match_context | startup_only | 24 | match_context | 3.270 | 4.812 |  | risk_guard(extreme)+capacity_guard |
| ctx100000-seq24-small | startup_only | 24 | small | 31.550 | 3.953 |  | risk_guard(extreme) |
| ctx100000-seq32-match_context | startup_only | 32 | match_context | 3.300 | 4.864 |  | risk_guard(extreme)+capacity_guard |
| ctx100000-seq32-small | startup_only | 32 | small | 31.440 | 4.048 |  | risk_guard(extreme)+capacity_guard |

## Latest Measurements

| candidate | status | context | seqs | idle GiB | tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| ctx98304-seq32-match_context | startup_only | 98304 | 32 | 4.856 |  |
| ctx98304-seq32-small | startup_only | 98304 | 32 | 4.989 |  |
| ctx131072-seq24-match_context | startup_only | 131072 | 24 | 4.809 |  |
| ctx131072-seq24-small | startup_only | 131072 | 24 | 4.002 |  |
| ctx196608-seq16-match_context | startup_service_exit | 196608 | 16 |  |  |
| ctx196608-seq16-small | startup_only | 196608 | 16 | 3.986 |  |
| ctx100000-seq32-match_context | startup_only | 100000 | 32 | 4.864 |  |
| ctx100000-seq32-small | startup_only | 100000 | 32 | 4.048 |  |
| ctx131072-seq32-match_context | startup_only | 131072 | 32 | 4.856 |  |
| ctx131072-seq32-small | startup_only | 131072 | 32 | 4.061 |  |
| ctx262144-seq16-match_context | startup_service_exit | 262144 | 16 |  |  |
| ctx262144-seq16-small | startup_only | 262144 | 16 | 4.091 |  |
| ctx196608-seq24-match_context | startup_service_exit | 196608 | 24 |  |  |
| ctx196608-seq24-small | startup_only | 196608 | 24 | 4.024 |  |
| ctx196608-seq32-match_context | startup_service_exit | 196608 | 32 |  |  |
| ctx196608-seq32-small | startup_only | 196608 | 32 | 4.005 |  |
| ctx262144-seq24-match_context | startup_service_exit | 262144 | 24 |  |  |
| ctx262144-seq24-small | startup_only | 262144 | 24 | 4.182 |  |
| ctx262144-seq32-match_context | startup_service_exit | 262144 | 32 |  |  |
| ctx262144-seq32-small | startup_only | 262144 | 32 | 4.051 |  |

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

- Input JSONL: `results/tegrastats-sweep-20260623T075153Z-results.jsonl`.
- `tables/measurements.csv`: flattened table for spreadsheets.
- `plots/idle-memory-by-context.svg`
- `plots/load-peak-memory-by-context.svg`
- `plots/system-memory-drop-by-context.svg`
- `plots/tegrastats-ram-delta-by-context.svg`
- `plots/tegrastats-temperature-by-context.svg`
- `plots/reported-concurrency-by-context.svg`
- `plots/throughput-by-concurrency.svg`
- `plots/latency-p95-by-concurrency.svg`
- `models/linear-models.json`: fitted coefficients and error metrics.
