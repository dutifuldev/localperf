# Gemma 4 vLLM Resource Sweep Implementation Plan

Date: 2026-06-20

## Goal

Measure how vLLM startup and request-load parameters affect memory use,
startup feasibility, throughput, and latency for
`nvidia/Gemma-4-26B-A4B-NVFP4` on this GB10 machine.

The required end state is:

- At least 100 parameter samples considered and recorded in machine-readable
  form.
- Executed samples must include idle server measurements and loaded request
  measurements where safe.
- The sweep must cover small contexts such as 4k and high contexts including
  about 100k tokens, and candidate concurrency up to 32.
- Results must include tables, plots, and fitted relationships or formulas for
  parameter-to-resource behavior.
- Work must stay in this scratch repo and be pushed frequently to the PR branch.
- The machine must not be OOMed.

## Model And Runtime

Model:

```text
nvidia/Gemma-4-26B-A4B-NVFP4
```

Local model metadata observed from the Hugging Face cache:

```text
text_config.max_position_embeddings = 262144
text_config.sliding_window = 1024
text_config.num_hidden_layers = 30
text_config.hidden_size = 2816
text_config.num_attention_heads = 16
text_config.num_key_value_heads = 8
text_config.head_dim = 256
text_config.global_head_dim = 512
```

Baseline working vLLM flags, inherited from the earlier LocalPager Gemma run:

```text
--trust-remote-code
--enable-prefix-caching
--reasoning-parser gemma4
--enable-auto-tool-choice
--tool-call-parser gemma4
--kv-cache-dtype fp8
--mm-processor-cache-gb 0
--moe-backend cutlass
--language-model-only
--no-enable-flashinfer-autotune
```

The sweep varies only the resource-shaping flags unless a controlled follow-up
explicitly says otherwise.

## Parameters

Primary sweep axes:

| parameter | values |
| --- | --- |
| `max_model_len` | 4096, 8192, 16384, 32768, 65536, 98304, 100000, 131072, 196608, 262144 |
| `max_num_seqs` | 1, 2, 4, 8, 16, 24, 32 |
| `max_num_batched_tokens` policy | `small`, `match_context`, `wide` |
| `gpu_memory_utilization` | 0.55, 0.65, 0.75 for selected calibration points |

The first matrix targets at least 140 candidate settings:

```text
10 context windows * 7 max_num_seqs * 2 batch policies = 140 candidates
```

Some candidates may be marked `skipped_risk` after a safe startup/capacity
check. A skipped candidate still records the attempted parameter tuple and
reason, but it does not count as an executed measurement.

## Measurement Phases Per Candidate

Each candidate produces one JSONL record with nested phase results.

1. `preflight`
   - Stop conflicting local LLM services.
   - Record system RAM, swap, process list, vLLM version, model snapshot.
   - Estimate risk from previous successful rows and the candidate's
     `context * concurrency` token budget.

2. `startup`
   - Start vLLM through a transient user systemd service with `MemoryMax`.
   - Wait for `/v1/models`.
   - Record startup wall time.
   - Parse vLLM logs for:
     - available KV cache memory,
     - GPU KV cache token capacity,
     - maximum reported concurrency for the configured context.
   - Record idle system memory and service cgroup memory.

3. `load_short_decode`
   - Send `max_num_seqs` concurrent short prompts, capped by the candidate load
     cap.
   - Use low output length, default 64 tokens, to keep samples fast.
   - Record successes, errors, wall time, aggregate output tokens/sec, total
     tokens/sec, latency distribution, and service memory peak.

4. `load_prefill_probe`
   - For a smaller selected subset, send long prompts at 25% and 75% of the
     context window, with output capped at 16 tokens.
   - This phase is skipped for high-risk candidates or when the estimated prompt
     would be too slow.

5. `shutdown`
   - Stop the transient service.
   - Record final cgroup status and whether cleanup completed.

## OOM Safety Rules

The harness must prefer `skipped_risk` over trying a dangerous sample.

Hard rules:

- Stop `localpager-vllm-qwen36-nvfp4.service`,
  `localpager-vllm-gemma4-26b-a4b-nvfp4.service`, and `localpager-worker.service`
  before every candidate unless running in `--dry-run`.
- Start each vLLM candidate as a transient user systemd service with an explicit
  `MemoryMax` cap.
- Kill the service if available system memory falls below the configured floor
  for more than one monitor tick.
- Never run load if startup logs report maximum concurrency below the requested
  load concurrency.
- Never run load if idle memory already violates the safety floor.
- High-context samples at 100k and above start with low concurrency first.
- Concurrency 24 and 32 are attempted only after nearby lower-concurrency
  samples succeed for the same context range.

Initial safety defaults:

```text
MemoryMax = 95 GiB
minimum available system memory while running = 12 GiB
minimum free swap while running = 4 GiB
startup timeout = 15 minutes
load timeout = 5 minutes
```

The user asked not to OOM the machine. If the safety guard conflicts with
collecting a data point, the data point is skipped and recorded.

## Machine-Readable Outputs

Required outputs:

```text
results/sweep-candidates.jsonl
results/sweep-results.jsonl
results/sweep-summary.json
results/events.jsonl
```

Each result row records:

- candidate id,
- parameters,
- command flags,
- service unit name,
- startup status,
- parsed vLLM capacity,
- idle memory,
- load memory,
- throughput,
- latency,
- skip/failure reason,
- log path.

## Analysis Outputs

Generated by the plotting/report script:

```text
reports/summary.md
reports/index.html
reports/plots/*.svg
reports/tables/*.csv
reports/models/*.json
```

Required analysis:

- Feasible/infeasible heatmaps for context by concurrency.
- Idle memory versus context for each concurrency level.
- Peak loaded memory versus context and concurrency.
- Aggregate output tok/s versus concurrency.
- Latency p50/p95 versus concurrency.
- Fitted linear or interaction models such as:

```text
idle_memory_gib ~= b0 + b1 * context_k + b2 * max_num_seqs
peak_memory_gib ~= b0 + b1 * context_k + b2 * max_num_seqs + b3 * context_k * max_num_seqs
output_tok_s ~= b0 + b1 * max_num_seqs + b2 * context_k
```

The formulas are descriptive, not universal hardware laws.

## Execution Strategy

1. Land this plan and push the PR branch.
2. Add the harness and dry-run matrix generation.
3. Run 3-5 calibration samples:
   - 4k / c1,
   - 16k / c4,
   - 32k / c8,
   - 32k / c16,
   - 100k / c1.
4. Push calibration results.
5. Expand in batches of 10-20 candidates, pushing after each batch.
6. Generate plots after each batch so partial results remain inspectable.
7. Continue until at least 100 candidates are recorded and enough executed
   samples exist to support the final report.

## Current Known Constraints

- `nvidia-smi` is present but reports `N/A` for GB10 memory fields on this
  machine, so GPU memory must be inferred from vLLM logs, cgroup/process memory,
  and system memory.
- `matplotlib` and `pandas` are not installed in the system Python, so plotting
  uses Python standard-library SVG/HTML generation unless dependencies are
  intentionally added later.
- Existing Qwen vLLM may be loaded before the sweep starts; the harness must
  unload it before Gemma samples.
