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
- Work must stay in this localperf repo and be pushed frequently to the PR branch.
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
| `max_num_batched_tokens` policy | `small`, `match_context` |
| `gpu_memory_utilization` | fixed at 0.65 for this sweep |

The first matrix targets at least 140 candidate settings:

```text
10 context windows * 7 max_num_seqs * 2 batch policies = 140 candidates
```

Some candidates may be recorded as startup-only after a safe startup/capacity
check. A startup-only candidate still records the attempted parameter tuple,
vLLM capacity, idle memory, telemetry, and reason; it does not run request load.

The runner orders candidates by increasing estimated risk, not by the table
order above. Progress should therefore be judged by recorded coverage and the
completion verifier, not by expecting a monotonic walk through every context at
one concurrency before moving to the next.

## Measurement Phases Per Candidate

Each candidate produces one JSONL record with nested phase results.

1. `preflight`
   - Stop conflicting local LLM services.
   - Start a per-candidate `tegrastats` sampler unless explicitly disabled.
   - Record system RAM, swap, conflicting service state, vLLM version, and
     model snapshot.
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
   - Record `/proc/meminfo`, service cgroup memory, `nvidia-smi` telemetry
     where available, and `tegrastats` samples.

3. `load_short_decode`
   - Send `max_num_seqs` concurrent short prompts, capped by the candidate load
     cap.
   - Use low output length, default 64 tokens, to keep samples fast.
   - Record successes, errors, wall time, aggregate output tokens/sec, total
     tokens/sec, latency distribution, service memory peak, and load-phase
     telemetry.
   - If startup succeeds but the capacity or memory guard rejects request load,
     record the row as startup-only with the guard reason. This is planned safe
     behavior, not a failed sweep shortcut.

4. `load_prefill_probe`
   - Follow-up only. It is not part of the current 140-candidate sweep.
   - For a smaller selected subset, a later run can send long prompts at 25%
     and 75% of the context window, with output capped at 16 tokens.

5. `shutdown`
   - Stop the transient service.
   - Stop `tegrastats`.
   - Record final cgroup status, telemetry summary, and whether cleanup
     completed.

## OOM Safety Rules

The harness must prefer startup-only or skipped-load records over trying a
dangerous load sample.

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
results/<run-id>-results.jsonl
results/<run-id>-summary.json
results/<run-id>-events.jsonl
```

The timestamped raw JSONL files are local-only because they can include machine
paths, process details, and verbose logs. Final reports and sanitized summaries
can be committed.

Each result row records:

- candidate id,
- parameters,
- command flags,
- service unit name,
- startup status,
- parsed vLLM capacity,
- `tegrastats` total RAM/swap/temperature samples and summary,
- `/proc/meminfo` snapshots,
- systemd cgroup memory,
- `nvidia-smi` fields where available,
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
- Total machine RAM pressure versus context and concurrency using `tegrastats`
  and `MemAvailable` drop.
- Cgroup idle/load memory versus context for process-accounting diagnostics.
- Aggregate output tok/s versus concurrency.
- Latency p50/p95 versus concurrency.
- Fitted linear or interaction models such as:

```text
tegrastats_ram_delta_gib ~= b0 + b1 * context_k + b2 * max_num_seqs + b3 * context_k * max_num_seqs
system_memory_drop_gib ~= b0 + b1 * context_k + b2 * max_num_seqs + b3 * context_k * max_num_seqs
output_tok_s ~= b0 + b1 * max_num_seqs + b2 * context_k
```

The formulas are descriptive, not universal hardware laws.

Committed report artifacts must be regenerated from the active run before the
final report is treated as current. Older HTML, CSV, SVG, or model JSON files in
`reports/` are placeholders or previous partial outputs until they explicitly
reference the active run id.

## Execution Strategy

1. Land this plan and push the PR branch. Done.
2. Add the harness and dry-run matrix generation. Done.
3. Add corrected memory telemetry. Done:
   - per-candidate `tegrastats`,
   - `/proc/meminfo` available-memory drop,
   - systemd cgroup process accounting,
   - `nvidia-smi` where exposed by the platform.
4. Run 3-5 calibration samples:
   - 4k / c1,
   - 16k / c4,
   - 32k / c8,
   - 32k / c16,
   - 100k / c1.
5. Push calibration/smoke harness results. Done for code/docs; raw local smoke
   JSONL remains uncommitted.
6. Start the full safe 140-candidate sweep under a user systemd runner. Done
   on 2026-06-23:

   ```text
   runner unit = localperf-gemma-sweep-20260623T075153Z.service
   results = results/tegrastats-sweep-20260623T075153Z-results.jsonl
   events = results/tegrastats-sweep-20260623T075153Z-events.jsonl
   ```

7. Generate partial plots/reports once enough rows exist to show useful
   surfaces. Done.
8. Generate the final report after the sweep finishes. Done from the completed
   140-row run. Passing 100 rows is a minimum reportability threshold, not a
   reason to stop a healthy 140-candidate sweep.
9. Commit sanitized reports and summaries. Keep raw logs and machine-path JSONL
   local unless explicitly sanitized first. Done for report artifacts; raw
   timestamped JSONL and logs remain uncommitted.

## Current Known Constraints

- `nvidia-smi` is present but reports `N/A` for GB10 memory fields on this
  machine, so total memory pressure must be read from `tegrastats` and
  `/proc/meminfo`. vLLM logs and cgroup/process memory remain useful but do not
  fully represent unified memory.
- `startup_only` rows are expected for candidates where vLLM can start but the
  harness decides request load would be unsafe or misleading. These rows should
  remain in the dataset because they define the feasible boundary.
- `tegrastats` on this machine reports RAM, swap, CPU, and temperature. It does
  not currently expose a `GR3D`/GPU utilization field in the observed output.
- `matplotlib` and `pandas` are not installed in the system Python, so plotting
  uses Python standard-library SVG/HTML generation unless dependencies are
  intentionally added later.
- Existing Qwen vLLM may be loaded before the sweep starts; the harness must
  unload it before Gemma samples.
- The sweep runner itself should be launched with `systemd-run --user`; plain
  shell backgrounding can leave an orphaned candidate vLLM service if the
  parent shell exits unexpectedly.
