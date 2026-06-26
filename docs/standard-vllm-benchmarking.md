---
title: Standard vLLM Benchmarking
author: Bob <dutifulbob@gmail.com>
date: 2026-06-26
---

# Standard vLLM Benchmarking

LocalPerf now has a reusable vLLM benchmark runner:

```sh
go run ./cmd/localperf-vllm-bench plan \
  --spec examples/diffusiongemma-vllm-standard/spec.json

go run ./cmd/localperf-vllm-bench run \
  --spec examples/diffusiongemma-vllm-standard/spec.json \
  --timeout 4h

go run ./cmd/localperf-vllm-bench report \
  --run-dir runs/<run-id>
```

Use `--dry-run` on `run` when changing a spec:

```sh
go run ./cmd/localperf-vllm-bench run \
  --spec examples/diffusiongemma-vllm-standard/spec.json \
  --run-dir /tmp/localperf-vllm-dry-run \
  --dry-run
```

Run a small subset before the full matrix:

```sh
go run ./cmd/localperf-vllm-bench run \
  --spec examples/diffusiongemma-vllm-standard/spec.json \
  --profile 8k \
  --workload prefill-8k-out16-fixed \
  --concurrency 4 \
  --vllm-command /home/bob/scratch/vllm-latest-dgxspark-20260626/.venv/bin/vllm \
  --timeout 1h
```

## What It Standardizes

The spec file defines:

- model and environment variables,
- vLLM server profiles, including `--max-model-len`, `--max-num-seqs`,
  batching, backend, ports, and sleep-mode settings,
- benchmark workloads and concurrency ladders,
- warmup traffic,
- the minimum allowed `/proc/meminfo` `MemAvailable` floor,
- output paths for JSONL events, raw vLLM result JSON, summary JSON, logs, and
  the Markdown report.

This replaces one-off terminal sessions with a repeatable run directory.

## Workload Types

Use separate workloads for separate questions.

| Question | Dataset shape | Output length | Primary metric |
| --- | --- | ---: | --- |
| Prefill speed | `random` with long input | 16 or 32 | total token/s and TTFT |
| Decode throughput | `random` with short or medium input | 256 or 512 | output token/s |
| Claim reproduction | exact published prompt/output/concurrency shape | as claimed | per-user and aggregate output token/s |
| Realistic chat | ShareGPT or a checked-in custom prompt set | task-dependent | latency and success rate |
| Prefix cache | `prefix_repetition` | task-dependent | cache hit benefit |

There is no single standard prompt. For controlled prefill and decode, fixed
token-count synthetic inputs are better than prose because they make token count
and concurrency exact. For product-like behavior, use a separate realistic chat
set and do not mix those numbers with raw throughput claims.

## Sleep-Mode Profile Pools

The runner supports vLLM sleep mode through profile settings:

```json
{
  "managed": true,
  "enable_sleep_mode": true,
  "sleep_level": 2
}
```

The safe default is one active profile at a time. A profile is started, warmed,
benchmarked, slept or stopped, and then the next profile starts. This avoids
holding multiple awake model profiles in unified memory.

For hot profile pools, set:

```json
{
  "runner": {
    "preboot_profiles": true,
    "one_awake_profile": true
  }
}
```

That starts each managed profile sequentially, waits for readiness, warms it,
sleeps it, and moves to the next profile. During measurement it wakes one
profile, runs all workloads for that profile, then sleeps it again. Use level 2
sleep first on the Spark/GB10 machine because it drops GPU/unified-memory
pressure much more aggressively than level 1.

## OOM Avoidance

The benchmark runner checks `MemAvailable` before profile startup, wake, warmup,
sleep, and each measured workload:

```json
{
  "safety": {
    "min_mem_available_gib": 40
  }
}
```

If available memory is below the floor, the run records a skipped/failed event
instead of launching the next heavy step. This is the primary guard. `earlyoom`
is still useful as a machine-level backstop, but the benchmark should not rely
on the kernel or earlyoom to recover from unsafe scheduling.

The runner also monitors the same floor during server startup and benchmark
subprocesses. If memory drops below the floor mid-step, the step is canceled and
the profile process is stopped.

Do not lower the memory floor automatically after a failure. Change the profile
settings, reduce `gpu_memory_utilization`, reduce `max_num_seqs`, or split the
workload.

## DiffusionGemma Example

The checked-in spec is:

```text
examples/diffusiongemma-vllm-standard/spec.json
```

It includes:

- a `4k-reference` profile for the earlier 1k prompt / 1024 output claim-repro
  shape,
- 8k, 16k, and 32k profiles for practical long-context characterization,
- fixed-token decode and prefill workloads,
- warmup before measured traffic,
- `TRITON_ATTN` attention and `cutlass` MoE backend,
- sleep-mode enabled on all managed profiles,
- a 40 GiB `MemAvailable` floor.

The checked-in known-results fixture is:

```text
examples/diffusiongemma-vllm-standard/known-run/report.md
```

It records the reproduced 311 tok/s single-worker reference point, the 557 tok/s
20-worker claim attempt, and the 8k 4/8/16 worker decode grid measured on
2026-06-26. See also
`docs/2026-06-26-diffusiongemma-throughput-notes.md`.

## Report Interpretation

The generated report separates:

- configured context window,
- benchmark concurrency,
- input and output token shape,
- completed and failed requests,
- aggregate output token/s,
- per-user output token/s,
- total token/s,
- TTFT when vLLM reports it.

Aggregate output token/s answers "how much throughput did the server produce?"
Per-user output token/s answers "what did each concurrent user experience?"
For capacity planning, use both, plus the memory telemetry described in
`docs/2026-06-23-measurement-methods.md`.
