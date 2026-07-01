---
title: Inference Engine Architecture Plan
author: Bob <dutifulbob@gmail.com>
date: 2026-06-29
---

# Inference Engine Architecture Plan

The long-term target is:

```text
one benchmark spec -> many inference engines -> comparable results
```

localperf should own workload definition, safety, telemetry, result
normalization, and reporting. Engines should be thin adapters that know how to
start, stop, configure, and health-check one backend.

This keeps vLLM-specific features like sleep mode useful without making the
whole product a vLLM wrapper.

## Design Goals

- Run the same benchmark plan against vLLM, SGLang, TGI, llama.cpp, Ollama, LM
  Studio, and any OpenAI-compatible endpoint.
- Preserve exact engine behavior when serve-time settings matter, especially
  `max_model_len`, KV cache sizing, scheduler settings, and batching knobs.
- Keep memory guardrails engine-independent.
- Record enough engine metadata to explain results later.
- Make hot profile pools an adapter feature, not a core assumption.
- Keep reports comparable even when engines expose different detailed metrics.

## Non-Goals

- Do not hide every backend option behind a lowest-common-denominator schema.
  Engine-specific flags still need an escape hatch.
- Do not force all engines to support managed startup. External endpoints must
  be first-class.
- Do not require sleep/wake support. It is an optional capability.
- Do not merge quality benchmarks and throughput benchmarks into one ambiguous
  score.

## Target CLI

The user-facing shape should stay simple:

```sh
localperf bench plan --spec examples/diffusiongemma/spec.json
localperf bench run --spec examples/diffusiongemma/spec.json --timeout 8h
localperf bench report --run-dir runs/<run-id>
```

Short aliases can come later. The important part is that `bench` is not named
after one engine.

## Spec Shape

Use an engine-neutral top level with engine-specific profile blocks:

```json
{
  "version": "localperf.bench/v1",
  "name": "diffusiongemma-standard",
  "model": "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
  "safety": {
    "min_mem_available_gib": 40
  },
  "runner": {
    "one_awake_profile": true,
    "preboot_profiles": false
  },
  "engines": [
    {
      "name": "vllm",
      "type": "vllm-managed",
      "command": "vllm"
    }
  ],
  "profiles": [
    {
      "name": "32k-vllm",
      "engine": "vllm",
      "port": 8132,
      "serve": {
        "max_model_len": 32768,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.35
      },
      "engine_args": [
        "--attention-backend", "TRITON_ATTN",
        "--moe-backend", "cutlass"
      ]
    }
  ],
  "workloads": [
    {
      "name": "decode-1k-out512",
      "profiles": ["32k-vllm"],
      "traffic": {
        "protocol": "openai-chat",
        "dataset": "random",
        "random_input_len": 1000,
        "random_output_len": 512,
        "random_range_ratio": "0",
        "request_rate": "inf",
        "ignore_eos": true,
        "temperature": 0
      },
      "concurrency": [4, 8, 16],
      "samples": 32,
      "repeats": 2,
      "save_detailed": true
    }
  ]
}
```

Rules:

- `workloads[].traffic` is engine-neutral request traffic.
- `profiles[].serve` is normalized when the option exists across several
  engines.
- `profiles[].engine_args` is the raw escape hatch.
- `engines[].type` picks the adapter.
- `profiles[].engine` links a profile to an adapter instance.
- `samples`, `repeats`, and `save_detailed` are first-class so variance is not
  accidental.

## Core Interfaces

Create a generic benchmark core under `internal/bench`.

```go
type Engine interface {
    Name() string
    Start(ctx context.Context, profile Profile, run RunPaths) error
    Stop(ctx context.Context) error
    Health(ctx context.Context) error
    Endpoint(profile Profile) Endpoint
    Metadata(ctx context.Context) (EngineMetadata, error)
}

type SleepCapable interface {
    Sleep(ctx context.Context, level int) error
    Wake(ctx context.Context) error
}

type LoadGenerator interface {
    Run(ctx context.Context, endpoint Endpoint, workload Workload, concurrency int, run RunPaths) (RawResult, error)
}

type Reporter interface {
    Normalize(raw RawResult, metadata RunMetadata) (Result, error)
    Write(run Run, paths RunPaths) error
}
```

The core runner should only depend on these interfaces. It should not import
`internal/vllmbench`.

## Adapter Set

Implement adapters in this order.

| Adapter | Purpose | Managed startup | Sleep mode | Traffic path |
| --- | --- | --- | --- | --- |
| `openai-endpoint` | Any already-running OpenAI-compatible server | no | no | native HTTP |
| `vllm-managed` | Current local vLLM runner behavior | yes | yes | `vllm bench` or HTTP |
| `vllm-endpoint` | Already-running vLLM server | no | admin API if enabled | `vllm bench` or HTTP |
| `sglang-managed` | SGLang server profiles | yes | adapter-specific if available | HTTP |
| `llama.cpp-managed` | local `llama-server` profiles | yes | no | HTTP |
| `ollama-endpoint` | Running Ollama | no | no | Ollama/OpenAI HTTP |
| `lmstudio-endpoint` | Running LM Studio local server | no | no | OpenAI HTTP |

Start with `openai-endpoint` and `vllm-managed`. That proves the abstraction
without boiling the ocean.

## Load Generation

Split traffic generation from engine control.

Initial load generators:

- `vllm-bench`: wraps `vllm bench serve`; best for vLLM-specific benchmark
  metrics.
- `openai-http`: built into localperf; sends OpenAI-compatible requests and
  records per-request rows.

`openai-http` should become the universal fallback. It must record:

- request start/end timestamps,
- success or error,
- prompt tokens when available,
- completion tokens when available,
- total tokens when available,
- TTFT when streaming is enabled,
- inter-token timing when streaming is enabled,
- status code and error class.

Use `vllm-bench` when the engine is vLLM and the goal is exact vLLM benchmark
parity. Use `openai-http` when comparing engines.

## Result Schema

Every result row should normalize into the same shape:

```json
{
  "run_id": "...",
  "engine": "vllm",
  "engine_type": "vllm-managed",
  "profile": "32k-vllm",
  "workload": "decode-1k-out512",
  "concurrency": 16,
  "samples": 32,
  "repeat": 1,
  "context_window": 32768,
  "input_tokens_requested": 1000,
  "output_tokens_requested": 512,
  "completed": 32,
  "failed": 0,
  "aggregate_output_tok_s": 308.1,
  "per_user_output_tok_s": 19.3,
  "aggregate_total_tok_s": 910.0,
  "latency_ms": {
    "mean": 22800,
    "p50": 22100,
    "p95": 31000,
    "p99": 33000,
    "stddev": 4900
  },
  "ttft_ms": {
    "mean": 1200,
    "p50": 1100,
    "p95": 1800,
    "p99": 2100,
    "stddev": 300
  },
  "memory": {
    "min_mem_available_gib": 44.2,
    "system_memory_drop_gib": 78.5,
    "cgroup_peak_gib": null
  }
}
```

Keep raw engine output next to normalized output. Never throw away the raw JSON.

## Hot Profile Pools

Support hot profile pools as a scheduler strategy:

```json
{
  "runner": {
    "preboot_profiles": true,
    "one_awake_profile": true
  }
}
```

Core behavior:

- Start one managed profile.
- Wait for readiness.
- Warm it.
- Sleep it if the adapter supports sleep.
- Move to the next profile.
- During measurement, wake one profile, run all workload points for that
  profile, sleep it, then continue.

Safety rules:

- If `one_awake_profile=true`, reject specs where prebooted managed profiles
  cannot sleep.
- Check `MemAvailable` before every start, wake, warmup, workload, and preboot
  transition.
- Stop the active profile if memory drops below the configured floor.
- Keep `preboot_profiles=false` as the safe default on large models.

## Migration Plan

### Phase 0: Reconcile Baseline

- Bring local `main` to a clean state with the merged vLLM runner present.
- Preserve local-only docs commits and untracked raw run directories.
- Confirm `go test ./...` passes before refactoring.

### Phase 1: Rename Without Behavior Change

- Add `cmd/localperf-bench`.
- Move generic config, planning, safety, run directory, and report code from
  `internal/vllmbench` into `internal/bench`.
- Keep `cmd/localperf-vllm-bench` as a compatibility wrapper.
- Keep current vLLM spec working through a compatibility loader.

Acceptance:

- Existing DiffusionGemma spec still plans the same 36 cases.
- Existing known-run report can still be regenerated.
- No benchmark behavior changes.

### Phase 2: Add Engine Registry

- Add `internal/bench/engines`.
- Register `openai-endpoint` and `vllm-managed`.
- Move vLLM command construction, health checks, sleep/wake, and admin calls
  into the vLLM adapter.
- Add endpoint-only mode for already-running servers.

Acceptance:

- Same workload can run against `vllm-managed`.
- Same workload can run against an external OpenAI-compatible endpoint.
- Reports show `engine`, `engine_type`, `profile`, and endpoint metadata.

### Phase 3: Add Built-In OpenAI HTTP Load Generator

- Implement streaming and non-streaming request modes.
- Save per-request detailed rows by default for benchmark runs.
- Compute throughput and latency variance from per-request data.
- Keep `vllm-bench` available as an optional load generator.

Acceptance:

- localperf can benchmark any OpenAI-compatible endpoint without `vllm bench`.
- Reports include mean, p50, p95, p99, and stddev for latency and TTFT.
- Aggregate token/s variance can be calculated across repeats.

### Phase 4: Spec V1 and Validation

- Introduce `localperf.bench/v1`.
- Add strict validation with clear errors for unsafe or ambiguous specs.
- Add compatibility conversion from the current vLLM-only spec.
- Document all stable fields.

Acceptance:

- Invalid hot-profile specs are rejected before launching a model.
- Engine-specific args are preserved.
- Normalized specs are written into every run directory.

### Phase 5: Comparative Reports

- Add report mode that compares multiple run directories.
- Group by workload and concurrency.
- Show engine-to-engine throughput, latency, failures, memory pressure, and
  variance.
- Include a machine-readable `comparison.json` and human `comparison.md`.

Acceptance:

- One report can compare vLLM vs SGLang vs llama.cpp for the same workload.
- The report warns when workload shapes are not comparable.

### Phase 6: More Adapters

- Add `sglang-managed`.
- Add `llama.cpp-managed`.
- Add `ollama-endpoint`.
- Add `lmstudio-endpoint`.
- Keep each adapter small and covered by command-construction tests.

Acceptance:

- New adapters do not change core runner behavior.
- Endpoint-only adapters require no model lifecycle management.

## Testing Plan

- Unit-test spec parsing, defaults, validation, and compatibility conversion.
- Unit-test command construction for each managed adapter.
- Unit-test safety behavior with fake memory probes.
- Unit-test result normalization with checked-in raw result fixtures.
- Add an integration smoke test against a fake OpenAI-compatible server.
- Keep real GPU/model tests manual or opt-in.

## Documentation Plan

- Update `README.md` after `localperf bench` exists.
- Keep vLLM-specific instructions in a vLLM adapter doc.
- Add an engine adapter authoring guide.
- Add one endpoint-only example spec.
- Add one managed vLLM example spec.
- Add one comparison report fixture.

## First PR Scope

The first implementation PR should be small:

- Add `cmd/localperf-bench`.
- Introduce `internal/bench` interfaces and package boundaries.
- Register `vllm-managed`.
- Keep the existing vLLM runner behavior working.
- Add tests proving the old vLLM spec still plans and runs through the new
  command path.

Do not add SGLang, llama.cpp, Ollama, or LM Studio in the first PR. The first
PR should prove the API shape without increasing backend surface area.

## Open Decisions

- Whether `vllm bench serve` remains the default load generator for vLLM, or
  whether localperf's HTTP generator becomes default everywhere.
- Whether `samples` means total requests per workload point or requests per
  repeat. The likely answer is total requests per repeat.
- Whether engine adapters should live under `internal/bench/engines/<name>` or
  `internal/engines/<name>`.
- Whether hot profile preboot should ever default to true. Current answer:
  no, because large models can still put too much pressure on unified memory.
