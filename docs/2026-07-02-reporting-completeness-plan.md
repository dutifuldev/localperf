---
title: Reporting Completeness and Capture Plan
author: Bob <dutifulbob@gmail.com>
date: 2026-07-02
---

# Reporting Completeness and Capture Plan

This doc is the implementation plan for making localperf reports complete and
unambiguous, measured against the reporting requirements in the BentoML LLM
Inference Handbook (https://github.com/bentoml/llm-inference-handbook). It
consolidates the context-semantics contract
(`2026-07-02-context-semantics.md`) with the metric, capture, and goodput
gaps found when auditing localperf against that source.

## Principle

One rule runs through every phase:

- The runner records raw facts only (timestamps, token counts, samples,
  probes). It never precomputes a number a reader will see.
- Every reported metric is defined exactly once, in a metrics registry in
  code. The report legend and the computation come from the same definition,
  so they cannot drift.
- The report is a pure view over the SQLite artifact. Anything derivable at
  render time is derived at render time, which makes fixes retroactive for
  existing artifacts.
- Semantic claims (context, SLO, engine identity) are declared in the spec
  and rendered only when measurement agrees. Declared, checked, then shown.

## Phase 1: honest context labels

Implements `2026-07-02-context-semantics.md`:

- `context_target` and `context_semantics` workload fields with hard
  validation in `internal/vllmbench/config.go` (90% to 100% band,
  `max_model_len >= context_target`, teaching error message).
- Report labeling in `internal/report/html.go`: group titles only from
  declared-and-measured targets; measured shape as fallback; long-output rows
  display the `active_start -> active_end` range; `max_model_len` renders
  only as a server limit attribute.

Acceptance: the old Gemma artifact re-renders as `~1k -> 5k active` instead
of "32k context", and its spec is refused by validation.

## Phase 2: report what the artifact already knows

Render-only, retroactive for all existing artifacts, in `internal/report/`:

- Metrics registry: each reported quantity defined once with name, unit,
  formula, and weighting. Feeds both computation and an auto-generated
  legend footer. `2026-06-23-measurement-methods.md` references it.
- Token-weighted ITL derived from the `requests` table:
  `sum(itl_mean_ms * (completion_tokens - 1)) / sum(completion_tokens - 1)`
  over completed requests. Rendered as the ITL column. TPOT stays the
  request-weighted per-user number. Today's "ITL mean" is request-weighted
  and near-duplicates TPOT, which is exactly the ambiguity the handbook
  warns about.
- Median (p50) and p99 for TTFT and end-to-end latency. Already stored in
  `metric_stats`, never rendered.
- Requests per second: `completed_requests / wall_time`.
- Achieved concurrency, time-weighted from request start/end timestamps,
  shown as `~19 (of 32)` when it diverges from requested. Requested
  concurrency displayed as achieved load is the same conflation shape as
  capacity displayed as context.
- Failure breakdown by `error_type` instead of a bare failed count.
- Repeat aggregation: one primary row per (profile, workload, concurrency)
  with mean and standard deviation across `repeat_index`; per-repeat rows
  secondary. Comparisons without variance present noise as signal.

## Phase 3: capture what is genuinely missing

Runner changes; benefits future runs, older artifacts render "-":

- Hardware inventory into the existing `run.host_json`: GPU name, count,
  VRAM, driver from `nvidia-smi --query-gpu`, plus CPU and RAM. Report
  header gains a Hardware line. Degrade gracefully without nvidia-smi.
- GPU telemetry into the existing (currently unused) `telemetry_series` and
  `telemetry_samples` tables: utilization and memory-used sampled about
  every 2s during measurement phases, tagged with `measurement_id`. Report
  shows avg/peak GPU utilization and peak VRAM per row.
- Engine identity probes: GET `/version` (vLLM) and `/v1/models` (any
  OpenAI-compatible server, including LM Studio and llama.cpp) at startup.
  Fill the currently always-NULL `engines.version` and store the server's
  self-reported identity in `engines.metadata_json`. External engines become
  verified rather than trusted; a spec/server mismatch becomes visible.
- Capture `enable_prefix_caching` into `profiles.serve_json` and render it
  next to the KV cache dtype. Prefix caching can make prefill numbers
  partly fictional; readers must be able to see whether it was on.

## Phase 4: goodput and the sweep generator

- Optional `slo` block on workloads (for example
  `{"ttft_p95_ms": 500, "e2el_p95_ms": 30000}`), stored in the existing
  `workloads.metadata_json`; no schema change. At render time, compute the
  fraction of completed requests meeting the SLO and goodput as SLO-met
  requests per second. Rendered only when declared; the report never invents
  a quality bar. Add an SLO section to
  `2026-06-23-measurement-methods.md` before implementing.
- `localperf sweep plan`: emits the default context/concurrency grid with
  contract-compliant shapes and declared context semantics, per
  `2026-07-02-default-inference-sweep.md`.

## Deliberately excluded

- Cost per token: defer until reports drive buy/rent decisions; revisit
  with an optional GPU cost-per-hour input.
- Persisted claim verdicts, verifier registries, and typed comparison
  enforcement: revisit only if localperf becomes shared CI infrastructure
  where verdicts must be audit-stable across time and tools.

## Constraints

- No schema migrations. Every phase fills columns and tables that already
  exist.
- Each phase lands independently behind the standard gates (`go test`,
  `go vet`, simpledoc, slophammer) plus one dry-run benchmark case and
  `localperf artifact check`.
- Missing data renders as "-", never fabricated and never silently omitted.
- Keep benchmark safety behavior conservative throughout; none of this
  loosens memory floors or guardrails.

## Status

- Contract docs: done (`2026-07-02-context-semantics.md`, sweep doc).
- Phases 1 through 4: not started. Build in order; phases 1 and 2 upgrade
  every existing artifact, so they come first.
