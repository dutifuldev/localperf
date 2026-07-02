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

## Data availability rule

Per-request rows (`requests` table) exist only for measurements recorded
with `save_detailed=1`. Every derivation in this plan therefore has two
paths:

- detailed path: derive from `requests` rows,
- fallback path: derive from `measurements` aggregates when possible,
  otherwise render `-`.

Never fabricate a value on the fallback path. `-` means "not measured", and
that is information.

## Phase 1: honest context labels

Implements `2026-07-02-context-semantics.md`.

### Spec fields and validation (`internal/vllmbench/config.go`)

Add to the `Workload` struct (next to the embedded
`BenchmarkTrafficConfig`):

```go
ContextTarget    int    `json:"context_target,omitempty"`
ContextSemantics string `json:"context_semantics,omitempty"` // "active" | "capacity"
```

Constants, defined once:

```go
const (
    contextTargetMinFrac = 0.90
    contextTargetMaxFrac = 1.00
)
```

Add `validateWorkloadContextSemantics(prefix string, workload Workload,
profiles []Profile) []string`, called from `validateWorkloads` (which
already receives profile names; extend it to receive the profiles
themselves). Rules:

1. Both fields set or both empty. One without the other is an error.
2. `context_semantics` must be `"active"` or `"capacity"`.
3. For `"active"`: `random_input_len + random_output_len` must be within
   `[0.90, 1.00] * context_target`, and every profile the workload pairs
   with in planned-run expansion must have `max_model_len >=
   context_target`. Use the exact error message wording from the contract
   doc; the message must name the measured-versus-claimed percentage and
   the three remediations (lower the target, raise the input, or declare
   capacity semantics).
4. For `"capacity"`: `context_target` must equal the paired profile's
   `max_model_len`.

These are hard errors from `ValidateSpec`, so invalid specs die before any
server starts. Workloads with neither field remain valid (legacy) and get
no context claim.

The fields flow into the artifact automatically: `insertWorkloads` in
`internal/vllmbench/sqlite_artifact.go` serializes the embedded traffic
config into `workloads.traffic_json`, and the normalized spec row keeps the
full workload JSON. No schema change.

### Report labeling (`internal/report/html.go`)

Load two new inputs per measurement:

- declared claim: parse `context_target`/`context_semantics` out of
  `workloads.traffic_json` in the existing workload query.
- measured active context, detailed path:

```sql
SELECT AVG(prompt_tokens), AVG(completion_tokens)
FROM requests
WHERE measurement_id = ? AND status = 'completed';
```

  Fallback path: `measurements.prompt_tokens / completed_requests` and
  `measurements.completion_tokens / completed_requests`.

Add to `SQLiteReportMeasurement`: `ContextTarget int`, `ContextSemantics
string`, `ActiveStartMean float64`, `ActiveEndMean float64` (plus known
flags matching the existing `...Known` convention).

Replace capacity-based titling:

- `throughputGroupKey` changes from `{profile, contextWindow}` to
  `{profile, contextLabel}` where `contextLabel` is computed by a new
  `contextRowLabel(m)`:
  1. If `ContextSemantics == "active"` and `ActiveEndMean` is within
     `[0.90, 1.00] * ContextTarget`: label is
     `contextLabel(ContextTarget) + " active context"`.
  2. If the row's mean completion tokens exceed a few tokens, the row also
     carries the range string
     `activeRange = contextLabel(start) + " -> " + contextLabel(end)`,
     rendered in the group header and the Shape column.
  3. If a target is declared but measurement lands outside the band: label
     falls back to measured shape and the row gets a `mismatch` badge
     showing declared versus measured (new CSS class next to the existing
     status pills).
  4. No declared target: label is the measured shape from the existing
     `requestShape` helper.
- `contextTitle(profile.context_window)` is deleted. `context_window`
  renders only as `server limit: 32k` in the group axis items and the
  profile table.
- `sqliteReportMetadataItems`: the `Contexts` line lists verified active
  targets only; add a separate `Server limits` line listing distinct
  `max_model_len` values.
- Sorting (`sqliteReportThroughputRows`) orders by `ActiveEndMean` (or
  target when verified) instead of `ContextWindow`.

### Tests and acceptance

- `config_test.go`: table-driven cases for each validation rule, including
  the exact Gemma shape (target 32768, 1024+4096) asserting refusal and
  message content.
- `html_test.go`: fixture artifacts for verified-active, capacity,
  mismatch, and legacy rows; assert group titles, range strings, badge.
- Acceptance: re-render
  `gemma4-merged-practical-long-output-20260701.sqlite`; the 32k-profile
  decode rows must render as `~1k -> 5k active` (measured shape path), not
  "32k context".

## Phase 2: report what the artifact already knows

Render-only; retroactive for all existing artifacts. All in
`internal/report/`.

### Metrics registry (`internal/report/metrics.go`, new file)

```go
type MetricDef struct {
    Key        string // stable id, e.g. "itl_token_weighted"
    Label      string // column header, e.g. "ITL (tok-weighted)"
    Unit       string
    Weighting  string // "per-request" | "per-token" | "aggregate"
    Definition string // one-sentence formula, rendered in the legend
}

var ReportMetrics = []MetricDef{ ... }
```

Entries: TTFT, E2EL, TPOT (request-weighted), ITL (token-weighted), RPS,
input/output/total tok/s, per-user output tok/s, goodput (phase 4), GPU
utilization (phase 3). The HTML template gains a `<details>` legend footer
ranging over `ReportMetrics`. `2026-06-23-measurement-methods.md` names
this registry as the canonical definition list.

### Token-weighted ITL

Detailed path, one query per measurement (CASE instead of FILTER for older
SQLite):

```sql
SELECT SUM(itl_mean_ms * (completion_tokens - 1)) * 1.0
         / SUM(completion_tokens - 1)
FROM requests
WHERE measurement_id = ?
  AND status = 'completed'
  AND completion_tokens > 1;
```

This is exact, not an approximation: per-request `itl_mean_ms` times its
gap count recovers that request's gap sum. Fallback path: `-` (the
request-weighted value is not an acceptable substitute; substituting it is
the current bug). Rendered as the ITL column; the existing request-weighted
TPOT column stays and is labeled `TPOT (per-req)`.

Unit test: build a fixture with known gap arrays, assert the SQL result
equals brute-force `sum(all gaps)/count(all gaps)` to within float
tolerance, and assert it differs from mean-of-means on a skewed fixture.

### Percentiles, RPS, failure breakdown

- p50/p99: already loaded by the `metric_stats` scan; add
  `TTFTP50MS/TTFTP99MS/LatencyP50MS/LatencyP99MS` via the existing
  `metricDisplayFirst(metrics, "P50"|"P99", "request_ttft", "ttft")`
  pattern, and matching columns in the detail table template.
- RPS: `completed_requests / (wall_time_ms / 1000)`, computed in Go beside
  the existing throughput display fields.
- Failure breakdown:

```sql
SELECT COALESCE(error_type, 'unknown'), COUNT(*)
FROM requests
WHERE measurement_id = ? AND status != 'completed'
GROUP BY 1 ORDER BY 2 DESC;
```

  Rendered as `3 failed (2 timeout, 1 http_5xx)` in place of the bare
  failed count; fallback path keeps the bare count.

### Achieved concurrency

Detailed path: load `(started_at, completed_at)` per completed request,
sweep-line in Go: sort all endpoints, walk them accumulating
`in_flight * dt`, divide by total span. Display as `~19 (of 32)` when
`|achieved - requested| / requested > 0.10`, else just the requested
number. Fallback path: requested number only. Unit test against a
hand-computed interval fixture.

### Repeat aggregation

Group loaded measurements by `(profile, workload, concurrency)` before row
construction. When a group has more than one `repeat_index`:

- primary row shows mean and stddev (`123.4 ± 5.6`) for throughput, TTFT,
  and latency columns, plus an `xN` repeat count,
- per-repeat rows render in a collapsed `<details>` section under the
  group,
- status: worst status in the group wins the pill.

Stddev is sample stddev (n-1); with n=1 render the plain value, no `±`.

### Template changes

The detail table (`Profile / Workload / ...` thead in the HTML template)
gains: RPS, TTFT p50/p99, Latency p50/p99, ITL (tok-weighted), achieved
concurrency, failure breakdown. The compact comparison table keeps its
current column set (avg/p95) to stay readable; it gains only the repeat
`±` treatment. Golden-file tests: extend `html_test.go` fixtures and
assert rendered fragments, following the existing test style.

## Phase 3: capture what is genuinely missing

Runner changes; benefits future runs, older artifacts render `-`.

### Hardware inventory (`internal/vllmbench/hostinfo.go`, new file)

```go
type HostInfo struct {
    CPUModel  string    `json:"cpu,omitempty"`
    RAMGiB    float64   `json:"ram_gib,omitempty"`
    GPUs      []GPUInfo `json:"gpus,omitempty"`
    Telemetry []string  `json:"telemetry_sources,omitempty"`
}
type GPUInfo struct {
    Name    string  `json:"name"`
    VRAMGiB float64 `json:"vram_gib,omitempty"`
    Driver  string  `json:"driver,omitempty"`
}
```

Collect once in `newRunSession`: CPU/RAM from `/proc/cpuinfo` and
`/proc/meminfo`; GPUs from
`nvidia-smi --query-gpu=name,memory.total,driver_version
--format=csv,noheader` with a 5s timeout; `Telemetry` lists which of
`tegrastats`/`nvml`/`nvidia-smi` respond. Every field optional; on a
machine with none of them, record what exists. Serialize into the
`host_json` parameter that `insertRun` already accepts (currently unset).
Report: add a `Hardware` item to `sqliteReportMetadataItems`, formatted
`GB10 x1 (unified, driver 580.x)` style; `-` when absent. Column
conventions are documented in `2026-06-29-sqlite-run-artifact-format.md`.

### GPU telemetry sampler

Source preference per `2026-06-23-measurement-methods.md`: `tegrastats`
on unified-memory systems, then NVML/DCGM, then `nvidia-smi`. Series names
per the artifact format doc (`gpu_utilization_percent`,
`gpu_memory_used_bytes`, with `source` set to the actual tool).

Implementation: a `telemetrySampler` goroutine started in
`runProfileWorkload` around measurement execution and stopped after it,
sampling every 2s. It appends JSONL telemetry events to the run directory
(same event-stream pattern the runner already uses for commands and
phases), carrying the measurement key. `insertTelemetry` in
`sqlite_artifact.go` ingests them into `telemetry_series` /
`telemetry_samples` with `measurement_id` set. On unified-memory systems
GPU memory counters may be unreliable; record the source and cross-check
against `MemAvailable` drop rather than trusting one signal.

Report: per measurement,

```sql
SELECT AVG(ts.value), MAX(ts.value)
FROM telemetry_samples ts
JOIN telemetry_series s ON s.id = ts.series_id
WHERE ts.measurement_id = ? AND s.metric = 'gpu_utilization_percent';
```

rendered as `GPU util avg/peak` and (same shape for memory) `peak VRAM`,
with the source named in the legend.

### Engine identity probes

After `readinessWaiter` reports ready (`probeReady` success path), issue
two GETs with a 2s timeout each against the profile base URL:

- `/version` (vLLM): fills `engines.version`, today always NULL at the
  `insertEngines` call site.
- `/v1/models` (any OpenAI-compatible server, including LM Studio and
  llama.cpp): raw JSON stored under `identity` in
  `engines.metadata_json`.

For managed vLLM, fall back to parsing the version line from the captured
startup log if `/version` fails. The report engine summary
(`engineSummaries`) appends the version when present. A report-level
warning renders when the identity's served model does not match the
profile's `model` — declared versus self-reported, the same
declared-then-verified pattern as context.

### Prefix caching flag

Add `EnablePrefixCaching *bool` to the `Profile` serve options, emit the
corresponding vLLM flag in `internal/vllmbench/commands.go`, include it in
the `serve_json` map in `insertProfiles`, and render it in the profile
table next to KV cache dtype. Tri-state: on / off / unknown (older
artifacts).

## Phase 4: goodput and the sweep generator

### SLO block and goodput

Spec (`config.go`):

```go
type SLOConfig struct {
    TTFTP95Millis float64 `json:"ttft_p95_ms,omitempty"`
    E2ELP95Millis float64 `json:"e2el_p95_ms,omitempty"`
}
// on Workload:
SLO *SLOConfig `json:"slo,omitempty"`
```

Serialized into `workloads.metadata_json` by `insertWorkloads`; no schema
change. Validation: values must be positive when set.

Render-time derivation, detailed path only:

```sql
SELECT SUM(CASE WHEN ttft_ms <= ? AND latency_ms <= ? THEN 1 ELSE 0 END),
       COUNT(*)
FROM requests
WHERE measurement_id = ? AND status = 'completed';
```

Columns (only when an SLO is declared): `% in SLO` and `goodput req/s` =
SLO-met count / wall time. The report never invents a quality bar. Note:
the spec already passes a `goodput` field through to `vllm bench serve`
(see `2026-06-26-standard-vllm-benchmarking.md`); the render-time
derivation is canonical because it is engine-agnostic and re-derivable,
but the two must agree on SLO definitions, not compete. Add an SLO section
to `2026-06-23-measurement-methods.md` before implementing.

### `localperf sweep plan` (new `internal/sweepplan`)

Pure generator, no I/O in the core:

```go
type PlanRequest struct {
    Model       string
    Engine      string // "vllm-managed" first
    Contexts    []int  // active-context ladder, e.g. 8k..128k
    Concurrency []int  // e.g. 1,4,8,16,32
    Repeats     int
}
func Plan(req PlanRequest) (vllmbench.Spec, error)
```

Shape derivation per active-context point `N`, from
`2026-07-02-default-inference-sweep.md`:

```go
headroom := max(64, N/64)
// prefill: input = N - headroom - 1, output = 1
// decode:  output = min(4096, N/4), input = N - output - headroom
```

Every generated workload sets `context_target: N` and
`context_semantics: "active"`; the `max-throughput-reference` family sets
`"capacity"`. Generated specs must round-trip `ValidateSpec` with zero
issues — enforced by a test, which is the guarantee that generator and
validator can never drift.

CLI: a `sweep plan` subcommand in `internal/benchcli` (the dispatch that
already routes `bench run` and `artifact check`), flags `--model`,
`--engine`, `--contexts 8k,16k,...`, `--concurrency 1,4,...`,
`--repeats`, `--out spec.json` (default stdout). Golden-file test: fixed
request in, byte-stable spec out.

## Deliberately excluded

- Cost per token: defer until reports drive buy/rent decisions; revisit
  with an optional GPU cost-per-hour input.
- Persisted claim verdicts, verifier registries, and typed comparison
  enforcement: revisit only if localperf becomes shared CI infrastructure
  where verdicts must be audit-stable across time and tools.

## Constraints

- No schema migrations. Every phase fills columns and tables that already
  exist; new spec fields ride in existing JSON columns.
- Each phase lands independently behind the standard gates (`go test`,
  `go vet`, simpledoc, slophammer) plus one dry-run benchmark case and
  `localperf artifact check`.
- Missing data renders as `-`, never fabricated and never silently
  omitted.
- Keep benchmark safety behavior conservative throughout; none of this
  loosens memory floors or guardrails.

## Status

- Contract docs: done (`2026-07-02-context-semantics.md`, sweep doc).
- Phases 1 through 4: not started. Build in order; phases 1 and 2 upgrade
  every existing artifact, so they come first.
