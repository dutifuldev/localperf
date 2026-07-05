---
title: Report Integrity Plan
author: Bob <dutifulbob@gmail.com>
date: 2026-07-05
---

# Report Integrity Plan

Date: 2026-07-05. Status: planned.

Fixes every problem found while inspecting the live viewer serving the
4-model GB10 sweep of 2026-07-05, each verified against the code.

Problems, ranked:

1. TTFT columns show end-to-end latency: `bench http-load` hardcodes
   `"stream": false` and silently falls back to first-byte time
   (`internal/vllmbench/http_load.go`, request body build and
   `applyRequestStats`). Real TTFT is unmeasurable through this path.
2. TTFT p95 is rendered but never recorded (http-load stores mean+p99
   only); the recorded p99 is shown nowhere.
3. Skipped/failed points fragment into duplicate "unverified" tables:
   throughput grouping keys on `(profile, contextLabel)`
   (`internal/report/html.go`) and no-token points get a different label.
   This also splits one concurrency point's prefill/decode across two
   tables and shows the wrong warning for skips.
4. Adaptive skip reasons render only in the detail view; table cells show
   a bare "skipped". Failed cells show "0.000" tok/s instead of a failed
   marker.
5. The viewer measurement detail is config-only: no throughput, latency,
   ITL, failure, telemetry, or SLO numbers, though all exist in the
   artifact.
6. Prefix cache renders "unknown" when the flag is passed via extra args
   (capture reads only the structured spec field).
7. Spec-authoring drift: the sweeps ran from hand-authored specs with
   silently trimmed 32k/64k ladders and a non-streaming backend. Nothing
   distinguishes generated specs from hand-written ones, and nothing
   records deliberate ladder trims.
8. Legacy exports still exist: every run writes `report.md`,
   `report.json`, and `report.csv` (`internal/vllmbench/report.go`,
   `bench report`) with pre-contract labeling, and the artifact ingests
   them as `normalized_report` attachments. They contradict the HTML
   report and would keep publishing the fake TTFT values.

Design principle throughout, same contract as
[Context Semantics](2026-07-02-context-semantics.md): declared, then
verified, then labeled; anything unverifiable is labeled as such, never
silently trusted. No legacy paths; cutover.

## PR 1: real TTFT (streaming load generator)

The measurement fix; everything else is presentation or provenance.

- `internal/vllmbench/http_load.go`: add SSE streaming to the openai-chat
  backend. Send `"stream": true` plus
  `"stream_options": {"include_usage": true}`. Parse `data: {...}` chunks;
  TTFT = first chunk with non-empty delta content; per-chunk timestamps
  give ITL (`itl_mean = (last-first)/(n-1)`); token counts from the final
  usage chunk. Set `Streamed: true`, `TTFTMillis`, `ITLMeanMillis` on
  samples.
- Streaming is the default for all http-load workloads. Workload/CLI
  opt-out (`"stream": false`, `--no-stream`) exists for engines without
  SSE, and a non-streaming run records NO TTFT — delete the first-byte
  fallback in `applyRequestStats`. E2E latency already has its own
  columns.
- Record full TTFT stats from streamed samples: mean/p50/p95/p99 (the
  stats helper already computes them; write them all).
- Measurement metadata records `ttft_source: "stream"` when TTFT came from
  streamed samples. The report trusts TTFT stats only with that marker:
  everything else renders "-" (with a "not streamed" note in the legend).
  This retroactively makes the four existing GB10 reports honest — they
  show "-" instead of E2E-as-TTFT — without any artifact surgery.
- Tests: SSE fake server (chunk timing controlled) asserting TTFT/ITL/
  usage parsing; non-streaming run asserting TTFT absent; report test
  asserting the ttft_source gate.

## PR 2: report and viewer rendering

- Table grouping (`internal/report/html.go`): group by
  `(profile, declared context claim)` — target + semantics — not by final
  label. Skipped/failed/unverified points stay as rows inside their
  declared context's table with row-level status. The table title comes
  from the verified state of its completed rows; the "unverified: not
  confirmed by token counts" warning applies only to
  completed-but-unverifiable rows, never to skips (a skip's message is its
  reason). Completed-but-mismatched rows keep their measured-shape label
  and mismatch note at row level. This also fixes the prefill/decode phase
  split (one group key for both phases).
- Failure display: when status is failed/skipped, the tok/s cell shows the
  failure label ("failed 0/32", "skipped"), not "0.000"
  (`displayFailureMetric` currently substitutes only for empty values).
- Skip/fail reasons inline: add `failure_reason` to the throughput cell
  payload (`internal/reportmodel/reportmodel.go`) and render it in the SPA
  table (tooltip or sub-line) and in the static HTML cells, not just the
  detail view.
- Percentile columns: tables render TTFT mean + p99 (what is recorded and
  what the handbook prefers); p50/p95 appear in the detail view. Update
  the metrics registry, `report.gohtml`, and the SPA columns together so
  the legend matches reality.
- Viewer measurement detail (`internal/reportmodel` + `web/src/main.tsx`):
  include the numbers that exist in the artifact — throughput stats,
  latency percentiles, TTFT (when real), ITL, token counts, failure
  breakdown by error_type, telemetry summary, SLO/goodput. Reuse the
  structs the static HTML report already computes; do not re-derive.
- Engine comparability: when the viewer serves multiple reports with
  different engine versions, show a note in the report list/header.
- Legacy export cutover: delete the `report.md`/`report.json`/`report.csv`
  pipeline outright — `internal/vllmbench/report.go`, the finalize-time
  markdown write in `runner.go`, the `bench report` rendering command, and
  the `normalized_report` attachment ingestion in `sqlite_artifact.go`.
  The run directory keeps raw data only (spec, events, results, logs,
  datasets, summary.json); the SQLite artifact is the canonical record and
  the HTML report/viewer the only rendered views. Machine-readable export
  is the artifact itself. Update the README Outputs section and any docs
  referencing the deleted files. Coordinate with the viewer-mode branch:
  its rebuild-from-run-dir path must key off events/results/summary.json,
  never the deleted exports.
- Rebuild `web/dist` (npm build) and re-embed.

## PR 3: spec provenance and intent compilation

- Intent surface: extend `sweep plan` (`internal/sweepplan`) with the
  fields that forced hand-authoring — runtime path (vllm binary),
  gpu-memory-utilization, kv-cache-memory-bytes — plus declared ladder
  trims: `--trim 64k=8:"12 GiB KV budget"` →
  `{"context": 65536, "max_concurrency": 8, "reason": "..."}`.
- Trims compile into the spec as metadata AND remove the trimmed points
  from the grid; the report synthesizes "trimmed by author: <reason>" rows
  from that metadata so trimmed points render like adaptive skips, never
  as holes.
- Provenance stamp: generated specs carry
  `generator: {tool, version, intent, content_hash}` where the hash covers
  the canonical spec minus the stamp. `bench run` recomputes the hash and
  records the result; report/viewer header shows "Generated default sweep"
  or "Custom grid" for anything unstamped, edited, or stale. Hand-written
  specs keep working — visibly labeled.
- The spec stays the record in the run dir; intent is the only editing
  surface. Docs: update
  [Default Inference Sweep](2026-07-02-default-inference-sweep.md) and the
  README to route spec creation through `sweep plan` only.

## Engine config capture fix (rides in PR 2 or 3)

- Compute the effective `enable_prefix_caching` tri-state where the serve
  command is assembled: structured field wins; otherwise parse the merged
  args (`--enable-prefix-caching`, `--no-enable-prefix-caching`,
  `--enable-prefix-caching=false`). Store the resolved value in serve_json
  so "unknown" only appears when genuinely unknown.

## After merge

- Regenerate the four GB10 specs through the new compiler (declaring the
  64k/32k trims with reasons) and rerun the sweeps to get real TTFT data.
  Until then the existing reports render TTFT as "-", which is the truth.

## Invariants

- No legacy paths; cutover. No silent fallbacks: a metric renders only
  when actually measured.
- Every skip/trim/failure renders with a reason.
- Generated specs keep round-tripping ValidateSpec; the sweepplan golden
  is updated.
- Gates: go test/vet, gofmt, simpledoc, slophammer check/dry/crap,
  golangci-lint, AGENTS.md dry-run + artifact check. No mutation tests.
- PRs at the end of each phase; do not merge without explicit approval.
