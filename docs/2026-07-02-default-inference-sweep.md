---
title: Default Inference Sweep
author: Bob <dutifulbob@gmail.com>
date: 2026-07-02
---

# Default Inference Sweep

Use this as the default localperf sweep shape when the user asks to characterize
an inference engine across context length and concurrency.

## User Instruction

The original benchmark instruction was:

```text
I want to create 2 reference points

max possible throughput we can have, even though not useful. min 4k context window. maximize concurrency and token throughput we can get

then set to 8k, 16k, 32k, 64k and 128k context window, and try to see how concurrent we can get. try to maximize concurrency

note per worker throughput and total aggregate throughput in each case

try to use as much memory we can use. like 80% of the memory. without ooming the device

first document this the type of benchmark we aim for. then execute the benchmark without OOM'in the device
```

## Default Grid

Run two benchmark families:

- `max-throughput-reference`: minimum `4k` context, intentionally optimized for
  maximum aggregate token throughput even if the setting is not practically
  useful. These are capacity points (`context_semantics: "capacity"`).
- `practical-context-sweep`: **active** context points `8k`, `16k`, `32k`,
  `64k`, and `128k`, with concurrency `1`, `4`, `8`, `16`, and `32` where the
  hardware can safely run them.

The ladder points mean active context, not server capacity: a `32k` point must
actually move ~32k tokens per request through the KV cache. Setting
`max_model_len=32768` while requesting a ~1k prompt is a capacity experiment,
not a 32k-context data point; see `2026-07-02-context-semantics.md` for the
contract and validation rules.

Each active-context point `N` includes both workload shapes, with input
lengths that track `N`:

- `prefill`: long prompt, short output. Default shape:
  `input = N - output - headroom`, `output = 256`.
- `decode`: long prompt, long output within `N`. Default shape:
  `output = min(4096, N/4)`, `input = N - output - headroom`.

with `headroom = max(64, N/64)` to absorb chat template and tokenizer drift.
Prefill rows are reported as aggregate and per-user prefill tok/s; decode rows
as aggregate and per-user output tok/s.

Default sweeps must be generated with `localperf sweep plan` once it lands;
until then, hand-author specs to these shapes and declare `context_target` and
`context_semantics` on every workload.

## Extension Rule

The default grid is not a hard ceiling. If the hardware has clear memory and
latency headroom, continue the same ladder with further powers of two:

- context: `256k`, `512k`, and onward,
- concurrency: `64`, `128`, and onward.

Only extend one dimension at a time, and stop before the machine OOMs. A failed
startup or memory-guard kill is a result; do not keep pushing the same shape
without changing the profile.

## Model-Level Artifacts

For a specific model, keep repeated runs in one SQLite artifact and 1 HTML
report per model. Use model-scoped paths such as:

```text
runs/models/<model-slug>.sqlite
runs/models/<model-slug>.html
```

The shared SQLite file should contain one `run` row per benchmark attempt or
batch, with all profiles, workloads, measurements, requests, telemetry, raw
outputs, and report exports tied back by `run_id`. Do not split `c1`, `c4`,
`8k`, `16k`, or separate retry attempts into separate SQLite files unless the
split is only for debugging or recovering from a failed run.

Render the HTML from the shared SQLite artifact after each batch. If the runner
cannot append directly yet, append or merge the new run into the model-level
artifact before treating the sweep as complete.

The final step of every default sweep is to render the completed SQLite
artifact into 1 HTML report per model. Do not call the sweep complete until the
model-level SQLite artifact and the matching model-level HTML report both
exist.

## Reporting Requirements

Every row must record:

- model and inference engine,
- exact engine profile, including context limit and concurrency cap,
- workload name and token shape,
- requested concurrency and completed/error count,
- aggregate throughput and per-user throughput,
- average TTFT and p95 TTFT,
- memory headroom or peak memory pressure,
- exact artifact path for the raw result.

Report context and concurrency together. The primary comparison table should make
it easy to answer: "At this context length, how many users can we run, and what
aggregate and per-user throughput do they get?" Context labels in reports must
follow `2026-07-02-context-semantics.md`: label rows by declared-and-measured
active context or by measured token shape, never by `max_model_len` alone.

## Safety Rules

Use at most about 80% of usable memory unless the user explicitly asks to go
higher. Keep the machine responsive and use memory guards where available.

Prefer a sparse search first:

- start with `c1`, then `c4`, `c8`, `c16`, `c32`,
- skip higher concurrency once memory, TTFT, or error rate makes the profile
  impractical,
- confirm near the best working point with the exact production-style profile.

Do not call a sweep complete until all completed, skipped, and failed cases are
recorded with enough metadata to reproduce the run.
