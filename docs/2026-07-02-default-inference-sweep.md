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
  and `64k`, with concurrency `1`, `4`, `8`, `16`, and `32` where the
  hardware can safely run them. `128k` is opt-in and capped at `c4`; at that
  KV budget, higher concurrency is an hours-long stress exercise, not a
  default grid point.

The ladder points mean active context, not server capacity: a `32k` point must
actually move ~32k tokens per request through the KV cache. Setting
`max_model_len=32768` while requesting a ~1k prompt is a capacity experiment,
not a 32k-context data point; see `2026-07-02-context-semantics.md` for the
contract and validation rules.

Each active-context point `N` includes both workload shapes, with input
lengths that track `N`:

- `prefill`: long prompt, minimal output, measured through TTFT. Default
  shape: `input = N - headroom`, `output = 1`. Keep output at 1 (at most a
  few tokens); anything larger spends run time decoding and pollutes the
  prefill numbers.
- `decode`: long prompt, `1024` generated tokens. Default shape:
  `output = min(1024, N/4)`, `input = N - output - headroom`. 1024 tokens is
  still hundreds of steady-state decode steps, and the shorter output keeps
  the measured active range close to `N`. A decode row sweeps an active
  context range from `input` up to `N` as output tokens accumulate; reports
  label it with that range per `2026-07-02-context-semantics.md`.

with `headroom = max(64, N/64)` to absorb chat template and tokenizer drift.
Prefill rows are reported as aggregate and per-user prefill tok/s; decode rows
as aggregate and per-user output tok/s.

Request counts scale with concurrency: `prompts_per_user` (default 2) gives
`num_prompts = max(8, 2 x concurrency)` per point, so `c1` points stop paying
for `c32`-sized sample counts. Do not go below the floor of 8 requests per
point; that trades hours for noise.

Long-output behavior is a stress preset, not the default:
`localperf sweep plan --stress` adds `4096`-token decode spot checks at
`32k c4` and `64k c1/c4` plus the `128k` points at `c1/c4`.

Default sweeps must be generated with `localperf sweep plan`, for example:

```sh
localperf sweep plan --model <model-id> \
  --contexts 8k,16k,32k,64k --concurrency 1,4,8,16,32 \
  --out spec.json
```

Hand-authored specs stay legal but must declare `context_target` and
`context_semantics` on every workload and pass the same validation.

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

Point every batch at the model-level artifact and the runner appends:

```sh
localperf bench run --spec spec.json --artifact runs/models/<model-slug>.sqlite ...
```

Existing per-run artifacts combine with:

```sh
localperf artifact merge --into runs/models/<model-slug>.sqlite runs/batch-1.sqlite runs/batch-2.sqlite
```

Re-running the same run directory replaces that run's rows; merging an
already-present run is skipped. Render the HTML from the shared SQLite
artifact after each batch; the report lists every run and aggregates repeated
points across runs with mean ± spread.

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

The runner enforces this automatically (`runner.adaptive`, on by default):
per profile and workload, concurrency runs ascending and higher points are
skipped, with the reason recorded, when throughput improves less than 10%
over the previous point, a point's TTFT p99 exceeds a configured ceiling
(`ttft_p99_ceiling_ms`, off unless set), the previous point failed, or the
concurrency exceeds 2x vLLM's reported maximum concurrency. Disable with
`"runner": {"adaptive": {"enabled": false}}`; negative thresholds disable
individual rules.

Long sweeps are crash-tolerant: the artifact is refreshed after every point
(run status `running` until the sweep finishes), so partial results render at
any time, and `bench run --resume` with the same `--run-dir` skips points
whose results already completed.

Do not call a sweep complete until all completed, skipped, and failed cases are
recorded with enough metadata to reproduce the run.
