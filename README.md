# localperf

localperf is a local LLM inference benchmark CLI.
It runs benchmark plans against local inference servers, collects all evidence
in one portable SQLite artifact per model, and renders reports that only label
what the measurements actually confirm.

It is currently focused on vLLM-managed runs, with an engine-neutral benchmark
spec and CLI shape:

```sh
go run ./cmd/localperf sweep plan --model <model-id> --out spec.json
go run ./cmd/localperf bench run  --spec spec.json --artifact runs/models/<model-slug>.sqlite
go run ./cmd/localperf artifact render runs/models/<model-slug>.sqlite
```

## Requirements

- Go 1.26 or newer.
- vLLM installed and available as `vllm` for real managed benchmark runs.
- Enough available system memory for the model profile you run.
- `sqlite3` if you want to inspect artifacts from the shell.

The included DiffusionGemma example targets
`nvidia/diffusiongemma-26B-A4B-it-NVFP4` on a GB10/DGX Spark-class local
machine. Edit the spec before using it on a different machine or model.

## Quick Start

Generate the default context/concurrency sweep spec instead of hand-writing
the grid:

```sh
go run ./cmd/localperf sweep plan \
  --model nvidia/diffusiongemma-26B-A4B-it-NVFP4 \
  --contexts 8k,16k,32k --concurrency 1,4,8 \
  --out spec.json
```

Preview the planned runs without starting a model:

```sh
go run ./cmd/localperf bench plan --spec spec.json
```

Run one dry benchmark case and validate the artifact:

```sh
go run ./cmd/localperf bench run \
  --dry-run \
  --spec examples/diffusiongemma-vllm-standard/spec.json \
  --profile 4k-reference \
  --workload claim-repro-1k-out1024 \
  --concurrency 1 \
  --run-dir /tmp/localperf-onecase-dry

go run ./cmd/localperf artifact check /tmp/localperf-onecase-dry.sqlite
```

Run the full spec only when the machine is ready for it, pointing batches at
one model-level artifact:

```sh
go run ./cmd/localperf bench run --spec spec.json --timeout 4h \
  --artifact runs/models/<model-slug>.sqlite
```

Render the HTML report:

```sh
go run ./cmd/localperf artifact render runs/models/<model-slug>.sqlite
```

## Model-Level Artifacts

Keep every run of one model in a single SQLite file and render one HTML
report from it. Pointing `bench run --artifact` at an existing artifact
appends the new run; re-running the same run directory replaces that run.
Combine existing per-run artifacts with:

```sh
go run ./cmd/localperf artifact merge \
  --into runs/models/<model-slug>.sqlite runs/batch-1.sqlite runs/batch-2.sqlite
```

Merges are idempotent: runs already present are skipped, and a run id that
collides with different provenance is refused instead of silently replaced.
The report lists every run and aggregates repeated points across runs with
mean ± spread.

## Context Semantics

Every workload must declare what its context number means:

```json
"context_target": 32768,
"context_semantics": "active"
```

`"active"` claims the workload actually pushes ~N tokens through the KV cache
and is validated: the requested input+output must land within 90–100% of the
target, on the random dataset, with a fixed range ratio. `"capacity"` marks a
server-limit/concurrency point and must match the profile's `max_model_len`.
Specs that conflate the two are refused before any GPU time is spent, and the
report labels rows only by declared-and-measured active context or by the
measured token shape — never by `max_model_len` alone. See
[Context Semantics](docs/2026-07-02-context-semantics.md) for the contract.

Workloads may also declare latency targets for goodput:

```json
"slo": {"ttft_p95_ms": 500}
```

The report then shows the fraction of requests meeting the target and goodput
in requests per second.

## Outputs

Each run writes:

- the SQLite artifact (`--artifact` path, or `runs/<run-id>.sqlite`): the
  canonical record — specs, engine/profile/workload definitions, measurements,
  per-request rows, metric stats, GPU telemetry, hardware inventory, engine
  identity probes, events, commands, and logs.
- `runs/<run-id>/events.jsonl`, `results/*.json`, `logs/*.log`: raw run data.
- `runs/<run-id>/report.md`, `report.json`, `report.csv`: legacy exports; the
  HTML report rendered from the artifact is the authoritative view.

Example inspection:

```sh
sqlite3 runs/models/<model-slug>.sqlite \
  "select run_id, profile_id, workload_id, concurrency, status, aggregate_output_tok_s from measurements"
```

## Memory Safety

Specs include a `safety.min_mem_available_gib` floor. localperf checks
`/proc/meminfo` before major steps and while subprocesses run. If available
memory drops below the floor, the current step is stopped and skipped/failed
rows are recorded.

On unified-memory systems, do not treat process/cgroup memory as total model
memory. For capacity planning, compare multiple signals:

- whole-machine `MemAvailable` drop,
- process/cgroup memory,
- vLLM KV-cache capacity lines,
- GPU or platform telemetry when available.

localperf samples GPU utilization and memory during measurements from every
available source (`tegrastats`, `nvidia-smi`) and names the source in the
report. See [Measurement Methods](docs/2026-06-23-measurement-methods.md) for
the memory reporting policy.

## Example Data

The repo includes two useful examples:

- `examples/diffusiongemma-vllm-standard/`: a reusable vLLM benchmark spec plus
  a completed known-run fixture.
- `examples/gemma4-vllm-resource-sweep-20260620/`: an earlier Gemma 4 resource
  sweep with generated tables, plots, and an HTML report.

Open the Gemma 4 report locally:

```sh
python3 -m http.server 8766 --directory examples/gemma4-vllm-resource-sweep-20260620/reports
```

Then visit `http://127.0.0.1:8766/`.
