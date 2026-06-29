# LocalPerf

LocalPerf benchmarks local LLM inference servers and keeps the evidence in one
portable run artifact.

It is currently focused on vLLM-managed runs, with an engine-neutral benchmark
spec and CLI shape:

```sh
go run ./cmd/localperf bench plan --spec examples/diffusiongemma-vllm-standard/spec.json
go run ./cmd/localperf bench run  --spec examples/diffusiongemma-vllm-standard/spec.json
go run ./cmd/localperf artifact check runs/<run-id>.sqlite
```

The legacy command still works:

```sh
go run ./cmd/localperf-vllm-bench ...
```

## Requirements

- Go 1.26 or newer.
- `sqlite3` if you want to inspect artifacts from the shell.
- vLLM installed and available as `vllm` for real managed benchmark runs.
- Enough available system memory for the model profile you run.

The included DiffusionGemma example targets
`nvidia/diffusiongemma-26B-A4B-it-NVFP4` on a GB10/DGX Spark-class local
machine. Edit the spec before using it on a different machine or model.

## Quick Start

Preview the planned runs without starting a model:

```sh
go run ./cmd/localperf bench plan \
  --spec examples/diffusiongemma-vllm-standard/spec.json
```

Run one dry benchmark case and write the reports/artifact:

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

Run the full spec only when the machine is ready for it:

```sh
go run ./cmd/localperf bench run \
  --spec examples/diffusiongemma-vllm-standard/spec.json \
  --timeout 4h
```

Generate reports again from an existing run directory:

```sh
go run ./cmd/localperf bench report --run-dir runs/<run-id>
```

## Outputs

Each run writes:

- `runs/<run-id>.sqlite`: canonical run artifact.
- `runs/<run-id>/events.jsonl`: lifecycle and diagnostic events.
- `runs/<run-id>/results/*.json`: raw benchmark results.
- `runs/<run-id>/logs/*.log`: server and benchmark logs.
- `runs/<run-id>/report.md`, `report.json`, `report.csv`: report exports.

Use the SQLite artifact as the source of truth. It stores the original and
normalized specs, engine/profile/workload definitions, measurements, metric
stats, events, commands, raw result artifacts, logs, and rendered reports.

Example inspection:

```sh
sqlite3 runs/<run-id>.sqlite \
  "select profile_id, workload_id, concurrency, status, aggregate_output_tok_s from measurements"
```

## Memory Safety

Specs include a `safety.min_mem_available_gib` floor. LocalPerf checks
`/proc/meminfo` before major steps and while subprocesses run. If available
memory drops below the floor, the current step is stopped and skipped/failed
rows are recorded.

On unified-memory systems, do not treat process/cgroup memory as total model
memory. For capacity planning, compare multiple signals:

- whole-machine `MemAvailable` drop,
- process/cgroup memory,
- vLLM KV-cache capacity lines,
- GPU or platform telemetry when available.

See [Measurement Methods](docs/2026-06-23-measurement-methods.md) for the
memory reporting policy.

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
