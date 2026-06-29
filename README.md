# LocalPerf

LocalPerf is a local inference performance benchmarking and profiling workspace.

The current repo starts from the Gemma 4 vLLM resource sweep we ran on the local
GB10 machine. It records vLLM startup settings, context windows, concurrency,
throughput, latency, capacity, and memory-pressure signals.

## Current Snapshot

The first copied experiment is here:

```text
examples/gemma4-vllm-resource-sweep-20260620/
```

It includes:

- the sweep harness,
- candidate definitions,
- generated CSV tables,
- generated SVG plots,
- an interactive HTML report,
- implementation notes.

Raw per-run JSONL, event streams, env profiles, and vLLM logs are not committed
because they can contain local machine paths and verbose service details.

Open the report locally with:

```sh
python3 -m http.server 8766 --directory examples/gemma4-vllm-resource-sweep-20260620/reports
```

Then visit:

```text
http://127.0.0.1:8766/
```

## Measurement Caveat

On the GB10 unified-memory setup, one memory number is not enough.

The copied sweep currently records:

- Linux cgroup memory from `systemctl show ... MemoryCurrent MemoryPeak`,
- system memory from `/proc/meminfo`,
- vLLM-reported KV cache and max concurrency from vLLM logs,
- throughput and latency from OpenAI-compatible requests.

The cgroup memory columns in the generated report are process/cgroup memory,
not total model memory. For total machine pressure on unified memory, use:

```text
MemAvailable before startup - lowest MemAvailable during startup/load
```

Future LocalPerf runs should record and report these as separate signals:

- process/cgroup memory,
- whole-machine `MemAvailable` drop,
- vLLM KV cache capacity,
- GPU telemetry from `nvtop`, NVML, DCGM, or the best available GB10 source.

See [Measurement Methods](docs/2026-06-23-measurement-methods.md) for the full
recording and reporting policy.

## Direction

The goal is a reusable local model performance characterization harness:

- run parameter sweeps over context, concurrency, batching, and backend flags,
- avoid OOM by using guardrails and staged sampling,
- capture machine-readable telemetry,
- generate human-readable reports,
- compare local serving configurations scientifically.

## Standard Benchmarks

Use the reusable benchmark runner for repeatable specs, warmups, memory
guardrails, raw result capture, SQLite artifacts, and Markdown/JSON/CSV
reports:

```sh
go run ./cmd/localperf bench plan \
  --spec examples/diffusiongemma-vllm-standard/spec.json

go run ./cmd/localperf bench run \
  --spec examples/diffusiongemma-vllm-standard/spec.json \
  --timeout 4h

go run ./cmd/localperf bench report \
  --run-dir runs/<run-id>

go run ./cmd/localperf artifact check \
  runs/<run-id>.sqlite
```

The run command writes a canonical `runs/<run-id>.sqlite` artifact and keeps
the run directory for raw local files. The report command writes `report.md`,
`report.json`, and `report.csv` exports by default. Use the SQLite artifact as
the source of truth and the CSV for plotting or spreadsheet workflows.

The legacy `go run ./cmd/localperf-vllm-bench ...` command still works as a
compatibility wrapper around the same implementation.

The DiffusionGemma NVFP4 example and completed 36-case known-results fixture
live under `examples/diffusiongemma-vllm-standard/`. The benchmark policy is
documented in
[Standard vLLM Benchmarking](docs/2026-06-26-standard-vllm-benchmarking.md).
