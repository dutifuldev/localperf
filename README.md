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

## Direction

The goal is a reusable local model performance characterization harness:

- run parameter sweeps over context, concurrency, batching, and backend flags,
- avoid OOM by using guardrails and staged sampling,
- capture machine-readable telemetry,
- generate human-readable reports,
- compare local serving configurations scientifically.
