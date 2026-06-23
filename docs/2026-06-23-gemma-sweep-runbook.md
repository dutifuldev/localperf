---
title: Gemma Sweep Runbook
author: Bob <dutifulbob@gmail.com>
date: 2026-06-23
---

# Gemma Sweep Runbook

This runbook tracks the active Gemma 4 NVFP4 vLLM resource sweep. The goal is
at least 100 recorded parameter samples with accurate memory and performance
telemetry, without OOMing the machine.

## Active Run

The active run was started on 2026-06-23 through a user systemd runner:

```text
runner unit = localperf-gemma-sweep-20260623T075153Z.service
results = examples/gemma4-vllm-resource-sweep-20260620/results/tegrastats-sweep-20260623T075153Z-results.jsonl
events = examples/gemma4-vllm-resource-sweep-20260620/results/tegrastats-sweep-20260623T075153Z-events.jsonl
runner log = examples/gemma4-vllm-resource-sweep-20260620/results/logs/tegrastats-sweep-20260623T075153Z.runner.log
```

The raw JSONL and logs are intentionally local-only because they contain machine
paths and service details. Commit sanitized reports, tables, plots, and summary
metadata after enough samples are recorded.

## Runtime Settings

The sweep uses `nvidia/Gemma-4-26B-A4B-NVFP4` with the working vLLM NVFP4 setup:

```text
--gpu-memory-utilization 0.65
--kv-cache-dtype fp8
--moe-backend cutlass
--language-model-only
--no-enable-flashinfer-autotune
--enable-prefix-caching
--reasoning-parser gemma4
--enable-auto-tool-choice
--tool-call-parser gemma4
```

The candidate matrix is:

```text
10 context windows * 7 max_num_seqs * 2 batch policies = 140 candidates
```

The batch policies are `small` and `match_context`. The run does not include the
older `wide` policy or a `gpu_memory_utilization` sweep.

The runner executes candidates in increasing estimated risk order. This means a
high context such as 98k can appear before every lower-context concurrency pair
has finished. Use the progress reporter and verifier for coverage, not visual
ordering in the event log.

## Memory Telemetry

Treat these memory signals as separate measurements:

- `telemetry.tegrastats`: primary total machine RAM/swap/temperature signal.
- `system_memory`: `/proc/meminfo`, especially `MemAvailable` drop.
- `service`: systemd cgroup memory for the vLLM process tree.
- `gpu`: `nvidia-smi` fields where GB10 exposes them.

On this GB10 machine, `nvidia-smi` reports `N/A` for memory fields and observed
`tegrastats` output does not include `GR3D`. Total pressure should therefore be
read from `tegrastats` RAM delta and `/proc/meminfo`, not from cgroup alone.

## Safety Rules

The active run uses these guardrails:

```text
MemoryMax = 95 GiB
minimum available system memory = 12 GiB
minimum free swap = 4 GiB
startup timeout = 15 minutes
load timeout = 5 minutes
```

High-risk candidates may start and record idle capacity, but request load is
skipped unless the capacity and memory guards allow it. Startup-only rows still
count as recorded parameter samples if they include telemetry and a clear
reason.

Treat `startup_only` as a boundary measurement. It should not be rerun with
weaker guards just to force a load row unless the user explicitly approves a
riskier experiment.

## Monitor Commands

Use these commands to inspect progress:

```sh
systemctl --user status localperf-gemma-sweep-20260623T075153Z
systemctl --user list-units 'localperf-gemma-sweep-*' 'gemma4-vllm-sweep-*' --no-pager --plain
tail -f examples/gemma4-vllm-resource-sweep-20260620/results/tegrastats-sweep-20260623T075153Z-events.jsonl
```

Count rows and inspect the latest records:

```sh
go run ./cmd/localperf-sweep-progress \
  --results examples/gemma4-vllm-resource-sweep-20260620/results/tegrastats-sweep-20260623T075153Z-results.jsonl \
  --target-rows 100
```

For lower-level inspection:

```sh
python3 - <<'PY'
import collections
import json
from pathlib import Path

path = Path("examples/gemma4-vllm-resource-sweep-20260620/results/tegrastats-sweep-20260623T075153Z-results.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
print("rows", len(rows))
print("statuses", dict(collections.Counter(row.get("status") for row in rows)))
for row in rows[-10:]:
    tegrastats = (row.get("telemetry") or {}).get("tegrastats") or {}
    load = row.get("load_short_decode") or {}
    candidate = row.get("candidate") or {}
    print(
        row.get("candidate_id"),
        row.get("status"),
        "ctx", candidate.get("max_model_len"),
        "seq", candidate.get("max_num_seqs"),
        "samples", tegrastats.get("sample_count"),
        "ram_delta_gib", tegrastats.get("ram_used_delta_gib"),
        "completion_tok_s", load.get("completion_tokens_per_second"),
    )
PY
```

Run the mechanical completion check:

```sh
go run ./cmd/localperf-sweep-check \
  --results examples/gemma4-vllm-resource-sweep-20260620/results/tegrastats-sweep-20260623T075153Z-results.jsonl \
  --min-rows 100 \
  --require-context 100000 \
  --require-max-context 262144 \
  --require-max-seqs 32
```

That command should fail while the sweep is still below 100 rows. It should pass
before the final report is treated as complete.

Passing the command proves the minimum acceptable sweep coverage. It does not by
itself mean the run should be stopped; if the 140-candidate runner is still
healthy, let it continue and generate the final report from the most complete
safe dataset.

## Completion Checklist

Do not call the sweep complete until all of these are proven from current files
and command output:

- At least 100 candidate rows are recorded.
- Every recorded row has candidate parameters, status, startup/shutdown data,
  and telemetry metadata.
- Rows with `tegrastats_available=true` have parsed `tegrastats` samples.
- Request-load rows include successes/errors, throughput, latency, and memory
  monitor data.
- Startup-only or skipped-load rows include an explicit reason.
- No machine OOM occurred.
- If fewer than 140 planned candidates are recorded, the report explains why the
  run stopped early.
- Sanitized CSV, SVG, HTML, Markdown, and model-fit outputs are generated.
- The final report states which memory metric is appropriate for each claim.
- Local checks pass or unrelated pre-existing failures are documented.
- Codex review finds no P0/P1 issues before final PR handoff.
