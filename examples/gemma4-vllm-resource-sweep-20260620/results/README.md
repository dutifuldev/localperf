# Results

Machine-readable sweep outputs live here.

Committed files:

- `sweep-candidates.jsonl`

Raw run outputs are intentionally not committed because they can contain local
machine paths, service process details, and verbose runtime logs.

Local-only generated files:

- `sweep-results.jsonl`
- `sweep-summary.json`
- `events.jsonl`
- timestamped `*-results.jsonl`, `*-summary.json`, and `*-events.jsonl`
- `logs/`
- `profiles/`

Each result row records several memory signals because DGX Spark/GB10 uses
unified memory and no single tool tells the whole story:

- `telemetry.tegrastats`: total machine RAM used, swap, CPU, and temperature.
- `system_memory`: `/proc/meminfo`, mainly `MemAvailable` before/after load.
- `service`: systemd cgroup accounting for the vLLM process tree.
- `gpu`: `nvidia-smi` telemetry when that tool exposes a field on this device.

For total pressure on the machine, prefer `tegrastats` RAM delta and the
`MemAvailable` drop. The cgroup values are still useful, but they undercount
unified GPU memory on this setup.
