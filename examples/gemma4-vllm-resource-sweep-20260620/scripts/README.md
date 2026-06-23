# Scripts

Sweep and analysis scripts live here.

The scripts should use only the Python standard library unless a later commit
explicitly adds and documents dependencies.

`vllm_gemma_sweep.py` records:

- `tegrastats` samples once per candidate, from preflight through shutdown.
- `/proc/meminfo` snapshots during startup, idle, and load.
- systemd cgroup memory for the launched vLLM transient unit.
- `nvidia-smi` snapshots when available.

Run with `--disable-tegrastats` only when the NVIDIA utility is unavailable.
On DGX Spark/GB10, `tegrastats` plus `/proc/meminfo` is the most useful memory
view because GPU memory pressure is reflected in system RAM.
