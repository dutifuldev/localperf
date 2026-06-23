# Gemma 4 vLLM Single 16-Concurrency Rerun With tegrastats Check

Run status: `load_complete`.

## Configuration

- Model: `nvidia/Gemma-4-26B-A4B-NVFP4`
- Backend: vLLM
- vLLM version: `0.23.1rc1.dev49+ga7fdfeef7`
- Context window: `16384`
- Requested concurrency: `16`
- `--max-num-seqs`: `16`
- `--max-num-batched-tokens`: `8192`
- KV cache dtype: `fp8`
- MoE backend: `cutlass`
- GPU memory utilization target: `0.65`
- Text-only serving: enabled with `--language-model-only`
- FlashInfer autotune: disabled
- `ninja` available during retry: `True` (`1.11.1`)

## Telemetry Tools

- `tegrastats` available: `False`
- `tegrastats` samples recorded: `0`
- `nvtop` available: `True`
- `nvidia-smi` available: `True`
- `nvidia-smi` GPU memory counter available: `False`

`tegrastats` was requested for this rerun, but it was not installed on this machine. The run records that absence explicitly and falls back to system `MemAvailable`, cgroup memory, vLLM capacity logs, and `nvidia-smi` utilization/power samples.

## Startup Result

- Ready: `True`
- Time to OpenAI API readiness: `177.839` seconds
- vLLM model-loading line: `17.05` GiB and `93.506495` seconds
- Weight-loading line: `90.59` seconds

## Request Result

- Successes/errors: `16/0`
- Wall time: `2.182` seconds
- Completion tokens/s: `271.783`
- Total tokens/s: `7289.559`
- Latency p50/p95/max: `1.646` / `1.648` / `1.649` seconds

## Memory And Capacity Signals

- System `MemAvailable` drop: `80.664` GiB
- Minimum `MemAvailable`: `33.691` GiB
- Max cgroup `MemoryCurrent`: `10.745` GiB
- Max cgroup `MemoryPeak`: `10.821` GiB
- vLLM available KV cache: `57.85` GiB
- vLLM GPU KV cache tokens: `914200`
- vLLM reported max concurrency: `55.8` at `16384` tokens/request
- Max GPU utilization sample: `96.0`%
- Max memory-controller utilization sample: `0.0`%
- Max power draw sample: `42.25` W

## Which Number Is More Accurate?

For total machine pressure, use system `MemAvailable` drop: `80.664` GiB. That is the best available total-memory number from this rerun because `tegrastats` was not installed and `nvidia-smi` did not expose GPU memory counters.

For process accounting, use cgroup peak: `10.821` GiB. This is useful, but it is not total model memory on this unified-memory machine.

For fit/capacity, use vLLM's KV-cache report: `57.85` GiB, `914200` tokens, and `55.8`x at `16384` tokens/request.

## Failed Attempt Note

An earlier rerun attempt failed before readiness because `ninja` was missing. That was a startup dependency issue, not an OOM. After installing `ninja-build`, this run was retried with the same vLLM and request settings.

## Artifacts

- `record.json`: sanitized machine-readable run record.
- Raw vLLM log is not committed.
