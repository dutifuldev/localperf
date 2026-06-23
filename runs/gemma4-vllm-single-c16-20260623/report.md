# Gemma 4 vLLM Single 16-Concurrency Run

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

The vLLM server was started for this one run, tested, and stopped afterward.

## Startup Result

- Ready: `true`
- Time to OpenAI API readiness: `155.996` seconds
- vLLM model-loading line: `17.05` GiB and `93.798497` seconds
- Weight-loading line: `90.78` seconds

## Request Result

- Successes/errors: `16/0`
- Wall time: `4.333` seconds
- Completion tokens/s: `236.304`
- Total tokens/s: `6900.803`
- Latency p50/p95/max: `3.937` / `3.939` / `3.939` seconds

## Memory And Capacity Signals

- System `MemAvailable` drop: `79.559` GiB
- Minimum `MemAvailable`: `33.861` GiB
- Max cgroup `MemoryCurrent`: `11.262` GiB
- Max cgroup `MemoryPeak`: `11.266` GiB
- vLLM available KV cache: `58.21` GiB
- vLLM GPU KV cache tokens: `919891`
- vLLM reported max concurrency: `56.15` at `16384` tokens/request
- GPU memory counter available from `nvidia-smi`: `False`
- Max GPU utilization sample: `96.0`%
- Max memory-controller utilization sample: `0.0`%
- Max power draw sample: `42.69` W

## Which Memory Number Is More Accurate?

For the question "how much total memory pressure did this run put on the machine?", the system `MemAvailable` drop is the most accurate number in this report: `79.559` GiB. This GB10 setup uses unified memory, and `nvidia-smi` did not expose normal GPU-memory counters.

For the question "how much memory did the vLLM process account for in systemd?", use the cgroup peak: `11.266` GiB. That number is real, but it under-reports total model/unified-memory pressure here, so it should not be described as total model memory.

For the question "can this context/concurrency fit?", use vLLM's KV-cache capacity report: `58.21` GiB available KV cache, `919891` KV tokens, and `56.15x` reported max concurrency at `16384` tokens/request.

`nvidia-smi` did not expose GPU memory counters on this GB10 run (`[N/A]`), so GPU memory telemetry is not the source of truth for this report.

## Artifacts

- `record.json`: sanitized machine-readable run record.
- Raw vLLM log is not committed.
