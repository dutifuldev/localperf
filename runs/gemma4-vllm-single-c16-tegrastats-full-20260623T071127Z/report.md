# Gemma 4 vLLM Single 16-Concurrency Full tegrastats Run

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
- Service memory cap: `95G`

## Telemetry Coverage

- `tegrastats` available: `True`
- `tegrastats` install source: NVIDIA Jetson Linux BSP R38.4.0 `nvidia-l4t-tools`, extracted binary only
- `tegrastats` samples recorded: `149`
- `tegrastats` parsed samples: `149`
- `tegrastats` GPU/GR3D field seen: `False`
- `nvtop` available: `True`
- `nvidia-smi` available: `True`
- `nvidia-smi` GPU memory counter available: `False`

## Startup Result

- Ready: `True`
- Time to OpenAI API readiness: `142.495` seconds
- vLLM model-loading line: `17.05` GiB and `94.464916` seconds
- Weight-loading line: `91.35` seconds
- Engine init line: `13.36` seconds
- CUTLASS MoE: `True`
- Triton attention: `True`
- FlashInfer sampling: `True`

## Request Result

- Successes/errors: `16/0`
- Wall time: `2.167` seconds
- Completion tokens/s: `269.453`
- Total tokens/s: `7334.279`
- Latency p50/p95/max: `1.717` / `1.81` / `1.811` seconds

## Memory And Capacity Signals

- System `MemAvailable` drop: `81.986` GiB
- Minimum `MemAvailable`: `32.535` GiB
- Max cgroup `MemoryCurrent`: `12.201` GiB
- Max cgroup `MemoryPeak`: `12.311` GiB
- `tegrastats` baseline RAM used: `9091` MB
- `tegrastats` max RAM used: `92004` MB
- `tegrastats` RAM used delta: `82913` MB (`80.97` GiB)
- `tegrastats` max swap used: `5173` MB
- `tegrastats` max temperature: `74.1` C
- vLLM available KV cache: `58.51` GiB
- vLLM GPU KV cache tokens: `924767`
- vLLM reported max concurrency: `56.44` at `16384` tokens/request
- Max GPU utilization sample from `nvidia-smi`: `96.0`%
- Max memory-controller utilization sample from `nvidia-smi`: `0.0`%
- Max power draw sample from `nvidia-smi`: `40.51` W

## Which Number Is More Accurate?

For total machine memory pressure, the best cross-check is now system `MemAvailable` drop plus `tegrastats` RAM used delta. They should be read together: system `MemAvailable` drop was `81.986` GiB, while `tegrastats` RAM used delta was `82913` MB (`80.97` GiB).

For process accounting, use cgroup peak: `12.311` GiB. It is not total model memory on this unified-memory machine.

For fit/capacity, use vLLM's KV-cache report: `58.51` GiB, `924767` tokens, and `56.44`x at `16384` tokens/request.

`nvidia-smi` still did not expose GPU memory counters on this GB10 run.

## Artifacts

- `record.json`: sanitized machine-readable run record with raw and parsed `tegrastats` samples.
- Raw vLLM journal log is not committed because it contains local runtime paths.
