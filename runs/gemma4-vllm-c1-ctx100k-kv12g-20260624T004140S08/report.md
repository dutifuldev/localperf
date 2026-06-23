# Gemma 4 vLLM 100k Context c1 KV12G Run

Run status: `load_complete_and_95k_prompt_succeeded`.

## Configuration

- Model: `nvidia/Gemma-4-26B-A4B-NVFP4`
- Backend: vLLM `0.23.1rc1.dev49+ga7fdfeef7`
- Context window: `100000`
- Requested concurrency: `1`
- `--max-num-seqs`: `1`
- `--max-num-batched-tokens`: `100000`
- KV cache dtype: `fp8`
- Explicit KV cache memory: `12G`
- GPU memory utilization target: `0.35`
- MoE backend: `cutlass`
- Attention backend: `TRITON_ATTN`
- Text-only serving: enabled with `--language-model-only`
- FlashInfer autotune: disabled

## Startup Result

- Service start: `2026-06-24T00:41:40+08:00`
- API ready: `2026-06-24T00:44:44+08:00`
- Time to API readiness: `184` seconds
- Weight loading: `90.74` seconds
- Model loading: `17.05` GiB and `93.682655` seconds
- Torch compile: `35.49` seconds
- Initial profiling/warmup: `5.98` seconds
- Engine init: `58.20` seconds
- Graph capture: `3` seconds, reported `-0.73` GiB

## vLLM Capacity

vLLM accepted the explicit KV setting:

```text
reserved 12.0 GiB memory for KV Cache as specified by kv_cache_memory_bytes
GPU KV cache size: 114,371 tokens
Maximum concurrency for 100,000 tokens per request: 1.14x
```

This means `12G` fp8 KV is viable for one 100k-token request, with about 14% headroom by vLLM's own capacity calculation.

## Request Checks

- Small ready check: `18` prompt tokens, `2` completion tokens, response `ready`.
- First large prompt attempt: `45,013` prompt tokens, `1` completion token, `22.076` seconds.
- Main large prompt check: `95,013` prompt tokens, `1` completion token, `60.794` seconds.

The main 95k-token request succeeded without OOM.

## Memory

- Unloaded baseline via `tegrastats`: `8645` MB RAM used
- Loaded idle before large request via `tegrastats`: about `55859` MB RAM used
- Peak during 95k-token request via `tegrastats`: `68475` MB RAM used
- Peak above unloaded baseline: `59830` MB, about `58.43` GiB
- Total peak RAM used: `68475` MB, about `66.87` GiB
- `nvidia-smi` vLLM engine memory before large request: `42944` MiB
- `nvidia-smi` vLLM engine memory after large request: `55202` MiB
- Systemd cgroup memory current/peak after request: `4.8` GiB / `11.7` GiB

For this unified-memory GB10 machine, the best fit counter is `tegrastats` RAM used. `nvidia-smi` process memory is useful as a cross-check, but the top-level GPU memory counter is not supported. Systemd cgroup memory undercounts the model/KV memory pressure.

## Answer

For this exact setting:

```text
Gemma 4 26B A4B NVFP4
100k context
concurrency 1
fp8 KV cache
12G explicit KV
```

Measured memory need is about:

```text
68.5 GB total RAM peak
60 GB above unloaded baseline
75 GB available RAM recommended
```

## Artifacts

- `record.json`: machine-readable run record.
- `tegrastats-45k-prompt.log`: raw `tegrastats` samples for the 45k-token prompt attempt.
- `tegrastats-95k-prompt.log`: raw `tegrastats` samples for the 95k-token prompt check.
- `vllm-key-lines.log`: sanitized vLLM key log lines.
