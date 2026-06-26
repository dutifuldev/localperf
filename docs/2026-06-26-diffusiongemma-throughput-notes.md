---
title: DiffusionGemma Throughput Notes
author: Bob <dutifulbob@gmail.com>
date: 2026-06-26
---

# DiffusionGemma Throughput Notes - 2026-06-26

Model: `nvidia/diffusiongemma-26B-A4B-it-NVFP4`

Common serving stack:

- vLLM
- Attention backend: `TRITON_ATTN`
- NVFP4 MoE backend: `VLLM_CUTLASS`
- `VLLM_USE_V2_MODEL_RUNNER=1`
- `FLASHINFER_DISABLE_VERSION_CHECK=1`
- `CUTE_DSL_ARCH=sm_121a`

## Results

| Case | Context | Workers | Prompt | Output | Thinking | EOS | Server cap | Aggregate tok/s | Per-user tok/s |
| --- | ---: | ---: | --- | ---: | --- | --- | ---: | ---: | ---: |
| 300 tok/s repro | 4k | 1 | 1k tokens | 1024 | off | `ignore_eos=true` | 4 seqs | 311.4 | 311.4 |
| 20-user claim attempt, cold-ish | 4k | 20 | 1k tokens | 1024 | off | `ignore_eos=true` | 20 seqs | 479.4 | 24.0 |
| 20-user claim attempt, warm | 4k | 20 | 1k tokens | 1024 | off | `ignore_eos=true` | 20 seqs | 557.1 | 27.9 |
| LocalPerf runner smoke | 4k | 1 | 1k tokens | 1024 | off | `ignore_eos=true` | 20 seqs | 334.1 | 334.1 |
| 8k grid | 8k | 4 | short prompt | 512 | on | normal EOS | 16 seqs | 100.4 | 25.1 |
| 8k grid | 8k | 8 | short prompt | 512 | on | normal EOS | 16 seqs | 107.5 | 13.4 |
| 8k grid | 8k | 16 | short prompt | 512 | on | normal EOS | 16 seqs | 119.4 | 7.5 |

## Exact Settings

### 300 tok/s repro

Server:

- `--max-model-len 4096`
- `--max-num-seqs 4`
- `--max-num-batched-tokens 4096`
- `--gpu-memory-utilization 0.30`
- `--attention-backend TRITON_ATTN`
- `--moe-backend cutlass`
- `--default-chat-template-kwargs '{"enable_thinking": false}'`

Benchmark:

- `vllm bench serve`
- `--backend openai-chat`
- `--dataset-name random`
- `--random-input-len 1000`
- `--random-output-len 1024`
- `--num-prompts 4`
- `--max-concurrency 1`
- `--request-rate inf`
- `--ignore-eos`
- `--temperature 0`

Saved result:

- `results/diffusiongemma-vllmbench-thinkoff-c1-r4-random1000-out1024.json`

### 20-user claim attempt

Server:

- `--max-model-len 4096`
- `--max-num-seqs 20`
- `--max-num-batched-tokens 8192`
- `--gpu-memory-utilization 0.30`
- `--attention-backend TRITON_ATTN`
- `--moe-backend cutlass`
- `--default-chat-template-kwargs '{"enable_thinking": false}'`

Benchmark:

- `vllm bench serve`
- `--backend openai-chat`
- `--dataset-name random`
- `--random-input-len 1000`
- `--random-output-len 1024`
- `--num-prompts 20`
- `--max-concurrency 20`
- `--request-rate inf`
- `--ignore-eos`
- `--temperature 0`

Saved results:

- `results/diffusiongemma-vllmbench-thinkoff-c20-r20-random1000-out1024-gpu030.json`
- `results/diffusiongemma-vllmbench-thinkoff-c20-r20-random1000-out1024-gpu030-warm.json`

Claim comparison:

- Claimed: `165 tok/s/user` at `20` users, about `3,300 aggregate tok/s`.
- Reproduced here: best `557.1 aggregate tok/s`, about `27.9 tok/s/user`.
- Conclusion: we did not reproduce the 20-user claim.

### LocalPerf runner smoke

Server:

- `--max-model-len 4096`
- `--max-num-seqs 20`
- `--max-num-batched-tokens 8192`
- `--gpu-memory-utilization 0.30`
- `--attention-backend TRITON_ATTN`
- `--moe-backend cutlass`
- `--default-chat-template-kwargs '{"enable_thinking": false}'`
- `--enable-sleep-mode`

Benchmark:

- `go run ./cmd/localperf-vllm-bench run`
- Spec: `examples/diffusiongemma-vllm-standard/spec.json`
- Filter: `--profile 4k-reference --workload claim-repro-1k-out1024 --concurrency 1`
- vLLM command override: `/home/bob/scratch/vllm-latest-dgxspark-20260626/.venv/bin/vllm`
- Memory floor: `40 GiB MemAvailable`
- Warmup: random 256 input / 16 output, 4 prompts, concurrency 1
- Measured workload: random 1000 input / 1024 output, 20 prompts, concurrency 1

Saved report:

- `runs/diffusiongemma-localperf-c1-endpoint-20260626T1330Z/report.md`

Result:

- Completed `20 / 20` requests with `0` failures.
- Aggregate output throughput: `334.1 tok/s`.
- Total token throughput: `664.8 tok/s`.
- Mean TTFT: `2311.6 ms`.
- Server startup reached a low observed pre-workload memory state of about
  `45.8 GiB MemAvailable`, above the configured `40 GiB` floor.
- The runner slept and stopped the vLLM process after the run; the machine did
  not OOM.

### 8k grid

Server:

- `--max-model-len 8192`
- `--max-num-seqs 16`
- `--max-num-batched-tokens 8192`
- `--gpu-memory-utilization 0.35`
- `--attention-backend TRITON_ATTN`
- `--moe-backend cutlass`
- `--default-chat-template-kwargs '{"enable_thinking": true}'`
- vLLM reported `24.70x` full-context concurrency for 8k.

Benchmark:

- Custom `benchmark_profile.mjs`
- OpenAI chat completions endpoint
- Short prompt: `Write plain lowercase words separated by spaces...`
- `max_tokens=512`
- Normal EOS behavior
- Worker counts: `4, 8, 16`

Saved result:

- `results/grid-l8192-s16-gpu035-c4-8-16-t512.jsonl`

## Notes

- The `300 tok/s` repro is a benchmark-only setup because `ignore_eos=true` forces output even after the model wants to stop.
- The 8k grid is not a full 8k-prompt prefill benchmark. It measures decode throughput with an 8k context window configured.
- The interrupted 16k run did not produce valid 16k throughput numbers.
