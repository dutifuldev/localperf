---
title: GB10 / DGX Spark NVFP4 Optimized Sweep Specs
author: Bob <dutifulbob@gmail.com>
date: 2026-07-07
---

# GB10 / DGX Spark NVFP4 Optimized Sweep Specs

Throughput-tuned default-sweep specs for four NVFP4 models on a GB10
(DGX Spark, aarch64, ~122 GiB unified memory), vLLM 0.24.0:

- `qwen27b.json` — `nvidia/Qwen3.6-27B-NVFP4` (hybrid linear-attention)
- `qwen35b.json` — `nvidia/Qwen3.6-35B-A3B-NVFP4` (hybrid linear-attention MoE)
- `gemma4-26b.json` — `nvidia/Gemma-4-26B-A4B-NVFP4`
- `gemma4-31b.json` — `nvidia/Gemma-4-31B-IT-NVFP4` (dense)

These are machine-specific: the `runner.vllm_command`, env
(`CUTE_DSL_ARCH=sm_121a`, etc.), and ports point at a particular runtime.
Edit them before use elsewhere. `PROFILES-old-vs-new.txt` records each
model's old serve command vs the tuned one.

## What was tuned (and why)

Measured against a public DGX Spark benchmark, the original hand-authored
specs served vLLM in a decode-hostile way. The tuned specs drop three flags
and raise one:

- **Drop `--mamba-cache-mode none`** — the Qwen3.6 models are hybrid
  linear-attention (only every 4th layer is full attention; the rest keep a
  recurrent SSM state). `none` disables caching that state, so every decode
  step recomputes it over the whole sequence and decode throughput falls with
  context. Default (cache on) keeps it flat.
- **Drop `--enforce-eager`** — enables CUDA graphs.
- **Drop `--no-async-scheduling`** — enables scheduler/compute overlap.
- **Raise `--max-num-batched-tokens` to 16384** — large-context prefill was
  step-throttled at 2048.

Result on qwen27b: aggregate output throughput at concurrency rose ~20–30%
(4k max-throughput reference c32: 157 → 186 tok/s; decode-8k c32: 61 → 80;
decode-16k c16: 31 → 40). The Gemma specs were already free of the bad flags.

## Notes on other levers (tested, not adopted here)

- **MTP / speculative decoding** (`--speculative-config '{"method":"mtp",
  "num_speculative_tokens":1}'`) roughly 1.6x's *single-stream* decode on the
  Qwen models (they ship an MTP head), but adds overhead and memory pressure
  under concurrency, so it is left off for max throughput. Enable it when
  single-request latency matters more than aggregate throughput.
- **flashinfer autotune + sampler** boot fine on sm_121 but gave no aggregate
  throughput gain here, so they stay disabled.

## Usage

```sh
localperf bench run --spec qwen27b.json \
  --run-dir runs/qwen27b --artifact runs/qwen27b.sqlite --timeout 6h
localperf artifact render runs/qwen27b.sqlite
```
