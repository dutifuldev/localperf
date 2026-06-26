# vLLM Benchmark Report

Run directory: `examples/diffusiongemma-vllm-standard/known-run`

Generated: `2026-06-26T13:13:58Z`

## Event Summary

| Event | Count |
| --- | ---: |
| `workload_finish` | 6 |

## Throughput

| Profile | Workload | Dataset | Context | Concurrency | Input | Output | Completed | Failed | Output tok/s | Per-user tok/s | Total tok/s | TTFT mean ms | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4k-reference | claim-repro-1k-out1024 | random | 4096 | 1 | 1000 | 1024 | 4 | 0 | 311.4 | 311.4 | 619.6 | 2597.3 | `results/4k-reference__claim-repro-1k-out1024__c1.json` |
| 4k-reference | claim-repro-1k-out1024 | random | 4096 | 4 | 1000 | 1024 | 20 | 0 | 450.1 | 112.5 | 895.4 | 6402.4 | `results/4k-reference__claim-repro-1k-out1024__c4.json` |
| 4k-reference | claim-repro-1k-out1024 | random | 4096 | 20 | 1000 | 1024 | 20 | 0 | 557.1 | 27.9 | 1108.5 | 23209.5 | `results/4k-reference__claim-repro-1k-out1024__c20.json` |
| 8k | decode-short-eos | custom-short-chat | 8192 | 4 | - | 512 | 4 | 0 | 100.4 | 25.1 | 107.8 | - | `results/8k__decode-short-eos__c4.json` |
| 8k | decode-short-eos | custom-short-chat | 8192 | 8 | - | 512 | 8 | 0 | 107.5 | 13.4 | 115.4 | - | `results/8k__decode-short-eos__c8.json` |
| 8k | decode-short-eos | custom-short-chat | 8192 | 16 | - | 512 | 16 | 0 | 119.4 | 7.5 | 128.4 | - | `results/8k__decode-short-eos__c16.json` |
