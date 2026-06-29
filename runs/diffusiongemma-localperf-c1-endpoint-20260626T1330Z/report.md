# vLLM Benchmark Report

Run directory: `runs/diffusiongemma-localperf-c1-endpoint-20260626T1330Z`

Generated: `2026-06-26T13:32:34Z`

## Event Summary

| Event | Count |
| --- | ---: |
| `workload_finish` | 1 |

## Throughput

| Profile | Workload | Dataset | Context | Concurrency | Input | Output | Completed | Failed | Output tok/s | Per-user tok/s | Total tok/s | TTFT mean ms | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4k-reference | claim-repro-1k-out1024 | random | 4096 | 1 | 1000 | 1024 | 20 | 0 | 334.1 | 334.1 | 664.8 | 2311.6 | `results/4k-reference__claim-repro-1k-out1024__c1.json` |
