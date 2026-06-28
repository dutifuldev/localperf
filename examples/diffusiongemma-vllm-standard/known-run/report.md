# vLLM Benchmark Report

Run directory: `examples/diffusiongemma-vllm-standard/known-run`

Generated: `2026-06-28T21:04:51Z`

## Event Summary

| Event | Count |
| --- | ---: |
| `before_profile` | 4 |
| `before_sleep` | 4 |
| `before_wake` | 4 |
| `before_warmup` | 4 |
| `before_workload` | 36 |
| `planned_run` | 36 |
| `profile_sleep` | 4 |
| `profile_wake_skipped` | 4 |
| `run_finish` | 1 |
| `run_start` | 1 |
| `server_ready` | 4 |
| `server_start` | 4 |
| `warmup_finish` | 4 |
| `workload_finish` | 36 |
| `workload_start` | 36 |

## Throughput

| Profile | Workload | Dataset | Context | Concurrency | Input | Output | Completed | Failed | Output tok/s | Per-user tok/s | Total tok/s | TTFT mean ms | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16k | decode-1k-out512-fixed | random | 16384 | 4 | 1024 | 512 | 64 | 0 | 266.1 | 66.5 | 805.3 | 6886.8 | `results/16k__decode-1k-out512-fixed__c4.json` |
| 16k | decode-1k-out512-fixed | random | 16384 | 8 | 1024 | 512 | 64 | 0 | 320.3 | 40.0 | 969.1 | 11306.6 | `results/16k__decode-1k-out512-fixed__c8.json` |
| 16k | decode-1k-out512-fixed | random | 16384 | 16 | 1024 | 512 | 64 | 0 | 329.2 | 20.6 | 996.3 | 21050.3 | `results/16k__decode-1k-out512-fixed__c16.json` |
| 16k | prefill-16k-out16-fixed | random | 16384 | 1 | 14336 | 16 | 64 | 0 | 1.9 | 1.9 | 1693.5 | 8482.6 | `results/16k__prefill-16k-out16-fixed__c1.json` |
| 16k | prefill-16k-out16-fixed | random | 16384 | 4 | 14336 | 16 | 64 | 0 | 2.1 | 0.5 | 1873.7 | 30471.4 | `results/16k__prefill-16k-out16-fixed__c4.json` |
| 16k | prefill-16k-out16-fixed | random | 16384 | 8 | 14336 | 16 | 64 | 0 | 2.0 | 0.3 | 1833.0 | 61820.4 | `results/16k__prefill-16k-out16-fixed__c8.json` |
| 16k | prefill-16k-out16-fixed | random | 16384 | 16 | 14336 | 16 | 64 | 0 | 1.7 | 0.1 | 1560.7 | 141780.7 | `results/16k__prefill-16k-out16-fixed__c16.json` |
| 16k | prefill-8k-out16-fixed | random | 16384 | 1 | 7168 | 16 | 64 | 0 | 3.5 | 3.5 | 1559.6 | 4614.5 | `results/16k__prefill-8k-out16-fixed__c1.json` |
| 16k | prefill-8k-out16-fixed | random | 16384 | 4 | 7168 | 16 | 64 | 0 | 4.0 | 1.0 | 1818.6 | 15629.9 | `results/16k__prefill-8k-out16-fixed__c4.json` |
| 16k | prefill-8k-out16-fixed | random | 16384 | 8 | 7168 | 16 | 64 | 0 | 4.2 | 0.5 | 1899.2 | 29750.1 | `results/16k__prefill-8k-out16-fixed__c8.json` |
| 16k | prefill-8k-out16-fixed | random | 16384 | 16 | 7168 | 16 | 64 | 0 | 3.9 | 0.2 | 1755.8 | 62766.8 | `results/16k__prefill-8k-out16-fixed__c16.json` |
| 32k | decode-1k-out512-fixed | random | 32768 | 4 | 1024 | 512 | 64 | 0 | 261.2 | 65.3 | 790.3 | 6974.4 | `results/32k__decode-1k-out512-fixed__c4.json` |
| 32k | decode-1k-out512-fixed | random | 32768 | 8 | 1024 | 512 | 64 | 0 | 295.6 | 37.0 | 894.6 | 12017.4 | `results/32k__decode-1k-out512-fixed__c8.json` |
| 32k | decode-1k-out512-fixed | random | 32768 | 16 | 1024 | 512 | 64 | 0 | 308.1 | 19.3 | 932.4 | 22792.9 | `results/32k__decode-1k-out512-fixed__c16.json` |
| 32k | prefill-16k-out16-fixed | random | 32768 | 1 | 14336 | 16 | 64 | 0 | 1.9 | 1.9 | 1690.4 | 8497.9 | `results/32k__prefill-16k-out16-fixed__c1.json` |
| 32k | prefill-16k-out16-fixed | random | 32768 | 4 | 14336 | 16 | 64 | 0 | 1.9 | 0.5 | 1711.8 | 33374.1 | `results/32k__prefill-16k-out16-fixed__c4.json` |
| 32k | prefill-16k-out16-fixed | random | 32768 | 8 | 14336 | 16 | 64 | 0 | 1.7 | 0.2 | 1525.0 | 74628.7 | `results/32k__prefill-16k-out16-fixed__c8.json` |
| 32k | prefill-16k-out16-fixed | random | 32768 | 16 | 14336 | 16 | 64 | 0 | 1.5 | 0.1 | 1373.6 | 163132.5 | `results/32k__prefill-16k-out16-fixed__c16.json` |
| 32k | prefill-32k-out16-fixed | random | 32768 | 1 | 28672 | 16 | 64 | 0 | 0.8 | 0.8 | 1378.4 | 20821.8 | `results/32k__prefill-32k-out16-fixed__c1.json` |
| 32k | prefill-32k-out16-fixed | random | 32768 | 4 | 28672 | 16 | 64 | 0 | 0.7 | 0.2 | 1320.1 | 86883.7 | `results/32k__prefill-32k-out16-fixed__c4.json` |
| 32k | prefill-32k-out16-fixed | random | 32768 | 8 | 28672 | 16 | 64 | 0 | 0.7 | 0.1 | 1276.4 | 178664.9 | `results/32k__prefill-32k-out16-fixed__c8.json` |
| 32k | prefill-32k-out16-fixed | random | 32768 | 16 | 28672 | 16 | 64 | 0 | 0.7 | 0.0 | 1259.3 | 349836.4 | `results/32k__prefill-32k-out16-fixed__c16.json` |
| 32k | prefill-8k-out16-fixed | random | 32768 | 1 | 7168 | 16 | 64 | 0 | 3.5 | 3.5 | 1565.7 | 4596.7 | `results/32k__prefill-8k-out16-fixed__c1.json` |
| 32k | prefill-8k-out16-fixed | random | 32768 | 4 | 7168 | 16 | 64 | 0 | 4.0 | 1.0 | 1807.8 | 15788.8 | `results/32k__prefill-8k-out16-fixed__c4.json` |
| 32k | prefill-8k-out16-fixed | random | 32768 | 8 | 7168 | 16 | 64 | 0 | 3.6 | 0.5 | 1628.8 | 34672.2 | `results/32k__prefill-8k-out16-fixed__c8.json` |
| 32k | prefill-8k-out16-fixed | random | 32768 | 16 | 7168 | 16 | 64 | 0 | 2.7 | 0.2 | 1213.4 | 91628.7 | `results/32k__prefill-8k-out16-fixed__c16.json` |
| 4k-reference | claim-repro-1k-out1024 | random | 4096 | 1 | 1000 | 1024 | 20 | 0 | 333.4 | 333.4 | 663.3 | 2315.9 | `results/4k-reference__claim-repro-1k-out1024__c1.json` |
| 4k-reference | claim-repro-1k-out1024 | random | 4096 | 4 | 1000 | 1024 | 20 | 0 | 423.1 | 105.8 | 841.8 | 7157.7 | `results/4k-reference__claim-repro-1k-out1024__c4.json` |
| 4k-reference | claim-repro-1k-out1024 | random | 4096 | 20 | 1000 | 1024 | 20 | 0 | 549.9 | 27.5 | 1094.1 | 24475.7 | `results/4k-reference__claim-repro-1k-out1024__c20.json` |
| 8k | decode-1k-out512-fixed | random | 8192 | 4 | 1024 | 512 | 64 | 0 | 271.4 | 67.8 | 821.3 | 6751.5 | `results/8k__decode-1k-out512-fixed__c4.json` |
| 8k | decode-1k-out512-fixed | random | 8192 | 8 | 1024 | 512 | 64 | 0 | 321.0 | 40.1 | 971.4 | 11153.7 | `results/8k__decode-1k-out512-fixed__c8.json` |
| 8k | decode-1k-out512-fixed | random | 8192 | 16 | 1024 | 512 | 64 | 0 | 334.6 | 20.9 | 1012.6 | 20526.2 | `results/8k__decode-1k-out512-fixed__c16.json` |
| 8k | prefill-8k-out16-fixed | random | 8192 | 1 | 7168 | 16 | 64 | 0 | 3.6 | 3.6 | 1625.1 | 4428.6 | `results/8k__prefill-8k-out16-fixed__c1.json` |
| 8k | prefill-8k-out16-fixed | random | 8192 | 4 | 7168 | 16 | 64 | 0 | 4.1 | 1.0 | 1834.3 | 15596.4 | `results/8k__prefill-8k-out16-fixed__c4.json` |
| 8k | prefill-8k-out16-fixed | random | 8192 | 8 | 7168 | 16 | 64 | 0 | 4.2 | 0.5 | 1881.4 | 29747.2 | `results/8k__prefill-8k-out16-fixed__c8.json` |
| 8k | prefill-8k-out16-fixed | random | 8192 | 16 | 7168 | 16 | 64 | 0 | 4.5 | 0.3 | 2009.8 | 54787.6 | `results/8k__prefill-8k-out16-fixed__c16.json` |
