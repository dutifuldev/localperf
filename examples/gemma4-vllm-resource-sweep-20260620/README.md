# Gemma 4 vLLM Resource Sweep

Created: 2026-06-20

This experiment measures how `nvidia/Gemma-4-26B-A4B-NVFP4` behaves under
different vLLM startup parameters on the local GB10 machine.

Primary question:

> What is the memory and throughput behavior across context window,
> concurrency, and batching configurations, from small 4k context up through
> high-context settings including about 100k tokens, without OOMing the device?

Current files:

- `docs/implementation-plan.md`: experiment plan and safety rules.
- `configs/`: sweep configuration files.
- `scripts/`: harness and analysis scripts.
- `results/`: machine-readable run outputs.
- `reports/`: generated plots, tables, and final reports.

The final report should include both raw measurement tables and fitted
relationships between parameters and resource use.
