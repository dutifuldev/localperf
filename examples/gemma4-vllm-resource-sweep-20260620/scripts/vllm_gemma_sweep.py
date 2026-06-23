#!/usr/bin/env python3
"""Run a safety-first vLLM resource sweep for Gemma 4 NVFP4.

The script intentionally avoids third-party dependencies.  It writes JSONL so
partial sweeps can be resumed and audited.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "nvidia/Gemma-4-26B-A4B-NVFP4"
DEFAULT_WORKDIR = Path(os.environ.get("LOCALPERF_VLLM_WORKDIR", ".")).expanduser()
DEFAULT_VENV = Path(os.environ.get("LOCALPERF_VLLM_VENV", DEFAULT_WORKDIR / ".venv")).expanduser()
DEFAULT_VLLM_BIN = Path(os.environ.get("LOCALPERF_VLLM_BIN", DEFAULT_VENV / "bin/vllm")).expanduser()
CONFLICTING_SERVICES = [
    "localpager-worker.service",
    "localpager-vllm-qwen36-nvfp4.service",
    "localpager-vllm-gemma4-26b-a4b-nvfp4.service",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write the candidate matrix")
    gen.add_argument("--out", default=str(ROOT / "results/sweep-candidates.jsonl"))

    run = sub.add_parser("run", help="run candidates")
    run.add_argument("--candidates", default=str(ROOT / "results/sweep-candidates.jsonl"))
    run.add_argument("--results", default=str(ROOT / "results/sweep-results.jsonl"))
    run.add_argument("--events", default=str(ROOT / "results/events.jsonl"))
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--candidate-id", action="append", default=[])
    run.add_argument("--calibration", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--rerun", action="store_true")
    run.add_argument("--stop-conflicting-services", action="store_true")
    run.add_argument("--run-load", action="store_true")
    run.add_argument("--allow-risky-load", action="store_true")
    run.add_argument("--memory-max-gib", type=float, default=95.0)
    run.add_argument("--min-available-gib", type=float, default=12.0)
    run.add_argument("--min-swap-free-gib", type=float, default=4.0)
    run.add_argument("--startup-timeout-sec", type=int, default=900)
    run.add_argument("--load-timeout-sec", type=int, default=300)
    run.add_argument("--idle-sleep-sec", type=float, default=5.0)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8000)

    summary = sub.add_parser("summarize", help="write summary JSON from results")
    summary.add_argument("--results", default=str(ROOT / "results/sweep-results.jsonl"))
    summary.add_argument("--out", default=str(ROOT / "results/sweep-summary.json"))

    args = parser.parse_args()
    if args.command == "generate":
        return generate_command(args)
    if args.command == "run":
        return run_command(args)
    if args.command == "summarize":
        return summarize_command(args)
    raise AssertionError(args.command)


def generate_command(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    candidates = generate_candidates()
    write_jsonl(out, candidates)
    print(f"wrote {len(candidates)} candidates to {out}")
    return 0


def generate_candidates() -> list[dict]:
    contexts = [4096, 8192, 16384, 32768, 65536, 98304, 100000, 131072, 196608, 262144]
    seqs = [1, 2, 4, 8, 16, 24, 32]
    policies = ["small", "match_context"]
    candidates: list[dict] = []
    index = 1
    for context in contexts:
        for max_num_seqs in seqs:
            for policy in policies:
                max_num_batched_tokens = batch_tokens(context, max_num_seqs, policy)
                token_budget = context * max_num_seqs
                candidates.append(
                    {
                        "candidate_id": f"ctx{context}-seq{max_num_seqs}-{policy}",
                        "index": index,
                        "model": DEFAULT_MODEL,
                        "max_model_len": context,
                        "max_num_seqs": max_num_seqs,
                        "max_num_batched_tokens": max_num_batched_tokens,
                        "batch_policy": policy,
                        "gpu_memory_utilization": 0.65,
                        "kv_cache_dtype": "fp8",
                        "moe_backend": "cutlass",
                        "language_model_only": True,
                        "disable_flashinfer_autotune": True,
                        "load_concurrency": max_num_seqs,
                        "load_prompt_words": 256,
                        "load_max_tokens": 64,
                        "risk_token_budget": token_budget,
                        "risk_tier": risk_tier(context, max_num_seqs),
                        "calibration": (context, max_num_seqs, policy)
                        in {
                            (4096, 1, "small"),
                            (16384, 4, "small"),
                            (32768, 8, "small"),
                            (32768, 16, "match_context"),
                            (100000, 1, "match_context"),
                        },
                    }
                )
                index += 1
    candidates.sort(key=lambda row: (row["risk_token_budget"], row["max_model_len"], row["max_num_seqs"], row["batch_policy"]))
    for index, row in enumerate(candidates, start=1):
        row["index"] = index
    return candidates


def batch_tokens(context: int, seqs: int, policy: str) -> int:
    if policy == "small":
        return min(max(context, 4096), 8192)
    if policy == "match_context":
        return context
    if policy == "wide":
        return min(context * max(1, min(seqs, 4)), 65536)
    raise ValueError(f"unknown batch policy {policy}")


def risk_tier(context: int, seqs: int) -> str:
    if context >= 196608 or (context >= 131072 and seqs >= 8) or (context >= 100000 and seqs >= 16):
        return "extreme"
    if context >= 100000 or seqs >= 24 or context * seqs >= 1_000_000:
        return "high"
    if context >= 32768 or seqs >= 8:
        return "medium"
    return "low"


def run_command(args: argparse.Namespace) -> int:
    candidates = read_jsonl(Path(args.candidates))
    if args.calibration:
        candidates = [row for row in candidates if row.get("calibration")]
    if args.candidate_id:
        wanted = set(args.candidate_id)
        candidates = [row for row in candidates if row["candidate_id"] in wanted]
    if args.limit:
        candidates = candidates[: args.limit]

    results_path = Path(args.results)
    events_path = Path(args.events)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    if results_path.exists() and not args.rerun:
        existing = {row.get("candidate_id") for row in read_jsonl(results_path)}

    for candidate in candidates:
        if candidate["candidate_id"] in existing:
            event(events_path, "skip_existing", candidate_id=candidate["candidate_id"])
            continue
        result = run_candidate(candidate, args, events_path)
        append_jsonl(results_path, result)
        summarize_results(results_path, summary_path_for(results_path))
    return 0


def run_candidate(candidate: dict, args: argparse.Namespace, events_path: Path) -> dict:
    started_at = now()
    candidate_id = candidate["candidate_id"]
    event(events_path, "candidate_start", candidate_id=candidate_id, dry_run=args.dry_run)
    result = {
        "candidate_id": candidate_id,
        "started_at": started_at,
        "finished_at": None,
        "candidate": candidate,
        "status": "unknown",
        "preflight": {
            "system_memory": system_memory(),
            "conflicting_services": service_states(CONFLICTING_SERVICES),
        },
        "startup": None,
        "idle": None,
        "load_short_decode": None,
        "shutdown": None,
        "notes": [],
    }

    if args.dry_run:
        result["status"] = "dry_run"
        result["finished_at"] = now()
        return finish_candidate(result, events_path, candidate_id)

    if args.stop_conflicting_services:
        stop_services(CONFLICTING_SERVICES, events_path)

    preflight_mem = system_memory()
    if not memory_above_floor(preflight_mem, args):
        result["status"] = "skipped_preflight_memory"
        result["notes"].append("available memory or swap was below configured floor before startup")
        result["finished_at"] = now()
        return finish_candidate(result, events_path, candidate_id)

    unit = unit_name(candidate_id)
    log_path = ROOT / "results/logs" / f"{candidate_id}.log"
    profile_path = ROOT / "results/profiles" / f"{candidate_id}.env"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(render_profile(candidate, args), encoding="utf-8")

    startup = start_and_wait(candidate, args, unit, profile_path, log_path, events_path)
    result["startup"] = startup
    if startup["status"] != "ready":
        result["status"] = startup["status"]
        result["shutdown"] = stop_unit(unit)
        result["finished_at"] = now()
        return finish_candidate(result, events_path, candidate_id)

    time.sleep(args.idle_sleep_sec)
    idle = {
        "measured_at": now(),
        "system_memory": system_memory(),
        "service": service_show(unit),
        "vllm_capacity": parse_vllm_capacity(log_path.read_text(errors="replace")),
        "metrics": fetch_metrics(args.host, args.port),
    }
    result["idle"] = idle

    if not memory_above_floor(idle["system_memory"], args):
        result["status"] = "skipped_load_idle_memory"
        result["notes"].append("idle memory was below configured safety floor")
    elif should_skip_load(candidate, idle, args):
        result["status"] = "startup_only"
        result["notes"].append("load skipped by capacity or risk guard")
    elif args.run_load:
        result["load_short_decode"] = run_short_load(candidate, args, unit, events_path)
        result["status"] = "load_complete" if result["load_short_decode"]["errors"] == 0 else "load_errors"
    else:
        result["status"] = "startup_only"
        result["notes"].append("load not requested")

    result["shutdown"] = stop_unit(unit)
    result["finished_at"] = now()
    return finish_candidate(result, events_path, candidate_id)


def finish_candidate(result: dict, events_path: Path, candidate_id: str) -> dict:
    if not result.get("finished_at"):
        result["finished_at"] = now()
    event(events_path, "candidate_done", candidate_id=candidate_id, status=result["status"])
    return result


def render_profile(candidate: dict, args: argparse.Namespace) -> str:
    return "\n".join(
        [
            f"VLLM_WORKDIR={quote(str(DEFAULT_WORKDIR))}",
            f"VLLM_VENV={quote(str(DEFAULT_VENV))}",
            f"VLLM_BIN={quote(str(DEFAULT_VLLM_BIN))}",
            f"VLLM_MODEL={quote(candidate['model'])}",
            f"VLLM_HOST={quote(args.host)}",
            f"VLLM_PORT={quote(str(args.port))}",
            'FLASHINFER_DISABLE_VERSION_CHECK="1"',
            'CUTE_DSL_ARCH="sm_121a"',
            f"VLLM_GPU_MEMORY_UTILIZATION={quote(str(candidate['gpu_memory_utilization']))}",
            f"VLLM_MAX_MODEL_LEN={quote(str(candidate['max_model_len']))}",
            f"VLLM_MAX_NUM_SEQS={quote(str(candidate['max_num_seqs']))}",
            f"VLLM_MAX_NUM_BATCHED_TOKENS={quote(str(candidate['max_num_batched_tokens']))}",
            'VLLM_LANGUAGE_MODEL_ONLY="1"',
            'VLLM_SKIP_MM_PROFILING="0"',
            'VLLM_REASONING_PARSER="gemma4"',
            'VLLM_TOOL_CALL_PARSER="gemma4"',
            'VLLM_QUANTIZATION=""',
            f"VLLM_KV_CACHE_DTYPE={quote(candidate['kv_cache_dtype'])}",
            'VLLM_MM_PROCESSOR_CACHE_GB="0"',
            'VLLM_ENFORCE_EAGER="0"',
            f"VLLM_MOE_BACKEND={quote(candidate['moe_backend'])}",
            'VLLM_DISABLE_FLASHINFER_AUTOTUNE="1"',
            "",
        ]
    )


def start_and_wait(candidate: dict, args: argparse.Namespace, unit: str, profile_path: Path, log_path: Path, events_path: Path) -> dict:
    stop_unit(unit)
    command = [
        "systemd-run",
        "--user",
        f"--unit={unit}",
        "--collect",
        "--property=MemoryAccounting=yes",
        f"--property=MemoryMax={args.memory_max_gib}G",
        "--property=TimeoutStopSec=20s",
        f"--property=WorkingDirectory={ROOT}",
        f"--property=StandardOutput=append:{log_path}",
        f"--property=StandardError=append:{log_path}",
        *systemd_environment(),
        *vllm_serve_command(candidate, args),
    ]
    started = time.time()
    run(command, check=True)
    ready = False
    status = "startup_timeout"
    last_error = ""
    monitor_samples = []
    while time.time() - started < args.startup_timeout_sec:
        sample = {
            "at": now(),
            "service": service_show(unit),
            "system_memory": system_memory(),
        }
        monitor_samples.append(sample)
        if not memory_above_floor(sample["system_memory"], args):
            status = "startup_memory_guard"
            last_error = "memory below safety floor"
            stop_unit(unit)
            break
        state = sample["service"].get("ActiveState")
        if state in {"failed", "inactive"}:
            status = "startup_service_exit"
            last_error = state or "service exited"
            break
        try:
            models = http_json(f"http://{args.host}:{args.port}/v1/models", timeout=2)
            ids = [entry.get("id") for entry in models.get("data", [])]
            if candidate["model"] in ids:
                ready = True
                status = "ready"
                break
            last_error = f"model ids={ids}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(5)
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    return {
        "status": status,
        "ready": ready,
        "startup_seconds": round(time.time() - started, 3),
        "last_error": last_error,
        "unit": unit,
        "profile_path": rel(profile_path),
        "log_path": rel(log_path),
        "command": command,
        "service": service_show(unit),
        "system_memory": system_memory(),
        "capacity": parse_vllm_capacity(text),
        "monitor_samples": monitor_samples,
    }


def vllm_serve_command(candidate: dict, args: argparse.Namespace) -> list[str]:
    command = [
        str(DEFAULT_VLLM_BIN),
        "serve",
        candidate["model"],
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(candidate["gpu_memory_utilization"]),
        "--max-model-len",
        str(candidate["max_model_len"]),
        "--max-num-seqs",
        str(candidate["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(candidate["max_num_batched_tokens"]),
        "--enable-prefix-caching",
        "--reasoning-parser",
        "gemma4",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "gemma4",
        "--kv-cache-dtype",
        candidate["kv_cache_dtype"],
        "--mm-processor-cache-gb",
        "0",
        "--moe-backend",
        candidate["moe_backend"],
        "--language-model-only",
        "--no-enable-flashinfer-autotune",
    ]
    return command


def systemd_environment() -> list[str]:
    path = f"{DEFAULT_VENV / 'bin'}:{os.environ.get('PATH', '')}"
    return [
        "--setenv=FLASHINFER_DISABLE_VERSION_CHECK=1",
        "--setenv=CUTE_DSL_ARCH=sm_121a",
        f"--setenv=VIRTUAL_ENV={DEFAULT_VENV}",
        f"--setenv=PATH={path}",
    ]


def should_skip_load(candidate: dict, idle: dict, args: argparse.Namespace) -> bool:
    if candidate["risk_tier"] in {"high", "extreme"} and not args.allow_risky_load:
        # Startup is still useful for capacity/memory.  Load high-risk rows only
        # after nearby lower-risk rows establish headroom.
        return True
    capacity = idle.get("vllm_capacity") or {}
    reported = capacity.get("max_reported_concurrency")
    if reported is not None and reported + 1e-9 < candidate["load_concurrency"]:
        return True
    return False


def run_short_load(candidate: dict, args: argparse.Namespace, unit: str, events_path: Path) -> dict:
    concurrency = int(candidate["load_concurrency"])
    prompt = ("Measure vLLM resource behavior. " * int(candidate["load_prompt_words"])).strip()
    payload = {
        "model": candidate["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": int(candidate["load_max_tokens"]),
    }
    monitor = MemoryMonitor(unit)
    monitor.start()
    started = time.time()
    records = []
    errors = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(chat_completion, args.host, args.port, payload, args.load_timeout_sec)
                for _ in range(concurrency)
            ]
            for future in concurrent.futures.as_completed(futures, timeout=args.load_timeout_sec + 30):
                try:
                    records.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    records.append({"ok": False, "error": str(exc), "elapsed_seconds": None})
    finally:
        monitor.stop()
    wall = time.time() - started
    usage = [row.get("usage") or {} for row in records if row.get("ok")]
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in usage)
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in usage)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in usage)
    latencies = sorted(row["elapsed_seconds"] for row in records if row.get("elapsed_seconds") is not None)
    return {
        "status": "done",
        "requested_concurrency": concurrency,
        "successes": sum(1 for row in records if row.get("ok")),
        "errors": errors + sum(1 for row in records if not row.get("ok")),
        "wall_seconds": round(wall, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tokens_per_second": round(completion_tokens / wall, 3) if wall > 0 else None,
        "total_tokens_per_second": round(total_tokens / wall, 3) if wall > 0 else None,
        "latency_seconds": {
            "min": percentile(latencies, 0.0),
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "max": percentile(latencies, 1.0),
        },
        "service_after": service_show(unit),
        "system_memory_after": system_memory(),
        "memory_monitor": monitor.summary(),
        "responses": records,
    }


def chat_completion(host: str, port: int, payload: dict, timeout: int) -> dict:
    started = time.time()
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        parsed = json.loads(response.read().decode("utf-8"))
    return {
        "ok": True,
        "elapsed_seconds": round(time.time() - started, 3),
        "usage": parsed.get("usage") or {},
        "finish_reason": ((parsed.get("choices") or [{}])[0].get("finish_reason")),
    }


class MemoryMonitor:
    def __init__(self, unit: str):
        self.unit = unit
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append({"at": now(), "service": service_show(self.unit), "system_memory": system_memory()})
            self._stop.wait(1)

    def summary(self) -> dict:
        memory_current = [parse_int(sample["service"].get("MemoryCurrent")) for sample in self.samples]
        memory_peak = [parse_int(sample["service"].get("MemoryPeak")) for sample in self.samples]
        avail = [sample["system_memory"].get("mem_available_bytes", 0) for sample in self.samples]
        return {
            "samples": len(self.samples),
            "max_memory_current_bytes": max(memory_current) if memory_current else None,
            "max_memory_peak_bytes": max(memory_peak) if memory_peak else None,
            "min_available_memory_bytes": min(avail) if avail else None,
        }


def summarize_command(args: argparse.Namespace) -> int:
    summarize_results(Path(args.results), Path(args.out))
    return 0


def summarize_results(results_path: Path, out_path: Path) -> None:
    rows = read_jsonl(results_path) if results_path.exists() else []
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row.get("status", "unknown")] = statuses.get(row.get("status", "unknown"), 0) + 1
    executed = [row for row in rows if row.get("status") in {"load_complete", "load_errors", "startup_only"}]
    loaded = [row for row in rows if row.get("load_short_decode")]
    best_completion = max(
        loaded,
        key=lambda row: row["load_short_decode"].get("completion_tokens_per_second") or 0,
        default=None,
    )
    best_total = max(
        loaded,
        key=lambda row: row["load_short_decode"].get("total_tokens_per_second") or 0,
        default=None,
    )
    summary = {
        "generated_at": now(),
        "results_file": rel(results_path),
        "rows": len(rows),
        "executed_or_started": len(executed),
        "statuses": statuses,
        "best_completion_tps": summarize_throughput_row(best_completion, "completion_tokens_per_second"),
        "best_total_tps": summarize_throughput_row(best_total, "total_tokens_per_second"),
        "latest": rows[-1] if rows else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def summarize_throughput_row(row: dict | None, metric: str) -> dict | None:
    if not row:
        return None
    load = row.get("load_short_decode") or {}
    candidate = row.get("candidate") or {}
    return {
        "candidate_id": row.get("candidate_id"),
        "metric": metric,
        "value": load.get(metric),
        "context": candidate.get("max_model_len"),
        "seqs": candidate.get("max_num_seqs"),
        "batch_policy": candidate.get("batch_policy"),
        "total_tokens_per_second": load.get("total_tokens_per_second"),
        "completion_tokens_per_second": load.get("completion_tokens_per_second"),
    }


def summary_path_for(results_path: Path) -> Path:
    if results_path.name.endswith("-results.jsonl"):
        return results_path.with_name(results_path.name.removesuffix("-results.jsonl") + "-summary.json")
    return results_path.with_suffix(".summary.json")


def parse_vllm_capacity(text: str) -> dict:
    capacity: dict[str, float | int | None] = {
        "available_kv_cache_memory_gib": None,
        "gpu_kv_cache_tokens": None,
        "max_reported_context_tokens": None,
        "max_reported_concurrency": None,
    }
    m = re.search(r"Available KV cache memory:\s*([0-9.]+)\s*GiB", text)
    if m:
        capacity["available_kv_cache_memory_gib"] = float(m.group(1))
    m = re.search(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", text)
    if m:
        capacity["gpu_kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Maximum concurrency for\s*([0-9,]+)\s*tokens per request:\s*([0-9.]+)x", text)
    if m:
        capacity["max_reported_context_tokens"] = int(m.group(1).replace(",", ""))
        capacity["max_reported_concurrency"] = float(m.group(2))
    return capacity


def fetch_metrics(host: str, port: int) -> dict:
    try:
        text = http_text(f"http://{host}:{port}/metrics", timeout=2)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    selected = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if any(key in line for key in ["vllm:num_requests", "vllm:gpu_cache_usage_perc", "vllm:prompt_tokens_total", "vllm:generation_tokens_total"]):
            name, _, value = line.partition(" ")
            selected[name] = value
    return {"ok": True, "selected": selected}


def stop_services(services: list[str], events_path: Path) -> None:
    for service in services:
        state = service_show(service).get("ActiveState")
        if state == "active":
            event(events_path, "stop_service", service=service)
            run(["systemctl", "--user", "stop", service], check=False)


def stop_unit(unit: str) -> dict:
    before = service_show(unit)
    if before.get("LoadState") not in {None, "not-found"} and before.get("ActiveState") not in {"inactive", "failed"}:
        run(["systemctl", "--user", "stop", unit], check=False)
    time.sleep(1)
    return {"before": before, "after": service_show(unit)}


def service_states(services: list[str]) -> dict:
    return {service: service_show(service).get("ActiveState") for service in services}


def service_show(unit: str) -> dict:
    result = run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=LoadState,ActiveState,SubState,MainPID,MemoryCurrent,MemoryPeak,NRestarts,Result",
        ],
        check=False,
        capture=True,
    )
    data: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key] = value
    return data


def system_memory() -> dict:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        amount = int(value.strip().split()[0]) * 1024
        values[key] = amount
    return {
        "mem_total_bytes": values.get("MemTotal", 0),
        "mem_available_bytes": values.get("MemAvailable", 0),
        "swap_total_bytes": values.get("SwapTotal", 0),
        "swap_free_bytes": values.get("SwapFree", 0),
    }


def memory_above_floor(memory: dict, args: argparse.Namespace) -> bool:
    return (
        memory.get("mem_available_bytes", 0) >= args.min_available_gib * 1024**3
        and memory.get("swap_free_bytes", 0) >= args.min_swap_free_gib * 1024**3
    )


def http_json(url: str, timeout: int) -> dict:
    return json.loads(http_text(url, timeout))


def http_text(url: str, timeout: int) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return response.read().decode("utf-8")


def run(command: list[str], check: bool, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check, text=True, capture_output=capture)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def event(path: Path, event_type: str, **fields: object) -> None:
    append_jsonl(path, {"at": now(), "event": event_type, **fields})


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    index = q * (len(values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    frac = index - lower
    return round(values[lower] * (1 - frac) + values[upper] * frac, 3)


def parse_int(value: str | None) -> int:
    with contextlib.suppress(Exception):
        return int(value or "0")
    return 0


def quote(value: str) -> str:
    return json.dumps(value)


def rel(path: Path) -> str:
    with contextlib.suppress(ValueError):
        return str(path.resolve().relative_to(ROOT))
    return str(path)


def now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def unit_name(candidate_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", candidate_id)
    return f"gemma4-vllm-sweep-{safe}"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        os.kill(os.getpid(), signal.SIGINT)
