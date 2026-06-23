#!/usr/bin/env python3
"""Generate simple CSV, SVG, and Markdown summaries for sweep JSONL results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT / "results/sweep-results.jsonl"))
    parser.add_argument("--reports-dir", default=str(ROOT / "reports"))
    args = parser.parse_args()
    rows = read_jsonl(Path(args.results))
    reports = Path(args.reports_dir)
    (reports / "plots").mkdir(parents=True, exist_ok=True)
    (reports / "tables").mkdir(parents=True, exist_ok=True)
    (reports / "models").mkdir(parents=True, exist_ok=True)
    table_rows = flatten_rows(rows)
    write_csv(reports / "tables/measurements.csv", table_rows)
    models = build_models(table_rows)
    write_models(reports / "models/linear-models.json", models)
    write_plots(reports / "plots", table_rows)
    write_summary(reports / "summary.md", table_rows, rows, models)
    write_html(reports / "index.html", table_rows, raw_rows=rows, models=models)
    print(f"wrote report files under {reports}")
    return 0


def flatten_rows(rows: list[dict]) -> list[dict]:
    flat = []
    for row in rows:
        c = row.get("candidate", {})
        idle = row.get("idle") or {}
        load = row.get("load_short_decode") or {}
        startup = row.get("startup") or {}
        capacity = idle.get("vllm_capacity") or startup.get("capacity") or {}
        idle_service = idle.get("service") or startup.get("service") or {}
        load_service = load.get("service_after") or {}
        flat.append(
            {
                "candidate_id": row.get("candidate_id"),
                "status": row.get("status"),
                "max_model_len": c.get("max_model_len"),
                "max_num_seqs": c.get("max_num_seqs"),
                "max_num_batched_tokens": c.get("max_num_batched_tokens"),
                "batch_policy": c.get("batch_policy"),
                "risk_tier": c.get("risk_tier"),
                "token_budget_m": safe_product(c.get("max_model_len"), c.get("max_num_seqs"), scale=1_000_000),
                "startup_seconds": (startup or {}).get("startup_seconds"),
                "kv_cache_tokens": capacity.get("gpu_kv_cache_tokens"),
                "reported_concurrency": capacity.get("max_reported_concurrency"),
                "idle_memory_gib": bytes_to_gib(idle_service.get("MemoryCurrent")),
                "idle_memory_peak_gib": bytes_to_gib(idle_service.get("MemoryPeak")),
                "load_memory_gib": bytes_to_gib(load_service.get("MemoryCurrent")),
                "load_memory_peak_gib": bytes_to_gib(load_service.get("MemoryPeak")),
                "load_successes": load.get("successes"),
                "load_errors": load.get("errors"),
                "completion_tok_s": load.get("completion_tokens_per_second"),
                "total_tok_s": load.get("total_tokens_per_second"),
                "latency_p50": (load.get("latency_seconds") or {}).get("p50"),
                "latency_p95": (load.get("latency_seconds") or {}).get("p95"),
            }
        )
    return flat


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_models(rows: list[dict]) -> dict:
    usable = [r for r in rows if number(r.get("idle_memory_gib")) is not None]
    models = {
        "idle_memory_gib": fit_linear(
            usable,
            "idle_memory_gib",
            [
                ("intercept", lambda r: 1.0),
                ("context_k", lambda r: number(r["max_model_len"]) / 1024),
                ("max_num_seqs", lambda r: number(r["max_num_seqs"])),
                ("context_k_x_seqs", lambda r: (number(r["max_model_len"]) / 1024) * number(r["max_num_seqs"])),
                ("match_context_policy", lambda r: 1.0 if r.get("batch_policy") == "match_context" else 0.0),
            ],
        )
    }
    models["idle_memory_peak_gib"] = fit_linear(
        [r for r in rows if number(r.get("idle_memory_peak_gib")) is not None],
        "idle_memory_peak_gib",
        [
            ("intercept", lambda r: 1.0),
            ("context_k", lambda r: number(r["max_model_len"]) / 1024),
            ("max_num_seqs", lambda r: number(r["max_num_seqs"])),
            ("context_k_x_seqs", lambda r: (number(r["max_model_len"]) / 1024) * number(r["max_num_seqs"])),
            ("match_context_policy", lambda r: 1.0 if r.get("batch_policy") == "match_context" else 0.0),
        ],
    )
    models["load_memory_peak_gib"] = fit_linear(
        [r for r in rows if number(r.get("load_memory_peak_gib")) is not None],
        "load_memory_peak_gib",
        [
            ("intercept", lambda r: 1.0),
            ("context_k", lambda r: number(r["max_model_len"]) / 1024),
            ("max_num_seqs", lambda r: number(r["max_num_seqs"])),
            ("context_k_x_seqs", lambda r: (number(r["max_model_len"]) / 1024) * number(r["max_num_seqs"])),
        ],
    )
    models["reported_concurrency"] = fit_linear(
        [r for r in rows if number(r.get("reported_concurrency")) is not None],
        "reported_concurrency",
        [
            ("intercept", lambda r: 1.0),
            ("context_k", lambda r: number(r["max_model_len"]) / 1024),
            ("max_num_seqs", lambda r: number(r["max_num_seqs"])),
            ("context_k_x_seqs", lambda r: (number(r["max_model_len"]) / 1024) * number(r["max_num_seqs"])),
            ("match_context_policy", lambda r: 1.0 if r.get("batch_policy") == "match_context" else 0.0),
        ],
    )
    models["reported_concurrency_log"] = fit_log_response(
        [r for r in rows if number(r.get("reported_concurrency")) is not None],
        "reported_concurrency",
        [
            ("intercept", lambda r: 1.0),
            ("log_context", lambda r: math.log(number(r["max_model_len"]))),
            ("max_num_seqs", lambda r: number(r["max_num_seqs"])),
            ("match_context_policy", lambda r: 1.0 if r.get("batch_policy") == "match_context" else 0.0),
        ],
    )
    models["reported_concurrency_empirical"] = fit_piecewise_loglog_by_policy(
        [r for r in rows if number(r.get("reported_concurrency")) is not None],
        "reported_concurrency",
    )
    loaded = [r for r in rows if number(r.get("completion_tok_s")) is not None]
    models["completion_tok_s"] = fit_linear(
        loaded,
        "completion_tok_s",
        [
            ("intercept", lambda r: 1.0),
            ("context_k", lambda r: number(r["max_model_len"]) / 1024),
            ("max_num_seqs", lambda r: number(r["max_num_seqs"])),
        ],
    )
    models["latency_p95"] = fit_linear(
        [r for r in rows if number(r.get("latency_p95")) is not None],
        "latency_p95",
        [
            ("intercept", lambda r: 1.0),
            ("context_k", lambda r: number(r["max_model_len"]) / 1024),
            ("max_num_seqs", lambda r: number(r["max_num_seqs"])),
        ],
    )
    return models


def write_models(path: Path, models: dict) -> None:
    path.write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")


def fit_linear(rows: list[dict], target: str, features: list[tuple[str, object]]) -> dict:
    clean = []
    for row in rows:
        y = number(row.get(target))
        if y is None:
            continue
        xs = []
        ok = True
        for _, fn in features:
            value = fn(row)
            if value is None or not math.isfinite(value):
                ok = False
                break
            xs.append(float(value))
        if ok:
            clean.append((xs, float(y)))
    if len(clean) < len(features):
        return {"ok": False, "reason": "not enough rows", "rows": len(clean)}
    xtx = [[0.0 for _ in features] for _ in features]
    xty = [0.0 for _ in features]
    for xs, y in clean:
        for i, xi in enumerate(xs):
            xty[i] += xi * y
            for j, xj in enumerate(xs):
                xtx[i][j] += xi * xj
    beta = solve(xtx, xty)
    if beta is None:
        return {"ok": False, "reason": "singular matrix", "rows": len(clean)}
    ys = [y for _, y in clean]
    predictions = [sum(beta[i] * xs[i] for i in range(len(beta))) for xs, _ in clean]
    residuals = [ys[i] - predictions[i] for i in range(len(ys))]
    y_mean = sum(ys) / len(ys)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum(value**2 for value in residuals)
    mae = sum(abs(value) for value in residuals) / len(residuals)
    rmse = math.sqrt(ss_res / len(residuals))
    return {
        "ok": True,
        "rows": len(clean),
        "target": target,
        "coefficients": {name: beta[index] for index, (name, _) in enumerate(features)},
        "r2": None if ss_tot == 0 else 1 - ss_res / ss_tot,
        "mae": mae,
        "rmse": rmse,
    }


def fit_log_response(rows: list[dict], target: str, features: list[tuple[str, object]]) -> dict:
    clean = []
    for row in rows:
        y = number(row.get(target))
        if y is None or y <= 0:
            continue
        xs = []
        ok = True
        for _, fn in features:
            value = fn(row)
            if value is None or not math.isfinite(value):
                ok = False
                break
            xs.append(float(value))
        if ok:
            clean.append((xs, float(y), math.log(float(y))))
    if len(clean) < len(features):
        return {"ok": False, "reason": "not enough rows", "rows": len(clean), "transform": "log_response"}
    xtx = [[0.0 for _ in features] for _ in features]
    xty = [0.0 for _ in features]
    for xs, _, log_y in clean:
        for i, xi in enumerate(xs):
            xty[i] += xi * log_y
            for j, xj in enumerate(xs):
                xtx[i][j] += xi * xj
    beta = solve(xtx, xty)
    if beta is None:
        return {"ok": False, "reason": "singular matrix", "rows": len(clean), "transform": "log_response"}
    ys = [y for _, y, _ in clean]
    predictions = [math.exp(sum(beta[i] * xs[i] for i in range(len(beta)))) for xs, _, _ in clean]
    residuals = [ys[i] - predictions[i] for i in range(len(ys))]
    y_mean = sum(ys) / len(ys)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum(value**2 for value in residuals)
    mae = sum(abs(value) for value in residuals) / len(residuals)
    rmse = math.sqrt(ss_res / len(residuals))
    return {
        "ok": True,
        "rows": len(clean),
        "target": target,
        "transform": "log_response",
        "coefficients": {name: beta[index] for index, (name, _) in enumerate(features)},
        "r2": None if ss_tot == 0 else 1 - ss_res / ss_tot,
        "mae": mae,
        "rmse": rmse,
    }


def fit_piecewise_loglog_by_policy(rows: list[dict], target: str) -> dict:
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        y = number(row.get(target))
        c = number(row.get("max_model_len"))
        policy = row.get("batch_policy")
        if y is None or y <= 0 or c is None or c <= 0 or not policy:
            continue
        grouped.setdefault(str(policy), {}).setdefault(int(c), []).append(float(y))
    points: dict[str, list[list[float]]] = {}
    for policy, by_context in grouped.items():
        policy_points = []
        for context, values in sorted(by_context.items()):
            ordered = sorted(values)
            mid = len(ordered) // 2
            median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
            policy_points.append([float(context), median])
        points[policy] = policy_points
    clean = []
    for row in rows:
        y = number(row.get(target))
        c = number(row.get("max_model_len"))
        policy = row.get("batch_policy")
        if y is None or c is None or not policy:
            continue
        predicted = predict_piecewise_loglog(points.get(str(policy), []), float(c))
        if predicted is not None:
            clean.append((float(y), predicted))
    if len(clean) < 2:
        return {"ok": False, "reason": "not enough rows", "rows": len(clean), "transform": "piecewise_loglog_by_policy", "points": points}
    ys = [y for y, _ in clean]
    predictions = [pred for _, pred in clean]
    residuals = [ys[i] - predictions[i] for i in range(len(ys))]
    y_mean = sum(ys) / len(ys)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum(value**2 for value in residuals)
    mae = sum(abs(value) for value in residuals) / len(residuals)
    rmse = math.sqrt(ss_res / len(residuals))
    return {
        "ok": True,
        "rows": len(clean),
        "target": target,
        "transform": "piecewise_loglog_by_policy",
        "points": points,
        "r2": None if ss_tot == 0 else 1 - ss_res / ss_tot,
        "mae": mae,
        "rmse": rmse,
    }


def predict_piecewise_loglog(points: list[list[float]], context: float) -> float | None:
    clean = [(float(c), float(y)) for c, y in points if c and y and c > 0 and y > 0]
    if not clean:
        return None
    clean.sort()
    if context <= clean[0][0]:
        return clean[0][1]
    if context >= clean[-1][0]:
        return clean[-1][1]
    for index in range(1, len(clean)):
        left_c, left_y = clean[index - 1]
        right_c, right_y = clean[index]
        if context <= right_c:
            t = (math.log(context) - math.log(left_c)) / (math.log(right_c) - math.log(left_c))
            return math.exp(math.log(left_y) + t * (math.log(right_y) - math.log(left_y)))
    return clean[-1][1]


def solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [value / div for value in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [m[row][i] - factor * m[col][i] for i in range(n + 1)]
    return [m[i][n] for i in range(n)]


def write_plots(path: Path, rows: list[dict]) -> None:
    plot_svg(path / "idle-memory-by-context.svg", rows, "max_model_len", "idle_memory_gib", "Idle memory by context", "context tokens", "GiB")
    plot_svg(path / "load-peak-memory-by-context.svg", rows, "max_model_len", "load_memory_peak_gib", "Loaded peak memory by context", "context tokens", "GiB")
    plot_svg(path / "reported-concurrency-by-context.svg", rows, "max_model_len", "reported_concurrency", "Reported max concurrency by context", "context tokens", "reported max concurrency")
    plot_svg(path / "throughput-by-concurrency.svg", rows, "max_num_seqs", "completion_tok_s", "Output throughput by concurrency", "max_num_seqs", "completion tok/s")
    plot_svg(path / "latency-p95-by-concurrency.svg", rows, "max_num_seqs", "latency_p95", "P95 latency by concurrency", "max_num_seqs", "seconds")


def plot_svg(path: Path, rows: list[dict], x_key: str, y_key: str, title: str, x_label: str, y_label: str) -> None:
    points = [(number(r.get(x_key)), number(r.get(y_key)), r.get("status")) for r in rows]
    points = [(x, y, s) for x, y, s in points if x is not None and y is not None]
    width, height = 900, 520
    pad = 70
    if not points:
        path.write_text(f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'><text x='20' y='40'>{title}: no data</text></svg>\n", encoding="utf-8")
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmax += 1
    if ymin == ymax:
        ymax += 1
    def sx(x: float) -> float:
        return pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)
    def sy(y: float) -> float:
        return height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)
    circles = []
    for x, y, status in points:
        color = "#1f77b4" if status == "load_complete" else "#ff7f0e"
        circles.append(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='5' fill='{color}'><title>{x_key}={x}, {y_key}={y}, status={status}</title></circle>")
    svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="32" text-anchor="middle" font-size="20">{escape(title)}</text>
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="black"/>
<text x="{width/2}" y="{height-20}" text-anchor="middle">{escape(x_label)}</text>
<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle">{escape(y_label)}</text>
{''.join(circles)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_summary(path: Path, rows: list[dict], raw_rows: list[dict], models: dict) -> None:
    statuses = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    contexts = sorted({int(number(row["max_model_len"]) or 0) for row in rows})
    max_requested_seqs = max(int(number(row["max_num_seqs"]) or 0) for row in rows) if rows else 0
    top_throughput = top_rows(rows, "completion_tok_s", 10)
    top_total = top_rows(rows, "total_tok_s", 5)
    context_rows = context_summary(rows)
    hundred_k_rows = [row for row in rows if int(number(row.get("max_model_len")) or 0) == 100000]
    lines = [
        "# Gemma 4 vLLM Resource Sweep Report",
        "",
        "This report is generated from `results/sweep-results.jsonl`. It covers a standalone vLLM sweep for `nvidia/Gemma-4-26B-A4B-NVFP4`; it does not use LocalPager or OpenClaw runtime paths.",
        "",
        f"Rows recorded: {len(rows)}.",
        f"Context windows covered: {', '.join(str(value) for value in contexts)}.",
        f"Highest requested concurrency covered: {max_requested_seqs}.",
        "",
        "The load probe uses concurrent short prompts with 64 requested output tokens. High-risk rows are still started and measured at idle, but load is skipped when vLLM reports insufficient capacity or when the harness risk guard blocks it.",
        "",
        "## Status Counts",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(statuses.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            f"- Best measured output throughput was `{fmt(top_throughput[0].get('completion_tok_s'))}` completion tok/s at `{top_throughput[0]['candidate_id']}` with `{top_throughput[0]['max_model_len']}` context and `{top_throughput[0]['max_num_seqs']}` requested concurrency." if top_throughput else "- No load-complete rows were available for throughput ranking.",
            f"- Best measured total token throughput was `{fmt(top_total[0].get('total_tok_s'))}` total tok/s at `{top_total[0]['candidate_id']}`." if top_total else "- No total token throughput rows were available.",
            "- The `small` batching policy (`max_num_batched_tokens` capped at 8192) is the stable high-context path in this sweep. It starts successfully at 100k, 131k, 196k, and 262k context, though high-risk load was intentionally skipped.",
            "- The `match_context` batching policy often reduces reported concurrency at high context and hit a CUTLASS FP4 MoE kernel/config boundary at 196k and 262k contexts.",
            "- No run was classified as a machine OOM. The startup exits were vLLM/kernel assertions, and the guard stopped load when capacity or risk was unsafe.",
        ]
    )
    boundaries = startup_boundaries(raw_rows)
    if boundaries:
        lines.extend(["", "## Startup Boundaries", ""])
        for boundary in boundaries:
            lines.append(f"- {boundary}")
    lines.extend(capacity_simplification_markdown())
    lines.extend(abstract_model_markdown())
    lines.extend(
        [
            "",
            "## Fit Quality",
            "",
            "The coefficients for these model families are stored in `models/linear-models.json`. They are intentionally not printed in the report formulas; the report formulas define the system, and this table only says how well each family fit the observed rows.",
            "",
            "| target | rows | R2 | MAE | RMSE |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, model in models.items():
        lines.append(model_quality_table_row(name, model))
    lines.extend(["", "## Context Summary", "", "| context | rows | load complete | startup only | startup exits | max loaded seqs | best completion tok/s | max reported concurrency |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in context_rows:
        lines.append(
            f"| {row['context']} | {row['rows']} | {row['load_complete']} | {row['startup_only']} | {row['startup_service_exit']} | {row['max_loaded_seqs']} | {fmt(row['best_completion_tok_s'])} | {fmt(row['max_reported_concurrency'])} |"
        )
    lines.extend(["", "## Top Output Throughput Rows", "", "| rank | candidate | context | seqs | policy | completion tok/s | total tok/s | p95 latency | idle GiB | load peak GiB |", "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |"])
    for index, row in enumerate(top_throughput, start=1):
        lines.append(
            f"| {index} | {row['candidate_id']} | {row['max_model_len']} | {row['max_num_seqs']} | {row['batch_policy']} | {fmt(row['completion_tok_s'])} | {fmt(row['total_tok_s'])} | {fmt(row['latency_p95'])} | {fmt(row['idle_memory_gib'])} | {fmt(row['load_memory_peak_gib'])} |"
        )
    if hundred_k_rows:
        lines.extend(["", "## 100k Context Rows", "", "| candidate | status | seqs | policy | reported concurrency | idle GiB | completion tok/s | notes |", "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |"])
        raw_by_id = {row.get("candidate_id"): row for row in raw_rows}
        for row in sorted(hundred_k_rows, key=lambda item: (number(item["max_num_seqs"]) or 0, item["batch_policy"])):
            notes = "; ".join(raw_by_id.get(row["candidate_id"], {}).get("notes") or [])
            lines.append(
                f"| {row['candidate_id']} | {row['status']} | {row['max_num_seqs']} | {row['batch_policy']} | {fmt(row['reported_concurrency'])} | {fmt(row['idle_memory_gib'])} | {fmt(row['completion_tok_s'])} | {notes} |"
            )
    lines.extend(["", "## Latest Measurements", "", "| candidate | status | context | seqs | idle GiB | tok/s |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in rows[-20:]:
        lines.append(
            f"| {row['candidate_id']} | {row['status']} | {row['max_model_len']} | {row['max_num_seqs']} | {fmt(row.get('idle_memory_gib'))} | {fmt(row.get('completion_tok_s'))} |"
        )
    lines.extend(
        [
            "",
            "## Runtime And Safety",
            "",
            "- vLLM was started directly as transient user systemd services from the standalone harness.",
            "- Model: `nvidia/Gemma-4-26B-A4B-NVFP4`.",
            "- Fixed vLLM flags included `--gpu-memory-utilization 0.65`, `--kv-cache-dtype fp8`, `--moe-backend cutlass`, `--language-model-only`, and `--no-enable-flashinfer-autotune`.",
            "- Each candidate used `MemoryMax=95G`, `min_available_gib=12`, and `min_swap_free_gib=4` guardrails.",
            "- Conflicting local LLM services were stopped before each candidate to isolate the measurement; the harness itself is independent from LocalPager.",
            "- `nvidia-smi` memory fields are not useful on this GB10 setup, so the report uses vLLM logs, cgroup memory, and system memory snapshots.",
            "",
            "## Limitations",
            "",
            "- The load phase is a short-decode probe, not a full long-prefill benchmark.",
            "- High-risk rows are intentionally startup/capacity measurements unless explicitly allowed for risky load.",
            "- The fitted formulas summarize this run's measured relationships and can be distorted by guardrails, cold-start compile behavior, and skipped load rows.",
            "- cgroup memory is the best available process-level memory signal here; it is not a direct GPU-memory counter.",
            "",
            "## Artifacts",
            "",
            "- `results/sweep-results.jsonl`: machine-readable per-candidate measurements.",
            "- `results/sweep-summary.json`: machine-readable aggregate summary.",
            "- `tables/measurements.csv`: flattened table for spreadsheets.",
            "- `plots/idle-memory-by-context.svg`",
            "- `plots/load-peak-memory-by-context.svg`",
            "- `plots/reported-concurrency-by-context.svg`",
            "- `plots/throughput-by-concurrency.svg`",
            "- `plots/latency-p95-by-concurrency.svg`",
            "- `models/linear-models.json`: fitted coefficients and error metrics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def startup_boundaries(raw_rows: list[dict]) -> list[str]:
    startup_exits = [row for row in raw_rows if row.get("status") == "startup_service_exit"]
    if not startup_exits:
        return []
    cutlass_limit = []
    oom_like = []
    for row in startup_exits:
        startup = row.get("startup") or {}
        log_ref = startup.get("log_path")
        text = ""
        if log_ref:
            log_path = ROOT / log_ref
            if log_path.exists():
                text = log_path.read_text(errors="replace")
        if "MAX_TOKENS_PER_EXPERT" in text and "cutlass_moe_fp4" in text:
            cutlass_limit.append(row.get("candidate_id"))
        if "out of memory" in text.lower() or "oom" in text.lower():
            oom_like.append(row.get("candidate_id"))
    notes = [f"{len(startup_exits)} rows exited during startup."]
    if cutlass_limit:
        notes.append(
            f"{len(cutlass_limit)} of those hit the CUTLASS FP4 MoE MAX_TOKENS_PER_EXPERT assertion, which is a kernel/config boundary rather than an observed machine OOM: {', '.join(cutlass_limit)}."
        )
    if oom_like:
        notes.append(f"OOM-like text appeared in these startup-exit logs and needs separate review: {', '.join(oom_like)}.")
    return notes


def page_css() -> str:
    return """<style>
:root{color-scheme:light;--ink:#1d2522;--muted:#5f6d68;--line:#d8dfdc;--paper:#fbfcfa;--panel:#f2f5f1;--accent:#176b64;--accent-2:#9b5f00;--bad:#9d2f2f;--good:#286f3d}
body{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45;margin:32px;max-width:1240px;color:var(--ink);background:var(--paper)}
h1,h2,h3{line-height:1.15;margin:28px 0 12px}h1{font-size:2rem}h2{font-size:1.35rem}h3{font-size:1rem}
p{max-width:78ch}table{border-collapse:collapse;margin:16px 0;width:100%;font-size:.92rem}th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:middle}th{background:#eef2ee}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
img{max-width:100%;border:1px solid var(--line);background:white}code{background:#eef2ee;padding:1px 3px;border-radius:3px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}
.explorer{display:grid;grid-template-columns:minmax(360px,.76fr) minmax(640px,1.24fr);gap:20px;align-items:start;margin:18px 0 28px}
.capacity-planner{display:grid;grid-template-columns:minmax(320px,.62fr) minmax(620px,1.38fr);gap:20px;align-items:start;margin:18px 0 28px}.capacity-planner .control-grid{grid-template-columns:1fr}.planner-table{font-size:.84rem;margin-top:14px;table-layout:fixed}.planner-table th,.planner-table td{padding:5px 6px}.planner-table code{white-space:normal;overflow-wrap:anywhere}
.tool-panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;min-width:0;overflow-x:auto}.control-grid{display:grid;grid-template-columns:repeat(2,minmax(180px,1fr));gap:14px}
.control{display:grid;gap:6px}.control label{font-size:.8rem;font-weight:700;color:#2f3d39;text-transform:uppercase;letter-spacing:.02em}.control output{font-variant-numeric:tabular-nums;color:var(--accent)}
input[type=range]{width:100%;accent-color:var(--accent)}select{width:100%;min-height:34px;border:1px solid #c5cfca;border-radius:6px;background:white;color:var(--ink);padding:4px 8px}
.readouts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:14px}.readout{background:white;border:1px solid var(--line);border-radius:6px;padding:10px;min-width:0}.readout b{display:block;font-size:.78rem;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.02em}.readout span{display:block;margin-top:3px;font-size:1.15rem;font-variant-numeric:tabular-nums}
.graph-frame{background:white;border:1px solid var(--line);border-radius:8px;padding:10px;min-height:540px}.graph-frame svg{display:block;width:100%;height:auto;overflow:visible}.legend{display:flex;flex-wrap:wrap;gap:10px 14px;margin:8px 0 0;color:var(--muted);font-size:.85rem}.legend span{display:inline-flex;align-items:center;gap:6px}.swatch{width:10px;height:10px;border-radius:999px;background:var(--accent)}.swatch.square{border-radius:2px;background:#694f9e}.swatch.hollow{background:white;border:1px solid #71807a}.swatch.surface{width:20px;height:10px;border-radius:2px;background:linear-gradient(90deg,#f7efd3,#65a98e,#33428d)}.swatch.line{width:20px;height:2px;border-radius:0;background:var(--accent)}
.sample-panel{grid-column:1/-1;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto}.sample-table{font-size:.82rem;margin:0;min-width:900px}.sample-table th,.sample-table td{padding:5px 6px}.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 7px;background:white;font-size:.78rem}.ok{color:var(--good)}.warn{color:var(--accent-2)}.bad{color:var(--bad)}
@media (max-width:900px){body{margin:18px}.explorer,.capacity-planner{grid-template-columns:1fr}.control-grid,.readouts{grid-template-columns:1fr}.grid{grid-template-columns:1fr}.graph-frame{min-height:420px}}
</style>"""


def capacity_simplification_markdown() -> list[str]:
    return [
        "",
        "## Simplified Capacity Model",
        "",
        "The capacity view deliberately simplifies the parameter sweep. For capacity planning, context length is the main input. The batch policy is the main choice. The other capacity parameters can be derived from those two choices.",
        "",
        "| symbol | meaning | simplified role |",
        "| --- | --- | --- |",
        "| $c$ | context window / `--max-model-len` | primary input |",
        "| $p$ | batch policy | user chooses `small` or `match_context` |",
        "| $b$ | `--max-num-batched-tokens` | derived from $c,p$ |",
        "| $q$ | vLLM-reported max concurrency | measured/estimated capacity curve $Q_p(c)$ |",
        "| $s$ | requested max sequences / `--max-num-seqs` | recommended from $q$ with a safety margin |",
        "",
        "Policy definitions used in this sweep:",
        "",
        "$$b_{small}(c)=\\min(c,8192)$$",
        "",
        "$$b_{match\\_context}(c)=c$$",
        "",
        "Recommended requested concurrency:",
        "",
        "$$s_{recommended}(c,p,m)=\\left\\lfloor m\\,Q_p(c)\\right\\rfloor$$",
        "",
        "where $m$ is the safety margin. The interactive dashboard defaults to $m=0.8$.",
        "",
        "For a fixed memory budget, the capacity question is not `what is memory as a smooth function of every knob?` The edge-case question is `what requested session count can survive for this context window?` In this report that frontier is:",
        "",
        "$$s \\le \\left\\lfloor m\\,Q_p(c)\\right\\rfloor$$",
        "",
        "The planner therefore answers high-context questions by reading the measured capacity frontier. For example, at a 150k context target, choose the context slider, choose the safety margin, and compare the `small` and `match_context` rows. The table gives the derived batch-token value and the recommended session count.",
        "",
        "This simplification is valid for capacity planning: deciding whether a context window can fit and choosing a conservative `--max-num-seqs`. It is not sufficient for throughput or latency, because throughput and latency still depend on the actual requested concurrency, prompt/load shape, and whether the load phase was run.",
        "",
        "The capacity curve in the dashboard is policy-specific and empirical. It uses the measured vLLM-reported concurrency points from this sweep and connects them continuously on a log-context scale. That avoids forcing a bad global linear model onto a clearly non-linear capacity curve.",
    ]


def capacity_simplification_html() -> list[str]:
    return [
        "<h2>Capacity Model Notes</h2>",
        "<p>The simplified capacity view treats context length as the main input and policy as the main choice. Batch tokens and recommended concurrency are derived from those.</p>",
        "<table><tr><th>symbol</th><th>meaning</th><th>simplified role</th></tr>",
        "<tr><td>\\(c\\)</td><td>context window / <code>--max-model-len</code></td><td>primary input</td></tr>",
        "<tr><td>\\(p\\)</td><td>batch policy</td><td>choose <code>small</code> or <code>match_context</code></td></tr>",
        "<tr><td>\\(b\\)</td><td><code>--max-num-batched-tokens</code></td><td>derived from \\(c,p\\)</td></tr>",
        "<tr><td>\\(q\\)</td><td>vLLM-reported max concurrency</td><td>measured capacity curve \\(Q_p(c)\\)</td></tr>",
        "<tr><td>\\(s\\)</td><td><code>--max-num-seqs</code></td><td>recommended from \\(q\\) with a safety margin</td></tr>",
        "</table>",
        "<p>Policy definitions:</p>",
        "<p>\\[b_{small}(c)=\\min(c,8192)\\]</p>",
        "<p>\\[b_{match\\_context}(c)=c\\]</p>",
        "<p>Recommended concurrency:</p>",
        "<p>\\[s_{recommended}(c,p,m)=\\left\\lfloor m\\,Q_p(c)\\right\\rfloor\\]</p>",
        "<p>The dashboard defaults to \\(m=0.8\\). For a fixed memory budget, this is the practical frontier: requested sessions should stay below \\(\\left\\lfloor m\\,Q_p(c)\\right\\rfloor\\). At 150k context, use the slider and compare the two policy rows instead of reading a global linear interpolation.</p>",
        "<p>This simplification is for capacity planning only. Throughput and latency still need the separate metric surface because they depend on actual requested concurrency and load shape.</p>",
    ]


def capacity_planner_html() -> list[str]:
    return [
        "<h2>Capacity Planner</h2>",
        '<section class="capacity-planner" aria-label="Capacity planner">',
        '<div class="tool-panel">',
        '<div class="control-grid">',
        '<div class="control"><label for="capacityContextSlider">Context window</label><input id="capacityContextSlider" type="range" min="4096" max="262144" step="1" value="150000"><output id="capacityContextValue">150000</output></div>',
        '<div class="control"><label for="marginSlider">Safety margin</label><input id="marginSlider" type="range" min="0.5" max="1" step="0.05" value="0.8"><output id="marginValue">0.80</output></div>',
        "</div>",
        '<div class="readouts">',
        '<div class="readout"><b>small policy</b><span id="smallPolicyReadout"></span></div>',
        '<div class="readout"><b>match-context policy</b><span id="matchPolicyReadout"></span></div>',
        '<div class="readout"><b>small b(c)</b><span id="smallBatchReadout"></span></div>',
        '<div class="readout"><b>match b(c)</b><span id="matchBatchReadout"></span></div>',
        "</div>",
        '<table class="planner-table" id="capacityTable"><thead><tr><th>policy</th><th class="num">b(c)</th><th class="num">q(c)</th><th class="num">rec. s</th></tr></thead><tbody></tbody></table>',
        "</div>",
        '<div class="graph-frame">',
        '<svg id="capacitySvg" viewBox="0 0 940 560" role="img" aria-label="Capacity by context and policy"></svg>',
        '<div class="legend"><span><i class="swatch"></i>small q(c)</span><span><i class="swatch line"></i>small safe s</span><span><i class="swatch square"></i>match-context q(c)</span><span><i class="swatch hollow"></i>measured points</span></div>',
        "</div>",
        "</section>",
    ]


def explorer_rows(rows: list[dict]) -> list[dict]:
    keep = [
        "candidate_id",
        "status",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "batch_policy",
        "risk_tier",
        "token_budget_m",
        "startup_seconds",
        "kv_cache_tokens",
        "reported_concurrency",
        "idle_memory_gib",
        "idle_memory_peak_gib",
        "load_memory_peak_gib",
        "completion_tok_s",
        "total_tok_s",
        "latency_p95",
    ]
    out = []
    for row in rows:
        item = {key: row.get(key) for key in keep}
        for key in keep:
            if key in {"candidate_id", "status", "batch_policy", "risk_tier"}:
                continue
            n = number(item.get(key))
            item[key] = None if n is None else round(n, 6)
        out.append(item)
    return out


def interactive_explorer_html(rows: list[dict], models: dict) -> list[str]:
    data = json.dumps(explorer_rows(rows), separators=(",", ":")).replace("</", "<\\/")
    model_data = json.dumps(models, separators=(",", ":")).replace("</", "<\\/")
    return [
        "<h2>Metric Surface Explorer</h2>",
        '<section class="explorer" aria-label="Interactive sweep explorer">',
        '<div class="tool-panel">',
        '<div class="control-grid">',
        '<div class="control"><label for="contextSlider">Context window</label><input id="contextSlider" type="range" min="4096" max="262144" step="1" value="150000"><output id="contextValue">150000</output></div>',
        '<div class="control"><label for="seqSlider">Requested concurrency</label><input id="seqSlider" type="range" min="1" max="32" step="1" value="16"><output id="seqValue">16</output></div>',
        '<div class="control"><label for="policySelect">Batch policy</label><select id="policySelect"><option value="small">small</option><option value="match_context">match_context</option></select></div>',
        '<div class="control"><label for="metricSelect">Metric surface</label><select id="metricSelect"><option value="completion_tok_s">completion tok/s</option><option value="idle_memory_gib">idle memory GiB</option><option value="load_memory_peak_gib">load peak GiB</option><option value="latency_p95">p95 latency</option><option value="reported_concurrency">reported concurrency (diagnostic)</option></select></div>',
        "</div>",
        '<div class="readouts">',
        '<div class="readout"><b>requested budget</b><span id="budgetReadout"></span></div>',
        '<div class="readout"><b>model prediction</b><span id="predictionReadout"></span></div>',
        '<div class="readout"><b>nearest measured</b><span id="nearestReadout"></span></div>',
        '<div class="readout"><b>fit quality</b><span id="fitReadout"></span></div>',
        "</div>",
        '<p id="nearestText"></p>',
        "</div>",
        '<div class="graph-frame">',
        '<svg id="explorerSvg" viewBox="0 0 940 590" role="img" aria-label="Continuous fitted model surface with measured samples"></svg>',
        '<div class="legend"><span><i class="swatch surface"></i>continuous fitted surface</span><span><i class="swatch"></i>sample with actual metric</span><span><i class="swatch hollow"></i>sample without that metric</span><span><i class="swatch square"></i>selected point</span></div>',
        "</div>",
        '<div class="sample-panel"><table class="sample-table" id="nearestTable"><thead><tr><th>sample</th><th class="num">ctx</th><th class="num">seq</th><th>status</th><th class="num">actual</th><th class="num">model</th><th class="num">residual</th><th class="num">q</th><th class="num">tok/s</th></tr></thead><tbody></tbody></table></div>',
        "</section>",
        f'<script id="sweep-data" type="application/json">{data}</script>',
        f'<script id="sweep-models" type="application/json">{model_data}</script>',
        explorer_script(),
    ]


def explorer_script() -> str:
    return """<script>
(function(){
  const samples = JSON.parse(document.getElementById('sweep-data').textContent);
  const models = JSON.parse(document.getElementById('sweep-models').textContent);
  const contextSlider = document.getElementById('contextSlider');
  const seqSlider = document.getElementById('seqSlider');
  const policySelect = document.getElementById('policySelect');
  const metricSelect = document.getElementById('metricSelect');
  const contextValue = document.getElementById('contextValue');
  const seqValue = document.getElementById('seqValue');
  const budgetReadout = document.getElementById('budgetReadout');
  const predictionReadout = document.getElementById('predictionReadout');
  const nearestReadout = document.getElementById('nearestReadout');
  const fitReadout = document.getElementById('fitReadout');
  const nearestText = document.getElementById('nearestText');
  const nearestBody = document.querySelector('#nearestTable tbody');
  const svg = document.getElementById('explorerSvg');
  const capacityContextSlider = document.getElementById('capacityContextSlider');
  const capacityContextValue = document.getElementById('capacityContextValue');
  const marginSlider = document.getElementById('marginSlider');
  const marginValue = document.getElementById('marginValue');
  const smallPolicyReadout = document.getElementById('smallPolicyReadout');
  const matchPolicyReadout = document.getElementById('matchPolicyReadout');
  const smallBatchReadout = document.getElementById('smallBatchReadout');
  const matchBatchReadout = document.getElementById('matchBatchReadout');
  const capacityTableBody = document.querySelector('#capacityTable tbody');
  const capacitySvg = document.getElementById('capacitySvg');
  const contexts = [...new Set(samples.map(s => s.max_model_len).filter(Boolean))].sort((a,b)=>a-b);
  const fmt = new Intl.NumberFormat('en-US');
  const fmt1 = new Intl.NumberFormat('en-US', {maximumFractionDigits: 1});
  const finite = v => typeof v === 'number' && Number.isFinite(v);
  const metricLabels = {
    reported_concurrency: 'reported concurrency',
    completion_tok_s: 'completion tok/s',
    idle_memory_gib: 'idle GiB',
    load_memory_peak_gib: 'load peak GiB',
    latency_p95: 'p95 latency'
  };
  const metricModels = {
    reported_concurrency: 'reported_concurrency_empirical',
    completion_tok_s: 'completion_tok_s',
    idle_memory_gib: 'idle_memory_gib',
    load_memory_peak_gib: 'load_memory_peak_gib',
    latency_p95: 'latency_p95'
  };
  function predict(metric, ctx, seq, policy) {
    const model = models[metricModels[metric] || metric];
    if (!model || !model.ok || !model.coefficients) return null;
    const contextK = ctx / 1024;
    const features = {
      intercept: 1,
      context_k: contextK,
      log_context: Math.log(ctx),
      max_num_seqs: seq,
      context_k_x_seqs: contextK * seq,
      match_context_policy: policy === 'match_context' ? 1 : 0
    };
    let value = 0;
    for (const [name, coefficient] of Object.entries(model.coefficients)) {
      if (!finite(features[name])) return null;
      value += coefficient * features[name];
    }
    if (model.transform === 'log_response') return Math.max(0, Math.exp(value));
    return Math.max(0, value);
  }
  function predictFromModel(metric, ctx, seq, policy) {
    const model = models[metricModels[metric] || metric];
    if (!model || !model.ok) return null;
    if (model.transform === 'piecewise_loglog_by_policy') return predictPiecewise(model.points && model.points[policy], ctx);
    return predict(metric, ctx, seq, policy);
  }
  function predictPiecewise(points, ctx) {
    const clean = (points || []).filter(point => point[0] > 0 && point[1] > 0).sort((a,b)=>a[0]-b[0]);
    if (!clean.length) return null;
    if (ctx <= clean[0][0]) return clean[0][1];
    if (ctx >= clean[clean.length - 1][0]) return clean[clean.length - 1][1];
    for (let i = 1; i < clean.length; i++) {
      const left = clean[i - 1], right = clean[i];
      if (ctx <= right[0]) {
        const t = (Math.log(ctx) - Math.log(left[0])) / (Math.log(right[0]) - Math.log(left[0]));
        return Math.exp(Math.log(left[1]) + t * (Math.log(right[1]) - Math.log(left[1])));
      }
    }
    return clean[clean.length - 1][1];
  }
  function batchTokens(policy, ctx) {
    return policy === 'small' ? Math.min(ctx, 8192) : ctx;
  }
  function capacityFor(policy, ctx) {
    const model = models.reported_concurrency_empirical;
    if (!model || !model.ok) return null;
    return predictPiecewise(model.points && model.points[policy], ctx);
  }
  function recommendedSessions(capacity, margin) {
    return finite(capacity) ? Math.max(1, Math.floor(capacity * margin)) : null;
  }
  function shortPolicyLabel(policy) {
    return policy === 'match_context' ? 'match' : policy;
  }
  function drawCapacityPlanner(ctx, margin) {
    if (!capacitySvg) return;
    const width = 940, height = 560;
    const pad = {left: 76, right: 34, top: 32, bottom: 66};
    const xmin = Math.min(...contexts);
    const xmax = Math.max(...contexts);
    const model = models.reported_concurrency_empirical || {};
    const smallPoints = ((model.points && model.points.small) || []).filter(point => point[0] > 0 && point[1] > 0);
    const matchPoints = ((model.points && model.points.match_context) || []).filter(point => point[0] > 0 && point[1] > 0);
    const allValues = smallPoints.concat(matchPoints).map(point => point[1]);
    const maxValue = Math.max(2, ...allValues, capacityFor('small', ctx) || 1, capacityFor('match_context', ctx) || 1);
    const yMax = Math.pow(2, Math.ceil(Math.log2(maxValue)));
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const x = value => pad.left + (Math.log(value) - Math.log(xmin)) / (Math.log(xmax) - Math.log(xmin)) * plotW;
    const y = value => pad.top + (1 - (Math.log(Math.max(1, value)) / Math.log(yMax))) * plotH;
    const linePath = (policy, multiplier) => {
      const steps = 100;
      const parts = [];
      for (let i = 0; i <= steps; i++) {
        const c = Math.exp(Math.log(xmin) + (Math.log(xmax) - Math.log(xmin)) * i / steps);
        const q = capacityFor(policy, c);
        if (!finite(q)) continue;
        const px = x(c).toFixed(1);
        const py = y(Math.max(1, q * multiplier)).toFixed(1);
        parts.push(`${parts.length ? 'L' : 'M'}${px},${py}`);
      }
      return parts.join(' ');
    };
    const xTicks = contexts.filter(tick => tick !== 98304 || !contexts.includes(100000));
    const yTicks = [1, 2, 4, 8, 16, 32, 64, 128, 256].filter(value => value <= yMax);
    let out = [`<rect x="0" y="0" width="${width}" height="${height}" fill="white"/>`];
    out.push(`<text x="${pad.left}" y="22" font-size="14" font-weight="700" fill="#1d2522">capacity curve Qp(c) and recommended s</text>`);
    for (const tick of xTicks) {
      const px = x(tick);
      out.push(`<line x1="${px.toFixed(1)}" y1="${pad.top}" x2="${px.toFixed(1)}" y2="${height-pad.bottom}" stroke="#eef2ee"/>`);
      out.push(`<text x="${px.toFixed(1)}" y="${height-34}" text-anchor="middle" font-size="10" fill="#5f6d68">${fmtCompact(tick)}</text>`);
    }
    for (const tick of yTicks) {
      const py = y(tick);
      out.push(`<line x1="${pad.left}" y1="${py.toFixed(1)}" x2="${width-pad.right}" y2="${py.toFixed(1)}" stroke="#eef2ee"/>`);
      out.push(`<text x="${pad.left-10}" y="${(py+4).toFixed(1)}" text-anchor="end" font-size="11" fill="#5f6d68">${tick}</text>`);
    }
    out.push(`<line x1="${pad.left}" y1="${height-pad.bottom}" x2="${width-pad.right}" y2="${height-pad.bottom}" stroke="#44514d"/>`);
    out.push(`<line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height-pad.bottom}" stroke="#44514d"/>`);
    out.push(`<text x="${width/2}" y="${height-8}" text-anchor="middle" font-size="12" fill="#33413c">context window tokens, log scale</text>`);
    out.push(`<text x="18" y="${height/2}" transform="rotate(-90 18 ${height/2})" text-anchor="middle" font-size="12" fill="#33413c">concurrent sessions, log scale</text>`);
    out.push(`<path d="${linePath('small', 1)}" fill="none" stroke="#176b64" stroke-width="3"/>`);
    out.push(`<path d="${linePath('small', margin)}" fill="none" stroke="#176b64" stroke-width="2" stroke-dasharray="7 6"/>`);
    out.push(`<path d="${linePath('match_context', 1)}" fill="none" stroke="#694f9e" stroke-width="3"/>`);
    out.push(`<path d="${linePath('match_context', margin)}" fill="none" stroke="#694f9e" stroke-width="2" stroke-dasharray="7 6"/>`);
    for (const [policy, points, color] of [['small', smallPoints, '#176b64'], ['match_context', matchPoints, '#694f9e']]) {
      for (const point of points) {
        out.push(`<circle cx="${x(point[0]).toFixed(1)}" cy="${y(point[1]).toFixed(1)}" r="5.8" fill="white" stroke="${color}" stroke-width="2"><title>${policy}\\ncontext=${fmt.format(point[0])}\\nreported q=${formatMaybe(point[1])}</title></circle>`);
      }
    }
    const selectedX = x(ctx);
    out.push(`<line x1="${selectedX.toFixed(1)}" y1="${pad.top}" x2="${selectedX.toFixed(1)}" y2="${height-pad.bottom}" stroke="#1d2522" stroke-dasharray="4 5"/>`);
    for (const [policy, color] of [['small', '#176b64'], ['match_context', '#694f9e']]) {
      const q = capacityFor(policy, ctx);
      const s = recommendedSessions(q, margin);
      if (!finite(q) || !finite(s)) continue;
      out.push(`<circle cx="${selectedX.toFixed(1)}" cy="${y(q).toFixed(1)}" r="7" fill="${color}" stroke="white" stroke-width="2"><title>${policy} q=${formatMaybe(q)}</title></circle>`);
      out.push(`<rect x="${(selectedX-5).toFixed(1)}" y="${(y(s)-5).toFixed(1)}" width="10" height="10" fill="white" stroke="${color}" stroke-width="2"><title>${policy} recommended s=${s}</title></rect>`);
    }
    capacitySvg.innerHTML = out.join('');
  }
  function renderCapacityPlanner() {
    if (!capacitySvg || !capacityContextSlider || !marginSlider) return;
    const ctx = Number(capacityContextSlider.value);
    const margin = Number(marginSlider.value);
    const smallQ = capacityFor('small', ctx);
    const matchQ = capacityFor('match_context', ctx);
    const smallS = recommendedSessions(smallQ, margin);
    const matchS = recommendedSessions(matchQ, margin);
    capacityContextValue.value = fmt.format(ctx);
    marginValue.value = margin.toFixed(2);
    smallPolicyReadout.textContent = finite(smallQ) ? `q ${formatMaybe(smallQ)}, s ${smallS}` : '';
    matchPolicyReadout.textContent = finite(matchQ) ? `q ${formatMaybe(matchQ)}, s ${matchS}` : '';
    smallBatchReadout.textContent = `${fmt.format(batchTokens('small', ctx))} tokens`;
    matchBatchReadout.textContent = `${fmt.format(batchTokens('match_context', ctx))} tokens`;
    capacityTableBody.innerHTML = [
      ['small', batchTokens('small', ctx), smallQ, smallS],
      ['match_context', batchTokens('match_context', ctx), matchQ, matchS]
    ].map(row => `<tr><td><code>${shortPolicyLabel(row[0])}</code></td><td class="num">${fmt.format(row[1])}</td><td class="num">${formatMaybe(row[2])}</td><td class="num">${finite(row[3]) ? row[3] : ''}</td></tr>`).join('');
    drawCapacityPlanner(ctx, margin);
  }
  function nearestRows(ctx, seq, policy, metric) {
    return samples.map(row => {
      const dc = Math.abs(Math.log(row.max_model_len) - Math.log(ctx));
      const ds = Math.abs(row.max_num_seqs - seq) / 32;
      const dp = row.batch_policy === policy ? 0 : 0.8;
      const missing = finite(row[metric]) ? 0 : 0.25;
      return {row, score: dc + ds + dp + missing};
    }).sort((a,b)=>a.score-b.score).slice(0, 9).map(item => item.row);
  }
  function statusClass(row) {
    if (row.status === 'load_complete') return 'ok';
    if (row.status === 'startup_service_exit') return 'bad';
    return 'warn';
  }
  function colorAt(t) {
    const stops = [
      [0.00, [247, 239, 211]],
      [0.45, [101, 169, 142]],
      [1.00, [51, 66, 141]]
    ];
    const clamped = Math.max(0, Math.min(1, t));
    let left = stops[0], right = stops[stops.length - 1];
    for (let i = 1; i < stops.length; i++) {
      if (clamped <= stops[i][0]) {
        left = stops[i - 1];
        right = stops[i];
        break;
      }
    }
    const local = (clamped - left[0]) / (right[0] - left[0] || 1);
    const rgb = left[1].map((value, index) => Math.round(value + (right[1][index] - value) * local));
    return `rgb(${rgb.join(',')})`;
  }
  function gridValues(metric, policy) {
    const xmin = Math.min(...contexts);
    const xmax = Math.max(...contexts);
    const values = [];
    for (let ix = 0; ix < 60; ix++) {
      const ctx = Math.exp(Math.log(xmin) + (Math.log(xmax) - Math.log(xmin)) * ix / 59);
      for (let iy = 0; iy < 32; iy++) {
        const seq = 1 + iy;
        const value = predictFromModel(metric, ctx, seq, policy);
        if (finite(value)) values.push(value);
      }
    }
    values.sort((a,b)=>a-b);
    const lo = values[Math.floor(values.length * 0.03)] ?? 0;
    const hi = values[Math.floor(values.length * 0.97)] ?? 1;
    return {lo, hi: hi <= lo ? lo + 1 : hi};
  }
  function draw(ctx, seq, policy, metric) {
    const width = 940, height = 590;
    const pad = {left: 76, right: 102, top: 34, bottom: 66};
    const xmin = Math.min(...contexts);
    const xmax = Math.max(...contexts);
    const policyRows = samples.filter(s => s.batch_policy === policy);
    const yTop = 32;
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const x = value => pad.left + (Math.log(value) - Math.log(xmin)) / (Math.log(xmax) - Math.log(xmin)) * plotW;
    const y = value => pad.top + (1 - (value - 1) / (yTop - 1)) * plotH;
    const yTicks = [1, 4, 8, 16, 24, 32];
    const xTicks = contexts;
    const domain = gridValues(metric, policy);
    let out = [`<rect x="0" y="0" width="${width}" height="${height}" fill="white"/>`];
    out.push(`<text x="${pad.left}" y="22" font-size="14" font-weight="700" fill="#1d2522">${escapeHtml(metricLabels[metric])} fitted surface</text>`);
    for (let ix = 0; ix < 72; ix++) {
      const c0 = Math.exp(Math.log(xmin) + (Math.log(xmax) - Math.log(xmin)) * ix / 72);
      const c1 = Math.exp(Math.log(xmin) + (Math.log(xmax) - Math.log(xmin)) * (ix + 1) / 72);
      for (let iy = 0; iy < 32; iy++) {
        const s0 = 1 + iy;
        const value = predictFromModel(metric, (c0 + c1) / 2, s0 + 0.5, policy);
        if (!finite(value)) continue;
        const t = (value - domain.lo) / (domain.hi - domain.lo);
        out.push(`<rect x="${x(c0).toFixed(1)}" y="${y(s0+1).toFixed(1)}" width="${Math.max(1, x(c1)-x(c0)+0.4).toFixed(1)}" height="${Math.max(1, y(s0)-y(s0+1)+0.4).toFixed(1)}" fill="${colorAt(t)}" opacity="0.88"/>`);
      }
    }
    let lastXLabel = -Infinity;
    for (const tick of xTicks) {
      const px = x(tick);
      out.push(`<line x1="${px.toFixed(1)}" y1="${pad.top}" x2="${px.toFixed(1)}" y2="${height-pad.bottom}" stroke="rgba(255,255,255,.72)"/>`);
      const suppressCrowdedLabel = tick === 98304 && contexts.includes(100000);
      if (!suppressCrowdedLabel && px - lastXLabel >= 44) {
        out.push(`<text x="${px.toFixed(1)}" y="${height-34}" text-anchor="middle" font-size="10" fill="#5f6d68">${fmtCompact(tick)}</text>`);
        lastXLabel = px;
      }
    }
    for (const tick of yTicks) {
      const py = y(tick);
      out.push(`<line x1="${pad.left}" y1="${py.toFixed(1)}" x2="${width-pad.right}" y2="${py.toFixed(1)}" stroke="rgba(255,255,255,.72)"/>`);
      out.push(`<text x="${pad.left-10}" y="${(py+4).toFixed(1)}" text-anchor="end" font-size="11" fill="#5f6d68">${tick}</text>`);
    }
    out.push(`<line x1="${pad.left}" y1="${height-pad.bottom}" x2="${width-pad.right}" y2="${height-pad.bottom}" stroke="#44514d"/>`);
    out.push(`<line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height-pad.bottom}" stroke="#44514d"/>`);
    out.push(`<text x="${width/2}" y="${height-8}" text-anchor="middle" font-size="12" fill="#33413c">context window tokens, log scale</text>`);
    out.push(`<text x="18" y="${height/2}" transform="rotate(-90 18 ${height/2})" text-anchor="middle" font-size="12" fill="#33413c">requested concurrency</text>`);
    for (const row of policyRows) {
      const px = x(row.max_model_len);
      const py = y(row.max_num_seqs);
      const actual = row[metric];
      const predicted = predictFromModel(metric, row.max_model_len, row.max_num_seqs, policy);
      const stroke = row.status === 'startup_service_exit' ? '#9d2f2f' : row.status === 'startup_only' ? '#9b5f00' : '#1d2522';
      if (finite(actual)) {
        const t = (actual - domain.lo) / (domain.hi - domain.lo);
        const residual = finite(predicted) ? actual - predicted : null;
        out.push(`<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="6.2" fill="${colorAt(t)}" stroke="${stroke}" stroke-width="1.7"><title>${escapeHtml(row.candidate_id)}\\nactual=${formatMaybe(actual)}\\nmodel=${formatMaybe(predicted)}\\nresidual=${formatMaybe(residual)}\\nstatus=${row.status}</title></circle>`);
      } else {
        out.push(`<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="5.5" fill="white" fill-opacity="0.72" stroke="${stroke}" stroke-width="1.5" stroke-dasharray="2 2"><title>${escapeHtml(row.candidate_id)}\\nno actual ${escapeHtml(metricLabels[metric])}\\nstatus=${row.status}</title></circle>`);
      }
    }
    const sx = x(ctx);
    const sy = y(seq);
    out.push(`<line x1="${sx.toFixed(1)}" y1="${pad.top}" x2="${sx.toFixed(1)}" y2="${height-pad.bottom}" stroke="#176b64" stroke-dasharray="4 5"/>`);
    out.push(`<line x1="${pad.left}" y1="${sy.toFixed(1)}" x2="${width-pad.right}" y2="${sy.toFixed(1)}" stroke="#176b64" stroke-dasharray="4 5"/>`);
    out.push(`<rect x="${(sx-7).toFixed(1)}" y="${(sy-7).toFixed(1)}" width="14" height="14" fill="none" stroke="#176b64" stroke-width="3"><title>selected: context=${ctx}, seqs=${seq}</title></rect>`);
    for (let i = 0; i <= 8; i++) {
      const t = i / 8;
      out.push(`<rect x="${width-72}" y="${(pad.top + i * 36).toFixed(1)}" width="18" height="36.5" fill="${colorAt(1-t)}"/>`);
    }
    out.push(`<text x="${width-47}" y="${pad.top+8}" font-size="10" fill="#33413c">${formatMaybe(domain.hi)}</text>`);
    out.push(`<text x="${width-47}" y="${pad.top+292}" font-size="10" fill="#33413c">${formatMaybe(domain.lo)}</text>`);
    svg.innerHTML = out.join('');
  }
  function fmtCompact(value) {
    if (value >= 1000) return `${Math.round(value / 1000)}k`;
    return String(value);
  }
  function formatMaybe(value, digits = 1) {
    return finite(value) ? new Intl.NumberFormat('en-US', {maximumFractionDigits: digits}).format(value) : '';
  }
  function escapeHtml(value) {
    return String(value).replace(/[&<>"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]));
  }
  function render() {
    const ctx = Number(contextSlider.value);
    const seq = Number(seqSlider.value);
    const policy = policySelect.value;
    const metric = metricSelect.value;
    const budget = ctx * seq;
    const prediction = predictFromModel(metric, ctx, seq, policy);
    const model = models[metricModels[metric] || metric] || {};
    contextValue.value = fmt.format(ctx);
    seqValue.value = fmt.format(seq);
    budgetReadout.textContent = `${fmt.format(budget)} tokens`;
    predictionReadout.textContent = `${formatMaybe(prediction)} ${metricLabels[metric]}`;
    fitReadout.textContent = model.ok ? `R2 ${formatMaybe(model.r2, 2)}, MAE ${formatMaybe(model.mae, 2)}` : 'no model';
    const nearest = nearestRows(ctx, seq, policy, metric);
    const nearestWithActual = nearest.find(row => finite(row[metric]));
    nearestReadout.textContent = nearestWithActual ? `${formatMaybe(nearestWithActual[metric])} at ${nearestWithActual.candidate_id}` : 'no nearby actual';
    nearestText.textContent = `Surface is the fitted metric model for the selected policy and requested concurrency. Use the capacity planner above for max safe concurrency.`;
    nearestBody.innerHTML = nearest.map(row => `<tr>
      <td><code>${escapeHtml(row.candidate_id)}</code></td>
      <td class="num">${fmt.format(row.max_model_len)}</td>
      <td class="num">${fmt.format(row.max_num_seqs)}</td>
      <td><span class="badge ${statusClass(row)}">${escapeHtml(row.status)}</span></td>
      <td class="num">${formatMaybe(row[metric])}</td>
      <td class="num">${formatMaybe(predictFromModel(metric, row.max_model_len, row.max_num_seqs, row.batch_policy))}</td>
      <td class="num">${finite(row[metric]) ? formatMaybe(row[metric] - predictFromModel(metric, row.max_model_len, row.max_num_seqs, row.batch_policy)) : ''}</td>
      <td class="num">${formatMaybe(row.reported_concurrency)}</td>
      <td class="num">${formatMaybe(row.completion_tok_s, 1)}</td>
    </tr>`).join('');
    draw(ctx, seq, policy, metric);
  }
  for (const el of [capacityContextSlider, marginSlider]) {
    if (el) el.addEventListener('input', renderCapacityPlanner);
  }
  for (const el of [contextSlider, seqSlider, policySelect, metricSelect]) el.addEventListener('input', render);
  renderCapacityPlanner();
  render();
})();
</script>"""


def write_html(path: Path, rows: list[dict], raw_rows: list[dict], models: dict) -> None:
    statuses = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    top_throughput = top_rows(rows, "completion_tok_s", 8)
    context_rows = context_summary(rows)
    boundaries = startup_boundaries(raw_rows)
    html = [
        "<!doctype html>",
        '<meta charset="utf-8">',
        "<title>Gemma 4 vLLM Resource Sweep</title>",
        '<script>window.MathJax={tex:{inlineMath:[[\'\\\\(\',\'\\\\)\'],[\'$\',\'$\']]}};</script>',
        '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>',
        page_css(),
        "<h1>Gemma 4 vLLM Resource Sweep</h1>",
        "<p>Standalone vLLM sweep for <code>nvidia/Gemma-4-26B-A4B-NVFP4</code>. Generated from machine-readable JSONL results.</p>",
        f"<p><strong>Rows:</strong> {len(rows)}. <strong>Status counts:</strong> {escape(', '.join(f'{k}={v}' for k, v in sorted(statuses.items())))}.</p>",
        "<h2>Key Findings</h2>",
        "<ul>",
    ]
    if top_throughput:
        html.append(f"<li>Best output throughput: <code>{fmt(top_throughput[0]['completion_tok_s'])}</code> completion tok/s at <code>{escape(top_throughput[0]['candidate_id'])}</code>.</li>")
    html.extend(
        [
            "<li>The small batched-token policy is the stable high-context path; match-context batching hits capacity limits earlier.</li>",
            "<li>Startup exits were CUTLASS FP4 MoE assertion boundaries, not observed machine OOMs.</li>",
            "<li>High-risk rows were measured at startup/idle and load was skipped when the guard said it was unsafe.</li>",
            "</ul>",
            *capacity_planner_html(),
            *capacity_simplification_html(),
            *interactive_explorer_html(rows, models),
            "<h2>Plots</h2>",
            '<div class="grid">',
            '<figure><img src="plots/idle-memory-by-context.svg" alt="Idle memory by context"><figcaption>Idle memory by context.</figcaption></figure>',
            '<figure><img src="plots/load-peak-memory-by-context.svg" alt="Load peak memory by context"><figcaption>Loaded peak memory by context.</figcaption></figure>',
            '<figure><img src="plots/reported-concurrency-by-context.svg" alt="Reported concurrency by context"><figcaption>vLLM reported concurrency capacity by context.</figcaption></figure>',
            '<figure><img src="plots/throughput-by-concurrency.svg" alt="Throughput by concurrency"><figcaption>Output throughput by requested concurrency.</figcaption></figure>',
            '<figure><img src="plots/latency-p95-by-concurrency.svg" alt="Latency by concurrency"><figcaption>P95 latency by requested concurrency.</figcaption></figure>',
            "</div>",
        ]
    )
    html.extend(abstract_model_html())
    html.extend(
        [
            "<h2>Fit Quality</h2>",
            "<p>The coefficients for these model families are stored in <code>models/linear-models.json</code>. The report formulas define the abstract system; this table only reports fit quality.</p>",
            '<table><tr><th>target</th><th class="num">rows</th><th class="num">R2</th><th class="num">MAE</th><th class="num">RMSE</th></tr>',
        ]
    )
    for name, model in models.items():
        html.append(model_quality_html_row(name, model))
    html.extend(["</table>", "<h2>Context Summary</h2>", '<table><tr><th class="num">context</th><th class="num">rows</th><th class="num">load complete</th><th class="num">startup only</th><th class="num">startup exits</th><th class="num">max loaded seqs</th><th class="num">best completion tok/s</th><th class="num">max reported concurrency</th></tr>'])
    for row in context_rows:
        html.append(
            f"<tr><td class=\"num\">{row['context']}</td><td class=\"num\">{row['rows']}</td><td class=\"num\">{row['load_complete']}</td><td class=\"num\">{row['startup_only']}</td><td class=\"num\">{row['startup_service_exit']}</td><td class=\"num\">{row['max_loaded_seqs']}</td><td class=\"num\">{fmt(row['best_completion_tok_s'])}</td><td class=\"num\">{fmt(row['max_reported_concurrency'])}</td></tr>"
        )
    html.extend(["</table>", "<h2>Top Throughput Rows</h2>", '<table><tr><th>candidate</th><th class="num">context</th><th class="num">seqs</th><th>policy</th><th class="num">completion tok/s</th><th class="num">total tok/s</th><th class="num">p95 latency</th></tr>'])
    for row in top_throughput:
        html.append(
            f"<tr><td><code>{escape(row['candidate_id'])}</code></td><td class=\"num\">{row['max_model_len']}</td><td class=\"num\">{row['max_num_seqs']}</td><td>{escape(row['batch_policy'])}</td><td class=\"num\">{fmt(row['completion_tok_s'])}</td><td class=\"num\">{fmt(row['total_tok_s'])}</td><td class=\"num\">{fmt(row['latency_p95'])}</td></tr>"
        )
    html.extend(["</table>", "<h2>Startup Boundaries</h2>", "<ul>"])
    for boundary in boundaries:
        html.append(f"<li>{escape(boundary)}</li>")
    html.extend(["</ul>", '<p>Full Markdown report: <a href="summary.md">summary.md</a>. Flattened CSV: <a href="tables/measurements.csv">measurements.csv</a>. Model JSON: <a href="models/linear-models.json">linear-models.json</a>.</p>'])
    path.write_text("\n".join(html) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bytes_to_gib(value: object) -> float | None:
    n = number(value)
    if n is None:
        return None
    return round(n / 1024**3, 3)


def safe_product(left: object, right: object, scale: float = 1.0) -> float | None:
    a = number(left)
    b = number(right)
    if a is None or b is None:
        return None
    return round((a * b) / scale, 6)


def number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: object) -> str:
    n = number(value)
    return "" if n is None else f"{n:.3f}"


def top_rows(rows: list[dict], key: str, limit: int) -> list[dict]:
    usable = [row for row in rows if number(row.get(key)) is not None]
    return sorted(usable, key=lambda row: number(row.get(key)) or float("-inf"), reverse=True)[:limit]


def context_summary(rows: list[dict]) -> list[dict]:
    by_context: dict[int, list[dict]] = {}
    for row in rows:
        context = int(number(row.get("max_model_len")) or 0)
        by_context.setdefault(context, []).append(row)
    out = []
    for context, group in sorted(by_context.items()):
        loaded = [row for row in group if row.get("status") == "load_complete"]
        out.append(
            {
                "context": context,
                "rows": len(group),
                "load_complete": sum(1 for row in group if row.get("status") == "load_complete"),
                "startup_only": sum(1 for row in group if row.get("status") == "startup_only"),
                "startup_service_exit": sum(1 for row in group if row.get("status") == "startup_service_exit"),
                "max_loaded_seqs": max([int(number(row.get("max_num_seqs")) or 0) for row in loaded], default=0),
                "best_completion_tok_s": max([number(row.get("completion_tok_s")) or 0 for row in loaded], default=None),
                "max_reported_concurrency": max([number(row.get("reported_concurrency")) or 0 for row in group], default=None),
            }
        )
    return out


def abstract_model_markdown() -> list[str]:
    return [
        "",
        "## Abstract System",
        "",
        "The sweep is a system with startup parameters as inputs and resource/performance measurements as outputs. The symbols below are single-letter quantities used by the formulas.",
        "",
        "| symbol | quantity |",
        "| --- | --- |",
        "| $c$ | context window, measured in 1024-token units |",
        "| $s$ | requested maximum concurrent sequences |",
        "| $b$ | maximum batched tokens |",
        "| $u$ | vLLM GPU memory utilization target |",
        "| $p$ | batching policy indicator, where $p=0$ for `small` and $p=1$ for `match_context` |",
        "| $v$ | requested token budget |",
        "| $k$ | vLLM-reported KV cache token capacity |",
        "| $q$ | vLLM-reported maximum concurrency for the configured context |",
        "| $m$ | idle service memory |",
        "| $h$ | idle service peak memory |",
        "| $a$ | loaded service peak memory |",
        "| $y$ | completion token throughput |",
        "| $z$ | total token throughput |",
        "| $l$ | p95 request latency |",
        "| $w$ | free swap memory |",
        "| $r$ | risk guard indicator, where $r=1$ means load is blocked by policy |",
        "| $o$ | startup outcome |",
        "| $d$ | load decision, where $d=1$ means load was run |",
        "",
        "Derived budget:",
        "",
        "$$v = cs$$",
        "",
        "Startup and capacity:",
        "",
        "$$o = O(c,s,b,u,p)$$",
        "",
        "$$k = K(c,s,b,u,p)$$",
        "",
        "$$q = Q(c,s,b,u,p)$$",
        "",
        "Load is run only when the configured concurrency is inside reported capacity and safety bounds:",
        "",
        "$$d = \\mathbb{1}\\{q \\ge s \\;\\land\\; m < m_{\\max} \\;\\land\\; w > w_{\\min} \\;\\land\\; r = 0\\}$$",
        "",
        "Memory model family:",
        "",
        "$$m = M(c,s,b,u,p,cs) + \\epsilon_m$$",
        "",
        "$$h = H(c,s,b,u,p,cs) + \\epsilon_h$$",
        "",
        "$$a = A(c,s,b,u,p,cs) + \\epsilon_a, \\quad d=1$$",
        "",
        "Performance model family:",
        "",
        "$$y = Y(c,s,b,u,p,cs) + \\epsilon_y, \\quad d=1$$",
        "",
        "$$z = Z(c,s,b,u,p,cs) + \\epsilon_z, \\quad d=1$$",
        "",
        "$$l = L(c,s,b,u,p,cs) + \\epsilon_l, \\quad d=1$$",
        "",
        "In this run, the fitted instances of $M,H,A,Q,Y,L$ are simple continuous response models over the observed rows. Most are linear interaction models; the interactive capacity surface uses a policy-specific piecewise log-log empirical model for $Q$ so it follows the measured capacity curve. The abstract system is the important part; the numeric coefficients are only run-specific estimates.",
    ]


def abstract_model_html() -> list[str]:
    return [
        "<h2>Abstract System</h2>",
        "<p>The sweep is a system with startup parameters as inputs and resource/performance measurements as outputs. The symbols below are single-letter quantities used by the formulas.</p>",
        "<table><tr><th>symbol</th><th>quantity</th></tr>",
        "<tr><td>\\(c\\)</td><td>context window, measured in 1024-token units</td></tr>",
        "<tr><td>\\(s\\)</td><td>requested maximum concurrent sequences</td></tr>",
        "<tr><td>\\(b\\)</td><td>maximum batched tokens</td></tr>",
        "<tr><td>\\(u\\)</td><td>vLLM GPU memory utilization target</td></tr>",
        "<tr><td>\\(p\\)</td><td>batching policy indicator, where \\(p=0\\) for <code>small</code> and \\(p=1\\) for <code>match_context</code></td></tr>",
        "<tr><td>\\(v\\)</td><td>requested token budget</td></tr>",
        "<tr><td>\\(k\\)</td><td>vLLM-reported KV cache token capacity</td></tr>",
        "<tr><td>\\(q\\)</td><td>vLLM-reported maximum concurrency for the configured context</td></tr>",
        "<tr><td>\\(m\\)</td><td>idle service memory</td></tr>",
        "<tr><td>\\(h\\)</td><td>idle service peak memory</td></tr>",
        "<tr><td>\\(a\\)</td><td>loaded service peak memory</td></tr>",
        "<tr><td>\\(y\\)</td><td>completion token throughput</td></tr>",
        "<tr><td>\\(z\\)</td><td>total token throughput</td></tr>",
        "<tr><td>\\(l\\)</td><td>p95 request latency</td></tr>",
        "<tr><td>\\(w\\)</td><td>free swap memory</td></tr>",
        "<tr><td>\\(r\\)</td><td>risk guard indicator, where \\(r=1\\) means load is blocked by policy</td></tr>",
        "<tr><td>\\(o\\)</td><td>startup outcome</td></tr>",
        "<tr><td>\\(d\\)</td><td>load decision, where \\(d=1\\) means load was run</td></tr>",
        "</table>",
        "<p>Derived budget:</p>",
        "<p>\\[v = cs\\]</p>",
        "<p>Startup and capacity:</p>",
        "<p>\\[o = O(c,s,b,u,p)\\]</p>",
        "<p>\\[k = K(c,s,b,u,p)\\]</p>",
        "<p>\\[q = Q(c,s,b,u,p)\\]</p>",
        "<p>Load is run only when the configured concurrency is inside reported capacity and safety bounds:</p>",
        "<p>\\[d = \\mathbb{1}\\{q \\ge s \\;\\land\\; m < m_{\\max} \\;\\land\\; w > w_{\\min} \\;\\land\\; r = 0\\}\\]</p>",
        "<p>Memory model family:</p>",
        "<p>\\[m = M(c,s,b,u,p,cs) + \\epsilon_m\\]</p>",
        "<p>\\[h = H(c,s,b,u,p,cs) + \\epsilon_h\\]</p>",
        "<p>\\[a = A(c,s,b,u,p,cs) + \\epsilon_a, \\quad d=1\\]</p>",
        "<p>Performance model family:</p>",
        "<p>\\[y = Y(c,s,b,u,p,cs) + \\epsilon_y, \\quad d=1\\]</p>",
        "<p>\\[z = Z(c,s,b,u,p,cs) + \\epsilon_z, \\quad d=1\\]</p>",
        "<p>\\[l = L(c,s,b,u,p,cs) + \\epsilon_l, \\quad d=1\\]</p>",
        "<p>In this run, the fitted instances of \\(M,H,A,Q,Y,L\\) are simple continuous response models over the observed rows. Most are linear interaction models; the interactive capacity surface uses a policy-specific piecewise log-log empirical model for \\(Q\\) so it follows the measured capacity curve. The abstract system is the important part; the numeric coefficients are only run-specific estimates.</p>",
    ]


def model_quality_table_row(name: str, model: dict) -> str:
    if not model.get("ok"):
        return f"| {name} | {model.get('rows', 0)} |  |  |  |"
    return f"| {name} | {model.get('rows')} | {fmt(model.get('r2'))} | {fmt(model.get('mae'))} | {fmt(model.get('rmse'))} |"


def model_quality_html_row(name: str, model: dict) -> str:
    if not model.get("ok"):
        return f"<tr><td>{escape(name)}</td><td class=\"num\">{model.get('rows', 0)}</td><td></td><td></td><td></td></tr>"
    return (
        f"<tr><td>{escape(name)}</td><td class=\"num\">{model.get('rows')}</td>"
        f"<td class=\"num\">{fmt(model.get('r2'))}</td><td class=\"num\">{fmt(model.get('mae'))}</td><td class=\"num\">{fmt(model.get('rmse'))}</td></tr>"
    )


def formula(name: str, model: dict) -> str:
    coefficients = model.get("coefficients") or {}
    pieces = []
    intercept = coefficients.get("intercept")
    if intercept is not None:
        pieces.append(f"{intercept:.4g}")
    for key, value in coefficients.items():
        if key == "intercept":
            continue
        sign = "+" if value >= 0 else "-"
        pieces.append(f"{sign} {abs(value):.4g}*{key}")
    return f"{name} ~= {' '.join(pieces)}"


def escape(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    raise SystemExit(main())
