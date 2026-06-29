---
title: SQLite Run Artifact Format
author: Bob <dutifulbob@gmail.com>
date: 2026-06-29
---

# SQLite Run Artifact Format

LocalPerf should write one canonical run artifact:

```text
runs/<run-id>.sqlite
```

That SQLite file is the source of truth for the run. Markdown, JSON, CSV, and
Parquet outputs are exports derived from it.

## Minimal Artifact

A minimal valid artifact is a SQLite database with:

- the `metadata` table,
- one `run` row,
- one `specs` row for the original spec,
- one `specs` row for the normalized spec.

Example inspection:

```sh
sqlite3 runs/diffusiongemma-20260629.sqlite \
  "select key, value from metadata order by key"
```

Required metadata:

```text
format_name     localperf_run
format_version  1
```

## File Rules

- The file extension should be `.sqlite`.
- The database must be readable by standard SQLite tooling.
- `PRAGMA foreign_keys=ON` must be used by LocalPerf.
- `PRAGMA user_version=1` should be set.
- Timestamps are UTC ISO 8601 strings with `Z`, for example
  `2026-06-29T02:30:55Z`.
- Durations are stored as milliseconds in `REAL` columns.
- Token counts and byte counts are stored as integers.
- Booleans are stored as `0` or `1`.
- JSON columns are stored as `TEXT` and should have `json_valid(...)` checks.
- Large raw logs and engine-native outputs live in `artifacts.content`.

SQLite may create temporary journal files while a benchmark is running. A
finished run should leave one canonical `.sqlite` file plus any optional
exports.

## Schema Overview

| Table | Purpose |
| --- | --- |
| `metadata` | Format and artifact-level key/value metadata. |
| `run` | One row describing the benchmark run. |
| `specs` | Original and normalized specs. |
| `engines` | Engine adapter definitions and detected versions. |
| `profiles` | Serve-time engine profiles. |
| `workloads` | Workload definitions. |
| `measurements` | One row per profile/workload/concurrency/repeat result. |
| `requests` | Per-request timing, token, and error rows. |
| `telemetry_series` | Definitions for sampled telemetry metrics. |
| `telemetry_samples` | Memory, GPU, process, and system telemetry samples. |
| `events` | Append-only lifecycle and diagnostic events. |
| `commands` | Exact subprocess commands and outcomes. |
| `artifacts` | Raw logs, raw benchmark JSON, stdout/stderr, generated blobs. |
| `reports` | Rendered report exports stored inside the artifact. |

## Schema DDL

This is the planned v1 schema. Implementation can add indexes and views, but
must preserve these table meanings.

```sql
PRAGMA foreign_keys = ON;
PRAGMA user_version = 1;

CREATE TABLE metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE run (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  status TEXT NOT NULL CHECK (status IN (
    'created', 'running', 'completed', 'failed', 'canceled'
  )),
  created_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  localperf_version TEXT,
  localperf_git_commit TEXT,
  hostname TEXT,
  username TEXT,
  cwd TEXT,
  command_line_json TEXT CHECK (
    command_line_json IS NULL OR json_valid(command_line_json)
  ),
  host_json TEXT CHECK (host_json IS NULL OR json_valid(host_json)),
  labels_json TEXT CHECK (labels_json IS NULL OR json_valid(labels_json)),
  notes TEXT
);

CREATE TABLE specs (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK (kind IN ('original', 'normalized')),
  format TEXT NOT NULL CHECK (format IN ('json')),
  content TEXT NOT NULL CHECK (json_valid(content)),
  sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (run_id, kind)
);

CREATE TABLE engines (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  type TEXT NOT NULL,
  managed INTEGER NOT NULL CHECK (managed IN (0, 1)),
  command TEXT,
  version TEXT,
  git_commit TEXT,
  endpoint_base_url TEXT,
  env_json TEXT CHECK (env_json IS NULL OR json_valid(env_json)),
  metadata_json TEXT CHECK (
    metadata_json IS NULL OR json_valid(metadata_json)
  ),
  UNIQUE (run_id, name)
);

CREATE TABLE profiles (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  engine_id TEXT NOT NULL REFERENCES engines(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  model TEXT NOT NULL,
  host TEXT,
  port INTEGER,
  endpoint_base_url TEXT,
  managed INTEGER NOT NULL CHECK (managed IN (0, 1)),
  context_window INTEGER,
  max_num_seqs INTEGER,
  max_num_batched_tokens INTEGER,
  gpu_memory_utilization REAL,
  enable_sleep_mode INTEGER CHECK (enable_sleep_mode IN (0, 1)),
  sleep_level INTEGER,
  serve_json TEXT CHECK (serve_json IS NULL OR json_valid(serve_json)),
  engine_args_json TEXT CHECK (
    engine_args_json IS NULL OR json_valid(engine_args_json)
  ),
  env_json TEXT CHECK (env_json IS NULL OR json_valid(env_json)),
  metadata_json TEXT CHECK (
    metadata_json IS NULL OR json_valid(metadata_json)
  ),
  UNIQUE (run_id, name)
);

CREATE TABLE workloads (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  traffic_json TEXT NOT NULL CHECK (json_valid(traffic_json)),
  concurrency_json TEXT NOT NULL CHECK (json_valid(concurrency_json)),
  samples INTEGER NOT NULL,
  repeats INTEGER NOT NULL DEFAULT 1,
  save_detailed INTEGER NOT NULL DEFAULT 1 CHECK (save_detailed IN (0, 1)),
  capture_prompt_text INTEGER NOT NULL DEFAULT 0 CHECK (
    capture_prompt_text IN (0, 1)
  ),
  metadata_json TEXT CHECK (
    metadata_json IS NULL OR json_valid(metadata_json)
  ),
  UNIQUE (run_id, name)
);

CREATE TABLE measurements (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  workload_id TEXT NOT NULL REFERENCES workloads(id) ON DELETE CASCADE,
  repeat_index INTEGER NOT NULL DEFAULT 0,
  concurrency INTEGER NOT NULL,
  samples_requested INTEGER NOT NULL,
  status TEXT NOT NULL CHECK (status IN (
    'planned', 'running', 'completed', 'failed', 'skipped', 'canceled'
  )),
  started_at TEXT,
  completed_at TEXT,
  wall_time_ms REAL,
  completed_requests INTEGER NOT NULL DEFAULT 0,
  failed_requests INTEGER NOT NULL DEFAULT 0,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  total_tokens INTEGER,
  aggregate_output_tok_s REAL,
  per_user_output_tok_s REAL,
  aggregate_total_tok_s REAL,
  latency_stats_json TEXT CHECK (
    latency_stats_json IS NULL OR json_valid(latency_stats_json)
  ),
  ttft_stats_json TEXT CHECK (
    ttft_stats_json IS NULL OR json_valid(ttft_stats_json)
  ),
  tpot_stats_json TEXT CHECK (
    tpot_stats_json IS NULL OR json_valid(tpot_stats_json)
  ),
  itl_stats_json TEXT CHECK (
    itl_stats_json IS NULL OR json_valid(itl_stats_json)
  ),
  memory_stats_json TEXT CHECK (
    memory_stats_json IS NULL OR json_valid(memory_stats_json)
  ),
  raw_result_artifact_id INTEGER REFERENCES artifacts(id),
  error_type TEXT,
  error_message TEXT,
  metadata_json TEXT CHECK (
    metadata_json IS NULL OR json_valid(metadata_json)
  ),
  UNIQUE (
    run_id,
    profile_id,
    workload_id,
    repeat_index,
    concurrency
  )
);

CREATE TABLE requests (
  id INTEGER PRIMARY KEY,
  measurement_id INTEGER NOT NULL REFERENCES measurements(id) ON DELETE CASCADE,
  request_index INTEGER NOT NULL,
  request_id TEXT,
  status TEXT NOT NULL CHECK (status IN ('completed', 'failed', 'canceled')),
  started_at TEXT NOT NULL,
  first_token_at TEXT,
  completed_at TEXT,
  latency_ms REAL,
  ttft_ms REAL,
  tpot_ms REAL,
  itl_mean_ms REAL,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  total_tokens INTEGER,
  output_tok_s REAL,
  prompt_sha256 TEXT,
  prompt_text TEXT,
  response_sha256 TEXT,
  response_text TEXT,
  error_type TEXT,
  error_code TEXT,
  error_message TEXT,
  response_metadata_json TEXT CHECK (
    response_metadata_json IS NULL OR json_valid(response_metadata_json)
  ),
  UNIQUE (measurement_id, request_index)
);

CREATE TABLE telemetry_series (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  source TEXT NOT NULL,
  name TEXT NOT NULL,
  unit TEXT,
  scope TEXT,
  tags_json TEXT CHECK (tags_json IS NULL OR json_valid(tags_json)),
  UNIQUE (run_id, source, name, scope, tags_json)
);

CREATE TABLE telemetry_samples (
  id INTEGER PRIMARY KEY,
  series_id INTEGER NOT NULL REFERENCES telemetry_series(id) ON DELETE CASCADE,
  timestamp TEXT NOT NULL,
  value_real REAL,
  value_integer INTEGER,
  value_text TEXT,
  measurement_id INTEGER REFERENCES measurements(id) ON DELETE SET NULL
);

CREATE TABLE events (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  timestamp TEXT NOT NULL,
  level TEXT NOT NULL CHECK (level IN (
    'debug', 'info', 'warn', 'error'
  )),
  type TEXT NOT NULL,
  profile_id TEXT REFERENCES profiles(id) ON DELETE SET NULL,
  workload_id TEXT REFERENCES workloads(id) ON DELETE SET NULL,
  measurement_id INTEGER REFERENCES measurements(id) ON DELETE SET NULL,
  message TEXT,
  data_json TEXT CHECK (data_json IS NULL OR json_valid(data_json))
);

CREATE TABLE commands (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  measurement_id INTEGER REFERENCES measurements(id) ON DELETE SET NULL,
  profile_id TEXT REFERENCES profiles(id) ON DELETE SET NULL,
  phase TEXT NOT NULL,
  cwd TEXT,
  argv_json TEXT NOT NULL CHECK (json_valid(argv_json)),
  env_json TEXT CHECK (env_json IS NULL OR json_valid(env_json)),
  started_at TEXT,
  completed_at TEXT,
  exit_code INTEGER,
  status TEXT NOT NULL CHECK (status IN (
    'planned', 'running', 'completed', 'failed', 'canceled'
  )),
  stdout_artifact_id INTEGER REFERENCES artifacts(id),
  stderr_artifact_id INTEGER REFERENCES artifacts(id),
  metadata_json TEXT CHECK (
    metadata_json IS NULL OR json_valid(metadata_json)
  )
);

CREATE TABLE artifacts (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  media_type TEXT NOT NULL,
  compression TEXT NOT NULL DEFAULT 'none' CHECK (
    compression IN ('none', 'gzip')
  ),
  content BLOB NOT NULL,
  content_size_bytes INTEGER NOT NULL,
  uncompressed_size_bytes INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  original_path TEXT,
  created_at TEXT NOT NULL,
  metadata_json TEXT CHECK (
    metadata_json IS NULL OR json_valid(metadata_json)
  ),
  UNIQUE (run_id, kind, name)
);

CREATE TABLE reports (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  format TEXT NOT NULL CHECK (format IN ('markdown', 'json', 'csv', 'html')),
  media_type TEXT NOT NULL,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  UNIQUE (run_id, name, format)
);
```

Recommended indexes:

```sql
CREATE INDEX idx_measurements_lookup
  ON measurements (run_id, profile_id, workload_id, concurrency);

CREATE INDEX idx_requests_measurement
  ON requests (measurement_id, status);

CREATE INDEX idx_telemetry_samples_series_time
  ON telemetry_samples (series_id, timestamp);

CREATE INDEX idx_events_run_time
  ON events (run_id, timestamp);
```

## Required Performance Data

Each completed measurement must be enough to answer these questions without
rerunning the benchmark:

- What exact spec, profile, engine, command, and environment produced it?
- How many requests were attempted, completed, failed, or canceled?
- How many prompt, completion, and total tokens were processed?
- What were aggregate output token/s, per-user output token/s, and total
  token/s?
- What were latency, TTFT, TPOT, and inter-token latency distributions?
- What memory and GPU telemetry was observed before, during, and after the
  measured workload?
- What raw engine output supports the normalized row?
- What errors occurred, and at which phase?

`measurements` stores aggregate values. `requests` stores per-request rows.
`telemetry_samples` stores time series. `artifacts` stores raw engine output.

## Statistics JSON

Statistics JSON fields should use the same shape everywhere:

```json
{
  "mean": 22800.4,
  "stddev": 4900.1,
  "min": 18000.0,
  "p50": 22100.0,
  "p90": 28900.0,
  "p95": 31000.0,
  "p99": 33000.0,
  "max": 34500.0,
  "unit": "ms",
  "count": 32
}
```

For token throughput variance, store one `measurements` row per repeat and
calculate cross-repeat stats in reports.

## Telemetry Names

Use stable telemetry series names where possible.

| Source | Name | Unit |
| --- | --- | --- |
| `proc_meminfo` | `mem_available_bytes` | `bytes` |
| `proc_meminfo` | `swap_free_bytes` | `bytes` |
| `process` | `rss_bytes` | `bytes` |
| `cgroup` | `memory_current_bytes` | `bytes` |
| `cgroup` | `memory_peak_bytes` | `bytes` |
| `vllm` | `kv_cache_available_bytes` | `bytes` |
| `vllm` | `kv_cache_tokens` | `tokens` |
| `vllm` | `reported_max_concurrency` | `requests` |
| `tegrastats` | `ram_used_bytes` | `bytes` |
| `tegrastats` | `swap_used_bytes` | `bytes` |
| `tegrastats` | `gpu_utilization_percent` | `percent` |
| `nvml` | `gpu_memory_used_bytes` | `bytes` |
| `nvml` | `gpu_utilization_percent` | `percent` |

Unknown telemetry is allowed, but it must use a clear `source`, `name`, and
`unit`.

## Artifact Kinds

`artifacts.kind` should use these values first:

| Kind | Meaning |
| --- | --- |
| `server_log` | Engine server stdout/stderr or combined log. |
| `bench_raw_result` | Engine-native benchmark result JSON. |
| `command_stdout` | Captured command stdout. |
| `command_stderr` | Captured command stderr. |
| `normalized_report` | Generated report body. |
| `raw_telemetry` | Original telemetry stream, if captured separately. |
| `debug` | Extra diagnostic blob. |

Use `gzip` compression for large text artifacts. Store the SHA-256 of the
uncompressed bytes. `content_size_bytes` is the stored blob size.
`uncompressed_size_bytes` is the original byte length before compression.

## Privacy Rules

Prompt and response text can leak data. Default behavior:

- save token counts,
- save timings,
- save hashes,
- do not save prompt text,
- do not save response text,
- redact environment variables that look like tokens, keys, passwords, or
  credentials.

`workloads.capture_prompt_text=1` is required before LocalPerf stores
`requests.prompt_text` or `requests.response_text`.

## Lifecycle

The runner writes the SQLite file incrementally:

1. Create the artifact and schema.
2. Insert `metadata`, `run`, `specs`, `engines`, `profiles`, and `workloads`.
3. Mark the run `running`.
4. Insert events and command rows as phases start.
5. Insert one `measurements` row for each planned point.
6. Append request rows, telemetry samples, raw artifacts, and events.
7. Update each measurement with final aggregate metrics.
8. Generate report artifacts.
9. Mark the run `completed`, `failed`, or `canceled`.

Each measurement should be finalized in a transaction so partial runs are still
queryable.

## Validation

Future command:

```sh
localperf artifact check runs/<run-id>.sqlite
```

The validator must check:

- `metadata.format_name = localperf_run`.
- `metadata.format_version = 1`.
- required tables exist.
- there is exactly one `run` row.
- original and normalized specs exist and match their SHA-256 values.
- JSON columns contain valid JSON.
- foreign keys are valid.
- completed measurements have throughput, token, and request counts.
- completed measurements with `save_detailed=1` have request rows.
- referenced artifact hashes match stored content.
- prompt and response text are absent unless capture was enabled.
- final run status is one of the allowed values.

## Compatibility

Readers must reject unsupported major format versions. For v1, the supported
format is:

```text
format_name     localperf_run
format_version  1
```

Future schema changes should be additive when possible. Breaking changes should
use `format_version = 2`.

## Boundaries

The `.sqlite` artifact stores benchmark evidence. It does not store model
weights, Hugging Face cache files, credentials, or enough private environment
state to recreate a user's machine.

Raw artifacts should be useful for debugging and reproduction, but the
normalized tables are the stable API.
