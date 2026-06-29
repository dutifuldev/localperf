package vllmbench

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/user"
	"path/filepath"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

const (
	sqliteFormatName    = "localperf_run"
	sqliteFormatVersion = "1"
)

func SQLiteArtifactPath(runDir, override string) string {
	if strings.TrimSpace(override) != "" {
		return override
	}
	clean := strings.TrimRight(filepath.Clean(runDir), string(filepath.Separator))
	if clean == "." || clean == "" {
		return "localperf-run.sqlite"
	}
	return clean + ".sqlite"
}

func WriteSQLiteArtifact(runDir, artifactPath string, spec Spec, summary RunSummary, plan []PlannedRun, originalSpecPath string) error {
	if strings.TrimSpace(artifactPath) == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(artifactPath), 0o755); err != nil {
		return err
	}
	_ = os.Remove(artifactPath)
	db, err := sql.Open("sqlite", artifactPath)
	if err != nil {
		return err
	}
	defer db.Close()
	if _, err := db.Exec("PRAGMA foreign_keys = ON"); err != nil {
		return err
	}
	if _, err := db.Exec(sqliteSchema); err != nil {
		return err
	}
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if err := writeSQLiteRun(tx, runDir, spec, summary, plan, originalSpecPath); err != nil {
		return err
	}
	return tx.Commit()
}

func CheckSQLiteArtifact(path string) error {
	if _, err := os.Stat(path); err != nil {
		return err
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return err
	}
	defer db.Close()
	if _, err := db.Exec("PRAGMA foreign_keys = ON"); err != nil {
		return err
	}
	var integrity string
	if err := db.QueryRow("PRAGMA integrity_check").Scan(&integrity); err != nil {
		return err
	}
	if integrity != "ok" {
		return fmt.Errorf("sqlite integrity_check = %s", integrity)
	}
	if err := checkMetadata(db); err != nil {
		return err
	}
	if err := checkRequiredTables(db); err != nil {
		return err
	}
	if err := checkSpecHashes(db); err != nil {
		return err
	}
	var runRows int
	if err := db.QueryRow("SELECT COUNT(*) FROM run").Scan(&runRows); err != nil {
		return err
	}
	if runRows != 1 {
		return fmt.Errorf("run rows = %d, want 1", runRows)
	}
	for _, kind := range []string{"original", "normalized"} {
		var count int
		if err := db.QueryRow("SELECT COUNT(*) FROM specs WHERE kind = ?", kind).Scan(&count); err != nil {
			return err
		}
		if count != 1 {
			return fmt.Errorf("spec kind %s rows = %d, want 1", kind, count)
		}
	}
	rows, err := db.Query("PRAGMA foreign_key_check")
	if err != nil {
		return err
	}
	defer rows.Close()
	if rows.Next() {
		return errors.New("foreign key check reported at least one failure")
	}
	return checkArtifactHashes(db)
}

func writeSQLiteRun(tx *sql.Tx, runDir string, spec Spec, summary RunSummary, plan []PlannedRun, originalSpecPath string) error {
	now := time.Now().UTC()
	runID := filepath.Base(filepath.Clean(runDir))
	if runID == "." || runID == "" {
		runID = spec.Name
	}
	if err := insertMetadata(tx, "format_name", sqliteFormatName); err != nil {
		return err
	}
	if err := insertMetadata(tx, "format_version", sqliteFormatVersion); err != nil {
		return err
	}
	if err := insertMetadata(tx, "generated_at", now.Format(time.RFC3339)); err != nil {
		return err
	}
	status := "completed"
	if summary.FailedRuns > 0 {
		status = "failed"
	}
	hostname, _ := os.Hostname()
	currentUser, _ := user.Current()
	username := ""
	if currentUser != nil {
		username = currentUser.Username
	}
	cwd, _ := os.Getwd()
	commandLineJSON := mustJSONString(os.Args)
	hostJSON := mustJSONString(map[string]string{"hostname": hostname})
	_, err := tx.Exec(`INSERT INTO run (
		id, name, description, status, created_at, started_at, completed_at,
		hostname, username, cwd, command_line_json, host_json
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		runID, spec.Name, spec.Description, status, summary.StartedAt.Format(time.RFC3339),
		timeOrNull(summary.StartedAt), timeOrNull(summary.FinishedAt), hostname, username, cwd,
		commandLineJSON, hostJSON)
	if err != nil {
		return err
	}
	specData, err := json.MarshalIndent(RedactedSpec(spec), "", "  ")
	if err != nil {
		return err
	}
	originalData, err := originalSpecBytes(originalSpecPath, specData)
	if err != nil {
		return err
	}
	if err := insertSpec(tx, runID, "original", originalData, now); err != nil {
		return err
	}
	if normalized, err := os.ReadFile(filepath.Join(runDir, "spec.normalized.json")); err == nil {
		if err := insertSpec(tx, runID, "normalized", normalized, now); err != nil {
			return err
		}
	} else if err := insertSpec(tx, runID, "normalized", specData, now); err != nil {
		return err
	}
	if err := insertEngines(tx, runID, spec); err != nil {
		return err
	}
	if err := insertProfiles(tx, runID, spec); err != nil {
		return err
	}
	if err := insertWorkloads(tx, runID, spec); err != nil {
		return err
	}
	events, _ := readEvents(filepath.Join(runDir, "events.jsonl"))
	artifactIDs, err := insertRunArtifacts(tx, runID, runDir, events)
	if err != nil {
		return err
	}
	phaseIDs, err := insertPhases(tx, runID, plan, events)
	if err != nil {
		return err
	}
	measurementIDs, err := insertMeasurements(tx, runID, runDir, plan, events, artifactIDs, phaseIDs)
	if err != nil {
		return err
	}
	if err := insertEvents(tx, runID, events, phaseIDs, measurementIDs); err != nil {
		return err
	}
	if err := insertCommands(tx, runID, events, phaseIDs, measurementIDs, artifactIDs); err != nil {
		return err
	}
	if err := insertTelemetry(tx, runID, events, phaseIDs, measurementIDs); err != nil {
		return err
	}
	if err := insertReports(tx, runID, artifactIDs, now); err != nil {
		return err
	}
	return nil
}

func insertMetadata(tx *sql.Tx, key, value string) error {
	_, err := tx.Exec("INSERT INTO metadata (key, value) VALUES (?, ?)", key, value)
	return err
}

func insertSpec(tx *sql.Tx, runID, kind string, data []byte, createdAt time.Time) error {
	content := strings.TrimSpace(string(data))
	_, err := tx.Exec(`INSERT INTO specs (run_id, kind, format, content, sha256, created_at)
		VALUES (?, ?, 'json', ?, ?, ?)`,
		runID, kind, content, sha256Hex([]byte(content)), createdAt.Format(time.RFC3339))
	return err
}

func originalSpecBytes(path string, fallback []byte) ([]byte, error) {
	if strings.TrimSpace(path) == "" {
		return fallback, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	redacted, err := redactedJSONDocument(data)
	if err != nil {
		return nil, err
	}
	return redacted, nil
}

func redactedJSONDocument(data []byte) ([]byte, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return nil, fmt.Errorf("extra content after JSON document")
	}
	value = redactSensitiveJSONValue(value)
	return json.MarshalIndent(value, "", "  ")
}

func redactSensitiveJSONValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		for key, child := range typed {
			if strings.EqualFold(key, "env") {
				typed[key] = redactJSONEnv(child)
				continue
			}
			if isSensitiveJSONKey(key) {
				typed[key] = "<redacted>"
				continue
			}
			typed[key] = redactSensitiveJSONValue(child)
		}
		return typed
	case []any:
		for i, child := range typed {
			typed[i] = redactSensitiveJSONValue(child)
		}
		return typed
	default:
		return value
	}
}

func redactJSONEnv(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		for key, child := range typed {
			if isSensitiveEnvKey(key) {
				typed[key] = "<redacted>"
				continue
			}
			typed[key] = child
		}
		return typed
	default:
		return redactSensitiveJSONValue(value)
	}
}

func isSensitiveJSONKey(key string) bool {
	upper := strings.ToUpper(strings.ReplaceAll(key, "-", "_"))
	switch upper {
	case "AUTH", "AUTHORIZATION", "COOKIE", "CREDENTIAL", "CREDENTIALS", "KEY", "PASS", "PASSWORD", "SECRET", "TOKEN",
		"API_KEY", "ACCESS_TOKEN", "REFRESH_TOKEN", "CLIENT_SECRET":
		return true
	}
	for _, suffix := range []string{"_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_CREDENTIAL", "_CREDENTIALS"} {
		if strings.HasSuffix(upper, suffix) {
			return true
		}
	}
	return false
}

func insertEngines(tx *sql.Tx, runID string, spec Spec) error {
	for _, engine := range spec.Engines {
		managed := boolToInt(engine.Type == "vllm-managed")
		if engine.Managed != nil {
			managed = boolToInt(*engine.Managed)
		}
		if _, err := tx.Exec(`INSERT INTO engines (
			id, run_id, name, type, managed, command, endpoint_base_url, env_json, metadata_json
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			engine.Name, runID, engine.Name, engine.Type, managed, engine.Command,
			emptyNull(engine.EndpointBaseURL), nullableJSON(redactedEnv(engine.Env)), nullableJSON(engine.Metadata)); err != nil {
			return err
		}
	}
	return nil
}

func insertProfiles(tx *sql.Tx, runID string, spec Spec) error {
	for _, profile := range spec.Profiles {
		serveJSON := mustJSONString(map[string]any{
			"max_model_len":          profile.MaxModelLen,
			"max_num_seqs":           profile.MaxNumSeqs,
			"max_num_batched_tokens": profile.MaxNumBatchedTokens,
			"gpu_memory_utilization": profile.GPUMemoryUtilization,
			"kv_cache_dtype":         profile.KVCacheDType,
			"attention_backend":      profile.AttentionBackend,
			"moe_backend":            profile.MoEBackend,
			"enable_sleep_mode":      profile.EnableSleepMode,
			"sleep_level":            SleepLevelValue(profile),
		})
		if _, err := tx.Exec(`INSERT INTO profiles (
			id, run_id, engine_id, name, model, host, port, endpoint_base_url,
			managed, context_window, max_num_seqs, max_num_batched_tokens,
			gpu_memory_utilization, enable_sleep_mode, sleep_level,
			serve_json, engine_args_json, env_json
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			profile.Name, runID, profile.Engine, profile.Name, profile.Model, profile.Host, profile.Port,
			baseURL(profile), boolToInt(profile.Managed), intNull(profile.MaxModelLen), intNull(profile.MaxNumSeqs),
			intNull(profile.MaxNumBatchedTokens), floatNull(profile.GPUMemoryUtilization), boolToInt(profile.EnableSleepMode),
			SleepLevelValue(profile), serveJSON, nullableJSON(profileExtraArgs(profile)), nullableJSON(redactedEnv(profile.Env))); err != nil {
			return err
		}
	}
	return nil
}

func insertWorkloads(tx *sql.Tx, runID string, spec Spec) error {
	for _, workload := range spec.Workloads {
		trafficJSON := mustJSONString(workload.BenchmarkTrafficConfig)
		concurrencyJSON := mustJSONString(workload.MaxConcurrency)
		if _, err := tx.Exec(`INSERT INTO workloads (
			id, run_id, name, traffic_json, concurrency_json, samples, repeats,
			save_detailed, capture_payload_artifacts
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			workload.Name, runID, workload.Name, trafficJSON, concurrencyJSON, workload.NumPrompts,
			workload.Repeats, boolToInt(workload.BenchmarkTrafficConfig.SaveDetailed),
			boolToInt(workload.CapturePayloadArtifacts)); err != nil {
			return err
		}
	}
	return nil
}

func insertRunArtifacts(tx *sql.Tx, runID, runDir string, events []Event) (map[string]int64, error) {
	artifactIDs := map[string]int64{}
	addPath := func(kind, name, path, mediaType string) error {
		if strings.TrimSpace(path) == "" {
			return nil
		}
		resolved := resolveResultPath(runDir, path)
		if _, err := os.Stat(resolved); err != nil {
			if !os.IsNotExist(err) {
				return err
			}
			return nil
		}
		id, err := insertArtifactPath(tx, runID, kind, name, resolved, mediaType)
		if err != nil {
			return err
		}
		artifactIDs[resolved] = id
		artifactIDs[path] = id
		if rel, err := filepath.Rel(runDir, resolved); err == nil {
			artifactIDs[rel] = id
		}
		return nil
	}
	for _, artifact := range []struct {
		kind      string
		name      string
		path      string
		mediaType string
	}{
		{"debug", "events.jsonl", filepath.Join(runDir, "events.jsonl"), "application/x-ndjson"},
		{"debug", "summary.json", filepath.Join(runDir, "summary.json"), "application/json"},
		{"normalized_report", "report.md", filepath.Join(runDir, "report.md"), "text/markdown"},
		{"normalized_report", "report.json", filepath.Join(runDir, "report.json"), "application/json"},
		{"normalized_report", "report.csv", filepath.Join(runDir, "report.csv"), "text/csv"},
	} {
		if err := addPath(artifact.kind, artifact.name, artifact.path, artifact.mediaType); err != nil {
			return nil, err
		}
	}
	for _, event := range events {
		if event.ResultFile != "" {
			name := rawResultArtifactName(event)
			if err := addPath("bench_raw_result", name, event.ResultFile, "application/json"); err != nil {
				return nil, err
			}
		}
		if event.LogFile != "" {
			name := filepath.Base(event.LogFile)
			if err := addPath("server_log", name, event.LogFile, "text/plain"); err != nil {
				return nil, err
			}
		}
	}
	return artifactIDs, nil
}

func rawResultArtifactName(event Event) string {
	profile := Slug(event.Profile)
	if profile == "" {
		profile = "profile"
	}
	workload := Slug(event.Workload)
	if workload == "" {
		workload = "workload"
	}
	return fmt.Sprintf("result-%s__%s__c%d__r%d.json", profile, workload, event.Concurrency, event.Repeat+1)
}

func insertArtifactPath(tx *sql.Tx, runID, kind, name, path, mediaType string) (int64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	content := data
	compression := "none"
	if len(data) > 64*1024 && (strings.HasPrefix(mediaType, "text/") || strings.Contains(mediaType, "json")) {
		var compressed bytes.Buffer
		gzipWriter := gzip.NewWriter(&compressed)
		if _, err := gzipWriter.Write(data); err != nil {
			return 0, err
		}
		if err := gzipWriter.Close(); err != nil {
			return 0, err
		}
		content = compressed.Bytes()
		compression = "gzip"
	}
	result, err := tx.Exec(`INSERT OR IGNORE INTO artifacts (
		run_id, kind, name, media_type, compression, content, content_size_bytes,
		uncompressed_size_bytes, sha256, original_path, created_at
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		runID, kind, name, mediaType, compression, content, len(content), len(data),
		sha256Hex(data), path, time.Now().UTC().Format(time.RFC3339))
	if err != nil {
		return 0, err
	}
	id, _ := result.LastInsertId()
	if id != 0 {
		return id, nil
	}
	err = tx.QueryRow(`SELECT id FROM artifacts WHERE run_id = ? AND kind = ? AND name = ?`, runID, kind, name).Scan(&id)
	return id, err
}

func insertPhases(tx *sql.Tx, runID string, plan []PlannedRun, events []Event) (map[string]int64, error) {
	phaseIDs := map[string]int64{}
	for _, planned := range plan {
		key := measurementKey(planned.Profile.Name, planned.Workload.Name, planned.Concurrency, planned.Repeat)
		startedAt, completedAt := measurementTimes(events, planned)
		status := measurementStatus(events, planned)
		id, err := insertPhase(tx, runID, planned.Profile.Name, planned.Workload.Name,
			fmt.Sprintf("%s/%s c%d r%d", planned.Profile.Name, planned.Workload.Name, planned.Concurrency, planned.Repeat+1),
			"measurement", status, startedAt, completedAt)
		if err != nil {
			return nil, err
		}
		phaseIDs[key] = id
	}
	return phaseIDs, nil
}

func insertPhase(tx *sql.Tx, runID, profile, workload, name, typ, status string, startedAt, completedAt *time.Time) (int64, error) {
	result, err := tx.Exec(`INSERT INTO phases (
		run_id, profile_id, workload_id, name, type, status, started_at, completed_at
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		runID, nullString(profile), nullString(workload), name, typ, status, timePtrString(startedAt), timePtrString(completedAt))
	if err != nil {
		return 0, err
	}
	return result.LastInsertId()
}

func insertMeasurements(tx *sql.Tx, runID, runDir string, plan []PlannedRun, events []Event, artifactIDs map[string]int64, phaseIDs map[string]int64) (map[string]int64, error) {
	measurementIDs := map[string]int64{}
	reportRows := rowsByMeasurement(runDir, events)
	for _, planned := range plan {
		key := measurementKey(planned.Profile.Name, planned.Workload.Name, planned.Concurrency, planned.Repeat)
		row := reportRows[key]
		status := measurementStatus(events, planned)
		startedAt, completedAt := measurementTimes(events, planned)
		rawID := int64(0)
		if row.ResultFile != "" {
			rawID = artifactIDForPath(artifactIDs, row.ResultFile)
		} else if event := finishEvent(events, planned); event.ResultFile != "" {
			rawID = artifactIDForPath(artifactIDs, event.ResultFile)
		}
		result, err := tx.Exec(`INSERT INTO measurements (
			run_id, profile_id, workload_id, phase_id, repeat_index, concurrency,
			samples_requested, status, started_at, completed_at, wall_time_ms,
			completed_requests, failed_requests, prompt_tokens, completion_tokens,
			total_tokens, aggregate_output_tok_s, per_user_output_tok_s,
			aggregate_total_tok_s, raw_result_artifact_id, error_type, error_message
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			runID, planned.Profile.Name, planned.Workload.Name, zeroNullInt(phaseIDs[key]),
			planned.Repeat, planned.Concurrency, planned.Workload.NumPrompts, status,
			timePtrString(startedAt), timePtrString(completedAt), durationMillis(startedAt, completedAt),
			row.Completed, row.Failed, nil, nil, nil, floatNull(row.OutputTokensPerSec),
			floatNull(row.PerUserOutputTokSec), floatNull(row.TotalTokensPerSec),
			zeroNullInt(rawID), nil, measurementError(events, planned))
		if err != nil {
			return nil, err
		}
		id, _ := result.LastInsertId()
		measurementIDs[key] = id
		if err := insertMetricStats(tx, id, row); err != nil {
			return nil, err
		}
	}
	return measurementIDs, nil
}

func insertMetricStats(tx *sql.Tx, measurementID int64, row ReportRow) error {
	stats := []struct {
		metric string
		unit   string
		mean   float64
		p99    float64
		count  int
	}{
		{"ttft", "ms", row.MeanTTFTMillis, row.P99TTFTMillis, row.Completed},
		{"tpot", "ms", row.MeanTPOTMillis, 0, row.Completed},
		{"output_throughput", "tok/s", row.OutputTokensPerSec, 0, 1},
		{"total_throughput", "tok/s", row.TotalTokensPerSec, 0, 1},
	}
	for _, stat := range stats {
		if stat.mean == 0 && stat.p99 == 0 {
			continue
		}
		if _, err := tx.Exec(`INSERT INTO metric_stats (
			measurement_id, metric, unit, mean, p99, count
		) VALUES (?, ?, ?, ?, ?, ?)`, measurementID, stat.metric, stat.unit, stat.mean, floatNull(stat.p99), stat.count); err != nil {
			return err
		}
	}
	return nil
}

func insertEvents(tx *sql.Tx, runID string, events []Event, phaseIDs, measurementIDs map[string]int64) error {
	for _, event := range events {
		key := eventMeasurementKey(event)
		_, err := tx.Exec(`INSERT INTO events (
			run_id, timestamp, level, type, phase_id, profile_id, workload_id,
			measurement_id, message, data_json
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			runID, event.Timestamp.Format(time.RFC3339), eventLevel(event), event.Type,
			zeroNullInt(phaseIDs[key]), nullString(event.Profile), nullString(event.Workload),
			zeroNullInt(measurementIDs[key]), event.Error, nullableRawJSON(event.Details))
		if err != nil {
			return err
		}
	}
	return nil
}

func insertCommands(tx *sql.Tx, runID string, events []Event, phaseIDs, measurementIDs, artifactIDs map[string]int64) error {
	for _, event := range events {
		if event.Command == "" && len(event.Args) == 0 {
			continue
		}
		key := eventMeasurementKey(event)
		status := commandStatus(event)
		_, err := tx.Exec(`INSERT INTO commands (
			run_id, phase_id, measurement_id, profile_id, phase, argv_json,
			started_at, completed_at, exit_code, status, stdout_artifact_id, stderr_artifact_id
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			runID, zeroNullInt(phaseIDs[key]), zeroNullInt(measurementIDs[key]),
			nullString(event.Profile), event.Type, mustJSONString(event.Args),
			commandStartedAt(event, status), commandCompletedAt(event, status),
			commandExitCode(event, status), status, zeroNullInt(artifactIDForPath(artifactIDs, event.LogFile)), nil)
		if err != nil {
			return err
		}
	}
	return nil
}

func commandStatus(event Event) string {
	if event.Type == "planned_run" {
		return "planned"
	}
	if strings.HasSuffix(event.Type, "_start") {
		return "running"
	}
	if event.Error != "" {
		return "failed"
	}
	return "completed"
}

func commandStartedAt(event Event, status string) any {
	if status == "planned" {
		return nil
	}
	return event.Timestamp.Format(time.RFC3339)
}

func commandCompletedAt(event Event, status string) any {
	if status == "completed" || status == "failed" || status == "canceled" {
		return event.Timestamp.Format(time.RFC3339)
	}
	return nil
}

func commandExitCode(event Event, status string) any {
	if status == "completed" || status == "failed" || status == "canceled" {
		return event.ExitCode
	}
	return nil
}

func insertTelemetry(tx *sql.Tx, runID string, events []Event, phaseIDs, measurementIDs map[string]int64) error {
	seriesID, err := ensureTelemetrySeries(tx, runID, "proc_meminfo", "mem_available_bytes", "bytes", "run", "{}")
	if err != nil {
		return err
	}
	for _, event := range events {
		if event.MemAvailableGiB == 0 {
			continue
		}
		key := eventMeasurementKey(event)
		_, err := tx.Exec(`INSERT INTO telemetry_samples (
			series_id, timestamp, value, phase_id, measurement_id
		) VALUES (?, ?, ?, ?, ?)`,
			seriesID, event.Timestamp.Format(time.RFC3339), event.MemAvailableGiB*1024*1024*1024,
			zeroNullInt(phaseIDs[key]), zeroNullInt(measurementIDs[key]))
		if err != nil {
			return err
		}
	}
	return nil
}

func ensureTelemetrySeries(tx *sql.Tx, runID, source, metric, unit, target, tags string) (int64, error) {
	if _, err := tx.Exec(`INSERT OR IGNORE INTO telemetry_series (
		run_id, source, metric, unit, target, tags_json
	) VALUES (?, ?, ?, ?, ?, ?)`, runID, source, metric, unit, target, tags); err != nil {
		return 0, err
	}
	var id int64
	err := tx.QueryRow(`SELECT id FROM telemetry_series
		WHERE run_id = ? AND source = ? AND metric = ? AND target = ? AND tags_json = ?`,
		runID, source, metric, target, tags).Scan(&id)
	return id, err
}

func insertReports(tx *sql.Tx, runID string, artifactIDs map[string]int64, createdAt time.Time) error {
	for _, report := range []struct {
		name      string
		format    string
		mediaType string
	}{
		{"report.md", "markdown", "text/markdown"},
		{"report.json", "json", "application/json"},
		{"report.csv", "csv", "text/csv"},
	} {
		id := artifactIDs[report.name]
		if id == 0 {
			continue
		}
		if _, err := tx.Exec(`INSERT INTO reports (
			run_id, name, format, media_type, artifact_id, created_at
		) VALUES (?, ?, ?, ?, ?, ?)`, runID, report.name, report.format, report.mediaType, id, createdAt.Format(time.RFC3339)); err != nil {
			return err
		}
	}
	return nil
}

func rowsByMeasurement(runDir string, events []Event) map[string]ReportRow {
	out := map[string]ReportRow{}
	for _, event := range events {
		if event.Type != "workload_finish" || event.ResultFile == "" || event.Error != "" {
			continue
		}
		rows, err := ParseResultFile(resolveResultPath(runDir, event.ResultFile))
		if err != nil || len(rows) == 0 {
			continue
		}
		row := rows[0]
		enrichRowFromEvent(&row, event, nil)
		out[eventMeasurementKey(event)] = row
	}
	return out
}

func measurementStatus(events []Event, planned PlannedRun) string {
	status := "planned"
	for _, event := range events {
		if eventMatchesPlanned(event, planned) {
			if event.Type == "workload_skipped" {
				return "skipped"
			}
			if event.Type == "workload_failed" || event.Error != "" {
				return "failed"
			}
			if event.Type == "workload_finish" && event.Error == "" {
				status = "completed"
			}
		}
	}
	return status
}

func measurementError(events []Event, planned PlannedRun) any {
	for _, event := range events {
		if eventMatchesPlanned(event, planned) && event.Error != "" {
			return event.Error
		}
	}
	return nil
}

func measurementTimes(events []Event, planned PlannedRun) (*time.Time, *time.Time) {
	var start, end *time.Time
	for _, event := range events {
		if !eventMatchesPlanned(event, planned) {
			continue
		}
		if event.Type == "workload_start" {
			t := event.Timestamp
			start = &t
		}
		if event.Type == "workload_finish" || event.Type == "workload_failed" || event.Type == "workload_skipped" {
			t := event.Timestamp
			end = &t
		}
	}
	return start, end
}

func finishEvent(events []Event, planned PlannedRun) Event {
	for _, event := range events {
		if eventMatchesPlanned(event, planned) && event.Type == "workload_finish" {
			return event
		}
	}
	return Event{}
}

func eventMatchesPlanned(event Event, planned PlannedRun) bool {
	return event.Profile == planned.Profile.Name &&
		event.Workload == planned.Workload.Name &&
		event.Concurrency == planned.Concurrency &&
		event.Repeat == planned.Repeat
}

func eventMeasurementKey(event Event) string {
	if event.Profile == "" || event.Workload == "" || event.Concurrency == 0 {
		return ""
	}
	return measurementKey(event.Profile, event.Workload, event.Concurrency, event.Repeat)
}

func measurementKey(profile, workload string, concurrency, repeat int) string {
	return fmt.Sprintf("%s\x00%s\x00%d\x00%d", profile, workload, concurrency, repeat)
}

func checkMetadata(db *sql.DB) error {
	values := map[string]string{}
	rows, err := db.Query("SELECT key, value FROM metadata")
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var key, value string
		if err := rows.Scan(&key, &value); err != nil {
			return err
		}
		values[key] = value
	}
	if values["format_name"] != sqliteFormatName {
		return fmt.Errorf("format_name = %q, want %q", values["format_name"], sqliteFormatName)
	}
	if values["format_version"] != sqliteFormatVersion {
		return fmt.Errorf("format_version = %q, want %q", values["format_version"], sqliteFormatVersion)
	}
	return nil
}

func checkRequiredTables(db *sql.DB) error {
	required := []string{"metadata", "run", "specs", "engines", "profiles", "workloads", "phases", "measurements", "metric_stats", "requests", "request_stream_events", "telemetry_series", "telemetry_samples", "events", "commands", "artifacts", "reports"}
	present := map[string]bool{}
	rows, err := db.Query("SELECT name FROM sqlite_master WHERE type = 'table'")
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			return err
		}
		present[name] = true
	}
	for _, table := range required {
		if !present[table] {
			return fmt.Errorf("missing required table %s", table)
		}
	}
	return nil
}

func checkSpecHashes(db *sql.DB) error {
	rows, err := db.Query("SELECT kind, content, sha256 FROM specs")
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var kind, content, want string
		if err := rows.Scan(&kind, &content, &want); err != nil {
			return err
		}
		if got := sha256Hex([]byte(content)); got != want {
			return fmt.Errorf("spec %s sha256 = %s, want %s", kind, got, want)
		}
	}
	return rows.Err()
}

func checkArtifactHashes(db *sql.DB) error {
	rows, err := db.Query("SELECT name, compression, content, sha256 FROM artifacts")
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var name, compression, want string
		var content []byte
		if err := rows.Scan(&name, &compression, &content, &want); err != nil {
			return err
		}
		data := content
		if compression == "gzip" {
			reader, err := gzip.NewReader(bytes.NewReader(content))
			if err != nil {
				return fmt.Errorf("artifact %s gzip decode: %w", name, err)
			}
			data, err = io.ReadAll(reader)
			_ = reader.Close()
			if err != nil {
				return fmt.Errorf("artifact %s gzip read: %w", name, err)
			}
		}
		if got := sha256Hex(data); got != want {
			return fmt.Errorf("artifact %s sha256 = %s, want %s", name, got, want)
		}
	}
	return rows.Err()
}

func artifactIDForPath(ids map[string]int64, path string) int64 {
	if path == "" {
		return 0
	}
	if id := ids[path]; id != 0 {
		return id
	}
	if id := ids[filepath.Clean(path)]; id != 0 {
		return id
	}
	if rel := strings.TrimPrefix(filepath.Clean(path), "."+string(filepath.Separator)); rel != path {
		return ids[rel]
	}
	return 0
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func nullableJSON(value any) any {
	data, err := json.Marshal(value)
	if err != nil || string(data) == "null" || string(data) == "{}" {
		return nil
	}
	return string(data)
}

func nullableRawJSON(value json.RawMessage) any {
	if len(value) == 0 {
		return nil
	}
	return string(value)
}

func mustJSONString(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		return "{}"
	}
	return string(data)
}

func timeOrNull(value time.Time) any {
	if value.IsZero() {
		return nil
	}
	return value.Format(time.RFC3339)
}

func timePtrString(value *time.Time) any {
	if value == nil || value.IsZero() {
		return nil
	}
	return value.UTC().Format(time.RFC3339)
}

func durationMillis(start, end *time.Time) any {
	if start == nil || end == nil {
		return nil
	}
	return end.Sub(*start).Seconds() * 1000
}

func boolToInt(value bool) int {
	if value {
		return 1
	}
	return 0
}

func nullString(value string) any {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	return value
}

func emptyNull(value string) any {
	return nullString(value)
}

func intNull(value int) any {
	if value == 0 {
		return nil
	}
	return value
}

func zeroNullInt(value int64) any {
	if value == 0 {
		return nil
	}
	return value
}

func floatNull(value float64) any {
	if value == 0 {
		return nil
	}
	return value
}

func eventLevel(event Event) string {
	if event.Error != "" {
		return "error"
	}
	if strings.Contains(event.Type, "failed") || strings.Contains(event.Type, "floor") {
		return "warn"
	}
	return "info"
}

const sqliteSchema = `
PRAGMA foreign_keys = ON;
PRAGMA user_version = 1;

CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE run (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  status TEXT NOT NULL CHECK (status IN ('created', 'running', 'completed', 'failed', 'canceled')),
  created_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  localperf_version TEXT,
  localperf_git_commit TEXT,
  hostname TEXT,
  username TEXT,
  cwd TEXT,
  command_line_json TEXT CHECK (command_line_json IS NULL OR json_valid(command_line_json)),
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
  metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json)),
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
  engine_args_json TEXT CHECK (engine_args_json IS NULL OR json_valid(engine_args_json)),
  env_json TEXT CHECK (env_json IS NULL OR json_valid(env_json)),
  metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json)),
  UNIQUE (run_id, name)
);
CREATE TABLE workloads (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  traffic_json TEXT NOT NULL CHECK (json_valid(traffic_json)),
  concurrency_json TEXT NOT NULL CHECK (json_valid(concurrency_json)),
  samples INTEGER NOT NULL CHECK (samples > 0),
  repeats INTEGER NOT NULL DEFAULT 1 CHECK (repeats > 0),
  save_detailed INTEGER NOT NULL DEFAULT 1 CHECK (save_detailed IN (0, 1)),
  capture_payload_artifacts INTEGER NOT NULL DEFAULT 0 CHECK (capture_payload_artifacts IN (0, 1)),
  metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json)),
  UNIQUE (run_id, name)
);
CREATE TABLE phases (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  profile_id TEXT REFERENCES profiles(id) ON DELETE SET NULL,
  workload_id TEXT REFERENCES workloads(id) ON DELETE SET NULL,
  name TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('startup', 'health_check', 'warmup', 'measurement', 'sleep', 'wake', 'shutdown', 'report', 'other')),
  status TEXT NOT NULL CHECK (status IN ('planned', 'running', 'completed', 'failed', 'skipped', 'canceled')),
  started_at TEXT,
  completed_at TEXT,
  metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json))
);
CREATE TABLE measurements (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  workload_id TEXT NOT NULL REFERENCES workloads(id) ON DELETE CASCADE,
  phase_id INTEGER REFERENCES phases(id) ON DELETE SET NULL,
  repeat_index INTEGER NOT NULL DEFAULT 0,
  concurrency INTEGER NOT NULL CHECK (concurrency > 0),
  samples_requested INTEGER NOT NULL CHECK (samples_requested > 0),
  status TEXT NOT NULL CHECK (status IN ('planned', 'running', 'completed', 'failed', 'skipped', 'canceled')),
  started_at TEXT,
  completed_at TEXT,
  wall_time_ms REAL,
  completed_requests INTEGER NOT NULL DEFAULT 0 CHECK (completed_requests >= 0),
  failed_requests INTEGER NOT NULL DEFAULT 0 CHECK (failed_requests >= 0),
  prompt_tokens INTEGER CHECK (prompt_tokens >= 0),
  completion_tokens INTEGER CHECK (completion_tokens >= 0),
  total_tokens INTEGER CHECK (total_tokens >= 0),
  aggregate_output_tok_s REAL,
  per_user_output_tok_s REAL,
  aggregate_total_tok_s REAL,
  raw_result_artifact_id INTEGER REFERENCES artifacts(id),
  error_type TEXT,
  error_message TEXT,
  metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json)),
  UNIQUE (run_id, profile_id, workload_id, repeat_index, concurrency)
);
CREATE TABLE metric_stats (
  id INTEGER PRIMARY KEY,
  measurement_id INTEGER NOT NULL REFERENCES measurements(id) ON DELETE CASCADE,
  metric TEXT NOT NULL,
  unit TEXT NOT NULL,
  mean REAL,
  stddev REAL,
  min REAL,
  p50 REAL,
  p90 REAL,
  p95 REAL,
  p99 REAL,
  max REAL,
  count INTEGER NOT NULL CHECK (count >= 0),
  metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json)),
  UNIQUE (measurement_id, metric, unit)
);
CREATE TABLE requests (
  id INTEGER PRIMARY KEY,
  measurement_id INTEGER NOT NULL REFERENCES measurements(id) ON DELETE CASCADE,
  request_index INTEGER NOT NULL,
  request_id TEXT,
  status TEXT NOT NULL CHECK (status IN ('completed', 'failed', 'canceled')),
  streamed INTEGER NOT NULL DEFAULT 0 CHECK (streamed IN (0, 1)),
  http_status_code INTEGER,
  started_at TEXT NOT NULL,
  first_token_at TEXT,
  completed_at TEXT,
  latency_ms REAL,
  ttft_ms REAL,
  tpot_ms REAL,
  itl_mean_ms REAL,
  prompt_tokens INTEGER CHECK (prompt_tokens >= 0),
  completion_tokens INTEGER CHECK (completion_tokens >= 0),
  total_tokens INTEGER CHECK (total_tokens >= 0),
  prompt_sha256 TEXT,
  response_sha256 TEXT,
  prompt_artifact_id INTEGER REFERENCES artifacts(id),
  response_artifact_id INTEGER REFERENCES artifacts(id),
  error_type TEXT,
  error_code TEXT,
  error_message TEXT,
  response_metadata_json TEXT CHECK (response_metadata_json IS NULL OR json_valid(response_metadata_json)),
  UNIQUE (measurement_id, request_index)
);
CREATE TABLE request_stream_events (
  id INTEGER PRIMARY KEY,
  request_row_id INTEGER NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
  event_index INTEGER NOT NULL,
  timestamp TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('queued', 'sent', 'first_token', 'chunk', 'token', 'completed', 'error')),
  token_count_delta INTEGER CHECK (token_count_delta IS NULL OR token_count_delta >= 0),
  text_byte_count_delta INTEGER CHECK (text_byte_count_delta IS NULL OR text_byte_count_delta >= 0),
  metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json)),
  UNIQUE (request_row_id, event_index)
);
CREATE TABLE telemetry_series (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  source TEXT NOT NULL,
  metric TEXT NOT NULL,
  unit TEXT,
  target TEXT NOT NULL DEFAULT 'run',
  tags_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(tags_json)),
  UNIQUE (run_id, source, metric, target, tags_json)
);
CREATE TABLE telemetry_samples (
  id INTEGER PRIMARY KEY,
  series_id INTEGER NOT NULL REFERENCES telemetry_series(id) ON DELETE CASCADE,
  timestamp TEXT NOT NULL,
  value REAL NOT NULL,
  phase_id INTEGER REFERENCES phases(id) ON DELETE SET NULL,
  measurement_id INTEGER REFERENCES measurements(id) ON DELETE SET NULL
);
CREATE TABLE events (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  timestamp TEXT NOT NULL,
  level TEXT NOT NULL CHECK (level IN ('debug', 'info', 'warn', 'error')),
  type TEXT NOT NULL,
  phase_id INTEGER REFERENCES phases(id) ON DELETE SET NULL,
  profile_id TEXT REFERENCES profiles(id) ON DELETE SET NULL,
  workload_id TEXT REFERENCES workloads(id) ON DELETE SET NULL,
  measurement_id INTEGER REFERENCES measurements(id) ON DELETE SET NULL,
  message TEXT,
  data_json TEXT CHECK (data_json IS NULL OR json_valid(data_json))
);
CREATE TABLE commands (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  phase_id INTEGER REFERENCES phases(id) ON DELETE SET NULL,
  measurement_id INTEGER REFERENCES measurements(id) ON DELETE SET NULL,
  profile_id TEXT REFERENCES profiles(id) ON DELETE SET NULL,
  phase TEXT NOT NULL,
  cwd TEXT,
  argv_json TEXT NOT NULL CHECK (json_valid(argv_json)),
  env_json TEXT CHECK (env_json IS NULL OR json_valid(env_json)),
  started_at TEXT,
  completed_at TEXT,
  exit_code INTEGER,
  status TEXT NOT NULL CHECK (status IN ('planned', 'running', 'completed', 'failed', 'canceled')),
  stdout_artifact_id INTEGER REFERENCES artifacts(id),
  stderr_artifact_id INTEGER REFERENCES artifacts(id),
  metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json))
);
CREATE TABLE artifacts (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  media_type TEXT NOT NULL,
  compression TEXT NOT NULL DEFAULT 'none' CHECK (compression IN ('none', 'gzip')),
  content BLOB NOT NULL,
  content_size_bytes INTEGER NOT NULL,
  uncompressed_size_bytes INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  original_path TEXT,
  created_at TEXT NOT NULL,
  metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json)),
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
CREATE INDEX idx_measurements_lookup ON measurements (run_id, profile_id, workload_id, concurrency);
CREATE INDEX idx_metric_stats_metric ON metric_stats (metric, unit, measurement_id);
CREATE INDEX idx_requests_measurement ON requests (measurement_id, status);
CREATE INDEX idx_request_stream_events_request ON request_stream_events (request_row_id, event_index);
CREATE INDEX idx_telemetry_samples_series_time ON telemetry_samples (series_id, timestamp);
CREATE INDEX idx_telemetry_samples_phase_time ON telemetry_samples (phase_id, timestamp);
CREATE INDEX idx_phases_run_time ON phases (run_id, started_at);
CREATE INDEX idx_events_run_time ON events (run_id, timestamp);
`
