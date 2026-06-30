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
	db, err := createSQLiteArtifact(artifactPath)
	if err != nil {
		return err
	}
	defer db.Close()
	return withSQLiteTx(db, func(tx *sql.Tx) error {
		return writeSQLiteRun(tx, runDir, spec, summary, plan, originalSpecPath)
	})
}

func createSQLiteArtifact(path string) (*sql.DB, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	_ = os.Remove(path)
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	if _, err := db.Exec("PRAGMA foreign_keys = ON"); err != nil {
		_ = db.Close()
		return nil, err
	}
	if _, err := db.Exec(sqliteSchema); err != nil {
		_ = db.Close()
		return nil, err
	}
	return db, nil
}

func withSQLiteTx(db *sql.DB, run func(*sql.Tx) error) error {
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if err := run(tx); err != nil {
		return err
	}
	return tx.Commit()
}

func CheckSQLiteArtifact(path string) error {
	db, err := openSQLiteArtifactForCheck(path)
	if err != nil {
		return err
	}
	defer db.Close()
	for _, check := range []func(*sql.DB) error{
		checkIntegrity,
		checkMetadata,
		checkRequiredTables,
		checkSpecHashes,
		checkRunRowCount,
		checkSpecKindRows,
		checkForeignKeys,
		checkArtifactHashes,
	} {
		if err := check(db); err != nil {
			return err
		}
	}
	return nil
}

func openSQLiteArtifactForCheck(path string) (*sql.DB, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	if _, err := db.Exec("PRAGMA foreign_keys = ON"); err != nil {
		_ = db.Close()
		return nil, err
	}
	return db, nil
}

func checkIntegrity(db *sql.DB) error {
	var integrity string
	if err := db.QueryRow("PRAGMA integrity_check").Scan(&integrity); err != nil {
		return err
	}
	if integrity != "ok" {
		return fmt.Errorf("sqlite integrity_check = %s", integrity)
	}
	return nil
}

func checkRunRowCount(db *sql.DB) error {
	var runRows int
	if err := db.QueryRow("SELECT COUNT(*) FROM run").Scan(&runRows); err != nil {
		return err
	}
	if runRows != 1 {
		return fmt.Errorf("run rows = %d, want 1", runRows)
	}
	return nil
}

func checkSpecKindRows(db *sql.DB) error {
	for _, kind := range []string{"original", "normalized"} {
		var count int
		if err := db.QueryRow("SELECT COUNT(*) FROM specs WHERE kind = ?", kind).Scan(&count); err != nil {
			return err
		}
		if count != 1 {
			return fmt.Errorf("spec kind %s rows = %d, want 1", kind, count)
		}
	}
	return nil
}

func checkForeignKeys(db *sql.DB) error {
	rows, err := db.Query("PRAGMA foreign_key_check")
	if err != nil {
		return err
	}
	defer rows.Close()
	if rows.Next() {
		return errors.New("foreign key check reported at least one failure")
	}
	return rows.Err()
}

func writeSQLiteRun(tx *sql.Tx, runDir string, spec Spec, summary RunSummary, plan []PlannedRun, originalSpecPath string) error {
	now := time.Now().UTC()
	runID := sqliteRunID(runDir, spec)
	if err := insertRunMetadata(tx, now); err != nil {
		return err
	}
	if err := insertRunRow(tx, runID, spec, summary); err != nil {
		return err
	}
	if err := insertRunSpecs(tx, runID, runDir, spec, originalSpecPath, now); err != nil {
		return err
	}
	return insertRunData(tx, runID, runDir, spec, summary, plan, now)
}

func sqliteRunID(runDir string, spec Spec) string {
	runID := filepath.Base(filepath.Clean(runDir))
	if runID == "." || runID == "" {
		return spec.Name
	}
	return runID
}

func insertRunMetadata(tx *sql.Tx, now time.Time) error {
	values := map[string]string{
		"format_name":    sqliteFormatName,
		"format_version": sqliteFormatVersion,
		"generated_at":   now.Format(time.RFC3339),
	}
	for key, value := range values {
		if err := insertMetadata(tx, key, value); err != nil {
			return err
		}
	}
	return nil
}

func insertRunRow(tx *sql.Tx, runID string, spec Spec, summary RunSummary) error {
	hostname, _ := os.Hostname()
	username := currentUsername()
	cwd, _ := os.Getwd()
	_, err := tx.Exec(`INSERT INTO run (
		id, name, description, status, created_at, started_at, completed_at,
		hostname, username, cwd, command_line_json, host_json
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		runID, spec.Name, spec.Description, runStatus(summary), summary.StartedAt.Format(time.RFC3339),
		timeOrNull(summary.StartedAt), timeOrNull(summary.FinishedAt), hostname, username, cwd,
		mustJSONString(os.Args), mustJSONString(map[string]string{"hostname": hostname}))
	return err
}

func runStatus(summary RunSummary) string {
	if summary.FailedRuns > 0 {
		return "failed"
	}
	return "completed"
}

func currentUsername() string {
	currentUser, _ := user.Current()
	if currentUser == nil {
		return ""
	}
	return currentUser.Username
}

func insertRunSpecs(tx *sql.Tx, runID, runDir string, spec Spec, originalSpecPath string, now time.Time) error {
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
	return insertSpec(tx, runID, "normalized", normalizedSpecBytes(runDir, specData), now)
}

func normalizedSpecBytes(runDir string, fallback []byte) []byte {
	if normalized, err := os.ReadFile(filepath.Join(runDir, "spec.normalized.json")); err == nil {
		return normalized
	}
	return fallback
}

func insertRunData(tx *sql.Tx, runID, runDir string, spec Spec, _ RunSummary, plan []PlannedRun, now time.Time) error {
	if err := insertRunDimensions(tx, runID, runDir, spec); err != nil {
		return err
	}
	events, _ := readEvents(filepath.Join(runDir, "events.jsonl"))
	return insertRunExecutionData(tx, runID, runDir, spec, plan, events, now)
}

func insertRunDimensions(tx *sql.Tx, runID, runDir string, spec Spec) error {
	for _, insert := range []func(*sql.Tx, string, Spec) error{insertEngines, insertProfiles, insertWorkloads} {
		if err := insert(tx, runID, spec); err != nil {
			return err
		}
	}
	return insertCanonicalDatasets(tx, runID, runDir, spec)
}

func insertRunExecutionData(tx *sql.Tx, runID, runDir string, spec Spec, plan []PlannedRun, events []Event, now time.Time) error {
	artifactIDs, err := insertRunArtifacts(tx, runID, runDir, spec, events)
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
	return insertTelemetryAndReports(tx, runID, events, phaseIDs, measurementIDs, artifactIDs, now)
}

func insertTelemetryAndReports(tx *sql.Tx, runID string, events []Event, phaseIDs, measurementIDs, artifactIDs map[string]int64, now time.Time) error {
	if err := insertTelemetry(tx, runID, events, phaseIDs, measurementIDs); err != nil {
		return err
	}
	return insertReports(tx, runID, artifactIDs, now)
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
			id, run_id, name, phase, traffic_json, concurrency_json, samples, repeats,
			save_detailed, capture_payload_artifacts, dataset_json, request_json, load_json
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			workload.Name, runID, workload.Name, workload.Phase, trafficJSON, concurrencyJSON, workload.NumPrompts,
			workload.Repeats, boolToInt(workload.BenchmarkTrafficConfig.SaveDetailed),
			boolToInt(workload.CapturePayloadArtifacts), structuredWorkloadJSON(workload, workload.Dataset),
			structuredWorkloadJSON(workload, workload.Request), structuredWorkloadJSON(workload, workload.Load)); err != nil {
			return err
		}
	}
	return nil
}

func structuredWorkloadJSON(workload Workload, value any) any {
	if !hasStructuredDataset(workload) {
		return nil
	}
	return nullableJSON(value)
}

func insertCanonicalDatasets(tx *sql.Tx, runID, runDir string, spec Spec) error {
	for _, workload := range spec.Workloads {
		if !hasStructuredDataset(workload) || strings.TrimSpace(workload.Dataset.Prepared.CanonicalPath) == "" {
			continue
		}
		if err := insertCanonicalDataset(tx, runID, runDir, workload); err != nil {
			return err
		}
	}
	return nil
}

func insertCanonicalDataset(tx *sql.Tx, runID, runDir string, workload Workload) error {
	datasetID := firstNonEmpty(workload.Dataset.Prepared.DatasetID, datasetIDForWorkload(workload.Name))
	requests, err := readCanonicalRequestFile(resolveResultPath(runDir, workload.Dataset.Prepared.CanonicalPath))
	if err != nil {
		return err
	}
	_, err = tx.Exec(`INSERT INTO datasets (
		id, run_id, workload_id, type, uri, path, split, selection, sample_count,
		seed, config_json, canonical_path, rendered_path, request_count, sha256
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		datasetID, runID, workload.Name, workload.Dataset.Type, emptyNull(workload.Dataset.URI),
		emptyNull(workload.Dataset.Path), emptyNull(workload.Dataset.Split), emptyNull(workload.Dataset.Selection),
		workload.Dataset.SampleCount, intPointerNull(workload.Dataset.Seed), mustJSONString(workload.Dataset),
		workload.Dataset.Prepared.CanonicalPath, emptyNull(workload.Dataset.Prepared.VLLMCustomPath),
		len(requests), workload.Dataset.Prepared.SHA256)
	if err != nil {
		return err
	}
	if !workload.CapturePayloadArtifacts {
		return nil
	}
	sourceIDs := map[string]int64{}
	for _, request := range requests {
		sourceID, err := insertSourceRecord(tx, runID, datasetID, request, sourceIDs)
		if err != nil {
			return err
		}
		if err := insertCanonicalRequest(tx, runID, datasetID, workload.Name, sourceID, request); err != nil {
			return err
		}
	}
	return nil
}

func readCanonicalRequestFile(path string) ([]CanonicalRequest, error) {
	rawRows, err := readJSONLines(path)
	if err != nil {
		return nil, err
	}
	requests := make([]CanonicalRequest, 0, len(rawRows))
	for _, raw := range rawRows {
		var request CanonicalRequest
		if err := json.Unmarshal(raw, &request); err != nil {
			return nil, err
		}
		requests = append(requests, request)
	}
	return requests, nil
}

func insertSourceRecord(tx *sql.Tx, runID, datasetID string, request CanonicalRequest, seen map[string]int64) (int64, error) {
	sourceRecordID := firstNonEmpty(request.SourceRecordID, request.ID)
	if id := seen[sourceRecordID]; id != 0 {
		return id, nil
	}
	content := request.Raw
	if len(content) == 0 {
		content = mustJSON(request)
	}
	result, err := tx.Exec(`INSERT INTO source_records (
		run_id, dataset_id, source_record_id, ordinal, content_json, sha256, metadata_json
	) VALUES (?, ?, ?, ?, ?, ?, ?)`,
		runID, datasetID, sourceRecordID, request.Ordinal, string(content),
		sha256Hex(content), nullableJSON(request.Metadata))
	if err != nil {
		return 0, err
	}
	id, _ := result.LastInsertId()
	seen[sourceRecordID] = id
	return id, nil
}

func insertCanonicalRequest(tx *sql.Tx, runID, datasetID, workloadID string, sourceRecordRowID int64, request CanonicalRequest) error {
	prompt := request.Prompt
	if strings.TrimSpace(prompt) == "" {
		prompt = messagesPrompt(request.Messages)
	}
	_, err := tx.Exec(`INSERT INTO canonical_requests (
		id, run_id, dataset_id, source_record_row_id, workload_id, request_id, ordinal,
		conversation_id, turn_index, mode, prompt_sha256, max_output_tokens,
		input_tokens_expected, output_tokens_expected, arrival_offset_ms,
		messages_json, attachments_json, metadata_json, canonical_json
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		canonicalRequestRowID(datasetID, request), runID, datasetID, sourceRecordRowID, workloadID, request.ID, request.Ordinal,
		emptyNull(request.ConversationID), request.TurnIndex, request.Mode, emptyNull(sha256Maybe(prompt)),
		intNull(request.MaxOutputTokens), intNull(request.InputTokensExpected), intNull(request.OutputTokensExpected),
		request.ArrivalOffsetMillis, nullableJSON(request.Messages), nullableJSON(request.Attachments),
		nullableJSON(request.Metadata), mustJSONString(request))
	return err
}

func canonicalRequestRowID(datasetID string, request CanonicalRequest) string {
	return fmt.Sprintf("%s:%06d:%s", datasetID, request.Ordinal, firstNonEmpty(request.ID, "request"))
}

func insertRunArtifacts(tx *sql.Tx, runID, runDir string, spec Spec, events []Event) (map[string]int64, error) {
	inserter := artifactInserter{tx: tx, runID: runID, runDir: runDir, ids: map[string]int64{}}
	if err := inserter.addDefaultArtifacts(); err != nil {
		return nil, err
	}
	if err := inserter.addDatasetArtifacts(spec); err != nil {
		return nil, err
	}
	if err := inserter.addEventArtifacts(events); err != nil {
		return nil, err
	}
	return inserter.ids, nil
}

type artifactInserter struct {
	tx     *sql.Tx
	runID  string
	runDir string
	ids    map[string]int64
}

type artifactSpec struct {
	kind      string
	name      string
	path      string
	mediaType string
}

func (inserter artifactInserter) addDefaultArtifacts() error {
	for _, artifact := range []artifactSpec{
		{"debug", "events.jsonl", filepath.Join(inserter.runDir, "events.jsonl"), "application/x-ndjson"},
		{"debug", "summary.json", filepath.Join(inserter.runDir, "summary.json"), "application/json"},
		{"normalized_report", "report.md", filepath.Join(inserter.runDir, "report.md"), "text/markdown"},
		{"normalized_report", "report.json", filepath.Join(inserter.runDir, "report.json"), "application/json"},
		{"normalized_report", "report.csv", filepath.Join(inserter.runDir, "report.csv"), "text/csv"},
	} {
		if err := inserter.add(artifact); err != nil {
			return err
		}
	}
	return nil
}

func (inserter artifactInserter) addDatasetArtifacts(spec Spec) error {
	for _, workload := range spec.Workloads {
		for _, artifact := range datasetArtifactsForWorkload(workload) {
			if err := inserter.add(artifact); err != nil {
				return err
			}
		}
	}
	return nil
}

func datasetArtifactsForWorkload(workload Workload) []artifactSpec {
	if !workload.CapturePayloadArtifacts {
		return nil
	}
	prepared := workload.Dataset.Prepared
	rows := []struct {
		kind string
		path string
	}{
		{"canonical_dataset", prepared.CanonicalPath},
		{"engine_dataset", prepared.VLLMCustomPath},
	}
	artifacts := make([]artifactSpec, 0, len(rows))
	for _, row := range rows {
		if strings.TrimSpace(row.path) == "" {
			continue
		}
		artifacts = append(artifacts, artifactSpec{
			kind:      row.kind,
			name:      filepath.Base(row.path),
			path:      row.path,
			mediaType: "application/x-ndjson",
		})
	}
	return artifacts
}

func (inserter artifactInserter) addEventArtifacts(events []Event) error {
	for _, event := range events {
		if eventHasArtifactResult(event) {
			name := rawResultArtifactName(event)
			if err := inserter.add(artifactSpec{"bench_raw_result", name, event.ResultFile, "application/json"}); err != nil {
				return err
			}
		}
		if event.LogFile != "" {
			name := filepath.Base(event.LogFile)
			if err := inserter.add(artifactSpec{"server_log", name, event.LogFile, "text/plain"}); err != nil {
				return err
			}
		}
	}
	return nil
}

func (inserter artifactInserter) add(artifact artifactSpec) error {
	if strings.TrimSpace(artifact.path) == "" {
		return nil
	}
	resolved, ok, err := existingArtifactPath(inserter.runDir, artifact.path)
	if err != nil || !ok {
		return err
	}
	id, err := insertArtifactPath(inserter.tx, inserter.runID, artifact.kind, artifact.name, resolved, artifact.mediaType)
	if err != nil {
		return err
	}
	inserter.recordPathIDs(artifact.path, resolved, id)
	return nil
}

func existingArtifactPath(runDir, path string) (string, bool, error) {
	resolved := resolveResultPath(runDir, path)
	if _, err := os.Stat(resolved); err != nil {
		return resolved, false, nonMissingFileError(err)
	}
	return resolved, true, nil
}

func nonMissingFileError(err error) error {
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

func (inserter artifactInserter) recordPathIDs(original, resolved string, id int64) {
	inserter.ids[resolved] = id
	inserter.ids[original] = id
	if rel, err := filepath.Rel(inserter.runDir, resolved); err == nil {
		inserter.ids[rel] = id
	}
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
	content, compression, err := artifactContent(data, mediaType)
	if err != nil {
		return 0, err
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

func artifactContent(data []byte, mediaType string) ([]byte, string, error) {
	if !shouldCompressArtifact(data, mediaType) {
		return data, "none", nil
	}
	content, err := gzipBytes(data)
	return content, "gzip", err
}

func shouldCompressArtifact(data []byte, mediaType string) bool {
	return len(data) > 64*1024 && (strings.HasPrefix(mediaType, "text/") || strings.Contains(mediaType, "json"))
}

func gzipBytes(data []byte) ([]byte, error) {
	var compressed bytes.Buffer
	gzipWriter := gzip.NewWriter(&compressed)
	if _, err := gzipWriter.Write(data); err != nil {
		return nil, err
	}
	if err := gzipWriter.Close(); err != nil {
		return nil, err
	}
	return compressed.Bytes(), nil
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
		startedAt, completedAt := measurementTimes(events, planned)
		id, err := insertMeasurement(tx, measurementInsert{
			runID:       runID,
			planned:     planned,
			row:         row,
			phaseID:     phaseIDs[key],
			rawID:       measurementRawArtifactID(row, events, planned, artifactIDs),
			status:      measurementStatus(events, planned),
			errorText:   measurementError(events, planned),
			startedAt:   startedAt,
			completedAt: completedAt,
		})
		if err != nil {
			return nil, err
		}
		measurementIDs[key] = id
		if err := insertMeasurementDetails(tx, runDir, id, row, measurementResultFile(row, events, planned)); err != nil {
			return nil, err
		}
	}
	return measurementIDs, nil
}

type measurementInsert struct {
	runID       string
	planned     PlannedRun
	row         ReportRow
	phaseID     int64
	rawID       int64
	status      string
	errorText   any
	startedAt   *time.Time
	completedAt *time.Time
}

func insertMeasurement(tx *sql.Tx, insert measurementInsert) (int64, error) {
	result, err := tx.Exec(`INSERT INTO measurements (
		run_id, profile_id, workload_id, phase_id, repeat_index, concurrency,
		samples_requested, status, started_at, completed_at, wall_time_ms,
		completed_requests, failed_requests, prompt_tokens, completion_tokens,
		total_tokens, aggregate_output_tok_s, per_user_output_tok_s,
		aggregate_total_tok_s, raw_result_artifact_id, error_type, error_message
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		insert.runID, insert.planned.Profile.Name, insert.planned.Workload.Name, zeroNullInt(insert.phaseID),
		insert.planned.Repeat, insert.planned.Concurrency, insert.planned.Workload.NumPrompts, insert.status,
		timePtrString(insert.startedAt), timePtrString(insert.completedAt), durationMillis(insert.startedAt, insert.completedAt),
		insert.row.Completed, insert.row.Failed, knownIntNull(insert.row.promptTokensKnown, insert.row.PromptTokens),
		knownIntNull(insert.row.completionTokensKnown, insert.row.CompletionTokens), knownIntNull(insert.row.totalTokensKnown, insert.row.TotalTokens),
		knownFloatNull(insert.row.outputTokensPerSecKnown, insert.row.OutputTokensPerSec),
		knownFloatNull(insert.row.perUserOutputTokSecKnown, insert.row.PerUserOutputTokSec),
		knownFloatNull(insert.row.totalTokensPerSecKnown, insert.row.TotalTokensPerSec),
		zeroNullInt(insert.rawID), nil, insert.errorText)
	if err != nil {
		return 0, err
	}
	return result.LastInsertId()
}

func measurementRawArtifactID(row ReportRow, events []Event, planned PlannedRun, artifactIDs map[string]int64) int64 {
	if row.ResultFile != "" {
		return artifactIDForPath(artifactIDs, row.ResultFile)
	}
	if event := artifactFinishEvent(events, planned); event.ResultFile != "" {
		return artifactIDForPath(artifactIDs, event.ResultFile)
	}
	return 0
}

func measurementResultFile(row ReportRow, events []Event, planned PlannedRun) string {
	if row.ResultFile != "" {
		return row.ResultFile
	}
	return importableFinishEvent(events, planned).ResultFile
}

func insertMeasurementDetails(tx *sql.Tx, runDir string, measurementID int64, row ReportRow, resultFile string) error {
	samples, err := requestSamplesForResult(runDir, resultFile)
	if err != nil {
		if !os.IsNotExist(err) {
			return err
		}
		samples = nil
	}
	if err := insertRequestSamples(tx, measurementID, samples); err != nil {
		return err
	}
	return insertMetricStats(tx, measurementID, row, samples)
}

func insertMetricStats(tx *sql.Tx, measurementID int64, row ReportRow, samples []RequestSample) error {
	if err := insertAggregateMetricStats(tx, measurementID, row); err != nil {
		return err
	}
	return insertSampleMetricStats(tx, measurementID, samples)
}

type aggregateMetricStat struct {
	metric string
	unit   string
	mean   float64
	p99    float64
	count  int
}

func insertAggregateMetricStats(tx *sql.Tx, measurementID int64, row ReportRow) error {
	for _, stat := range aggregateMetricStats(row) {
		if err := insertAggregateMetricStat(tx, measurementID, stat); err != nil {
			return err
		}
	}
	return nil
}

func aggregateMetricStats(row ReportRow) []aggregateMetricStat {
	return []aggregateMetricStat{
		{"ttft", "ms", row.MeanTTFTMillis, row.P99TTFTMillis, row.Completed},
		{"tpot", "ms", row.MeanTPOTMillis, 0, row.Completed},
		{"output_throughput", "tok/s", row.OutputTokensPerSec, 0, 1},
		{"total_throughput", "tok/s", row.TotalTokensPerSec, 0, 1},
	}
}

func insertAggregateMetricStat(tx *sql.Tx, measurementID int64, stat aggregateMetricStat) error {
	if stat.mean == 0 && stat.p99 == 0 {
		return nil
	}
	_, err := tx.Exec(`INSERT INTO metric_stats (
		measurement_id, metric, unit, mean, p99, count
	) VALUES (?, ?, ?, ?, ?, ?)`, measurementID, stat.metric, stat.unit, stat.mean, floatNull(stat.p99), stat.count)
	return err
}

func insertSampleMetricStats(tx *sql.Tx, measurementID int64, samples []RequestSample) error {
	for _, distribution := range sampleMetricDistributions(samples) {
		if distribution.stats.Count == 0 {
			continue
		}
		if err := insertMetricDistribution(tx, measurementID, distribution.metric, distribution.unit, distribution.stats); err != nil {
			return err
		}
	}
	return nil
}

type sampleMetricDistribution struct {
	metric string
	unit   string
	stats  numericStats
}

func sampleMetricDistributions(samples []RequestSample) []sampleMetricDistribution {
	return []sampleMetricDistribution{
		{"request_output_throughput", "tok/s", statsFromSamples(samples, true, func(sample RequestSample) float64 { return sample.OutputTokensPerSecond })},
		{"request_total_throughput", "tok/s", statsFromSamples(samples, true, func(sample RequestSample) float64 { return sample.TotalTokensPerSecond })},
		{"latency", "ms", statsFromSamples(samples, false, func(sample RequestSample) float64 { return sample.LatencyMillis })},
		{"first_byte", "ms", statsFromSamples(samples, false, func(sample RequestSample) float64 { return sample.FirstByteMillis })},
	}
}

func insertMetricDistribution(tx *sql.Tx, measurementID int64, metric, unit string, stats numericStats) error {
	_, err := tx.Exec(`INSERT INTO metric_stats (
		measurement_id, metric, unit, mean, stddev, min, p50, p90, p95, p99, max, count
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		measurementID, metric, unit, stats.Mean, stats.StdDev, stats.Min, stats.P50,
		stats.P90, stats.P95, stats.P99, stats.Max, stats.Count)
	return err
}

func requestSamplesForResult(runDir, resultFile string) ([]RequestSample, error) {
	path := resolveResultPath(runDir, resultFile)
	if strings.TrimSpace(path) == "" {
		return nil, nil
	}
	data, err := readTrimmedFile(path)
	if err != nil {
		return nil, err
	}
	return requestSamplesFromResultData(data)
}

func requestSamplesFromResultData(data []byte) ([]RequestSample, error) {
	if len(data) == 0 || data[0] != '{' {
		return nil, nil
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, nil
	}
	samplesRaw, ok := raw["request_samples"]
	if !ok {
		return nil, nil
	}
	var samples []RequestSample
	if err := json.Unmarshal(samplesRaw, &samples); err != nil {
		return nil, err
	}
	return samples, nil
}

func insertRequestSamples(tx *sql.Tx, measurementID int64, samples []RequestSample) error {
	for _, sample := range samples {
		completedSample := sample.Status == "completed"
		if _, err := tx.Exec(`INSERT INTO requests (
			measurement_id, request_index, request_id, status, streamed,
			http_status_code, started_at, first_byte_at, first_byte_ms,
			first_token_at, completed_at, latency_ms, ttft_ms, tpot_ms,
			itl_mean_ms, prompt_tokens, completion_tokens, total_tokens,
			output_tok_s, total_tok_s, prompt_sha256, response_sha256,
			error_type, error_code, error_message, response_metadata_json
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			measurementID, sample.RequestIndex, nullString(sample.RequestID),
			firstNonEmpty(sample.Status, "failed"), boolToInt(sample.Streamed),
			intNull(sample.HTTPStatusCode), sample.StartedAt.Format(time.RFC3339),
			timePtrString(sample.FirstByteAt), floatNull(sample.FirstByteMillis),
			nil, timePtrString(sample.CompletedAt), floatNull(sample.LatencyMillis),
			nil, nil, nil, knownIntNull(completedSample, sample.PromptTokens), knownIntNull(completedSample, sample.CompletionTokens),
			knownIntNull(completedSample, sample.TotalTokens), knownFloatNull(completedSample, sample.OutputTokensPerSecond),
			knownFloatNull(completedSample, sample.TotalTokensPerSecond), nullString(sample.PromptSHA256),
			nullString(sample.ResponseSHA256), nullString(sample.ErrorType),
			nullString(sample.ErrorCode), nullString(sample.ErrorMessage),
			nullableJSON(sample.ResponseMetadata)); err != nil {
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
		if !eventHasImportableResult(event) {
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

func importableFinishEvent(events []Event, planned PlannedRun) Event {
	for _, event := range events {
		if eventMatchesPlanned(event, planned) && eventHasImportableResult(event) {
			return event
		}
	}
	return Event{}
}

func artifactFinishEvent(events []Event, planned PlannedRun) Event {
	for _, event := range events {
		if eventMatchesPlanned(event, planned) && event.Type == "workload_finish" && eventHasArtifactResult(event) {
			return event
		}
	}
	return Event{}
}

func eventHasImportableResult(event Event) bool {
	return event.Type == "workload_finish" && event.ResultFile != "" && (event.Error == "" || eventDetailBool(event, "result_written"))
}

func eventHasArtifactResult(event Event) bool {
	return event.ResultFile != "" && (event.Type == "workload_finish" || event.Type == "warmup_finish")
}

func eventDetailBool(event Event, key string) bool {
	if len(event.Details) == 0 {
		return false
	}
	var details map[string]bool
	if err := json.Unmarshal(event.Details, &details); err != nil {
		return false
	}
	return details[key]
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
		data, err := artifactHashContent(name, compression, content)
		if err != nil {
			return err
		}
		if got := sha256Hex(data); got != want {
			return fmt.Errorf("artifact %s sha256 = %s, want %s", name, got, want)
		}
	}
	return rows.Err()
}

func artifactHashContent(name, compression string, content []byte) ([]byte, error) {
	if compression != "gzip" {
		return content, nil
	}
	reader, err := gzip.NewReader(bytes.NewReader(content))
	if err != nil {
		return nil, fmt.Errorf("artifact %s gzip decode: %w", name, err)
	}
	data, err := io.ReadAll(reader)
	_ = reader.Close()
	if err != nil {
		return nil, fmt.Errorf("artifact %s gzip read: %w", name, err)
	}
	return data, nil
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

func sha256Maybe(value string) string {
	if strings.TrimSpace(value) == "" {
		return ""
	}
	return sha256Hex([]byte(value))
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

func knownIntNull(known bool, value int) any {
	return knownNull(known, value, intNull(value))
}

func intPointerNull(value *int) any {
	if value == nil {
		return nil
	}
	return *value
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

func knownFloatNull(known bool, value float64) any {
	return knownNull(known, value, floatNull(value))
}

func knownNull(known bool, value, fallback any) any {
	if known {
		return value
	}
	return fallback
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
  phase TEXT NOT NULL DEFAULT 'mixed',
  traffic_json TEXT NOT NULL CHECK (json_valid(traffic_json)),
  concurrency_json TEXT NOT NULL CHECK (json_valid(concurrency_json)),
  samples INTEGER NOT NULL CHECK (samples > 0),
  repeats INTEGER NOT NULL DEFAULT 1 CHECK (repeats > 0),
  save_detailed INTEGER NOT NULL DEFAULT 1 CHECK (save_detailed IN (0, 1)),
  capture_payload_artifacts INTEGER NOT NULL DEFAULT 0 CHECK (capture_payload_artifacts IN (0, 1)),
  dataset_json TEXT CHECK (dataset_json IS NULL OR json_valid(dataset_json)),
  request_json TEXT CHECK (request_json IS NULL OR json_valid(request_json)),
  load_json TEXT CHECK (load_json IS NULL OR json_valid(load_json)),
  metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json)),
  UNIQUE (run_id, name)
);
CREATE TABLE datasets (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  workload_id TEXT NOT NULL REFERENCES workloads(id) ON DELETE CASCADE,
  type TEXT NOT NULL,
  uri TEXT,
  path TEXT,
  split TEXT,
  selection TEXT,
  sample_count INTEGER NOT NULL CHECK (sample_count > 0),
  seed INTEGER,
  config_json TEXT NOT NULL CHECK (json_valid(config_json)),
  canonical_path TEXT,
  rendered_path TEXT,
  request_count INTEGER NOT NULL CHECK (request_count >= 0),
  sha256 TEXT,
  metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json)),
  UNIQUE (run_id, workload_id)
);
CREATE TABLE source_records (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  source_record_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
  content_json TEXT NOT NULL CHECK (json_valid(content_json)),
  sha256 TEXT NOT NULL,
  metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json)),
  UNIQUE (dataset_id, source_record_id)
);
CREATE TABLE canonical_requests (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  source_record_row_id INTEGER REFERENCES source_records(id) ON DELETE SET NULL,
  workload_id TEXT NOT NULL REFERENCES workloads(id) ON DELETE CASCADE,
  request_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
  conversation_id TEXT,
  turn_index INTEGER,
  mode TEXT NOT NULL,
  prompt_sha256 TEXT,
  max_output_tokens INTEGER CHECK (max_output_tokens IS NULL OR max_output_tokens >= 0),
  input_tokens_expected INTEGER CHECK (input_tokens_expected IS NULL OR input_tokens_expected >= 0),
  output_tokens_expected INTEGER CHECK (output_tokens_expected IS NULL OR output_tokens_expected >= 0),
  arrival_offset_ms INTEGER NOT NULL DEFAULT 0,
  messages_json TEXT CHECK (messages_json IS NULL OR json_valid(messages_json)),
  attachments_json TEXT CHECK (attachments_json IS NULL OR json_valid(attachments_json)),
  metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json)),
  canonical_json TEXT NOT NULL CHECK (json_valid(canonical_json)),
  UNIQUE (dataset_id, ordinal)
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
  first_byte_at TEXT,
  first_byte_ms REAL,
  first_token_at TEXT,
  completed_at TEXT,
  latency_ms REAL,
  ttft_ms REAL,
  tpot_ms REAL,
  itl_mean_ms REAL,
  prompt_tokens INTEGER CHECK (prompt_tokens >= 0),
  completion_tokens INTEGER CHECK (completion_tokens >= 0),
  total_tokens INTEGER CHECK (total_tokens >= 0),
  output_tok_s REAL,
  total_tok_s REAL,
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
CREATE INDEX idx_canonical_requests_dataset ON canonical_requests (dataset_id, ordinal);
CREATE INDEX idx_source_records_dataset ON source_records (dataset_id, ordinal);
CREATE INDEX idx_requests_measurement ON requests (measurement_id, status);
CREATE INDEX idx_request_stream_events_request ON request_stream_events (request_row_id, event_index);
CREATE INDEX idx_telemetry_samples_series_time ON telemetry_samples (series_id, timestamp);
CREATE INDEX idx_telemetry_samples_phase_time ON telemetry_samples (phase_id, timestamp);
CREATE INDEX idx_phases_run_time ON phases (run_id, started_at);
CREATE INDEX idx_events_run_time ON events (run_id, timestamp);
`
