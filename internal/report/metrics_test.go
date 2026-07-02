package report

import (
	"database/sql"
	"path/filepath"
	"strings"
	"testing"
)

// TestTokenWeightedITLMatchesBruteForce checks the SQL derivation
// sum(itl_mean * (completion-1)) / sum(completion-1) against gap arrays
// computed by hand, and that it differs from the request-weighted
// mean-of-means on a skewed fixture.
func TestTokenWeightedITLAndDerivedRequestMetrics(t *testing.T) {
	artifactPath := filepath.Join(t.TempDir(), "run.sqlite")
	createTestSQLiteHTMLArtifact(t, artifactPath, "Derived")
	db, err := sql.Open("sqlite", artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	// Replace the fixture's request rows with hand-computed ones.
	if _, err := db.Exec(`DELETE FROM requests`); err != nil {
		t.Fatal(err)
	}
	// Request A: gaps [10,10,10] -> itl_mean 10ms, 4 completion tokens.
	// Request B: gaps [100] -> itl_mean 100ms, 2 completion tokens.
	// Token-weighted ITL = (30+100)/4 = 32.5ms; mean-of-means would be 55ms.
	insertDerivedRequest(t, db, 1, 0, "completed", "2026-01-01T00:00:00Z", "2026-01-01T00:00:10Z", 10, 4, "")
	insertDerivedRequest(t, db, 1, 1, "completed", "2026-01-01T00:00:00Z", "2026-01-01T00:00:10Z", 100, 2, "")
	insertDerivedRequest(t, db, 1, 2, "failed", "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", 0, 0, "timeout")
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	doc, err := LoadSQLiteReport(artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(doc.Measurements) != 1 {
		t.Fatalf("measurements = %d, want 1", len(doc.Measurements))
	}
	measurement := doc.Measurements[0]
	if measurement.ITLTokenWeightedMS != "32.500" {
		t.Fatalf("token-weighted ITL = %q, want 32.500 (request-weighted mean-of-means would be 55)", measurement.ITLTokenWeightedMS)
	}
	// Two 10s requests over a 10s span: achieved concurrency 2 of requested 4.
	if measurement.AchievedConcurrency != "~2 (of 4)" {
		t.Fatalf("achieved concurrency = %q, want ~2 (of 4)", measurement.AchievedConcurrency)
	}
	if measurement.FailureBreakdown != "1 timeout" {
		t.Fatalf("failure breakdown = %q, want 1 timeout", measurement.FailureBreakdown)
	}
	// 2 completed requests over 1000ms wall time.
	if measurement.RPS != "2.000" {
		t.Fatalf("RPS = %q, want 2.000", measurement.RPS)
	}
}

func TestRepeatAggregationRendersSpreadAndRepeatRows(t *testing.T) {
	artifactPath := filepath.Join(t.TempDir(), "run.sqlite")
	createTestSQLiteHTMLArtifact(t, artifactPath, "Repeats")
	db, err := sql.Open("sqlite", artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	// Second repeat of the fixture's (profile-1, workload-1, c4) point.
	result, err := db.Exec(`INSERT INTO measurements (
		run_id, profile_id, workload_id, repeat_index, concurrency, samples_requested,
		status, started_at, completed_at, wall_time_ms, completed_requests, failed_requests,
		prompt_tokens, completion_tokens, total_tokens, aggregate_output_tok_s,
		per_user_output_tok_s, aggregate_total_tok_s
	) VALUES (
		'run-1', 'profile-1', 'workload-1', 1, 4, 8, 'completed',
		'2026-01-01T00:02:00Z', '2026-01-01T00:03:00Z', 1000, 2, 0, 200, 20, 220,
		133.4, 66.7, 233.4
	)`)
	if err != nil {
		t.Fatal(err)
	}
	measurementID, err := result.LastInsertId()
	if err != nil {
		t.Fatal(err)
	}
	seedSQLiteHTMLMetrics(t, db, measurementID)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	doc, err := LoadSQLiteReport(artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(doc.Measurements) != 1 {
		t.Fatalf("aggregated measurements = %d, want 1", len(doc.Measurements))
	}
	combined := doc.Measurements[0]
	if combined.RepeatCount != 2 {
		t.Fatalf("repeat count = %d, want 2", combined.RepeatCount)
	}
	// mean(123.4, 133.4) = 128.4, sample stddev = 7.071
	if !strings.HasPrefix(combined.OutputTokS, "128.400 ±") {
		t.Fatalf("aggregated output tok/s = %q, want mean ± spread", combined.OutputTokS)
	}
	if combined.CompletedRequests != 4 {
		t.Fatalf("aggregated completed = %d, want summed 4", combined.CompletedRequests)
	}
	// Token totals must sum with request counts so per-request derivations
	// stay exact: 400 prompt / 4 requests keeps the 100 in / 10 out shape.
	if combined.PromptTokensValue != 400 || combined.CompletionTokensValue != 40 {
		t.Fatalf("aggregated tokens = %d/%d, want 400/40", combined.PromptTokensValue, combined.CompletionTokensValue)
	}
	if shape := requestShape(combined); shape != "100 in / 10 out" {
		t.Fatalf("aggregated shape = %q, want 100 in / 10 out", shape)
	}
	if combined.WallTimeMSValue != 2000 {
		t.Fatalf("aggregated wall time = %f, want summed 2000", combined.WallTimeMSValue)
	}
	if len(doc.RepeatDetails) != 2 {
		t.Fatalf("repeat details = %d, want 2", len(doc.RepeatDetails))
	}

	var out strings.Builder
	if err := RenderHTMLReport(&out, doc, HTMLReportOptions{}); err != nil {
		t.Fatal(err)
	}
	html := out.String()
	for _, want := range []string{"Repeats", "Per-repeat rows", "±", "&times;2"} {
		if !strings.Contains(html, want) {
			t.Fatalf("HTML report missing %q", want)
		}
	}
}

func TestSLOGoodputDerivation(t *testing.T) {
	artifactPath := filepath.Join(t.TempDir(), "run.sqlite")
	createTestSQLiteHTMLArtifact(t, artifactPath, "SLO")
	db, err := sql.Open("sqlite", artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`UPDATE workloads SET metadata_json = '{"slo":{"ttft_p95_ms":500}}' WHERE id = 'workload-1'`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`DELETE FROM requests`); err != nil {
		t.Fatal(err)
	}
	for index, ttft := range []float64{100, 900} {
		if _, err := db.Exec(`INSERT INTO requests (
			measurement_id, request_index, status, streamed, started_at, completed_at,
			ttft_ms, latency_ms, prompt_tokens, completion_tokens
		) VALUES (1, ?, 'completed', 1, '2026-01-01T00:00:00Z', '2026-01-01T00:00:05Z', ?, 5000, 100, 10)`,
			index, ttft); err != nil {
			t.Fatal(err)
		}
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	doc, err := LoadSQLiteReport(artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	if !doc.HasSLO {
		t.Fatal("HasSLO = false, want true when a workload declares an SLO")
	}
	measurement := doc.Measurements[0]
	if measurement.SLOMetPct != "50%" {
		t.Fatalf("SLO met = %q, want 50%%", measurement.SLOMetPct)
	}
	// 1 SLO-met request over 1000ms wall time.
	if measurement.GoodputRPS != "1.000" {
		t.Fatalf("goodput = %q, want 1.000", measurement.GoodputRPS)
	}
	if measurement.SLONote != "ttft<=500ms" {
		t.Fatalf("SLO note = %q, want ttft<=500ms", measurement.SLONote)
	}
	var out strings.Builder
	if err := RenderHTMLReport(&out, doc, HTMLReportOptions{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "% in SLO") || !strings.Contains(out.String(), "Goodput req/s") {
		t.Fatal("HTML report missing SLO columns when an SLO is declared")
	}
	// Goodput must be visible in the headline table, not only the hidden
	// detail sections.
	if !strings.Contains(out.String(), "SLO / goodput") || !strings.Contains(out.String(), "50% / 1.000") {
		t.Fatal("HTML report missing visible SLO/goodput in the throughput table")
	}
}

func insertDerivedRequest(t *testing.T, db *sql.DB, measurementID int64, index int, status, startedAt, completedAt string, itlMeanMS float64, completionTokens int, errorType string) {
	t.Helper()
	var itl any
	if itlMeanMS > 0 {
		itl = itlMeanMS
	}
	var errType any
	if errorType != "" {
		errType = errorType
	}
	if _, err := db.Exec(`INSERT INTO requests (
		measurement_id, request_index, status, streamed, started_at, completed_at,
		itl_mean_ms, prompt_tokens, completion_tokens
	) VALUES (?, ?, ?, 1, ?, ?, ?, 100, ?)`,
		measurementID, index, status, startedAt, completedAt, itl, completionTokens); err != nil {
		t.Fatal(err)
	}
	if errType != nil {
		if _, err := db.Exec(`UPDATE requests SET error_type = ? WHERE measurement_id = ? AND request_index = ?`, errType, measurementID, index); err != nil {
			t.Fatal(err)
		}
	}
}
