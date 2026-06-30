package vllmbench

import (
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRenderSQLiteHTMLReportEscapesAndIsStandalone(t *testing.T) {
	artifactPath := testSQLiteHTMLArtifact(t, "Escaping <script>alert(1)</script>")
	doc, err := LoadSQLiteReport(artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	var out strings.Builder
	if err := RenderHTMLReport(&out, doc, HTMLReportOptions{}); err != nil {
		t.Fatal(err)
	}
	html := out.String()
	for _, want := range []string{
		"<!doctype html>",
		"<style>",
		"<svg class=\"svg-chart\"",
		"Escaping &lt;script&gt;alert(1)&lt;/script&gt;",
		"Measurements",
		"Privacy",
	} {
		if !strings.Contains(html, want) {
			t.Fatalf("HTML report missing %q:\n%s", want, html)
		}
	}
	for _, forbidden := range []string{"<script>alert(1)</script>", "https://", "http://", "<link ", "src="} {
		if strings.Contains(html, forbidden) {
			t.Fatalf("HTML report contains forbidden %q:\n%s", forbidden, html)
		}
	}
}

func TestWriteSQLiteHTMLReportStoresArtifact(t *testing.T) {
	artifactPath := testSQLiteHTMLArtifact(t, "Stored HTML")
	outputPath := filepath.Join(t.TempDir(), "report.html")
	if err := WriteSQLiteHTMLReport(artifactPath, outputPath, HTMLReportOptions{Store: true}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "Stored HTML") {
		t.Fatalf("rendered HTML missing run title:\n%s", data)
	}
	if err := CheckSQLiteArtifact(artifactPath); err != nil {
		t.Fatalf("artifact check after storing HTML: %v", err)
	}
	db, err := sql.Open("sqlite", artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var reportRows, artifactRows int
	if err := db.QueryRow(`SELECT COUNT(*) FROM reports WHERE name = 'report.html' AND format = 'html'`).Scan(&reportRows); err != nil {
		t.Fatal(err)
	}
	if err := db.QueryRow(`SELECT COUNT(*) FROM artifacts WHERE kind = 'normalized_report' AND name = 'report.html' AND media_type = 'text/html'`).Scan(&artifactRows); err != nil {
		t.Fatal(err)
	}
	if reportRows != 1 || artifactRows != 1 {
		t.Fatalf("stored report/artifact rows = %d/%d, want 1/1", reportRows, artifactRows)
	}
}

func TestWriteSQLiteHTMLReportUsesDefaultOutput(t *testing.T) {
	dir := t.TempDir()
	artifactPath := testSQLiteHTMLArtifact(t, "Default Output")
	copyPath := filepath.Join(dir, "run.sqlite")
	data, err := os.ReadFile(artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(copyPath, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := WriteSQLiteHTMLReport(copyPath, "", HTMLReportOptions{}); err != nil {
		t.Fatal(err)
	}
	outputPath := filepath.Join(dir, "run.html")
	html, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(html), "Default Output") {
		t.Fatalf("default HTML output missing run title:\n%s", html)
	}
}

func TestWriteSQLiteHTMLReportRejectsOverwritingSourceArtifact(t *testing.T) {
	artifactPath := testSQLiteHTMLArtifact(t, "No Overwrite")
	if err := WriteSQLiteHTMLReport(artifactPath, artifactPath, HTMLReportOptions{}); err == nil {
		t.Fatal("WriteSQLiteHTMLReport error = nil, want same-path rejection")
	}
	if err := CheckSQLiteArtifact(artifactPath); err != nil {
		t.Fatalf("source artifact was corrupted after rejected render: %v", err)
	}
}

func TestWriteSQLiteHTMLReportRejectsSymlinkOutputToSourceArtifact(t *testing.T) {
	artifactPath := testSQLiteHTMLArtifact(t, "No Symlink Overwrite")
	outputPath := filepath.Join(t.TempDir(), "report.html")
	if err := os.Symlink(artifactPath, outputPath); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	if err := WriteSQLiteHTMLReport(artifactPath, outputPath, HTMLReportOptions{}); err == nil {
		t.Fatal("WriteSQLiteHTMLReport error = nil, want symlink output rejection")
	}
	if err := CheckSQLiteArtifact(artifactPath); err != nil {
		t.Fatalf("source artifact was corrupted after rejected symlink render: %v", err)
	}
}

func TestWriteSQLiteHTMLReportRejectsHardlinkOutputToSourceArtifact(t *testing.T) {
	artifactPath := testSQLiteHTMLArtifact(t, "No Hardlink Overwrite")
	outputPath := filepath.Join(t.TempDir(), "report.html")
	if err := os.Link(artifactPath, outputPath); err != nil {
		t.Skipf("hardlink unavailable: %v", err)
	}
	if err := WriteSQLiteHTMLReport(artifactPath, outputPath, HTMLReportOptions{}); err == nil {
		t.Fatal("WriteSQLiteHTMLReport error = nil, want hardlink output rejection")
	}
	if err := CheckSQLiteArtifact(artifactPath); err != nil {
		t.Fatalf("source artifact was corrupted after rejected hardlink render: %v", err)
	}
}

func TestWriteSQLiteHTMLReportRejectsDefaultOutputOverSourceArtifact(t *testing.T) {
	dir := t.TempDir()
	artifactPath := testSQLiteHTMLArtifact(t, "HTML Named Artifact")
	htmlNamedArtifact := filepath.Join(dir, "run.html")
	data, err := os.ReadFile(artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(htmlNamedArtifact, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := WriteSQLiteHTMLReport(htmlNamedArtifact, "", HTMLReportOptions{}); err == nil {
		t.Fatal("WriteSQLiteHTMLReport error = nil, want default same-path rejection")
	}
	if err := CheckSQLiteArtifact(htmlNamedArtifact); err != nil {
		t.Fatalf("HTML-named source artifact was corrupted after rejected render: %v", err)
	}
}

func TestLoadSQLiteReportFallsBackToAggregateOnlyArtifact(t *testing.T) {
	artifactPath := testSQLiteHTMLArtifact(t, "Aggregate Only")
	db, err := sql.Open("sqlite", artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`UPDATE workloads SET save_detailed = 0`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`DELETE FROM requests`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`DELETE FROM metric_stats WHERE metric IN (
		'request_output_throughput', 'request_ttft', 'request_tpot', 'request_itl_mean', 'latency'
	)`); err != nil {
		t.Fatal(err)
	}
	var measurementID int64
	if err := db.QueryRow(`SELECT id FROM measurements ORDER BY id LIMIT 1`).Scan(&measurementID); err != nil {
		t.Fatal(err)
	}
	for _, metric := range []struct {
		name  string
		mean  float64
		count int
	}{
		{"ttft", 321, 2},
		{"tpot", 45, 2},
	} {
		if _, err := db.Exec(`INSERT INTO metric_stats (
			measurement_id, metric, unit, mean, count
		) VALUES (?, ?, 'ms', ?, ?)`, measurementID, metric.name, metric.mean, metric.count); err != nil {
			t.Fatal(err)
		}
	}
	doc, err := LoadSQLiteReport(artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	if doc.RequestSummary.Total != 2 || doc.RequestSummary.Completed != 2 || doc.RequestSummary.Failed != 0 {
		t.Fatalf("request summary = %+v, want aggregate measurement counts", doc.RequestSummary)
	}
	if doc.RequestSummary.OutputTokSMean != "123.400" || doc.RequestSummary.TTFTMeanMS != "321.000" || doc.RequestSummary.TPOTMeanMS != "45.000" {
		t.Fatalf("aggregate request summary = %+v", doc.RequestSummary)
	}
	if got := doc.Measurements[0].TTFTMeanMS; got != "321.000" {
		t.Fatalf("measurement TTFT = %q, want aggregate fallback", got)
	}
	if got := doc.Measurements[0].TPOTMeanMS; got != "45.000" {
		t.Fatalf("measurement TPOT = %q, want aggregate fallback", got)
	}
}

func TestLoadSQLiteReportMergesDetailedAndAggregateOnlySummary(t *testing.T) {
	artifactPath := testSQLiteHTMLArtifact(t, "Mixed Summary")
	db, err := sql.Open("sqlite", artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	insertAggregateOnlyMeasurement(t, db, 3, 300, 50, 200)

	doc, err := LoadSQLiteReport(artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	if doc.RequestSummary.Total != 5 || doc.RequestSummary.Completed != 5 || doc.RequestSummary.Failed != 0 {
		t.Fatalf("mixed request summary = %+v, want detailed plus aggregate counts", doc.RequestSummary)
	}
	if doc.RequestSummary.TTFTMeanMS != "260.000" || doc.RequestSummary.TPOTMeanMS != "42.000" {
		t.Fatalf("mixed request summary did not merge weighted metrics: %+v", doc.RequestSummary)
	}
}

func TestLoadSQLiteReportIgnoresEmptyNotableEventMessages(t *testing.T) {
	artifactPath := testSQLiteHTMLArtifact(t, "Empty Events")
	doc, err := LoadSQLiteReport(artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(doc.NotableEvents) != 0 {
		t.Fatalf("notable events = %+v, want empty routine events ignored", doc.NotableEvents)
	}
}

func insertAggregateOnlyMeasurement(t *testing.T, db *sql.DB, completed int, ttft, tpot, outputTokS float64) int64 {
	t.Helper()
	if _, err := db.Exec(`INSERT INTO workloads (
		id, run_id, name, phase, traffic_json, concurrency_json, samples, repeats,
		save_detailed, capture_payload_artifacts
	) SELECT 'aggregate-only', run_id, 'aggregate-only', phase, traffic_json, concurrency_json, ?, 1, 0, capture_payload_artifacts
		FROM workloads ORDER BY id LIMIT 1`, completed); err != nil {
		t.Fatal(err)
	}
	result, err := db.Exec(`INSERT INTO measurements (
		run_id, profile_id, workload_id, repeat_index, concurrency, samples_requested, status,
		completed_requests, failed_requests, prompt_tokens, completion_tokens, total_tokens,
		aggregate_output_tok_s, per_user_output_tok_s, aggregate_total_tok_s
	) SELECT run_id, profile_id, 'aggregate-only', 0, 2, ?, 'completed',
		?, 0, 300, 30, 330, ?, ? / 2.0, ? + 100.0
		FROM measurements ORDER BY id LIMIT 1`, completed, completed, outputTokS, outputTokS, outputTokS)
	if err != nil {
		t.Fatal(err)
	}
	measurementID, err := result.LastInsertId()
	if err != nil {
		t.Fatal(err)
	}
	insertAggregateMetric(t, db, measurementID, "ttft", "ms", ttft, completed)
	insertAggregateMetric(t, db, measurementID, "tpot", "ms", tpot, completed)
	return measurementID
}

func insertAggregateMetric(t *testing.T, db *sql.DB, measurementID int64, name, unit string, mean float64, count int) {
	t.Helper()
	if _, err := db.Exec(`INSERT INTO metric_stats (
		measurement_id, metric, unit, mean, count
	) VALUES (?, ?, ?, ?, ?)`, measurementID, name, unit, mean, count); err != nil {
		t.Fatal(err)
	}
}

func testSQLiteHTMLArtifact(t *testing.T, name string) string {
	t.Helper()
	spec := testSpec()
	spec.Name = name
	spec.OutputDir = t.TempDir()
	appendTimestamp := false
	spec.Runner.AppendTimestampToRun = &appendTimestamp
	spec.Warmup.Enabled = false
	spec.Workloads[0].NumPrompts = 1
	spec.Workloads[0].MaxConcurrency = []int{1}
	ApplyDefaults(&spec)
	runDir := filepath.Join(t.TempDir(), "run")
	summary, err := Execute(context.Background(), spec, RunOptions{DryRun: true, RunDir: runDir})
	if err != nil {
		t.Fatal(err)
	}
	if summary.ArtifactPath == "" {
		t.Fatal("dry run did not write an artifact")
	}
	seedSQLiteHTMLMetrics(t, summary.ArtifactPath)
	return summary.ArtifactPath
}

func seedSQLiteHTMLMetrics(t *testing.T, artifactPath string) {
	t.Helper()
	db, err := sql.Open("sqlite", artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`UPDATE measurements SET
		status = 'completed',
		completed_requests = 2,
		failed_requests = 0,
		prompt_tokens = 200,
		completion_tokens = 20,
		total_tokens = 220,
		aggregate_output_tok_s = 123.4,
		per_user_output_tok_s = 61.7,
		aggregate_total_tok_s = 223.4
		WHERE id = (SELECT id FROM measurements ORDER BY id LIMIT 1)`); err != nil {
		t.Fatal(err)
	}
	var measurementID int64
	if err := db.QueryRow(`SELECT id FROM measurements ORDER BY id LIMIT 1`).Scan(&measurementID); err != nil {
		t.Fatal(err)
	}
	for _, metric := range []struct {
		name   string
		unit   string
		mean   float64
		stddev float64
	}{
		{"request_output_throughput", "tok/s", 61.7, 2.5},
		{"latency", "ms", 1200, 50},
		{"request_ttft", "ms", 200, 10},
		{"request_tpot", "ms", 30, 3},
		{"request_itl_mean", "ms", 28, 2},
	} {
		if _, err := db.Exec(`INSERT INTO metric_stats (
			measurement_id, metric, unit, mean, stddev, min, p50, p90, p95, p99, max, count
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 2)`,
			measurementID, metric.name, metric.unit, metric.mean, metric.stddev,
			metric.mean-1, metric.mean, metric.mean+1, metric.mean+2, metric.mean+3, metric.mean+4); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 2; i++ {
		if _, err := db.Exec(`INSERT INTO requests (
			measurement_id, request_index, request_id, status, streamed, started_at,
			completed_at, latency_ms, ttft_ms, tpot_ms, itl_mean_ms,
			prompt_tokens, completion_tokens, total_tokens, output_tok_s, total_tok_s
		) VALUES (?, ?, ?, 'completed', 1, '2026-01-01T00:00:00Z',
			'2026-01-01T00:00:01Z', 1000, 200, 30, 28, 100, 10, 110, 61.7, 111.7)`,
			measurementID, i, "request"); err != nil {
			t.Fatal(err)
		}
	}
}
