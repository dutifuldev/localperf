package vllmbench

import (
	"database/sql"
	"path/filepath"
	"testing"
)

func TestInsertMeasurementDetailsImportsVLLMBenchRequestSamples(t *testing.T) {
	runDir := t.TempDir()
	resultFile := filepath.Join("results", "vllm.json")
	writeFile(t, filepath.Join(runDir, resultFile), `{
  "date": "20260102-030405",
  "completed": 2,
  "failed": 0,
  "input_lens": [100, 200],
  "output_lens": [3, 2],
  "ttfts": [0.1, 0.2],
  "itls": [[0.05, 0.07], [0.1]],
  "start_times": [10.0, 11.5]
}`)
	db, err := createSQLiteArtifact(filepath.Join(t.TempDir(), "artifact.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err := withSQLiteTx(db, func(tx *sql.Tx) error {
		seedMeasurementParents(t, tx)
		id, err := insertMeasurement(tx, measurementInsert{
			runID: "run",
			planned: PlannedRun{
				Profile:     Profile{Name: "profile"},
				Workload:    Workload{Name: "workload", NumPrompts: 2},
				Concurrency: 2,
			},
			status: "completed",
		})
		if err != nil {
			return err
		}
		return insertMeasurementDetails(tx, runDir, id, ReportRow{}, resultFile)
	}); err != nil {
		t.Fatal(err)
	}
	var requestRows, promptTokens, completionTokens int
	var minTTFT, maxTPOT float64
	if err := db.QueryRow(`
		SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0),
		       COALESCE(MIN(ttft_ms), 0), COALESCE(MAX(tpot_ms), 0)
		FROM requests`).Scan(&requestRows, &promptTokens, &completionTokens, &minTTFT, &maxTPOT); err != nil {
		t.Fatal(err)
	}
	if requestRows != 2 || promptTokens != 300 || completionTokens != 5 {
		t.Fatalf("request rows/tokens = %d/%d/%d, want 2/300/5", requestRows, promptTokens, completionTokens)
	}
	if !near(minTTFT, 100) || !near(maxTPOT, 100) {
		t.Fatalf("request timing fields = min ttft %.3f max tpot %.3f, want 100/100", minTTFT, maxTPOT)
	}
	for _, metric := range []string{"request_output_throughput", "request_ttft", "request_tpot", "request_itl_mean"} {
		var count int
		var stddev sql.NullFloat64
		if err := db.QueryRow(`SELECT count, stddev FROM metric_stats WHERE metric = ?`, metric).Scan(&count, &stddev); err != nil {
			t.Fatalf("%s metric missing: %v", metric, err)
		}
		if count != 2 || !stddev.Valid || stddev.Float64 <= 0 {
			t.Fatalf("%s metric count/stddev = %d/%v, want 2/positive", metric, count, stddev)
		}
	}
}
