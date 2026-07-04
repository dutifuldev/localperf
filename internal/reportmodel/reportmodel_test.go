package reportmodel

import (
	"testing"
	"time"

	"github.com/dutifuldev/localperf/internal/report"
)

func TestBuildMergesRunsButSplitsActiveContexts(t *testing.T) {
	doc := report.SQLiteReportDocument{
		GeneratedAt: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
		ThroughputRows: []report.SQLiteReportThroughputRow{
			throughputRow("run-1", "32k", "16k active context", 16384, 1),
			throughputRow("run-2", "32k", "16k active context", 16384, 4),
			throughputRow("run-3", "32k", "32k active context", 32768, 8),
		},
	}

	model := Build("/tmp/report.sqlite", doc)
	if len(model.Throughput.Tables) != 2 {
		t.Fatalf("tables = %d, want 2", len(model.Throughput.Tables))
	}
	first := model.Throughput.Tables[0]
	if first.ContextLabel != "16k active context" {
		t.Fatalf("first context = %q, want 16k active context", first.ContextLabel)
	}
	if first.RunID != "" || len(first.RunIDs) != 2 {
		t.Fatalf("first run IDs = %q/%v, want merged multi-run table", first.RunID, first.RunIDs)
	}
	if len(first.Rows) != 2 {
		t.Fatalf("first rows = %d, want two merged concurrency points", len(first.Rows))
	}
	second := model.Throughput.Tables[1]
	if second.ContextLabel != "32k active context" || len(second.Rows) != 1 {
		t.Fatalf("second table = %+v, want separate 32k context table", second)
	}
}

func TestBuildLabelsUnverifiedContext(t *testing.T) {
	doc := report.SQLiteReportDocument{
		GeneratedAt: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
		ThroughputRows: []report.SQLiteReportThroughputRow{
			{
				Mode:          "decode",
				RunID:         "run-1",
				MeasurementID: 1,
				ProfileID:     "profile-1",
				Profile:       "64k",
				Model:         "test/model",
				ContextWindow: 65536,
				ContextLabel:  "unverified (declared 64k active)",
				Concurrency:   1,
				Shape:         "-",
				Detail: report.SQLiteReportCellDetail{
					Available:     true,
					Mode:          "decode",
					MeasurementID: 1,
					ContextLabel:  "unverified (declared 64k active)",
				},
			},
		},
	}

	table := Build("/tmp/report.sqlite", doc).Throughput.Tables[0]
	if table.ContextStatus != "unverified" {
		t.Fatalf("context status = %q, want unverified", table.ContextStatus)
	}
	if table.Warning == "" {
		t.Fatal("warning is empty, want unverified warning")
	}
}

func throughputRow(runID, profile, contextLabel string, contextSortKey, concurrency int) report.SQLiteReportThroughputRow {
	return report.SQLiteReportThroughputRow{
		Mode:              "decode",
		RunID:             runID,
		MeasurementID:     int64(concurrency),
		ProfileID:         "profile-1",
		Profile:           profile,
		Model:             "test/model",
		ContextWindow:     32768,
		ContextLabel:      contextLabel,
		ContextSortKey:    contextSortKey,
		Concurrency:       concurrency,
		Shape:             contextLabel,
		ThroughputTokS:    "100",
		PerUserTokS:       "100",
		CompletedRequests: concurrency,
		Detail: report.SQLiteReportCellDetail{
			Available:     true,
			Mode:          "decode",
			RunID:         runID,
			MeasurementID: int64(concurrency),
			Profile:       profile,
			ContextLabel:  contextLabel,
			Shape:         contextLabel,
		},
	}
}
