package main

import (
	"strings"
	"testing"
)

func TestCheckResultsAcceptsLoadedRow(t *testing.T) {
	input := `{"candidate_id":"ctx4096-seq1-small","status":"load_complete","candidate":{"max_model_len":4096},"startup":{},"shutdown":{},"telemetry":{"tools":{"tegrastats_available":true},"tegrastats":{"sample_count":3,"parsed_sample_count":3,"ram_used_delta_gib":81.2}},"load_short_decode":{"successes":1,"errors":0,"wall_seconds":2.1,"completion_tokens_per_second":12.3,"total_tokens_per_second":400.0,"latency_seconds":{"p50":2.1},"memory_monitor":{"samples":2}}}`

	sum, err := checkResults(strings.NewReader(input+"\n"), config{minRows: 1, requireTegrastats: true})
	if err != nil {
		t.Fatalf("checkResults returned unexpected error: %v", err)
	}
	if len(sum.Issues) != 0 {
		t.Fatalf("expected no issues, got %#v", sum.Issues)
	}
	if sum.Rows != 1 || sum.LoadRows != 1 || sum.RowsWithTegrastats != 1 {
		t.Fatalf("unexpected summary: %#v", sum)
	}
}

func TestCheckResultsRequiresMinimumRows(t *testing.T) {
	input := `{"candidate_id":"ctx4096-seq1-small","status":"load_complete","candidate":{"max_model_len":4096},"startup":{},"shutdown":{},"telemetry":{"tools":{"tegrastats_available":true},"tegrastats":{"sample_count":3,"parsed_sample_count":3,"ram_used_delta_gib":81.2}},"load_short_decode":{"successes":1,"errors":0,"wall_seconds":2.1,"completion_tokens_per_second":12.3,"total_tokens_per_second":400.0,"latency_seconds":{"p50":2.1},"memory_monitor":{"samples":2}}}`

	sum, err := checkResults(strings.NewReader(input+"\n"), config{minRows: 2, requireTegrastats: true})
	if err != nil {
		t.Fatalf("checkResults returned unexpected error: %v", err)
	}
	if len(sum.Issues) == 0 {
		t.Fatal("expected minimum-row issue")
	}
	if !strings.Contains(strings.Join(sum.Issues, "\n"), "need at least 2") {
		t.Fatalf("expected min-row issue, got %#v", sum.Issues)
	}
}

func TestCheckResultsRejectsMissingTelemetry(t *testing.T) {
	input := `{"candidate_id":"ctx4096-seq1-small","status":"load_complete","candidate":{"max_model_len":4096},"startup":{},"shutdown":{},"load_short_decode":{"successes":1,"errors":0,"wall_seconds":2.1,"completion_tokens_per_second":12.3,"total_tokens_per_second":400.0,"latency_seconds":{"p50":2.1},"memory_monitor":{"samples":2}}}`

	sum, err := checkResults(strings.NewReader(input+"\n"), config{minRows: 1, requireTegrastats: true})
	if err != nil {
		t.Fatalf("checkResults returned unexpected error: %v", err)
	}
	if len(sum.Issues) == 0 {
		t.Fatal("expected telemetry issue")
	}
	if !strings.Contains(strings.Join(sum.Issues, "\n"), "missing telemetry") {
		t.Fatalf("expected telemetry issue, got %#v", sum.Issues)
	}
}
