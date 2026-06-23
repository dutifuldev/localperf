package main

import (
	"strings"
	"testing"
)

func TestCheckResultsAcceptsLoadedRow(t *testing.T) {
	input := `{"candidate_id":"ctx4096-seq1-small","status":"load_complete","candidate":{"max_model_len":4096},"startup":{},"shutdown":{},"telemetry":{"tools":{"tegrastats_available":true},"tegrastats":{"sample_count":3,"parsed_sample_count":3,"ram_used_delta_gib":81.2}},"load_short_decode":{"successes":1,"errors":0,"wall_seconds":2.1,"completion_tokens_per_second":12.3,"total_tokens_per_second":400.0,"latency_seconds":{"p50":2.1,"p95":2.1},"memory_monitor":{"samples":2}}}`

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
	input := `{"candidate_id":"ctx4096-seq1-small","status":"load_complete","candidate":{"max_model_len":4096},"startup":{},"shutdown":{},"telemetry":{"tools":{"tegrastats_available":true},"tegrastats":{"sample_count":3,"parsed_sample_count":3,"ram_used_delta_gib":81.2}},"load_short_decode":{"successes":1,"errors":0,"wall_seconds":2.1,"completion_tokens_per_second":12.3,"total_tokens_per_second":400.0,"latency_seconds":{"p50":2.1,"p95":2.1},"memory_monitor":{"samples":2}}}`

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
	input := `{"candidate_id":"ctx4096-seq1-small","status":"load_complete","candidate":{"max_model_len":4096},"startup":{},"shutdown":{},"load_short_decode":{"successes":1,"errors":0,"wall_seconds":2.1,"completion_tokens_per_second":12.3,"total_tokens_per_second":400.0,"latency_seconds":{"p50":2.1,"p95":2.1},"memory_monitor":{"samples":2}}}`

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

func TestCheckResultsRejectsMissingCoverage(t *testing.T) {
	input := `{"candidate_id":"ctx4096-seq1-small","status":"load_complete","candidate":{"max_model_len":4096,"max_num_seqs":1},"startup":{},"shutdown":{},"telemetry":{"tools":{"tegrastats_available":true},"tegrastats":{"sample_count":3,"parsed_sample_count":3,"ram_used_delta_gib":81.2}},"load_short_decode":{"successes":1,"errors":0,"wall_seconds":2.1,"completion_tokens_per_second":12.3,"total_tokens_per_second":400.0,"latency_seconds":{"p50":2.1,"p95":2.1},"memory_monitor":{"samples":2}}}`

	sum, err := checkResults(strings.NewReader(input+"\n"), config{
		minRows:           1,
		requireTegrastats: true,
		requireContexts:   intList{100000},
		requireMaxContext: 100000,
		requireMaxSeqs:    32,
	})
	if err != nil {
		t.Fatalf("checkResults returned unexpected error: %v", err)
	}
	joined := strings.Join(sum.Issues, "\n")
	for _, want := range []string{"required context 100000", "max context 4096", "max max_num_seqs 1"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("expected %q in issues, got %#v", want, sum.Issues)
		}
	}
}

func TestCheckResultsRejectsDuplicateCandidates(t *testing.T) {
	row := `{"candidate_id":"ctx4096-seq1-small","status":"startup_only","candidate":{"max_model_len":4096,"max_num_seqs":1},"startup":{},"shutdown":{},"telemetry":{"tools":{"tegrastats_available":false},"tegrastats":{"sample_count":0,"parsed_sample_count":0}},"notes":["load skipped"]}`

	sum, err := checkResults(strings.NewReader(row+"\n"+row+"\n"), config{minRows: 1, requireTegrastats: true})
	if err != nil {
		t.Fatalf("checkResults returned unexpected error: %v", err)
	}
	if !strings.Contains(strings.Join(sum.Issues, "\n"), "duplicate candidate_id") {
		t.Fatalf("expected duplicate issue, got %#v", sum.Issues)
	}
}

func TestCheckResultsAllowsPreflightMemorySkipWithoutStartup(t *testing.T) {
	input := `{"candidate_id":"ctx4096-seq1-small","status":"skipped_preflight_memory","candidate":{"max_model_len":4096,"max_num_seqs":1},"startup":null,"shutdown":null,"telemetry":{"tools":{"tegrastats_available":false},"tegrastats":{"sample_count":0,"parsed_sample_count":0}},"notes":["available memory or swap was below configured floor before startup"]}`

	sum, err := checkResults(strings.NewReader(input+"\n"), config{minRows: 1, requireTegrastats: true})
	if err != nil {
		t.Fatalf("checkResults returned unexpected error: %v", err)
	}
	if len(sum.Issues) != 0 {
		t.Fatalf("expected no issues, got %#v", sum.Issues)
	}
	if sum.StartupOnlyRows != 1 {
		t.Fatalf("expected skipped row to count as startup-only/skipped, got %#v", sum)
	}
}

func TestCheckResultsRejectsNullLoadMetrics(t *testing.T) {
	input := `{"candidate_id":"ctx4096-seq1-small","status":"load_complete","candidate":{"max_model_len":4096,"max_num_seqs":1},"startup":{},"shutdown":{},"telemetry":{"tools":{"tegrastats_available":true},"tegrastats":{"sample_count":3,"parsed_sample_count":3,"ram_used_delta_gib":81.2}},"load_short_decode":{"successes":null,"errors":0,"wall_seconds":2.1,"completion_tokens_per_second":null,"total_tokens_per_second":400.0,"latency_seconds":{"p50":2.1,"p95":null},"memory_monitor":{"samples":2}}}`

	sum, err := checkResults(strings.NewReader(input+"\n"), config{minRows: 1, requireTegrastats: true})
	if err != nil {
		t.Fatalf("checkResults returned unexpected error: %v", err)
	}
	joined := strings.Join(sum.Issues, "\n")
	for _, want := range []string{"successes", "completion_tokens_per_second", "p95"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("expected %q in issues, got %#v", want, sum.Issues)
		}
	}
}
