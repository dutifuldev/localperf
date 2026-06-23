package main

import (
	"strings"
	"testing"
)

func TestReadProgressSummarizesRows(t *testing.T) {
	input := strings.Join([]string{
		`{"candidate_id":"ctx4096-seq1-small","status":"load_complete","started_at":"2026-06-23T00:00:00Z","finished_at":"2026-06-23T00:03:00Z","candidate":{"max_model_len":4096,"max_num_seqs":1}}`,
		`{"candidate_id":"ctx8192-seq2-small","status":"startup_only","started_at":"2026-06-23T00:03:00Z","finished_at":"2026-06-23T00:06:00Z","candidate":{"max_model_len":8192,"max_num_seqs":2}}`,
	}, "\n") + "\n"

	report, err := readProgress(strings.NewReader(input), 100)
	if err != nil {
		t.Fatalf("readProgress returned error: %v", err)
	}
	if report.Rows != 2 {
		t.Fatalf("expected 2 rows, got %d", report.Rows)
	}
	if report.Statuses["load_complete"] != 1 || report.Statuses["startup_only"] != 1 {
		t.Fatalf("unexpected statuses: %#v", report.Statuses)
	}
	if !report.Contexts[4096] || !report.Contexts[8192] {
		t.Fatalf("unexpected contexts: %#v", report.Contexts)
	}
	if !report.Seqs[1] || !report.Seqs[2] {
		t.Fatalf("unexpected seqs: %#v", report.Seqs)
	}
	if report.LatestCandidate != "ctx8192-seq2-small" {
		t.Fatalf("unexpected latest candidate: %s", report.LatestCandidate)
	}
}
