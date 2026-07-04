package vllmbench

import (
	"context"
	"testing"
)

func TestFinalRunErrorAllowsPartialSweepFailures(t *testing.T) {
	err := finalRunError(RunSummary{CompletedRuns: 3, FailedRuns: 1}, nil)
	if err != nil {
		t.Fatalf("finalRunError() = %v, want nil for partial sweep failure", err)
	}
}

func TestFinalRunErrorKeepsFatalRunErrorAfterPartialProgress(t *testing.T) {
	err := finalRunError(RunSummary{CompletedRuns: 3, FailedRuns: 1}, context.Canceled)
	if err == nil {
		t.Fatal("finalRunError() = nil, want fatal run error to stay fatal")
	}
}

func TestFinalRunErrorFailsAllFailedSweep(t *testing.T) {
	err := finalRunError(RunSummary{CompletedRuns: 0, FailedRuns: 2}, nil)
	if err == nil {
		t.Fatal("finalRunError() = nil, want error when every attempted run failed")
	}
}

func TestRunStatusCompletedWhenSweepHasSomeSuccess(t *testing.T) {
	if got := runStatus(RunSummary{CompletedRuns: 1, FailedRuns: 1}); got != "completed" {
		t.Fatalf("runStatus() = %q, want completed for partial sweep failure", got)
	}
}

func TestRunStatusFailedWhenFatalRunErrorHasSomeSuccess(t *testing.T) {
	if got := runStatus(RunSummary{CompletedRuns: 1, FailedRuns: 1, Error: "context canceled"}); got != "failed" {
		t.Fatalf("runStatus() = %q, want failed for partial fatal run", got)
	}
}

func TestRunStatusFailedWhenSweepHasNoSuccess(t *testing.T) {
	if got := runStatus(RunSummary{CompletedRuns: 0, FailedRuns: 1}); got != "failed" {
		t.Fatalf("runStatus() = %q, want failed when every attempted run failed", got)
	}
}

func TestMeasurementStatusLastDecisiveEventWins(t *testing.T) {
	planned := PlannedRun{Profile: Profile{Name: "p"}, Workload: Workload{Name: "w"}, Concurrency: 1}
	event := func(kind, errText string) Event {
		return Event{Type: kind, Profile: "p", Workload: "w", Concurrency: 1, Error: errText}
	}
	cases := []struct {
		name       string
		events     []Event
		wantStatus string
		wantError  bool
	}{
		{"no events", nil, "planned", false},
		{"clean finish", []Event{event("workload_finish", "")}, "completed", false},
		{"failure", []Event{event("workload_finish", "boom"), event("workload_failed", "boom")}, "failed", true},
		{"skip", []Event{event("workload_skipped", "reason")}, "skipped", true},
		{"retry succeeds after failure", []Event{event("workload_failed", "boom"), event("workload_start", ""), event("workload_finish", "")}, "completed", false},
		{"resumed adoption completes", []Event{event("workload_failed", "boom"), event("workload_resumed", "")}, "completed", false},
		{"unrelated events ignored", []Event{{Type: "workload_failed", Profile: "other", Concurrency: 1, Error: "x"}}, "planned", false},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			if got := measurementStatus(testCase.events, planned); got != testCase.wantStatus {
				t.Fatalf("status = %q, want %q", got, testCase.wantStatus)
			}
			if gotError := measurementError(testCase.events, planned) != nil; gotError != testCase.wantError {
				t.Fatalf("error present = %t, want %t", gotError, testCase.wantError)
			}
		})
	}
}
