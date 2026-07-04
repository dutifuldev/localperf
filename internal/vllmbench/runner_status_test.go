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
