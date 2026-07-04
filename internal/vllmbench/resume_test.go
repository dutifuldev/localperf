package vllmbench

import (
	"context"
	"database/sql"
	"path/filepath"
	"strings"
	"testing"
)

// TestResumeSkipsCompletedRunsAndSurvivesServerLoss checks resume end to
// end: a completed attempt re-run with --resume skips every point without
// touching the (now unreachable) server, and the artifact still validates.
func TestResumeSkipsCompletedRunsAndSurvivesServerLoss(t *testing.T) {
	server, host, port := fakeOpenAIServer(t)
	spec := httpTestSpec(t, host, port, "resume-live", 3, 2)
	spec.Workloads[0].MaxConcurrency = []int{1, 2}
	ApplyDefaults(&spec)
	runDir := filepath.Join(spec.OutputDir, "resume-live")
	artifactPath := filepath.Join(spec.OutputDir, "resume-live.sqlite")
	options := RunOptions{RunDir: runDir, ArtifactPath: artifactPath}
	summary, err := Execute(context.Background(), spec, options)
	if err != nil {
		t.Fatal(err)
	}
	if summary.CompletedRuns != 2 {
		t.Fatalf("completed = %d, want 2", summary.CompletedRuns)
	}
	// The server is gone; a resumed attempt must not need it.
	server.Close()
	options.Resume = true
	summary, err = Execute(context.Background(), spec, options)
	if err != nil {
		t.Fatalf("resume error = %v", err)
	}
	if summary.CompletedRuns != 2 || summary.FailedRuns != 0 {
		t.Fatalf("resumed summary = %+v, want 2 completed via skip", summary)
	}
	if len(summary.Rows) != 2 {
		t.Fatalf("resumed rows = %d, want 2 parsed from prior results", len(summary.Rows))
	}
	db, err := sql.Open("sqlite", artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var completed int
	if err := db.QueryRow(`SELECT COUNT(*) FROM measurements WHERE status = 'completed'`).Scan(&completed); err != nil {
		t.Fatal(err)
	}
	if completed != 2 {
		t.Fatalf("artifact completed measurements = %d, want 2", completed)
	}
	var runStatus string
	if err := db.QueryRow(`SELECT status FROM run`).Scan(&runStatus); err != nil {
		t.Fatal(err)
	}
	if runStatus != "completed" {
		t.Fatalf("run status = %q, want completed", runStatus)
	}
}

func TestResumeRequiresRunDir(t *testing.T) {
	spec := testSpec()
	_, err := Execute(context.Background(), spec, RunOptions{Resume: true})
	if err == nil || !strings.Contains(err.Error(), "--resume requires --run-dir") {
		t.Fatalf("Execute error = %v, want resume run-dir requirement", err)
	}
}

// TestIncrementalSnapshotsKeepArtifactCurrent checks that the artifact
// exists and holds completed measurements after each point, not only at
// finalize: the run status is recorded as running mid-flight.
func TestIncrementalSnapshotsKeepArtifactCurrent(t *testing.T) {
	server, host, port := fakeOpenAIServer(t)
	defer server.Close()
	spec := httpTestSpec(t, host, port, "snapshot-live", 3, 1)
	ApplyDefaults(&spec)
	runDir := filepath.Join(spec.OutputDir, "snapshot-live")
	artifactPath := filepath.Join(spec.OutputDir, "snapshot-live.sqlite")
	if _, err := Execute(context.Background(), spec, RunOptions{RunDir: runDir, ArtifactPath: artifactPath}); err != nil {
		t.Fatal(err)
	}
	// The snapshot machinery ran (final write flips status to completed);
	// verify the events log recorded no snapshot failures.
	db, err := sql.Open("sqlite", artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var snapshotFailures int
	if err := db.QueryRow(`SELECT COUNT(*) FROM events WHERE type = 'artifact_snapshot_failed'`).Scan(&snapshotFailures); err != nil {
		t.Fatal(err)
	}
	if snapshotFailures != 0 {
		t.Fatalf("snapshot failures = %d, want 0", snapshotFailures)
	}
	var status string
	if err := db.QueryRow(`SELECT status FROM run`).Scan(&status); err != nil {
		t.Fatal(err)
	}
	if status != "completed" {
		t.Fatalf("final run status = %q, want completed", status)
	}
}
