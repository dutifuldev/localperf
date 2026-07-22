package vllmbench

import (
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/osolmaz/localperf/internal/artifact"
)

// TestModelLevelArtifactAppendsRuns checks the model-level accumulation
// contract from docs/2026-07-02-default-inference-sweep.md: repeated runs
// append to one artifact, and re-running the same run directory replaces
// that run instead of duplicating it.
func TestModelLevelArtifactAppendsRuns(t *testing.T) {
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "models", "example.sqlite")
	runDry := func(name string) {
		spec := testSpec()
		spec.OutputDir = dir
		if _, err := Execute(context.Background(), spec, RunOptions{
			DryRun:       true,
			RunDir:       filepath.Join(dir, name),
			ArtifactPath: artifactPath,
		}); err != nil {
			t.Fatal(err)
		}
	}
	runDry("batch-1")
	runDry("batch-2")
	if err := artifact.Check(artifactPath); err != nil {
		t.Fatalf("multi-run artifact check failed: %v", err)
	}
	assertRunCount(t, artifactPath, 2)

	// Re-running a batch replaces it.
	runDry("batch-2")
	if err := artifact.Check(artifactPath); err != nil {
		t.Fatalf("artifact check after replace failed: %v", err)
	}
	assertRunCount(t, artifactPath, 2)
}

func TestModelLevelArtifactInitializesEmptyExistingPath(t *testing.T) {
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "model.sqlite")
	if err := os.WriteFile(artifactPath, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	spec := testSpec()
	spec.OutputDir = dir
	if _, err := Execute(context.Background(), spec, RunOptions{
		DryRun:       true,
		RunDir:       filepath.Join(dir, "run"),
		ArtifactPath: artifactPath,
	}); err != nil {
		t.Fatal(err)
	}
	if err := artifact.Check(artifactPath); err != nil {
		t.Fatalf("empty-path artifact check failed: %v", err)
	}
	assertRunCount(t, artifactPath, 1)
}

func TestArtifactMergeCombinesRunsAndSkipsDuplicates(t *testing.T) {
	dir := t.TempDir()
	makeSingle := func(name string) string {
		spec := testSpec()
		spec.OutputDir = dir
		path := filepath.Join(dir, name+".sqlite")
		if _, err := Execute(context.Background(), spec, RunOptions{
			DryRun:       true,
			RunDir:       filepath.Join(dir, name),
			ArtifactPath: path,
		}); err != nil {
			t.Fatal(err)
		}
		return path
	}
	first := makeSingle("run-a")
	second := makeSingle("run-b")
	dst := filepath.Join(dir, "models", "example.sqlite")

	summary, err := artifact.Merge(dst, []string{first, second})
	if err != nil {
		t.Fatal(err)
	}
	if len(summary.MergedRuns) != 2 || len(summary.SkippedRuns) != 0 {
		t.Fatalf("merge summary = %+v, want 2 merged", summary)
	}
	assertRunCount(t, dst, 2)

	// Merging the same source again is a safe no-op.
	summary, err = artifact.Merge(dst, []string{first})
	if err != nil {
		t.Fatal(err)
	}
	if len(summary.MergedRuns) != 0 || len(summary.SkippedRuns) != 1 {
		t.Fatalf("re-merge summary = %+v, want 1 skipped", summary)
	}
	assertRunCount(t, dst, 2)

	db, err := sql.Open("sqlite", dst)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var measurements, profiles int
	if err := db.QueryRow(`SELECT COUNT(*) FROM measurements`).Scan(&measurements); err != nil {
		t.Fatal(err)
	}
	if err := db.QueryRow(`SELECT COUNT(DISTINCT id) FROM profiles`).Scan(&profiles); err != nil {
		t.Fatal(err)
	}
	if measurements != 4 || profiles != 2 {
		t.Fatalf("merged measurements/profiles = %d/%d, want 4 measurements and 2 namespaced profiles", measurements, profiles)
	}
}

func TestModelLevelArtifactRejectsBasenameCollisions(t *testing.T) {
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "model.sqlite")
	runDry := func(parent string) error {
		spec := testSpec()
		spec.OutputDir = dir
		_, err := Execute(context.Background(), spec, RunOptions{
			DryRun:       true,
			RunDir:       filepath.Join(dir, parent, "run"),
			ArtifactPath: artifactPath,
		})
		return err
	}
	if err := runDry("a"); err != nil {
		t.Fatal(err)
	}
	// Same basename "run" from a different parent directory is a collision,
	// not a retry; replacing it would silently destroy unrelated results.
	err := runDry("b")
	if err == nil || !strings.Contains(err.Error(), "different or unknown run directory") {
		t.Fatalf("Execute error = %v, want run-id collision refusal", err)
	}
	assertRunCount(t, artifactPath, 1)
}

func TestModelLevelArtifactRejectsRunsWithoutProvenance(t *testing.T) {
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "model.sqlite")
	runDry := func() error {
		spec := testSpec()
		spec.OutputDir = dir
		_, err := Execute(context.Background(), spec, RunOptions{
			DryRun:       true,
			RunDir:       filepath.Join(dir, "run"),
			ArtifactPath: artifactPath,
		})
		return err
	}
	if err := runDry(); err != nil {
		t.Fatal(err)
	}
	// Cutover: a same-id run without a recorded run directory has unknown
	// provenance and is refused, not silently replaced.
	db, err := sql.Open("sqlite", artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`UPDATE run SET labels_json = '{}'`); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	err = runDry()
	if err == nil || !strings.Contains(err.Error(), "unknown run directory") {
		t.Fatalf("Execute error = %v, want unknown-provenance refusal", err)
	}
}

func TestMergeRejectsRunIDCollisions(t *testing.T) {
	dir := t.TempDir()
	makeSingle := func(parent string) string {
		spec := testSpec()
		spec.OutputDir = dir
		path := filepath.Join(dir, parent+".sqlite")
		if _, err := Execute(context.Background(), spec, RunOptions{
			DryRun:       true,
			RunDir:       filepath.Join(dir, parent, "run"),
			ArtifactPath: path,
		}); err != nil {
			t.Fatal(err)
		}
		return path
	}
	first := makeSingle("a")
	second := makeSingle("b")
	dst := filepath.Join(dir, "model.sqlite")
	// Both sources carry run id "run" from different directories: merging
	// must refuse rather than silently drop the second run as a duplicate.
	_, err := artifact.Merge(dst, []string{first, second})
	if err == nil || !strings.Contains(err.Error(), "different provenance") {
		t.Fatalf("merge error = %v, want provenance collision refusal", err)
	}
}

func TestMergeDoesNotLeaveEmptyDestinationOnBadSource(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "model.sqlite")
	if _, err := artifact.Merge(dst, []string{filepath.Join(dir, "missing.sqlite")}); err == nil {
		t.Fatal("merge with missing source succeeded")
	}
	if _, err := os.Stat(dst); err == nil {
		t.Fatal("failed merge left a destination artifact behind")
	}
}

func assertRunCount(t *testing.T, path string, want int) {
	t.Helper()
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var runs int
	if err := db.QueryRow(`SELECT COUNT(*) FROM run`).Scan(&runs); err != nil {
		t.Fatal(err)
	}
	if runs != want {
		t.Fatalf("run rows = %d, want %d", runs, want)
	}
}

func TestWriteSQLiteArtifactNoopWithoutPath(t *testing.T) {
	if err := WriteSQLiteArtifact(t.TempDir(), "", testSpec(), RunSummary{}, nil, ""); err != nil {
		t.Fatalf("WriteSQLiteArtifact without path = %v, want nil", err)
	}
}

func TestWriteSQLiteArtifactRemovesFreshFileOnFailedWrite(t *testing.T) {
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "model.sqlite")
	spec := testSpec()
	// Duplicate workload names violate the artifact's unique constraint,
	// failing the first write; the fresh file must not be left behind.
	spec.Workloads = append(spec.Workloads, spec.Workloads[0])
	err := WriteSQLiteArtifact(filepath.Join(dir, "run"), artifactPath, spec, RunSummary{StartedAt: time.Now()}, nil, "")
	if err == nil {
		t.Fatal("write with duplicate workloads succeeded")
	}
	if _, statErr := os.Stat(artifactPath); statErr == nil {
		t.Fatal("failed first write left a schema-only artifact behind")
	}
}

func TestMergeCleansUpEmptyDestinationOnCollision(t *testing.T) {
	dir := t.TempDir()
	makeSingle := func(parent string) string {
		spec := testSpec()
		spec.OutputDir = dir
		path := filepath.Join(dir, parent+".sqlite")
		if _, err := Execute(context.Background(), spec, RunOptions{
			DryRun:       true,
			RunDir:       filepath.Join(dir, parent, "run"),
			ArtifactPath: path,
		}); err != nil {
			t.Fatal(err)
		}
		return path
	}
	first := makeSingle("c")
	second := makeSingle("d")
	dst := filepath.Join(dir, "empty-model.sqlite")
	// A pre-existing zero-byte destination is initialized by the merge; a
	// later collision must remove it like a freshly created file, not
	// leave a partially merged artifact behind.
	if err := os.WriteFile(dst, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.Merge(dst, []string{first, second}); err == nil {
		t.Fatal("expected provenance collision failure")
	}
	if _, err := os.Stat(dst); !os.IsNotExist(err) {
		t.Fatalf("destination stat = %v, want removed after failed merge into an empty file", err)
	}
}
