package vllmbench

import (
	"bytes"
	"compress/gzip"
	"database/sql"
	"strings"
	"testing"
)

func TestCheckArtifactHashesHandlesPlainGzipAndFailures(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`CREATE TABLE artifacts (name TEXT, compression TEXT, content BLOB, sha256 TEXT)`); err != nil {
		t.Fatal(err)
	}
	plain := []byte("plain artifact")
	if _, err := db.Exec(`INSERT INTO artifacts VALUES (?, ?, ?, ?)`, "plain", "none", plain, sha256Hex(plain)); err != nil {
		t.Fatal(err)
	}
	var compressed bytes.Buffer
	writer := gzip.NewWriter(&compressed)
	if _, err := writer.Write([]byte("gzipped artifact")); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO artifacts VALUES (?, ?, ?, ?)`, "gzipped", "gzip", compressed.Bytes(), sha256Hex([]byte("gzipped artifact"))); err != nil {
		t.Fatal(err)
	}
	if err := checkArtifactHashes(db); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`UPDATE artifacts SET sha256 = 'bad' WHERE name = 'plain'`); err != nil {
		t.Fatal(err)
	}
	if err := checkArtifactHashes(db); err == nil || !strings.Contains(err.Error(), "plain sha256") {
		t.Fatalf("hash error = %v", err)
	}
	if _, err := db.Exec(`UPDATE artifacts SET sha256 = ? WHERE name = 'plain'`, sha256Hex(plain)); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`UPDATE artifacts SET content = ? WHERE name = 'gzipped'`, []byte("not gzip")); err != nil {
		t.Fatal(err)
	}
	if err := checkArtifactHashes(db); err == nil || !strings.Contains(err.Error(), "gzip decode") {
		t.Fatalf("gzip error = %v", err)
	}
}

func TestSQLiteArtifactPathEdges(t *testing.T) {
	if got := SQLiteArtifactPath("run-dir", "/tmp/custom.sqlite"); got != "/tmp/custom.sqlite" {
		t.Fatalf("override artifact path = %s", got)
	}
	if got := SQLiteArtifactPath(".", ""); got != "localperf-run.sqlite" {
		t.Fatalf("default artifact path = %s", got)
	}
	if got := SQLiteArtifactPath("runs/example/", ""); got != "runs/example.sqlite" {
		t.Fatalf("run artifact path = %s", got)
	}
}

func TestInsertArtifactPathStoresPlainAndCompressedContent(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`CREATE TABLE artifacts (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		run_id TEXT,
		kind TEXT,
		name TEXT,
		media_type TEXT,
		compression TEXT,
		content BLOB,
		content_size_bytes INTEGER,
		uncompressed_size_bytes INTEGER,
		sha256 TEXT,
		original_path TEXT,
		created_at TEXT
	)`); err != nil {
		t.Fatal(err)
	}
	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	plainPath := writeTempArtifact(t, "plain.txt", strings.Repeat("a", 16))
	if id, err := insertArtifactPath(tx, "run", "log", "plain", plainPath, "text/plain"); err != nil || id == 0 {
		t.Fatalf("plain artifact id=%d err=%v", id, err)
	}
	largePath := writeTempArtifact(t, "large.json", strings.Repeat(`{"x":1}`, 10000))
	if id, err := insertArtifactPath(tx, "run", "json", "large", largePath, "application/json"); err != nil || id == 0 {
		t.Fatalf("large artifact id=%d err=%v", id, err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
	rows, err := db.Query(`SELECT name, compression FROM artifacts ORDER BY name`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	got := map[string]string{}
	for rows.Next() {
		var name, compression string
		if err := rows.Scan(&name, &compression); err != nil {
			t.Fatal(err)
		}
		got[name] = compression
	}
	if got["plain"] != "none" || got["large"] != "gzip" {
		t.Fatalf("compressions = %#v", got)
	}
}

func writeTempArtifact(t *testing.T, name, content string) string {
	t.Helper()
	path := t.TempDir() + "/" + name
	writeFile(t, path, content)
	return path
}
