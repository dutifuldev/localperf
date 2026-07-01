package artifact

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
	if _, err := db.Exec(`INSERT INTO artifacts VALUES (?, ?, ?, ?)`, "plain", "none", plain, SHA256Hex(plain)); err != nil {
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
	if _, err := db.Exec(`INSERT INTO artifacts VALUES (?, ?, ?, ?)`, "gzipped", "gzip", compressed.Bytes(), SHA256Hex([]byte("gzipped artifact"))); err != nil {
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
	if _, err := db.Exec(`UPDATE artifacts SET sha256 = ? WHERE name = 'plain'`, SHA256Hex(plain)); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`UPDATE artifacts SET content = ? WHERE name = 'gzipped'`, []byte("not gzip")); err != nil {
		t.Fatal(err)
	}
	if err := checkArtifactHashes(db); err == nil || !strings.Contains(err.Error(), "gzip decode") {
		t.Fatalf("gzip error = %v", err)
	}
}
