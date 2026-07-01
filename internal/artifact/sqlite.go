package artifact

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

const (
	FormatName    = "localperf_run"
	FormatVersion = "1"
)

func Path(runDir, override string) string {
	if strings.TrimSpace(override) != "" {
		return override
	}
	clean := strings.TrimRight(filepath.Clean(runDir), string(filepath.Separator))
	if clean == "." || clean == "" {
		return "localperf-run.sqlite"
	}
	return clean + ".sqlite"
}

func Create(path, schema string) (*sql.DB, error) {
	if err := preparePath(path); err != nil {
		return nil, err
	}
	return openWithSchema(path, schema)
}

func preparePath(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	_ = os.Remove(path)
	return nil
}

func WithTx(db *sql.DB, run func(*sql.Tx) error) error {
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if err := run(tx); err != nil {
		return err
	}
	return tx.Commit()
}

func openWithSchema(path, schema string) (*sql.DB, error) {
	db, err := open(path, "")
	if err != nil {
		return nil, err
	}
	if _, err := db.Exec(schema); err != nil {
		_ = db.Close()
		return nil, err
	}
	return db, nil
}

func OpenExisting(path, rawQuery string) (*sql.DB, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}
	return open(path, rawQuery)
}

func OpenReadOnly(path string) (*sql.DB, error) {
	return OpenExisting(path, "mode=ro")
}

func OpenWritable(path string) (*sql.DB, error) {
	return OpenExisting(path, "")
}

func open(path, rawQuery string) (*sql.DB, error) {
	dsn, err := fileDSNWithQuery(path, rawQuery)
	if err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, err
	}
	if _, err := db.Exec("PRAGMA foreign_keys = ON"); err != nil {
		_ = db.Close()
		return nil, err
	}
	return db, nil
}

func fileDSNWithQuery(path, rawQuery string) (string, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	uri := url.URL{Scheme: "file", Path: absolute, RawQuery: rawQuery}
	return uri.String(), nil
}

func Check(path string) error {
	db, err := OpenWritable(path)
	if err != nil {
		return err
	}
	defer db.Close()
	for _, check := range []func(*sql.DB) error{
		checkIntegrity,
		checkMetadata,
		checkRequiredTables,
		checkSpecHashes,
		checkRunRowCount,
		checkSpecKindRows,
		checkForeignKeys,
		checkArtifactHashes,
	} {
		if err := check(db); err != nil {
			return err
		}
	}
	return nil
}

func checkIntegrity(db *sql.DB) error {
	var integrity string
	if err := db.QueryRow("PRAGMA integrity_check").Scan(&integrity); err != nil {
		return err
	}
	if integrity != "ok" {
		return fmt.Errorf("sqlite integrity_check = %s", integrity)
	}
	return nil
}

func checkRunRowCount(db *sql.DB) error {
	var runRows int
	if err := db.QueryRow("SELECT COUNT(*) FROM run").Scan(&runRows); err != nil {
		return err
	}
	if runRows != 1 {
		return fmt.Errorf("run rows = %d, want 1", runRows)
	}
	return nil
}

func checkSpecKindRows(db *sql.DB) error {
	for _, kind := range []string{"original", "normalized"} {
		var count int
		if err := db.QueryRow("SELECT COUNT(*) FROM specs WHERE kind = ?", kind).Scan(&count); err != nil {
			return err
		}
		if count != 1 {
			return fmt.Errorf("spec kind %s rows = %d, want 1", kind, count)
		}
	}
	return nil
}

func checkForeignKeys(db *sql.DB) error {
	rows, err := db.Query("PRAGMA foreign_key_check")
	if err != nil {
		return err
	}
	defer rows.Close()
	if rows.Next() {
		return errors.New("foreign key check reported at least one failure")
	}
	return rows.Err()
}

func checkMetadata(db *sql.DB) error {
	values := map[string]string{}
	rows, err := db.Query("SELECT key, value FROM metadata")
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var key, value string
		if err := rows.Scan(&key, &value); err != nil {
			return err
		}
		values[key] = value
	}
	if values["format_name"] != FormatName {
		return fmt.Errorf("format_name = %q, want %q", values["format_name"], FormatName)
	}
	if values["format_version"] != FormatVersion {
		return fmt.Errorf("format_version = %q, want %q", values["format_version"], FormatVersion)
	}
	return nil
}

func checkRequiredTables(db *sql.DB) error {
	required := []string{"metadata", "run", "specs", "engines", "profiles", "workloads", "phases", "measurements", "metric_stats", "requests", "request_stream_events", "telemetry_series", "telemetry_samples", "events", "commands", "artifacts", "reports"}
	present := map[string]bool{}
	rows, err := db.Query("SELECT name FROM sqlite_master WHERE type = 'table'")
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			return err
		}
		present[name] = true
	}
	for _, table := range required {
		if !present[table] {
			return fmt.Errorf("missing required table %s", table)
		}
	}
	return nil
}

func checkSpecHashes(db *sql.DB) error {
	rows, err := db.Query("SELECT kind, content, sha256 FROM specs")
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var kind, content, want string
		if err := rows.Scan(&kind, &content, &want); err != nil {
			return err
		}
		if got := SHA256Hex([]byte(content)); got != want {
			return fmt.Errorf("spec %s sha256 = %s, want %s", kind, got, want)
		}
	}
	return rows.Err()
}

func checkArtifactHashes(db *sql.DB) error {
	rows, err := db.Query("SELECT name, compression, content, sha256 FROM artifacts")
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var name, compression, want string
		var content []byte
		if err := rows.Scan(&name, &compression, &content, &want); err != nil {
			return err
		}
		data, err := hashContent(name, compression, content)
		if err != nil {
			return err
		}
		if got := SHA256Hex(data); got != want {
			return fmt.Errorf("artifact %s sha256 = %s, want %s", name, got, want)
		}
	}
	return rows.Err()
}

func hashContent(name, compression string, content []byte) ([]byte, error) {
	if compression != "gzip" {
		return content, nil
	}
	reader, err := gzip.NewReader(bytes.NewReader(content))
	if err != nil {
		return nil, fmt.Errorf("artifact %s gzip decode: %w", name, err)
	}
	data, err := io.ReadAll(reader)
	_ = reader.Close()
	if err != nil {
		return nil, fmt.Errorf("artifact %s gzip read: %w", name, err)
	}
	return data, nil
}

func Content(data []byte, mediaType string) ([]byte, string, error) {
	if !shouldCompress(data, mediaType) {
		return data, "none", nil
	}
	content, err := gzipBytes(data)
	return content, "gzip", err
}

func shouldCompress(data []byte, mediaType string) bool {
	return len(data) > 64*1024 && (strings.HasPrefix(mediaType, "text/") || strings.Contains(mediaType, "json"))
}

func gzipBytes(data []byte) ([]byte, error) {
	var compressed bytes.Buffer
	gzipWriter := gzip.NewWriter(&compressed)
	if _, err := gzipWriter.Write(data); err != nil {
		return nil, err
	}
	if err := gzipWriter.Close(); err != nil {
		return nil, err
	}
	return compressed.Bytes(), nil
}

func SHA256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func NullString(value string) any {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	return value
}

func StoreReport(path, name, mediaType, originalPath string, content []byte) error {
	db, err := OpenWritable(path)
	if err != nil {
		return err
	}
	defer db.Close()
	return WithTx(db, func(tx *sql.Tx) error {
		runID, err := singleRunID(tx)
		if err != nil {
			return err
		}
		artifactID, err := upsertReportArtifact(tx, runID, name, mediaType, originalPath, content, time.Now().UTC())
		if err != nil {
			return err
		}
		return upsertReportRow(tx, runID, name, mediaType, artifactID, time.Now().UTC())
	})
}

func singleRunID(tx *sql.Tx) (string, error) {
	var runID string
	err := tx.QueryRow("SELECT id FROM run LIMIT 1").Scan(&runID)
	return runID, err
}

func upsertReportArtifact(tx *sql.Tx, runID, name, mediaType, originalPath string, data []byte, createdAt time.Time) (int64, error) {
	content, compression, err := Content(data, mediaType)
	if err != nil {
		return 0, err
	}
	_, err = tx.Exec(`INSERT INTO artifacts (
		run_id, kind, name, media_type, compression, content, content_size_bytes,
		uncompressed_size_bytes, sha256, original_path, created_at
	) VALUES (?, 'normalized_report', ?, ?, ?, ?, ?, ?, ?, ?, ?)
	ON CONFLICT(run_id, kind, name) DO UPDATE SET
		media_type = excluded.media_type,
		compression = excluded.compression,
		content = excluded.content,
		content_size_bytes = excluded.content_size_bytes,
		uncompressed_size_bytes = excluded.uncompressed_size_bytes,
		sha256 = excluded.sha256,
		original_path = excluded.original_path,
		created_at = excluded.created_at`,
		runID, name, mediaType, compression, content, len(content), len(data), SHA256Hex(data),
		NullString(originalPath), createdAt.Format(time.RFC3339))
	if err != nil {
		return 0, err
	}
	var artifactID int64
	err = tx.QueryRow(`SELECT id FROM artifacts WHERE run_id = ? AND kind = 'normalized_report' AND name = ?`, runID, name).Scan(&artifactID)
	return artifactID, err
}

func upsertReportRow(tx *sql.Tx, runID, name, mediaType string, artifactID int64, createdAt time.Time) error {
	_, err := tx.Exec(`INSERT INTO reports (
		run_id, name, format, media_type, artifact_id, created_at
	) VALUES (?, ?, 'html', ?, ?, ?)
	ON CONFLICT(run_id, name, format) DO UPDATE SET
		media_type = excluded.media_type,
		artifact_id = excluded.artifact_id,
		created_at = excluded.created_at`,
		runID, name, mediaType, artifactID, createdAt.Format(time.RFC3339))
	return err
}
