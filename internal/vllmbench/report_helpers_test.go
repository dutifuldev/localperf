package vllmbench

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/dutifuldev/localperf/internal/collections"
)

func TestParseResultFileVariants(t *testing.T) {
	dir := t.TempDir()
	arrayPath := filepath.Join(dir, "array.json")
	writeFile(t, arrayPath, `[
  {
    "profile": "p1",
    "workload": "w1",
    "dataset": "random",
    "context": 4096,
    "concurrency": 4,
    "random_input_len": 1024,
    "random_output_len": 256,
    "completed": 8,
    "output_throughput": 40,
    "total_token_throughput": 80,
    "mean_ttft_ms": 12.5
  }
]`)
	rows, err := ParseResultFile(arrayPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(rows))
	}
	row := rows[0]
	if row.DatasetName != "random" || row.InputLen != 1024 || row.OutputLen != 256 {
		t.Fatalf("row lengths were not derived: %+v", row)
	}
	if row.PerUserOutputTokSec != 10 {
		t.Fatalf("per-user throughput = %v, want 10", row.PerUserOutputTokSec)
	}

	objectPath := filepath.Join(dir, "object.json")
	writeFile(t, objectPath, `{"ok": 2, "failed": 1, "duration": 3.5}`)
	rows, err = ParseResultFile(objectPath)
	if err != nil {
		t.Fatal(err)
	}
	if rows[0].Completed != 2 || rows[0].Failed != 1 || rows[0].DurationSeconds != 3.5 {
		t.Fatalf("object row not parsed: %+v", rows[0])
	}

	jsonlPath := filepath.Join(dir, "rows.jsonl")
	writeFile(t, jsonlPath, "\n{\"profile\":\"p2\",\"successes\":3}\n{\"profile\":\"p3\",\"errors\":2}\n")
	rows, err = ParseResultFile(jsonlPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || rows[0].Completed != 3 || rows[1].Failed != 2 {
		t.Fatalf("jsonl rows not parsed: %+v", rows)
	}

	emptyPath := filepath.Join(dir, "empty.json")
	writeFile(t, emptyPath, "  \n")
	rows, err = ParseResultFile(emptyPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 0 {
		t.Fatalf("empty rows = %d, want 0", len(rows))
	}

	badPath := filepath.Join(dir, "bad.json")
	writeFile(t, badPath, `{"profile":`)
	if _, err := ParseResultFile(badPath); err == nil {
		t.Fatal("expected malformed result to fail")
	}
}

func TestParseResultDirectorySortsAndSkipsInvalidFiles(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "nested"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(dir, "notes.txt"), "ignored")
	writeFile(t, filepath.Join(dir, "bad.json"), "{")
	writeFile(t, filepath.Join(dir, "b.json"), `{"profile":"b","workload":"w","concurrency":2}`)
	writeFile(t, filepath.Join(dir, "a.jsonl"), `{"profile":"a","workload":"w","concurrency":1}`)

	rows, err := parseResultDirectory(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got := []string{rows[0].Profile, rows[1].Profile}; !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Fatalf("profiles = %#v, want sorted a,b", got)
	}
}

func TestReportTrafficLengthHelpers(t *testing.T) {
	customOutput := 77
	shareGPTOutput := 88
	cases := []struct {
		name      string
		traffic   BenchmarkTrafficConfig
		wantInput int
		wantOut   int
	}{
		{
			name:      "random",
			traffic:   BenchmarkTrafficConfig{DatasetName: "random", RandomInputLen: 100, RandomOutputLen: 20},
			wantInput: 100,
			wantOut:   20,
		},
		{
			name:      "sonnet",
			traffic:   BenchmarkTrafficConfig{DatasetName: "sonnet", SonnetInputLen: 200, SonnetOutputLen: 30},
			wantInput: 200,
			wantOut:   30,
		},
		{
			name:      "prefix repetition",
			traffic:   BenchmarkTrafficConfig{DatasetName: "prefix_repetition", PrefixRepetitionPrefixLen: 300, PrefixRepetitionSuffixLen: 40, PrefixRepetitionOutputLen: 50},
			wantInput: 340,
			wantOut:   50,
		},
		{
			name:    "custom",
			traffic: BenchmarkTrafficConfig{DatasetName: "custom", CustomOutputLen: &customOutput},
			wantOut: customOutput,
		},
		{
			name:    "sharegpt",
			traffic: BenchmarkTrafficConfig{DatasetName: "sharegpt", ShareGPTOutputLen: &shareGPTOutput},
			wantOut: shareGPTOutput,
		},
		{
			name:    "speed bench",
			traffic: BenchmarkTrafficConfig{DatasetName: "speed_bench", SpeedBenchOutputLen: 99},
			wantOut: 99,
		},
		{
			name:    "unknown",
			traffic: BenchmarkTrafficConfig{DatasetName: "unknown"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := trafficInputLen(tc.traffic); got != tc.wantInput {
				t.Fatalf("input len = %d, want %d", got, tc.wantInput)
			}
			if got := trafficOutputLen(tc.traffic); got != tc.wantOut {
				t.Fatalf("output len = %d, want %d", got, tc.wantOut)
			}
		})
	}
}

func TestReportValueAndCellHelpers(t *testing.T) {
	row := map[string]any{
		"string":      "value",
		"string_int":  12,
		"int":         3,
		"int64":       int64(4),
		"float":       5.9,
		"json_int":    json.Number("6"),
		"json_float":  json.Number("7.25"),
		"invalid_int": "x",
	}
	if stringValue(row, "string_int") != "12" || stringValue(row, "missing") != "" {
		t.Fatalf("stringValue returned unexpected values")
	}
	if intValue(row, "int") != 3 || intValue(row, "int64") != 4 || intValue(row, "float") != 5 || intValue(row, "json_int") != 6 {
		t.Fatalf("intValue returned unexpected values")
	}
	if intValue(row, "invalid_int") != 0 || intValue(row, "missing") != 0 {
		t.Fatalf("intValue fallback failed")
	}
	if floatValue(row, "int") != 3 || floatValue(row, "int64") != 4 || floatValue(row, "float") != 5.9 || floatValue(row, "json_float") != 7.25 {
		t.Fatalf("floatValue returned unexpected values")
	}
	if floatValue(row, "invalid_int") != 0 || floatValue(row, "missing") != 0 {
		t.Fatalf("floatValue fallback failed")
	}
	if cell(" a|b ") != "a\\|b" || cell(" ") != "-" {
		t.Fatalf("cell escaping failed")
	}
	if intCell(0) != "-" || intCell(9) != "9" || intCSV(0) != "" || intCSV(9) != "9" {
		t.Fatalf("integer cell helpers failed")
	}
	if floatCell(0) != "-" || floatCell(math.NaN()) != "-" || floatCell(math.Inf(1)) != "-" || floatCell(1.25) != "1.2" {
		t.Fatalf("floatCell returned unexpected values")
	}
	if floatCSV(0) != "" || floatCSV(math.NaN()) != "" || floatCSV(math.Inf(1)) != "" || floatCSV(1.25) != "1.25" {
		t.Fatalf("floatCSV returned unexpected values")
	}
}

func TestReportPathHelpers(t *testing.T) {
	dir := t.TempDir()
	runDir := filepath.Join(dir, "run")
	resultPath := filepath.Join(runDir, "results", "one.json")
	writeFile(t, resultPath, "{}")

	if resolveResultPath(runDir, "") != "" {
		t.Fatal("empty result path should remain empty")
	}
	if resolveResultPath(runDir, resultPath) != resultPath {
		t.Fatal("absolute result path should remain unchanged")
	}
	if got := resolveResultPath(runDir, filepath.Join("results", "one.json")); got != resultPath {
		t.Fatalf("resolved path = %s, want %s", got, resultPath)
	}
	if got := resolveResultPath(runDir, filepath.Join(filepath.Base(runDir), "results", "one.json")); got != resultPath {
		t.Fatalf("stripped path = %s, want %s", got, resultPath)
	}
	if got := fileCell(runDir, resultPath); got != filepath.Join("results", "one.json") {
		t.Fatalf("file cell = %s", got)
	}
	if got := fileCell(runDir, filepath.Join(dir, "outside.json")); got != filepath.Join(dir, "outside.json") {
		t.Fatalf("outside file cell = %s", got)
	}
	if stripped, ok := stripRunDirPrefix(runDir, filepath.Join(filepath.Base(runDir), "results", "one.json")); !ok || stripped != filepath.Join("results", "one.json") {
		t.Fatalf("stripRunDirPrefix = %q, %t", stripped, ok)
	}
}

func TestSortedProfileNames(t *testing.T) {
	got := collections.SortedKeys(map[string]Profile{
		"z": {Name: "z"},
		"a": {Name: "a"},
		"m": {Name: "m"},
	})
	if !reflect.DeepEqual(got, []string{"a", "m", "z"}) {
		t.Fatalf("sorted names = %#v", got)
	}
}
