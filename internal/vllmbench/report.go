package vllmbench

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type Report struct {
	RunDir    string      `json:"run_dir"`
	Generated time.Time   `json:"generated"`
	Rows      []ReportRow `json:"rows"`
	Events    EventCounts `json:"events"`
}

type EventCounts struct {
	Total          int            `json:"total"`
	ByType         map[string]int `json:"by_type"`
	FailedWorkload int            `json:"failed_workload"`
}

type ReportRow struct {
	Profile             string  `json:"profile,omitempty"`
	Workload            string  `json:"workload,omitempty"`
	DatasetName         string  `json:"dataset_name,omitempty"`
	Context             int     `json:"context,omitempty"`
	ServerMaxNumSeqs    int     `json:"server_max_num_seqs,omitempty"`
	Concurrency         int     `json:"concurrency,omitempty"`
	RandomInputLen      int     `json:"random_input_len,omitempty"`
	RandomOutputLen     int     `json:"random_output_len,omitempty"`
	Completed           int     `json:"completed,omitempty"`
	Failed              int     `json:"failed,omitempty"`
	DurationSeconds     float64 `json:"duration_seconds,omitempty"`
	OutputTokensPerSec  float64 `json:"output_tokens_per_second,omitempty"`
	TotalTokensPerSec   float64 `json:"total_tokens_per_second,omitempty"`
	PerUserOutputTokSec float64 `json:"per_user_output_tokens_per_second,omitempty"`
	MeanTTFTMillis      float64 `json:"mean_ttft_ms,omitempty"`
	P99TTFTMillis       float64 `json:"p99_ttft_ms,omitempty"`
	MeanTPOTMillis      float64 `json:"mean_tpot_ms,omitempty"`
	ResultFile          string  `json:"result_file,omitempty"`
}

func BuildReport(runDir string) (Report, error) {
	report := Report{
		RunDir:    runDir,
		Generated: time.Now().UTC(),
		Events: EventCounts{
			ByType: map[string]int{},
		},
	}
	spec, _ := loadNormalizedSpec(filepath.Join(runDir, "spec.normalized.json"))
	eventRows, err := readEvents(filepath.Join(runDir, "events.jsonl"))
	if err == nil {
		report.Events.Total = len(eventRows)
		resultEvents := map[string]Event{}
		for _, event := range eventRows {
			report.Events.ByType[event.Type]++
			if event.Type == "workload_failed" {
				report.Events.FailedWorkload++
			}
			if event.Type == "workload_finish" && event.ResultFile != "" && event.Error == "" {
				resultEvents[event.ResultFile] = event
			}
		}
		for resultFile, event := range resultEvents {
			resultPath := resolveResultPath(runDir, resultFile)
			rows, err := ParseResultFile(resultPath)
			if err != nil {
				continue
			}
			event.ResultFile = resultPath
			for _, row := range rows {
				enrichRowFromEvent(&row, event, spec)
				report.Rows = append(report.Rows, row)
			}
		}
	} else {
		rows, err := parseResultDirectory(filepath.Join(runDir, "results"))
		if err != nil {
			return report, err
		}
		report.Rows = rows
	}
	sortReportRows(report.Rows)
	return report, nil
}

func ParseResultFile(path string) ([]ReportRow, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	data, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}
	data = []byte(strings.TrimSpace(string(data)))
	if len(data) == 0 {
		return nil, nil
	}
	if data[0] == '[' {
		var raw []map[string]any
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, err
		}
		return rowsFromRaw(raw, path), nil
	}
	if data[0] == '{' {
		var raw map[string]any
		if err := json.Unmarshal(data, &raw); err == nil {
			return rowsFromRaw([]map[string]any{raw}, path), nil
		}
	}
	scanner := bufio.NewScanner(strings.NewReader(string(data)))
	scanner.Buffer(make([]byte, 1024*1024), 64*1024*1024)
	var raw []map[string]any
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var row map[string]any
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, err
		}
		raw = append(raw, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return rowsFromRaw(raw, path), nil
}

func loadNormalizedSpec(path string) (*Spec, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var spec Spec
	if err := json.Unmarshal(data, &spec); err != nil {
		return nil, err
	}
	ApplyDefaults(&spec)
	return &spec, nil
}

func enrichRowFromEvent(row *ReportRow, event Event, spec *Spec) {
	row.Profile = event.Profile
	row.Workload = event.Workload
	row.Concurrency = event.Concurrency
	row.ResultFile = event.ResultFile
	if spec == nil {
		return
	}
	for _, profile := range spec.Profiles {
		if profile.Name != event.Profile {
			continue
		}
		if row.Context == 0 {
			row.Context = profile.MaxModelLen
		}
		if row.ServerMaxNumSeqs == 0 {
			row.ServerMaxNumSeqs = profile.MaxNumSeqs
		}
		break
	}
	for _, workload := range spec.Workloads {
		if workload.Name != event.Workload {
			continue
		}
		if row.DatasetName == "" {
			row.DatasetName = workload.DatasetName
		}
		if row.RandomInputLen == 0 {
			row.RandomInputLen = workload.RandomInputLen
		}
		if row.RandomOutputLen == 0 {
			row.RandomOutputLen = workload.RandomOutputLen
		}
		break
	}
}

func RenderMarkdown(report Report) string {
	var out strings.Builder
	out.WriteString("# vLLM Benchmark Report\n\n")
	out.WriteString(fmt.Sprintf("Run directory: `%s`\n\n", report.RunDir))
	out.WriteString(fmt.Sprintf("Generated: `%s`\n\n", report.Generated.Format(time.RFC3339)))
	if len(report.Events.ByType) > 0 {
		out.WriteString("## Event Summary\n\n")
		out.WriteString("| Event | Count |\n")
		out.WriteString("| --- | ---: |\n")
		for _, key := range sortedEventKeys(report.Events.ByType) {
			out.WriteString(fmt.Sprintf("| `%s` | %d |\n", key, report.Events.ByType[key]))
		}
		out.WriteString("\n")
	}
	out.WriteString("## Throughput\n\n")
	if len(report.Rows) == 0 {
		out.WriteString("No parseable benchmark result rows were found.\n")
		return out.String()
	}
	out.WriteString("| Profile | Workload | Dataset | Context | Concurrency | Input | Output | Completed | Failed | Output tok/s | Per-user tok/s | Total tok/s | TTFT mean ms | Result |\n")
	out.WriteString("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
	for _, row := range report.Rows {
		out.WriteString(fmt.Sprintf(
			"| %s | %s | %s | %s | %s | %s | %s | %d | %d | %s | %s | %s | %s | `%s` |\n",
			cell(row.Profile),
			cell(row.Workload),
			cell(row.DatasetName),
			intCell(row.Context),
			intCell(row.Concurrency),
			intCell(row.RandomInputLen),
			intCell(row.RandomOutputLen),
			row.Completed,
			row.Failed,
			floatCell(row.OutputTokensPerSec),
			floatCell(row.PerUserOutputTokSec),
			floatCell(row.TotalTokensPerSec),
			floatCell(row.MeanTTFTMillis),
			fileCell(report.RunDir, row.ResultFile),
		))
	}
	return out.String()
}

func WriteReportFiles(report Report, outputPath string) error {
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(outputPath, []byte(RenderMarkdown(report)), 0o644); err != nil {
		return err
	}
	jsonPath := strings.TrimSuffix(outputPath, filepath.Ext(outputPath)) + ".json"
	return writeJSONFile(jsonPath, report)
}

func rowsFromRaw(rawRows []map[string]any, path string) []ReportRow {
	rows := make([]ReportRow, 0, len(rawRows))
	for _, raw := range rawRows {
		row := ReportRow{
			Profile:            stringValue(raw, "profile"),
			Workload:           stringValue(raw, "workload"),
			DatasetName:        stringValue(raw, "dataset_name"),
			Context:            intValue(raw, "max_model_len"),
			ServerMaxNumSeqs:   intValue(raw, "server_max_num_seqs"),
			Concurrency:        intValue(raw, "max_concurrency"),
			RandomInputLen:     intValue(raw, "random_input_len"),
			RandomOutputLen:    intValue(raw, "random_output_len"),
			Completed:          firstInt(raw, "completed", "ok", "successes"),
			Failed:             firstInt(raw, "failed", "errors"),
			DurationSeconds:    firstFloat(raw, "duration", "wall_seconds"),
			OutputTokensPerSec: firstFloat(raw, "output_throughput", "aggregate_completion_tokens_per_second", "completion_tokens_per_second", "diffusion_committed_throughput"),
			TotalTokensPerSec:  firstFloat(raw, "total_token_throughput", "aggregate_total_tokens_per_second", "total_tokens_per_second"),
			MeanTTFTMillis:     firstFloat(raw, "mean_ttft_ms"),
			P99TTFTMillis:      firstFloat(raw, "p99_ttft_ms"),
			MeanTPOTMillis:     firstFloat(raw, "mean_tpot_ms"),
			ResultFile:         path,
		}
		if row.Context == 0 {
			row.Context = intValue(raw, "context")
		}
		if row.Concurrency == 0 {
			row.Concurrency = intValue(raw, "concurrency")
		}
		if row.RandomInputLen == 0 {
			row.RandomInputLen = intValue(raw, "prompt_len")
		}
		if row.RandomOutputLen == 0 {
			row.RandomOutputLen = intValue(raw, "max_tokens")
		}
		if row.DatasetName == "" {
			row.DatasetName = stringValue(raw, "dataset")
		}
		if row.Concurrency > 0 && row.OutputTokensPerSec > 0 {
			row.PerUserOutputTokSec = row.OutputTokensPerSec / float64(row.Concurrency)
		}
		rows = append(rows, row)
	}
	return rows
}

func parseResultDirectory(dir string) ([]ReportRow, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var rows []ReportRow
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasSuffix(name, ".json") && !strings.HasSuffix(name, ".jsonl") {
			continue
		}
		parsed, err := ParseResultFile(filepath.Join(dir, name))
		if err != nil {
			continue
		}
		rows = append(rows, parsed...)
	}
	sortReportRows(rows)
	return rows, nil
}

func readEvents(path string) ([]Event, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 64*1024*1024)
	var events []Event
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var event Event
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			return nil, err
		}
		events = append(events, event)
	}
	return events, scanner.Err()
}

func sortReportRows(rows []ReportRow) {
	sort.Slice(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		if a.Profile != b.Profile {
			return a.Profile < b.Profile
		}
		if a.Context != b.Context {
			return a.Context < b.Context
		}
		if a.Workload != b.Workload {
			return a.Workload < b.Workload
		}
		if a.Concurrency != b.Concurrency {
			return a.Concurrency < b.Concurrency
		}
		return a.ResultFile < b.ResultFile
	})
}

func sortedEventKeys(values map[string]int) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func stringValue(row map[string]any, key string) string {
	value, ok := row[key]
	if !ok || value == nil {
		return ""
	}
	switch typed := value.(type) {
	case string:
		return typed
	default:
		return fmt.Sprint(typed)
	}
}

func firstInt(row map[string]any, keys ...string) int {
	for _, key := range keys {
		if value := intValue(row, key); value != 0 {
			return value
		}
	}
	return 0
}

func intValue(row map[string]any, key string) int {
	value, ok := row[key]
	if !ok || value == nil {
		return 0
	}
	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	case json.Number:
		number, _ := typed.Int64()
		return int(number)
	default:
		return 0
	}
}

func firstFloat(row map[string]any, keys ...string) float64 {
	for _, key := range keys {
		if value := floatValue(row, key); value != 0 {
			return value
		}
	}
	return 0
}

func floatValue(row map[string]any, key string) float64 {
	value, ok := row[key]
	if !ok || value == nil {
		return 0
	}
	switch typed := value.(type) {
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	case float64:
		return typed
	case json.Number:
		number, _ := typed.Float64()
		return number
	default:
		return 0
	}
}

func cell(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "-"
	}
	return strings.ReplaceAll(value, "|", "\\|")
}

func intCell(value int) string {
	if value == 0 {
		return "-"
	}
	return fmt.Sprint(value)
}

func floatCell(value float64) string {
	if value == 0 || math.IsNaN(value) || math.IsInf(value, 0) {
		return "-"
	}
	return fmt.Sprintf("%.1f", value)
}

func fileCell(runDir, path string) string {
	if path == "" {
		return ""
	}
	if rel, err := filepath.Rel(runDir, path); err == nil && !strings.HasPrefix(rel, "..") {
		return rel
	}
	return path
}

func resolveResultPath(runDir, path string) string {
	if path == "" || filepath.IsAbs(path) {
		return path
	}
	if _, err := os.Stat(path); err == nil {
		return path
	}
	return filepath.Join(runDir, path)
}
