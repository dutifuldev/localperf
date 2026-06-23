package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
)

type config struct {
	resultsPath       string
	minRows           int
	requireTegrastats bool
	jsonOutput        bool
	requireContexts   intList
	requireMaxContext int
	requireMaxSeqs    int
}

type summary struct {
	Rows                  int            `json:"rows"`
	MinRows               int            `json:"min_rows"`
	Statuses              map[string]int `json:"statuses"`
	LoadRows              int            `json:"load_rows"`
	StartupOnlyRows       int            `json:"startup_only_rows"`
	RowsWithTegrastats    int            `json:"rows_with_tegrastats"`
	RowsMissingTelemetry  int            `json:"rows_missing_telemetry"`
	RowsMissingLoadFields int            `json:"rows_missing_load_fields"`
	UniqueCandidates      int            `json:"unique_candidates"`
	MaxContext            int            `json:"max_context"`
	MaxSeqs               int            `json:"max_seqs"`
	Contexts              []int          `json:"contexts"`
	Seqs                  []int          `json:"seqs"`
	Issues                []string       `json:"issues"`
}

func main() {
	cfg := parseFlags()
	file, err := os.Open(cfg.resultsPath)
	if err != nil {
		fatal(cfg, summary{MinRows: cfg.minRows, Issues: []string{err.Error()}})
	}
	defer file.Close()

	sum, err := checkResults(file, cfg)
	if err != nil {
		sum.Issues = append(sum.Issues, err.Error())
	}
	if len(sum.Issues) > 0 {
		fatal(cfg, sum)
	}
	printSummary(cfg, sum)
}

func parseFlags() config {
	cfg := config{}
	flag.StringVar(&cfg.resultsPath, "results", "", "path to sweep results JSONL")
	flag.IntVar(&cfg.minRows, "min-rows", 100, "minimum required result rows")
	flag.BoolVar(&cfg.requireTegrastats, "require-tegrastats", true, "require parsed tegrastats samples when available")
	flag.BoolVar(&cfg.jsonOutput, "json", false, "print machine-readable JSON summary")
	flag.Var(&cfg.requireContexts, "require-context", "context window that must appear at least once; may be repeated")
	flag.IntVar(&cfg.requireMaxContext, "require-max-context", 0, "minimum required maximum context window")
	flag.IntVar(&cfg.requireMaxSeqs, "require-max-seqs", 0, "minimum required maximum max_num_seqs")
	flag.Parse()
	if strings.TrimSpace(cfg.resultsPath) == "" {
		fmt.Fprintln(os.Stderr, "missing --results")
		os.Exit(2)
	}
	return cfg
}

func checkResults(reader io.Reader, cfg config) (summary, error) {
	sum := summary{
		MinRows:  cfg.minRows,
		Statuses: map[string]int{},
	}
	state := checkState{
		candidates: map[string]int{},
		contexts:   map[int]bool{},
		seqs:       map[int]bool{},
	}
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 1024*1024), 64*1024*1024)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var row map[string]any
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			sum.Issues = append(sum.Issues, fmt.Sprintf("line %d: invalid JSON: %v", lineNo, err))
			continue
		}
		checkRow(&sum, &state, lineNo, row, cfg)
	}
	if err := scanner.Err(); err != nil {
		return sum, err
	}
	finalizeCoverage(&sum, state, cfg)
	if sum.Rows < cfg.minRows {
		sum.Issues = append(sum.Issues, fmt.Sprintf("only %d rows recorded; need at least %d", sum.Rows, cfg.minRows))
	}
	if sum.Rows == 0 {
		sum.Issues = append(sum.Issues, "no result rows found")
	}
	return sum, nil
}

type checkState struct {
	candidates map[string]int
	contexts   map[int]bool
	seqs       map[int]bool
}

func checkRow(sum *summary, state *checkState, lineNo int, row map[string]any, cfg config) {
	sum.Rows++
	id := stringField(row, "candidate_id")
	if id == "" {
		id = fmt.Sprintf("line %d", lineNo)
		sum.Issues = append(sum.Issues, fmt.Sprintf("%s: missing candidate_id", id))
	} else {
		state.candidates[id]++
		if state.candidates[id] > 1 {
			sum.Issues = append(sum.Issues, fmt.Sprintf("%s: duplicate candidate_id", id))
		}
	}
	status := stringField(row, "status")
	if status == "" {
		sum.Issues = append(sum.Issues, fmt.Sprintf("%s: missing status", id))
	} else {
		sum.Statuses[status]++
	}
	candidate := objectField(row, "candidate")
	if candidate == nil {
		sum.Issues = append(sum.Issues, fmt.Sprintf("%s: missing candidate parameters", id))
	} else {
		context := int(numericField(candidate, "max_model_len"))
		seqs := int(numericField(candidate, "max_num_seqs"))
		if context > 0 {
			state.contexts[context] = true
			if context > sum.MaxContext {
				sum.MaxContext = context
			}
		}
		if seqs > 0 {
			state.seqs[seqs] = true
			if seqs > sum.MaxSeqs {
				sum.MaxSeqs = seqs
			}
		}
	}
	if status != "dry_run" && objectField(row, "startup") == nil {
		sum.Issues = append(sum.Issues, fmt.Sprintf("%s: missing startup record", id))
	}
	if status != "dry_run" && objectField(row, "shutdown") == nil {
		sum.Issues = append(sum.Issues, fmt.Sprintf("%s: missing shutdown record", id))
	}

	if !checkTelemetry(sum, id, row, cfg) {
		sum.RowsMissingTelemetry++
	}

	switch status {
	case "load_complete", "load_errors":
		sum.LoadRows++
		if !checkLoadRecord(id, row, sum) {
			sum.RowsMissingLoadFields++
		}
	case "startup_only", "skipped_load_idle_memory", "skipped_preflight_memory":
		sum.StartupOnlyRows++
		if len(arrayField(row, "notes")) == 0 {
			sum.Issues = append(sum.Issues, fmt.Sprintf("%s: startup-only/skipped row has no note", id))
		}
	}
}

func finalizeCoverage(sum *summary, state checkState, cfg config) {
	sum.UniqueCandidates = len(state.candidates)
	sum.Contexts = sortedInts(state.contexts)
	sum.Seqs = sortedInts(state.seqs)
	for _, context := range cfg.requireContexts {
		if !state.contexts[context] {
			sum.Issues = append(sum.Issues, fmt.Sprintf("required context %d was not recorded", context))
		}
	}
	if cfg.requireMaxContext > 0 && sum.MaxContext < cfg.requireMaxContext {
		sum.Issues = append(sum.Issues, fmt.Sprintf("max context %d is below required %d", sum.MaxContext, cfg.requireMaxContext))
	}
	if cfg.requireMaxSeqs > 0 && sum.MaxSeqs < cfg.requireMaxSeqs {
		sum.Issues = append(sum.Issues, fmt.Sprintf("max max_num_seqs %d is below required %d", sum.MaxSeqs, cfg.requireMaxSeqs))
	}
}

func checkTelemetry(sum *summary, id string, row map[string]any, cfg config) bool {
	telemetry := objectField(row, "telemetry")
	if telemetry == nil {
		sum.Issues = append(sum.Issues, fmt.Sprintf("%s: missing telemetry", id))
		return false
	}
	tools := objectField(telemetry, "tools")
	if tools == nil {
		sum.Issues = append(sum.Issues, fmt.Sprintf("%s: missing telemetry.tools", id))
		return false
	}
	tegrastats := objectField(telemetry, "tegrastats")
	if tegrastats == nil {
		sum.Issues = append(sum.Issues, fmt.Sprintf("%s: missing telemetry.tegrastats", id))
		return false
	}
	available := boolField(tools, "tegrastats_available")
	if available {
		sum.RowsWithTegrastats++
	}
	if available && cfg.requireTegrastats {
		if numericField(tegrastats, "sample_count") <= 0 {
			sum.Issues = append(sum.Issues, fmt.Sprintf("%s: tegrastats has no samples", id))
			return false
		}
		if numericField(tegrastats, "parsed_sample_count") <= 0 {
			sum.Issues = append(sum.Issues, fmt.Sprintf("%s: tegrastats has no parsed samples", id))
			return false
		}
		if _, ok := tegrastats["ram_used_delta_gib"]; !ok {
			sum.Issues = append(sum.Issues, fmt.Sprintf("%s: tegrastats missing ram_used_delta_gib", id))
			return false
		}
	}
	return true
}

func checkLoadRecord(id string, row map[string]any, sum *summary) bool {
	load := objectField(row, "load_short_decode")
	if load == nil {
		sum.Issues = append(sum.Issues, fmt.Sprintf("%s: missing load_short_decode", id))
		return false
	}
	ok := true
	for _, key := range []string{"successes", "errors", "wall_seconds", "completion_tokens_per_second", "total_tokens_per_second"} {
		if _, exists := load[key]; !exists {
			sum.Issues = append(sum.Issues, fmt.Sprintf("%s: load_short_decode missing %s", id, key))
			ok = false
		}
	}
	if objectField(load, "latency_seconds") == nil {
		sum.Issues = append(sum.Issues, fmt.Sprintf("%s: load_short_decode missing latency_seconds", id))
		ok = false
	}
	monitor := objectField(load, "memory_monitor")
	if monitor == nil {
		sum.Issues = append(sum.Issues, fmt.Sprintf("%s: load_short_decode missing memory_monitor", id))
		ok = false
	} else if numericField(monitor, "samples") <= 0 {
		sum.Issues = append(sum.Issues, fmt.Sprintf("%s: memory_monitor has no samples", id))
		ok = false
	}
	return ok
}

func fatal(cfg config, sum summary) {
	printSummary(cfg, sum)
	os.Exit(1)
}

func printSummary(cfg config, sum summary) {
	if cfg.jsonOutput {
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		_ = encoder.Encode(sum)
		return
	}
	fmt.Printf("rows: %d / %d\n", sum.Rows, sum.MinRows)
	fmt.Printf("load rows: %d\n", sum.LoadRows)
	fmt.Printf("startup-only/skipped rows: %d\n", sum.StartupOnlyRows)
	fmt.Printf("rows with tegrastats: %d\n", sum.RowsWithTegrastats)
	fmt.Printf("unique candidates: %d\n", sum.UniqueCandidates)
	fmt.Printf("max context: %d\n", sum.MaxContext)
	fmt.Printf("max max_num_seqs: %d\n", sum.MaxSeqs)
	fmt.Println("statuses:")
	for _, status := range sortedKeys(sum.Statuses) {
		fmt.Printf("  %s: %d\n", status, sum.Statuses[status])
	}
	if len(sum.Issues) == 0 {
		fmt.Println("result: ok")
		return
	}
	fmt.Println("result: failed")
	fmt.Println("issues:")
	for _, issue := range sum.Issues {
		fmt.Printf("  - %s\n", issue)
	}
}

func sortedKeys(values map[string]int) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func sortedInts(values map[int]bool) []int {
	out := make([]int, 0, len(values))
	for value := range values {
		out = append(out, value)
	}
	sort.Ints(out)
	return out
}

func objectField(row map[string]any, key string) map[string]any {
	value, ok := row[key]
	if !ok || value == nil {
		return nil
	}
	object, ok := value.(map[string]any)
	if !ok {
		return nil
	}
	return object
}

func arrayField(row map[string]any, key string) []any {
	value, ok := row[key]
	if !ok || value == nil {
		return nil
	}
	array, ok := value.([]any)
	if !ok {
		return nil
	}
	return array
}

func stringField(row map[string]any, key string) string {
	value, ok := row[key]
	if !ok || value == nil {
		return ""
	}
	text, ok := value.(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(text)
}

func boolField(row map[string]any, key string) bool {
	value, ok := row[key]
	if !ok || value == nil {
		return false
	}
	flag, ok := value.(bool)
	return ok && flag
}

func numericField(row map[string]any, key string) float64 {
	value, ok := row[key]
	if !ok || value == nil {
		return 0
	}
	number, ok := value.(float64)
	if !ok {
		return 0
	}
	return number
}

type intList []int

func (values *intList) String() string {
	if values == nil {
		return ""
	}
	parts := make([]string, 0, len(*values))
	for _, value := range *values {
		parts = append(parts, fmt.Sprint(value))
	}
	return strings.Join(parts, ",")
}

func (values *intList) Set(raw string) error {
	parsed, err := strconv.Atoi(raw)
	if err != nil || parsed <= 0 {
		return fmt.Errorf("invalid positive integer %q", raw)
	}
	*values = append(*values, parsed)
	return nil
}
