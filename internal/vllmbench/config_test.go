package vllmbench

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

func TestBuildPlanAndBenchCommand(t *testing.T) {
	spec := testSpec()
	runDir := filepath.Join("runs", "example")
	plan := BuildPlan(spec, runDir)
	if len(plan) != 2 {
		t.Fatalf("plan length = %d, want 2", len(plan))
	}
	if plan[0].Profile.Name != "8k" || plan[0].Workload.Name != "prefill-8k" || plan[0].Concurrency != 4 {
		t.Fatalf("unexpected first plan row: %+v", plan[0])
	}
	command := BenchCommand(spec, plan[0])
	got := ShellQuote(command.Args)
	for _, want := range []string{
		"vllm bench serve",
		"--backend openai-chat",
		"--dataset-name random",
		"--random-input-len 8192",
		"--random-output-len 16",
		"--endpoint /v1/chat/completions",
		"--max-concurrency 4",
		"--result-filename runs/example/results/8k__prefill-8k__c4.json",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("command %q missing %q", got, want)
		}
	}
}

func TestValidateSpecRequiresMemoryFloor(t *testing.T) {
	spec := testSpec()
	spec.Safety.MinMemAvailableGiB = 0
	if err := ValidateSpec(spec); err == nil || !strings.Contains(err.Error(), "min_mem_available_gib") {
		t.Fatalf("ValidateSpec error = %v, want min_mem_available_gib issue", err)
	}
}

func TestApplyFilter(t *testing.T) {
	spec := testSpec()
	err := ApplyFilter(&spec, Filter{
		Profiles:      []string{"8k"},
		Workloads:     []string{"prefill-8k"},
		Concurrencies: []int{8},
	})
	if err != nil {
		t.Fatal(err)
	}
	plan := BuildPlan(spec, "runs/example")
	if len(plan) != 1 {
		t.Fatalf("plan length = %d, want 1", len(plan))
	}
	if plan[0].Profile.Name != "8k" || plan[0].Workload.Name != "prefill-8k" || plan[0].Concurrency != 8 {
		t.Fatalf("unexpected filtered plan: %+v", plan[0])
	}
}

func TestApplyFilterDropsWorkloadsWithoutMatchingProfile(t *testing.T) {
	spec := testSpec()
	spec.Profiles = append(spec.Profiles, Profile{
		Name:        "16k",
		Model:       spec.Model,
		Host:        "127.0.0.1",
		Port:        8116,
		Managed:     true,
		MaxModelLen: 16384,
		MaxNumSeqs:  16,
	})
	spec.Workloads = append(spec.Workloads, Workload{
		Name:            "prefill-16k",
		Profiles:        []string{"16k"},
		Backend:         "openai-chat",
		DatasetName:     "random",
		RandomInputLen:  14336,
		RandomOutputLen: 16,
		NumPrompts:      8,
		MaxConcurrency:  []int{4},
	})
	ApplyDefaults(&spec)
	if err := ApplyFilter(&spec, Filter{Profiles: []string{"8k"}}); err != nil {
		t.Fatal(err)
	}
	plan := BuildPlan(spec, "runs/example")
	if len(plan) != 2 {
		t.Fatalf("plan length = %d, want only the two 8k workload concurrencies", len(plan))
	}
	for _, run := range plan {
		if run.Profile.Name != "8k" || run.Workload.Name != "prefill-8k" {
			t.Fatalf("unexpected filtered run: %+v", run)
		}
	}
}

func TestApplyFilterDropsWorkloadsWithoutMatchingConcurrency(t *testing.T) {
	spec := testSpec()
	spec.Workloads = append(spec.Workloads, Workload{
		Name:            "claim-repro",
		Profiles:        []string{"8k"},
		Backend:         "openai-chat",
		DatasetName:     "random",
		RandomInputLen:  1000,
		RandomOutputLen: 1024,
		NumPrompts:      20,
		MaxConcurrency:  []int{20},
	})
	ApplyDefaults(&spec)
	if err := ApplyFilter(&spec, Filter{Concurrencies: []int{20}}); err != nil {
		t.Fatal(err)
	}
	plan := BuildPlan(spec, "runs/example")
	if len(plan) != 1 {
		t.Fatalf("plan length = %d, want only the c20 workload", len(plan))
	}
	if plan[0].Workload.Name != "claim-repro" || plan[0].Concurrency != 20 {
		t.Fatalf("unexpected filtered run: %+v", plan[0])
	}
}

func TestParseMeminfo(t *testing.T) {
	meminfo := strings.NewReader(`MemTotal:       131072000 kB
MemFree:         1000000 kB
MemAvailable:    65536000 kB
SwapFree:         4194304 kB
`)
	snapshot, err := ParseMeminfo(meminfo)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.MemAvailableGiB != 62.5 {
		t.Fatalf("MemAvailableGiB = %v, want 62.5", snapshot.MemAvailableGiB)
	}
	if snapshot.SwapFreeGiB != 4 {
		t.Fatalf("SwapFreeGiB = %v, want 4", snapshot.SwapFreeGiB)
	}
}

func TestParseVLLMBenchResult(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "result.json")
	writeFile(t, path, `{"backend":"openai-chat","model_id":"nvidia/diffusiongemma-26B-A4B-it-NVFP4","num_prompts":4,"max_concurrency":1,"duration":13.1517,"completed":4,"failed":0,"output_throughput":311.441,"total_token_throughput":619.612,"mean_ttft_ms":2597.32}`)
	rows, err := ParseResultFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(rows))
	}
	row := rows[0]
	if row.Concurrency != 1 || row.Completed != 4 || row.OutputTokensPerSec != 311.441 {
		t.Fatalf("unexpected row: %+v", row)
	}
	if row.PerUserOutputTokSec != 311.441 {
		t.Fatalf("per-user throughput = %v, want 311.441", row.PerUserOutputTokSec)
	}
}

func TestParseCustomJSONLResult(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "result.jsonl")
	writeFile(t, path, `{"profile":"grid-l8192-s16-gpu035","max_model_len":8192,"server_max_num_seqs":16,"concurrency":8,"requests":8,"max_tokens":512,"wall_seconds":38.117,"ok":8,"failed":0,"completion_tokens":4096,"total_tokens":4400,"aggregate_completion_tokens_per_second":107.459,"aggregate_total_tokens_per_second":115.434}`)
	rows, err := ParseResultFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(rows))
	}
	row := rows[0]
	if row.Context != 8192 || row.Concurrency != 8 || row.RandomOutputLen != 512 {
		t.Fatalf("unexpected row: %+v", row)
	}
	if row.PerUserOutputTokSec < 13.4 || row.PerUserOutputTokSec > 13.5 {
		t.Fatalf("per-user throughput = %v, want about 13.4", row.PerUserOutputTokSec)
	}
}

func TestExecuteDryRunAndReport(t *testing.T) {
	spec := testSpec()
	spec.OutputDir = t.TempDir()
	appendTimestamp := false
	spec.Runner.AppendTimestampToRun = &appendTimestamp
	summary, err := Execute(context.Background(), spec, RunOptions{DryRun: true})
	if err != nil {
		t.Fatal(err)
	}
	if summary.PlannedRuns != 2 {
		t.Fatalf("planned runs = %d, want 2", summary.PlannedRuns)
	}
	for _, path := range []string{
		filepath.Join(summary.RunDir, "events.jsonl"),
		filepath.Join(summary.RunDir, "spec.normalized.json"),
		filepath.Join(summary.RunDir, "summary.json"),
	} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("expected artifact %s: %v", path, err)
		}
	}
}

func TestExecuteWithFakeVLLMEndToEnd(t *testing.T) {
	spec := testSpec()
	spec.Name = "fake-vllm-e2e"
	spec.OutputDir = t.TempDir()
	appendTimestamp := false
	spec.Runner.AppendTimestampToRun = &appendTimestamp
	spec.Runner.VLLMCommand = fakeVLLMScript(t)
	spec.Runner.VLLMBenchCommand = spec.Runner.VLLMCommand
	spec.Safety.MinMemAvailableGiB = 0.1
	spec.Safety.StartupTimeoutSec = 10
	spec.Safety.WorkloadTimeoutSec = 10
	spec.Safety.HTTPTimeoutSec = 2
	spec.Warmup.Enabled = true
	spec.Profiles = spec.Profiles[:1]
	spec.Profiles[0].Port = freeTestPort()
	spec.Workloads = []Workload{{
		Name:            "fake-random",
		Profiles:        []string{spec.Profiles[0].Name},
		Backend:         "openai-chat",
		DatasetName:     "random",
		RandomInputLen:  128,
		RandomOutputLen: 16,
		NumPrompts:      2,
		MaxConcurrency:  []int{2},
		RequestRate:     "inf",
	}}
	ApplyDefaults(&spec)
	summary, err := Execute(context.Background(), spec, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if summary.CompletedRuns != 1 || summary.FailedRuns != 0 {
		t.Fatalf("summary = %+v, want one completed run", summary)
	}
	report, err := BuildReport(summary.RunDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Rows) != 1 {
		t.Fatalf("report rows = %d, want 1", len(report.Rows))
	}
	if report.Rows[0].OutputTokensPerSec != 20 {
		t.Fatalf("output throughput = %v, want 20", report.Rows[0].OutputTokensPerSec)
	}
}

func TestExecuteFailsWhenSleepFails(t *testing.T) {
	spec := testSpec()
	spec.Name = "fake-vllm-sleep-failure"
	spec.OutputDir = t.TempDir()
	appendTimestamp := false
	spec.Runner.AppendTimestampToRun = &appendTimestamp
	spec.Runner.VLLMCommand = fakeVLLMScript(t)
	spec.Runner.VLLMBenchCommand = spec.Runner.VLLMCommand
	spec.Safety.MinMemAvailableGiB = 0.1
	spec.Safety.StartupTimeoutSec = 10
	spec.Safety.WorkloadTimeoutSec = 10
	spec.Safety.HTTPTimeoutSec = 2
	spec.Warmup.Enabled = false
	spec.Profiles = spec.Profiles[:1]
	spec.Profiles[0].Port = freeTestPort()
	spec.Profiles[0].Env = map[string]string{"FAKE_SLEEP_FAIL": "1"}
	spec.Workloads = []Workload{{
		Name:            "fake-random",
		Profiles:        []string{spec.Profiles[0].Name},
		Backend:         "openai-chat",
		DatasetName:     "random",
		RandomInputLen:  128,
		RandomOutputLen: 16,
		NumPrompts:      1,
		MaxConcurrency:  []int{1},
		RequestRate:     "inf",
	}}
	ApplyDefaults(&spec)
	summary, err := Execute(context.Background(), spec, RunOptions{})
	if err == nil || !strings.Contains(err.Error(), "sleep failed") {
		t.Fatalf("Execute error = %v, want sleep failure", err)
	}
	if summary.CompletedRuns != 1 {
		t.Fatalf("completed runs = %d, want measured workload to complete before sleep failure", summary.CompletedRuns)
	}
}

func TestBuildReportEnrichesFromSpec(t *testing.T) {
	spec := testSpec()
	runDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(runDir, "results"), 0o755); err != nil {
		t.Fatal(err)
	}
	ApplyDefaults(&spec)
	if err := writeJSONFile(filepath.Join(runDir, "spec.normalized.json"), spec); err != nil {
		t.Fatal(err)
	}
	resultPath := filepath.Join(runDir, "results", "8k__prefill-8k__c4.json")
	writeFile(t, resultPath, `{"max_concurrency":4,"completed":8,"failed":0,"output_throughput":200,"total_token_throughput":250}`)
	events := `{"timestamp":"2026-06-26T00:00:00Z","type":"workload_finish","profile":"8k","workload":"prefill-8k","concurrency":4,"result_file":"` + resultPath + `"}`
	writeFile(t, filepath.Join(runDir, "events.jsonl"), events+"\n")
	report, err := BuildReport(runDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(report.Rows))
	}
	row := report.Rows[0]
	if row.Context != 8192 || row.RandomInputLen != 8192 || row.RandomOutputLen != 16 || row.DatasetName != "random" {
		t.Fatalf("row was not enriched from spec: %+v", row)
	}
}

func testSpec() Spec {
	temp := 0.0
	oneAwake := true
	stopManaged := true
	spec := Spec{
		Version: "1",
		Name:    "DiffusionGemma Standard",
		Model:   "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
		Env: map[string]string{
			"VLLM_USE_V2_MODEL_RUNNER": "1",
		},
		Runner: RunnerConfig{
			VLLMCommand:       "vllm",
			VLLMBenchCommand:  "vllm",
			OneAwakeProfile:   &oneAwake,
			StopManagedOnExit: &stopManaged,
		},
		Safety: SafetyConfig{
			MinMemAvailableGiB: 40,
		},
		Profiles: []Profile{
			{
				Name:                 "8k",
				Host:                 "127.0.0.1",
				Port:                 8108,
				Managed:              true,
				EnableSleepMode:      true,
				SleepLevel:           2,
				MaxModelLen:          8192,
				MaxNumSeqs:           16,
				MaxNumBatchedTokens:  8192,
				GPUMemoryUtilization: 0.35,
				AttentionBackend:     "TRITON_ATTN",
				MoEBackend:           "cutlass",
			},
		},
		Workloads: []Workload{
			{
				Name:            "prefill-8k",
				Profiles:        []string{"8k"},
				Backend:         "openai-chat",
				DatasetName:     "random",
				RandomInputLen:  8192,
				RandomOutputLen: 16,
				NumPrompts:      8,
				MaxConcurrency:  []int{4, 8},
				RequestRate:     "inf",
				Temperature:     &temp,
			},
		},
	}
	ApplyDefaults(&spec)
	return spec
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func fakeVLLMScript(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fake-vllm")
	script := fmt.Sprintf("#!/bin/sh\nGO_WANT_VLLMBENCH_HELPER=1 exec %s -test.run=TestHelperProcess -- \"$@\"\n", shellSingleQuote(os.Args[0]))
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func shellSingleQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\"'\"'") + "'"
}

func TestHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_VLLMBENCH_HELPER") != "1" {
		return
	}
	args := helperArgs()
	if len(args) == 0 {
		os.Exit(2)
	}
	switch args[0] {
	case "serve":
		runFakeServe(args[1:])
	case "bench":
		runFakeBench(args[1:])
	default:
		os.Exit(2)
	}
}

func helperArgs() []string {
	for i, arg := range os.Args {
		if arg == "--" {
			return os.Args[i+1:]
		}
	}
	return nil
}

func runFakeServe(args []string) {
	port := flagValue(args, "--port")
	if port == "" {
		os.Exit(2)
	}
	var sleeping atomic.Bool
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{}})
	})
	mux.HandleFunc("/is_sleeping", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"is_sleeping": sleeping.Load()})
	})
	mux.HandleFunc("/sleep", func(w http.ResponseWriter, _ *http.Request) {
		if os.Getenv("FAKE_SLEEP_FAIL") == "1" {
			http.Error(w, "sleep failed", http.StatusInternalServerError)
			return
		}
		sleeping.Store(true)
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
	})
	mux.HandleFunc("/wake_up", func(w http.ResponseWriter, _ *http.Request) {
		sleeping.Store(false)
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
	})
	server := &http.Server{Addr: "127.0.0.1:" + port, Handler: mux}
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		<-signals
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		_ = server.Shutdown(ctx)
	}()
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		os.Exit(1)
	}
	os.Exit(0)
}

func runFakeBench(args []string) {
	if len(args) == 0 || args[0] != "serve" {
		os.Exit(2)
	}
	resultPath := flagValue(args, "--result-filename")
	if resultPath == "" {
		os.Exit(2)
	}
	concurrency, _ := strconv.Atoi(flagValue(args, "--max-concurrency"))
	numPrompts, _ := strconv.Atoi(flagValue(args, "--num-prompts"))
	if concurrency <= 0 {
		concurrency = 1
	}
	if numPrompts <= 0 {
		numPrompts = concurrency
	}
	if err := os.MkdirAll(filepath.Dir(resultPath), 0o755); err != nil {
		os.Exit(1)
	}
	row := map[string]any{
		"max_concurrency":        concurrency,
		"completed":              numPrompts,
		"failed":                 0,
		"duration":               1.0,
		"output_throughput":      float64(concurrency * 10),
		"total_token_throughput": float64(concurrency * 12),
	}
	data, _ := json.Marshal(row)
	if err := os.WriteFile(resultPath, append(data, '\n'), 0o644); err != nil {
		os.Exit(1)
	}
	os.Exit(0)
}

func flagValue(args []string, name string) string {
	for i, arg := range args {
		if arg == name && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

func freeTestPort() int {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 19191
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port
}
