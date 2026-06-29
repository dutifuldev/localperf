package vllmbench

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
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

func TestBenchCommandSupportsStandardDatasetKnobs(t *testing.T) {
	spec := testSpec()
	seed := 7
	customOutputLen := -1
	shareGPTOutputLen := 256
	spec.Workloads = []Workload{{
		Name:     "standard-knobs",
		Profiles: []string{"8k"},
		BenchmarkTrafficConfig: BenchmarkTrafficConfig{
			Backend:                     "openai-chat",
			DatasetName:                 "sonnet",
			DatasetPath:                 "examples/prompts/sonnet.txt",
			SonnetInputLen:              4096,
			SonnetOutputLen:             64,
			SonnetPrefixLen:             128,
			PrefixRepetitionPrefixLen:   256,
			PrefixRepetitionSuffixLen:   512,
			PrefixRepetitionNumPrefixes: 4,
			PrefixRepetitionOutputLen:   32,
			CustomOutputLen:             &customOutputLen,
			ShareGPTOutputLen:           &shareGPTOutputLen,
			SpeedBenchDatasetSubset:     "reasoning",
			SpeedBenchOutputLen:         128,
			SpeedBenchCategory:          "math",
			Seed:                        &seed,
			DisableShuffle:              true,
			NoOversample:                true,
			SkipChatTemplate:            true,
			SaveDetailed:                true,
			PlotDatasetStats:            true,
			ExtraBody:                   `{"guided_decoding_backend":"outlines"}`,
			Metadata:                    []string{"suite=standard", "shape=sonnet"},
			Goodput:                     []string{"ttft:5000"},
			RequestRate:                 "inf",
			ExtraArgs:                   []string{"--request-id-prefix", "standard"},
		},
		NumPrompts:     2,
		MaxConcurrency: []int{1},
	}}
	ApplyDefaults(&spec)
	command := BenchCommand(spec, BuildPlan(spec, "runs/example")[0])
	got := ShellQuote(command.Args)
	for _, want := range []string{
		"--dataset-name sonnet",
		"--dataset-path examples/prompts/sonnet.txt",
		"--seed 7",
		"--disable-shuffle",
		"--no-oversample",
		"--skip-chat-template",
		"--save-detailed",
		"--plot-dataset-stats",
		"--custom-output-len -1",
		"--sharegpt-output-len 256",
		"--sonnet-input-len 4096",
		"--sonnet-output-len 64",
		"--sonnet-prefix-len 128",
		"--prefix-repetition-prefix-len 256",
		"--prefix-repetition-suffix-len 512",
		"--prefix-repetition-num-prefixes 4",
		"--prefix-repetition-output-len 32",
		"--speed-bench-dataset-subset reasoning",
		"--speed-bench-output-len 128",
		"--speed-bench-category math",
		"--extra-body '{\"guided_decoding_backend\":\"outlines\"}'",
		"--metadata suite=standard",
		"--metadata shape=sonnet",
		"--goodput ttft:5000",
		"--request-id-prefix standard",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("command %q missing %q", got, want)
		}
	}
}

func TestLoadSpecSupportsEngineNeutralShape(t *testing.T) {
	path := filepath.Join(t.TempDir(), "spec.json")
	writeFile(t, path, `{
  "version": "localperf.bench/v1",
  "name": "engine-neutral",
  "model": "example/model",
  "safety": {"min_mem_available_gib": 1},
  "engines": [
    {"name": "vllm", "type": "vllm-managed", "command": "vllm-custom"}
  ],
  "profiles": [
    {
      "name": "4k",
      "engine": "vllm",
      "managed": true,
      "port": 8104,
      "serve": {
        "max_model_len": 4096,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 4096,
        "gpu_memory_utilization": 0.25
      },
      "engine_args": ["--disable-log-requests"]
    }
  ],
  "workloads": [
    {
      "name": "decode",
      "profiles": ["4k"],
      "traffic": {
        "backend": "openai-chat",
        "dataset_name": "random",
        "random_input_len": 128,
        "random_output_len": 16,
        "request_rate": "inf"
      },
      "samples": 3,
      "repeats": 2,
      "concurrency": [1, 2]
    }
  ]
}`)
	spec, err := LoadSpec(path)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Profiles[0].MaxModelLen != 4096 || spec.Profiles[0].MaxNumSeqs != 8 {
		t.Fatalf("serve fields were not lifted into profile: %+v", spec.Profiles[0])
	}
	if spec.Workloads[0].NumPrompts != 3 || spec.Workloads[0].Repeats != 2 {
		t.Fatalf("samples/repeats not normalized: %+v", spec.Workloads[0])
	}
	if got := len(BuildPlan(spec, "runs/example")); got != 4 {
		t.Fatalf("plan length = %d, want 4", got)
	}
	command := ServeCommand(spec, spec.Profiles[0])
	got := ShellQuote(command.Args)
	for _, want := range []string{"vllm-custom serve", "--max-model-len 4096", "--disable-log-requests"} {
		if !strings.Contains(got, want) {
			t.Fatalf("serve command %q missing %q", got, want)
		}
	}
}

func TestCommandSummaryRedactsSensitiveEnv(t *testing.T) {
	summary := CommandSummary(CommandSpec{
		Env: map[string]string{
			"CUTE_DSL_ARCH":  "sm_121a",
			"HF_TOKEN":       "hf_secret",
			"OPENAI_API_KEY": "sk-secret",
		},
		Args: []string{"vllm", "serve", "model"},
	})
	for _, secret := range []string{"hf_secret", "sk-secret"} {
		if strings.Contains(summary, secret) {
			t.Fatalf("summary leaked secret %q: %s", secret, summary)
		}
	}
	for _, want := range []string{"HF_TOKEN='<redacted>'", "OPENAI_API_KEY='<redacted>'", "CUTE_DSL_ARCH=sm_121a"} {
		if !strings.Contains(summary, want) {
			t.Fatalf("summary %q missing %q", summary, want)
		}
	}
}

func TestValidateSpecRejectsInvalidWarmupTraffic(t *testing.T) {
	spec := testSpec()
	spec.Warmup.Enabled = true
	spec.Warmup.DatasetName = "random"
	spec.Warmup.RandomInputLen = -1
	if err := ValidateSpec(spec); err == nil || !strings.Contains(err.Error(), "warmup: random_input_len") {
		t.Fatalf("ValidateSpec error = %v, want warmup random input issue", err)
	}
}

func TestValidateSpecRequiresMemoryFloor(t *testing.T) {
	spec := testSpec()
	spec.Safety.MinMemAvailableGiB = 0
	if err := ValidateSpec(spec); err == nil || !strings.Contains(err.Error(), "min_mem_available_gib") {
		t.Fatalf("ValidateSpec error = %v, want min_mem_available_gib issue", err)
	}
}

func TestValidatePrebootOneAwakeRequiresSleepMode(t *testing.T) {
	spec := testSpec()
	spec.Runner.PrebootProfiles = true
	spec.Profiles[0].EnableSleepMode = false
	if err := ValidateSpec(spec); err == nil || !strings.Contains(err.Error(), "enable_sleep_mode") {
		t.Fatalf("ValidateSpec error = %v, want enable_sleep_mode issue", err)
	}
	oneAwake := false
	spec.Runner.OneAwakeProfile = &oneAwake
	if err := ValidateSpec(spec); err != nil {
		t.Fatalf("ValidateSpec with one_awake_profile=false = %v", err)
	}
}

func TestValidateSpecRejectsProfileSlugCollisions(t *testing.T) {
	spec := testSpec()
	colliding := spec.Profiles[0]
	colliding.Name = "8K"
	colliding.Port = 8109
	spec.Profiles = append(spec.Profiles, colliding)
	if err := ValidateSpec(spec); err == nil || !strings.Contains(err.Error(), "collides") {
		t.Fatalf("ValidateSpec error = %v, want slug collision issue", err)
	}
}

func TestValidateSpecRejectsWorkloadSlugCollisions(t *testing.T) {
	spec := testSpec()
	colliding := spec.Workloads[0]
	colliding.Name = "prefill/8k"
	spec.Workloads = append(spec.Workloads, colliding)
	if err := ValidateSpec(spec); err == nil || !strings.Contains(err.Error(), "collides") {
		t.Fatalf("ValidateSpec error = %v, want slug collision issue", err)
	}
}

func TestValidateSpecRejectsDuplicateConcurrencyValues(t *testing.T) {
	spec := testSpec()
	spec.Workloads[0].MaxConcurrency = []int{4, 4}
	if err := ValidateSpec(spec); err == nil || !strings.Contains(err.Error(), "duplicate max_concurrency") {
		t.Fatalf("ValidateSpec error = %v, want duplicate concurrency issue", err)
	}
}

func TestValidateSpecRejectsDuplicateWorkloadProfileReferences(t *testing.T) {
	spec := testSpec()
	spec.Workloads[0].Profiles = []string{"8k", "8k"}
	if err := ValidateSpec(spec); err == nil || !strings.Contains(err.Error(), "duplicate profile reference") {
		t.Fatalf("ValidateSpec error = %v, want duplicate profile reference issue", err)
	}
}

func TestApplyDefaultsHonorsSleepLevelZero(t *testing.T) {
	spec := testSpec()
	spec.Profiles[0].SleepLevel = testIntPointer(0)
	ApplyDefaults(&spec)
	if got := SleepLevelValue(spec.Profiles[0]); got != 0 {
		t.Fatalf("sleep level = %d, want explicit zero", got)
	}
}

func TestApplyDefaultsSetsOmittedSleepLevelToTwo(t *testing.T) {
	spec := testSpec()
	spec.Profiles[0].SleepLevel = nil
	ApplyDefaults(&spec)
	if got := SleepLevelValue(spec.Profiles[0]); got != 2 {
		t.Fatalf("sleep level = %d, want default level 2", got)
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
		Name:     "prefill-16k",
		Profiles: []string{"16k"},
		BenchmarkTrafficConfig: BenchmarkTrafficConfig{
			Backend:         "openai-chat",
			DatasetName:     "random",
			RandomInputLen:  14336,
			RandomOutputLen: 16,
		},
		NumPrompts:     8,
		MaxConcurrency: []int{4},
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
		Name:     "claim-repro",
		Profiles: []string{"8k"},
		BenchmarkTrafficConfig: BenchmarkTrafficConfig{
			Backend:         "openai-chat",
			DatasetName:     "random",
			RandomInputLen:  1000,
			RandomOutputLen: 1024,
		},
		NumPrompts:     20,
		MaxConcurrency: []int{20},
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
	if row.Context != 8192 || row.Concurrency != 8 || row.DisplayOutputLen() != 512 {
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

func TestExecuteRedactsSensitiveEnvInArtifacts(t *testing.T) {
	spec := testSpec()
	spec.OutputDir = t.TempDir()
	appendTimestamp := false
	spec.Runner.AppendTimestampToRun = &appendTimestamp
	spec.Env["HF_TOKEN"] = "hf_secret"
	spec.Profiles[0].Env = map[string]string{"OPENAI_API_KEY": "sk-secret"}
	summary, err := Execute(context.Background(), spec, RunOptions{DryRun: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{summary.SpecPath, summary.EventsPath} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		text := string(data)
		for _, secret := range []string{"hf_secret", "sk-secret"} {
			if strings.Contains(text, secret) {
				t.Fatalf("%s leaked secret %q:\n%s", path, secret, text)
			}
		}
		if !containsRedactedMarker(text) {
			t.Fatalf("%s did not contain redacted marker:\n%s", path, text)
		}
	}
}

func containsRedactedMarker(text string) bool {
	return strings.Contains(text, "<redacted>") || strings.Contains(text, `\u003credacted\u003e`)
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
	spec.Workloads = []Workload{testRandomWorkload("fake-random", []string{spec.Profiles[0].Name}, 128, 16, 2, []int{2})}
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
	assertSQLiteArtifact(t, summary.ArtifactPath)
}

func assertSQLiteArtifact(t *testing.T, path string) {
	t.Helper()
	if path == "" {
		t.Fatal("summary artifact path is empty")
	}
	if err := CheckSQLiteArtifact(path); err != nil {
		t.Fatalf("artifact check failed: %v", err)
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for table, want := range map[string]int{
		"run":              1,
		"specs":            2,
		"engines":          1,
		"profiles":         1,
		"workloads":        1,
		"measurements":     1,
		"metric_stats":     2,
		"artifacts":        1,
		"events":           1,
		"telemetry_series": 1,
	} {
		var got int
		if err := db.QueryRow("SELECT COUNT(*) FROM " + table).Scan(&got); err != nil {
			t.Fatalf("count %s: %v", table, err)
		}
		if got < want {
			t.Fatalf("%s rows = %d, want at least %d", table, got, want)
		}
	}
	var outputThroughput float64
	if err := db.QueryRow("SELECT aggregate_output_tok_s FROM measurements LIMIT 1").Scan(&outputThroughput); err != nil {
		t.Fatal(err)
	}
	if outputThroughput != 20 {
		t.Fatalf("artifact aggregate_output_tok_s = %v, want 20", outputThroughput)
	}
}

func TestExecuteFailsWhenBenchmarkReportsRequestFailures(t *testing.T) {
	spec := testSpec()
	spec.Name = "fake-vllm-request-failure"
	spec.OutputDir = t.TempDir()
	appendTimestamp := false
	spec.Runner.AppendTimestampToRun = &appendTimestamp
	spec.Runner.VLLMCommand = fakeVLLMScript(t)
	spec.Runner.VLLMBenchCommand = spec.Runner.VLLMCommand
	spec.Env["FAKE_BENCH_FAILED"] = "1"
	spec.Safety.MinMemAvailableGiB = 0.1
	spec.Safety.StartupTimeoutSec = 10
	spec.Safety.WorkloadTimeoutSec = 10
	spec.Safety.HTTPTimeoutSec = 2
	spec.Warmup.Enabled = false
	spec.Profiles = spec.Profiles[:1]
	spec.Profiles[0].Port = freeTestPort()
	spec.Profiles[0].EnableSleepMode = false
	spec.Workloads = []Workload{testRandomWorkload("fake-random", []string{spec.Profiles[0].Name}, 128, 16, 2, []int{2})}
	ApplyDefaults(&spec)
	summary, err := Execute(context.Background(), spec, RunOptions{})
	if err == nil || !strings.Contains(err.Error(), "benchmark run") {
		t.Fatalf("Execute error = %v, want failed benchmark run", err)
	}
	if summary.CompletedRuns != 0 || summary.FailedRuns != 1 {
		t.Fatalf("summary = %+v, want failed workload", summary)
	}
	events, err := os.ReadFile(filepath.Join(summary.RunDir, "events.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(events), "failed request") {
		t.Fatalf("events did not record failed request:\n%s", events)
	}
	report, err := BuildReport(summary.RunDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Rows) != 1 || report.Rows[0].Failed != 1 {
		t.Fatalf("report rows = %+v, want failed request row", report.Rows)
	}
}

func TestExecuteDerivesPerUserAfterPlannedRunEnrichment(t *testing.T) {
	spec := testSpec()
	spec.Name = "fake-vllm-derived-per-user"
	spec.OutputDir = t.TempDir()
	appendTimestamp := false
	spec.Runner.AppendTimestampToRun = &appendTimestamp
	spec.Runner.VLLMCommand = fakeVLLMScript(t)
	spec.Runner.VLLMBenchCommand = spec.Runner.VLLMCommand
	spec.Env["FAKE_BENCH_OMIT_CONCURRENCY"] = "1"
	spec.Safety.MinMemAvailableGiB = 0.1
	spec.Safety.StartupTimeoutSec = 10
	spec.Safety.WorkloadTimeoutSec = 10
	spec.Safety.HTTPTimeoutSec = 2
	spec.Warmup.Enabled = false
	spec.Profiles = spec.Profiles[:1]
	spec.Profiles[0].Port = freeTestPort()
	spec.Profiles[0].EnableSleepMode = false
	spec.Workloads = []Workload{testRandomWorkload("fake-random", []string{spec.Profiles[0].Name}, 128, 16, 2, []int{2})}
	ApplyDefaults(&spec)
	summary, err := Execute(context.Background(), spec, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(summary.Rows) != 1 {
		t.Fatalf("summary rows = %d, want 1", len(summary.Rows))
	}
	if got := summary.Rows[0].PerUserOutputTokSec; got != 10 {
		t.Fatalf("per-user throughput = %v, want 10", got)
	}
}

func TestExecuteStopsManagedProfileAfterWorkloadMemoryFloorAbort(t *testing.T) {
	startFile := filepath.Join(t.TempDir(), "bench.started")
	oldCheckMemoryFloor := checkMemoryFloor
	checkMemoryFloor = func(minGiB float64) (MemorySnapshot, error) {
		snapshot := MemorySnapshot{MemTotalGiB: 128, MemAvailableGiB: minGiB + 1}
		if _, err := os.Stat(startFile); err == nil {
			snapshot.MemAvailableGiB = minGiB - 1
			return snapshot, &MemoryFloorError{Snapshot: snapshot, MinGiB: minGiB}
		}
		return snapshot, nil
	}
	defer func() {
		checkMemoryFloor = oldCheckMemoryFloor
	}()

	spec := testSpec()
	spec.Name = "fake-vllm-memory-floor-abort"
	spec.OutputDir = t.TempDir()
	appendTimestamp := false
	spec.Runner.AppendTimestampToRun = &appendTimestamp
	spec.Runner.VLLMCommand = fakeVLLMScript(t)
	spec.Runner.VLLMBenchCommand = spec.Runner.VLLMCommand
	spec.Env["FAKE_BENCH_STARTED_FILE"] = startFile
	spec.Env["FAKE_BENCH_SLEEP_MS"] = "500"
	spec.Safety.MinMemAvailableGiB = 40
	spec.Safety.PollIntervalMillis = 20
	spec.Safety.StartupTimeoutSec = 10
	spec.Safety.WorkloadTimeoutSec = 10
	spec.Safety.HTTPTimeoutSec = 2
	spec.Warmup.Enabled = false
	spec.Profiles = spec.Profiles[:1]
	spec.Profiles[0].Port = freeTestPort()
	spec.Profiles[0].EnableSleepMode = false
	spec.Workloads = []Workload{testRandomWorkload("fake-random", []string{spec.Profiles[0].Name}, 128, 16, 1, []int{1, 2})}
	ApplyDefaults(&spec)
	summary, err := Execute(context.Background(), spec, RunOptions{})
	if err == nil || !strings.Contains(err.Error(), "benchmark run") {
		t.Fatalf("Execute error = %v, want failed benchmark run", err)
	}
	if summary.CompletedRuns != 0 || summary.FailedRuns != 2 {
		t.Fatalf("summary = %+v, want current and remaining profile runs failed", summary)
	}
	events, err := os.ReadFile(filepath.Join(summary.RunDir, "events.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(events), `"type":"profile_memory_floor_abort"`) {
		t.Fatalf("events did not record profile memory-floor abort:\n%s", events)
	}
	client := &http.Client{Timeout: 200 * time.Millisecond}
	resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/v1/models", spec.Profiles[0].Port))
	if err == nil {
		_ = resp.Body.Close()
		t.Fatalf("expected managed server to be stopped after memory-floor abort, got HTTP %d", resp.StatusCode)
	}
}

func TestExecuteFailsWhenWarmupReportsRequestFailures(t *testing.T) {
	spec := testSpec()
	spec.Name = "fake-vllm-warmup-failure"
	spec.OutputDir = t.TempDir()
	appendTimestamp := false
	spec.Runner.AppendTimestampToRun = &appendTimestamp
	spec.Runner.VLLMCommand = fakeVLLMScript(t)
	spec.Runner.VLLMBenchCommand = spec.Runner.VLLMCommand
	spec.Env["FAKE_BENCH_FAILED"] = "1"
	spec.Safety.MinMemAvailableGiB = 0.1
	spec.Safety.StartupTimeoutSec = 10
	spec.Safety.WorkloadTimeoutSec = 10
	spec.Safety.HTTPTimeoutSec = 2
	spec.Warmup.Enabled = true
	spec.Profiles = spec.Profiles[:1]
	spec.Profiles[0].Port = freeTestPort()
	spec.Profiles[0].EnableSleepMode = false
	spec.Workloads = []Workload{testRandomWorkload("fake-random", []string{spec.Profiles[0].Name}, 128, 16, 1, []int{1})}
	ApplyDefaults(&spec)
	summary, err := Execute(context.Background(), spec, RunOptions{})
	if err == nil || !strings.Contains(err.Error(), "benchmark run") {
		t.Fatalf("Execute error = %v, want failed benchmark run", err)
	}
	if summary.CompletedRuns != 0 || summary.FailedRuns != 1 {
		t.Fatalf("summary = %+v, want failed profile before workload", summary)
	}
	events, err := os.ReadFile(filepath.Join(summary.RunDir, "events.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(events), `"type":"warmup_finish"`) || !strings.Contains(string(events), "warmup result reported") {
		t.Fatalf("events did not record failed warmup:\n%s", events)
	}
}

func TestExecuteWarmsPrebootedProfileAfterWake(t *testing.T) {
	spec := testSpec()
	spec.Name = "fake-vllm-preboot-warm-after-wake"
	spec.OutputDir = t.TempDir()
	appendTimestamp := false
	spec.Runner.AppendTimestampToRun = &appendTimestamp
	spec.Runner.PrebootProfiles = true
	spec.Runner.VLLMCommand = fakeVLLMScript(t)
	spec.Runner.VLLMBenchCommand = spec.Runner.VLLMCommand
	spec.Safety.MinMemAvailableGiB = 0.1
	spec.Safety.StartupTimeoutSec = 10
	spec.Safety.WorkloadTimeoutSec = 10
	spec.Safety.HTTPTimeoutSec = 2
	spec.Warmup.Enabled = true
	spec.Profiles = spec.Profiles[:1]
	spec.Profiles[0].Port = freeTestPort()
	spec.Workloads = []Workload{testRandomWorkload("fake-random", []string{spec.Profiles[0].Name}, 128, 16, 1, []int{1})}
	ApplyDefaults(&spec)
	summary, err := Execute(context.Background(), spec, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if summary.CompletedRuns != 1 || summary.FailedRuns != 0 {
		t.Fatalf("summary = %+v, want one completed run", summary)
	}
	events, err := os.ReadFile(filepath.Join(summary.RunDir, "events.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(string(events), `"type":"warmup_finish"`); got != 2 {
		t.Fatalf("warmup_finish events = %d, want preboot and post-wake warmups:\n%s", got, events)
	}
}

func TestSleepProfileWaitsForSleepingState(t *testing.T) {
	spec := testSpec()
	spec.Name = "fake-vllm-delayed-sleep"
	spec.OutputDir = t.TempDir()
	spec.Runner.VLLMCommand = fakeVLLMScript(t)
	spec.Runner.VLLMBenchCommand = spec.Runner.VLLMCommand
	spec.Safety.MinMemAvailableGiB = 0.1
	spec.Safety.PollIntervalMillis = 20
	spec.Safety.StartupTimeoutSec = 5
	spec.Safety.HTTPTimeoutSec = 2
	spec.Warmup.Enabled = false
	spec.Profiles = spec.Profiles[:1]
	spec.Profiles[0].Port = freeTestPort()
	spec.Profiles[0].Env = map[string]string{"FAKE_SLEEP_DELAY_MS": "250"}
	ApplyDefaults(&spec)
	runDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(runDir, "logs"), 0o755); err != nil {
		t.Fatal(err)
	}
	events, err := newEventWriter(filepath.Join(runDir, "events.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer events.Close()
	proc, err := prepareProfile(context.Background(), spec, runDir, spec.Profiles[0], events, false)
	if err != nil {
		t.Fatal(err)
	}
	defer stopProcess(proc)
	start := time.Now()
	if err := sleepProfile(context.Background(), spec, spec.Profiles[0], events); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed < 200*time.Millisecond {
		t.Fatalf("sleepProfile returned before delayed sleep completed: %s", elapsed)
	}
}

func TestWakeProfileWaitsForAwakeState(t *testing.T) {
	spec := testSpec()
	spec.Name = "fake-vllm-delayed-wake"
	spec.OutputDir = t.TempDir()
	spec.Runner.VLLMCommand = fakeVLLMScript(t)
	spec.Runner.VLLMBenchCommand = spec.Runner.VLLMCommand
	spec.Safety.MinMemAvailableGiB = 0.1
	spec.Safety.PollIntervalMillis = 20
	spec.Safety.StartupTimeoutSec = 5
	spec.Safety.HTTPTimeoutSec = 2
	spec.Warmup.Enabled = false
	spec.Profiles = spec.Profiles[:1]
	spec.Profiles[0].Port = freeTestPort()
	spec.Profiles[0].Env = map[string]string{"FAKE_WAKE_DELAY_MS": "250"}
	ApplyDefaults(&spec)
	runDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(runDir, "logs"), 0o755); err != nil {
		t.Fatal(err)
	}
	events, err := newEventWriter(filepath.Join(runDir, "events.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer events.Close()
	proc, err := prepareProfile(context.Background(), spec, runDir, spec.Profiles[0], events, false)
	if err != nil {
		t.Fatal(err)
	}
	defer stopProcess(proc)
	if err := sleepProfile(context.Background(), spec, spec.Profiles[0], events); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	if err := wakeProfile(context.Background(), spec, spec.Profiles[0], events); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed < 200*time.Millisecond {
		t.Fatalf("wakeProfile returned before delayed wake completed: %s", elapsed)
	}
}

func TestExecuteStopsPrebootedProfileAfterWakeFailure(t *testing.T) {
	spec := testSpec()
	spec.Name = "fake-vllm-preboot-wake-failure"
	spec.OutputDir = t.TempDir()
	appendTimestamp := false
	spec.Runner.AppendTimestampToRun = &appendTimestamp
	spec.Runner.PrebootProfiles = true
	spec.Runner.VLLMCommand = fakeVLLMScript(t)
	spec.Runner.VLLMBenchCommand = spec.Runner.VLLMCommand
	spec.Safety.MinMemAvailableGiB = 0.1
	spec.Safety.StartupTimeoutSec = 10
	spec.Safety.WorkloadTimeoutSec = 10
	spec.Safety.HTTPTimeoutSec = 2
	spec.Warmup.Enabled = false
	spec.Profiles = spec.Profiles[:1]
	spec.Profiles[0].Port = freeTestPort()
	spec.Profiles[0].Env = map[string]string{"FAKE_WAKE_FAIL": "1"}
	spec.Workloads = []Workload{testRandomWorkload("fake-random", []string{spec.Profiles[0].Name}, 128, 16, 1, []int{1})}
	ApplyDefaults(&spec)
	summary, err := Execute(context.Background(), spec, RunOptions{})
	if err == nil || !strings.Contains(err.Error(), "benchmark run") {
		t.Fatalf("Execute error = %v, want failed benchmark run", err)
	}
	if summary.CompletedRuns != 0 || summary.FailedRuns != 1 {
		t.Fatalf("summary = %+v, want failed profile run", summary)
	}
	events, err := os.ReadFile(filepath.Join(summary.RunDir, "events.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(events), `"type":"profile_failed"`) || !strings.Contains(string(events), "wake failed") {
		t.Fatalf("events did not record wake failure:\n%s", events)
	}
	client := &http.Client{Timeout: 200 * time.Millisecond}
	resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/v1/models", spec.Profiles[0].Port))
	if err == nil {
		_ = resp.Body.Close()
		t.Fatalf("expected prebooted server to be stopped after wake failure, got HTTP %d", resp.StatusCode)
	}
}

func TestExecuteStopsManagedProfileOnInterrupt(t *testing.T) {
	startFile := filepath.Join(t.TempDir(), "bench.started")
	spec := testSpec()
	spec.Name = "fake-vllm-interrupt"
	spec.OutputDir = t.TempDir()
	appendTimestamp := false
	spec.Runner.AppendTimestampToRun = &appendTimestamp
	spec.Runner.VLLMCommand = fakeVLLMScript(t)
	spec.Runner.VLLMBenchCommand = spec.Runner.VLLMCommand
	spec.Env["FAKE_BENCH_STARTED_FILE"] = startFile
	spec.Env["FAKE_BENCH_SLEEP_MS"] = "5000"
	spec.Safety.MinMemAvailableGiB = 0.1
	spec.Safety.StartupTimeoutSec = 10
	spec.Safety.WorkloadTimeoutSec = 10
	spec.Safety.HTTPTimeoutSec = 2
	spec.Warmup.Enabled = false
	spec.Profiles = spec.Profiles[:1]
	spec.Profiles[0].Port = freeTestPort()
	spec.Profiles[0].EnableSleepMode = false
	spec.Workloads = []Workload{testRandomWorkload("fake-random", []string{spec.Profiles[0].Name}, 128, 16, 1, []int{1})}
	ApplyDefaults(&spec)
	type result struct {
		summary RunSummary
		err     error
	}
	done := make(chan result, 1)
	go func() {
		summary, err := Execute(context.Background(), spec, RunOptions{})
		done <- result{summary: summary, err: err}
	}()
	waitForFile(t, startFile)
	process, err := os.FindProcess(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	if err := process.Signal(os.Interrupt); err != nil {
		t.Fatal(err)
	}
	var got result
	select {
	case got = <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Execute did not return after interrupt")
	}
	if got.err == nil {
		t.Fatalf("Execute error = nil, want interrupted run failure; summary = %+v", got.summary)
	}
	client := &http.Client{Timeout: 200 * time.Millisecond}
	resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/v1/models", spec.Profiles[0].Port))
	if err == nil {
		_ = resp.Body.Close()
		t.Fatalf("expected managed server to be stopped after interrupt, got HTTP %d", resp.StatusCode)
	}
}

func TestStopProcessUsesSavedProcessGroupAfterParentExit(t *testing.T) {
	childFile := filepath.Join(t.TempDir(), "child.pid")
	cmd := exec.Command("sh", "-c", fmt.Sprintf("sleep 60 & echo $! > %s", shellSingleQuote(childFile)))
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	pgid, err := syscall.Getpgid(cmd.Process.Pid)
	if err != nil {
		t.Fatal(err)
	}
	proc := &serverProcess{cmd: cmd, pgid: pgid, done: make(chan error, 1)}
	go func() {
		proc.done <- cmd.Wait()
	}()
	childPID := waitForPIDFile(t, childFile)
	defer func() {
		_ = syscall.Kill(-pgid, syscall.SIGKILL)
	}()
	select {
	case err := <-proc.done:
		proc.done <- err
	case <-time.After(2 * time.Second):
		t.Fatal("launcher did not exit")
	}
	if !processExists(childPID) {
		t.Fatalf("child process %d exited before cleanup", childPID)
	}
	stopProcess(proc)
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if !processExists(childPID) {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("child process %d survived process-group cleanup", childPID)
}

func TestExecuteHonorsStopManagedOnExitFalse(t *testing.T) {
	spec := testSpec()
	spec.Name = "fake-vllm-keepalive"
	spec.OutputDir = t.TempDir()
	appendTimestamp := false
	stopManaged := false
	spec.Runner.AppendTimestampToRun = &appendTimestamp
	spec.Runner.StopManagedOnExit = &stopManaged
	spec.Runner.VLLMCommand = fakeVLLMScript(t)
	spec.Runner.VLLMBenchCommand = spec.Runner.VLLMCommand
	spec.Safety.MinMemAvailableGiB = 0.1
	spec.Safety.StartupTimeoutSec = 10
	spec.Safety.WorkloadTimeoutSec = 10
	spec.Safety.HTTPTimeoutSec = 2
	spec.Warmup.Enabled = false
	spec.Profiles = spec.Profiles[:1]
	spec.Profiles[0].Port = freeTestPort()
	spec.Profiles[0].EnableSleepMode = false
	spec.Workloads = []Workload{testRandomWorkload("fake-random", []string{spec.Profiles[0].Name}, 128, 16, 1, []int{1})}
	ApplyDefaults(&spec)
	summary, err := Execute(context.Background(), spec, RunOptions{})
	defer shutdownFakeServer(spec.Profiles[0].Port)
	if err != nil {
		t.Fatal(err)
	}
	if summary.CompletedRuns != 1 || summary.FailedRuns != 0 {
		t.Fatalf("summary = %+v, want one completed run", summary)
	}
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/v1/models", spec.Profiles[0].Port))
	if err != nil {
		t.Fatalf("expected managed server to remain alive: %v", err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("health status = %d, want 200", resp.StatusCode)
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
	spec.Workloads = []Workload{testRandomWorkload("fake-random", []string{spec.Profiles[0].Name}, 128, 16, 1, []int{1})}
	ApplyDefaults(&spec)
	summary, err := Execute(context.Background(), spec, RunOptions{})
	if err == nil || !strings.Contains(err.Error(), "sleep failed") {
		t.Fatalf("Execute error = %v, want sleep failure", err)
	}
	if summary.CompletedRuns != 1 {
		t.Fatalf("completed runs = %d, want measured workload to complete before sleep failure", summary.CompletedRuns)
	}
}

func TestExecuteFinalizesArtifactsWhenPrebootFails(t *testing.T) {
	spec := testSpec()
	spec.Name = "fake-vllm-preboot-failure"
	spec.OutputDir = t.TempDir()
	appendTimestamp := false
	spec.Runner.AppendTimestampToRun = &appendTimestamp
	spec.Runner.PrebootProfiles = true
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
	spec.Workloads = []Workload{testRandomWorkload("fake-random", []string{spec.Profiles[0].Name}, 128, 16, 1, []int{1})}
	ApplyDefaults(&spec)
	summary, err := Execute(context.Background(), spec, RunOptions{})
	if err == nil || !strings.Contains(err.Error(), "preboot profiles failed") {
		t.Fatalf("Execute error = %v, want preboot failure", err)
	}
	if summary.CompletedRuns != 0 || summary.FailedRuns != 1 || summary.FinishedAt.IsZero() {
		t.Fatalf("summary = %+v, want finalized failed run", summary)
	}
	for _, path := range []string{
		filepath.Join(summary.RunDir, "events.jsonl"),
		filepath.Join(summary.RunDir, "summary.json"),
		filepath.Join(summary.RunDir, "report.md"),
		filepath.Join(summary.RunDir, "report.json"),
	} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("expected artifact %s: %v", path, err)
		}
	}
	events, err := os.ReadFile(filepath.Join(summary.RunDir, "events.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(events), `"type":"preboot_failed"`) || !strings.Contains(string(events), `"type":"run_finish"`) {
		t.Fatalf("events did not record preboot failure and run finish:\n%s", events)
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

func TestBuildReportEnrichesGenericLengthsFromSpec(t *testing.T) {
	spec := testSpec()
	spec.Workloads = []Workload{{
		Name:     "sonnet-prefill",
		Profiles: []string{"8k"},
		BenchmarkTrafficConfig: BenchmarkTrafficConfig{
			Backend:         "openai-chat",
			DatasetName:     "sonnet",
			SonnetInputLen:  4096,
			SonnetOutputLen: 32,
			RequestRate:     "inf",
		},
		NumPrompts:     4,
		MaxConcurrency: []int{2},
	}}
	ApplyDefaults(&spec)
	runDir := t.TempDir()
	if err := writeJSONFile(filepath.Join(runDir, "spec.normalized.json"), spec); err != nil {
		t.Fatal(err)
	}
	resultPath := filepath.Join(runDir, "results", "8k__sonnet-prefill__c2.json")
	writeFile(t, resultPath, `{"completed":4,"failed":0,"output_throughput":80}`)
	writeFile(t, filepath.Join(runDir, "events.jsonl"), `{"timestamp":"2026-06-26T00:00:00Z","type":"workload_finish","profile":"8k","workload":"sonnet-prefill","concurrency":2,"result_file":"results/8k__sonnet-prefill__c2.json"}`+"\n")
	report, err := BuildReport(runDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(report.Rows))
	}
	row := report.Rows[0]
	if row.InputLen != 4096 || row.OutputLen != 32 || row.DisplayInputLen() != 4096 || row.DisplayOutputLen() != 32 {
		t.Fatalf("row was not enriched with generic lengths: %+v", row)
	}
}

func TestBuildReportDerivesPerUserAfterEventEnrichment(t *testing.T) {
	runDir := t.TempDir()
	resultPath := filepath.Join(runDir, "results", "p__w__c4.json")
	writeFile(t, resultPath, `{"completed":4,"failed":0,"output_throughput":100}`)
	writeFile(t, filepath.Join(runDir, "events.jsonl"), `{"timestamp":"2026-06-26T00:00:00Z","type":"workload_finish","profile":"p","workload":"w","concurrency":4,"result_file":"results/p__w__c4.json"}`+"\n")
	report, err := BuildReport(runDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(report.Rows))
	}
	if got := report.Rows[0].PerUserOutputTokSec; got != 25 {
		t.Fatalf("per-user throughput = %v, want 25", got)
	}
}

func TestWriteReportFilesRejectsSidecarOutputPaths(t *testing.T) {
	for _, name := range []string{"report.json", "report.csv"} {
		err := WriteReportFiles(Report{RunDir: t.TempDir(), Generated: time.Now()}, filepath.Join(t.TempDir(), name))
		if err == nil || !strings.Contains(err.Error(), "must not end in .json or .csv") {
			t.Fatalf("WriteReportFiles(%s) error = %v, want sidecar extension rejection", name, err)
		}
	}
}

func TestWriteReportFilesWritesCSV(t *testing.T) {
	runDir := t.TempDir()
	outputPath := filepath.Join(runDir, "report.md")
	report := Report{
		RunDir:    runDir,
		Generated: time.Date(2026, 6, 26, 0, 0, 0, 0, time.UTC),
		Rows: []ReportRow{{
			Profile:             "8k",
			Workload:            "prefill",
			DatasetName:         "random",
			Context:             8192,
			ServerMaxNumSeqs:    16,
			Concurrency:         4,
			RandomInputLen:      7168,
			RandomOutputLen:     16,
			Completed:           8,
			Failed:              0,
			DurationSeconds:     10.5,
			OutputTokensPerSec:  200,
			TotalTokensPerSec:   250,
			PerUserOutputTokSec: 50,
			MeanTTFTMillis:      1234.5,
			ResultFile:          filepath.Join(runDir, "results", "8k__prefill__c4.json"),
		}},
	}
	if err := WriteReportFiles(report, outputPath); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{outputPath, filepath.Join(runDir, "report.json"), filepath.Join(runDir, "report.csv")} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("expected report artifact %s: %v", path, err)
		}
	}
	csvData, err := os.ReadFile(filepath.Join(runDir, "report.csv"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(csvData)
	for _, want := range []string{
		"profile,workload,dataset_name,context",
		"8k,prefill,random,8192,16,4,,7168,16,7168,16,8,0,10.5,200,250,50,1234.5",
		"results/8k__prefill__c4.json",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("CSV %q missing %q", text, want)
		}
	}
}

func TestBuildReportResolvesCWDRelativeResultPathsWithAbsoluteRunDir(t *testing.T) {
	root := t.TempDir()
	runDir := filepath.Join(root, "runs", "foo")
	resultPath := filepath.Join(runDir, "results", "p__w__c1.json")
	if err := os.MkdirAll(filepath.Dir(resultPath), 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, resultPath, `{"max_concurrency":1,"completed":1,"output_throughput":10}`)
	writeFile(t, filepath.Join(runDir, "events.jsonl"), `{"timestamp":"2026-06-26T00:00:00Z","type":"workload_finish","profile":"p","workload":"w","concurrency":1,"result_file":"runs/foo/results/p__w__c1.json"}`+"\n")
	t.Chdir(t.TempDir())
	report, err := BuildReport(runDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(report.Rows))
	}
	if report.Rows[0].ResultFile != resultPath {
		t.Fatalf("result path = %q, want %q", report.Rows[0].ResultFile, resultPath)
	}
}

func TestBuildReportPrefersRunRelativeResultPaths(t *testing.T) {
	root := t.TempDir()
	t.Chdir(root)
	runDir := filepath.Join("runs", "foo")
	resultPath := filepath.Join(runDir, "results", "p__w__c1.json")
	cwdResultPath := filepath.Join("results", "p__w__c1.json")
	writeFile(t, resultPath, `{"max_concurrency":1,"completed":1,"output_throughput":10}`)
	writeFile(t, cwdResultPath, `{"max_concurrency":1,"completed":1,"output_throughput":999}`)
	writeFile(t, filepath.Join(runDir, "events.jsonl"), `{"timestamp":"2026-06-26T00:00:00Z","type":"workload_finish","profile":"p","workload":"w","concurrency":1,"result_file":"results/p__w__c1.json"}`+"\n")
	report, err := BuildReport(runDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(report.Rows))
	}
	if report.Rows[0].OutputTokensPerSec != 10 {
		t.Fatalf("output throughput = %v, want run-relative result", report.Rows[0].OutputTokensPerSec)
	}
	if report.Rows[0].ResultFile != resultPath {
		t.Fatalf("result path = %q, want %q", report.Rows[0].ResultFile, resultPath)
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
				SleepLevel:           testIntPointer(2),
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
				Name:     "prefill-8k",
				Profiles: []string{"8k"},
				BenchmarkTrafficConfig: BenchmarkTrafficConfig{
					Backend:         "openai-chat",
					DatasetName:     "random",
					RandomInputLen:  8192,
					RandomOutputLen: 16,
					RequestRate:     "inf",
				},
				NumPrompts:     8,
				MaxConcurrency: []int{4, 8},
				Temperature:    &temp,
			},
		},
	}
	ApplyDefaults(&spec)
	return spec
}

func testRandomWorkload(name string, profiles []string, inputLen, outputLen, numPrompts int, concurrencies []int) Workload {
	return Workload{
		Name:     name,
		Profiles: profiles,
		BenchmarkTrafficConfig: BenchmarkTrafficConfig{
			Backend:         "openai-chat",
			DatasetName:     "random",
			RandomInputLen:  inputLen,
			RandomOutputLen: outputLen,
			RequestRate:     "inf",
		},
		NumPrompts:     numPrompts,
		MaxConcurrency: concurrencies,
	}
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

func testIntPointer(value int) *int {
	return &value
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
	sleepDelay := durationFromEnv("FAKE_SLEEP_DELAY_MS")
	wakeDelay := durationFromEnv("FAKE_WAKE_DELAY_MS")
	var sleeping atomic.Bool
	mux := http.NewServeMux()
	var server *http.Server
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
		setSleepingAfter(&sleeping, true, sleepDelay)
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
	})
	mux.HandleFunc("/wake_up", func(w http.ResponseWriter, _ *http.Request) {
		if os.Getenv("FAKE_WAKE_FAIL") == "1" {
			http.Error(w, "wake failed", http.StatusInternalServerError)
			return
		}
		setSleepingAfter(&sleeping, false, wakeDelay)
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
	})
	mux.HandleFunc("/shutdown", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			_ = server.Shutdown(ctx)
		}()
	})
	server = &http.Server{Addr: "127.0.0.1:" + port, Handler: mux}
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

func durationFromEnv(key string) time.Duration {
	value, _ := strconv.Atoi(os.Getenv(key))
	if value <= 0 {
		return 0
	}
	return time.Duration(value) * time.Millisecond
}

func setSleepingAfter(sleeping *atomic.Bool, value bool, delay time.Duration) {
	if delay <= 0 {
		sleeping.Store(value)
		return
	}
	go func() {
		time.Sleep(delay)
		sleeping.Store(value)
	}()
}

func runFakeBench(args []string) {
	if len(args) == 0 || args[0] != "serve" {
		os.Exit(2)
	}
	resultPath := flagValue(args, "--result-filename")
	if resultPath == "" {
		os.Exit(2)
	}
	if startFile := os.Getenv("FAKE_BENCH_STARTED_FILE"); startFile != "" {
		_ = os.MkdirAll(filepath.Dir(startFile), 0o755)
		_ = os.WriteFile(startFile, []byte("1\n"), 0o644)
	}
	if rawSleepMillis := os.Getenv("FAKE_BENCH_SLEEP_MS"); rawSleepMillis != "" {
		sleepMillis, _ := strconv.Atoi(rawSleepMillis)
		if sleepMillis > 0 {
			time.Sleep(time.Duration(sleepMillis) * time.Millisecond)
		}
	}
	concurrency, _ := strconv.Atoi(flagValue(args, "--max-concurrency"))
	numPrompts, _ := strconv.Atoi(flagValue(args, "--num-prompts"))
	if concurrency <= 0 {
		concurrency = 1
	}
	if numPrompts <= 0 {
		numPrompts = concurrency
	}
	failed, _ := strconv.Atoi(os.Getenv("FAKE_BENCH_FAILED"))
	if failed < 0 {
		failed = 0
	}
	if failed > numPrompts {
		failed = numPrompts
	}
	if err := os.MkdirAll(filepath.Dir(resultPath), 0o755); err != nil {
		os.Exit(1)
	}
	row := map[string]any{
		"completed":              numPrompts - failed,
		"failed":                 failed,
		"duration":               1.0,
		"output_throughput":      float64(concurrency * 10),
		"total_token_throughput": float64(concurrency * 12),
	}
	if os.Getenv("FAKE_BENCH_OMIT_CONCURRENCY") != "1" {
		row["max_concurrency"] = concurrency
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

func shutdownFakeServer(port int) {
	client := &http.Client{Timeout: time.Second}
	req, err := http.NewRequest(http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/shutdown", port), nil)
	if err == nil {
		if resp, err := client.Do(req); err == nil {
			_ = resp.Body.Close()
		}
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/v1/models", port))
		if err != nil {
			return
		}
		_ = resp.Body.Close()
		time.Sleep(20 * time.Millisecond)
	}
}

func waitForPIDFile(t *testing.T, path string) int {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		data, err := os.ReadFile(path)
		if err == nil {
			pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
			if err == nil && pid > 0 {
				return pid
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("pid file %s was not written", path)
	return 0
}

func waitForFile(t *testing.T, path string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(path); err == nil {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("file %s was not written", path)
}

func processExists(pid int) bool {
	err := syscall.Kill(pid, 0)
	return err == nil || err == syscall.EPERM
}

func freeTestPort() int {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 19191
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port
}
