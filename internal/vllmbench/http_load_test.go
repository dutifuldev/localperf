package vllmbench

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"
)

func TestValidateLocalPerfHTTPResult(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "good.json")
	writeFile(t, good, `{"completed":1,"failed":0,"output_throughput":12.5}`)
	if err := validateParsedResult(good, "localperf_http"); err != nil {
		t.Fatalf("validate good result: %v", err)
	}

	failed := filepath.Join(dir, "failed.json")
	writeFile(t, failed, `{"completed":1,"failed":2}`)
	if err := validateParsedResult(failed, "localperf_http"); err == nil {
		t.Fatal("expected failed request result to fail validation")
	}

	empty := filepath.Join(dir, "empty.json")
	writeFile(t, empty, "")
	if err := validateParsedResult(empty, "localperf_http"); err == nil {
		t.Fatal("expected empty result to fail validation")
	}
}

func TestStructuredHTTPRequestsReadsPreparedCanonicalDataset(t *testing.T) {
	runDir := filepath.Join(t.TempDir(), "run")
	canonicalPath := filepath.Join(runDir, "datasets", "chat.canonical.jsonl")
	writeFile(t, canonicalPath, `{"id":"one","mode":"chat","messages":[{"role":"user","content":"hello"}],"max_output_tokens":8}
{"id":"two","mode":"chat","messages":[{"role":"user","content":"bye"}],"max_output_tokens":8}
`)
	planned := PlannedRun{
		ResultFile: filepath.Join(runDir, "results", "result.json"),
		Workload: Workload{
			NumPrompts: 1,
			Dataset: DatasetSpec{Prepared: DatasetMaterialization{
				CanonicalPath: filepath.Join("datasets", "chat.canonical.jsonl"),
			}},
		},
	}
	requests, err := structuredHTTPRequests(planned)
	if err != nil {
		t.Fatal(err)
	}
	if len(requests) != 1 || requests[0].ID != "one" {
		t.Fatalf("requests = %+v, want first canonical request only", requests)
	}

	planned.Workload.NumPrompts = 3
	if _, err := structuredHTTPRequests(planned); err == nil {
		t.Fatal("expected too-few prepared requests to fail")
	}
}

func TestRandomHTTPRequestsUseWorkloadBackend(t *testing.T) {
	workload := Workload{
		Name:       "random",
		NumPrompts: 1,
		BenchmarkTrafficConfig: BenchmarkTrafficConfig{
			Backend:         "openai",
			DatasetName:     "random",
			RandomInputLen:  4,
			RandomOutputLen: 8,
		},
	}
	requests, err := randomHTTPRequests(workload)
	if err != nil {
		t.Fatal(err)
	}
	if requests[0].Mode != "" {
		t.Fatalf("random request mode = %q, want workload backend to decide", requests[0].Mode)
	}
	client := openAIHTTPClient{profile: Profile{Model: "model"}, workload: workload}
	body, endpoint, err := client.requestBody(requests[0])
	if err != nil {
		t.Fatal(err)
	}
	if endpoint != "/v1/completions" {
		t.Fatalf("endpoint = %q, want completions endpoint", endpoint)
	}
	if body["prompt"] == nil || body["messages"] != nil {
		t.Fatalf("body = %+v, want completion body", body)
	}
}

func TestRequestEndpointFollowsRequestModeDefaults(t *testing.T) {
	client := openAIHTTPClient{
		profile: Profile{Model: "model"},
		workload: Workload{
			BenchmarkTrafficConfig: BenchmarkTrafficConfig{
				Backend:  "openai-chat",
				Endpoint: "/v1/chat/completions",
			},
		},
	}
	body, endpoint, err := client.requestBody(CanonicalRequest{
		ID:              "one",
		Mode:            "completion",
		Prompt:          "hello",
		MaxOutputTokens: 8,
	})
	if err != nil {
		t.Fatal(err)
	}
	if endpoint != "/v1/completions" {
		t.Fatalf("endpoint = %q, want mode-specific completions endpoint", endpoint)
	}
	if body["prompt"] != "hello" || body["messages"] != nil {
		t.Fatalf("body = %+v, want completion body", body)
	}

	client.workload.Endpoint = "/custom/completions"
	_, endpoint, err = client.requestBody(CanonicalRequest{
		ID:              "two",
		Mode:            "completion",
		Prompt:          "hello",
		MaxOutputTokens: 8,
	})
	if err != nil {
		t.Fatal(err)
	}
	if endpoint != "/custom/completions" {
		t.Fatalf("endpoint = %q, want custom endpoint", endpoint)
	}
}

func TestFeedHTTPJobsAndSleepContext(t *testing.T) {
	jobs := make(chan localPerfHTTPJob)
	done := make(chan []localPerfHTTPJob, 1)
	go func() {
		var got []localPerfHTTPJob
		for job := range jobs {
			got = append(got, job)
		}
		done <- got
	}()
	requests := []CanonicalRequest{{ID: "one"}, {ID: "two"}}
	if err := feedHTTPJobs(context.Background(), jobs, requests, time.Nanosecond); err != nil {
		t.Fatalf("feedHTTPJobs error = %v", err)
	}
	got := <-done
	if len(got) != 2 || got[1].request.ID != "two" {
		t.Fatalf("jobs = %+v, want two ordered jobs", got)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := sleepContext(ctx, time.Hour); !errors.Is(err, context.Canceled) {
		t.Fatalf("sleepContext error = %v, want context canceled", err)
	}

	jobs = make(chan localPerfHTTPJob)
	if err := feedHTTPJobs(ctx, jobs, []CanonicalRequest{{ID: "blocked"}}, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled feedHTTPJobs error = %v, want context canceled", err)
	}
}

func TestRequestRateDelayValues(t *testing.T) {
	cases := []struct {
		value string
		want  time.Duration
		ok    bool
	}{
		{"", 0, true},
		{"inf", 0, true},
		{"infinity", 0, true},
		{"2", 500 * time.Millisecond, true},
		{"0", 0, false},
		{"NaN", 0, false},
		{"+Inf", 0, false},
		{"bad", 0, false},
	}
	for _, tt := range cases {
		got, err := requestRateDelay(tt.value)
		if tt.ok && (err != nil || got != tt.want) {
			t.Fatalf("requestRateDelay(%q) = %s, %v; want %s, nil", tt.value, got, err, tt.want)
		}
		if !tt.ok && err == nil {
			t.Fatalf("requestRateDelay(%q) error = nil, want error", tt.value)
		}
	}
}

func TestOpenAIHTTPClientApplyRequestOptions(t *testing.T) {
	workloadTemp := 0.25
	client := openAIHTTPClient{workload: Workload{
		BenchmarkTrafficConfig: BenchmarkTrafficConfig{Backend: "openai-chat"},
		Temperature:            &workloadTemp,
		IgnoreEOS:              true,
	}}
	body := map[string]any{}
	client.applyRequestOptions(body, CanonicalRequest{})
	if body["temperature"] != workloadTemp || body["ignore_eos"] != true {
		t.Fatalf("workload options not applied: %+v", body)
	}

	requestTemp := 0.75
	body = map[string]any{}
	client.applyRequestOptions(body, CanonicalRequest{Temperature: &requestTemp})
	if body["temperature"] != requestTemp {
		t.Fatalf("request temperature should override workload temperature: %+v", body)
	}
}

func TestRequestSamplesForResultReadsSamples(t *testing.T) {
	dir := t.TempDir()
	samples, err := requestSamplesForResult(dir, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 0 {
		t.Fatalf("empty result path samples = %+v, want none", samples)
	}

	path := filepath.Join(dir, "result.json")
	writeFile(t, path, `{
  "completed": 1,
  "request_samples": [
    {
      "request_index": 0,
      "request_id": "one",
      "status": "completed",
      "latency_ms": 10,
      "completion_tokens": 5,
      "output_tokens_per_second": 500
    }
  ]
}`)
	samples, err = requestSamplesForResult(dir, "result.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 1 || samples[0].RequestID != "one" || samples[0].OutputTokensPerSecond != 500 {
		t.Fatalf("samples = %+v, want parsed request sample", samples)
	}

	writeFile(t, filepath.Join(dir, "no-samples.json"), `{"completed":1}`)
	samples, err = requestSamplesForResult(dir, "no-samples.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 0 {
		t.Fatalf("missing request_samples parsed as %+v, want none", samples)
	}

	writeFile(t, filepath.Join(dir, "bad-samples.json"), `{"request_samples":{}}`)
	if _, err := requestSamplesForResult(dir, "bad-samples.json"); err == nil {
		t.Fatal("expected malformed request_samples to fail")
	}
}
