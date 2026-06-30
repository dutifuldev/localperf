package vllmbench

import (
	"testing"
	"time"
)

func TestRequestSamplesFromResultReadsVLLMBenchArrays(t *testing.T) {
	samples, err := requestSamplesFromResultData([]byte(`{
  "date": "20260102-030405",
  "completed": 2,
  "failed": 0,
  "input_lens": [100, 200],
  "output_lens": [3, 2],
  "ttfts": [0.1, 0.2],
  "itls": [[0.05, 0.07], [0.1]],
  "start_times": [10.0, 11.5]
}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 2 {
		t.Fatalf("samples = %d, want 2", len(samples))
	}
	first := samples[0]
	if first.RequestID != "vllm-bench-0" || first.Status != "completed" || !first.Streamed {
		t.Fatalf("first sample identity/status = %+v", first)
	}
	if first.PromptTokens != 100 || first.CompletionTokens != 3 || first.TotalTokens != 103 {
		t.Fatalf("first sample tokens = %+v", first)
	}
	if !near(first.TTFTMillis, 100) || !near(first.ITLMeanMillis, 60) || !near(first.TPOTMillis, 60) {
		t.Fatalf("first sample timings = ttft %.3f itl %.3f tpot %.3f, want 100/60/60", first.TTFTMillis, first.ITLMeanMillis, first.TPOTMillis)
	}
	if !near(first.LatencyMillis, 220) || !near(first.OutputTokensPerSecond, 13.636363636363637) {
		t.Fatalf("first sample latency/output = %.6f/%.6f", first.LatencyMillis, first.OutputTokensPerSecond)
	}
	if first.FirstByteAt == nil || first.CompletedAt == nil {
		t.Fatalf("first sample timestamps missing: %+v", first)
	}
	if got := samples[1].StartedAt.Sub(samples[0].StartedAt); got != 1500*time.Millisecond {
		t.Fatalf("start delta = %s, want 1.5s", got)
	}
	if samples[0].ResponseMetadata["source"] != "vllm_bench" {
		t.Fatalf("metadata = %+v", samples[0].ResponseMetadata)
	}
}

func TestRequestSamplesFromResultUsesVLLMBenchPerRequestErrors(t *testing.T) {
	samples, err := requestSamplesFromResultData([]byte(`{
  "date": "20260102-030405",
  "completed": 1,
  "failed": 1,
  "input_lens": [100, 200],
  "output_lens": [0, 10],
  "ttfts": [0.0, 0.1],
  "itls": [[], [0.2]],
  "errors": ["boom", ""]
}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 2 {
		t.Fatalf("samples = %d, want 2", len(samples))
	}
	if samples[0].Status != "failed" || samples[0].ErrorType != "vllm_bench_error" || samples[0].ErrorMessage != "boom" {
		t.Fatalf("first sample = %+v, want failed vLLM error", samples[0])
	}
	if samples[1].Status != "completed" || samples[1].CompletionTokens != 10 {
		t.Fatalf("second sample = %+v, want completed request", samples[1])
	}
}

func near(got, want float64) bool {
	const epsilon = 0.000001
	diff := got - want
	if diff < 0 {
		diff = -diff
	}
	return diff <= epsilon
}
