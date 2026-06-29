package benchcli

import (
	"path/filepath"
	"testing"
)

func TestHTTPLoadWorkloadCarriesConcurrency(t *testing.T) {
	workload, err := httpLoadWorkload("openai-chat", "random", "inf", "", "", "", "0", true, 3, 4, 128, 16)
	if err != nil {
		t.Fatal(err)
	}
	if len(workload.MaxConcurrency) != 1 || workload.MaxConcurrency[0] != 4 {
		t.Fatalf("max concurrency = %v, want [4]", workload.MaxConcurrency)
	}
	if workload.Temperature == nil || *workload.Temperature != 0 {
		t.Fatalf("temperature = %v, want 0", workload.Temperature)
	}
	if !workload.IgnoreEOS {
		t.Fatal("ignore_eos = false, want true")
	}
}

func TestHTTPLoadWorkloadCarriesCanonicalDatasetPath(t *testing.T) {
	workload, err := httpLoadWorkload("openai-chat", "random", "inf", "", "/tmp/canonical.jsonl", `{"top_p":0.95}`, "", false, 3, 2, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if workload.DatasetName != "custom" {
		t.Fatalf("dataset_name = %q, want custom for canonical path", workload.DatasetName)
	}
	if workload.DatasetPath != "/tmp/canonical.jsonl" {
		t.Fatalf("dataset_path = %q, want canonical path", workload.DatasetPath)
	}
	if workload.ExtraBody != `{"top_p":0.95}` {
		t.Fatalf("extra_body = %q, want carried through", workload.ExtraBody)
	}
	if workload.Dataset.Prepared.CanonicalPath != "/tmp/canonical.jsonl" || workload.Dataset.Prepared.RequestCount != 3 {
		t.Fatalf("prepared dataset = %+v, want canonical path and request count", workload.Dataset.Prepared)
	}
}

func TestHTTPLoadWorkloadAbsolutizesRelativeDatasetPath(t *testing.T) {
	workload, err := httpLoadWorkload("openai-chat", "random", "inf", "", "canonical.jsonl", "", "", false, 1, 1, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := filepath.Abs("canonical.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	if workload.Dataset.Prepared.CanonicalPath != want || workload.DatasetPath != want {
		t.Fatalf("dataset path = %q prepared = %q, want %q", workload.DatasetPath, workload.Dataset.Prepared.CanonicalPath, want)
	}
}
