package benchcli

import "testing"

func TestHTTPLoadWorkloadCarriesConcurrency(t *testing.T) {
	workload, err := httpLoadWorkload("openai-chat", "random", "inf", "", "", "0", true, 3, 4, 128, 16)
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
	workload, err := httpLoadWorkload("openai-chat", "custom", "inf", "", "/tmp/canonical.jsonl", "", false, 3, 2, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if workload.DatasetPath != "/tmp/canonical.jsonl" {
		t.Fatalf("dataset_path = %q, want canonical path", workload.DatasetPath)
	}
	if workload.Dataset.Prepared.CanonicalPath != "/tmp/canonical.jsonl" || workload.Dataset.Prepared.RequestCount != 3 {
		t.Fatalf("prepared dataset = %+v, want canonical path and request count", workload.Dataset.Prepared)
	}
}
