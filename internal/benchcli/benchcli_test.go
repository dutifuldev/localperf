package benchcli

import "testing"

func TestHTTPLoadWorkloadCarriesConcurrency(t *testing.T) {
	workload, err := httpLoadWorkload("openai-chat", "random", "inf", "", "0", true, 3, 4, 128, 16)
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
