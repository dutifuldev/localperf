package benchcli

import (
	"flag"
	"path/filepath"
	"testing"
	"time"
)

func TestProfileFromBaseURLPreservesPathPrefix(t *testing.T) {
	profile, err := profileFromBaseURL("http://127.0.0.1:8000/proxy/", "model")
	if err != nil {
		t.Fatal(err)
	}
	if profile.EndpointBaseURL != "http://127.0.0.1:8000/proxy" {
		t.Fatalf("endpoint base URL = %q, want path prefix preserved", profile.EndpointBaseURL)
	}
}

func TestProfileFromBaseURLAcceptsHTTPS(t *testing.T) {
	profile, err := profileFromBaseURL("https://example.test:443/proxy/", "model")
	if err != nil {
		t.Fatal(err)
	}
	if profile.EndpointBaseURL != "https://example.test:443/proxy" {
		t.Fatalf("endpoint base URL = %q, want HTTPS URL preserved", profile.EndpointBaseURL)
	}
}

func TestProfileFromBaseURLStripsAPIRootV1(t *testing.T) {
	cases := map[string]string{
		"http://127.0.0.1:8000/v1":        "http://127.0.0.1:8000",
		"http://127.0.0.1:8000/proxy/v1/": "http://127.0.0.1:8000/proxy",
	}
	for raw, want := range cases {
		profile, err := profileFromBaseURL(raw, "model")
		if err != nil {
			t.Fatal(err)
		}
		if profile.EndpointBaseURL != want {
			t.Fatalf("endpoint base URL for %q = %q, want %q", raw, profile.EndpointBaseURL, want)
		}
	}
}

func TestProfileFromBaseURLUsesDefaultPorts(t *testing.T) {
	cases := map[string]int{
		"https://api.example.test/v1": 443,
		"http://localhost/v1":         80,
	}
	for raw, wantPort := range cases {
		profile, err := profileFromBaseURL(raw, "model")
		if err != nil {
			t.Fatal(err)
		}
		if profile.Port != wantPort {
			t.Fatalf("profile port for %q = %d, want %d", raw, profile.Port, wantPort)
		}
	}
}

func TestTimeoutSecondsRoundsUp(t *testing.T) {
	if got := timeoutSeconds(500 * time.Millisecond); got != 1 {
		t.Fatalf("timeoutSeconds(500ms) = %d, want 1", got)
	}
	if got := timeoutSeconds(1500 * time.Millisecond); got != 2 {
		t.Fatalf("timeoutSeconds(1500ms) = %d, want 2", got)
	}
}

func TestParseArtifactRenderFlagsAllowsPathBeforeFlags(t *testing.T) {
	config, err := parseArtifactRenderFlags([]string{"run.sqlite", "--output", "report.html", "--store", "--title", "Run"}, flag.ContinueOnError)
	if err != nil {
		t.Fatal(err)
	}
	if config.path != "run.sqlite" || config.output != "report.html" || config.title != "Run" || !config.store {
		t.Fatalf("config = %+v, want path, output, title, store", config)
	}
}

func TestParseArtifactRenderFlagsAllowsPathFlag(t *testing.T) {
	config, err := parseArtifactRenderFlags([]string{"--path", "run.sqlite", "--output", "report.html"}, flag.ContinueOnError)
	if err != nil {
		t.Fatal(err)
	}
	if config.path != "run.sqlite" || config.output != "report.html" {
		t.Fatalf("config = %+v, want flag path and output", config)
	}
}

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
