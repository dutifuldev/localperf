package sweepplan

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/dutifuldev/localperf/internal/vllmbench"
)

// TestGeneratedSpecRoundTripsValidation is the guarantee that the generator
// and the validator can never drift: every generated spec must validate with
// zero issues.
func TestGeneratedSpecRoundTripsValidation(t *testing.T) {
	spec, err := Plan(PlanRequest{
		Model:            "example/model",
		Contexts:         []int{8192, 16384, 32768, 65536, 131072},
		Concurrency:      []int{1, 4, 8, 16, 32},
		Repeats:          2,
		IncludeReference: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	vllmbench.ApplyDefaults(&spec)
	if err := vllmbench.ValidateSpec(spec); err != nil {
		t.Fatalf("generated spec failed validation: %v", err)
	}
	// 1 reference + 2 workloads per ladder point.
	if len(spec.Workloads) != 11 {
		t.Fatalf("workloads = %d, want 11", len(spec.Workloads))
	}
	if len(spec.Profiles) != 6 {
		t.Fatalf("profiles = %d, want 6", len(spec.Profiles))
	}
}

func TestShapesStayInsideContractBand(t *testing.T) {
	for _, context := range []int{4096, 8192, 16384, 32768, 65536, 131072, 262144} {
		for _, shape := range []struct {
			name             string
			input, output    int
			minimalOutputMax int
		}{
			{"prefill", 0, 0, 1},
			{"decode", 0, 0, 0},
		} {
			input, output := PrefillShape(context)
			if shape.name == "decode" {
				input, output = DecodeShape(context)
			}
			sum := input + output
			if float64(sum) < vllmbench.ContextTargetMinFrac*float64(context) || sum > context {
				t.Fatalf("%s shape for %d = %d+%d (%d), outside [90%%, 100%%] band", shape.name, context, input, output, sum)
			}
			if shape.name == "prefill" && output != 1 {
				t.Fatalf("prefill output = %d, want 1", output)
			}
		}
	}
}

// TestPlanGolden pins byte-stable generator output: same request in, same
// spec out.
func TestPlanGolden(t *testing.T) {
	spec, err := Plan(PlanRequest{
		Model:            "example/model",
		Contexts:         []int{8192},
		Concurrency:      []int{1, 4},
		IncludeReference: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, '\n')
	goldenPath := filepath.Join("testdata", "default-sweep-8k.golden.json")
	if os.Getenv("UPDATE_GOLDEN") == "1" {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(goldenPath, got, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("generated spec differs from golden; run with UPDATE_GOLDEN=1 to update.\ngot:\n%s", got)
	}
}

func TestPlanRequiresModel(t *testing.T) {
	if _, err := Plan(PlanRequest{Contexts: []int{8192}}); err == nil || !strings.Contains(err.Error(), "model is required") {
		t.Fatalf("Plan error = %v, want model required", err)
	}
}
