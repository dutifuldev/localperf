package vllmbench

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadCPUModel(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cpuinfo")
	if err := os.WriteFile(path, []byte("processor\t: 0\nmodel name\t: Fictional CPU X1\nflags\t: fpu\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := readCPUModel(path); got != "Fictional CPU X1" {
		t.Fatalf("cpu model = %q, want Fictional CPU X1", got)
	}
	if got := readCPUModel(filepath.Join(t.TempDir(), "missing")); got != "" {
		t.Fatalf("cpu model for missing file = %q, want empty", got)
	}
}

func TestParseMiBField(t *testing.T) {
	if got := parseMiBField(" 121850 MiB"); got < 118 || got > 120 {
		t.Fatalf("parseMiBField = %f, want ~119 GiB", got)
	}
	if got := parseMiBField("[N/A]"); got != 0 {
		t.Fatalf("parseMiBField N/A = %f, want 0", got)
	}
}
