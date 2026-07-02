package vllmbench

import "testing"

func TestTegrastatsPatterns(t *testing.T) {
	line := "11-30-2026 12:00:00 RAM 41234/119896MB (lfb 8x4MB) SWAP 0/0MB CPU [1%@2000] GR3D_FREQ 47% VIC_FREQ 0% APE 200"
	utilMatch := tegraGR3DPattern.FindStringSubmatch(line)
	if utilMatch == nil || utilMatch[1] != "47" {
		t.Fatalf("GR3D match = %v, want 47", utilMatch)
	}
	ramMatch := tegraRAMPattern.FindStringSubmatch(line)
	if ramMatch == nil || ramMatch[1] != "41234" {
		t.Fatalf("RAM match = %v, want 41234", ramMatch)
	}
}
