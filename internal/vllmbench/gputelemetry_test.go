package vllmbench

import (
	"context"
	"testing"
)

func TestParseTegrastatsLine(t *testing.T) {
	line := "11-30-2026 12:00:00 RAM 41234/119896MB (lfb 8x4MB) SWAP 0/0MB CPU [1%@2000] GR3D_FREQ 47% VIC_FREQ 0% APE 200"
	sample, ok := parseTegrastatsLine(line)
	if !ok || sample.Source != "tegrastats" {
		t.Fatalf("sample = %+v ok=%t, want tegrastats sample", sample, ok)
	}
	if sample.GPUUtilizationPct == nil || *sample.GPUUtilizationPct != 47 {
		t.Fatalf("utilization = %v, want 47", sample.GPUUtilizationPct)
	}
	if sample.GPUMemoryUsedBytes == nil || *sample.GPUMemoryUsedBytes != 41234*1024*1024 {
		t.Fatalf("memory = %v, want 41234 MiB in bytes", sample.GPUMemoryUsedBytes)
	}
	// GB10-class output has RAM but no GR3D line.
	sample, ok = parseTegrastatsLine("RAM 38121/124547MB SWAP 3453/16384MB CPU [6%@2808]")
	if !ok || sample.GPUUtilizationPct != nil || sample.GPUMemoryUsedBytes == nil {
		t.Fatalf("sample = %+v ok=%t, want memory-only sample", sample, ok)
	}
	if _, ok := parseTegrastatsLine("no telemetry here"); ok {
		t.Fatal("unparseable line yielded a sample")
	}
}

func TestParseNvidiaSMISample(t *testing.T) {
	sample, ok := parseNvidiaSMISample("95, 1234\n")
	if !ok || sample.Source != "nvidia-smi" {
		t.Fatalf("sample = %+v ok=%t, want nvidia-smi sample", sample, ok)
	}
	if sample.GPUUtilizationPct == nil || *sample.GPUUtilizationPct != 95 {
		t.Fatalf("utilization = %v, want 95", sample.GPUUtilizationPct)
	}
	if sample.GPUMemoryUsedBytes == nil || *sample.GPUMemoryUsedBytes != 1234*1024*1024 {
		t.Fatalf("memory = %v, want 1234 MiB in bytes", sample.GPUMemoryUsedBytes)
	}
	// Unified-memory machines report utilization with memory N/A.
	sample, ok = parseNvidiaSMISample("95, [N/A]")
	if !ok || sample.GPUUtilizationPct == nil || sample.GPUMemoryUsedBytes != nil {
		t.Fatalf("sample = %+v ok=%t, want utilization-only sample", sample, ok)
	}
	if _, ok := parseNvidiaSMISample("[N/A], [N/A]"); ok {
		t.Fatal("all-N/A output yielded a sample")
	}
	if _, ok := parseNvidiaSMISample("garbage"); ok {
		t.Fatal("malformed output yielded a sample")
	}
}

func TestPollSamplesEmitsImmediatelyAndStopsOnCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	emitted := make(chan gpuTelemetrySample, 1)
	util := 12.0
	done := make(chan struct{})
	go func() {
		defer close(done)
		pollSamples(ctx, func(sample gpuTelemetrySample) { emitted <- sample }, func(context.Context) (gpuTelemetrySample, bool) {
			return gpuTelemetrySample{Source: "fake", GPUUtilizationPct: &util}, true
		})
	}()
	sample := <-emitted
	if sample.Source != "fake" || *sample.GPUUtilizationPct != 12 {
		t.Fatalf("sample = %+v, want immediate fake sample", sample)
	}
	cancel()
	<-done
}
