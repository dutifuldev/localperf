package vllmbench

import (
	"bufio"
	"context"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// GPU telemetry sampling during measurement phases. Source preference per
// docs/2026-06-23-measurement-methods.md: tegrastats on unified-memory
// systems, then nvidia-smi. Samples are emitted as run events and ingested
// into the telemetry tables tagged with the measurement; series names follow
// docs/2026-06-29-sqlite-run-artifact-format.md.
const gpuTelemetryInterval = 2 * time.Second

type gpuTelemetrySample struct {
	Source             string   `json:"source"`
	GPUUtilizationPct  *float64 `json:"gpu_utilization_percent,omitempty"`
	GPUMemoryUsedBytes *float64 `json:"gpu_memory_used_bytes,omitempty"`
}

type gpuTelemetrySampler struct {
	cancel context.CancelFunc
	done   chan struct{}
}

// startGPUTelemetrySampler samples GPU utilization and memory during one
// measurement, emitting gpu_telemetry events tagged with the planned run so
// artifact ingestion can attach them to the measurement. Returns nil when no
// telemetry source is available; recording nothing is honest, substituting
// nothing is not.
func startGPUTelemetrySampler(ctx context.Context, events *eventWriter, planned PlannedRun) *gpuTelemetrySampler {
	sampleCtx, cancel := context.WithCancel(ctx)
	sampler := &gpuTelemetrySampler{cancel: cancel, done: make(chan struct{})}
	emit := func(sample gpuTelemetrySample) {
		events.Write(Event{
			Timestamp:   time.Now().UTC(),
			Type:        "gpu_telemetry",
			Profile:     planned.Profile.Name,
			Workload:    planned.Workload.Name,
			Concurrency: planned.Concurrency,
			Repeat:      planned.Repeat,
			Details:     mustJSON(sample),
		})
	}
	switch {
	case commandAvailable("tegrastats"):
		go func() {
			defer close(sampler.done)
			sampleTegrastats(sampleCtx, emit)
		}()
	case commandAvailable("nvidia-smi"):
		go func() {
			defer close(sampler.done)
			pollNvidiaSMI(sampleCtx, emit)
		}()
	default:
		cancel()
		close(sampler.done)
		return nil
	}
	return sampler
}

func (sampler *gpuTelemetrySampler) Stop() {
	if sampler == nil {
		return
	}
	sampler.cancel()
	<-sampler.done
}

func commandAvailable(name string) bool {
	_, err := exec.LookPath(name)
	return err == nil
}

var (
	tegraGR3DPattern = regexp.MustCompile(`GR3D_FREQ (\d+)%`)
	tegraRAMPattern  = regexp.MustCompile(`RAM (\d+)/(\d+)MB`)
)

// sampleTegrastats streams `tegrastats --interval` output; each line yields
// one sample. On unified-memory systems the RAM line is the closest
// available GPU-memory signal and is recorded under the tegrastats source so
// readers can judge it.
func sampleTegrastats(ctx context.Context, emit func(gpuTelemetrySample)) {
	cmd := exec.CommandContext(ctx, "tegrastats", "--interval", strconv.Itoa(int(gpuTelemetryInterval.Milliseconds())))
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return
	}
	if err := cmd.Start(); err != nil {
		return
	}
	defer func() { _ = cmd.Wait() }()
	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		sample := gpuTelemetrySample{Source: "tegrastats"}
		if match := tegraGR3DPattern.FindStringSubmatch(scanner.Text()); match != nil {
			if value, err := strconv.ParseFloat(match[1], 64); err == nil {
				sample.GPUUtilizationPct = &value
			}
		}
		if match := tegraRAMPattern.FindStringSubmatch(scanner.Text()); match != nil {
			if value, err := strconv.ParseFloat(match[1], 64); err == nil {
				bytes := value * 1024 * 1024
				sample.GPUMemoryUsedBytes = &bytes
			}
		}
		if sample.GPUUtilizationPct != nil || sample.GPUMemoryUsedBytes != nil {
			emit(sample)
		}
	}
}

func pollNvidiaSMI(ctx context.Context, emit func(gpuTelemetrySample)) {
	ticker := time.NewTicker(gpuTelemetryInterval)
	defer ticker.Stop()
	for {
		if sample, ok := queryNvidiaSMI(ctx); ok {
			emit(sample)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func queryNvidiaSMI(ctx context.Context) (gpuTelemetrySample, bool) {
	queryCtx, cancel := context.WithTimeout(ctx, gpuTelemetryInterval)
	defer cancel()
	output, err := exec.CommandContext(queryCtx, "nvidia-smi",
		"--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return gpuTelemetrySample{}, false
	}
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(lines) == 0 {
		return gpuTelemetrySample{}, false
	}
	// First GPU only; multi-GPU inventories are in host_json and per-GPU
	// series can be added when a multi-GPU machine needs them.
	fields := strings.Split(lines[0], ",")
	if len(fields) < 2 {
		return gpuTelemetrySample{}, false
	}
	sample := gpuTelemetrySample{Source: "nvidia-smi"}
	if value, err := strconv.ParseFloat(strings.TrimSpace(fields[0]), 64); err == nil {
		sample.GPUUtilizationPct = &value
	}
	if value, err := strconv.ParseFloat(strings.TrimSpace(fields[1]), 64); err == nil {
		bytes := value * 1024 * 1024
		sample.GPUMemoryUsedBytes = &bytes
	}
	if sample.GPUUtilizationPct == nil && sample.GPUMemoryUsedBytes == nil {
		return gpuTelemetrySample{}, false
	}
	return sample, true
}
