package vllmbench

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

type resultSampleAdapter func(map[string]json.RawMessage) ([]RequestSample, bool, error)

type vllmBenchSampleData struct {
	inputLens  []int
	outputLens []int
	ttfts      []float64
	itls       [][]float64
	startTimes []float64
	completed  int
	failed     int
	baseTime   time.Time
}

func requestSamplesFromResultData(data []byte) ([]RequestSample, error) {
	if len(data) == 0 || data[0] != '{' {
		return nil, nil
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, nil
	}
	for _, adapter := range []resultSampleAdapter{
		localPerfHTTPSampleAdapter,
		vllmBenchSampleAdapter,
	} {
		samples, handled, err := adapter(raw)
		if handled || err != nil {
			return samples, err
		}
	}
	return nil, nil
}

func localPerfHTTPSampleAdapter(raw map[string]json.RawMessage) ([]RequestSample, bool, error) {
	samplesRaw, ok := raw["request_samples"]
	if !ok {
		return nil, false, nil
	}
	var samples []RequestSample
	if err := json.Unmarshal(samplesRaw, &samples); err != nil {
		return nil, true, err
	}
	return samples, true, nil
}

func vllmBenchSampleAdapter(raw map[string]json.RawMessage) ([]RequestSample, bool, error) {
	if !isVLLMBenchSampleResult(raw) {
		return nil, false, nil
	}
	data, err := readVLLMBenchSampleData(raw)
	if err != nil {
		return nil, true, err
	}
	return data.samples(), true, nil
}

func isVLLMBenchSampleResult(raw map[string]json.RawMessage) bool {
	_, hasTTFTs := raw["ttfts"]
	_, hasOutputLens := raw["output_lens"]
	return hasTTFTs && hasOutputLens
}

func readVLLMBenchSampleData(raw map[string]json.RawMessage) (vllmBenchSampleData, error) {
	inputLens, _, err := intArrayField(raw, "input_lens")
	if err != nil {
		return vllmBenchSampleData{}, err
	}
	outputLens, _, err := intArrayField(raw, "output_lens")
	if err != nil {
		return vllmBenchSampleData{}, err
	}
	ttfts, _, err := floatArrayField(raw, "ttfts")
	if err != nil {
		return vllmBenchSampleData{}, err
	}
	itls, _, err := nestedFloatArrayField(raw, "itls")
	if err != nil {
		return vllmBenchSampleData{}, err
	}
	startTimes, _, err := floatArrayField(raw, "start_times")
	if err != nil {
		return vllmBenchSampleData{}, err
	}
	completed, _ := intScalarField(raw, "completed")
	failed, _ := intScalarField(raw, "failed")
	return vllmBenchSampleData{
		inputLens:  inputLens,
		outputLens: outputLens,
		ttfts:      ttfts,
		itls:       itls,
		startTimes: startTimes,
		completed:  completed,
		failed:     failed,
		baseTime:   resultDate(raw),
	}, nil
}

func (data vllmBenchSampleData) samples() []RequestSample {
	count := data.sampleCount()
	if count == 0 {
		return nil
	}
	samples := make([]RequestSample, 0, count)
	minStart, hasStart := minFinite(data.startTimes)
	for i := 0; i < count; i++ {
		samples = append(samples, data.sample(i, minStart, hasStart))
	}
	return samples
}

func (data vllmBenchSampleData) sampleCount() int {
	return maxInts(
		len(data.inputLens),
		len(data.outputLens),
		len(data.ttfts),
		len(data.itls),
		len(data.startTimes),
		data.completed+data.failed,
	)
}

func (data vllmBenchSampleData) sample(index int, minStart float64, hasStart bool) RequestSample {
	promptTokens := intAt(data.inputLens, index)
	completionTokens := intAt(data.outputLens, index)
	startedAt := derivedStartTime(data.baseTime, data.startTimes, index, minStart, hasStart)
	sample := RequestSample{
		RequestIndex:     index,
		RequestID:        fmt.Sprintf("vllm-bench-%d", index),
		Status:           vllmBenchRequestStatus(index, data.completed, completionTokens),
		Streamed:         true,
		StartedAt:        startedAt,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      promptTokens + completionTokens,
		ResponseMetadata: vllmBenchResponseMetadata(),
	}
	return data.applySampleTimings(sample, index)
}

func (data vllmBenchSampleData) applySampleTimings(sample RequestSample, index int) RequestSample {
	ttftSeconds, hasTTFT := finiteAt(data.ttfts, index)
	itlValues := floatSliceAt(data.itls, index)
	latencySeconds := sampleLatencySeconds(ttftSeconds, hasTTFT, itlValues)
	if hasTTFT {
		sample.applyTTFT(ttftSeconds)
	}
	if len(itlValues) > 0 {
		sample.ITLMeanMillis = meanFinite(itlValues) * 1000
	}
	if sample.CompletionTokens > 1 && latencySeconds > ttftSeconds {
		sample.TPOTMillis = ((latencySeconds - ttftSeconds) / float64(sample.CompletionTokens-1)) * 1000
	}
	if latencySeconds > 0 {
		sample.applyLatency(latencySeconds)
	}
	return sample
}

func sampleLatencySeconds(ttftSeconds float64, hasTTFT bool, itlValues []float64) float64 {
	latencySeconds := sumFinite(itlValues)
	if hasTTFT {
		latencySeconds += ttftSeconds
	}
	return latencySeconds
}

func (sample *RequestSample) applyTTFT(ttftSeconds float64) {
	sample.TTFTMillis = ttftSeconds * 1000
	sample.FirstByteMillis = sample.TTFTMillis
	firstByteAt := sample.StartedAt.Add(secondsDuration(ttftSeconds))
	sample.FirstByteAt = &firstByteAt
}

func (sample *RequestSample) applyLatency(latencySeconds float64) {
	sample.LatencyMillis = latencySeconds * 1000
	completedAt := sample.StartedAt.Add(secondsDuration(latencySeconds))
	sample.CompletedAt = &completedAt
	sample.deriveThroughput()
}

func vllmBenchResponseMetadata() map[string]any {
	return map[string]any{
		"source":           "vllm_bench",
		"timing_source":    "client_observed",
		"timestamp_source": "derived_from_result_date_and_start_times",
	}
}

func vllmBenchRequestStatus(index, completed, outputTokens int) string {
	if completed > 0 {
		if index < completed {
			return "completed"
		}
		return "failed"
	}
	if outputTokens > 0 {
		return "completed"
	}
	return "failed"
}

func intArrayField(raw map[string]json.RawMessage, key string) ([]int, bool, error) {
	value, ok := raw[key]
	if !ok {
		return nil, false, nil
	}
	var out []int
	if err := json.Unmarshal(value, &out); err != nil {
		return nil, true, fmt.Errorf("%s: %w", key, err)
	}
	return out, true, nil
}

func floatArrayField(raw map[string]json.RawMessage, key string) ([]float64, bool, error) {
	value, ok := raw[key]
	if !ok {
		return nil, false, nil
	}
	var out []float64
	if err := json.Unmarshal(value, &out); err != nil {
		return nil, true, fmt.Errorf("%s: %w", key, err)
	}
	return out, true, nil
}

func nestedFloatArrayField(raw map[string]json.RawMessage, key string) ([][]float64, bool, error) {
	value, ok := raw[key]
	if !ok {
		return nil, false, nil
	}
	var out [][]float64
	if err := json.Unmarshal(value, &out); err != nil {
		return nil, true, fmt.Errorf("%s: %w", key, err)
	}
	return out, true, nil
}

func intScalarField(raw map[string]json.RawMessage, key string) (int, bool) {
	value, ok := raw[key]
	if !ok {
		return 0, false
	}
	var out int
	if err := json.Unmarshal(value, &out); err != nil {
		return 0, false
	}
	return out, true
}

func resultDate(raw map[string]json.RawMessage) time.Time {
	value, ok := raw["date"]
	if !ok {
		return time.Unix(0, 0).UTC()
	}
	var text string
	if err := json.Unmarshal(value, &text); err != nil {
		return time.Unix(0, 0).UTC()
	}
	text = strings.TrimSpace(text)
	for _, layout := range []string{
		"20060102-150405",
		time.RFC3339Nano,
		time.RFC3339,
		"2006-01-02 15:04:05.999999",
		"2006-01-02 15:04:05",
	} {
		if parsed, err := time.ParseInLocation(layout, text, time.UTC); err == nil {
			return parsed.UTC()
		}
	}
	return time.Unix(0, 0).UTC()
}

func derivedStartTime(base time.Time, startTimes []float64, index int, minStart float64, hasStart bool) time.Time {
	if hasStart {
		if value, ok := finiteAt(startTimes, index); ok {
			return base.Add(secondsDuration(value - minStart))
		}
	}
	return base.Add(time.Duration(index) * time.Millisecond)
}

func minFinite(values []float64) (float64, bool) {
	finite := make([]float64, 0, len(values))
	for _, value := range values {
		if isFinite(value) {
			finite = append(finite, value)
		}
	}
	if len(finite) == 0 {
		return 0, false
	}
	sort.Float64s(finite)
	return finite[0], true
}

func intAt(values []int, index int) int {
	if index < 0 || index >= len(values) {
		return 0
	}
	if values[index] < 0 {
		return 0
	}
	return values[index]
}

func finiteAt(values []float64, index int) (float64, bool) {
	if index < 0 || index >= len(values) {
		return 0, false
	}
	value := values[index]
	if !isFinite(value) || value < 0 {
		return 0, false
	}
	return value, true
}

func floatSliceAt(values [][]float64, index int) []float64 {
	if index < 0 || index >= len(values) {
		return nil
	}
	return values[index]
}

func sumFinite(values []float64) float64 {
	sum := 0.0
	for _, value := range values {
		if isFinite(value) && value > 0 {
			sum += value
		}
	}
	return sum
}

func meanFinite(values []float64) float64 {
	count := 0
	sum := 0.0
	for _, value := range values {
		if isFinite(value) && value >= 0 {
			sum += value
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func secondsDuration(seconds float64) time.Duration {
	return time.Duration(seconds * float64(time.Second))
}

func maxInts(values ...int) int {
	max := 0
	for _, value := range values {
		if value > max {
			max = value
		}
	}
	return max
}
