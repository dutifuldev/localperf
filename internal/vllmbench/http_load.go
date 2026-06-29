package vllmbench

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptrace"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type HTTPBenchmarkResult struct {
	Date                          string          `json:"date"`
	LoadGenerator                 string          `json:"load_generator"`
	EndpointType                  string          `json:"endpoint_type"`
	Backend                       string          `json:"backend"`
	ModelID                       string          `json:"model_id"`
	TokenizerID                   string          `json:"tokenizer_id,omitempty"`
	NumPrompts                    int             `json:"num_prompts"`
	RequestRate                   string          `json:"request_rate"`
	MaxConcurrency                int             `json:"max_concurrency"`
	Duration                      float64         `json:"duration"`
	Completed                     int             `json:"completed"`
	Failed                        int             `json:"failed"`
	TotalInputTokens              int             `json:"total_input_tokens"`
	TotalOutputTokens             int             `json:"total_output_tokens"`
	TotalTokens                   int             `json:"total_tokens"`
	RequestThroughput             float64         `json:"request_throughput"`
	OutputThroughput              float64         `json:"output_throughput"`
	TotalTokenThroughput          float64         `json:"total_token_throughput"`
	RequestOutputThroughputMean   float64         `json:"request_output_throughput_mean,omitempty"`
	RequestOutputThroughputStdDev float64         `json:"request_output_throughput_stddev,omitempty"`
	RequestOutputThroughputMin    float64         `json:"request_output_throughput_min,omitempty"`
	RequestOutputThroughputP50    float64         `json:"request_output_throughput_p50,omitempty"`
	RequestOutputThroughputP95    float64         `json:"request_output_throughput_p95,omitempty"`
	RequestOutputThroughputP99    float64         `json:"request_output_throughput_p99,omitempty"`
	RequestOutputThroughputMax    float64         `json:"request_output_throughput_max,omitempty"`
	RequestTotalThroughputMean    float64         `json:"request_total_throughput_mean,omitempty"`
	RequestTotalThroughputStdDev  float64         `json:"request_total_throughput_stddev,omitempty"`
	MeanLatencyMillis             float64         `json:"mean_latency_ms,omitempty"`
	StdLatencyMillis              float64         `json:"std_latency_ms,omitempty"`
	P50LatencyMillis              float64         `json:"p50_latency_ms,omitempty"`
	P95LatencyMillis              float64         `json:"p95_latency_ms,omitempty"`
	P99LatencyMillis              float64         `json:"p99_latency_ms,omitempty"`
	RequestSamples                []RequestSample `json:"request_samples,omitempty"`
}

type RequestSample struct {
	RequestIndex          int            `json:"request_index"`
	RequestID             string         `json:"request_id,omitempty"`
	Status                string         `json:"status"`
	Streamed              bool           `json:"streamed"`
	HTTPStatusCode        int            `json:"http_status_code,omitempty"`
	StartedAt             time.Time      `json:"started_at"`
	FirstByteAt           *time.Time     `json:"first_byte_at,omitempty"`
	CompletedAt           *time.Time     `json:"completed_at,omitempty"`
	LatencyMillis         float64        `json:"latency_ms,omitempty"`
	FirstByteMillis       float64        `json:"first_byte_ms,omitempty"`
	PromptTokens          int            `json:"prompt_tokens,omitempty"`
	CompletionTokens      int            `json:"completion_tokens,omitempty"`
	TotalTokens           int            `json:"total_tokens,omitempty"`
	OutputTokensPerSecond float64        `json:"output_tokens_per_second,omitempty"`
	TotalTokensPerSecond  float64        `json:"total_tokens_per_second,omitempty"`
	PromptSHA256          string         `json:"prompt_sha256,omitempty"`
	ResponseSHA256        string         `json:"response_sha256,omitempty"`
	ErrorType             string         `json:"error_type,omitempty"`
	ErrorCode             string         `json:"error_code,omitempty"`
	ErrorMessage          string         `json:"error_message,omitempty"`
	ResponseMetadata      map[string]any `json:"response_metadata,omitempty"`
}

type localPerfHTTPJob struct {
	index   int
	request CanonicalRequest
}

type openAIHTTPClient struct {
	baseURL  string
	profile  Profile
	workload Workload
	client   *http.Client
}

type openAIResponse struct {
	ID      string         `json:"id,omitempty"`
	Choices []openAIChoice `json:"choices,omitempty"`
	Usage   openAIUsage    `json:"usage,omitempty"`
	Error   *openAIError   `json:"error,omitempty"`
}

type openAIChoice struct {
	Message      *openAIMessage `json:"message,omitempty"`
	Text         string         `json:"text,omitempty"`
	FinishReason string         `json:"finish_reason,omitempty"`
}

type openAIMessage struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens,omitempty"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`
}

type openAIError struct {
	Message string `json:"message,omitempty"`
	Type    string `json:"type,omitempty"`
	Code    any    `json:"code,omitempty"`
}

type numericStats struct {
	Count  int
	Mean   float64
	StdDev float64
	Min    float64
	P50    float64
	P90    float64
	P95    float64
	P99    float64
	Max    float64
}

func executeLocalPerfHTTPBench(ctx context.Context, spec Spec, planned PlannedRun, logPath string) (commandResult, error) {
	if err := prepareCommandPaths(LocalPerfHTTPCommand(spec, planned), logPath); err != nil {
		return commandResult{ExitCode: -1}, err
	}
	runCtx, cancel := context.WithTimeout(ctx, time.Duration(spec.Safety.WorkloadTimeoutSec)*time.Second)
	defer cancel()
	if _, err := checkMemoryFloor(spec.Safety.MinMemAvailableGiB); err != nil {
		_ = writeLocalPerfHTTPLog(logPath, nil, 0, nil, err)
		return commandResult{ExitCode: 1}, err
	}
	memoryMonitor := monitorMemoryFloor(runCtx, cancel, spec.Safety.MinMemAvailableGiB, time.Duration(spec.Safety.PollIntervalMillis)*time.Millisecond)
	start := time.Now()
	result, runErr := runLocalPerfHTTPBenchmark(runCtx, planned)
	duration := time.Since(start)
	cancel()
	memoryErr := <-memoryMonitor
	exitCode := 0
	if runErr != nil || memoryErr != nil || runCtx.Err() == context.DeadlineExceeded {
		exitCode = 1
	}
	if result != nil {
		if err := writeJSONFile(planned.ResultFile, result); err != nil && runErr == nil {
			runErr = err
			exitCode = 1
		}
	}
	if err := writeLocalPerfHTTPLog(logPath, result, duration, runErr, memoryErr); err != nil && runErr == nil {
		runErr = err
		exitCode = 1
	}
	commandResult := commandResult{Duration: duration, ExitCode: exitCode}
	if runCtx.Err() == context.DeadlineExceeded {
		return commandResult, fmt.Errorf("localperf_http benchmark timed out after %s", time.Duration(spec.Safety.WorkloadTimeoutSec)*time.Second)
	}
	if memoryErr != nil {
		return commandResult, memoryErr
	}
	return commandResult, runErr
}

func RunLocalPerfHTTPBench(ctx context.Context, spec Spec, planned PlannedRun, logPath string) error {
	if _, err := executeLocalPerfHTTPBench(ctx, spec, planned, logPath); err != nil {
		return err
	}
	rows, err := ParseResultFile(planned.ResultFile)
	if err != nil {
		return err
	}
	if len(rows) == 0 {
		return errors.New("localperf_http result file did not contain a parseable row")
	}
	if failed := failedRequestCount(rows); failed > 0 {
		return fmt.Errorf("localperf_http result reported %d failed request(s)", failed)
	}
	return nil
}

func runLocalPerfHTTPBenchmark(ctx context.Context, planned PlannedRun) (*HTTPBenchmarkResult, error) {
	requests, err := plannedRunHTTPRequests(ctx, planned)
	if err != nil {
		return nil, err
	}
	client := openAIHTTPClient{
		baseURL:  baseURL(planned.Profile),
		profile:  planned.Profile,
		workload: planned.Workload,
		client:   &http.Client{},
	}
	start := time.Now().UTC()
	samples, err := scheduleHTTPRequests(ctx, client, requests, planned)
	end := time.Now().UTC()
	result := buildHTTPBenchmarkResult(planned, samples, start, end)
	return result, err
}

func plannedRunHTTPRequests(ctx context.Context, planned PlannedRun) ([]CanonicalRequest, error) {
	if hasPreparedCanonicalDataset(planned.Workload) || hasStructuredDataset(planned.Workload) {
		return structuredHTTPRequests(planned)
	}
	if planned.Workload.DatasetName == "random" {
		return randomHTTPRequests(planned.Workload)
	}
	return nil, fmt.Errorf("localperf_http supports random and structured datasets, not dataset_name %q", planned.Workload.DatasetName)
}

func hasPreparedCanonicalDataset(workload Workload) bool {
	return strings.TrimSpace(workload.Dataset.Prepared.CanonicalPath) != ""
}

func structuredHTTPRequests(planned PlannedRun) ([]CanonicalRequest, error) {
	path := resolveResultPath(filepath.Dir(filepath.Dir(planned.ResultFile)), planned.Workload.Dataset.Prepared.CanonicalPath)
	requests, err := readCanonicalRequestFile(path)
	if err != nil {
		return nil, err
	}
	if len(requests) < planned.Workload.NumPrompts {
		return nil, fmt.Errorf("prepared dataset has %d request(s), workload needs %d", len(requests), planned.Workload.NumPrompts)
	}
	return requests[:planned.Workload.NumPrompts], nil
}

func randomHTTPRequests(workload Workload) ([]CanonicalRequest, error) {
	if workload.NumPrompts <= 0 {
		return nil, errors.New("num_prompts must be positive")
	}
	if workload.RandomOutputLen <= 0 {
		return nil, errors.New("random_output_len must be positive")
	}
	requests := make([]CanonicalRequest, 0, workload.NumPrompts)
	for i := 0; i < workload.NumPrompts; i++ {
		id := fmt.Sprintf("%s-%06d", datasetIDForWorkload(workload.Name), i+1)
		requests = append(requests, CanonicalRequest{
			ID:                   id,
			Ordinal:              i,
			DatasetID:            datasetIDForWorkload(workload.Name),
			SourceRecordID:       id,
			Mode:                 "chat",
			Messages:             []Message{{Role: "user", Content: syntheticPrompt(workload.RandomInputLen)}},
			MaxOutputTokens:      workload.RandomOutputLen,
			Temperature:          workload.Temperature,
			IgnoreEOS:            workload.IgnoreEOS,
			InputTokensExpected:  workload.RandomInputLen,
			OutputTokensExpected: workload.RandomOutputLen,
		})
	}
	return requests, nil
}

func scheduleHTTPRequests(ctx context.Context, client openAIHTTPClient, requests []CanonicalRequest, planned PlannedRun) ([]RequestSample, error) {
	concurrency := planned.Concurrency
	if concurrency <= 0 {
		concurrency = 1
	}
	delay, err := requestRateDelay(planned.Workload.RequestRate)
	if err != nil {
		return nil, err
	}
	jobs := make(chan localPerfHTTPJob)
	results := make(chan RequestSample, len(requests))
	var workers sync.WaitGroup
	for i := 0; i < concurrency; i++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for job := range jobs {
				results <- client.Invoke(ctx, job.index, job.request)
			}
		}()
	}
	feedErr := feedHTTPJobs(ctx, jobs, requests, delay)
	workers.Wait()
	close(results)
	samples := collectHTTPSamples(results)
	if feedErr != nil {
		return samples, feedErr
	}
	if err := ctx.Err(); err != nil {
		return samples, err
	}
	return samples, nil
}

func feedHTTPJobs(ctx context.Context, jobs chan<- localPerfHTTPJob, requests []CanonicalRequest, delay time.Duration) error {
	defer close(jobs)
	for i, request := range requests {
		if i > 0 && delay > 0 {
			if err := sleepContext(ctx, delay); err != nil {
				return err
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case jobs <- localPerfHTTPJob{index: i, request: request}:
		}
	}
	return nil
}

func collectHTTPSamples(results <-chan RequestSample) []RequestSample {
	var samples []RequestSample
	for sample := range results {
		samples = append(samples, sample)
	}
	sort.Slice(samples, func(i, j int) bool {
		return samples[i].RequestIndex < samples[j].RequestIndex
	})
	return samples
}

func requestRateDelay(value string) (time.Duration, error) {
	value = strings.TrimSpace(strings.ToLower(value))
	if value == "" || value == "inf" || value == "infinity" {
		return 0, nil
	}
	rate, err := strconv.ParseFloat(value, 64)
	if err != nil || rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
		return 0, fmt.Errorf("unsupported request_rate %q", value)
	}
	return time.Duration(float64(time.Second) / rate), nil
}

func sleepContext(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func (client openAIHTTPClient) Invoke(ctx context.Context, index int, request CanonicalRequest) RequestSample {
	started := time.Now().UTC()
	sample := RequestSample{
		RequestIndex: index,
		RequestID:    request.ID,
		Status:       "failed",
		Streamed:     false,
		StartedAt:    started,
		PromptSHA256: sha256Maybe(requestPromptText(request)),
	}
	body, endpoint, err := client.requestBody(request)
	if err != nil {
		return sample.withError("request_render", "", err.Error(), time.Now().UTC(), nil)
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return sample.withError("request_encode", "", err.Error(), time.Now().UTC(), nil)
	}
	var firstByteAt *time.Time
	trace := &httptrace.ClientTrace{GotFirstResponseByte: func() {
		now := time.Now().UTC()
		firstByteAt = &now
	}}
	req, err := http.NewRequestWithContext(httptrace.WithClientTrace(ctx, trace), http.MethodPost, client.baseURL+endpoint, bytes.NewReader(payload))
	if err != nil {
		return sample.withError("request_create", "", err.Error(), time.Now().UTC(), firstByteAt)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.client.Do(req)
	if err != nil {
		return sample.withError("request_send", "", err.Error(), time.Now().UTC(), firstByteAt)
	}
	defer resp.Body.Close()
	data, readErr := io.ReadAll(io.LimitReader(resp.Body, 256*1024*1024))
	completed := time.Now().UTC()
	sample.HTTPStatusCode = resp.StatusCode
	if readErr != nil {
		return sample.withError("response_read", "", readErr.Error(), completed, firstByteAt)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return sample.withError("http_status", fmt.Sprint(resp.StatusCode), strings.TrimSpace(string(data)), completed, firstByteAt)
	}
	var decoded openAIResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		return sample.withError("response_decode", "", err.Error(), completed, firstByteAt)
	}
	if decoded.Error != nil {
		return sample.withError(firstNonEmpty(decoded.Error.Type, "api_error"), fmt.Sprint(decoded.Error.Code), decoded.Error.Message, completed, firstByteAt)
	}
	return sample.withSuccess(request, decoded, data, completed, firstByteAt)
}

func (client openAIHTTPClient) requestBody(request CanonicalRequest) (map[string]any, string, error) {
	maxTokens := firstNonZeroInt(request.MaxOutputTokens, request.OutputTokensExpected, client.workload.RandomOutputLen)
	if maxTokens <= 0 {
		return nil, "", fmt.Errorf("request %s missing max output tokens", request.ID)
	}
	body := map[string]any{
		"model":      client.profile.Model,
		"max_tokens": maxTokens,
		"stream":     false,
	}
	if request.Temperature != nil {
		body["temperature"] = *request.Temperature
	} else if client.workload.Temperature != nil {
		body["temperature"] = *client.workload.Temperature
	}
	if request.IgnoreEOS || client.workload.IgnoreEOS {
		body["ignore_eos"] = true
	}
	backend := client.requestBackend(request)
	endpoint := strings.TrimSpace(client.workload.Endpoint)
	if backend == "openai" {
		body["prompt"] = firstNonEmpty(request.Prompt, messagesPrompt(request.Messages))
		if endpoint == "" {
			endpoint = defaultEndpoint("openai")
		}
		if err := mergeExtraBody(body, client.workload.ExtraBody); err != nil {
			return nil, "", err
		}
		return body, endpoint, nil
	}
	messages := request.Messages
	if len(messages) == 0 {
		prompt := firstNonEmpty(request.Prompt, messagesPrompt(request.Messages))
		if strings.TrimSpace(prompt) == "" {
			return nil, "", fmt.Errorf("request %s has no prompt or messages", request.ID)
		}
		messages = []Message{{Role: "user", Content: prompt}}
	}
	body["messages"] = messages
	if endpoint == "" {
		endpoint = defaultEndpoint("openai-chat")
	}
	if err := mergeExtraBody(body, client.workload.ExtraBody); err != nil {
		return nil, "", err
	}
	return body, endpoint, nil
}

func (client openAIHTTPClient) requestBackend(request CanonicalRequest) string {
	if strings.TrimSpace(request.Mode) != "" {
		return requestModeBackendFallback(request.Mode)
	}
	return firstNonEmpty(client.workload.Backend, "openai-chat")
}

func mergeExtraBody(body map[string]any, raw string) error {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	decoder := json.NewDecoder(strings.NewReader(raw))
	decoder.UseNumber()
	extra := map[string]any{}
	if err := decoder.Decode(&extra); err != nil {
		return fmt.Errorf("invalid extra_body: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return fmt.Errorf("invalid extra_body: extra content after JSON object")
	}
	for key, value := range extra {
		body[key] = value
	}
	return nil
}

func requestModeBackendFallback(mode string) string {
	backend, ok := requestModeBackend(mode)
	if !ok {
		return "openai-chat"
	}
	return backend
}

func (sample RequestSample) withSuccess(request CanonicalRequest, response openAIResponse, data []byte, completed time.Time, firstByteAt *time.Time) RequestSample {
	sample.Status = "completed"
	sample.CompletedAt = &completed
	sample.FirstByteAt = firstByteAt
	sample.LatencyMillis = completed.Sub(sample.StartedAt).Seconds() * 1000
	if firstByteAt != nil {
		sample.FirstByteMillis = firstByteAt.Sub(sample.StartedAt).Seconds() * 1000
	}
	sample.PromptTokens = firstNonZeroInt(response.Usage.PromptTokens, request.InputTokensExpected)
	sample.CompletionTokens = firstNonZeroInt(response.Usage.CompletionTokens, request.OutputTokensExpected)
	sample.TotalTokens = firstNonZeroInt(response.Usage.TotalTokens, sample.PromptTokens+sample.CompletionTokens)
	sample.ResponseSHA256 = sha256Hex(data)
	sample.ResponseMetadata = responseMetadata(response)
	sample.deriveThroughput()
	return sample
}

func (sample RequestSample) withError(errorType, errorCode, message string, completed time.Time, firstByteAt *time.Time) RequestSample {
	sample.CompletedAt = &completed
	sample.FirstByteAt = firstByteAt
	sample.LatencyMillis = completed.Sub(sample.StartedAt).Seconds() * 1000
	if firstByteAt != nil {
		sample.FirstByteMillis = firstByteAt.Sub(sample.StartedAt).Seconds() * 1000
	}
	sample.ErrorType = errorType
	sample.ErrorCode = errorCode
	sample.ErrorMessage = message
	return sample
}

func (sample *RequestSample) deriveThroughput() {
	if sample.LatencyMillis <= 0 {
		return
	}
	seconds := sample.LatencyMillis / 1000
	if sample.CompletionTokens > 0 {
		sample.OutputTokensPerSecond = float64(sample.CompletionTokens) / seconds
	}
	if sample.TotalTokens > 0 {
		sample.TotalTokensPerSecond = float64(sample.TotalTokens) / seconds
	}
}

func responseMetadata(response openAIResponse) map[string]any {
	out := map[string]any{}
	if response.ID != "" {
		out["id"] = response.ID
	}
	if len(response.Choices) > 0 && response.Choices[0].FinishReason != "" {
		out["finish_reason"] = response.Choices[0].FinishReason
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func requestPromptText(request CanonicalRequest) string {
	if strings.TrimSpace(request.Prompt) != "" {
		return request.Prompt
	}
	return messagesPrompt(request.Messages)
}

func buildHTTPBenchmarkResult(planned PlannedRun, samples []RequestSample, started, completed time.Time) *HTTPBenchmarkResult {
	duration := completed.Sub(started).Seconds()
	result := &HTTPBenchmarkResult{
		Date:           completed.Format("20060102-150405"),
		LoadGenerator:  LoadGeneratorLocalPerfHTTP,
		EndpointType:   planned.Workload.Backend,
		Backend:        planned.Workload.Backend,
		ModelID:        planned.Profile.Model,
		TokenizerID:    planned.Profile.Model,
		NumPrompts:     planned.Workload.NumPrompts,
		RequestRate:    planned.Workload.RequestRate,
		MaxConcurrency: planned.Concurrency,
		Duration:       duration,
		RequestSamples: samples,
	}
	for _, sample := range samples {
		if sample.Status == "completed" {
			result.Completed++
			result.TotalInputTokens += sample.PromptTokens
			result.TotalOutputTokens += sample.CompletionTokens
			result.TotalTokens += sample.TotalTokens
			continue
		}
		result.Failed++
	}
	if duration > 0 {
		result.RequestThroughput = float64(result.Completed) / duration
		result.OutputThroughput = float64(result.TotalOutputTokens) / duration
		result.TotalTokenThroughput = float64(result.TotalTokens) / duration
	}
	applyRequestStats(result, samples)
	return result
}

func applyRequestStats(result *HTTPBenchmarkResult, samples []RequestSample) {
	outputStats := statsFromSamples(samples, func(sample RequestSample) float64 { return sample.OutputTokensPerSecond })
	totalStats := statsFromSamples(samples, func(sample RequestSample) float64 { return sample.TotalTokensPerSecond })
	latencyStats := statsFromSamples(samples, func(sample RequestSample) float64 { return sample.LatencyMillis })
	result.RequestOutputThroughputMean = outputStats.Mean
	result.RequestOutputThroughputStdDev = outputStats.StdDev
	result.RequestOutputThroughputMin = outputStats.Min
	result.RequestOutputThroughputP50 = outputStats.P50
	result.RequestOutputThroughputP95 = outputStats.P95
	result.RequestOutputThroughputP99 = outputStats.P99
	result.RequestOutputThroughputMax = outputStats.Max
	result.RequestTotalThroughputMean = totalStats.Mean
	result.RequestTotalThroughputStdDev = totalStats.StdDev
	result.MeanLatencyMillis = latencyStats.Mean
	result.StdLatencyMillis = latencyStats.StdDev
	result.P50LatencyMillis = latencyStats.P50
	result.P95LatencyMillis = latencyStats.P95
	result.P99LatencyMillis = latencyStats.P99
}

func statsFromSamples(samples []RequestSample, value func(RequestSample) float64) numericStats {
	values := make([]float64, 0, len(samples))
	for _, sample := range samples {
		if sample.Status != "completed" {
			continue
		}
		if v := value(sample); v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0) {
			values = append(values, v)
		}
	}
	return numericStatsFromValues(values)
}

func numericStatsFromValues(values []float64) numericStats {
	if len(values) == 0 {
		return numericStats{}
	}
	sort.Float64s(values)
	sum := 0.0
	for _, value := range values {
		sum += value
	}
	mean := sum / float64(len(values))
	varianceSum := 0.0
	for _, value := range values {
		delta := value - mean
		varianceSum += delta * delta
	}
	stddev := 0.0
	if len(values) > 1 {
		stddev = math.Sqrt(varianceSum / float64(len(values)-1))
	}
	return numericStats{
		Count:  len(values),
		Mean:   mean,
		StdDev: stddev,
		Min:    values[0],
		P50:    percentile(values, 50),
		P90:    percentile(values, 90),
		P95:    percentile(values, 95),
		P99:    percentile(values, 99),
		Max:    values[len(values)-1],
	}
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if len(sorted) == 1 {
		return sorted[0]
	}
	rank := (p / 100) * float64(len(sorted)-1)
	lower := int(math.Floor(rank))
	upper := int(math.Ceil(rank))
	if lower == upper {
		return sorted[lower]
	}
	weight := rank - float64(lower)
	return sorted[lower]*(1-weight) + sorted[upper]*weight
}

func writeLocalPerfHTTPLog(logPath string, result *HTTPBenchmarkResult, duration time.Duration, runErr, memoryErr error) error {
	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		return err
	}
	log := map[string]any{
		"load_generator":   LoadGeneratorLocalPerfHTTP,
		"duration_seconds": duration.Seconds(),
	}
	if result != nil {
		log["completed"] = result.Completed
		log["failed"] = result.Failed
		log["output_throughput"] = result.OutputThroughput
		log["request_output_throughput_stddev"] = result.RequestOutputThroughputStdDev
	}
	if runErr != nil {
		log["error"] = runErr.Error()
	}
	if memoryErr != nil {
		log["memory_error"] = memoryErr.Error()
	}
	data, err := json.MarshalIndent(log, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(logPath, data, 0o644)
}
