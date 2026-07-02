package report

import (
	"database/sql"
	"fmt"
	"math"
	"strings"
	"time"
)

// MetricDef defines one reported quantity exactly once: the report legend
// and the computation both come from this registry, so the definition shown
// to the reader cannot drift from what is rendered. See
// docs/2026-07-02-reporting-completeness-plan.md.
type MetricDef struct {
	Key        string
	Label      string
	Unit       string
	Weighting  string
	Definition string
}

var ReportMetrics = []MetricDef{
	{
		Key: "decode_tok_s", Label: "decode tok/s", Unit: "tok/s", Weighting: "aggregate",
		Definition: "generated output tokens divided by full run wall time.",
	},
	{
		Key: "prefill_tok_s", Label: "prefill tok/s", Unit: "tok/s", Weighting: "aggregate",
		Definition: "input prompt tokens divided by full run wall time; a workload-level approximation, not engine kernel time.",
	},
	{
		Key: "per_user_tok_s", Label: "/user", Unit: "tok/s", Weighting: "per-request",
		Definition: "row throughput divided by concurrent users.",
	},
	{
		Key: "rps", Label: "RPS", Unit: "req/s", Weighting: "aggregate",
		Definition: "completed requests divided by full run wall time.",
	},
	{
		Key: "ttft", Label: "TTFT", Unit: "ms", Weighting: "per-request",
		Definition: "time to first token; mean, p50, p95, and p99 over completed requests.",
	},
	{
		Key: "latency", Label: "latency", Unit: "ms", Weighting: "per-request",
		Definition: "end-to-end request latency (E2EL); p50, p95, and p99 over completed requests.",
	},
	{
		Key: "tpot", Label: "TPOT", Unit: "ms", Weighting: "per-request",
		Definition: "request-weighted mean of per-request time per output token; every request counts equally, describing per-user experience.",
	},
	{
		Key: "itl_token_weighted", Label: "ITL (tok-wt)", Unit: "ms", Weighting: "per-token",
		Definition: "token-weighted mean inter-token gap: sum of all gaps divided by total gaps across requests; longer responses weigh more, describing steady-state system behavior. Not interchangeable with TPOT.",
	},
	{
		Key: "achieved_concurrency", Label: "achieved concurrency", Unit: "requests", Weighting: "aggregate",
		Definition: "time-weighted mean in-flight requests (total request busy time / measurement span); shown when it diverges from the requested concurrency by more than 10%.",
	},
	{
		Key: "ok_err", Label: "OK / Err", Unit: "requests", Weighting: "aggregate",
		Definition: "completed requests / failed requests, with failures broken down by error type.",
	},
	{
		Key: "gpu_util", Label: "GPU util a/p", Unit: "percent", Weighting: "aggregate",
		Definition: "average / peak GPU utilization sampled every 2s during the measurement, with the telemetry source named; on unified-memory systems cross-check GPU memory against system memory drop.",
	},
	{
		Key: "gpu_mem_peak", Label: "GPU mem peak", Unit: "GiB", Weighting: "aggregate",
		Definition: "peak sampled GPU memory used during the measurement.",
	},
	{
		Key: "prefix_cache", Label: "prefix cache", Unit: "", Weighting: "",
		Definition: "whether engine prefix caching was enabled (on / off / unknown). When on, prefill throughput can be partly served from cache and must be read accordingly.",
	},
	{
		Key: "repeats", Label: "± spread", Unit: "", Weighting: "across repeats",
		Definition: "when a point ran multiple repeats, values render as mean ± sample standard deviation across repeats and per-repeat rows move to the Repeats section.",
	},
	{
		Key: "context_labels", Label: "context labels", Unit: "tokens", Weighting: "",
		Definition: "declared active-context targets confirmed by measured tokens, or the measured request shape. \"Server limit\" is the engine's max_model_len; it is capacity, not the context a workload exercised.",
	},
}

// sqliteRequestDerived carries per-measurement metrics derived from the
// requests table at render time. Absent when the measurement was recorded
// without detailed request rows; the fallback is "-", never a substitute
// number.
type sqliteRequestDerived struct {
	ITLTokenWeightedMS   float64
	ITLTokenWeightedOK   bool
	AchievedConcurrency  float64
	AchievedConcurrencyK bool
	FailureBreakdown     string
	GPUUtilAvg           float64
	GPUUtilPeak          float64
	GPUUtilKnown         bool
	GPUUtilSource        string
	GPUMemPeakBytes      float64
	GPUMemKnown          bool
}

func loadSQLiteReportRequestDerived(db *sql.DB, doc *SQLiteReportDocument) error {
	doc.RequestDerived = map[int64]sqliteRequestDerived{}
	hasITL, err := sqliteRequestTableHasColumn(db, "itl_mean_ms")
	if err != nil {
		return err
	}
	if hasITL {
		if err := loadTokenWeightedITL(db, doc); err != nil {
			return err
		}
	}
	if err := loadAchievedConcurrency(db, doc); err != nil {
		return err
	}
	if err := loadGPUTelemetryStats(db, doc); err != nil {
		return err
	}
	return loadFailureBreakdown(db, doc)
}

// loadGPUTelemetryStats aggregates sampled GPU telemetry per measurement;
// the source (tegrastats, nvidia-smi, nvml) is kept so readers can judge the
// signal, which matters on unified-memory systems.
func loadGPUTelemetryStats(db *sql.DB, doc *SQLiteReportDocument) error {
	rows, err := db.Query(`SELECT ts.measurement_id, s.metric, s.source, AVG(ts.value), MAX(ts.value)
		FROM telemetry_samples ts
		JOIN telemetry_series s ON s.id = ts.series_id
		WHERE ts.measurement_id IS NOT NULL
		  AND s.metric IN ('gpu_utilization_percent', 'gpu_memory_used_bytes')
		GROUP BY ts.measurement_id, s.metric, s.source`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var measurementID int64
		var metric, source string
		var avg, peak float64
		if err := rows.Scan(&measurementID, &metric, &source, &avg, &peak); err != nil {
			return err
		}
		derived := doc.RequestDerived[measurementID]
		switch metric {
		case "gpu_utilization_percent":
			derived.GPUUtilAvg = avg
			derived.GPUUtilPeak = peak
			derived.GPUUtilKnown = true
			derived.GPUUtilSource = source
		case "gpu_memory_used_bytes":
			derived.GPUMemPeakBytes = peak
			derived.GPUMemKnown = true
		}
		doc.RequestDerived[measurementID] = derived
	}
	return rows.Err()
}

// loadTokenWeightedITL computes sum(all gaps)/count(all gaps) exactly from
// per-request means: itl_mean_ms * (completion_tokens - 1) recovers each
// request's gap sum.
func loadTokenWeightedITL(db *sql.DB, doc *SQLiteReportDocument) error {
	rows, err := db.Query(`SELECT measurement_id,
		SUM(CASE WHEN completion_tokens > 1 AND itl_mean_ms IS NOT NULL THEN itl_mean_ms * (completion_tokens - 1) END),
		SUM(CASE WHEN completion_tokens > 1 AND itl_mean_ms IS NOT NULL THEN completion_tokens - 1 END)
		FROM requests WHERE status = 'completed' GROUP BY measurement_id`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var measurementID int64
		var gapSum, gapCount sql.NullFloat64
		if err := rows.Scan(&measurementID, &gapSum, &gapCount); err != nil {
			return err
		}
		if !gapSum.Valid || !gapCount.Valid || gapCount.Float64 <= 0 {
			continue
		}
		derived := doc.RequestDerived[measurementID]
		derived.ITLTokenWeightedMS = gapSum.Float64 / gapCount.Float64
		derived.ITLTokenWeightedOK = true
		doc.RequestDerived[measurementID] = derived
	}
	return rows.Err()
}

// loadAchievedConcurrency derives the time-weighted mean number of in-flight
// requests: the integral of in-flight count over the measurement span equals
// the sum of request durations, so achieved = sum(durations) / span.
func loadAchievedConcurrency(db *sql.DB, doc *SQLiteReportDocument) error {
	rows, err := db.Query(`SELECT measurement_id, started_at, completed_at
		FROM requests
		WHERE status = 'completed' AND started_at IS NOT NULL AND completed_at IS NOT NULL`)
	if err != nil {
		return err
	}
	defer rows.Close()
	type window struct {
		busy       time.Duration
		start, end time.Time
		count      int
	}
	windows := map[int64]*window{}
	for rows.Next() {
		var measurementID int64
		var startedAt, completedAt string
		if err := rows.Scan(&measurementID, &startedAt, &completedAt); err != nil {
			return err
		}
		started, err := parseRequestTime(startedAt)
		if err != nil {
			continue
		}
		completed, err := parseRequestTime(completedAt)
		if err != nil || completed.Before(started) {
			continue
		}
		current := windows[measurementID]
		if current == nil {
			current = &window{start: started, end: completed}
			windows[measurementID] = current
		}
		if started.Before(current.start) {
			current.start = started
		}
		if completed.After(current.end) {
			current.end = completed
		}
		current.busy += completed.Sub(started)
		current.count++
	}
	if err := rows.Err(); err != nil {
		return err
	}
	for measurementID, current := range windows {
		span := current.end.Sub(current.start)
		if span <= 0 || current.count == 0 {
			continue
		}
		derived := doc.RequestDerived[measurementID]
		derived.AchievedConcurrency = float64(current.busy) / float64(span)
		derived.AchievedConcurrencyK = true
		doc.RequestDerived[measurementID] = derived
	}
	return nil
}

func parseRequestTime(value string) (time.Time, error) {
	if parsed, err := time.Parse(time.RFC3339Nano, value); err == nil {
		return parsed, nil
	}
	return time.Parse("2006-01-02 15:04:05.999999999-07:00", value)
}

func loadFailureBreakdown(db *sql.DB, doc *SQLiteReportDocument) error {
	rows, err := db.Query(`SELECT measurement_id, COALESCE(NULLIF(error_type, ''), 'unknown'), COUNT(*)
		FROM requests WHERE status != 'completed' GROUP BY 1, 2 ORDER BY 3 DESC, 2`)
	if err != nil {
		return err
	}
	defer rows.Close()
	parts := map[int64][]string{}
	for rows.Next() {
		var measurementID int64
		var errorType string
		var count int
		if err := rows.Scan(&measurementID, &errorType, &count); err != nil {
			return err
		}
		parts[measurementID] = append(parts[measurementID], fmt.Sprintf("%d %s", count, errorType))
	}
	if err := rows.Err(); err != nil {
		return err
	}
	for measurementID, breakdown := range parts {
		derived := doc.RequestDerived[measurementID]
		derived.FailureBreakdown = strings.Join(breakdown, ", ")
		doc.RequestDerived[measurementID] = derived
	}
	return nil
}

func applyRequestDerived(measurement *SQLiteReportMeasurement, derived sqliteRequestDerived) {
	if derived.ITLTokenWeightedOK {
		measurement.ITLTokenWeightedMS = displayFloat(derived.ITLTokenWeightedMS)
	} else {
		measurement.ITLTokenWeightedMS = "-"
	}
	if derived.AchievedConcurrencyK && measurement.Concurrency > 0 {
		requested := float64(measurement.Concurrency)
		if math.Abs(derived.AchievedConcurrency-requested)/requested > 0.10 {
			measurement.AchievedConcurrency = fmt.Sprintf("~%.0f (of %d)", derived.AchievedConcurrency, measurement.Concurrency)
		}
	}
	measurement.FailureBreakdown = derived.FailureBreakdown
	if measurement.WallTimeMSKnown && measurement.WallTimeMSValue > 0 && measurement.CompletedRequests > 0 {
		measurement.RPS = displayFloat(float64(measurement.CompletedRequests) / (measurement.WallTimeMSValue / 1000))
	} else {
		measurement.RPS = "-"
	}
	if derived.GPUUtilKnown {
		measurement.GPUUtil = fmt.Sprintf("%.0f / %.0f%% (%s)", derived.GPUUtilAvg, derived.GPUUtilPeak, derived.GPUUtilSource)
	} else {
		measurement.GPUUtil = "-"
	}
	if derived.GPUMemKnown {
		measurement.GPUMemPeak = fmt.Sprintf("%.1f GiB", derived.GPUMemPeakBytes/(1024*1024*1024))
	} else {
		measurement.GPUMemPeak = "-"
	}
}

// aggregateRepeatMeasurements collapses repeats of the same
// (profile, workload, phase, concurrency) point into one primary row with
// mean ± sample stddev displays; per-repeat rows are returned separately for
// the secondary Repeats section. Comparisons without variance present noise
// as signal.
func aggregateRepeatMeasurements(measurements []SQLiteReportMeasurement) (aggregated, repeats []SQLiteReportMeasurement) {
	type groupKey struct {
		profile, workload, phase string
		concurrency              int
	}
	groups := map[groupKey][]SQLiteReportMeasurement{}
	order := []groupKey{}
	for _, measurement := range measurements {
		key := groupKey{measurement.Profile, measurement.Workload, measurement.Phase, measurement.Concurrency}
		if _, seen := groups[key]; !seen {
			order = append(order, key)
		}
		groups[key] = append(groups[key], measurement)
	}
	for _, key := range order {
		members := groups[key]
		if len(members) == 1 {
			aggregated = append(aggregated, members[0])
			continue
		}
		aggregated = append(aggregated, combineRepeats(members))
		repeats = append(repeats, members...)
	}
	return aggregated, repeats
}

func combineRepeats(members []SQLiteReportMeasurement) SQLiteReportMeasurement {
	combined := members[0]
	combined.RepeatCount = len(members)
	combined.CompletedRequests = 0
	combined.FailedRequests = 0
	for _, member := range members {
		combined.CompletedRequests += member.CompletedRequests
		combined.FailedRequests += member.FailedRequests
		if statusRank(member.Status) > statusRank(combined.Status) {
			combined.Status = member.Status
		}
	}
	mean, known := meanOverRepeats(members, func(m SQLiteReportMeasurement) (float64, bool) {
		return m.OutputTokSValue, m.OutputTokSKnown
	})
	if known {
		combined.OutputTokSValue = mean
	}
	fields := []struct {
		get func(SQLiteReportMeasurement) string
		set func(*SQLiteReportMeasurement, string)
	}{
		{func(m SQLiteReportMeasurement) string { return m.OutputTokS }, func(m *SQLiteReportMeasurement, v string) { m.OutputTokS = v }},
		{func(m SQLiteReportMeasurement) string { return m.TotalTokS }, func(m *SQLiteReportMeasurement, v string) { m.TotalTokS = v }},
		{func(m SQLiteReportMeasurement) string { return m.PerUserOutputTokS }, func(m *SQLiteReportMeasurement, v string) { m.PerUserOutputTokS = v }},
		{func(m SQLiteReportMeasurement) string { return m.RPS }, func(m *SQLiteReportMeasurement, v string) { m.RPS = v }},
		{func(m SQLiteReportMeasurement) string { return m.TTFTMeanMS }, func(m *SQLiteReportMeasurement, v string) { m.TTFTMeanMS = v }},
		{func(m SQLiteReportMeasurement) string { return m.TTFTP50MS }, func(m *SQLiteReportMeasurement, v string) { m.TTFTP50MS = v }},
		{func(m SQLiteReportMeasurement) string { return m.TTFTP95MS }, func(m *SQLiteReportMeasurement, v string) { m.TTFTP95MS = v }},
		{func(m SQLiteReportMeasurement) string { return m.TTFTP99MS }, func(m *SQLiteReportMeasurement, v string) { m.TTFTP99MS = v }},
		{func(m SQLiteReportMeasurement) string { return m.LatencyP50MS }, func(m *SQLiteReportMeasurement, v string) { m.LatencyP50MS = v }},
		{func(m SQLiteReportMeasurement) string { return m.LatencyP95MS }, func(m *SQLiteReportMeasurement, v string) { m.LatencyP95MS = v }},
		{func(m SQLiteReportMeasurement) string { return m.LatencyP99MS }, func(m *SQLiteReportMeasurement, v string) { m.LatencyP99MS = v }},
		{func(m SQLiteReportMeasurement) string { return m.TPOTMeanMS }, func(m *SQLiteReportMeasurement, v string) { m.TPOTMeanMS = v }},
		{func(m SQLiteReportMeasurement) string { return m.ITLMeanMS }, func(m *SQLiteReportMeasurement, v string) { m.ITLMeanMS = v }},
		{func(m SQLiteReportMeasurement) string { return m.ITLTokenWeightedMS }, func(m *SQLiteReportMeasurement, v string) { m.ITLTokenWeightedMS = v }},
	}
	for _, field := range fields {
		field.set(&combined, meanSpreadDisplay(members, field.get))
	}
	return combined
}

func meanOverRepeats(members []SQLiteReportMeasurement, value func(SQLiteReportMeasurement) (float64, bool)) (float64, bool) {
	sum := 0.0
	count := 0
	for _, member := range members {
		current, known := value(member)
		if !known {
			continue
		}
		sum += current
		count++
	}
	if count == 0 {
		return 0, false
	}
	return sum / float64(count), true
}

// meanSpreadDisplay renders "mean ± stddev" (sample stddev, n-1) across the
// repeats' displayed values; if any repeat has no value, the spread cannot
// be computed honestly and "-" is rendered.
func meanSpreadDisplay(members []SQLiteReportMeasurement, value func(SQLiteReportMeasurement) string) string {
	values := make([]float64, 0, len(members))
	for _, member := range members {
		parsed, ok := parseDisplayedFloat(value(member))
		if !ok {
			return "-"
		}
		values = append(values, parsed)
	}
	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))
	variance := 0.0
	for _, v := range values {
		variance += (v - mean) * (v - mean)
	}
	stddev := math.Sqrt(variance / float64(len(values)-1))
	return displayFloat(mean) + " ± " + displayFloat(stddev)
}

func statusRank(status string) int {
	ranks := map[string]int{"completed": 0, "planned": 1, "running": 1, "skipped": 2, "canceled": 3, "failed": 4}
	return ranks[strings.ToLower(strings.TrimSpace(status))]
}
