package vllmbench

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/dutifuldev/localperf/internal/collections"
)

const htmlReportName = "report.html"

type HTMLReportOptions struct {
	Title      string
	IncludeRaw bool
	Store      bool
}

type SQLiteReportDocument struct {
	ArtifactPath       string
	GeneratedAt        time.Time
	Metadata           map[string]string
	Run                SQLiteReportRun
	Profiles           []SQLiteReportProfile
	Workloads          []SQLiteReportWorkload
	Measurements       []SQLiteReportMeasurement
	PhaseSections      []SQLiteReportPhaseSection
	Charts             []SQLiteReportChart
	RequestSummary     SQLiteReportRequestSummary
	EventCounts        []SQLiteReportCount
	NotableEvents      []SQLiteReportEvent
	Commands           []SQLiteReportCommand
	ExistingReports    []SQLiteReportExport
	ArtifactSummaries  []SQLiteReportArtifactSummary
	MeasurementMetrics map[int64]map[string]SQLiteReportMetric
}

type SQLiteReportRun struct {
	ID          string
	Name        string
	Description string
	Status      string
	CreatedAt   string
	StartedAt   string
	CompletedAt string
	Hostname    string
	Username    string
	CWD         string
	GitCommit   string
}

type SQLiteReportProfile struct {
	ID                    string
	Name                  string
	Model                 string
	ContextWindow         int
	MaxNumSeqs            int
	MaxNumBatchedTokens   int
	GPUMemoryUtilization  float64
	GPUMemoryUtilizationS string
	Managed               bool
	EnableSleepMode       bool
}

type SQLiteReportWorkload struct {
	ID                      string
	Name                    string
	Phase                   string
	Samples                 int
	Repeats                 int
	SaveDetailed            bool
	CapturePayloadArtifacts bool
}

type SQLiteReportMeasurement struct {
	ID                int64
	Profile           string
	Workload          string
	Phase             string
	ContextWindow     int
	RepeatIndex       int
	Concurrency       int
	SamplesRequested  int
	Status            string
	StartedAt         string
	CompletedAt       string
	WallTimeMS        string
	CompletedRequests int
	FailedRequests    int
	PromptTokens      string
	CompletionTokens  string
	TotalTokens       string
	OutputTokS        string
	OutputTokSValue   float64
	OutputTokSKnown   bool
	OutputTokSStdDev  string
	PerUserOutputTokS string
	TotalTokS         string
	LatencyMeanMS     string
	TTFTMeanMS        string
	TPOTMeanMS        string
	ITLMeanMS         string
	ErrorType         string
	ErrorMessage      string
}

type SQLiteReportMetric struct {
	MeasurementID int64
	Metric        string
	Unit          string
	Mean          string
	StdDev        string
	Min           string
	P50           string
	P90           string
	P95           string
	P99           string
	Max           string
	Count         int
	MeanValue     float64
	StdDevValue   float64
	MeanKnown     bool
	StdDevKnown   bool
}

type SQLiteReportPhaseSection struct {
	Phase        string
	Title        string
	Measurements []SQLiteReportMeasurement
}

type SQLiteReportChart struct {
	Title  string
	Unit   string
	Height int
	Bars   []SQLiteReportBar
}

type SQLiteReportBar struct {
	Label  string
	Value  string
	LabelY int
	RectY  int
	Width  string
}

type SQLiteReportRequestSummary struct {
	Total          int
	Completed      int
	Failed         int
	Canceled       int
	LatencyMeanMS  string
	TTFTMeanMS     string
	TPOTMeanMS     string
	ITLMeanMS      string
	OutputTokSMean string
}

type SQLiteReportCount struct {
	Name  string
	Count int
}

type SQLiteReportEvent struct {
	Timestamp string
	Level     string
	Type      string
	Profile   string
	Workload  string
	Message   string
}

type SQLiteReportCommand struct {
	Phase     string
	Status    string
	ExitCode  string
	StartedAt string
	Completed string
	Argv      string
}

type SQLiteReportExport struct {
	Name      string
	Format    string
	MediaType string
	CreatedAt string
}

type SQLiteReportArtifactSummary struct {
	Kind                  string
	Count                 int
	UncompressedSizeBytes int64
}

func LoadSQLiteReport(path string) (SQLiteReportDocument, error) {
	db, err := openSQLiteArtifactReadOnly(path)
	if err != nil {
		return SQLiteReportDocument{}, err
	}
	defer db.Close()
	if err := checkMetadata(db); err != nil {
		return SQLiteReportDocument{}, err
	}
	if err := checkRunRowCount(db); err != nil {
		return SQLiteReportDocument{}, err
	}
	doc := SQLiteReportDocument{
		ArtifactPath:       path,
		GeneratedAt:        time.Now().UTC(),
		Metadata:           map[string]string{},
		MeasurementMetrics: map[int64]map[string]SQLiteReportMetric{},
	}
	for _, load := range []func(*sql.DB, *SQLiteReportDocument) error{
		loadSQLiteReportMetadata,
		loadSQLiteReportRun,
		loadSQLiteReportProfiles,
		loadSQLiteReportWorkloads,
		loadSQLiteReportMetrics,
		loadSQLiteReportMeasurements,
		loadSQLiteReportRequestSummary,
		loadSQLiteReportEventCounts,
		loadSQLiteReportNotableEvents,
		loadSQLiteReportCommands,
		loadSQLiteReportExports,
		loadSQLiteReportArtifactSummaries,
	} {
		if err := load(db, &doc); err != nil {
			return SQLiteReportDocument{}, err
		}
	}
	doc.PhaseSections = sqliteReportPhaseSections(doc.Measurements)
	doc.Charts = sqliteReportCharts(doc.Measurements)
	return doc, nil
}

func RenderHTMLReport(writer io.Writer, doc SQLiteReportDocument, opts HTMLReportOptions) error {
	title := strings.TrimSpace(opts.Title)
	if title == "" {
		title = firstNonEmpty(doc.Run.Name, "LocalPerf Report")
	}
	view := struct {
		Title string
		Doc   SQLiteReportDocument
		Raw   bool
	}{
		Title: title,
		Doc:   doc,
		Raw:   opts.IncludeRaw,
	}
	tmpl, err := template.New("html-report").Funcs(template.FuncMap{
		"statusClass": reportStatusClass,
	}).Parse(sqliteHTMLReportTemplate)
	if err != nil {
		return err
	}
	return tmpl.Execute(writer, view)
}

func WriteSQLiteHTMLReport(artifactPath, outputPath string, opts HTMLReportOptions) error {
	resolvedOutputPath, err := htmlReportOutputPath(artifactPath, outputPath)
	if err != nil {
		return err
	}
	doc, err := LoadSQLiteReport(artifactPath)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(resolvedOutputPath), 0o755); err != nil {
		return err
	}
	var builder strings.Builder
	if err := RenderHTMLReport(&builder, doc, opts); err != nil {
		return err
	}
	content := []byte(builder.String())
	if err := os.WriteFile(resolvedOutputPath, content, 0o644); err != nil {
		return err
	}
	if opts.Store {
		return StoreSQLiteHTMLReport(artifactPath, htmlReportName, resolvedOutputPath, content)
	}
	return nil
}

func htmlReportOutputPath(artifactPath, outputPath string) (string, error) {
	if strings.TrimSpace(outputPath) == "" {
		outputPath = defaultHTMLReportPath(artifactPath)
	}
	if err := rejectSourceArtifactOutput(artifactPath, outputPath); err != nil {
		return "", err
	}
	return outputPath, nil
}

func rejectSourceArtifactOutput(artifactPath, outputPath string) error {
	sameFile, err := sameSQLiteArtifactOutput(artifactPath, outputPath)
	if err != nil {
		return err
	}
	if sameFile {
		return fmt.Errorf("HTML output path must differ from SQLite artifact path: %s", outputPath)
	}
	return nil
}

func sameSQLiteArtifactOutput(artifactPath, outputPath string) (bool, error) {
	artifactAbs, err := filepath.Abs(artifactPath)
	if err != nil {
		return false, err
	}
	outputAbs, err := filepath.Abs(outputPath)
	if err != nil {
		return false, err
	}
	if filepath.Clean(artifactAbs) == filepath.Clean(outputAbs) {
		return true, nil
	}
	return sameExistingFile(artifactAbs, outputAbs)
}

func sameExistingFile(firstPath, secondPath string) (bool, error) {
	firstInfo, err := os.Stat(firstPath)
	if err != nil {
		return false, err
	}
	secondInfo, err := os.Stat(secondPath)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	return os.SameFile(firstInfo, secondInfo), nil
}

func StoreSQLiteHTMLReport(artifactPath, name, originalPath string, content []byte) error {
	if strings.TrimSpace(name) == "" {
		name = htmlReportName
	}
	db, err := openSQLiteArtifactWritable(artifactPath)
	if err != nil {
		return err
	}
	defer db.Close()
	return withSQLiteTx(db, func(tx *sql.Tx) error {
		runID, err := sqliteSingleRunID(tx)
		if err != nil {
			return err
		}
		artifactID, err := upsertHTMLReportArtifact(tx, runID, name, originalPath, content, time.Now().UTC())
		if err != nil {
			return err
		}
		return upsertHTMLReportRow(tx, runID, name, artifactID, time.Now().UTC())
	})
}

func openSQLiteArtifactReadOnly(path string) (*sql.DB, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	uri := url.URL{Scheme: "file", Path: absolute, RawQuery: "mode=ro"}
	db, err := sql.Open("sqlite", uri.String())
	if err != nil {
		return nil, err
	}
	if _, err := db.Exec("PRAGMA foreign_keys = ON"); err != nil {
		_ = db.Close()
		return nil, err
	}
	return db, nil
}

func openSQLiteArtifactWritable(path string) (*sql.DB, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	if _, err := db.Exec("PRAGMA foreign_keys = ON"); err != nil {
		_ = db.Close()
		return nil, err
	}
	return db, nil
}

func defaultHTMLReportPath(artifactPath string) string {
	ext := filepath.Ext(artifactPath)
	if ext == "" {
		return artifactPath + ".html"
	}
	return strings.TrimSuffix(artifactPath, ext) + ".html"
}

func loadSQLiteReportMetadata(db *sql.DB, doc *SQLiteReportDocument) error {
	rows, err := db.Query("SELECT key, value FROM metadata ORDER BY key")
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var key, value string
		if err := rows.Scan(&key, &value); err != nil {
			return err
		}
		doc.Metadata[key] = value
	}
	return rows.Err()
}

func loadSQLiteReportRun(db *sql.DB, doc *SQLiteReportDocument) error {
	var description, startedAt, completedAt, hostname, username, cwd, gitCommit sql.NullString
	err := db.QueryRow(`SELECT
		id, name, description, status, created_at, started_at, completed_at,
		hostname, username, cwd, localperf_git_commit
		FROM run LIMIT 1`).Scan(
		&doc.Run.ID, &doc.Run.Name, &description, &doc.Run.Status, &doc.Run.CreatedAt,
		&startedAt, &completedAt, &hostname, &username, &cwd, &gitCommit)
	if err != nil {
		return err
	}
	doc.Run.Description = nullStringValue(description)
	doc.Run.StartedAt = nullStringValue(startedAt)
	doc.Run.CompletedAt = nullStringValue(completedAt)
	doc.Run.Hostname = nullStringValue(hostname)
	doc.Run.Username = nullStringValue(username)
	doc.Run.CWD = nullStringValue(cwd)
	doc.Run.GitCommit = nullStringValue(gitCommit)
	return nil
}

func loadSQLiteReportProfiles(db *sql.DB, doc *SQLiteReportDocument) error {
	rows, err := db.Query(`SELECT
		id, name, model, COALESCE(context_window, 0), COALESCE(max_num_seqs, 0),
		COALESCE(max_num_batched_tokens, 0), COALESCE(gpu_memory_utilization, 0),
		managed, COALESCE(enable_sleep_mode, 0)
		FROM profiles ORDER BY context_window, name`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var profile SQLiteReportProfile
		var managed, sleep int
		if err := rows.Scan(
			&profile.ID, &profile.Name, &profile.Model, &profile.ContextWindow,
			&profile.MaxNumSeqs, &profile.MaxNumBatchedTokens, &profile.GPUMemoryUtilization,
			&managed, &sleep,
		); err != nil {
			return err
		}
		profile.GPUMemoryUtilizationS = displayFloat(profile.GPUMemoryUtilization)
		profile.Managed = managed != 0
		profile.EnableSleepMode = sleep != 0
		doc.Profiles = append(doc.Profiles, profile)
	}
	return rows.Err()
}

func loadSQLiteReportWorkloads(db *sql.DB, doc *SQLiteReportDocument) error {
	rows, err := db.Query(`SELECT
		id, name, phase, samples, repeats, save_detailed, capture_payload_artifacts
		FROM workloads ORDER BY phase, name`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var workload SQLiteReportWorkload
		var saveDetailed, capture int
		if err := rows.Scan(&workload.ID, &workload.Name, &workload.Phase, &workload.Samples, &workload.Repeats, &saveDetailed, &capture); err != nil {
			return err
		}
		workload.SaveDetailed = saveDetailed != 0
		workload.CapturePayloadArtifacts = capture != 0
		doc.Workloads = append(doc.Workloads, workload)
	}
	return rows.Err()
}

func loadSQLiteReportMeasurements(db *sql.DB, doc *SQLiteReportDocument) error {
	rows, err := db.Query(`SELECT
		m.id, p.name, w.name, w.phase, COALESCE(p.context_window, 0), m.repeat_index,
		m.concurrency, m.samples_requested, m.status, m.started_at, m.completed_at,
		m.wall_time_ms, m.completed_requests, m.failed_requests, m.prompt_tokens,
		m.completion_tokens, m.total_tokens, m.aggregate_output_tok_s,
		m.per_user_output_tok_s, m.aggregate_total_tok_s, m.error_type, m.error_message
		FROM measurements m
		JOIN profiles p ON p.id = m.profile_id
		JOIN workloads w ON w.id = m.workload_id
		ORDER BY w.phase, COALESCE(p.context_window, 0), p.name, w.name, m.concurrency, m.repeat_index`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var measurement SQLiteReportMeasurement
		var startedAt, completedAt, errorType, errorMessage sql.NullString
		var wallTime, outputTokS, perUserTokS, totalTokS sql.NullFloat64
		var promptTokens, completionTokens, totalTokens sql.NullInt64
		if err := rows.Scan(
			&measurement.ID, &measurement.Profile, &measurement.Workload, &measurement.Phase,
			&measurement.ContextWindow, &measurement.RepeatIndex, &measurement.Concurrency,
			&measurement.SamplesRequested, &measurement.Status, &startedAt, &completedAt,
			&wallTime, &measurement.CompletedRequests, &measurement.FailedRequests,
			&promptTokens, &completionTokens, &totalTokens, &outputTokS, &perUserTokS,
			&totalTokS, &errorType, &errorMessage,
		); err != nil {
			return err
		}
		applySQLiteMeasurementDisplay(&measurement, doc.MeasurementMetrics[measurement.ID], startedAt, completedAt, wallTime, promptTokens, completionTokens, totalTokens, outputTokS, perUserTokS, totalTokS, errorType, errorMessage)
		doc.Measurements = append(doc.Measurements, measurement)
	}
	return rows.Err()
}

func applySQLiteMeasurementDisplay(measurement *SQLiteReportMeasurement, metrics map[string]SQLiteReportMetric, startedAt, completedAt sql.NullString, wallTime sql.NullFloat64, promptTokens, completionTokens, totalTokens sql.NullInt64, outputTokS, perUserTokS, totalTokS sql.NullFloat64, errorType, errorMessage sql.NullString) {
	measurement.StartedAt = nullStringValue(startedAt)
	measurement.CompletedAt = nullStringValue(completedAt)
	measurement.WallTimeMS = displayNullFloat(wallTime)
	measurement.PromptTokens = displayNullInt(promptTokens)
	measurement.CompletionTokens = displayNullInt(completionTokens)
	measurement.TotalTokens = displayNullInt(totalTokens)
	measurement.OutputTokS = displayNullFloat(outputTokS)
	measurement.OutputTokSValue = nullFloatValue(outputTokS)
	measurement.OutputTokSKnown = outputTokS.Valid
	measurement.PerUserOutputTokS = displayNullFloat(perUserTokS)
	measurement.TotalTokS = displayNullFloat(totalTokS)
	measurement.ErrorType = nullStringValue(errorType)
	measurement.ErrorMessage = nullStringValue(errorMessage)
	measurement.OutputTokSStdDev = metricDisplay(metrics, "request_output_throughput", "StdDev")
	measurement.LatencyMeanMS = metricDisplay(metrics, "latency", "Mean")
	measurement.TTFTMeanMS = metricDisplayFirst(metrics, "Mean", "request_ttft", "ttft")
	measurement.TPOTMeanMS = metricDisplayFirst(metrics, "Mean", "request_tpot", "tpot")
	measurement.ITLMeanMS = metricDisplay(metrics, "request_itl_mean", "Mean")
}

func loadSQLiteReportMetrics(db *sql.DB, doc *SQLiteReportDocument) error {
	rows, err := db.Query(`SELECT
		measurement_id, metric, unit, mean, stddev, min, p50, p90, p95, p99, max, count
		FROM metric_stats ORDER BY measurement_id, metric, unit`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var metric SQLiteReportMetric
		var mean, stddev, min, p50, p90, p95, p99, max sql.NullFloat64
		if err := rows.Scan(&metric.MeasurementID, &metric.Metric, &metric.Unit, &mean, &stddev, &min, &p50, &p90, &p95, &p99, &max, &metric.Count); err != nil {
			return err
		}
		metric.Mean = displayNullFloat(mean)
		metric.StdDev = displayNullFloat(stddev)
		metric.Min = displayNullFloat(min)
		metric.P50 = displayNullFloat(p50)
		metric.P90 = displayNullFloat(p90)
		metric.P95 = displayNullFloat(p95)
		metric.P99 = displayNullFloat(p99)
		metric.Max = displayNullFloat(max)
		metric.MeanValue = nullFloatValue(mean)
		metric.StdDevValue = nullFloatValue(stddev)
		metric.MeanKnown = mean.Valid
		metric.StdDevKnown = stddev.Valid
		if doc.MeasurementMetrics[metric.MeasurementID] == nil {
			doc.MeasurementMetrics[metric.MeasurementID] = map[string]SQLiteReportMetric{}
		}
		doc.MeasurementMetrics[metric.MeasurementID][metric.Metric] = metric
	}
	return rows.Err()
}

func loadSQLiteReportRequestSummary(db *sql.DB, doc *SQLiteReportDocument) error {
	var latency, ttft, tpot, itl, outputTokS sqliteReportWeightedMean
	var latencyMean, ttftMean, tpotMean, itlMean, outputTokSMean sql.NullFloat64
	var latencyCount, ttftCount, tpotCount, itlCount, outputTokSCount int
	err := db.QueryRow(`SELECT
			COUNT(*),
			COALESCE(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END), 0),
			COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0),
			COALESCE(SUM(CASE WHEN status = 'canceled' THEN 1 ELSE 0 END), 0),
			AVG(latency_ms), COUNT(latency_ms),
			AVG(ttft_ms), COUNT(ttft_ms),
			AVG(tpot_ms), COUNT(tpot_ms),
			AVG(itl_mean_ms), COUNT(itl_mean_ms),
			AVG(output_tok_s), COUNT(output_tok_s)
			FROM requests`).Scan(
		&doc.RequestSummary.Total, &doc.RequestSummary.Completed, &doc.RequestSummary.Failed,
		&doc.RequestSummary.Canceled, &latencyMean, &latencyCount, &ttftMean, &ttftCount,
		&tpotMean, &tpotCount, &itlMean, &itlCount, &outputTokSMean, &outputTokSCount)
	if err != nil {
		return err
	}
	latency.addNullFloat(latencyMean, latencyCount)
	ttft.addNullFloat(ttftMean, ttftCount)
	tpot.addNullFloat(tpotMean, tpotCount)
	itl.addNullFloat(itlMean, itlCount)
	outputTokS.addNullFloat(outputTokSMean, outputTokSCount)
	requestRows, err := sqliteRequestRowsByMeasurement(db)
	if err != nil {
		return err
	}
	applySQLiteAggregateRequestSummary(doc, requestRows, &latency, &ttft, &tpot, &itl, &outputTokS)
	doc.RequestSummary.LatencyMeanMS = displayWeightedMean(latency)
	doc.RequestSummary.TTFTMeanMS = displayWeightedMean(ttft)
	doc.RequestSummary.TPOTMeanMS = displayWeightedMean(tpot)
	doc.RequestSummary.ITLMeanMS = displayWeightedMean(itl)
	doc.RequestSummary.OutputTokSMean = displayWeightedMean(outputTokS)
	return nil
}

func sqliteRequestRowsByMeasurement(db *sql.DB) (map[int64]int, error) {
	rows, err := db.Query(`SELECT measurement_id, COUNT(*) FROM requests GROUP BY measurement_id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[int64]int{}
	for rows.Next() {
		var measurementID int64
		var count int
		if err := rows.Scan(&measurementID, &count); err != nil {
			return nil, err
		}
		out[measurementID] = count
	}
	return out, rows.Err()
}

func applySQLiteAggregateRequestSummary(doc *SQLiteReportDocument, requestRows map[int64]int, latency, ttft, tpot, itl, outputTokS *sqliteReportWeightedMean) {
	for _, measurement := range doc.Measurements {
		if requestRows[measurement.ID] > 0 {
			continue
		}
		doc.RequestSummary.Completed += measurement.CompletedRequests
		doc.RequestSummary.Failed += measurement.FailedRequests
		doc.RequestSummary.Canceled += canceledRequestEstimate(measurement)
		if measurement.OutputTokSKnown {
			outputTokS.add(measurement.OutputTokSValue, 1)
		}
		metrics := doc.MeasurementMetrics[measurement.ID]
		latency.addMetric(metricFirst(metrics, "latency"))
		ttft.addMetric(metricFirst(metrics, "request_ttft", "ttft"))
		tpot.addMetric(metricFirst(metrics, "request_tpot", "tpot"))
		itl.addMetric(metricFirst(metrics, "request_itl_mean"))
	}
	doc.RequestSummary.Total = doc.RequestSummary.Completed + doc.RequestSummary.Failed + doc.RequestSummary.Canceled
}

func canceledRequestEstimate(measurement SQLiteReportMeasurement) int {
	if measurement.Status != "canceled" {
		return 0
	}
	remaining := measurement.SamplesRequested - measurement.CompletedRequests - measurement.FailedRequests
	if remaining < 0 {
		return 0
	}
	return remaining
}

type sqliteReportWeightedMean struct {
	total  float64
	weight int
}

func (mean *sqliteReportWeightedMean) add(value float64, weight int) {
	if weight <= 0 {
		weight = 1
	}
	mean.total += value * float64(weight)
	mean.weight += weight
}

func (mean *sqliteReportWeightedMean) addNullFloat(value sql.NullFloat64, weight int) {
	if !value.Valid {
		return
	}
	mean.add(value.Float64, weight)
}

func (mean *sqliteReportWeightedMean) addMetric(metric SQLiteReportMetric, ok bool) {
	if !ok || !metric.MeanKnown {
		return
	}
	mean.add(metric.MeanValue, metric.Count)
}

func displayWeightedMean(mean sqliteReportWeightedMean) string {
	if mean.weight == 0 {
		return "-"
	}
	return displayFloat(mean.total / float64(mean.weight))
}

func loadSQLiteReportEventCounts(db *sql.DB, doc *SQLiteReportDocument) error {
	rows, err := db.Query(`SELECT type, COUNT(*) FROM events GROUP BY type ORDER BY type`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var count SQLiteReportCount
		if err := rows.Scan(&count.Name, &count.Count); err != nil {
			return err
		}
		doc.EventCounts = append(doc.EventCounts, count)
	}
	return rows.Err()
}

func loadSQLiteReportNotableEvents(db *sql.DB, doc *SQLiteReportDocument) error {
	rows, err := db.Query(`SELECT
		timestamp, level, type, COALESCE(profile_id, ''), COALESCE(workload_id, ''), COALESCE(message, '')
		FROM events
		WHERE level IN ('warn', 'error') OR type LIKE '%failed%' OR TRIM(COALESCE(message, '')) <> ''
		ORDER BY timestamp DESC LIMIT 50`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var event SQLiteReportEvent
		if err := rows.Scan(&event.Timestamp, &event.Level, &event.Type, &event.Profile, &event.Workload, &event.Message); err != nil {
			return err
		}
		doc.NotableEvents = append(doc.NotableEvents, event)
	}
	return rows.Err()
}

func loadSQLiteReportCommands(db *sql.DB, doc *SQLiteReportDocument) error {
	rows, err := db.Query(`SELECT
		phase, status, exit_code, started_at, completed_at, argv_json
		FROM commands ORDER BY id LIMIT 100`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var command SQLiteReportCommand
		var exitCode sql.NullInt64
		var startedAt, completedAt sql.NullString
		var argvJSON string
		if err := rows.Scan(&command.Phase, &command.Status, &exitCode, &startedAt, &completedAt, &argvJSON); err != nil {
			return err
		}
		command.ExitCode = displayNullInt(exitCode)
		command.StartedAt = nullStringValue(startedAt)
		command.Completed = nullStringValue(completedAt)
		command.Argv = commandSummaryFromJSON(argvJSON)
		doc.Commands = append(doc.Commands, command)
	}
	return rows.Err()
}

func loadSQLiteReportExports(db *sql.DB, doc *SQLiteReportDocument) error {
	rows, err := db.Query(`SELECT name, format, media_type, created_at FROM reports ORDER BY created_at, name`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var export SQLiteReportExport
		if err := rows.Scan(&export.Name, &export.Format, &export.MediaType, &export.CreatedAt); err != nil {
			return err
		}
		doc.ExistingReports = append(doc.ExistingReports, export)
	}
	return rows.Err()
}

func loadSQLiteReportArtifactSummaries(db *sql.DB, doc *SQLiteReportDocument) error {
	rows, err := db.Query(`SELECT kind, COUNT(*), COALESCE(SUM(uncompressed_size_bytes), 0) FROM artifacts GROUP BY kind ORDER BY kind`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var summary SQLiteReportArtifactSummary
		if err := rows.Scan(&summary.Kind, &summary.Count, &summary.UncompressedSizeBytes); err != nil {
			return err
		}
		doc.ArtifactSummaries = append(doc.ArtifactSummaries, summary)
	}
	return rows.Err()
}

func sqliteReportPhaseSections(measurements []SQLiteReportMeasurement) []SQLiteReportPhaseSection {
	byPhase := map[string][]SQLiteReportMeasurement{}
	for _, measurement := range measurements {
		phase := normalizeReportPhase(measurement.Phase)
		byPhase[phase] = append(byPhase[phase], measurement)
	}
	phases := collections.SortedKeys(byPhase)
	sort.SliceStable(phases, func(i, j int) bool {
		left, right := phaseRank(phases[i]), phaseRank(phases[j])
		if left != right {
			return left < right
		}
		return phases[i] < phases[j]
	})
	out := make([]SQLiteReportPhaseSection, 0, len(phases))
	for _, phase := range phases {
		out = append(out, SQLiteReportPhaseSection{Phase: phase, Title: phaseTitle(phase), Measurements: byPhase[phase]})
	}
	return out
}

func sqliteReportCharts(measurements []SQLiteReportMeasurement) []SQLiteReportChart {
	bars := make([]SQLiteReportBar, 0, len(measurements))
	maxValue := 0.0
	for _, measurement := range measurements {
		if measurement.OutputTokSKnown && measurement.OutputTokSValue > maxValue {
			maxValue = measurement.OutputTokSValue
		}
	}
	if maxValue <= 0 {
		return nil
	}
	for _, measurement := range measurements {
		if !measurement.OutputTokSKnown {
			continue
		}
		index := len(bars)
		label := fmt.Sprintf("%s / %s / c%d", measurement.Profile, measurement.Workload, measurement.Concurrency)
		bars = append(bars, SQLiteReportBar{
			Label:  label,
			Value:  displayFloat(measurement.OutputTokSValue),
			LabelY: 24 + index*28,
			RectY:  10 + index*28,
			Width:  displayFloat(560 * measurement.OutputTokSValue / maxValue),
		})
	}
	return []SQLiteReportChart{{Title: "Aggregate Output Throughput", Unit: "tok/s", Height: 24 + len(bars)*28, Bars: bars}}
}

func sqliteSingleRunID(tx *sql.Tx) (string, error) {
	var runID string
	if err := tx.QueryRow("SELECT id FROM run LIMIT 1").Scan(&runID); err != nil {
		return "", err
	}
	return runID, nil
}

func upsertHTMLReportArtifact(tx *sql.Tx, runID, name, originalPath string, data []byte, createdAt time.Time) (int64, error) {
	content, compression, err := artifactContent(data, "text/html")
	if err != nil {
		return 0, err
	}
	_, err = tx.Exec(`INSERT INTO artifacts (
		run_id, kind, name, media_type, compression, content, content_size_bytes,
		uncompressed_size_bytes, sha256, original_path, created_at
	) VALUES (?, 'normalized_report', ?, 'text/html', ?, ?, ?, ?, ?, ?, ?)
	ON CONFLICT(run_id, kind, name) DO UPDATE SET
		media_type = excluded.media_type,
		compression = excluded.compression,
		content = excluded.content,
		content_size_bytes = excluded.content_size_bytes,
		uncompressed_size_bytes = excluded.uncompressed_size_bytes,
		sha256 = excluded.sha256,
		original_path = excluded.original_path,
		created_at = excluded.created_at`,
		runID, name, compression, content, len(content), len(data), sha256Hex(data),
		nullString(originalPath), createdAt.Format(time.RFC3339))
	if err != nil {
		return 0, err
	}
	var artifactID int64
	err = tx.QueryRow(`SELECT id FROM artifacts WHERE run_id = ? AND kind = 'normalized_report' AND name = ?`, runID, name).Scan(&artifactID)
	return artifactID, err
}

func upsertHTMLReportRow(tx *sql.Tx, runID, name string, artifactID int64, createdAt time.Time) error {
	_, err := tx.Exec(`INSERT INTO reports (
		run_id, name, format, media_type, artifact_id, created_at
	) VALUES (?, ?, 'html', 'text/html', ?, ?)
	ON CONFLICT(run_id, name, format) DO UPDATE SET
		media_type = excluded.media_type,
		artifact_id = excluded.artifact_id,
		created_at = excluded.created_at`,
		runID, name, artifactID, createdAt.Format(time.RFC3339))
	return err
}

func metricDisplay(metrics map[string]SQLiteReportMetric, name, field string) string {
	return metricDisplayFirst(metrics, field, name)
}

func metricDisplayFirst(metrics map[string]SQLiteReportMetric, field string, names ...string) string {
	for _, name := range names {
		metric, ok := metricFirst(metrics, name)
		if !ok {
			continue
		}
		if value, ok := metricFieldDisplay(metric, field); ok {
			return value
		}
	}
	return "-"
}

func metricFirst(metrics map[string]SQLiteReportMetric, names ...string) (SQLiteReportMetric, bool) {
	for _, name := range names {
		metric, ok := metrics[name]
		if ok {
			return metric, true
		}
	}
	return SQLiteReportMetric{}, false
}

func metricFieldDisplay(metric SQLiteReportMetric, field string) (string, bool) {
	switch field {
	case "StdDev":
		if metric.StdDevKnown {
			return metric.StdDev, true
		}
	default:
		if metric.MeanKnown {
			return metric.Mean, true
		}
	}
	return "", false
}

func commandSummaryFromJSON(data string) string {
	var args []string
	if err := json.Unmarshal([]byte(data), &args); err != nil || len(args) == 0 {
		return ""
	}
	return ShellQuote(args)
}

func reportStatusClass(status string) string {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "completed":
		return "status-ok"
	case "failed", "canceled":
		return "status-bad"
	case "skipped":
		return "status-warn"
	default:
		return "status-neutral"
	}
}

func nullStringValue(value sql.NullString) string {
	if value.Valid {
		return value.String
	}
	return ""
}

func nullFloatValue(value sql.NullFloat64) float64 {
	if value.Valid {
		return value.Float64
	}
	return 0
}

func displayNullInt(value sql.NullInt64) string {
	if value.Valid {
		return fmt.Sprintf("%d", value.Int64)
	}
	return "-"
}

func displayNullFloat(value sql.NullFloat64) string {
	if value.Valid {
		return displayFloat(value.Float64)
	}
	return "-"
}

func displayFloat(value float64) string {
	return fmt.Sprintf("%.3f", value)
}

const sqliteHTMLReportTemplate = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{.Title}}</title>
<style>
:root{color-scheme:light;--bg:#f7f8fa;--panel:#ffffff;--text:#151922;--muted:#647084;--line:#d9dee8;--accent:#0f766e;--accent2:#1d4ed8;--bad:#b42318;--warn:#a15c07;--ok:#067647}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1280px;margin:0 auto;padding:24px}header{border-bottom:1px solid var(--line);padding:28px 24px 20px;background:var(--panel)}h1{font-size:28px;line-height:1.1;margin:0 0 8px}h2{font-size:18px;margin:28px 0 12px}h3{font-size:15px;margin:20px 0 10px}.subtle{color:var(--muted)}.grid{display:grid;gap:12px}.summary{grid-template-columns:repeat(auto-fit,minmax(170px,1fr));margin-top:18px}.stat{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}.stat strong{display:block;font-size:20px;margin-top:4px}.section{margin-top:18px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px;background:var(--panel)}table{width:100%;border-collapse:collapse;min-width:900px}th,td{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top;white-space:nowrap}th{font-size:12px;color:var(--muted);background:#f0f3f7;font-weight:650}td.num,th.num{text-align:right}.pill{display:inline-block;border-radius:999px;padding:2px 8px;font-size:12px;border:1px solid var(--line)}.status-ok{color:var(--ok);background:#ecfdf3;border-color:#abefc6}.status-bad{color:var(--bad);background:#fef3f2;border-color:#fecdca}.status-warn{color:var(--warn);background:#fffaeb;border-color:#fedf89}.status-neutral{color:var(--muted);background:#f8fafc}.chart{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}.svg-chart{width:100%;height:auto;display:block}.svg-label{font-size:12px;fill:var(--text)}.svg-value{font-size:12px;fill:var(--muted);text-anchor:end}.svg-track{fill:#e7ebf2}.svg-bar{fill:var(--accent)}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.small{font-size:12px}.wrap{white-space:normal}.privacy{border:1px solid var(--line);background:#fff8e6;border-radius:8px;padding:10px 12px;color:#704b00}@media (max-width:720px){main{padding:14px}header{padding:22px 14px}.summary{grid-template-columns:1fr}table{min-width:760px}}
</style>
</head>
<body>
<header>
<h1>{{.Title}}</h1>
<div class="subtle">Artifact: <span class="mono">{{.Doc.ArtifactPath}}</span></div>
<div class="subtle">Generated: <span class="mono">{{.Doc.GeneratedAt.Format "2006-01-02T15:04:05Z07:00"}}</span></div>
<div class="grid summary">
<div class="stat"><span>Run status</span><strong><span class="pill {{statusClass .Doc.Run.Status}}">{{.Doc.Run.Status}}</span></strong></div>
<div class="stat"><span>Measurements</span><strong>{{len .Doc.Measurements}}</strong></div>
<div class="stat"><span>Requests</span><strong>{{.Doc.RequestSummary.Total}}</strong></div>
<div class="stat"><span>Profiles</span><strong>{{len .Doc.Profiles}}</strong></div>
</div>
</header>
<main>
<section class="section">
<h2>Run</h2>
<div class="table-wrap"><table>
<tbody>
<tr><th>ID</th><td>{{.Doc.Run.ID}}</td><th>Name</th><td>{{.Doc.Run.Name}}</td></tr>
<tr><th>Created</th><td>{{.Doc.Run.CreatedAt}}</td><th>Completed</th><td>{{.Doc.Run.CompletedAt}}</td></tr>
<tr><th>Host</th><td>{{.Doc.Run.Hostname}}</td><th>User</th><td>{{.Doc.Run.Username}}</td></tr>
<tr><th>CWD</th><td class="mono wrap" colspan="3">{{.Doc.Run.CWD}}</td></tr>
</tbody>
</table></div>
</section>
<section class="section">
<h2>Request Summary</h2>
<div class="grid summary">
<div class="stat"><span>Completed</span><strong>{{.Doc.RequestSummary.Completed}}</strong></div>
<div class="stat"><span>Failed</span><strong>{{.Doc.RequestSummary.Failed}}</strong></div>
<div class="stat"><span>Mean TTFT ms</span><strong>{{.Doc.RequestSummary.TTFTMeanMS}}</strong></div>
<div class="stat"><span>Mean TPOT ms</span><strong>{{.Doc.RequestSummary.TPOTMeanMS}}</strong></div>
<div class="stat"><span>Mean ITL ms</span><strong>{{.Doc.RequestSummary.ITLMeanMS}}</strong></div>
<div class="stat"><span>Mean output tok/s</span><strong>{{.Doc.RequestSummary.OutputTokSMean}}</strong></div>
</div>
</section>
{{range .Doc.Charts}}
<section class="section chart">
<h2>{{.Title}}</h2>
<svg class="svg-chart" viewBox="0 0 960 {{.Height}}" role="img" aria-label="{{.Title}}">
<line x1="320" y1="0" x2="320" y2="{{.Height}}" stroke="#d9dee8"/>
{{range .Bars}}
<text class="svg-label" x="0" y="{{.LabelY}}">{{.Label}}</text>
<rect class="svg-track" x="320" y="{{.RectY}}" width="560" height="16" rx="4"></rect>
<rect class="svg-bar" x="320" y="{{.RectY}}" width="{{.Width}}" height="16" rx="4"></rect>
<text class="svg-value" x="940" y="{{.LabelY}}">{{.Value}}</text>
{{end}}
</svg>
<div class="small subtle">{{.Unit}}</div>
</section>
{{end}}
{{range .Doc.PhaseSections}}
<section class="section">
<h2>{{.Title}} Measurements</h2>
<div class="table-wrap"><table>
<thead><tr><th>Profile</th><th>Workload</th><th class="num">Context</th><th class="num">Conc.</th><th>Status</th><th class="num">Done</th><th class="num">Failed</th><th class="num">Output tok/s</th><th class="num">Req tok/s sd</th><th class="num">Per-user tok/s</th><th class="num">TTFT ms</th><th class="num">TPOT ms</th><th class="num">ITL ms</th><th class="num">Latency ms</th></tr></thead>
<tbody>
{{range .Measurements}}
<tr><td>{{.Profile}}</td><td>{{.Workload}}</td><td class="num">{{.ContextWindow}}</td><td class="num">{{.Concurrency}}</td><td><span class="pill {{statusClass .Status}}">{{.Status}}</span></td><td class="num">{{.CompletedRequests}}</td><td class="num">{{.FailedRequests}}</td><td class="num">{{.OutputTokS}}</td><td class="num">{{.OutputTokSStdDev}}</td><td class="num">{{.PerUserOutputTokS}}</td><td class="num">{{.TTFTMeanMS}}</td><td class="num">{{.TPOTMeanMS}}</td><td class="num">{{.ITLMeanMS}}</td><td class="num">{{.LatencyMeanMS}}</td></tr>
{{end}}
</tbody>
</table></div>
</section>
{{end}}
<section class="section">
<h2>Profiles</h2>
<div class="table-wrap"><table>
<thead><tr><th>Name</th><th>Model</th><th class="num">Context</th><th class="num">Max seqs</th><th class="num">Batched tokens</th><th class="num">GPU memory util.</th><th>Managed</th><th>Sleep</th></tr></thead>
<tbody>{{range .Doc.Profiles}}<tr><td>{{.Name}}</td><td>{{.Model}}</td><td class="num">{{.ContextWindow}}</td><td class="num">{{.MaxNumSeqs}}</td><td class="num">{{.MaxNumBatchedTokens}}</td><td class="num">{{.GPUMemoryUtilizationS}}</td><td>{{.Managed}}</td><td>{{.EnableSleepMode}}</td></tr>{{end}}</tbody>
</table></div>
</section>
<section class="section">
<h2>Events</h2>
<div class="grid summary">{{range .Doc.EventCounts}}<div class="stat"><span>{{.Name}}</span><strong>{{.Count}}</strong></div>{{end}}</div>
{{if .Doc.NotableEvents}}<h3>Notable Events</h3><div class="table-wrap"><table><thead><tr><th>Time</th><th>Level</th><th>Type</th><th>Profile</th><th>Workload</th><th>Message</th></tr></thead><tbody>{{range .Doc.NotableEvents}}<tr><td>{{.Timestamp}}</td><td>{{.Level}}</td><td>{{.Type}}</td><td>{{.Profile}}</td><td>{{.Workload}}</td><td class="wrap">{{.Message}}</td></tr>{{end}}</tbody></table></div>{{end}}
</section>
<section class="section">
<h2>Commands</h2>
<div class="table-wrap"><table><thead><tr><th>Phase</th><th>Status</th><th>Exit</th><th>Started</th><th>Completed</th><th>Command</th></tr></thead><tbody>{{range .Doc.Commands}}<tr><td>{{.Phase}}</td><td><span class="pill {{statusClass .Status}}">{{.Status}}</span></td><td>{{.ExitCode}}</td><td>{{.StartedAt}}</td><td>{{.Completed}}</td><td class="mono wrap">{{.Argv}}</td></tr>{{end}}</tbody></table></div>
</section>
<section class="section">
<h2>Artifact Contents</h2>
<div class="table-wrap"><table><thead><tr><th>Kind</th><th class="num">Count</th><th class="num">Uncompressed bytes</th></tr></thead><tbody>{{range .Doc.ArtifactSummaries}}<tr><td>{{.Kind}}</td><td class="num">{{.Count}}</td><td class="num">{{.UncompressedSizeBytes}}</td></tr>{{end}}</tbody></table></div>
</section>
<section class="section">
<h2>Privacy</h2>
<div class="privacy">This standalone report is rendered from normalized SQLite metrics. It does not include raw prompts, generated text, log bodies, or raw artifact contents.</div>
</section>
</main>
</body>
</html>
`
