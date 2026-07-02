package report

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/dutifuldev/localperf/internal/artifact"
	"github.com/dutifuldev/localperf/internal/bench"
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
	Engines            []SQLiteReportEngine
	Profiles           []SQLiteReportProfile
	Workloads          []SQLiteReportWorkload
	Measurements       []SQLiteReportMeasurement
	MetadataItems      []SQLiteReportMetadataItem
	ThroughputRows     []SQLiteReportThroughputRow
	ThroughputGroups   []SQLiteReportThroughputGroup
	PhaseSections      []SQLiteReportPhaseSection
	Charts             []SQLiteReportChart
	RequestSummary     SQLiteReportRequestSummary
	EventCounts        []SQLiteReportCount
	NotableEvents      []SQLiteReportEvent
	Commands           []SQLiteReportCommand
	ExistingReports    []SQLiteReportExport
	ArtifactSummaries  []SQLiteReportArtifactSummary
	MeasurementMetrics map[int64]map[string]SQLiteReportMetric
	RequestDerived     map[int64]sqliteRequestDerived
	RepeatDetails      []SQLiteReportMeasurement
	Legend             []MetricDef
	HasSLO             bool
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
	Hardware    string
}

type SQLiteReportEngine struct {
	Name            string
	Type            string
	Managed         bool
	Command         string
	Version         string
	GitCommit       string
	EndpointBaseURL string
	ServedModel     string
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
	KVCacheDtype          string
	PrefixCaching         string
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
	ID                    int64
	Profile               string
	Workload              string
	Phase                 string
	ContextWindow         int
	ContextTarget         int
	ContextSemantics      string
	ContextLabel          string
	ContextSortKey        int
	ContextMismatch       bool
	ContextMismatchNote   string
	ActiveRange           string
	RepeatIndex           int
	Concurrency           int
	SamplesRequested      int
	Status                string
	StartedAt             string
	CompletedAt           string
	WallTimeMS            string
	WallTimeMSValue       float64
	WallTimeMSKnown       bool
	CompletedRequests     int
	FailedRequests        int
	PromptTokens          string
	PromptTokensValue     int64
	PromptTokensKnown     bool
	CompletionTokens      string
	CompletionTokensValue int64
	CompletionTokensKnown bool
	TotalTokens           string
	TotalTokensValue      int64
	TotalTokensKnown      bool
	OutputTokS            string
	OutputTokSValue       float64
	OutputTokSKnown       bool
	OutputTokSStdDev      string
	PerUserOutputTokS     string
	TotalTokS             string
	RPS                   string
	LatencyMeanMS         string
	LatencyP50MS          string
	LatencyP95MS          string
	LatencyP99MS          string
	TTFTMeanMS            string
	TTFTP50MS             string
	TTFTP95MS             string
	TTFTP99MS             string
	TPOTMeanMS            string
	ITLMeanMS             string
	ITLTokenWeightedMS    string
	AchievedConcurrency   string
	FailureBreakdown      string
	GPUUtil               string
	GPUMemPeak            string
	SLOTTFTMillis         float64
	SLOE2ELMillis         float64
	SLONote               string
	SLOMetPct             string
	GoodputRPS            string
	RepeatCount           int
	ErrorType             string
	ErrorMessage          string
}

type SQLiteReportMetadataItem struct {
	Label string
	Value string
}

type SQLiteReportThroughputRow struct {
	Phase             string
	Mode              string
	Profile           string
	Workload          string
	ContextWindow     int
	ContextLabel      string
	ContextSortKey    int
	ContextMismatch   bool
	MismatchNote      string
	ActiveRange       string
	Concurrency       int
	Shape             string
	InputTokS         string
	TotalTokS         string
	OutputTokS        string
	PerUserOutputTokS string
	ThroughputTokS    string
	PerUserTokS       string
	TTFTMeanMS        string
	TTFTP95MS         string
	LatencyP95MS      string
	CompletedRequests int
	FailedRequests    int
	Status            string
	InputHeat         string
	TotalHeat         string
	OutputHeat        string
	PerUserHeat       string
	ThroughputHeat    string
	PerUserTokSHeat   string
	TTFTHeat          string
	LatencyHeat       string
	FailureHeat       string
	Baseline          bool
}

type SQLiteReportThroughputGroup struct {
	Title          string
	Profile        string
	ContextSortKey int
	ServerLimit    int
	AxisItems      []SQLiteReportMetadataItem
	Rows           []SQLiteReportThroughputComparisonRow
}

type SQLiteReportThroughputComparisonRow struct {
	Concurrency         int
	Baseline            bool
	DecodeTokS          string
	DecodePerUserTokS   string
	DecodeTTFTMeanMS    string
	DecodeTTFTMS        string
	DecodeLatencyMS     string
	DecodeOK            int
	DecodeErr           int
	DecodeShape         string
	PrefillTokS         string
	PrefillPerUserTokS  string
	PrefillTTFTMeanMS   string
	PrefillTTFTMS       string
	PrefillLatencyMS    string
	PrefillOK           int
	PrefillErr          int
	PrefillShape        string
	OK                  int
	Err                 int
	Requests            string
	DecodeTokSHeat      string
	DecodeUserHeat      string
	DecodeTTFTMeanHeat  string
	DecodeTTFTHeat      string
	DecodeLatencyHeat   string
	PrefillTokSHeat     string
	PrefillUserHeat     string
	PrefillTTFTMeanHeat string
	PrefillTTFTHeat     string
	PrefillLatencyHeat  string
	ErrHeat             string
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
	db, err := artifact.OpenReadOnly(path)
	if err != nil {
		return SQLiteReportDocument{}, err
	}
	defer db.Close()
	if err := artifact.CheckHeader(db); err != nil {
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
		loadSQLiteReportEngines,
		loadSQLiteReportProfiles,
		loadSQLiteReportWorkloads,
		loadSQLiteReportMetrics,
		loadSQLiteReportRequestDerived,
		loadSQLiteReportMeasurements,
		loadSLOGoodput,
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
	doc.Measurements, doc.RepeatDetails = aggregateRepeatMeasurements(doc.Measurements)
	doc.Legend = ReportMetrics
	doc.MetadataItems = sqliteReportMetadataItems(doc)
	doc.ThroughputRows = sqliteReportThroughputRows(doc.Measurements)
	doc.ThroughputGroups = sqliteReportThroughputGroups(doc.ThroughputRows)
	doc.PhaseSections = sqliteReportPhaseSections(doc.Measurements)
	doc.Charts = sqliteReportCharts(doc.Measurements)
	return doc, nil
}

func RenderHTMLReport(writer io.Writer, doc SQLiteReportDocument, opts HTMLReportOptions) error {
	title := strings.TrimSpace(opts.Title)
	if title == "" {
		title = bench.FirstNonEmpty(doc.Run.Name, "localperf report")
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
		"statusClass":  reportStatusClass,
		"contextLabel": contextLabel,
		"tokps":        tokenThroughputMetric,
		"seconds":      compactMilliseconds,
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
	return artifact.StoreReport(artifactPath, name, "text/html", originalPath, content)
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
	var description, startedAt, completedAt, hostname, username, cwd, gitCommit, hostJSON sql.NullString
	err := db.QueryRow(`SELECT
		id, name, description, status, created_at, started_at, completed_at,
		hostname, username, cwd, localperf_git_commit, host_json
		FROM run LIMIT 1`).Scan(
		&doc.Run.ID, &doc.Run.Name, &description, &doc.Run.Status, &doc.Run.CreatedAt,
		&startedAt, &completedAt, &hostname, &username, &cwd, &gitCommit, &hostJSON)
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
	doc.Run.Hardware = hardwareSummary(nullStringValue(hostJSON))
	return nil
}

// hardwareSummary renders the host_json hardware inventory, for example
// "NVIDIA GB10 (119 GiB, driver 580.95) / 273 GiB RAM". Absent inventory
// renders "-": missing data is information.
func hardwareSummary(hostJSON string) string {
	var host struct {
		CPU    string  `json:"cpu"`
		RAMGiB float64 `json:"ram_gib"`
		GPUs   []struct {
			Name    string  `json:"name"`
			VRAMGiB float64 `json:"vram_gib"`
			Driver  string  `json:"driver"`
		} `json:"gpus"`
	}
	if err := json.Unmarshal([]byte(hostJSON), &host); err != nil {
		return "-"
	}
	parts := []string{}
	for _, gpu := range host.GPUs {
		detail := []string{}
		if gpu.VRAMGiB > 0 {
			detail = append(detail, fmt.Sprintf("%.0f GiB", gpu.VRAMGiB))
		}
		if gpu.Driver != "" {
			detail = append(detail, "driver "+gpu.Driver)
		}
		if len(detail) > 0 {
			parts = append(parts, fmt.Sprintf("%s (%s)", gpu.Name, strings.Join(detail, ", ")))
			continue
		}
		parts = append(parts, gpu.Name)
	}
	if host.RAMGiB > 0 {
		parts = append(parts, fmt.Sprintf("%.0f GiB RAM", host.RAMGiB))
	}
	if len(parts) == 0 {
		return "-"
	}
	return strings.Join(parts, " / ")
}

func loadSQLiteReportEngines(db *sql.DB, doc *SQLiteReportDocument) error {
	rows, err := db.Query(`SELECT
		name, type, managed, COALESCE(command, ''), COALESCE(version, ''),
		COALESCE(git_commit, ''), COALESCE(endpoint_base_url, ''),
		COALESCE(json_extract(metadata_json, '$.identity.models.data[0].id'), '')
		FROM engines ORDER BY name`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var engine SQLiteReportEngine
		var managed int
		if err := rows.Scan(
			&engine.Name, &engine.Type, &managed, &engine.Command, &engine.Version,
			&engine.GitCommit, &engine.EndpointBaseURL, &engine.ServedModel,
		); err != nil {
			return err
		}
		engine.Managed = managed != 0
		doc.Engines = append(doc.Engines, engine)
	}
	return rows.Err()
}

func loadSQLiteReportProfiles(db *sql.DB, doc *SQLiteReportDocument) error {
	rows, err := db.Query(`SELECT
		id, name, model, COALESCE(context_window, 0), COALESCE(max_num_seqs, 0),
		COALESCE(max_num_batched_tokens, 0), COALESCE(gpu_memory_utilization, 0),
		managed, COALESCE(enable_sleep_mode, 0),
		COALESCE(json_extract(serve_json, '$.kv_cache_dtype'), ''),
		json_extract(serve_json, '$.enable_prefix_caching')
		FROM profiles ORDER BY context_window, name`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var profile SQLiteReportProfile
		var managed, sleep int
		var prefixCaching sql.NullInt64
		if err := rows.Scan(
			&profile.ID, &profile.Name, &profile.Model, &profile.ContextWindow,
			&profile.MaxNumSeqs, &profile.MaxNumBatchedTokens, &profile.GPUMemoryUtilization,
			&managed, &sleep, &profile.KVCacheDtype, &prefixCaching,
		); err != nil {
			return err
		}
		profile.GPUMemoryUtilizationS = displayFloat(profile.GPUMemoryUtilization)
		profile.Managed = managed != 0
		profile.EnableSleepMode = sleep != 0
		// Tri-state: prefix caching changes how prefill numbers must be
		// read, and "unknown" (unmanaged engines) must stay visible.
		switch {
		case !prefixCaching.Valid:
			profile.PrefixCaching = "unknown"
		case prefixCaching.Int64 != 0:
			profile.PrefixCaching = "on"
		default:
			profile.PrefixCaching = "off"
		}
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
		m.id, p.name, w.name, w.phase, COALESCE(p.context_window, 0),
		COALESCE(json_extract(w.metadata_json, '$.context.target'), 0),
		COALESCE(json_extract(w.metadata_json, '$.context.semantics'), ''),
		COALESCE(json_extract(w.metadata_json, '$.slo.ttft_p95_ms'), 0),
		COALESCE(json_extract(w.metadata_json, '$.slo.e2el_p95_ms'), 0),
		m.repeat_index,
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
			&measurement.ContextWindow, &measurement.ContextTarget, &measurement.ContextSemantics,
			&measurement.SLOTTFTMillis, &measurement.SLOE2ELMillis,
			&measurement.RepeatIndex, &measurement.Concurrency,
			&measurement.SamplesRequested, &measurement.Status, &startedAt, &completedAt,
			&wallTime, &measurement.CompletedRequests, &measurement.FailedRequests,
			&promptTokens, &completionTokens, &totalTokens, &outputTokS, &perUserTokS,
			&totalTokS, &errorType, &errorMessage,
		); err != nil {
			return err
		}
		applySQLiteMeasurementDisplay(&measurement, doc.MeasurementMetrics[measurement.ID], startedAt, completedAt, wallTime, promptTokens, completionTokens, totalTokens, outputTokS, perUserTokS, totalTokS, errorType, errorMessage)
		applyContextLabel(&measurement)
		applyRequestDerived(&measurement, doc.RequestDerived[measurement.ID])
		doc.Measurements = append(doc.Measurements, measurement)
	}
	return rows.Err()
}

// longOutputTokenThreshold separates prefill-style rows (start and end
// active context coincide) from rows whose active context grows materially
// during decode and must display the measured range.
const longOutputTokenThreshold = 8

// applyContextLabel implements the labeling rules in
// docs/2026-07-02-context-semantics.md: a context label may only come from a
// declared target confirmed by measurement; contradicted or undeclared rows
// are labeled by their measured shape, and the server limit is never a
// context label.
func applyContextLabel(measurement *SQLiteReportMeasurement) {
	activeStart, activeEnd, measured := measuredActiveContext(*measurement)
	measurement.ContextSortKey = int(activeEnd)
	if measurement.ContextTarget > 0 {
		measurement.ContextSortKey = measurement.ContextTarget
	}
	longOutput := measured && (activeEnd-activeStart) > longOutputTokenThreshold
	switch {
	case measurement.ContextSemantics == "active" && measurement.ContextTarget > 0:
		switch {
		case measured && withinContextBand(activeEnd, measurement.ContextTarget):
			measurement.ContextLabel = contextLabel(measurement.ContextTarget) + " active context"
			if longOutput {
				measurement.ActiveRange = activeRangeLabel(activeStart, activeEnd)
			}
		case measured:
			measurement.ContextLabel = measuredShapeLabel(*measurement)
			measurement.ContextMismatch = true
			measurement.ContextMismatchNote = fmt.Sprintf(
				"declared %s active, measured %s",
				contextLabel(measurement.ContextTarget), activeRangeLabel(activeStart, activeEnd))
		default:
			measurement.ContextLabel = contextLabel(measurement.ContextTarget) + " active (unverified)"
		}
	case measurement.ContextSemantics == "capacity" && measurement.ContextTarget > 0:
		measurement.ContextLabel = contextLabel(measurement.ContextTarget) + " capacity"
		if longOutput {
			measurement.ActiveRange = activeRangeLabel(activeStart, activeEnd)
		}
	default:
		measurement.ContextLabel = measuredShapeLabel(*measurement)
	}
}

// measuredActiveContext derives per-request mean active context at the start
// (prompt only) and end (prompt plus completion) of decode from the
// measurement aggregates.
func measuredActiveContext(measurement SQLiteReportMeasurement) (start, end float64, ok bool) {
	if measurement.CompletedRequests <= 0 || !measurement.PromptTokensKnown || !measurement.CompletionTokensKnown {
		return 0, 0, false
	}
	requests := float64(measurement.CompletedRequests)
	start = float64(measurement.PromptTokensValue) / requests
	end = start + float64(measurement.CompletionTokensValue)/requests
	return start, end, true
}

// withinContextBand checks the measured active end against the declared
// target using the contract band [0.90, 1.00].
func withinContextBand(activeEnd float64, target int) bool {
	return activeEnd >= 0.90*float64(target) && activeEnd <= float64(target)
}

func measuredShapeLabel(measurement SQLiteReportMeasurement) string {
	if shape := requestShape(measurement); shape != "-" {
		return shape
	}
	return "unlabeled"
}

func activeRangeLabel(start, end float64) string {
	return approxTokenLabel(start) + " -> " + approxTokenLabel(end) + " active"
}

func approxTokenLabel(value float64) string {
	if value >= 1024 {
		return fmt.Sprintf("~%.0fk", value/1024)
	}
	return fmt.Sprintf("~%.0f", value)
}

func applySQLiteMeasurementDisplay(measurement *SQLiteReportMeasurement, metrics map[string]SQLiteReportMetric, startedAt, completedAt sql.NullString, wallTime sql.NullFloat64, promptTokens, completionTokens, totalTokens sql.NullInt64, outputTokS, perUserTokS, totalTokS sql.NullFloat64, errorType, errorMessage sql.NullString) {
	measurement.StartedAt = nullStringValue(startedAt)
	measurement.CompletedAt = nullStringValue(completedAt)
	measurement.WallTimeMS = displayNullFloat(wallTime)
	measurement.WallTimeMSValue = nullValue(wallTime.Valid, wallTime.Float64)
	measurement.WallTimeMSKnown = wallTime.Valid
	measurement.PromptTokens = displayNullInt(promptTokens)
	measurement.PromptTokensValue = nullValue(promptTokens.Valid, promptTokens.Int64)
	measurement.PromptTokensKnown = promptTokens.Valid
	measurement.CompletionTokens = displayNullInt(completionTokens)
	measurement.CompletionTokensValue = nullValue(completionTokens.Valid, completionTokens.Int64)
	measurement.CompletionTokensKnown = completionTokens.Valid
	measurement.TotalTokens = displayNullInt(totalTokens)
	measurement.TotalTokensValue = nullValue(totalTokens.Valid, totalTokens.Int64)
	measurement.TotalTokensKnown = totalTokens.Valid
	measurement.OutputTokS = displayNullFloat(outputTokS)
	measurement.OutputTokSValue = nullValue(outputTokS.Valid, outputTokS.Float64)
	measurement.OutputTokSKnown = outputTokS.Valid
	measurement.PerUserOutputTokS = displayNullFloat(perUserTokS)
	measurement.TotalTokS = displayNullFloat(totalTokS)
	measurement.ErrorType = nullStringValue(errorType)
	measurement.ErrorMessage = nullStringValue(errorMessage)
	measurement.OutputTokSStdDev = metricDisplay(metrics, "request_output_throughput", "StdDev")
	measurement.LatencyMeanMS = metricDisplay(metrics, "latency", "Mean")
	measurement.LatencyP50MS = metricDisplayFirst(metrics, "P50", "latency")
	measurement.LatencyP95MS = metricDisplayFirst(metrics, "P95", "latency")
	measurement.LatencyP99MS = metricDisplayFirst(metrics, "P99", "latency")
	measurement.TTFTMeanMS = metricDisplayFirst(metrics, "Mean", "request_ttft", "ttft")
	measurement.TTFTP50MS = metricDisplayFirst(metrics, "P50", "request_ttft", "ttft")
	measurement.TTFTP95MS = metricDisplayFirst(metrics, "P95", "request_ttft", "ttft")
	measurement.TTFTP99MS = metricDisplayFirst(metrics, "P99", "request_ttft", "ttft")
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
		metric.MeanValue = nullValue(mean.Valid, mean.Float64)
		metric.StdDevValue = nullValue(stddev.Valid, stddev.Float64)
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
	hasRequestOutputTokS, err := sqliteRequestTableHasColumn(db, "output_tok_s")
	if err != nil {
		return err
	}
	means, hasDetailedOutputTokS, err := loadSQLiteDetailedRequestSummary(db, doc, hasRequestOutputTokS)
	if err != nil {
		return err
	}
	requestRows, err := sqliteRequestRowsByMeasurement(db)
	if err != nil {
		return err
	}
	applySQLiteAggregateRequestSummary(doc, requestRows, hasDetailedOutputTokS, &means)
	doc.RequestSummary.LatencyMeanMS = displayWeightedMean(means.latency)
	doc.RequestSummary.TTFTMeanMS = displayWeightedMean(means.ttft)
	doc.RequestSummary.TPOTMeanMS = displayWeightedMean(means.tpot)
	doc.RequestSummary.ITLMeanMS = displayWeightedMean(means.itl)
	doc.RequestSummary.OutputTokSMean = displayWeightedMean(means.outputTokS)
	return nil
}

type sqliteReportSummaryMeans struct {
	latency    sqliteReportWeightedMean
	ttft       sqliteReportWeightedMean
	tpot       sqliteReportWeightedMean
	itl        sqliteReportWeightedMean
	outputTokS sqliteReportWeightedMean
}

func loadSQLiteDetailedRequestSummary(db *sql.DB, doc *SQLiteReportDocument, includeOutputTokS bool) (sqliteReportSummaryMeans, bool, error) {
	var means sqliteReportSummaryMeans
	var latencyMean, ttftMean, tpotMean, itlMean, outputTokSMean sql.NullFloat64
	var latencyCount, ttftCount, tpotCount, itlCount, outputTokSCount int
	err := db.QueryRow(sqliteDetailedRequestSummaryQuery(includeOutputTokS)).Scan(
		&doc.RequestSummary.Total, &doc.RequestSummary.Completed, &doc.RequestSummary.Failed,
		&doc.RequestSummary.Canceled, &latencyMean, &latencyCount, &ttftMean, &ttftCount,
		&tpotMean, &tpotCount, &itlMean, &itlCount, &outputTokSMean, &outputTokSCount)
	if err != nil {
		return sqliteReportSummaryMeans{}, false, err
	}
	means.latency.addNullFloat(latencyMean, latencyCount)
	means.ttft.addNullFloat(ttftMean, ttftCount)
	means.tpot.addNullFloat(tpotMean, tpotCount)
	means.itl.addNullFloat(itlMean, itlCount)
	means.outputTokS.addNullFloat(outputTokSMean, outputTokSCount)
	return means, outputTokSCount > 0, nil
}

func sqliteDetailedRequestSummaryQuery(includeOutputTokS bool) string {
	outputTokSSelect := "CAST(NULL AS REAL), 0"
	if includeOutputTokS {
		outputTokSSelect = "AVG(output_tok_s), COUNT(output_tok_s)"
	}
	return `SELECT
		COUNT(*),
		COALESCE(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END), 0),
		COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0),
		COALESCE(SUM(CASE WHEN status = 'canceled' THEN 1 ELSE 0 END), 0),
		AVG(latency_ms), COUNT(latency_ms),
		AVG(ttft_ms), COUNT(ttft_ms),
		AVG(tpot_ms), COUNT(tpot_ms),
		AVG(itl_mean_ms), COUNT(itl_mean_ms),
		` + outputTokSSelect + `
		FROM requests`
}

func sqliteRequestTableHasColumn(db *sql.DB, name string) (bool, error) {
	rows, err := db.Query(`PRAGMA table_info(requests)`)
	if err != nil {
		return false, err
	}
	defer rows.Close()
	for rows.Next() {
		var cid, notNull, primaryKey int
		var columnName, columnType string
		var defaultValue sql.NullString
		if err := rows.Scan(&cid, &columnName, &columnType, &notNull, &defaultValue, &primaryKey); err != nil {
			return false, err
		}
		if columnName == name {
			return true, nil
		}
	}
	return false, rows.Err()
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

func applySQLiteAggregateRequestSummary(doc *SQLiteReportDocument, requestRows map[int64]int, hasDetailedOutputTokS bool, means *sqliteReportSummaryMeans) {
	for _, measurement := range doc.Measurements {
		hasRequestRows := requestRows[measurement.ID] > 0
		if !hasDetailedOutputTokS || !hasRequestRows {
			means.addMeasurementOutputTokS(measurement)
		}
		if hasRequestRows {
			continue
		}
		means.addAggregateMeasurement(doc, measurement)
	}
	doc.RequestSummary.Total = doc.RequestSummary.Completed + doc.RequestSummary.Failed + doc.RequestSummary.Canceled
}

func (means *sqliteReportSummaryMeans) addMeasurementOutputTokS(measurement SQLiteReportMeasurement) {
	if measurement.OutputTokSKnown {
		means.outputTokS.add(measurement.OutputTokSValue, 1)
	}
}

func (means *sqliteReportSummaryMeans) addAggregateMeasurement(doc *SQLiteReportDocument, measurement SQLiteReportMeasurement) {
	doc.RequestSummary.Completed += measurement.CompletedRequests
	doc.RequestSummary.Failed += measurement.FailedRequests
	doc.RequestSummary.Canceled += canceledRequestEstimate(measurement)
	metrics := doc.MeasurementMetrics[measurement.ID]
	means.latency.addMetric(metricFirst(metrics, "latency"))
	means.ttft.addMetric(metricFirst(metrics, "request_ttft", "ttft"))
	means.tpot.addMetric(metricFirst(metrics, "request_tpot", "tpot"))
	means.itl.addMetric(metricFirst(metrics, "request_itl_mean"))
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
		phase := bench.NormalizeReportPhase(measurement.Phase)
		byPhase[phase] = append(byPhase[phase], measurement)
	}
	phases := collections.SortedKeys(byPhase)
	sort.SliceStable(phases, func(i, j int) bool {
		left, right := bench.PhaseRank(phases[i]), bench.PhaseRank(phases[j])
		if left != right {
			return left < right
		}
		return phases[i] < phases[j]
	})
	out := make([]SQLiteReportPhaseSection, 0, len(phases))
	for _, phase := range phases {
		out = append(out, SQLiteReportPhaseSection{Phase: phase, Title: bench.PhaseTitle(phase), Measurements: byPhase[phase]})
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

func sqliteReportMetadataItems(doc SQLiteReportDocument) []SQLiteReportMetadataItem {
	items := []SQLiteReportMetadataItem{
		{Label: "Engine", Value: joinUnique(engineSummaries(doc.Engines), ", ")},
		{Label: "Hardware", Value: bench.FirstNonEmpty(doc.Run.Hardware, "-")},
		{Label: "Quant", Value: bench.FirstNonEmpty(inferQuantization(doc.Profiles), "-")},
		{Label: "KV", Value: bench.FirstNonEmpty(joinUnique(profileKVDtypes(doc.Profiles), ", "), "-")},
		// Active contexts come only from declared-and-verified claims; the
		// server limit is reported separately and never as a context.
		{Label: "Active contexts", Value: formatContextList(measurementPositiveInts(doc.Measurements, func(measurement SQLiteReportMeasurement) int {
			if measurement.ContextSemantics == "active" && !measurement.ContextMismatch {
				return measurement.ContextTarget
			}
			return 0
		}))},
		{Label: "Server limits", Value: formatContextList(profilePositiveInts(doc.Profiles, func(profile SQLiteReportProfile) int {
			return profile.ContextWindow
		}))},
		{Label: "Users", Value: formatIntList(measurementPositiveInts(doc.Measurements, func(measurement SQLiteReportMeasurement) int {
			return measurement.Concurrency
		}))},
		{Label: "Requests", Value: fmt.Sprintf("%d ok / %d err", doc.RequestSummary.Completed, doc.RequestSummary.Failed)},
	}
	items = append(items, servedModelMismatchItems(doc)...)
	return items
}

// servedModelMismatchItems surfaces declared-versus-self-reported model
// disagreements from the engine identity probe. Declared, checked, then
// shown: a silent mismatch is how a benchmark reports the wrong model.
func servedModelMismatchItems(doc SQLiteReportDocument) []SQLiteReportMetadataItem {
	served := map[string]string{}
	for _, engine := range doc.Engines {
		if engine.ServedModel != "" {
			served[engine.Name] = engine.ServedModel
		}
	}
	if len(served) == 0 {
		return nil
	}
	var items []SQLiteReportMetadataItem
	seen := map[string]bool{}
	for _, profile := range doc.Profiles {
		for _, servedModel := range served {
			if profile.Model == "" || servedModel == profile.Model || seen[profile.Model+servedModel] {
				continue
			}
			seen[profile.Model+servedModel] = true
			items = append(items, SQLiteReportMetadataItem{
				Label: "Model mismatch",
				Value: fmt.Sprintf("spec declares %s, server reports %s", profile.Model, servedModel),
			})
		}
	}
	return items
}

func profilePositiveInts(profiles []SQLiteReportProfile, value func(SQLiteReportProfile) int) []int {
	values := make([]int, 0, len(profiles))
	for _, profile := range profiles {
		if current := value(profile); current > 0 {
			values = append(values, current)
		}
	}
	return uniqueSortedInts(values)
}

func sqliteReportThroughputRows(measurements []SQLiteReportMeasurement) []SQLiteReportThroughputRow {
	rows := make([]SQLiteReportThroughputRow, 0, len(measurements))
	for _, measurement := range measurements {
		mode := throughputMode(measurement.Phase)
		inputTokS := inputThroughput(measurement)
		throughputTokS, perUserTokS := phaseThroughputMetrics(mode, inputTokS, measurement)
		rows = append(rows, SQLiteReportThroughputRow{
			Phase:             bench.PhaseTitle(bench.NormalizeReportPhase(measurement.Phase)),
			Mode:              mode,
			Profile:           measurement.Profile,
			Workload:          measurement.Workload,
			ContextWindow:     measurement.ContextWindow,
			ContextLabel:      measurement.ContextLabel,
			ContextSortKey:    measurement.ContextSortKey,
			ContextMismatch:   measurement.ContextMismatch,
			MismatchNote:      measurement.ContextMismatchNote,
			ActiveRange:       measurement.ActiveRange,
			Concurrency:       measurement.Concurrency,
			Shape:             throughputRowShape(measurement),
			InputTokS:         inputTokS,
			TotalTokS:         measurement.TotalTokS,
			OutputTokS:        measurement.OutputTokS,
			PerUserOutputTokS: measurement.PerUserOutputTokS,
			ThroughputTokS:    throughputTokS,
			PerUserTokS:       perUserTokS,
			TTFTMeanMS:        measurement.TTFTMeanMS,
			TTFTP95MS:         measurement.TTFTP95MS,
			LatencyP95MS:      measurement.LatencyP95MS,
			CompletedRequests: measurement.CompletedRequests,
			FailedRequests:    measurement.FailedRequests,
			Status:            measurement.Status,
			Baseline:          measurement.Concurrency == 1,
		})
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].ContextSortKey != rows[j].ContextSortKey {
			return rows[i].ContextSortKey < rows[j].ContextSortKey
		}
		leftMode, rightMode := throughputModeRank(rows[i].Mode), throughputModeRank(rows[j].Mode)
		if leftMode != rightMode {
			return leftMode < rightMode
		}
		return rows[i].Concurrency < rows[j].Concurrency
	})
	return rows
}

// throughputRowShape renders the measured token shape, extended with the
// active-context range for long-output rows.
func throughputRowShape(measurement SQLiteReportMeasurement) string {
	shape := requestShape(measurement)
	if measurement.ActiveRange != "" && shape != "-" {
		return shape + " (" + measurement.ActiveRange + ")"
	}
	return shape
}

func throughputMode(phase string) string {
	normalized := bench.NormalizeReportPhase(phase)
	switch normalized {
	case "prefill":
		return "prefill"
	case "decode":
		return "decode"
	default:
		value := strings.ToLower(strings.TrimSpace(bench.PhaseTitle(normalized)))
		if value == "" || value == "-" {
			return "run"
		}
		return value
	}
}

func throughputModeRank(mode string) int {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "decode":
		return 0
	case "prefill":
		return 1
	default:
		return 2
	}
}

func phaseThroughputMetrics(mode string, inputTokS string, measurement SQLiteReportMeasurement) (string, string) {
	if mode == "prefill" {
		return inputTokS, perUserMetric(inputTokS, measurement.Concurrency)
	}
	return measurement.OutputTokS, measurement.PerUserOutputTokS
}

func perUserMetric(value string, concurrency int) string {
	parsed, ok := parseDisplayedFloat(value)
	if !ok || concurrency <= 0 {
		return "-"
	}
	return displayFloat(parsed / float64(concurrency))
}

type throughputGroupKey struct {
	profile      string
	contextLabel string
}

type throughputAxisVisibility struct {
	profile bool
}

func sqliteReportThroughputGroups(rows []SQLiteReportThroughputRow) []SQLiteReportThroughputGroup {
	visibility := throughputAxisVisibilityForRows(rows)
	groups := []SQLiteReportThroughputGroup{}
	groupIndexes := map[throughputGroupKey]int{}
	rowIndexes := []map[int]int{}
	mismatchNotes := make([]string, 0)
	for _, row := range rows {
		key := throughputGroupKey{
			profile:      row.Profile,
			contextLabel: row.ContextLabel,
		}
		index, ok := groupIndexes[key]
		if !ok {
			index = len(groups)
			groupIndexes[key] = index
			groups = append(groups, SQLiteReportThroughputGroup{
				Title:          key.contextLabel,
				Profile:        key.profile,
				ContextSortKey: row.ContextSortKey,
				ServerLimit:    row.ContextWindow,
			})
			rowIndexes = append(rowIndexes, map[int]int{})
			mismatchNotes = append(mismatchNotes, "")
		}
		if row.ContextMismatch && row.MismatchNote != "" {
			mismatchNotes[index] = row.MismatchNote
		}
		rowIndex, ok := rowIndexes[index][row.Concurrency]
		if !ok {
			rowIndex = len(groups[index].Rows)
			rowIndexes[index][row.Concurrency] = rowIndex
			groups[index].Rows = append(groups[index].Rows, emptyThroughputComparisonRow(row.Concurrency))
		}
		applyThroughputComparisonSource(&groups[index].Rows[rowIndex], row)
	}
	for index := range groups {
		sort.SliceStable(groups[index].Rows, func(i, j int) bool {
			return groups[index].Rows[i].Concurrency < groups[index].Rows[j].Concurrency
		})
		applyThroughputComparisonHeatmaps(groups[index].Rows)
		key := throughputGroupKey{profile: groups[index].Profile, contextLabel: groups[index].Title}
		groups[index].AxisItems = throughputGroupAxisItems(key, visibility, groups[index].ServerLimit, mismatchNotes[index], groups[index].Rows)
	}
	sort.SliceStable(groups, func(i, j int) bool {
		left, right := throughputGroupSortKey(groups[i]), throughputGroupSortKey(groups[j])
		return left < right
	})
	return groups
}

func emptyThroughputComparisonRow(concurrency int) SQLiteReportThroughputComparisonRow {
	return SQLiteReportThroughputComparisonRow{
		Concurrency:         concurrency,
		Baseline:            concurrency == 1,
		DecodeTokS:          "-",
		DecodePerUserTokS:   "-",
		DecodeTTFTMeanMS:    "-",
		DecodeTTFTMS:        "-",
		DecodeLatencyMS:     "-",
		DecodeShape:         "-",
		PrefillTokS:         "-",
		PrefillPerUserTokS:  "-",
		PrefillTTFTMeanMS:   "-",
		PrefillTTFTMS:       "-",
		PrefillLatencyMS:    "-",
		Requests:            "0 / 0",
		PrefillShape:        "-",
		DecodeTokSHeat:      "heat-neutral",
		DecodeUserHeat:      "heat-neutral",
		DecodeTTFTMeanHeat:  "heat-neutral",
		DecodeTTFTHeat:      "heat-neutral",
		DecodeLatencyHeat:   "heat-neutral",
		PrefillTokSHeat:     "heat-neutral",
		PrefillUserHeat:     "heat-neutral",
		PrefillTTFTMeanHeat: "heat-neutral",
		PrefillTTFTHeat:     "heat-neutral",
		PrefillLatencyHeat:  "heat-neutral",
		ErrHeat:             "heat-neutral",
	}
}

func applyThroughputComparisonSource(target *SQLiteReportThroughputComparisonRow, source SQLiteReportThroughputRow) {
	switch source.Mode {
	case "prefill":
		target.PrefillTokS = source.ThroughputTokS
		target.PrefillPerUserTokS = source.PerUserTokS
		target.PrefillTTFTMeanMS = source.TTFTMeanMS
		target.PrefillTTFTMS = source.TTFTP95MS
		target.PrefillLatencyMS = source.LatencyP95MS
		target.PrefillOK = source.CompletedRequests
		target.PrefillErr = source.FailedRequests
		target.PrefillShape = source.Shape
	case "decode":
		target.DecodeTokS = source.ThroughputTokS
		target.DecodePerUserTokS = source.PerUserTokS
		target.DecodeTTFTMeanMS = source.TTFTMeanMS
		target.DecodeTTFTMS = source.TTFTP95MS
		target.DecodeLatencyMS = source.LatencyP95MS
		target.DecodeOK = source.CompletedRequests
		target.DecodeErr = source.FailedRequests
		target.DecodeShape = source.Shape
	default:
		target.DecodeTokS = source.ThroughputTokS
		target.DecodePerUserTokS = source.PerUserTokS
		target.DecodeTTFTMeanMS = source.TTFTMeanMS
		target.DecodeTTFTMS = source.TTFTP95MS
		target.DecodeLatencyMS = source.LatencyP95MS
		target.DecodeOK = source.CompletedRequests
		target.DecodeErr = source.FailedRequests
		target.DecodeShape = source.Shape
	}
	target.OK = target.DecodeOK + target.PrefillOK
	target.Err = target.DecodeErr + target.PrefillErr
	target.Requests = fmt.Sprintf("%d / %d", target.OK, target.Err)
}

func throughputAxisVisibilityForRows(rows []SQLiteReportThroughputRow) throughputAxisVisibility {
	profiles := map[string]struct{}{}
	for _, row := range rows {
		if strings.TrimSpace(row.Profile) != "" {
			profiles[row.Profile] = struct{}{}
		}
	}
	return throughputAxisVisibility{
		profile: len(profiles) > 1,
	}
}

func throughputGroupAxisItems(key throughputGroupKey, visibility throughputAxisVisibility, serverLimit int, mismatchNote string, rows []SQLiteReportThroughputComparisonRow) []SQLiteReportMetadataItem {
	items := []SQLiteReportMetadataItem{}
	if visibility.profile && strings.TrimSpace(key.profile) != "" && key.profile != key.contextLabel {
		items = append(items, SQLiteReportMetadataItem{Label: "Profile", Value: key.profile})
	}
	if serverLimit > 0 {
		items = append(items, SQLiteReportMetadataItem{Label: "Server limit", Value: contextLabel(serverLimit)})
	}
	if mismatchNote != "" {
		items = append(items, SQLiteReportMetadataItem{Label: "Context mismatch", Value: mismatchNote})
	}
	if shape := comparisonShapeSummary(rows, "decode"); shape != "" {
		items = append(items, SQLiteReportMetadataItem{Label: "Decode", Value: shape})
	}
	if shape := comparisonShapeSummary(rows, "prefill"); shape != "" {
		items = append(items, SQLiteReportMetadataItem{Label: "Prefill", Value: shape})
	}
	return items
}

func comparisonShapeSummary(rows []SQLiteReportThroughputComparisonRow, mode string) string {
	values := make([]string, 0, len(rows))
	for _, row := range rows {
		value := row.DecodeShape
		if mode == "prefill" {
			value = row.PrefillShape
		}
		value = strings.TrimSpace(value)
		if value == "" || value == "-" {
			continue
		}
		values = append(values, value)
	}
	return joinUnique(values, " / ")
}

func throughputGroupSortKey(group SQLiteReportThroughputGroup) string {
	return fmt.Sprintf("%012d:%s:%s", group.ContextSortKey, group.Profile, group.Title)
}

type throughputComparisonHeatmapColumn struct {
	higherIsBetter bool
	value          func(SQLiteReportThroughputComparisonRow) (float64, bool)
	set            func(*SQLiteReportThroughputComparisonRow, string)
}

type comparisonHeatmapStats struct {
	values []float64
	valid  []bool
	min    float64
	max    float64
	count  int
}

type metricDisplayField struct {
	value string
	known bool
}

func applyThroughputComparisonHeatmaps(rows []SQLiteReportThroughputComparisonRow) {
	columns := []throughputComparisonHeatmapColumn{
		{
			higherIsBetter: true,
			value: func(row SQLiteReportThroughputComparisonRow) (float64, bool) {
				return parseDisplayedFloat(row.DecodeTokS)
			},
			set: func(row *SQLiteReportThroughputComparisonRow, class string) { row.DecodeTokSHeat = class },
		},
		{
			higherIsBetter: true,
			value: func(row SQLiteReportThroughputComparisonRow) (float64, bool) {
				return parseDisplayedFloat(row.DecodePerUserTokS)
			},
			set: func(row *SQLiteReportThroughputComparisonRow, class string) { row.DecodeUserHeat = class },
		},
		{
			higherIsBetter: false,
			value: func(row SQLiteReportThroughputComparisonRow) (float64, bool) {
				return parseDisplayedFloat(row.DecodeTTFTMeanMS)
			},
			set: func(row *SQLiteReportThroughputComparisonRow, class string) {
				row.DecodeTTFTMeanHeat = class
			},
		},
		{
			higherIsBetter: false,
			value: func(row SQLiteReportThroughputComparisonRow) (float64, bool) {
				return parseDisplayedFloat(row.DecodeTTFTMS)
			},
			set: func(row *SQLiteReportThroughputComparisonRow, class string) {
				row.DecodeTTFTHeat = class
			},
		},
		{
			higherIsBetter: false,
			value: func(row SQLiteReportThroughputComparisonRow) (float64, bool) {
				return parseDisplayedFloat(row.DecodeLatencyMS)
			},
			set: func(row *SQLiteReportThroughputComparisonRow, class string) { row.DecodeLatencyHeat = class },
		},
		{
			higherIsBetter: true,
			value: func(row SQLiteReportThroughputComparisonRow) (float64, bool) {
				return parseDisplayedFloat(row.PrefillTokS)
			},
			set: func(row *SQLiteReportThroughputComparisonRow, class string) { row.PrefillTokSHeat = class },
		},
		{
			higherIsBetter: true,
			value: func(row SQLiteReportThroughputComparisonRow) (float64, bool) {
				return parseDisplayedFloat(row.PrefillPerUserTokS)
			},
			set: func(row *SQLiteReportThroughputComparisonRow, class string) { row.PrefillUserHeat = class },
		},
		{
			higherIsBetter: false,
			value: func(row SQLiteReportThroughputComparisonRow) (float64, bool) {
				return parseDisplayedFloat(row.PrefillTTFTMeanMS)
			},
			set: func(row *SQLiteReportThroughputComparisonRow, class string) {
				row.PrefillTTFTMeanHeat = class
			},
		},
		{
			higherIsBetter: false,
			value: func(row SQLiteReportThroughputComparisonRow) (float64, bool) {
				return parseDisplayedFloat(row.PrefillTTFTMS)
			},
			set: func(row *SQLiteReportThroughputComparisonRow, class string) {
				row.PrefillTTFTHeat = class
			},
		},
		{
			higherIsBetter: false,
			value: func(row SQLiteReportThroughputComparisonRow) (float64, bool) {
				return parseDisplayedFloat(row.PrefillLatencyMS)
			},
			set: func(row *SQLiteReportThroughputComparisonRow, class string) { row.PrefillLatencyHeat = class },
		},
		{
			higherIsBetter: false,
			value:          func(row SQLiteReportThroughputComparisonRow) (float64, bool) { return float64(row.Err), true },
			set:            func(row *SQLiteReportThroughputComparisonRow, class string) { row.ErrHeat = class },
		},
	}
	for _, column := range columns {
		applyThroughputComparisonHeatmapColumn(rows, column)
	}
}

func applyThroughputComparisonHeatmapColumn(rows []SQLiteReportThroughputComparisonRow, column throughputComparisonHeatmapColumn) {
	stats := collectComparisonHeatmapStats(rows, column)
	if !stats.hasSpread() {
		setComparisonHeatmapClass(rows, column, "heat-neutral")
		return
	}
	for index := range rows {
		column.set(&rows[index], stats.class(index, column))
	}
}

func collectComparisonHeatmapStats(rows []SQLiteReportThroughputComparisonRow, column throughputComparisonHeatmapColumn) comparisonHeatmapStats {
	stats := comparisonHeatmapStats{
		values: make([]float64, len(rows)),
		valid:  make([]bool, len(rows)),
		min:    math.Inf(1),
		max:    math.Inf(-1),
	}
	for index, row := range rows {
		value, ok := column.value(row)
		if ok {
			stats.record(index, value)
		}
	}
	return stats
}

func (stats *comparisonHeatmapStats) record(index int, value float64) {
	stats.values[index] = value
	stats.valid[index] = true
	stats.count++
	stats.min = math.Min(stats.min, value)
	stats.max = math.Max(stats.max, value)
}

func (stats comparisonHeatmapStats) hasSpread() bool {
	return stats.count > 0 && stats.min != stats.max
}

func (stats comparisonHeatmapStats) class(index int, column throughputComparisonHeatmapColumn) string {
	if !stats.valid[index] {
		return "heat-neutral"
	}
	ratio := (stats.values[index] - stats.min) / (stats.max - stats.min)
	if !column.higherIsBetter {
		ratio = 1 - ratio
	}
	return heatClass(ratio)
}

func setComparisonHeatmapClass(rows []SQLiteReportThroughputComparisonRow, column throughputComparisonHeatmapColumn, class string) {
	for index := range rows {
		column.set(&rows[index], class)
	}
}

func heatClass(ratio float64) string {
	if ratio < 0 {
		ratio = 0
	}
	if ratio > 1 {
		ratio = 1
	}
	return fmt.Sprintf("heat-%d", int(math.Round(ratio*5)))
}

func profileKVDtypes(profiles []SQLiteReportProfile) []string {
	values := make([]string, 0, len(profiles))
	for _, profile := range profiles {
		values = append(values, profile.KVCacheDtype)
	}
	return values
}

func engineSummaries(engines []SQLiteReportEngine) []string {
	values := make([]string, 0, len(engines))
	for _, engine := range engines {
		name := bench.FirstNonEmpty(engine.Name, engine.Type, "engine")
		if engine.Version != "" {
			values = append(values, fmt.Sprintf("%s %s", name, engine.Version))
			continue
		}
		if engine.Command != "" {
			values = append(values, fmt.Sprintf("%s via %s", name, filepath.Base(engine.Command)))
			continue
		}
		values = append(values, name)
	}
	return values
}

func inferQuantization(profiles []SQLiteReportProfile) string {
	quantizations := []string{}
	for _, profile := range profiles {
		model := strings.ToLower(profile.Model)
		switch {
		case strings.Contains(model, "nvfp4"):
			quantizations = append(quantizations, "NVFP4")
		case strings.Contains(model, "fp8"):
			quantizations = append(quantizations, "FP8")
		case strings.Contains(model, "int4"):
			quantizations = append(quantizations, "INT4")
		case strings.Contains(model, "int8"):
			quantizations = append(quantizations, "INT8")
		}
	}
	return joinUnique(quantizations, ", ")
}

func measurementPositiveInts(measurements []SQLiteReportMeasurement, value func(SQLiteReportMeasurement) int) []int {
	values := make([]int, 0, len(measurements))
	for _, measurement := range measurements {
		current := value(measurement)
		if current > 0 {
			values = append(values, current)
		}
	}
	return uniqueSortedInts(values)
}

func requestShape(measurement SQLiteReportMeasurement) string {
	if measurement.CompletedRequests <= 0 || !measurement.PromptTokensKnown || !measurement.CompletionTokensKnown {
		return "-"
	}
	input := float64(measurement.PromptTokensValue) / float64(measurement.CompletedRequests)
	output := float64(measurement.CompletionTokensValue) / float64(measurement.CompletedRequests)
	return fmt.Sprintf("%s in / %s out", displayTokenCount(input), displayTokenCount(output))
}

func inputThroughput(measurement SQLiteReportMeasurement) string {
	if !measurement.PromptTokensKnown || !measurement.WallTimeMSKnown || measurement.WallTimeMSValue <= 0 {
		return "-"
	}
	return displayFloat(float64(measurement.PromptTokensValue) / (measurement.WallTimeMSValue / 1000))
}

func displayTokenCount(value float64) string {
	return fmt.Sprintf("%.0f", value)
}

func joinUnique(values []string, sep string) string {
	seen := map[string]struct{}{}
	out := []string{}
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	sort.Strings(out)
	return strings.Join(out, sep)
}

func uniqueSortedInts(values []int) []int {
	seen := map[int]struct{}{}
	out := []int{}
	for _, value := range values {
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	sort.Ints(out)
	return out
}

func formatContextList(values []int) string {
	if len(values) == 0 {
		return "-"
	}
	parts := make([]string, 0, len(values))
	for _, value := range values {
		if value%1024 == 0 {
			parts = append(parts, fmt.Sprintf("%dk", value/1024))
			continue
		}
		parts = append(parts, fmt.Sprintf("%d", value))
	}
	return strings.Join(parts, " / ")
}

func contextLabel(value int) string {
	if value > 0 && value%1024 == 0 {
		return fmt.Sprintf("%dk", value/1024)
	}
	return fmt.Sprintf("%d", value)
}

// Context labels for groups and rows come from applyContextLabel; the
// server limit (max_model_len) is never rendered as a context title. See
// docs/2026-07-02-context-semantics.md.

func formatIntList(values []int) string {
	if len(values) == 0 {
		return "-"
	}
	parts := make([]string, 0, len(values))
	for _, value := range values {
		parts = append(parts, fmt.Sprintf("%d", value))
	}
	return strings.Join(parts, " / ")
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
	value, ok := metricDisplayFields(metric)[field]
	if !ok {
		value = metricDisplayField{value: metric.Mean, known: metric.MeanKnown}
	}
	return value.display()
}

func metricDisplayFields(metric SQLiteReportMetric) map[string]metricDisplayField {
	return map[string]metricDisplayField{
		"StdDev": {value: metric.StdDev, known: metric.StdDevKnown},
		"P50":    {value: metric.P50, known: metric.P50 != "-"},
		"P90":    {value: metric.P90, known: metric.P90 != "-"},
		"P95":    {value: metric.P95, known: metric.P95 != "-"},
		"P99":    {value: metric.P99, known: metric.P99 != "-"},
	}
}

func (field metricDisplayField) display() (string, bool) {
	if !field.known {
		return "", false
	}
	return field.value, true
}

func commandSummaryFromJSON(data string) string {
	var args []string
	if err := json.Unmarshal([]byte(data), &args); err != nil || len(args) == 0 {
		return ""
	}
	return bench.ShellQuote(args)
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

func nullValue[T any](valid bool, value T) T {
	if valid {
		return value
	}
	var zero T
	return zero
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

func tokenThroughputMetric(value string) string {
	parsed, ok := parseDisplayedFloat(value)
	if !ok {
		return value
	}
	rounded := math.Round(parsed*10) / 10
	if math.Abs(rounded) >= 100 {
		return fmt.Sprintf("%.0f", rounded)
	}
	return fmt.Sprintf("%.1f", rounded)
}

func compactMilliseconds(value string) string {
	parsed, ok := parseDisplayedFloat(value)
	if !ok {
		return value
	}
	sign := ""
	if parsed < 0 {
		sign = "-"
	}
	absMS := math.Abs(parsed)
	switch {
	case absMS < 1000:
		return fmt.Sprintf("%s%.0fms", sign, absMS)
	case absMS < 10_000:
		return sign + trimTrailingZero(fmt.Sprintf("%.1fs", absMS/1000))
	default:
		totalSeconds := int(math.Round(absMS / 1000))
		if totalSeconds < 60 {
			return fmt.Sprintf("%s%ds", sign, totalSeconds)
		}
		return fmt.Sprintf("%s%dm%02ds", sign, totalSeconds/60, totalSeconds%60)
	}
}

func parseDisplayedFloat(value string) (float64, bool) {
	value = strings.TrimSpace(value)
	if value == "" || value == "-" {
		return 0, false
	}
	parsed, err := strconv.ParseFloat(value, 64)
	return parsed, err == nil
}

func trimTrailingZero(value string) string {
	suffix := ""
	if len(value) > 0 {
		last := value[len(value)-1]
		if (last >= 'a' && last <= 'z') || (last >= 'A' && last <= 'Z') {
			suffix = value[len(value)-1:]
			value = value[:len(value)-1]
		}
	}
	if strings.Contains(value, ".") {
		value = strings.TrimRight(value, "0")
		value = strings.TrimSuffix(value, ".")
	}
	return value + suffix
}

const sqliteHTMLReportTemplate = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{.Title}}</title>
<style>
:root{color-scheme:light;--bg:#f7f8fa;--panel:#ffffff;--text:#151922;--muted:#647084;--line:#d9dee8;--bad:#b42318;--warn:#a15c07;--ok:#067647}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:13px/1.35 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header,main{max-width:1180px;margin:0 auto}
header{padding:14px 16px 10px;background:var(--panel);border-bottom:1px solid var(--line)}
main{padding:0 16px 18px}
h1{font-size:22px;line-height:1.08;margin:0 0 8px}
h2{font-size:15px;margin:14px 0 8px}
.meta-strip{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.meta-item{display:flex;gap:4px;align-items:baseline;border:1px solid var(--line);border-radius:999px;background:#fbfcfe;padding:4px 8px;max-width:100%}
.meta-item span{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.02em}
.meta-item strong{font-size:11px;overflow-wrap:anywhere}
.section{margin-top:12px}
.throughput-group{margin-top:14px}
.group-head{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:10px 0 6px}
.group-head h2{margin:0}
.group-meta{display:flex;flex-wrap:wrap;gap:5px}
.axis-item{display:inline-flex;gap:4px;align-items:baseline;border:1px solid var(--line);border-radius:999px;background:#fbfcfe;padding:2px 7px}
.axis-item span{color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.02em}
.axis-item strong{font-size:10px}
.info-box{border:1px solid var(--line);border-radius:6px;background:#fbfcfe;padding:8px 10px;color:var(--muted);font-size:11px}
.info-box p{margin:3px 0}
.info-box strong{color:var(--text)}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--panel)}
table{width:100%;border-collapse:collapse;table-layout:fixed;min-width:0}
th,td{border-bottom:1px solid var(--line);padding:6px 7px;text-align:left;vertical-align:top;white-space:normal;overflow-wrap:anywhere}
th{font-size:11px;color:var(--muted);background:#f0f3f7;font-weight:650}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.baseline-row td{border-bottom:2px solid #cbd5e1}
.heat-0{background:#fee2e2;color:#7f1d1d}
.heat-1{background:#ffedd5;color:#7c2d12}
.heat-2{background:#fef9c3;color:#713f12}
.heat-3{background:#ecfccb;color:#365314}
.heat-4{background:#dcfce7;color:#14532d}
.heat-5{background:#bbf7d0;color:#14532d}
.heat-neutral{background:#f8fafc}
.pill{display:inline-block;border-radius:999px;padding:2px 7px;font-size:11px;border:1px solid var(--line);white-space:nowrap}
.status-ok{color:var(--ok);background:#ecfdf3;border-color:#abefc6}
.status-bad{color:var(--bad);background:#fef3f2;border-color:#fecdca}
.status-warn{color:var(--warn);background:#fffaeb;border-color:#fedf89}
.status-neutral{color:var(--muted);background:#f8fafc}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.wrap{white-space:normal}
.privacy{border:1px solid var(--line);background:#fff8e6;border-radius:6px;padding:9px 11px;color:#704b00}
.artifact-line,.secondary{display:none}
.phone-table-wrap{display:none}
@media (max-width:720px){
  body{font-size:12px;background:#fff}
  header{padding:12px 14px 8px}
  main{padding:0 14px 14px}
  h1{font-size:22px;margin-bottom:8px}
  h2{font-size:15px;margin:12px 0 8px}
  .meta-strip{gap:5px}
  .meta-item{padding:3px 7px}
  .group-head{display:block;margin-top:10px}
  .group-meta{margin-top:5px}
  .axis-item{padding:2px 6px}
  .info-box{font-size:10px;padding:7px 8px}
  .desktop-table{display:none}
  .phone-table-wrap{display:block}
  .phone-table{font-size:11px}
  .phone-table th,.phone-table td{padding:5px 3px;line-height:1.15}
  .phone-table th{font-size:9px;overflow-wrap:normal}
  .phone-table td:first-child,.phone-table th:first-child{font-weight:700;text-align:right;white-space:normal}
  .phone-table td.num,.phone-table th.num{white-space:normal}
  .phone-table th.num{white-space:nowrap}
}
@media print{
  @page{size:A4 landscape;margin:10mm}
  body{background:#fff;font-size:10px}
  header,main{max-width:none;margin:0;padding:0}
  header{border-bottom:1px solid #bbb;margin-bottom:8px}
  h1{font-size:17px;margin-bottom:8px}
  h2{font-size:13px;margin:10px 0 6px}
  .meta-item{padding:2px 6px;border-radius:0}
  .meta-item span{font-size:8px}
  .meta-item strong{font-size:9px}
  .throughput-group{break-inside:avoid;margin-top:8px}
  .group-head{margin:6px 0 4px}
  .axis-item{padding:1px 4px;border-radius:0}
  .info-box{font-size:8px;padding:4px 5px}
  .table-wrap{overflow:visible;border-radius:0}
  table{width:100%;min-width:0;table-layout:fixed;font-size:8px}
  th,td{padding:3px 4px;white-space:normal;overflow-wrap:anywhere}
  td.num,th.num{white-space:normal}
  .phone-table-wrap,.secondary{display:none}
}
</style>
</head>
<body>
<header>
<h1>{{.Title}}</h1>
<div class="meta-strip">
{{range .Doc.MetadataItems}}<div class="meta-item"><span>{{.Label}}</span><strong>{{.Value}}</strong></div>{{end}}
</div>
</header>
<main>
<section class="section">
<h2>Throughput</h2>
{{range .Doc.ThroughputGroups}}
<div class="throughput-group">
<div class="group-head"><h2>{{.Title}}</h2>{{if .AxisItems}}<div class="group-meta">{{range .AxisItems}}<span class="axis-item"><span>{{.Label}}</span><strong>{{.Value}}</strong></span>{{end}}</div>{{end}}</div>
<div class="table-wrap desktop-table"><table class="essential-table">
<colgroup><col style="width:6%"><col style="width:10%"><col style="width:9%"><col style="width:10%"><col style="width:10%"><col style="width:10%"><col style="width:9%"><col style="width:10%"><col style="width:10%"><col style="width:16%"></colgroup>
<thead><tr><th class="num">Users</th><th class="num">Decode tok/s</th><th class="num">Decode/user</th><th class="num">Decode TTFT avg</th><th class="num">Decode TTFT p95</th><th class="num">Prefill tok/s</th><th class="num">Prefill/user</th><th class="num">Prefill TTFT avg</th><th class="num">Prefill TTFT p95</th><th class="num">OK / Err</th></tr></thead>
<tbody>
{{range .Rows}}
<tr class="{{if .Baseline}}baseline-row{{end}}"><td class="num">{{.Concurrency}}</td><td class="num {{.DecodeTokSHeat}}">{{tokps .DecodeTokS}}</td><td class="num {{.DecodeUserHeat}}">{{tokps .DecodePerUserTokS}}</td><td class="num {{.DecodeTTFTMeanHeat}}">{{seconds .DecodeTTFTMeanMS}}</td><td class="num {{.DecodeTTFTHeat}}">{{seconds .DecodeTTFTMS}}</td><td class="num {{.PrefillTokSHeat}}">{{tokps .PrefillTokS}}</td><td class="num {{.PrefillUserHeat}}">{{tokps .PrefillPerUserTokS}}</td><td class="num {{.PrefillTTFTMeanHeat}}">{{seconds .PrefillTTFTMeanMS}}</td><td class="num {{.PrefillTTFTHeat}}">{{seconds .PrefillTTFTMS}}</td><td class="num {{.ErrHeat}}">{{.Requests}}</td></tr>
{{end}}
</tbody>
</table></div>
<div class="table-wrap phone-table-wrap"><table class="phone-table">
<colgroup><col style="width:9%"><col style="width:11%"><col style="width:10%"><col style="width:17%"><col style="width:12%"><col style="width:10%"><col style="width:17%"><col style="width:14%"></colgroup>
<thead><tr><th class="num">Users</th><th class="num">Decode</th><th class="num">D/user</th><th class="num">D avg/p95</th><th class="num">Prefill</th><th class="num">P/user</th><th class="num">P avg/p95</th><th class="num">OK/Err</th></tr></thead>
<tbody>
{{range .Rows}}
<tr class="{{if .Baseline}}baseline-row{{end}}"><td class="num">{{.Concurrency}}</td><td class="num {{.DecodeTokSHeat}}">{{tokps .DecodeTokS}}</td><td class="num {{.DecodeUserHeat}}">{{tokps .DecodePerUserTokS}}</td><td class="num {{.DecodeTTFTHeat}}">{{seconds .DecodeTTFTMeanMS}} / {{seconds .DecodeTTFTMS}}</td><td class="num {{.PrefillTokSHeat}}">{{tokps .PrefillTokS}}</td><td class="num {{.PrefillUserHeat}}">{{tokps .PrefillPerUserTokS}}</td><td class="num {{.PrefillTTFTHeat}}">{{seconds .PrefillTTFTMeanMS}} / {{seconds .PrefillTTFTMS}}</td><td class="num {{.ErrHeat}}">{{.Requests}}</td></tr>
{{end}}
</tbody>
</table></div>
</div>
{{end}}
<div class="info-box">
{{range .Doc.Legend}}<p><strong>{{.Label}}</strong> = {{.Definition}}</p>
{{end}}<p>Times are engine milliseconds rendered as compact durations. Missing data renders as "-", never a substitute number.</p>
</div>
</section>
{{range .Doc.PhaseSections}}
<section class="section secondary">
<h2>{{.Title}} Detail</h2>
<div class="table-wrap"><table>
<thead><tr><th>Profile</th><th>Workload</th><th>Context</th><th class="num">Conc.</th><th>Status</th><th class="num">Done</th><th class="num">Failed</th><th class="num">RPS</th><th class="num">Total tok/s</th><th class="num">Output tok/s</th><th class="num">Out/user</th><th class="num">TTFT mean</th><th class="num">TTFT p50</th><th class="num">TTFT p95</th><th class="num">TTFT p99</th><th class="num">Latency p50</th><th class="num">Latency p95</th><th class="num">Latency p99</th><th class="num">TPOT mean</th><th class="num">ITL tok-wt</th><th class="num">GPU util a/p</th><th class="num">GPU mem peak</th>{{if $.Doc.HasSLO}}<th class="num">% in SLO</th><th class="num">Goodput req/s</th>{{end}}</tr></thead>
<tbody>
{{range .Measurements}}
<tr><td>{{.Profile}}</td><td>{{.Workload}}</td><td>{{.ContextLabel}}{{if .ContextMismatch}} <span class="pill status-bad" title="{{.ContextMismatchNote}}">mismatch</span>{{end}}</td><td class="num">{{.Concurrency}}{{if gt .RepeatCount 1}} &times;{{.RepeatCount}}{{end}}{{if .AchievedConcurrency}} <span class="pill status-warn" title="time-weighted mean in-flight requests">{{.AchievedConcurrency}}</span>{{end}}</td><td><span class="pill {{statusClass .Status}}">{{.Status}}</span></td><td class="num">{{.CompletedRequests}}</td><td class="num">{{.FailedRequests}}{{if .FailureBreakdown}} <span title="{{.FailureBreakdown}}">({{.FailureBreakdown}})</span>{{end}}</td><td class="num">{{.RPS}}</td><td class="num">{{.TotalTokS}}</td><td class="num">{{.OutputTokS}}</td><td class="num">{{.PerUserOutputTokS}}</td><td class="num">{{.TTFTMeanMS}}</td><td class="num">{{.TTFTP50MS}}</td><td class="num">{{.TTFTP95MS}}</td><td class="num">{{.TTFTP99MS}}</td><td class="num">{{.LatencyP50MS}}</td><td class="num">{{.LatencyP95MS}}</td><td class="num">{{.LatencyP99MS}}</td><td class="num">{{.TPOTMeanMS}}</td><td class="num">{{.ITLTokenWeightedMS}}</td><td class="num">{{.GPUUtil}}</td><td class="num">{{.GPUMemPeak}}</td>{{if $.Doc.HasSLO}}<td class="num">{{if .SLOMetPct}}<span title="{{.SLONote}}">{{.SLOMetPct}}</span>{{else}}-{{end}}</td><td class="num">{{if .GoodputRPS}}{{.GoodputRPS}}{{else}}-{{end}}</td>{{end}}</tr>
{{end}}
</tbody>
</table></div>
</section>
{{end}}
{{if .Doc.RepeatDetails}}
<section class="section secondary">
<h2>Repeats</h2>
<details><summary>Per-repeat rows behind aggregated points</summary>
<div class="table-wrap"><table>
<thead><tr><th>Profile</th><th>Workload</th><th class="num">Conc.</th><th class="num">Repeat</th><th>Status</th><th class="num">Done</th><th class="num">Failed</th><th class="num">Output tok/s</th><th class="num">Out/user</th><th class="num">TTFT mean</th><th class="num">Latency p95</th></tr></thead>
<tbody>
{{range .Doc.RepeatDetails}}
<tr><td>{{.Profile}}</td><td>{{.Workload}}</td><td class="num">{{.Concurrency}}</td><td class="num">{{.RepeatIndex}}</td><td><span class="pill {{statusClass .Status}}">{{.Status}}</span></td><td class="num">{{.CompletedRequests}}</td><td class="num">{{.FailedRequests}}</td><td class="num">{{.OutputTokS}}</td><td class="num">{{.PerUserOutputTokS}}</td><td class="num">{{.TTFTMeanMS}}</td><td class="num">{{.LatencyP95MS}}</td></tr>
{{end}}
</tbody>
</table></div>
</details>
</section>
{{end}}
<section class="section secondary">
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
<section class="section secondary">
<h2>Profiles</h2>
<div class="table-wrap"><table>
<thead><tr><th>Name</th><th>Model</th><th class="num">Server limit</th><th class="num">Max seqs</th><th class="num">Batched tokens</th><th class="num">GPU memory util.</th><th>KV cache</th><th>Prefix cache</th><th>Managed</th><th>Sleep</th></tr></thead>
<tbody>{{range .Doc.Profiles}}<tr><td>{{.Name}}</td><td>{{.Model}}</td><td class="num">{{.ContextWindow}}</td><td class="num">{{.MaxNumSeqs}}</td><td class="num">{{.MaxNumBatchedTokens}}</td><td class="num">{{.GPUMemoryUtilizationS}}</td><td>{{.KVCacheDtype}}</td><td>{{.PrefixCaching}}</td><td>{{.Managed}}</td><td>{{.EnableSleepMode}}</td></tr>{{end}}</tbody>
</table></div>
</section>
<section class="section secondary">
<h2>Events</h2>
<div class="grid summary">{{range .Doc.EventCounts}}<div class="stat"><span>{{.Name}}</span><strong>{{.Count}}</strong></div>{{end}}</div>
{{if .Doc.NotableEvents}}<h3>Notable Events</h3><div class="table-wrap"><table><thead><tr><th>Time</th><th>Level</th><th>Type</th><th>Profile</th><th>Workload</th><th>Message</th></tr></thead><tbody>{{range .Doc.NotableEvents}}<tr><td>{{.Timestamp}}</td><td>{{.Level}}</td><td>{{.Type}}</td><td>{{.Profile}}</td><td>{{.Workload}}</td><td class="wrap">{{.Message}}</td></tr>{{end}}</tbody></table></div>{{end}}
</section>
<section class="section secondary">
<h2>Commands</h2>
<div class="table-wrap"><table><thead><tr><th>Phase</th><th>Status</th><th>Exit</th><th>Started</th><th>Completed</th><th>Command</th></tr></thead><tbody>{{range .Doc.Commands}}<tr><td>{{.Phase}}</td><td><span class="pill {{statusClass .Status}}">{{.Status}}</span></td><td>{{.ExitCode}}</td><td>{{.StartedAt}}</td><td>{{.Completed}}</td><td class="mono wrap">{{.Argv}}</td></tr>{{end}}</tbody></table></div>
</section>
<section class="section secondary">
<h2>Artifact Contents</h2>
<div class="table-wrap"><table><thead><tr><th>Kind</th><th class="num">Count</th><th class="num">Uncompressed bytes</th></tr></thead><tbody>{{range .Doc.ArtifactSummaries}}<tr><td>{{.Kind}}</td><td class="num">{{.Count}}</td><td class="num">{{.UncompressedSizeBytes}}</td></tr>{{end}}</tbody></table></div>
</section>
<section class="section secondary">
<h2>Privacy</h2>
<div class="privacy">This standalone report is rendered from normalized SQLite metrics. It does not include raw prompts, generated text, log bodies, or raw artifact contents.</div>
</section>
</main>
</body>
</html>
`
