package benchcli

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/dutifuldev/localperf/internal/artifact"
	"github.com/dutifuldev/localperf/internal/collections"
	"github.com/dutifuldev/localperf/internal/report"
	"github.com/dutifuldev/localperf/internal/sweepplan"
	"github.com/dutifuldev/localperf/internal/viewer"
	"github.com/dutifuldev/localperf/internal/vllmbench"
)

const defaultHTTPLoadMinMemAvailableGiB = 40.0

type commandHandlers map[string]func([]string)

var rootHandlers = commandHandlers{
	"bench":    VLLMBenchMain,
	"artifact": runArtifact,
	"sweep":    runSweep,
	"view":     runView,
}

var artifactHandlers = commandHandlers{
	"check":  runArtifactCheck,
	"render": runArtifactRender,
	"merge":  runArtifactMerge,
}

// runArtifactMerge combines run artifacts into one model-level SQLite file;
// see docs/2026-07-02-default-inference-sweep.md, Model-Level Artifacts.
func runArtifactMerge(args []string) {
	flags := flag.NewFlagSet("artifact merge", flag.ExitOnError)
	into := flags.String("into", "", "destination model-level artifact (required)")
	_ = flags.Parse(args)
	sources := flags.Args()
	if strings.TrimSpace(*into) == "" || len(sources) == 0 {
		fmt.Fprintln(os.Stderr, "usage: localperf artifact merge --into runs/models/model.sqlite src1.sqlite [src2.sqlite ...]")
		os.Exit(2)
	}
	summary, err := artifact.Merge(*into, sources)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	for _, run := range summary.MergedRuns {
		fmt.Printf("merged: %s\n", run)
	}
	for _, run := range summary.SkippedRuns {
		fmt.Printf("skipped (already present): %s\n", run)
	}
	fmt.Printf("artifact ok: %s\n", *into)
}

var sweepHandlers = commandHandlers{
	"plan": runSweepPlan,
}

func runSweep(args []string) {
	dispatchCommand(args, usageRoot, sweepHandlers)
}

// runSweepPlan emits the default context/concurrency sweep spec with
// contract-compliant shapes and declared context semantics; see
// docs/2026-07-02-default-inference-sweep.md.
func runSweepPlan(args []string) {
	flags := flag.NewFlagSet("sweep plan", flag.ExitOnError)
	model := flags.String("model", "", "model identifier to benchmark (required)")
	contexts := flags.String("contexts", "8k,16k,32k,64k", "comma-separated active-context ladder (e.g. 8k,16k,32k); 128k and above are capped at c4")
	concurrency := flags.String("concurrency", "1,4,8,16,32", "comma-separated concurrency levels")
	repeats := flags.Int("repeats", 1, "repeats per measurement")
	numPrompts := flags.Int("num-prompts", 0, "fixed prompts per measurement; default scales with concurrency")
	promptsPerUser := flags.Int("prompts-per-user", 0, "prompts per concurrent user (default 2, floor 8 per point)")
	reference := flags.Bool("reference", true, "include the 4k max-throughput-reference capacity family")
	stress := flags.Bool("stress", false, "add long-output decode spot checks (4096 tokens at 32k c4, 64k c1/c4) and the 128k points")
	memFloor := flags.Float64("min-mem-available-gib", 0, "safety memory floor in GiB (default 40)")
	out := flags.String("out", "", "output spec path (default stdout)")
	_ = flags.Parse(args)
	contextValues, err := parseTokenList(*contexts)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	concurrencyValues, err := parseIntList(*concurrency)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	spec, err := sweepplan.Plan(sweepplan.PlanRequest{
		Model:              *model,
		Contexts:           contextValues,
		Concurrency:        concurrencyValues,
		Repeats:            *repeats,
		NumPrompts:         *numPrompts,
		PromptsPerUser:     *promptsPerUser,
		IncludeReference:   *reference,
		IncludeStress:      *stress,
		MinMemAvailableGiB: *memFloor,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	encoded, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	encoded = append(encoded, '\n')
	if strings.TrimSpace(*out) == "" {
		_, _ = os.Stdout.Write(encoded)
		return
	}
	if err := os.WriteFile(*out, encoded, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("spec: %s\n", *out)
}

// parseTokenList parses values such as "8k,16k,32768" into token counts.
func parseTokenList(value string) ([]int, error) {
	var values []int
	for _, part := range strings.Split(value, ",") {
		part = strings.TrimSpace(strings.ToLower(part))
		if part == "" {
			continue
		}
		multiplier := 1
		if strings.HasSuffix(part, "k") {
			multiplier = 1024
			part = strings.TrimSuffix(part, "k")
		}
		parsed, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid context value %q", part)
		}
		values = append(values, parsed*multiplier)
	}
	return values, nil
}

func parseIntList(value string) ([]int, error) {
	var values []int
	for _, part := range strings.Split(value, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		parsed, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid concurrency value %q", part)
		}
		values = append(values, parsed)
	}
	return values, nil
}

func VLLMBenchMain(args []string) {
	if len(args) < 1 {
		usage()
		os.Exit(2)
	}
	switch args[0] {
	case "plan":
		runPlan(args[1:])
	case "run":
		runBench(args[1:])
	case "http-load":
		runHTTPLoad(args[1:])
	case "artifact":
		runArtifact(args[1:])
	default:
		usage()
		os.Exit(2)
	}
}

func Main(args []string) {
	dispatchCommand(args, usageRoot, rootHandlers)
}

func dispatchCommand(args []string, usageFunc func(), handlers commandHandlers) {
	if len(args) < 1 {
		usageFunc()
		os.Exit(2)
	}
	if handler := handlers[args[0]]; handler != nil {
		handler(args[1:])
		return
	}
	usageFunc()
	os.Exit(2)
}

func runPlan(args []string) {
	flags := flag.NewFlagSet("plan", flag.ExitOnError)
	specPath := flags.String("spec", "", "benchmark spec JSON file")
	runDir := flags.String("run-dir", "", "optional run directory for result path planning")
	jsonOutput := flags.Bool("json", false, "print JSON instead of text")
	overrides := addOverrideFlags(flags)
	filterFlags := addFilterFlags(flags)
	_ = flags.Parse(args)
	spec := mustLoadSpec(*specPath, filterFlags.Filter())
	applyOverrides(&spec, overrides)
	dir := vllmbench.RunDir(*runDir, spec, time.Now())
	if err := vllmbench.PrepareDatasets(context.Background(), &spec, dir); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	plan := vllmbench.BuildPlan(spec, dir)
	if *jsonOutput {
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		_ = encoder.Encode(plan)
		return
	}
	fmt.Printf("name: %s\n", spec.Name)
	fmt.Printf("model: %s\n", spec.Model)
	fmt.Printf("run dir: %s\n", dir)
	fmt.Printf("memory floor: %.1f GiB MemAvailable\n", spec.Safety.MinMemAvailableGiB)
	fmt.Printf("planned runs: %d\n", len(plan))
	fmt.Println("profiles:")
	for _, profile := range spec.Profiles {
		fmt.Printf("- profile=%s managed=%t sleep=%t port=%d\n", profile.Name, profile.Managed, profile.EnableSleepMode, profile.Port)
		if profile.Managed {
			fmt.Printf("  %s\n", vllmbench.CommandSummary(vllmbench.ServeCommand(spec, profile)))
		}
	}
	fmt.Println("workloads:")
	for _, planned := range plan {
		command := vllmbench.LoadCommand(spec, planned)
		fmt.Printf("- profile=%s workload=%s concurrency=%d result=%s\n", planned.Profile.Name, planned.Workload.Name, planned.Concurrency, planned.ResultFile)
		fmt.Printf("  %s\n", vllmbench.ShellQuote(command.Args))
	}
}

func runBench(args []string) {
	flags := flag.NewFlagSet("run", flag.ExitOnError)
	specPath := flags.String("spec", "", "benchmark spec JSON file")
	runDir := flags.String("run-dir", "", "optional run directory")
	artifactPath := flags.String("artifact", "", "optional artifact path; an existing artifact is appended to (model-level accumulation)")
	resume := flags.Bool("resume", false, "skip planned runs whose result files already completed; requires --run-dir of the previous attempt")
	dryRun := flags.Bool("dry-run", false, "write planned artifacts without launching vLLM or benchmark commands")
	timeout := flags.Duration("timeout", 0, "optional overall timeout, for example 2h")
	overrides := addOverrideFlags(flags)
	filterFlags := addFilterFlags(flags)
	_ = flags.Parse(args)
	spec := mustLoadSpec(*specPath, filterFlags.Filter())
	applyOverrides(&spec, overrides)
	ctx := context.Background()
	cancel := func() {}
	if *timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, *timeout)
	}
	defer cancel()
	summary, err := vllmbench.Execute(ctx, spec, vllmbench.RunOptions{
		RunDir:           *runDir,
		ArtifactPath:     *artifactPath,
		DryRun:           *dryRun,
		Resume:           *resume,
		OriginalSpecPath: *specPath,
	})
	fmt.Printf("run dir: %s\n", summary.RunDir)
	fmt.Printf("planned: %d completed: %d failed: %d skipped: %d\n", summary.PlannedRuns, summary.CompletedRuns, summary.FailedRuns, summary.SkippedRuns)
	if summary.ArtifactPath != "" {
		fmt.Printf("artifact: %s\n", summary.ArtifactPath)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runView(args []string) {
	config, err := parseViewFlags(args, flag.ExitOnError)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	err = viewer.Serve(ctx, viewer.ServerConfig{
		Addr:        config.addr,
		Title:       config.title,
		Paths:       config.paths,
		OpenBrowser: config.open,
		Out:         os.Stdout,
		Err:         os.Stderr,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

type viewConfig struct {
	addr  string
	title string
	paths []string
	open  bool
}

func parseViewFlags(args []string, errorHandling flag.ErrorHandling) (viewConfig, error) {
	flags := flag.NewFlagSet("view", errorHandling)
	addr := flags.String("addr", "127.0.0.1:0", "viewer listen address")
	title := flags.String("title", "localperf viewer", "viewer title")
	open := flags.Bool("open", false, "open the viewer in a browser")
	noOpen := flags.Bool("no-open", false, "do not open the viewer in a browser")
	positionalPaths, parseArgs := viewParseArgs(args)
	if err := flags.Parse(parseArgs); err != nil {
		return viewConfig{}, err
	}
	paths := append([]string{}, positionalPaths...)
	paths = append(paths, flags.Args()...)
	if len(paths) == 0 {
		return viewConfig{}, fmt.Errorf("missing SQLite report path")
	}
	if *open && *noOpen {
		return viewConfig{}, fmt.Errorf("--open and --no-open cannot both be set")
	}
	return viewConfig{
		addr:  *addr,
		title: *title,
		paths: paths,
		open:  *open && !*noOpen,
	}, nil
}

func viewParseArgs(args []string) ([]string, []string) {
	var paths []string
	parseArgs := make([]string, 0, len(args))
	for index := 0; index < len(args); index++ {
		arg := args[index]
		if !strings.HasPrefix(arg, "-") {
			paths = append(paths, arg)
			continue
		}
		parseArgs = append(parseArgs, arg)
		if viewFlagNeedsValue(arg) && !strings.Contains(arg, "=") && index+1 < len(args) {
			index++
			parseArgs = append(parseArgs, args[index])
		}
	}
	return paths, parseArgs
}

func viewFlagNeedsValue(arg string) bool {
	if equals := strings.Index(arg, "="); equals >= 0 {
		arg = arg[:equals]
	}
	switch arg {
	case "-addr", "--addr", "-title", "--title":
		return true
	default:
		return false
	}
}

func runHTTPLoad(args []string) {
	flags := flag.NewFlagSet("http-load", flag.ExitOnError)
	backend := flags.String("backend", "openai-chat", "OpenAI-compatible backend type")
	baseURL := flags.String("base-url", "", "OpenAI-compatible endpoint base URL")
	model := flags.String("model", "", "served model name")
	datasetName := flags.String("dataset-name", "random", "dataset name")
	numPrompts := flags.Int("num-prompts", 0, "number of prompts")
	maxConcurrency := flags.Int("max-concurrency", 1, "maximum concurrent requests")
	requestRate := flags.String("request-rate", "inf", "request rate")
	resultFilename := flags.String("result-filename", "", "result JSON path")
	datasetPath := flags.String("dataset-path", "", "canonical request JSONL path for structured datasets")
	randomInputLen := flags.Int("random-input-len", 0, "random dataset input tokens")
	randomOutputLen := flags.Int("random-output-len", 0, "random dataset output tokens")
	endpoint := flags.String("endpoint", "", "endpoint path")
	extraBody := flags.String("extra-body", "", "extra OpenAI request body JSON object")
	ignoreEOS := flags.Bool("ignore-eos", false, "ask the engine to ignore EOS")
	temperature := flags.String("temperature", "", "temperature")
	noStream := flags.Bool("no-stream", false, "disable SSE streaming; the run then records no TTFT")
	timeout := flags.Duration("timeout", 0, "optional timeout")
	minMemAvailableGiB := flags.Float64("min-mem-available-gib", defaultHTTPLoadMinMemAvailableGiB, "memory floor")
	logPath := flags.String("log", "", "log path")
	_ = flags.Parse(args)
	profile, err := profileFromBaseURL(*baseURL, *model)
	exitOnError(err)
	if *minMemAvailableGiB <= 0 {
		exitOnError(fmt.Errorf("--min-mem-available-gib must be positive"))
	}
	workload, err := httpLoadWorkload(*backend, *datasetName, *requestRate, *endpoint, *datasetPath, *extraBody, *temperature, *ignoreEOS, *noStream, *numPrompts, *maxConcurrency, *randomInputLen, *randomOutputLen)
	exitOnError(err)
	if strings.TrimSpace(*resultFilename) == "" {
		exitOnError(fmt.Errorf("missing --result-filename"))
	}
	spec := vllmbench.Spec{Model: *model, Safety: vllmbench.SafetyConfig{MinMemAvailableGiB: *minMemAvailableGiB}}
	vllmbench.ApplyDefaults(&spec)
	if *timeout > 0 {
		spec.Safety.WorkloadTimeoutSec = timeoutSeconds(*timeout)
	}
	planned := vllmbench.PlannedRun{Profile: profile, Workload: workload, Concurrency: *maxConcurrency, ResultFile: *resultFilename}
	if strings.TrimSpace(*logPath) == "" {
		*logPath = strings.TrimSuffix(*resultFilename, filepath.Ext(*resultFilename)) + ".log"
	}
	if err := vllmbench.RunHTTPBench(context.Background(), spec, planned, *logPath); err != nil {
		exitOnError(err)
	}
	rows, err := vllmbench.ParseResultFile(*resultFilename)
	exitOnError(err)
	if len(rows) == 0 {
		exitOnError(fmt.Errorf("no result rows"))
	}
	row := rows[0]
	fmt.Printf("completed: %d failed: %d output_tok_s: %.3f request_output_tok_s_stddev: %.3f\n", row.Completed, row.Failed, row.OutputTokensPerSec, row.OutputTokSecStdDev)
}

func runArtifact(args []string) {
	dispatchCommand(args, usage, artifactHandlers)
}

func runArtifactCheck(args []string) {
	flags := flag.NewFlagSet("artifact check", flag.ExitOnError)
	path := flags.String("path", "", "SQLite artifact path")
	_ = flags.Parse(args)
	if strings.TrimSpace(*path) == "" {
		if flags.NArg() == 1 {
			*path = flags.Arg(0)
		}
	}
	if strings.TrimSpace(*path) == "" {
		fmt.Fprintln(os.Stderr, "missing artifact path")
		os.Exit(2)
	}
	if err := artifact.Check(*path); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("artifact ok: %s\n", *path)
}

func runArtifactRender(args []string) {
	config, err := parseArtifactRenderFlags(args, flag.ExitOnError)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if config.includeRaw {
		fmt.Fprintln(os.Stderr, "--include-raw is not implemented yet")
		os.Exit(2)
	}
	if err := report.WriteSQLiteHTMLReport(config.path, config.output, report.HTMLReportOptions{Title: config.title, Store: config.store}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	outPath := config.output
	if strings.TrimSpace(outPath) == "" {
		outPath = strings.TrimSuffix(config.path, filepath.Ext(config.path)) + ".html"
	}
	fmt.Printf("html: %s\n", outPath)
	if config.store {
		fmt.Printf("stored: %s\n", config.path)
	}
}

type artifactRenderConfig struct {
	path       string
	output     string
	title      string
	store      bool
	includeRaw bool
}

func parseArtifactRenderFlags(args []string, errorHandling flag.ErrorHandling) (artifactRenderConfig, error) {
	flags := flag.NewFlagSet("artifact render", errorHandling)
	path := flags.String("path", "", "SQLite artifact path")
	output := flags.String("output", "", "standalone HTML output path; defaults beside the artifact")
	title := flags.String("title", "", "optional report title")
	store := flags.Bool("store", false, "store report.html back into the SQLite artifact")
	includeRaw := flags.Bool("include-raw", false, "reserved for explicit raw artifact rendering")
	positionalPath, parseArgs := artifactRenderParseArgs(args)
	if err := flags.Parse(parseArgs); err != nil {
		return artifactRenderConfig{}, err
	}
	if strings.TrimSpace(*path) == "" && positionalPath != "" {
		*path = positionalPath
	}
	if strings.TrimSpace(*path) == "" && flags.NArg() > 0 {
		*path = flags.Arg(0)
	}
	if strings.TrimSpace(*path) == "" {
		return artifactRenderConfig{}, fmt.Errorf("missing artifact path")
	}
	return artifactRenderConfig{
		path:       *path,
		output:     *output,
		title:      *title,
		store:      *store,
		includeRaw: *includeRaw,
	}, nil
}

func artifactRenderParseArgs(args []string) (string, []string) {
	positionalPath := ""
	parseArgs := make([]string, 0, len(args))
	for index := 0; index < len(args); index++ {
		arg := args[index]
		if positionalPath == "" && !strings.HasPrefix(arg, "-") {
			positionalPath = arg
			continue
		}
		parseArgs = append(parseArgs, arg)
		if artifactRenderFlagNeedsValue(arg) && !strings.Contains(arg, "=") && index+1 < len(args) {
			index++
			parseArgs = append(parseArgs, args[index])
		}
	}
	return positionalPath, parseArgs
}

func artifactRenderFlagNeedsValue(arg string) bool {
	if equals := strings.Index(arg, "="); equals >= 0 {
		arg = arg[:equals]
	}
	switch arg {
	case "-path", "--path", "-output", "--output", "-title", "--title":
		return true
	default:
		return false
	}
}

func profileFromBaseURL(rawURL, model string) (vllmbench.Profile, error) {
	if strings.TrimSpace(rawURL) == "" {
		return vllmbench.Profile{}, fmt.Errorf("missing --base-url")
	}
	if strings.TrimSpace(model) == "" {
		return vllmbench.Profile{}, fmt.Errorf("missing --model")
	}
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return vllmbench.Profile{}, err
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return vllmbench.Profile{}, fmt.Errorf("only http and https base URLs are supported")
	}
	if parsed.RawQuery != "" || parsed.Fragment != "" {
		return vllmbench.Profile{}, fmt.Errorf("base URL must not include query or fragment")
	}
	host := parsed.Hostname()
	if host == "" {
		return vllmbench.Profile{}, fmt.Errorf("base URL must include host")
	}
	port, err := parsedBaseURLPort(parsed)
	if err != nil {
		return vllmbench.Profile{}, err
	}
	return vllmbench.Profile{Name: "http-load", Host: host, Port: port, Model: model, EndpointBaseURL: normalizedBaseURL(parsed)}, nil
}

func parsedBaseURLPort(parsed *url.URL) (int, error) {
	if portString := parsed.Port(); portString != "" {
		return strconv.Atoi(portString)
	}
	if parsed.Scheme == "https" {
		return 443, nil
	}
	return 80, nil
}

func normalizedBaseURL(parsed *url.URL) string {
	return vllmbench.NormalizeEndpointBaseURL(parsed.Scheme + "://" + parsed.Host + parsed.EscapedPath())
}

func timeoutSeconds(timeout time.Duration) int {
	seconds := int(timeout / time.Second)
	if timeout%time.Second != 0 {
		seconds++
	}
	if seconds < 1 {
		return 1
	}
	return seconds
}

func httpLoadWorkload(backend, datasetName, requestRate, endpoint, datasetPath, extraBody, temperature string, ignoreEOS, noStream bool, numPrompts, maxConcurrency, randomInputLen, randomOutputLen int) (vllmbench.Workload, error) {
	if numPrompts <= 0 {
		return vllmbench.Workload{}, fmt.Errorf("--num-prompts must be positive")
	}
	if maxConcurrency <= 0 {
		return vllmbench.Workload{}, fmt.Errorf("--max-concurrency must be positive")
	}
	datasetPath, err := absoluteDatasetPath(datasetPath)
	if err != nil {
		return vllmbench.Workload{}, err
	}
	if strings.TrimSpace(datasetPath) != "" && strings.TrimSpace(datasetName) == "random" {
		datasetName = "custom"
	}
	var temp *float64
	if strings.TrimSpace(temperature) != "" {
		parsed, err := strconv.ParseFloat(temperature, 64)
		if err != nil {
			return vllmbench.Workload{}, err
		}
		temp = &parsed
	}
	// This synthetic single-workload spec exists only to validate the exec
	// boundary of `bench http-load`; it is never persisted to an artifact.
	// The real context claim lives in the parent spec that planned the run,
	// so declare the only thing knowable here: for random traffic the active
	// context is exactly the requested shape; otherwise a capacity point
	// against a profile with no locally declared limit.
	contextTarget := 1
	contextSemantics := vllmbench.ContextSemanticsCapacity
	if datasetName == "random" && randomInputLen+randomOutputLen > 0 {
		contextTarget = randomInputLen + randomOutputLen
		contextSemantics = vllmbench.ContextSemanticsActive
	}
	workload := vllmbench.Workload{
		Name:             "http-load",
		Profiles:         []string{"http-load"},
		ContextTarget:    contextTarget,
		ContextSemantics: contextSemantics,
		LoadGenerator:    vllmbench.LoadGeneratorHTTP,
		BenchmarkTrafficConfig: vllmbench.BenchmarkTrafficConfig{
			Backend:         backend,
			DatasetName:     datasetName,
			DatasetPath:     datasetPath,
			RequestRate:     requestRate,
			Endpoint:        endpoint,
			ExtraBody:       extraBody,
			RandomInputLen:  randomInputLen,
			RandomOutputLen: randomOutputLen,
		},
		NumPrompts:     numPrompts,
		MaxConcurrency: []int{maxConcurrency},
		Temperature:    temp,
		IgnoreEOS:      ignoreEOS,
	}
	if noStream {
		stream := false
		workload.Stream = &stream
	}
	if strings.TrimSpace(datasetPath) != "" {
		workload.Dataset.Prepared = vllmbench.DatasetMaterialization{
			CanonicalPath: datasetPath,
			RequestCount:  numPrompts,
		}
	}
	spec := vllmbench.Spec{
		Name:     "http-load",
		Model:    "model",
		Safety:   vllmbench.SafetyConfig{MinMemAvailableGiB: defaultHTTPLoadMinMemAvailableGiB},
		Profiles: []vllmbench.Profile{{Name: "http-load", Model: "model", Port: 1}},
		Workloads: []vllmbench.Workload{
			workload,
		},
	}
	vllmbench.ApplyDefaults(&spec)
	if err := vllmbench.ValidateSpec(spec); err != nil {
		return vllmbench.Workload{}, err
	}
	return spec.Workloads[0], nil
}

func absoluteDatasetPath(path string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" || filepath.IsAbs(path) {
		return path, nil
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	return absolute, nil
}

func mustLoadSpec(path string, filter vllmbench.Filter) vllmbench.Spec {
	if strings.TrimSpace(path) == "" {
		fmt.Fprintln(os.Stderr, "missing --spec")
		os.Exit(2)
	}
	spec, err := vllmbench.LoadSpec(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := vllmbench.ApplyFilter(&spec, filter); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	return spec
}

func usage() {
	fmt.Fprintln(os.Stderr, `usage:
  localperf-vllm-bench plan   --spec spec.json [--run-dir runs/example] [--profile 8k] [--workload prefill-8k-out16-fixed] [--concurrency 4] [--vllm-command /path/to/vllm] [--json]
  localperf-vllm-bench run    --spec spec.json [--run-dir runs/example] [--profile 8k] [--workload prefill-8k-out16-fixed] [--concurrency 4] [--vllm-command /path/to/vllm] [--dry-run] [--timeout 2h]
  localperf-vllm-bench http-load --base-url http://127.0.0.1:8000 --model model --dataset-name random --random-input-len 1024 --random-output-len 128 --num-prompts 8 --max-concurrency 4 --result-filename result.json [--dataset-path canonical.jsonl] [--extra-body '{"guided_decoding_backend":"outlines"}']
  localperf-vllm-bench artifact check runs/example.sqlite
  localperf-vllm-bench artifact render runs/example.sqlite [--output runs/example.html] [--store]`)
}

func usageRoot() {
	fmt.Fprintln(os.Stderr, `usage:
  localperf bench plan   --spec spec.json [--run-dir runs/example] [--profile 8k] [--workload prefill-8k-out16-fixed] [--concurrency 4] [--vllm-command /path/to/vllm] [--json]
  localperf bench run    --spec spec.json [--run-dir runs/example] [--profile 8k] [--workload prefill-8k-out16-fixed] [--concurrency 4] [--vllm-command /path/to/vllm] [--dry-run] [--timeout 2h]
  localperf bench http-load --base-url http://127.0.0.1:8000 --model model --dataset-name random --random-input-len 1024 --random-output-len 128 --num-prompts 8 --max-concurrency 4 --result-filename result.json [--dataset-path canonical.jsonl] [--extra-body '{"guided_decoding_backend":"outlines"}']
  localperf artifact check runs/example.sqlite
  localperf artifact render runs/example.sqlite [--output runs/example.html] [--store]
  localperf artifact merge --into runs/models/model.sqlite src1.sqlite [src2.sqlite ...]
  localperf sweep plan   --model model-id [--contexts 8k,16k,32k,64k] [--concurrency 1,4,8,16,32] [--repeats 1] [--reference] [--stress] [--out spec.json]
  localperf view runs/model.sqlite [runs/other.sqlite ...] [--addr 127.0.0.1:0] [--open]`)
}

func addOverrideFlags(flags *flag.FlagSet) *overrideFlags {
	out := &overrideFlags{}
	flags.StringVar(&out.vllmCommand, "vllm-command", "", "override vllm serve executable")
	flags.StringVar(&out.vllmBenchCommand, "vllm-bench-command", "", "override vllm bench executable; defaults to --vllm-command when set")
	flags.Float64Var(&out.minMemAvailableGiB, "min-mem-available-gib", 0, "override safety.min_mem_available_gib")
	return out
}

type overrideFlags struct {
	vllmCommand        string
	vllmBenchCommand   string
	minMemAvailableGiB float64
}

func applyOverrides(spec *vllmbench.Spec, overrides *overrideFlags) {
	if overrides == nil {
		return
	}
	if strings.TrimSpace(overrides.vllmCommand) != "" {
		spec.Runner.VLLMCommand = overrides.vllmCommand
		if strings.TrimSpace(overrides.vllmBenchCommand) == "" {
			spec.Runner.VLLMBenchCommand = overrides.vllmCommand
		}
	}
	if strings.TrimSpace(overrides.vllmBenchCommand) != "" {
		spec.Runner.VLLMBenchCommand = overrides.vllmBenchCommand
	}
	for i := range spec.Engines {
		if spec.Engines[i].Type != "vllm-managed" && spec.Engines[i].Type != "vllm-endpoint" {
			continue
		}
		if strings.TrimSpace(overrides.vllmCommand) != "" {
			spec.Engines[i].Command = overrides.vllmCommand
			if strings.TrimSpace(overrides.vllmBenchCommand) == "" {
				spec.Engines[i].BenchCommand = overrides.vllmCommand
			}
		}
		if strings.TrimSpace(overrides.vllmBenchCommand) != "" {
			spec.Engines[i].BenchCommand = overrides.vllmBenchCommand
		}
	}
	if overrides.minMemAvailableGiB > 0 {
		spec.Safety.MinMemAvailableGiB = overrides.minMemAvailableGiB
	}
}

func exitOnError(err error) {
	if err == nil {
		return
	}
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}

func addFilterFlags(flags *flag.FlagSet) *filterFlags {
	out := &filterFlags{}
	flags.Var(&out.profiles, "profile", "profile name to include; may be repeated")
	flags.Var(&out.workloads, "workload", "workload name to include; may be repeated")
	flags.Var(&out.concurrencies, "concurrency", "concurrency value to include; may be repeated")
	return out
}

type filterFlags struct {
	profiles      stringList
	workloads     stringList
	concurrencies intList
}

func (flags *filterFlags) Filter() vllmbench.Filter {
	return vllmbench.Filter{
		Profiles:      flags.profiles,
		Workloads:     flags.workloads,
		Concurrencies: flags.concurrencies,
	}
}

type stringList []string

func (values *stringList) String() string {
	return strings.Join(*values, ",")
}

func (values *stringList) Set(raw string) error {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return fmt.Errorf("empty value")
	}
	*values = append(*values, raw)
	return nil
}

type intList = collections.PositiveIntList
