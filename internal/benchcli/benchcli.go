package benchcli

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/dutifuldev/localperf/internal/collections"
	"github.com/dutifuldev/localperf/internal/vllmbench"
)

func Main(args []string) {
	if len(args) < 1 {
		usage()
		os.Exit(2)
	}
	switch args[0] {
	case "plan":
		runPlan(args[1:])
	case "run":
		runBench(args[1:])
	case "report":
		runReport(args[1:])
	case "artifact":
		runArtifact(args[1:])
	default:
		usage()
		os.Exit(2)
	}
}

func LocalPerfMain(args []string) {
	if len(args) < 1 {
		usageLocalPerf()
		os.Exit(2)
	}
	switch args[0] {
	case "bench":
		Main(args[1:])
	case "artifact":
		runArtifact(args[1:])
	default:
		usageLocalPerf()
		os.Exit(2)
	}
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
		command := vllmbench.BenchCommand(spec, planned)
		fmt.Printf("- profile=%s workload=%s concurrency=%d result=%s\n", planned.Profile.Name, planned.Workload.Name, planned.Concurrency, planned.ResultFile)
		fmt.Printf("  %s\n", vllmbench.ShellQuote(command.Args))
	}
}

func runBench(args []string) {
	flags := flag.NewFlagSet("run", flag.ExitOnError)
	specPath := flags.String("spec", "", "benchmark spec JSON file")
	runDir := flags.String("run-dir", "", "optional run directory")
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
		DryRun:           *dryRun,
		OriginalSpecPath: *specPath,
	})
	fmt.Printf("run dir: %s\n", summary.RunDir)
	fmt.Printf("planned: %d completed: %d failed: %d\n", summary.PlannedRuns, summary.CompletedRuns, summary.FailedRuns)
	if summary.ReportPath != "" {
		fmt.Printf("report: %s\n", summary.ReportPath)
	}
	if summary.ArtifactPath != "" {
		fmt.Printf("artifact: %s\n", summary.ArtifactPath)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runReport(args []string) {
	flags := flag.NewFlagSet("report", flag.ExitOnError)
	runDir := flags.String("run-dir", "", "run directory with events.jsonl and results")
	output := flags.String("output", "", "Markdown report output path; defaults to <run-dir>/report.md")
	jsonOutput := flags.Bool("json", false, "print report JSON to stdout instead of writing files")
	_ = flags.Parse(args)
	if strings.TrimSpace(*runDir) == "" {
		fmt.Fprintln(os.Stderr, "missing --run-dir")
		os.Exit(2)
	}
	report, err := vllmbench.BuildReport(*runDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if *jsonOutput {
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		_ = encoder.Encode(report)
		return
	}
	outPath := *output
	if strings.TrimSpace(outPath) == "" {
		outPath = *runDir + "/report.md"
	}
	if err := vllmbench.WriteReportFiles(report, outPath); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("report: %s\n", outPath)
	sidecarBase := strings.TrimSuffix(outPath, filepath.Ext(outPath))
	fmt.Printf("json: %s\n", sidecarBase+".json")
	fmt.Printf("csv: %s\n", sidecarBase+".csv")
}

func runArtifact(args []string) {
	if len(args) < 1 {
		usage()
		os.Exit(2)
	}
	switch args[0] {
	case "check":
		runArtifactCheck(args[1:])
	default:
		usage()
		os.Exit(2)
	}
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
	if err := vllmbench.CheckSQLiteArtifact(*path); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("artifact ok: %s\n", *path)
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
  localperf-vllm-bench report --run-dir runs/example [--output runs/example/report.md] [--json]
  localperf-vllm-bench artifact check runs/example.sqlite`)
}

func usageLocalPerf() {
	fmt.Fprintln(os.Stderr, `usage:
  localperf bench plan   --spec spec.json [--run-dir runs/example] [--profile 8k] [--workload prefill-8k-out16-fixed] [--concurrency 4] [--vllm-command /path/to/vllm] [--json]
  localperf bench run    --spec spec.json [--run-dir runs/example] [--profile 8k] [--workload prefill-8k-out16-fixed] [--concurrency 4] [--vllm-command /path/to/vllm] [--dry-run] [--timeout 2h]
  localperf bench report --run-dir runs/example [--output runs/example/report.md] [--json]
  localperf artifact check runs/example.sqlite`)
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
