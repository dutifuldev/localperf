package vllmbench

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
)

type RunOptions struct {
	RunDir           string
	ArtifactPath     string
	DryRun           bool
	OriginalSpecPath string
}

type RunSummary struct {
	RunDir        string      `json:"run_dir"`
	StartedAt     time.Time   `json:"started_at"`
	FinishedAt    time.Time   `json:"finished_at"`
	DryRun        bool        `json:"dry_run"`
	PlannedRuns   int         `json:"planned_runs"`
	CompletedRuns int         `json:"completed_runs"`
	FailedRuns    int         `json:"failed_runs"`
	Rows          []ReportRow `json:"rows,omitempty"`
	EventsPath    string      `json:"events_path"`
	ReportPath    string      `json:"report_path,omitempty"`
	ArtifactPath  string      `json:"artifact_path,omitempty"`
	SpecPath      string      `json:"spec_path"`
	MemoryFloor   float64     `json:"memory_floor_gib"`
}

type Event struct {
	Timestamp       time.Time       `json:"timestamp"`
	Type            string          `json:"type"`
	Profile         string          `json:"profile,omitempty"`
	Workload        string          `json:"workload,omitempty"`
	Concurrency     int             `json:"concurrency,omitempty"`
	Repeat          int             `json:"repeat,omitempty"`
	Command         string          `json:"command,omitempty"`
	Args            []string        `json:"args,omitempty"`
	ResultFile      string          `json:"result_file,omitempty"`
	LogFile         string          `json:"log_file,omitempty"`
	DurationSeconds float64         `json:"duration_seconds,omitempty"`
	ExitCode        int             `json:"exit_code,omitempty"`
	Error           string          `json:"error,omitempty"`
	MemAvailableGiB float64         `json:"mem_available_gib,omitempty"`
	Details         json.RawMessage `json:"details,omitempty"`
}

type serverProcess struct {
	profile Profile
	cmd     *exec.Cmd
	pgid    int
	logFile *os.File
	logPath string
	done    chan error
}

type eventWriter struct {
	mu   sync.Mutex
	file *os.File
	enc  *json.Encoder
}

func Execute(ctx context.Context, spec Spec, opts RunOptions) (RunSummary, error) {
	ctx, stopSignals := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer stopSignals()

	ApplyDefaults(&spec)
	if err := ValidateSpec(spec); err != nil {
		return RunSummary{}, err
	}
	runDir := RunDir(opts.RunDir, spec, time.Now())
	summary := RunSummary{
		RunDir:       runDir,
		StartedAt:    time.Now().UTC(),
		DryRun:       opts.DryRun,
		EventsPath:   filepath.Join(runDir, "events.jsonl"),
		ArtifactPath: SQLiteArtifactPath(runDir, opts.ArtifactPath),
		SpecPath:     filepath.Join(runDir, "spec.normalized.json"),
		MemoryFloor:  spec.Safety.MinMemAvailableGiB,
	}
	if err := os.MkdirAll(filepath.Join(runDir, "results"), 0o755); err != nil {
		return summary, err
	}
	if err := os.MkdirAll(filepath.Join(runDir, "logs"), 0o755); err != nil {
		return summary, err
	}
	if err := writeJSONFile(summary.SpecPath, RedactedSpec(spec)); err != nil {
		return summary, err
	}
	events, err := newEventWriter(summary.EventsPath)
	if err != nil {
		return summary, err
	}
	defer events.Close()

	plan := BuildPlan(spec, runDir)
	summary.PlannedRuns = len(plan)
	events.Write(Event{Timestamp: time.Now().UTC(), Type: "run_start", Details: mustJSON(map[string]any{
		"name":         spec.Name,
		"planned_runs": len(plan),
		"dry_run":      opts.DryRun,
	})})
	for _, planned := range plan {
		events.Write(Event{
			Timestamp:   time.Now().UTC(),
			Type:        "planned_run",
			Profile:     planned.Profile.Name,
			Workload:    planned.Workload.Name,
			Concurrency: planned.Concurrency,
			Repeat:      planned.Repeat,
			Command:     CommandSummary(BenchCommand(spec, planned)),
			Args:        BenchCommand(spec, planned).Args,
			ResultFile:  planned.ResultFile,
		})
	}
	if opts.DryRun {
		if err := finalizeRun(runDir, &summary, events, spec, opts.OriginalSpecPath, nil); err != nil {
			return summary, err
		}
		return summary, nil
	}

	processes := map[string]*serverProcess{}
	defer func() {
		if shouldStopManaged(spec) || ctx.Err() != nil {
			stopAll(processes)
		}
	}()
	if spec.Runner.PrebootProfiles {
		if err := prebootProfiles(ctx, spec, runDir, plan, events, processes); err != nil {
			summary.FailedRuns += remainingUnaccountedRuns(summary)
			events.Write(Event{Timestamp: time.Now().UTC(), Type: "preboot_failed", Error: err.Error()})
			if finishErr := finalizeRun(runDir, &summary, events, spec, opts.OriginalSpecPath, fmt.Errorf("preboot profiles failed: %w", err)); finishErr != nil {
				return summary, finishErr
			}
			return summary, err
		}
	}

	for _, profile := range profilesInPlanOrder(spec.Profiles, plan) {
		runs := runsForProfile(plan, profile.Name)
		if len(runs) == 0 {
			continue
		}
		proc := processes[profile.Name]
		if proc == nil {
			var err error
			proc, err = prepareProfile(ctx, spec, runDir, profile, events, true)
			if err != nil {
				summary.FailedRuns += len(runs)
				events.Write(Event{Timestamp: time.Now().UTC(), Type: "profile_failed", Profile: profile.Name, Error: err.Error()})
				continue
			}
			if proc != nil {
				processes[profile.Name] = proc
			}
		} else {
			if err := wakeProfile(ctx, spec, profile, events); err != nil {
				summary.FailedRuns += len(runs)
				events.Write(Event{Timestamp: time.Now().UTC(), Type: "profile_failed", Profile: profile.Name, Error: err.Error()})
				if proc != nil {
					stopProcess(proc)
					delete(processes, profile.Name)
				}
				continue
			}
			if spec.Warmup.Enabled {
				if err := runWarmup(ctx, spec, profile, runDir, events); err != nil {
					summary.FailedRuns += len(runs)
					events.Write(Event{Timestamp: time.Now().UTC(), Type: "profile_failed", Profile: profile.Name, Error: err.Error()})
					if profile.EnableSleepMode {
						if sleepErr := sleepProfile(ctx, spec, profile, events); sleepErr != nil {
							events.Write(Event{Timestamp: time.Now().UTC(), Type: "profile_sleep_failed", Profile: profile.Name, Error: sleepErr.Error()})
							if proc != nil {
								stopProcess(proc)
								delete(processes, profile.Name)
							}
						}
					}
					continue
				}
			}
		}
		profileAborted := false
		for i, planned := range runs {
			if err := checkMemoryEvent(spec, events, "before_workload", planned.Profile.Name); err != nil {
				summary.FailedRuns++
				events.Write(Event{Timestamp: time.Now().UTC(), Type: "workload_skipped", Profile: planned.Profile.Name, Workload: planned.Workload.Name, Concurrency: planned.Concurrency, Repeat: planned.Repeat, Error: err.Error()})
				if IsMemoryFloorError(err) {
					remaining := len(runs) - i - 1
					summary.FailedRuns += remaining
					stopProfileAfterMemoryFloor(events, profile, proc, processes, remaining)
					profileAborted = true
					break
				}
				continue
			}
			result, err := executeBench(ctx, spec, planned, runDir, events)
			if err != nil {
				summary.FailedRuns++
				events.Write(Event{Timestamp: time.Now().UTC(), Type: "workload_failed", Profile: planned.Profile.Name, Workload: planned.Workload.Name, Concurrency: planned.Concurrency, Repeat: planned.Repeat, Error: err.Error()})
				if IsMemoryFloorError(err) {
					remaining := len(runs) - i - 1
					summary.FailedRuns += remaining
					stopProfileAfterMemoryFloor(events, profile, proc, processes, remaining)
					profileAborted = true
					break
				}
				continue
			}
			summary.CompletedRuns++
			if result != nil {
				summary.Rows = append(summary.Rows, *result)
			}
		}
		if profileAborted {
			continue
		}
		if profile.EnableSleepMode {
			if err := sleepProfile(ctx, spec, profile, events); err != nil {
				events.Write(Event{Timestamp: time.Now().UTC(), Type: "profile_sleep_failed", Profile: profile.Name, Error: err.Error()})
				if proc != nil {
					stopProcess(proc)
					delete(processes, profile.Name)
				}
				summary.FailedRuns += remainingRunsAfterProfile(spec.Profiles, plan, profile.Name)
				runErr := fmt.Errorf("profile %s sleep failed: %w", profile.Name, err)
				if finishErr := finalizeRun(runDir, &summary, events, spec, opts.OriginalSpecPath, runErr); finishErr != nil {
					return summary, finishErr
				}
				return summary, runErr
			}
		}
		if proc != nil && shouldStopManaged(spec) && !spec.Runner.PrebootProfiles {
			stopProcess(proc)
			delete(processes, profile.Name)
		}
	}
	if err := finalizeRun(runDir, &summary, events, spec, opts.OriginalSpecPath, nil); err != nil {
		return summary, err
	}
	return summary, nil
}

func finalizeRun(runDir string, summary *RunSummary, events *eventWriter, spec Spec, originalSpecPath string, runErr error) error {
	summary.FinishedAt = time.Now().UTC()
	event := Event{Timestamp: summary.FinishedAt, Type: "run_finish", Details: mustJSON(map[string]any{
		"completed_runs": summary.CompletedRuns,
		"failed_runs":    summary.FailedRuns,
	})}
	if runErr != nil {
		event.Error = runErr.Error()
	}
	events.Write(event)
	report, err := BuildReport(runDir)
	if err == nil {
		reportPath := filepath.Join(runDir, "report.md")
		if err := WriteReportFiles(report, reportPath); err == nil {
			summary.ReportPath = reportPath
		}
	}
	if err := writeJSONFile(filepath.Join(runDir, "summary.json"), summary); err != nil {
		return err
	}
	if summary.ArtifactPath != "" {
		if err := WriteSQLiteArtifact(runDir, summary.ArtifactPath, spec, *summary, BuildPlan(spec, runDir), originalSpecPath); err != nil && runErr == nil {
			return err
		}
	}
	if runErr != nil {
		return runErr
	}
	if summary.FailedRuns > 0 {
		return fmt.Errorf("%d benchmark run(s) failed", summary.FailedRuns)
	}
	return nil
}

func remainingUnaccountedRuns(summary RunSummary) int {
	remaining := summary.PlannedRuns - summary.CompletedRuns - summary.FailedRuns
	if remaining < 0 {
		return 0
	}
	return remaining
}

func prebootProfiles(ctx context.Context, spec Spec, runDir string, plan []PlannedRun, events *eventWriter, processes map[string]*serverProcess) error {
	for _, profile := range profilesInPlanOrder(spec.Profiles, plan) {
		proc, err := prepareProfile(ctx, spec, runDir, profile, events, true)
		if err != nil {
			return err
		}
		if proc != nil {
			processes[profile.Name] = proc
		}
		if profile.EnableSleepMode {
			if err := sleepProfile(ctx, spec, profile, events); err != nil {
				return err
			}
		}
	}
	return nil
}

func prepareProfile(ctx context.Context, spec Spec, runDir string, profile Profile, events *eventWriter, shouldWarmup bool) (*serverProcess, error) {
	if err := checkMemoryEvent(spec, events, "before_profile", profile.Name); err != nil {
		return nil, err
	}
	var proc *serverProcess
	var err error
	if profile.Managed {
		proc, err = startServer(ctx, spec, runDir, profile, events)
		if err != nil {
			return nil, err
		}
	}
	if err := waitReady(ctx, spec, profile, events, proc); err != nil {
		if proc != nil {
			stopProcess(proc)
		}
		return nil, err
	}
	if profile.EnableSleepMode {
		if err := wakeProfile(ctx, spec, profile, events); err != nil {
			if proc != nil {
				stopProcess(proc)
			}
			return nil, err
		}
	}
	if shouldWarmup && spec.Warmup.Enabled {
		if err := runWarmup(ctx, spec, profile, runDir, events); err != nil {
			if proc != nil {
				stopProcess(proc)
			}
			return nil, err
		}
	}
	return proc, nil
}

func startServer(_ context.Context, spec Spec, runDir string, profile Profile, events *eventWriter) (*serverProcess, error) {
	command := ServeCommand(spec, profile)
	logPath := filepath.Join(runDir, "logs", Slug(profile.Name)+".server.log")
	logFile, err := os.Create(logPath)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(command.Args[0], command.Args[1:]...)
	cmd.Env = WithProcessEnv(command.Env)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		_ = logFile.Close()
		return nil, err
	}
	pgid, _ := syscall.Getpgid(cmd.Process.Pid)
	proc := &serverProcess{
		profile: profile,
		cmd:     cmd,
		pgid:    pgid,
		logFile: logFile,
		logPath: logPath,
		done:    make(chan error, 1),
	}
	go func() {
		proc.done <- cmd.Wait()
		_ = logFile.Close()
	}()
	events.Write(Event{
		Timestamp: time.Now().UTC(),
		Type:      "server_start",
		Profile:   profile.Name,
		Command:   CommandSummary(command),
		Args:      command.Args,
		LogFile:   logPath,
	})
	return proc, nil
}

func waitReady(ctx context.Context, spec Spec, profile Profile, events *eventWriter, proc *serverProcess) error {
	start := time.Now()
	timeout := time.Duration(spec.Safety.StartupTimeoutSec) * time.Second
	poll := time.Duration(spec.Safety.PollIntervalMillis) * time.Millisecond
	url := baseURL(profile) + profile.HealthPath
	client := &http.Client{Timeout: time.Duration(spec.Safety.HTTPTimeoutSec) * time.Second}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	ticker := time.NewTicker(poll)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timer.C:
			return fmt.Errorf("profile %s did not become ready within %s", profile.Name, timeout)
		default:
		}
		if proc != nil {
			select {
			case err := <-proc.done:
				proc.done <- err
				return fmt.Errorf("profile %s server exited before readiness: %v; log tail: %s", profile.Name, err, tailFile(proc.logPath, 4096))
			default:
			}
		}
		if snapshot, err := checkMemoryFloor(spec.Safety.MinMemAvailableGiB); err != nil {
			events.Write(Event{
				Timestamp:       time.Now().UTC(),
				Type:            "startup_memory_floor",
				Profile:         profile.Name,
				MemAvailableGiB: snapshot.MemAvailableGiB,
				Error:           err.Error(),
			})
			return err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err == nil {
			resp, err := client.Do(req)
			if err == nil {
				_ = resp.Body.Close()
				if resp.StatusCode >= 200 && resp.StatusCode < 300 {
					events.Write(Event{Timestamp: time.Now().UTC(), Type: "server_ready", Profile: profile.Name, DurationSeconds: time.Since(start).Seconds()})
					return nil
				}
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func runWarmup(ctx context.Context, spec Spec, profile Profile, runDir string, events *eventWriter) error {
	if err := checkMemoryEvent(spec, events, "before_warmup", profile.Name); err != nil {
		return err
	}
	command := WarmupCommand(spec, profile, runDir)
	logPath := filepath.Join(runDir, "logs", Slug(profile.Name)+"__warmup.log")
	result, err := executeCommand(ctx, command, logPath, time.Duration(spec.Safety.WorkloadTimeoutSec)*time.Second, spec.Safety.MinMemAvailableGiB, time.Duration(spec.Safety.PollIntervalMillis)*time.Millisecond)
	event := Event{
		Timestamp:       time.Now().UTC(),
		Type:            "warmup_finish",
		Profile:         profile.Name,
		Command:         CommandSummary(command),
		Args:            command.Args,
		ResultFile:      resultFromArgs(command.Args),
		LogFile:         logPath,
		DurationSeconds: result.Duration.Seconds(),
		ExitCode:        result.ExitCode,
	}
	if err != nil {
		event.Error = err.Error()
		events.Write(event)
		return err
	}
	if rows, err := ParseResultFile(event.ResultFile); err != nil {
		event.Error = err.Error()
		events.Write(event)
		return err
	} else if len(rows) == 0 {
		err := errors.New("warmup result file did not contain a parseable row")
		event.Error = err.Error()
		events.Write(event)
		return err
	} else if failed := failedRequestCount(rows); failed > 0 {
		err := fmt.Errorf("warmup result reported %d failed request(s)", failed)
		event.Error = err.Error()
		events.Write(event)
		return err
	}
	events.Write(event)
	return nil
}

func executeBench(ctx context.Context, spec Spec, planned PlannedRun, runDir string, events *eventWriter) (*ReportRow, error) {
	command := BenchCommand(spec, planned)
	logPath := benchmarkLogPath(runDir, planned)
	events.Write(Event{
		Timestamp:   time.Now().UTC(),
		Type:        "workload_start",
		Profile:     planned.Profile.Name,
		Workload:    planned.Workload.Name,
		Concurrency: planned.Concurrency,
		Repeat:      planned.Repeat,
		Command:     CommandSummary(command),
		Args:        command.Args,
		ResultFile:  planned.ResultFile,
		LogFile:     logPath,
	})
	result, err := executeCommand(ctx, command, logPath, time.Duration(spec.Safety.WorkloadTimeoutSec)*time.Second, spec.Safety.MinMemAvailableGiB, time.Duration(spec.Safety.PollIntervalMillis)*time.Millisecond)
	event := Event{
		Timestamp:       time.Now().UTC(),
		Type:            "workload_finish",
		Profile:         planned.Profile.Name,
		Workload:        planned.Workload.Name,
		Concurrency:     planned.Concurrency,
		Repeat:          planned.Repeat,
		Command:         CommandSummary(command),
		Args:            command.Args,
		ResultFile:      planned.ResultFile,
		LogFile:         logPath,
		DurationSeconds: result.Duration.Seconds(),
		ExitCode:        result.ExitCode,
	}
	if err != nil {
		event.Error = err.Error()
		events.Write(event)
		return nil, err
	}
	events.Write(event)
	rows, err := ParseResultFile(planned.ResultFile)
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, errors.New("benchmark result file did not contain a parseable row")
	}
	row := rows[0]
	row.Profile = planned.Profile.Name
	row.Workload = planned.Workload.Name
	row.Context = planned.Profile.MaxModelLen
	row.ServerMaxNumSeqs = planned.Profile.MaxNumSeqs
	row.Concurrency = planned.Concurrency
	row.RandomInputLen = planned.Workload.RandomInputLen
	row.RandomOutputLen = planned.Workload.RandomOutputLen
	row.DatasetName = planned.Workload.DatasetName
	row.ResultFile = planned.ResultFile
	deriveReportRowFields(&row)
	if failed := failedRequestCount(rows); failed > 0 {
		return &row, fmt.Errorf("benchmark result reported %d failed request(s)", failed)
	}
	return &row, nil
}

func benchmarkLogPath(runDir string, planned PlannedRun) string {
	name := fmt.Sprintf("%s__%s__c%d", Slug(planned.Profile.Name), Slug(planned.Workload.Name), planned.Concurrency)
	if planned.Workload.Repeats > 1 {
		name += fmt.Sprintf("__r%d", planned.Repeat+1)
	}
	return filepath.Join(runDir, "logs", name+".log")
}

func failedRequestCount(rows []ReportRow) int {
	failed := 0
	for _, row := range rows {
		failed += row.Failed
	}
	return failed
}

type commandResult struct {
	Duration time.Duration
	ExitCode int
}

func executeCommand(ctx context.Context, command CommandSpec, logPath string, timeout time.Duration, minMemAvailableGiB float64, pollInterval time.Duration) (commandResult, error) {
	if len(command.Args) == 0 {
		return commandResult{ExitCode: -1}, errors.New("empty command")
	}
	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		return commandResult{ExitCode: -1}, err
	}
	if result := resultFromArgs(command.Args); result != "" {
		if err := os.MkdirAll(filepath.Dir(result), 0o755); err != nil {
			return commandResult{ExitCode: -1}, err
		}
	}
	logFile, err := os.Create(logPath)
	if err != nil {
		return commandResult{ExitCode: -1}, err
	}
	defer logFile.Close()
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	memoryMonitor := monitorMemoryFloor(runCtx, cancel, minMemAvailableGiB, pollInterval)
	cmd := exec.CommandContext(runCtx, command.Args[0], command.Args[1:]...)
	cmd.Env = WithProcessEnv(command.Env)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	start := time.Now()
	err = cmd.Run()
	duration := time.Since(start)
	cancel()
	memoryErr := <-memoryMonitor
	result := commandResult{Duration: duration}
	if cmd.ProcessState != nil {
		result.ExitCode = cmd.ProcessState.ExitCode()
	} else {
		result.ExitCode = -1
	}
	if runCtx.Err() == context.DeadlineExceeded {
		return result, fmt.Errorf("command timed out after %s", timeout)
	}
	if memoryErr != nil {
		return result, memoryErr
	}
	if err != nil {
		return result, fmt.Errorf("%w; log tail: %s", err, strings.TrimSpace(tailFile(logPath, 4096)))
	}
	return result, nil
}

func monitorMemoryFloor(ctx context.Context, cancel context.CancelFunc, minGiB float64, interval time.Duration) <-chan error {
	done := make(chan error, 1)
	go func() {
		if minGiB <= 0 {
			<-ctx.Done()
			done <- nil
			return
		}
		if interval <= 0 {
			interval = time.Second
		}
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				done <- nil
				return
			case <-ticker.C:
				if _, err := checkMemoryFloor(minGiB); err != nil {
					cancel()
					done <- err
					return
				}
			}
		}
	}()
	return done
}

func sleepProfile(ctx context.Context, spec Spec, profile Profile, events *eventWriter) error {
	if !profile.EnableSleepMode {
		return nil
	}
	if err := checkMemoryEvent(spec, events, "before_sleep", profile.Name); err != nil {
		return err
	}
	url := fmt.Sprintf("%s/sleep?level=%d", baseURL(profile), SleepLevelValue(profile))
	start := time.Now()
	err := postAdmin(ctx, spec, url)
	event := Event{Timestamp: time.Now().UTC(), Type: "profile_sleep", Profile: profile.Name, DurationSeconds: time.Since(start).Seconds()}
	if err == nil {
		err = waitSleepState(ctx, spec, profile, true)
		event.DurationSeconds = time.Since(start).Seconds()
	}
	if err != nil {
		event.Error = err.Error()
		events.Write(event)
		return err
	}
	events.Write(event)
	return nil
}

func wakeProfile(ctx context.Context, spec Spec, profile Profile, events *eventWriter) error {
	if !profile.EnableSleepMode {
		return nil
	}
	if err := checkMemoryEvent(spec, events, "before_wake", profile.Name); err != nil {
		return err
	}
	sleeping, err := isSleeping(ctx, spec, profile)
	if err != nil {
		events.Write(Event{Timestamp: time.Now().UTC(), Type: "profile_wake_check_failed", Profile: profile.Name, Error: err.Error()})
		sleeping = true
	}
	if !sleeping {
		events.Write(Event{Timestamp: time.Now().UTC(), Type: "profile_wake_skipped", Profile: profile.Name})
		return nil
	}
	start := time.Now()
	err = postAdmin(ctx, spec, baseURL(profile)+"/wake_up")
	event := Event{Timestamp: time.Now().UTC(), Type: "profile_wake", Profile: profile.Name, DurationSeconds: time.Since(start).Seconds()}
	if err == nil {
		err = waitSleepState(ctx, spec, profile, false)
		event.DurationSeconds = time.Since(start).Seconds()
	}
	if err != nil {
		event.Error = err.Error()
		events.Write(event)
		return err
	}
	events.Write(event)
	return waitReady(ctx, spec, profile, events, nil)
}

func waitSleepState(ctx context.Context, spec Spec, profile Profile, wantSleeping bool) error {
	state := "awake"
	if wantSleeping {
		state = "sleeping"
	}
	start := time.Now()
	timeout := time.Duration(spec.Safety.StartupTimeoutSec) * time.Second
	poll := time.Duration(spec.Safety.PollIntervalMillis) * time.Millisecond
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	ticker := time.NewTicker(poll)
	defer ticker.Stop()
	var lastErr error
	for {
		sleeping, err := isSleeping(ctx, spec, profile)
		if err == nil && sleeping == wantSleeping {
			return nil
		}
		if err != nil {
			lastErr = err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timer.C:
			if lastErr != nil {
				return fmt.Errorf("profile %s did not become %s within %s: %w", profile.Name, state, time.Since(start).Round(time.Millisecond), lastErr)
			}
			return fmt.Errorf("profile %s did not become %s within %s", profile.Name, state, time.Since(start).Round(time.Millisecond))
		case <-ticker.C:
		}
	}
}

func isSleeping(ctx context.Context, spec Spec, profile Profile) (bool, error) {
	client := &http.Client{Timeout: time.Duration(spec.Safety.HTTPTimeoutSec) * time.Second}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL(profile)+"/is_sleeping", nil)
	if err != nil {
		return false, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return false, fmt.Errorf("%s/is_sleeping returned HTTP %d: %s", baseURL(profile), resp.StatusCode, strings.TrimSpace(string(data)))
	}
	text := strings.TrimSpace(strings.ToLower(string(data)))
	var object map[string]any
	if err := json.Unmarshal(data, &object); err == nil {
		for _, key := range []string{"is_sleeping", "sleeping"} {
			if value, ok := object[key].(bool); ok {
				return value, nil
			}
		}
	}
	return text == "true" || text == `"true"` || strings.Contains(text, ":true"), nil
}

func postAdmin(ctx context.Context, spec Spec, url string) error {
	client := &http.Client{Timeout: time.Duration(spec.Safety.HTTPTimeoutSec) * time.Second}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("%s returned HTTP %d: %s", url, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return nil
}

func checkMemoryEvent(spec Spec, events *eventWriter, eventType, profile string) error {
	snapshot, err := checkMemoryFloor(spec.Safety.MinMemAvailableGiB)
	event := Event{Timestamp: time.Now().UTC(), Type: eventType, Profile: profile, MemAvailableGiB: snapshot.MemAvailableGiB}
	if err != nil {
		event.Error = err.Error()
		events.Write(event)
		return err
	}
	events.Write(event)
	return nil
}

func newEventWriter(path string) (*eventWriter, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	file, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	return &eventWriter{file: file, enc: json.NewEncoder(file)}, nil
}

func (writer *eventWriter) Write(event Event) {
	writer.mu.Lock()
	defer writer.mu.Unlock()
	_ = writer.enc.Encode(event)
}

func (writer *eventWriter) Close() error {
	writer.mu.Lock()
	defer writer.mu.Unlock()
	return writer.file.Close()
}

func baseURL(profile Profile) string {
	return fmt.Sprintf("http://%s:%d", profile.Host, profile.Port)
}

func resultFromArgs(args []string) string {
	for i, arg := range args {
		if arg == "--result-filename" && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

func writeJSONFile(path string, value any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func mustJSON(value any) json.RawMessage {
	data, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	return data
}

func profilesInPlanOrder(profiles []Profile, plan []PlannedRun) []Profile {
	needed := map[string]bool{}
	for _, run := range plan {
		needed[run.Profile.Name] = true
	}
	out := make([]Profile, 0, len(needed))
	for _, profile := range profiles {
		if needed[profile.Name] {
			out = append(out, profile)
		}
	}
	return out
}

func runsForProfile(plan []PlannedRun, profileName string) []PlannedRun {
	var out []PlannedRun
	for _, run := range plan {
		if run.Profile.Name == profileName {
			out = append(out, run)
		}
	}
	return out
}

func remainingRunsAfterProfile(profiles []Profile, plan []PlannedRun, profileName string) int {
	seenProfile := false
	remaining := 0
	for _, profile := range profilesInPlanOrder(profiles, plan) {
		if profile.Name == profileName {
			seenProfile = true
			continue
		}
		if seenProfile {
			remaining += len(runsForProfile(plan, profile.Name))
		}
	}
	return remaining
}

func shouldStopManaged(spec Spec) bool {
	return spec.Runner.StopManagedOnExit == nil || *spec.Runner.StopManagedOnExit
}

func stopProfileAfterMemoryFloor(events *eventWriter, profile Profile, proc *serverProcess, processes map[string]*serverProcess, remaining int) {
	if proc != nil {
		stopProcess(proc)
		delete(processes, profile.Name)
	}
	events.Write(Event{Timestamp: time.Now().UTC(), Type: "profile_memory_floor_abort", Profile: profile.Name, Error: fmt.Sprintf("memory floor guard tripped; skipped %d remaining profile run(s)", remaining)})
}

func stopAll(processes map[string]*serverProcess) {
	for name, proc := range processes {
		stopProcess(proc)
		delete(processes, name)
	}
}

func stopProcess(proc *serverProcess) {
	if proc == nil || proc.cmd == nil || proc.cmd.Process == nil {
		return
	}
	pgid := proc.pgid
	if pgid <= 0 {
		if value, err := syscall.Getpgid(proc.cmd.Process.Pid); err == nil {
			pgid = value
		}
	}
	if pgid > 0 {
		_ = syscall.Kill(-pgid, syscall.SIGTERM)
	} else {
		_ = proc.cmd.Process.Signal(syscall.SIGTERM)
	}
	select {
	case <-proc.done:
	case <-time.After(20 * time.Second):
		if pgid > 0 {
			_ = syscall.Kill(-pgid, syscall.SIGKILL)
		} else {
			_ = proc.cmd.Process.Kill()
		}
		<-proc.done
	}
}

func tailFile(path string, maxBytes int64) string {
	file, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return ""
	}
	start := info.Size() - maxBytes
	if start < 0 {
		start = 0
	}
	if _, err := file.Seek(start, io.SeekStart); err != nil {
		return ""
	}
	data, _ := io.ReadAll(file)
	if start > 0 {
		if index := bytes.IndexByte(data, '\n'); index >= 0 && index+1 < len(data) {
			data = data[index+1:]
		}
	}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	var lines []string
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			lines = append(lines, line)
		}
	}
	if len(lines) > 12 {
		lines = lines[len(lines)-12:]
	}
	return strings.Join(lines, " | ")
}
