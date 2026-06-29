package vllmbench

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"time"
)

const DefaultHealthPath = "/v1/models"

type Spec struct {
	Version     string            `json:"version"`
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
	Model       string            `json:"model"`
	OutputDir   string            `json:"output_dir,omitempty"`
	Env         map[string]string `json:"env,omitempty"`
	Engines     []EngineConfig    `json:"engines,omitempty"`
	Runner      RunnerConfig      `json:"runner"`
	Safety      SafetyConfig      `json:"safety"`
	Warmup      WarmupConfig      `json:"warmup,omitempty"`
	Profiles    []Profile         `json:"profiles"`
	Workloads   []Workload        `json:"workloads"`
}

type EngineConfig struct {
	Name            string            `json:"name"`
	Type            string            `json:"type"`
	Command         string            `json:"command,omitempty"`
	BenchCommand    string            `json:"bench_command,omitempty"`
	Managed         *bool             `json:"managed,omitempty"`
	EndpointBaseURL string            `json:"endpoint_base_url,omitempty"`
	Env             map[string]string `json:"env,omitempty"`
	Metadata        map[string]any    `json:"metadata,omitempty"`
}

type RunnerConfig struct {
	VLLMCommand          string `json:"vllm_command,omitempty"`
	VLLMBenchCommand     string `json:"vllm_bench_command,omitempty"`
	OneAwakeProfile      *bool  `json:"one_awake_profile,omitempty"`
	PrebootProfiles      bool   `json:"preboot_profiles,omitempty"`
	StopManagedOnExit    *bool  `json:"stop_managed_on_exit,omitempty"`
	AppendTimestampToRun *bool  `json:"append_timestamp_to_run,omitempty"`
}

type SafetyConfig struct {
	MinMemAvailableGiB float64 `json:"min_mem_available_gib"`
	PollIntervalMillis int     `json:"poll_interval_millis,omitempty"`
	StartupTimeoutSec  int     `json:"startup_timeout_sec,omitempty"`
	WorkloadTimeoutSec int     `json:"workload_timeout_sec,omitempty"`
	HTTPTimeoutSec     int     `json:"http_timeout_sec,omitempty"`
}

type WarmupConfig struct {
	BenchmarkTrafficConfig
	Enabled        bool `json:"enabled"`
	NumPrompts     int  `json:"num_prompts,omitempty"`
	MaxConcurrency int  `json:"max_concurrency,omitempty"`
}

type Profile struct {
	Name                 string            `json:"name"`
	Engine               string            `json:"engine,omitempty"`
	Model                string            `json:"model,omitempty"`
	Host                 string            `json:"host,omitempty"`
	Port                 int               `json:"port"`
	Managed              bool              `json:"managed"`
	EnableSleepMode      bool              `json:"enable_sleep_mode,omitempty"`
	SleepLevel           *int              `json:"sleep_level,omitempty"`
	HealthPath           string            `json:"health_path,omitempty"`
	Serve                ServeConfig       `json:"serve,omitempty"`
	MaxModelLen          int               `json:"max_model_len,omitempty"`
	MaxNumSeqs           int               `json:"max_num_seqs,omitempty"`
	MaxNumBatchedTokens  int               `json:"max_num_batched_tokens,omitempty"`
	GPUMemoryUtilization float64           `json:"gpu_memory_utilization,omitempty"`
	KVCacheDType         string            `json:"kv_cache_dtype,omitempty"`
	AttentionBackend     string            `json:"attention_backend,omitempty"`
	MoEBackend           string            `json:"moe_backend,omitempty"`
	Env                  map[string]string `json:"env,omitempty"`
	Args                 []string          `json:"args,omitempty"`
	EngineArgs           []string          `json:"engine_args,omitempty"`
}

type Workload struct {
	BenchmarkTrafficConfig
	Name                    string                 `json:"name"`
	Profiles                []string               `json:"profiles,omitempty"`
	NumPrompts              int                    `json:"num_prompts"`
	Samples                 int                    `json:"samples,omitempty"`
	Repeats                 int                    `json:"repeats,omitempty"`
	MaxConcurrency          []int                  `json:"max_concurrency"`
	Concurrency             []int                  `json:"concurrency,omitempty"`
	Traffic                 BenchmarkTrafficConfig `json:"traffic,omitempty"`
	IgnoreEOS               bool                   `json:"ignore_eos,omitempty"`
	Temperature             *float64               `json:"temperature,omitempty"`
	CapturePayloadArtifacts bool                   `json:"capture_payload_artifacts,omitempty"`
}

type ServeConfig struct {
	MaxModelLen          int     `json:"max_model_len,omitempty"`
	MaxNumSeqs           int     `json:"max_num_seqs,omitempty"`
	MaxNumBatchedTokens  int     `json:"max_num_batched_tokens,omitempty"`
	GPUMemoryUtilization float64 `json:"gpu_memory_utilization,omitempty"`
	KVCacheDType         string  `json:"kv_cache_dtype,omitempty"`
	AttentionBackend     string  `json:"attention_backend,omitempty"`
	MoEBackend           string  `json:"moe_backend,omitempty"`
}

type BenchmarkTrafficConfig struct {
	Backend                     string   `json:"backend,omitempty"`
	Endpoint                    string   `json:"endpoint,omitempty"`
	DatasetName                 string   `json:"dataset_name,omitempty"`
	DatasetPath                 string   `json:"dataset_path,omitempty"`
	RequestRate                 string   `json:"request_rate,omitempty"`
	Seed                        *int     `json:"seed,omitempty"`
	RandomInputLen              int      `json:"random_input_len,omitempty"`
	RandomOutputLen             int      `json:"random_output_len,omitempty"`
	RandomRangeRatio            string   `json:"random_range_ratio,omitempty"`
	RandomPrefixLen             int      `json:"random_prefix_len,omitempty"`
	CustomOutputLen             *int     `json:"custom_output_len,omitempty"`
	ShareGPTOutputLen           *int     `json:"sharegpt_output_len,omitempty"`
	SonnetInputLen              int      `json:"sonnet_input_len,omitempty"`
	SonnetOutputLen             int      `json:"sonnet_output_len,omitempty"`
	SonnetPrefixLen             int      `json:"sonnet_prefix_len,omitempty"`
	PrefixRepetitionPrefixLen   int      `json:"prefix_repetition_prefix_len,omitempty"`
	PrefixRepetitionSuffixLen   int      `json:"prefix_repetition_suffix_len,omitempty"`
	PrefixRepetitionNumPrefixes int      `json:"prefix_repetition_num_prefixes,omitempty"`
	PrefixRepetitionOutputLen   int      `json:"prefix_repetition_output_len,omitempty"`
	SpeedBenchDatasetSubset     string   `json:"speed_bench_dataset_subset,omitempty"`
	SpeedBenchOutputLen         int      `json:"speed_bench_output_len,omitempty"`
	SpeedBenchCategory          string   `json:"speed_bench_category,omitempty"`
	DisableShuffle              bool     `json:"disable_shuffle,omitempty"`
	NoOversample                bool     `json:"no_oversample,omitempty"`
	SkipChatTemplate            bool     `json:"skip_chat_template,omitempty"`
	SaveDetailed                bool     `json:"save_detailed,omitempty"`
	PlotDatasetStats            bool     `json:"plot_dataset_stats,omitempty"`
	ExtraBody                   string   `json:"extra_body,omitempty"`
	Metadata                    []string `json:"metadata,omitempty"`
	Goodput                     []string `json:"goodput,omitempty"`
	ExtraArgs                   []string `json:"extra_args,omitempty"`
}

type PlannedRun struct {
	Profile     Profile  `json:"profile"`
	Workload    Workload `json:"workload"`
	Concurrency int      `json:"concurrency"`
	Repeat      int      `json:"repeat,omitempty"`
	ResultFile  string   `json:"result_file"`
}

func LoadSpec(path string) (Spec, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Spec{}, err
	}
	var spec Spec
	if err := json.Unmarshal(data, &spec); err != nil {
		return Spec{}, err
	}
	ApplyDefaults(&spec)
	if err := ValidateSpec(spec); err != nil {
		return Spec{}, err
	}
	return spec, nil
}

func ApplyDefaults(spec *Spec) {
	if strings.TrimSpace(spec.Runner.VLLMCommand) == "" {
		spec.Runner.VLLMCommand = "vllm"
	}
	if strings.TrimSpace(spec.Runner.VLLMBenchCommand) == "" {
		spec.Runner.VLLMBenchCommand = "vllm"
	}
	if len(spec.Engines) == 0 {
		spec.Engines = []EngineConfig{{
			Name:         "vllm",
			Type:         "vllm-managed",
			Command:      spec.Runner.VLLMCommand,
			BenchCommand: spec.Runner.VLLMBenchCommand,
		}}
	}
	for i := range spec.Engines {
		engine := &spec.Engines[i]
		if strings.TrimSpace(engine.Type) == "" {
			engine.Type = "vllm-managed"
		}
		if strings.TrimSpace(engine.Command) == "" && engine.Type == "vllm-managed" {
			engine.Command = spec.Runner.VLLMCommand
		}
		if strings.TrimSpace(engine.BenchCommand) == "" && engine.Type == "vllm-managed" {
			engine.BenchCommand = spec.Runner.VLLMBenchCommand
		}
	}
	if spec.Runner.OneAwakeProfile == nil {
		yes := true
		spec.Runner.OneAwakeProfile = &yes
	}
	if spec.Runner.StopManagedOnExit == nil {
		yes := true
		spec.Runner.StopManagedOnExit = &yes
	}
	if spec.Runner.AppendTimestampToRun == nil {
		yes := true
		spec.Runner.AppendTimestampToRun = &yes
	}
	if spec.Safety.PollIntervalMillis <= 0 {
		spec.Safety.PollIntervalMillis = 1000
	}
	if spec.Safety.StartupTimeoutSec <= 0 {
		spec.Safety.StartupTimeoutSec = 900
	}
	if spec.Safety.WorkloadTimeoutSec <= 0 {
		spec.Safety.WorkloadTimeoutSec = 1800
	}
	if spec.Safety.HTTPTimeoutSec <= 0 {
		spec.Safety.HTTPTimeoutSec = 15
	}
	for i := range spec.Profiles {
		profile := &spec.Profiles[i]
		if strings.TrimSpace(profile.Engine) == "" {
			profile.Engine = spec.Engines[0].Name
		}
		if strings.TrimSpace(profile.Model) == "" {
			profile.Model = spec.Model
		}
		if strings.TrimSpace(profile.Host) == "" {
			profile.Host = "127.0.0.1"
		}
		if strings.TrimSpace(profile.HealthPath) == "" {
			profile.HealthPath = DefaultHealthPath
		}
		applyServeDefaults(profile)
		if profile.EnableSleepMode && profile.SleepLevel == nil {
			profile.SleepLevel = intPointer(2)
		}
	}
	if spec.Warmup.Enabled {
		applyTrafficDefaults(&spec.Warmup.BenchmarkTrafficConfig, "random")
		if spec.Warmup.RandomInputLen <= 0 {
			spec.Warmup.RandomInputLen = 256
		}
		if spec.Warmup.RandomOutputLen <= 0 {
			spec.Warmup.RandomOutputLen = 16
		}
		if spec.Warmup.NumPrompts <= 0 {
			spec.Warmup.NumPrompts = 4
		}
		if spec.Warmup.MaxConcurrency <= 0 {
			spec.Warmup.MaxConcurrency = 1
		}
	}
	for i := range spec.Workloads {
		workload := &spec.Workloads[i]
		if !trafficConfigEmpty(workload.Traffic) {
			workload.BenchmarkTrafficConfig = overlayTrafficConfig(workload.BenchmarkTrafficConfig, workload.Traffic)
		}
		if workload.NumPrompts <= 0 && workload.Samples > 0 {
			workload.NumPrompts = workload.Samples
		}
		if len(workload.MaxConcurrency) == 0 && len(workload.Concurrency) > 0 {
			workload.MaxConcurrency = append([]int(nil), workload.Concurrency...)
		}
		if workload.Repeats <= 0 {
			workload.Repeats = 1
		}
		applyTrafficDefaults(&workload.BenchmarkTrafficConfig, "")
	}
}

func trafficConfigEmpty(traffic BenchmarkTrafficConfig) bool {
	return reflect.DeepEqual(traffic, BenchmarkTrafficConfig{})
}

func overlayTrafficConfig(base, override BenchmarkTrafficConfig) BenchmarkTrafficConfig {
	baseValue := reflect.ValueOf(&base).Elem()
	overrideValue := reflect.ValueOf(override)
	for i := 0; i < overrideValue.NumField(); i++ {
		field := overrideValue.Field(i)
		zero := reflect.Zero(field.Type())
		if reflect.DeepEqual(field.Interface(), zero.Interface()) {
			continue
		}
		baseValue.Field(i).Set(field)
	}
	return base
}

func applyServeDefaults(profile *Profile) {
	if profile.MaxModelLen == 0 {
		profile.MaxModelLen = profile.Serve.MaxModelLen
	}
	if profile.MaxNumSeqs == 0 {
		profile.MaxNumSeqs = profile.Serve.MaxNumSeqs
	}
	if profile.MaxNumBatchedTokens == 0 {
		profile.MaxNumBatchedTokens = profile.Serve.MaxNumBatchedTokens
	}
	if profile.GPUMemoryUtilization == 0 {
		profile.GPUMemoryUtilization = profile.Serve.GPUMemoryUtilization
	}
	if strings.TrimSpace(profile.KVCacheDType) == "" {
		profile.KVCacheDType = profile.Serve.KVCacheDType
	}
	if strings.TrimSpace(profile.AttentionBackend) == "" {
		profile.AttentionBackend = profile.Serve.AttentionBackend
	}
	if strings.TrimSpace(profile.MoEBackend) == "" {
		profile.MoEBackend = profile.Serve.MoEBackend
	}
}

func applyTrafficDefaults(traffic *BenchmarkTrafficConfig, defaultDataset string) {
	if strings.TrimSpace(traffic.DatasetName) == "" {
		traffic.DatasetName = defaultDataset
	}
	if strings.TrimSpace(traffic.Backend) == "" {
		traffic.Backend = "openai-chat"
	}
	if strings.TrimSpace(traffic.Endpoint) == "" {
		traffic.Endpoint = defaultEndpoint(traffic.Backend)
	}
	if strings.TrimSpace(traffic.RequestRate) == "" {
		traffic.RequestRate = "inf"
	}
}

func RedactedSpec(spec Spec) Spec {
	out := spec
	out.Env = redactedEnv(spec.Env)
	out.Engines = append([]EngineConfig(nil), spec.Engines...)
	for i := range out.Engines {
		out.Engines[i].Env = redactedEnv(spec.Engines[i].Env)
	}
	out.Profiles = append([]Profile(nil), spec.Profiles...)
	for i := range out.Profiles {
		out.Profiles[i].Env = redactedEnv(spec.Profiles[i].Env)
	}
	return out
}

func defaultEndpoint(backend string) string {
	switch backend {
	case "openai-chat":
		return "/v1/chat/completions"
	case "openai":
		return "/v1/completions"
	default:
		return ""
	}
}

func ValidateSpec(spec Spec) error {
	var issues []string
	if strings.TrimSpace(spec.Name) == "" {
		issues = append(issues, "name is required")
	}
	if strings.TrimSpace(spec.Model) == "" {
		issues = append(issues, "model is required")
	}
	if spec.Safety.MinMemAvailableGiB <= 0 {
		issues = append(issues, "safety.min_mem_available_gib must be positive")
	}
	engineNames := map[string]bool{}
	if len(spec.Engines) == 0 {
		issues = append(issues, "at least one engine is required")
	}
	for i, engine := range spec.Engines {
		prefix := fmt.Sprintf("engines[%d]", i)
		name := strings.TrimSpace(engine.Name)
		if name == "" {
			issues = append(issues, prefix+": name is required")
		}
		if engineNames[engine.Name] {
			issues = append(issues, prefix+": duplicate engine name "+engine.Name)
		}
		engineNames[engine.Name] = true
		if strings.TrimSpace(engine.Type) == "" {
			issues = append(issues, prefix+": type is required")
		}
	}
	profileNames := map[string]bool{}
	profileSlugs := map[string]string{}
	if len(spec.Profiles) == 0 {
		issues = append(issues, "at least one profile is required")
	}
	oneAwakeProfile := spec.Runner.OneAwakeProfile == nil || *spec.Runner.OneAwakeProfile
	for i, profile := range spec.Profiles {
		prefix := fmt.Sprintf("profiles[%d]", i)
		profileName := strings.TrimSpace(profile.Name)
		if profileName == "" {
			issues = append(issues, prefix+": name is required")
		}
		if profileNames[profile.Name] {
			issues = append(issues, prefix+": duplicate profile name "+profile.Name)
		}
		profileNames[profile.Name] = true
		if profileName != "" {
			profileSlug := Slug(profileName)
			if profileSlug == "" {
				issues = append(issues, prefix+": name must contain at least one ASCII letter or digit")
			} else if previous, ok := profileSlugs[profileSlug]; ok {
				issues = append(issues, prefix+": profile name "+profile.Name+" collides with "+previous+" after slug normalization")
			} else {
				profileSlugs[profileSlug] = profile.Name
			}
		}
		if profile.Port <= 0 {
			issues = append(issues, prefix+": port must be positive")
		}
		if strings.TrimSpace(profile.Model) == "" {
			issues = append(issues, prefix+": model is required")
		}
		if strings.TrimSpace(profile.Engine) == "" {
			issues = append(issues, prefix+": engine is required")
		} else if !engineNames[profile.Engine] {
			issues = append(issues, prefix+": unknown engine "+profile.Engine)
		}
		if profile.Managed {
			if profile.MaxModelLen <= 0 {
				issues = append(issues, prefix+": max_model_len must be positive for managed profiles")
			}
			if profile.MaxNumSeqs <= 0 {
				issues = append(issues, prefix+": max_num_seqs must be positive for managed profiles")
			}
			if spec.Runner.PrebootProfiles && oneAwakeProfile && !profile.EnableSleepMode {
				issues = append(issues, prefix+": enable_sleep_mode is required when runner.preboot_profiles and runner.one_awake_profile are true")
			}
		}
		if profile.SleepLevel != nil && (*profile.SleepLevel < 0 || *profile.SleepLevel > 2) {
			issues = append(issues, prefix+": sleep_level must be 0, 1, or 2")
		}
	}
	if spec.Warmup.Enabled {
		if spec.Warmup.NumPrompts <= 0 {
			issues = append(issues, "warmup: num_prompts must be positive")
		}
		if spec.Warmup.MaxConcurrency <= 0 {
			issues = append(issues, "warmup: max_concurrency must be positive")
		}
		if strings.TrimSpace(spec.Warmup.DatasetName) == "" {
			issues = append(issues, "warmup: dataset_name is required")
		}
		issues = append(issues, validateTrafficConfig("warmup", spec.Warmup.BenchmarkTrafficConfig)...)
	}
	if len(spec.Workloads) == 0 {
		issues = append(issues, "at least one workload is required")
	}
	workloadNames := map[string]bool{}
	workloadSlugs := map[string]string{}
	for i, workload := range spec.Workloads {
		prefix := fmt.Sprintf("workloads[%d]", i)
		workloadName := strings.TrimSpace(workload.Name)
		if workloadName == "" {
			issues = append(issues, prefix+": name is required")
		}
		if workloadNames[workload.Name] {
			issues = append(issues, prefix+": duplicate workload name "+workload.Name)
		}
		workloadNames[workload.Name] = true
		if workloadName != "" {
			workloadSlug := Slug(workloadName)
			if workloadSlug == "" {
				issues = append(issues, prefix+": name must contain at least one ASCII letter or digit")
			} else if workloadSlug == "warmup" {
				issues = append(issues, prefix+": workload name warmup is reserved for warmup artifacts")
			} else if previous, ok := workloadSlugs[workloadSlug]; ok {
				issues = append(issues, prefix+": workload name "+workload.Name+" collides with "+previous+" after slug normalization")
			} else {
				workloadSlugs[workloadSlug] = workload.Name
			}
		}
		if strings.TrimSpace(workload.DatasetName) == "" {
			issues = append(issues, prefix+": dataset_name is required")
		}
		if workload.NumPrompts <= 0 {
			issues = append(issues, prefix+": num_prompts must be positive")
		}
		if workload.Repeats <= 0 {
			issues = append(issues, prefix+": repeats must be positive")
		}
		if len(workload.MaxConcurrency) == 0 {
			issues = append(issues, prefix+": max_concurrency must not be empty")
		}
		seenConcurrency := map[int]bool{}
		for _, concurrency := range workload.MaxConcurrency {
			if concurrency <= 0 {
				issues = append(issues, prefix+": max_concurrency values must be positive")
			}
			if seenConcurrency[concurrency] {
				issues = append(issues, prefix+": duplicate max_concurrency value "+fmt.Sprint(concurrency))
			}
			seenConcurrency[concurrency] = true
		}
		seenProfileRefs := map[string]bool{}
		for _, profileName := range workload.Profiles {
			if seenProfileRefs[profileName] {
				issues = append(issues, prefix+": duplicate profile reference "+profileName)
			}
			seenProfileRefs[profileName] = true
			if !profileNames[profileName] {
				issues = append(issues, prefix+": unknown profile "+profileName)
			}
		}
		issues = append(issues, validateTrafficConfig(prefix, workload.BenchmarkTrafficConfig)...)
	}
	if len(issues) > 0 {
		return errors.New(strings.Join(issues, "\n"))
	}
	return nil
}

func validateTrafficConfig(prefix string, traffic BenchmarkTrafficConfig) []string {
	var issues []string
	if traffic.DatasetName == "random" {
		if traffic.RandomInputLen <= 0 {
			issues = append(issues, prefix+": random_input_len must be positive for random dataset")
		}
		if traffic.RandomOutputLen <= 0 {
			issues = append(issues, prefix+": random_output_len must be positive for random dataset")
		}
	}
	if traffic.Seed != nil && *traffic.Seed < 0 {
		issues = append(issues, prefix+": seed must be non-negative")
	}
	for field, value := range map[string]int{
		"random_prefix_len":              traffic.RandomPrefixLen,
		"sonnet_input_len":               traffic.SonnetInputLen,
		"sonnet_output_len":              traffic.SonnetOutputLen,
		"sonnet_prefix_len":              traffic.SonnetPrefixLen,
		"prefix_repetition_prefix_len":   traffic.PrefixRepetitionPrefixLen,
		"prefix_repetition_suffix_len":   traffic.PrefixRepetitionSuffixLen,
		"prefix_repetition_num_prefixes": traffic.PrefixRepetitionNumPrefixes,
		"prefix_repetition_output_len":   traffic.PrefixRepetitionOutputLen,
		"speed_bench_output_len":         traffic.SpeedBenchOutputLen,
	} {
		if value < 0 {
			issues = append(issues, prefix+": "+field+" must not be negative")
		}
	}
	if traffic.CustomOutputLen != nil && *traffic.CustomOutputLen < -1 {
		issues = append(issues, prefix+": custom_output_len must be -1 or greater")
	}
	if traffic.ShareGPTOutputLen != nil && *traffic.ShareGPTOutputLen <= 0 {
		issues = append(issues, prefix+": sharegpt_output_len must be positive")
	}
	for _, metadata := range traffic.Metadata {
		if strings.TrimSpace(metadata) == "" {
			issues = append(issues, prefix+": metadata values must not be empty")
		}
	}
	for _, goodput := range traffic.Goodput {
		if strings.TrimSpace(goodput) == "" {
			issues = append(issues, prefix+": goodput values must not be empty")
		}
	}
	return issues
}

func BuildPlan(spec Spec, runDir string) []PlannedRun {
	profiles := map[string]Profile{}
	for _, profile := range spec.Profiles {
		profiles[profile.Name] = profile
	}
	var runs []PlannedRun
	for _, workload := range spec.Workloads {
		names := workload.Profiles
		if len(names) == 0 {
			names = sortedProfileNames(profiles)
		}
		for _, profileName := range names {
			profile := profiles[profileName]
			for _, concurrency := range workload.MaxConcurrency {
				repeats := workload.Repeats
				if repeats <= 0 {
					repeats = 1
				}
				for repeat := 0; repeat < repeats; repeat++ {
					runs = append(runs, PlannedRun{
						Profile:     profile,
						Workload:    workload,
						Concurrency: concurrency,
						Repeat:      repeat,
						ResultFile:  ResultPath(runDir, profile.Name, workload.Name, concurrency, repeat, repeats),
					})
				}
			}
		}
	}
	return runs
}

func RunDir(base string, spec Spec, now time.Time) string {
	if strings.TrimSpace(base) != "" {
		return base
	}
	parent := strings.TrimSpace(spec.OutputDir)
	if parent == "" {
		parent = "runs"
	}
	name := Slug(spec.Name)
	if name == "" {
		name = "vllm-bench"
	}
	if spec.Runner.AppendTimestampToRun == nil || *spec.Runner.AppendTimestampToRun {
		name += "-" + now.UTC().Format("20060102T150405Z")
	}
	return filepath.Join(parent, name)
}

func ResultPath(runDir, profileName, workloadName string, concurrency int, repeatInfo ...int) string {
	repeat := 0
	repeats := 1
	if len(repeatInfo) > 0 {
		repeat = repeatInfo[0]
	}
	if len(repeatInfo) > 1 {
		repeats = repeatInfo[1]
	}
	name := fmt.Sprintf("%s__%s__c%d", Slug(profileName), Slug(workloadName), concurrency)
	if repeats > 1 {
		name += fmt.Sprintf("__r%d", repeat+1)
	}
	name += ".json"
	return filepath.Join(runDir, "results", name)
}

func SleepLevelValue(profile Profile) int {
	if profile.SleepLevel == nil {
		return 2
	}
	return *profile.SleepLevel
}

func intPointer(value int) *int {
	return &value
}

func sortedProfileNames(profiles map[string]Profile) []string {
	names := make([]string, 0, len(profiles))
	for name := range profiles {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func Slug(text string) string {
	text = strings.ToLower(strings.TrimSpace(text))
	var out strings.Builder
	lastDash := false
	for _, r := range text {
		isAlnum := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9')
		if isAlnum {
			out.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash {
			out.WriteByte('-')
			lastDash = true
		}
	}
	return strings.Trim(out.String(), "-")
}
