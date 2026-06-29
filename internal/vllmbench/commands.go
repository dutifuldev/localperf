package vllmbench

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/dutifuldev/localperf/internal/collections"
)

var safeShellArgPattern = regexp.MustCompile(`^[A-Za-z0-9_./:=,+@-]+$`)

type CommandSpec struct {
	Dir  string            `json:"dir,omitempty"`
	Env  map[string]string `json:"env,omitempty"`
	Args []string          `json:"args"`
}

func ServeCommand(spec Spec, profile Profile) CommandSpec {
	engine := EngineForProfile(spec, profile)
	command := engineCommand(spec, engine)
	args := []string{
		command,
		"serve",
		profile.Model,
		"--host", profile.Host,
		"--port", strconv.Itoa(profile.Port),
	}
	appendIntArg := func(flag string, value int) {
		if value > 0 {
			args = append(args, flag, strconv.Itoa(value))
		}
	}
	appendFloatArg := func(flag string, value float64) {
		if value > 0 {
			args = append(args, flag, trimFloat(value))
		}
	}
	appendStringArg := func(flag, value string) {
		if strings.TrimSpace(value) != "" {
			args = append(args, flag, value)
		}
	}
	appendIntArg("--max-model-len", profile.MaxModelLen)
	appendIntArg("--max-num-seqs", profile.MaxNumSeqs)
	appendIntArg("--max-num-batched-tokens", profile.MaxNumBatchedTokens)
	appendFloatArg("--gpu-memory-utilization", profile.GPUMemoryUtilization)
	appendStringArg("--kv-cache-dtype", profile.KVCacheDType)
	appendStringArg("--attention-backend", profile.AttentionBackend)
	appendStringArg("--moe-backend", profile.MoEBackend)
	if profile.EnableSleepMode {
		args = append(args, "--enable-sleep-mode")
	}
	args = append(args, profileExtraArgs(profile)...)
	return CommandSpec{
		Env:  mergeEnv(spec.Env, engine.Env, profile.Env, profile.EnableSleepMode),
		Args: args,
	}
}

func BenchCommand(spec Spec, run PlannedRun) CommandSpec {
	workload := run.Workload
	profile := run.Profile
	engine := EngineForProfile(spec, profile)
	command := engineBenchCommand(spec, engine)
	args := []string{
		command,
		"bench", "serve",
		"--backend", workload.Backend,
		"--host", profile.Host,
		"--port", strconv.Itoa(profile.Port),
		"--model", profile.Model,
		"--dataset-name", workload.DatasetName,
		"--num-prompts", strconv.Itoa(workload.NumPrompts),
		"--max-concurrency", strconv.Itoa(run.Concurrency),
		"--request-rate", workload.RequestRate,
		"--save-result",
		"--result-filename", run.ResultFile,
	}
	args = appendTrafficArgs(args, workload.BenchmarkTrafficConfig)
	if strings.TrimSpace(workload.Endpoint) != "" {
		args = append(args, "--endpoint", workload.Endpoint)
	}
	if workload.IgnoreEOS {
		args = append(args, "--ignore-eos")
	}
	if workload.Temperature != nil {
		args = append(args, "--temperature", trimFloat(*workload.Temperature))
	}
	return CommandSpec{
		Env:  mergeEnv(spec.Env, engine.Env, nil, false),
		Args: args,
	}
}

func LoadCommand(spec Spec, run PlannedRun) CommandSpec {
	if run.Workload.LoadGenerator == LoadGeneratorLocalPerfHTTP {
		return LocalPerfHTTPCommand(run)
	}
	return BenchCommand(spec, run)
}

func LocalPerfHTTPCommand(run PlannedRun) CommandSpec {
	args := []string{
		"localperf", "bench", "http-load",
		"--backend", run.Workload.Backend,
		"--base-url", baseURL(run.Profile),
		"--model", run.Profile.Model,
		"--dataset-name", run.Workload.DatasetName,
		"--num-prompts", strconv.Itoa(run.Workload.NumPrompts),
		"--max-concurrency", strconv.Itoa(run.Concurrency),
		"--request-rate", run.Workload.RequestRate,
		"--result-filename", run.ResultFile,
	}
	if run.Workload.DatasetName == "random" {
		if run.Workload.RandomInputLen > 0 {
			args = append(args, "--random-input-len", strconv.Itoa(run.Workload.RandomInputLen))
		}
		if run.Workload.RandomOutputLen > 0 {
			args = append(args, "--random-output-len", strconv.Itoa(run.Workload.RandomOutputLen))
		}
	}
	if strings.TrimSpace(run.Workload.Endpoint) != "" {
		args = append(args, "--endpoint", run.Workload.Endpoint)
	}
	if run.Workload.IgnoreEOS {
		args = append(args, "--ignore-eos")
	}
	if run.Workload.Temperature != nil {
		args = append(args, "--temperature", trimFloat(*run.Workload.Temperature))
	}
	return CommandSpec{Args: args}
}

func WarmupCommand(spec Spec, profile Profile, runDir string) CommandSpec {
	warmup := spec.Warmup
	engine := EngineForProfile(spec, profile)
	command := engineBenchCommand(spec, engine)
	resultFile := ResultPath(runDir, profile.Name, "warmup", warmup.MaxConcurrency)
	args := []string{
		command,
		"bench", "serve",
		"--backend", warmup.Backend,
		"--host", profile.Host,
		"--port", strconv.Itoa(profile.Port),
		"--model", profile.Model,
		"--dataset-name", warmup.DatasetName,
		"--num-prompts", strconv.Itoa(warmup.NumPrompts),
		"--max-concurrency", strconv.Itoa(warmup.MaxConcurrency),
		"--request-rate", warmup.RequestRate,
		"--save-result",
		"--result-filename", resultFile,
	}
	args = appendTrafficArgs(args, warmup.BenchmarkTrafficConfig)
	if strings.TrimSpace(warmup.Endpoint) != "" {
		args = append(args, "--endpoint", warmup.Endpoint)
	}
	return CommandSpec{
		Env:  mergeEnv(spec.Env, engine.Env, nil, false),
		Args: args,
	}
}

func EngineForProfile(spec Spec, profile Profile) EngineConfig {
	for _, engine := range spec.Engines {
		if engine.Name == profile.Engine {
			return engine
		}
	}
	if len(spec.Engines) > 0 {
		return spec.Engines[0]
	}
	return EngineConfig{Name: "vllm", Type: "vllm-managed", Command: spec.Runner.VLLMCommand, BenchCommand: spec.Runner.VLLMBenchCommand}
}

func engineCommand(spec Spec, engine EngineConfig) string {
	if strings.TrimSpace(engine.Command) == "" {
		return spec.Runner.VLLMCommand
	}
	if engine.Command == "vllm" && strings.TrimSpace(spec.Runner.VLLMCommand) != "" && spec.Runner.VLLMCommand != "vllm" {
		return spec.Runner.VLLMCommand
	}
	return engine.Command
}

func engineBenchCommand(spec Spec, engine EngineConfig) string {
	if strings.TrimSpace(engine.BenchCommand) != "" && !(engine.BenchCommand == "vllm" && spec.Runner.VLLMBenchCommand != "" && spec.Runner.VLLMBenchCommand != "vllm") {
		return engine.BenchCommand
	}
	if strings.TrimSpace(spec.Runner.VLLMBenchCommand) != "" {
		return spec.Runner.VLLMBenchCommand
	}
	return engineCommand(spec, engine)
}

func appendTrafficArgs(args []string, traffic BenchmarkTrafficConfig) []string {
	builder := argBuilder(args)
	appendCommonTrafficArgs(&builder, traffic)
	appendRandomTrafficArgs(&builder, traffic)
	appendDatasetTrafficArgs(&builder, traffic)
	builder.repeated("--metadata", traffic.Metadata)
	builder.repeated("--goodput", traffic.Goodput)
	return append(builder, traffic.ExtraArgs...)
}

type argBuilder []string

func (builder *argBuilder) string(flag, value string) {
	if strings.TrimSpace(value) != "" {
		*builder = append(*builder, flag, value)
	}
}

func (builder *argBuilder) integer(flag string, value int) {
	if value > 0 {
		*builder = append(*builder, flag, strconv.Itoa(value))
	}
}

func (builder *argBuilder) optionalInteger(flag string, value *int) {
	if value != nil {
		*builder = append(*builder, flag, strconv.Itoa(*value))
	}
}

func (builder *argBuilder) flag(enabled bool, flag string) {
	if enabled {
		*builder = append(*builder, flag)
	}
}

func (builder *argBuilder) repeated(flag string, values []string) {
	for _, value := range values {
		builder.string(flag, value)
	}
}

func appendCommonTrafficArgs(builder *argBuilder, traffic BenchmarkTrafficConfig) {
	builder.string("--dataset-path", traffic.DatasetPath)
	builder.optionalInteger("--seed", traffic.Seed)
	builder.flag(traffic.DisableShuffle, "--disable-shuffle")
	builder.flag(traffic.NoOversample, "--no-oversample")
	builder.flag(traffic.SkipChatTemplate, "--skip-chat-template")
	builder.flag(traffic.SaveDetailed, "--save-detailed")
	builder.flag(traffic.PlotDatasetStats, "--plot-dataset-stats")
}

func appendRandomTrafficArgs(builder *argBuilder, traffic BenchmarkTrafficConfig) {
	if traffic.DatasetName != "random" {
		return
	}
	builder.integer("--random-input-len", traffic.RandomInputLen)
	builder.integer("--random-output-len", traffic.RandomOutputLen)
	builder.string("--random-range-ratio", traffic.RandomRangeRatio)
	builder.integer("--random-prefix-len", traffic.RandomPrefixLen)
}

func appendDatasetTrafficArgs(builder *argBuilder, traffic BenchmarkTrafficConfig) {
	builder.optionalInteger("--custom-output-len", traffic.CustomOutputLen)
	builder.optionalInteger("--sharegpt-output-len", traffic.ShareGPTOutputLen)
	builder.integer("--sonnet-input-len", traffic.SonnetInputLen)
	builder.integer("--sonnet-output-len", traffic.SonnetOutputLen)
	builder.integer("--sonnet-prefix-len", traffic.SonnetPrefixLen)
	builder.integer("--prefix-repetition-prefix-len", traffic.PrefixRepetitionPrefixLen)
	builder.integer("--prefix-repetition-suffix-len", traffic.PrefixRepetitionSuffixLen)
	builder.integer("--prefix-repetition-num-prefixes", traffic.PrefixRepetitionNumPrefixes)
	builder.integer("--prefix-repetition-output-len", traffic.PrefixRepetitionOutputLen)
	builder.string("--speed-bench-dataset-subset", traffic.SpeedBenchDatasetSubset)
	builder.integer("--speed-bench-output-len", traffic.SpeedBenchOutputLen)
	builder.string("--speed-bench-category", traffic.SpeedBenchCategory)
	builder.string("--extra-body", traffic.ExtraBody)
}

func profileExtraArgs(profile Profile) []string {
	args := make([]string, 0, len(profile.Args)+len(profile.EngineArgs))
	args = append(args, profile.Args...)
	args = append(args, profile.EngineArgs...)
	return args
}

func ShellQuote(args []string) string {
	parts := make([]string, 0, len(args))
	for _, arg := range args {
		parts = append(parts, shellQuote(arg))
	}
	return strings.Join(parts, " ")
}

func WithProcessEnv(env map[string]string) []string {
	out := os.Environ()
	for _, key := range collections.SortedKeys(env) {
		out = append(out, key+"="+env[key])
	}
	return out
}

func mergeEnv(specEnv, engineEnv, profileEnv map[string]string, devMode bool) map[string]string {
	out := cloneMap(specEnv)
	for key, value := range engineEnv {
		out[key] = value
	}
	for key, value := range profileEnv {
		out[key] = value
	}
	if devMode {
		out["VLLM_SERVER_DEV_MODE"] = "1"
	}
	return out
}

func cloneMap(values map[string]string) map[string]string {
	out := map[string]string{}
	for key, value := range values {
		out[key] = value
	}
	return out
}

func trimFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}

func shellQuote(arg string) string {
	if arg == "" {
		return "''"
	}
	if safeShellArgPattern.MatchString(arg) {
		return arg
	}
	return "'" + strings.ReplaceAll(arg, "'", "'\"'\"'") + "'"
}

func CommandSummary(command CommandSpec) string {
	if len(command.Args) == 0 {
		return ""
	}
	if len(command.Env) == 0 {
		return ShellQuote(command.Args)
	}
	parts := make([]string, 0, len(command.Env)+len(command.Args))
	for _, key := range collections.SortedKeys(command.Env) {
		parts = append(parts, fmt.Sprintf("%s=%s", key, shellQuote(redactEnvValue(key, command.Env[key]))))
	}
	parts = append(parts, ShellQuote(command.Args))
	return strings.Join(parts, " ")
}

func redactedEnv(values map[string]string) map[string]string {
	if values == nil {
		return nil
	}
	out := make(map[string]string, len(values))
	for key, value := range values {
		out[key] = redactEnvValue(key, value)
	}
	return out
}

func redactEnvValue(key, value string) string {
	if isSensitiveEnvKey(key) {
		return "<redacted>"
	}
	return value
}

func isSensitiveEnvKey(key string) bool {
	upper := strings.ToUpper(key)
	return hasSensitiveEnvPart(upper) || containsSensitiveEnvMarker(upper)
}

func hasSensitiveEnvPart(upper string) bool {
	for _, part := range strings.FieldsFunc(upper, isEnvKeySeparator) {
		if sensitiveEnvParts[part] {
			return true
		}
	}
	return false
}

func isEnvKeySeparator(r rune) bool {
	return !((r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9'))
}

var sensitiveEnvParts = map[string]bool{
	"AUTH":          true,
	"AUTHORIZATION": true,
	"COOKIE":        true,
	"CREDENTIAL":    true,
	"CREDENTIALS":   true,
	"KEY":           true,
	"PASS":          true,
	"PASSWORD":      true,
	"SECRET":        true,
	"TOKEN":         true,
}

func containsSensitiveEnvMarker(upper string) bool {
	for _, marker := range []string{"TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"} {
		if strings.Contains(upper, marker) {
			return true
		}
	}
	return false
}
