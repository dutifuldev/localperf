package vllmbench

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

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
	args = append(args, profile.Args...)
	return CommandSpec{
		Env:  mergeEnv(spec.Env, profile.Env, profile.EnableSleepMode),
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
		Env:  cloneMap(spec.Env),
		Args: args,
	}
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
		Env:  cloneMap(spec.Env),
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
	appendStringArg := func(flag, value string) {
		if strings.TrimSpace(value) != "" {
			args = append(args, flag, value)
		}
	}
	appendIntArg := func(flag string, value int) {
		if value > 0 {
			args = append(args, flag, strconv.Itoa(value))
		}
	}
	appendIntPointerArg := func(flag string, value *int) {
		if value != nil {
			args = append(args, flag, strconv.Itoa(*value))
		}
	}
	appendRepeatedArg := func(flag string, values []string) {
		for _, value := range values {
			if strings.TrimSpace(value) != "" {
				args = append(args, flag, value)
			}
		}
	}

	appendStringArg("--dataset-path", traffic.DatasetPath)
	if traffic.Seed != nil {
		args = append(args, "--seed", strconv.Itoa(*traffic.Seed))
	}
	if traffic.DisableShuffle {
		args = append(args, "--disable-shuffle")
	}
	if traffic.NoOversample {
		args = append(args, "--no-oversample")
	}
	if traffic.SkipChatTemplate {
		args = append(args, "--skip-chat-template")
	}
	if traffic.SaveDetailed {
		args = append(args, "--save-detailed")
	}
	if traffic.PlotDatasetStats {
		args = append(args, "--plot-dataset-stats")
	}
	if traffic.DatasetName == "random" {
		appendIntArg("--random-input-len", traffic.RandomInputLen)
		appendIntArg("--random-output-len", traffic.RandomOutputLen)
		appendStringArg("--random-range-ratio", traffic.RandomRangeRatio)
		appendIntArg("--random-prefix-len", traffic.RandomPrefixLen)
	}
	appendIntPointerArg("--custom-output-len", traffic.CustomOutputLen)
	appendIntPointerArg("--sharegpt-output-len", traffic.ShareGPTOutputLen)
	appendIntArg("--sonnet-input-len", traffic.SonnetInputLen)
	appendIntArg("--sonnet-output-len", traffic.SonnetOutputLen)
	appendIntArg("--sonnet-prefix-len", traffic.SonnetPrefixLen)
	appendIntArg("--prefix-repetition-prefix-len", traffic.PrefixRepetitionPrefixLen)
	appendIntArg("--prefix-repetition-suffix-len", traffic.PrefixRepetitionSuffixLen)
	appendIntArg("--prefix-repetition-num-prefixes", traffic.PrefixRepetitionNumPrefixes)
	appendIntArg("--prefix-repetition-output-len", traffic.PrefixRepetitionOutputLen)
	appendStringArg("--speed-bench-dataset-subset", traffic.SpeedBenchDatasetSubset)
	appendIntArg("--speed-bench-output-len", traffic.SpeedBenchOutputLen)
	appendStringArg("--speed-bench-category", traffic.SpeedBenchCategory)
	appendStringArg("--extra-body", traffic.ExtraBody)
	appendRepeatedArg("--metadata", traffic.Metadata)
	appendRepeatedArg("--goodput", traffic.Goodput)
	args = append(args, traffic.ExtraArgs...)
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
	for _, key := range sortedMapKeys(env) {
		out = append(out, key+"="+env[key])
	}
	return out
}

func mergeEnv(specEnv, profileEnv map[string]string, devMode bool) map[string]string {
	out := cloneMap(specEnv)
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

func sortedMapKeys(values map[string]string) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func trimFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}

func shellQuote(arg string) string {
	if arg == "" {
		return "''"
	}
	if strings.IndexFunc(arg, func(r rune) bool {
		return !(r == '_' || r == '-' || r == '.' || r == '/' || r == ':' || r == '=' || r == ',' || r == '+' || r == '@' || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9'))
	}) == -1 {
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
	for _, key := range sortedMapKeys(command.Env) {
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
	parts := strings.FieldsFunc(upper, func(r rune) bool {
		return !((r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9'))
	})
	for _, part := range parts {
		switch part {
		case "AUTH", "AUTHORIZATION", "COOKIE", "CREDENTIAL", "CREDENTIALS", "KEY", "PASS", "PASSWORD", "SECRET", "TOKEN":
			return true
		}
	}
	for _, marker := range []string{"TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"} {
		if strings.Contains(upper, marker) {
			return true
		}
	}
	return false
}
