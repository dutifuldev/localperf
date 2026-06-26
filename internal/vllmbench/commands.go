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
	args := []string{
		spec.Runner.VLLMCommand,
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
	args := []string{
		spec.Runner.VLLMBenchCommand,
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
	if workload.DatasetName == "random" {
		args = append(args,
			"--random-input-len", strconv.Itoa(workload.RandomInputLen),
			"--random-output-len", strconv.Itoa(workload.RandomOutputLen),
		)
	}
	if strings.TrimSpace(workload.Endpoint) != "" {
		args = append(args, "--endpoint", workload.Endpoint)
	}
	if workload.IgnoreEOS {
		args = append(args, "--ignore-eos")
	}
	if workload.Temperature != nil {
		args = append(args, "--temperature", trimFloat(*workload.Temperature))
	}
	args = append(args, workload.ExtraArgs...)
	return CommandSpec{
		Env:  cloneMap(spec.Env),
		Args: args,
	}
}

func WarmupCommand(spec Spec, profile Profile, runDir string) CommandSpec {
	warmup := spec.Warmup
	resultFile := ResultPath(runDir, profile.Name, "warmup", warmup.MaxConcurrency)
	args := []string{
		spec.Runner.VLLMBenchCommand,
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
	if warmup.DatasetName == "random" {
		args = append(args,
			"--random-input-len", strconv.Itoa(warmup.RandomInputLen),
			"--random-output-len", strconv.Itoa(warmup.RandomOutputLen),
		)
	}
	if strings.TrimSpace(warmup.Endpoint) != "" {
		args = append(args, "--endpoint", warmup.Endpoint)
	}
	args = append(args, warmup.ExtraArgs...)
	return CommandSpec{
		Env:  cloneMap(spec.Env),
		Args: args,
	}
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
