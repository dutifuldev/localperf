package bench

import (
	"regexp"
	"strings"
)

var safeShellArgPattern = regexp.MustCompile(`^[A-Za-z0-9_./:=,+@-]+$`)

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

func NormalizeWorkloadPhase(phase string) string {
	return Slug(strings.ToLower(strings.TrimSpace(phase)))
}

func NormalizeReportPhase(phase string) string {
	if phase = NormalizeWorkloadPhase(phase); phase != "" {
		return phase
	}
	return "mixed"
}

func PhaseRank(phase string) int {
	switch phase {
	case "decode":
		return 0
	case "prefill":
		return 1
	case "mixed":
		return 2
	default:
		return 3
	}
}

func PhaseTitle(phase string) string {
	phase = NormalizeReportPhase(phase)
	parts := strings.Split(phase, "-")
	for i, part := range parts {
		if part == "" {
			continue
		}
		parts[i] = strings.ToUpper(part[:1]) + part[1:]
	}
	return strings.Join(parts, " ")
}

func FirstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func ShellQuote(args []string) string {
	parts := make([]string, 0, len(args))
	for _, arg := range args {
		if arg == "" {
			parts = append(parts, "''")
			continue
		}
		if safeShellArgPattern.MatchString(arg) {
			parts = append(parts, arg)
			continue
		}
		parts = append(parts, "'"+strings.ReplaceAll(arg, "'", "'\"'\"'")+"'")
	}
	return strings.Join(parts, " ")
}
