package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/dutifuldev/localperf/internal/collections"
)

type row struct {
	CandidateID string         `json:"candidate_id"`
	Status      string         `json:"status"`
	StartedAt   string         `json:"started_at"`
	FinishedAt  string         `json:"finished_at"`
	Candidate   map[string]any `json:"candidate"`
}

type progress struct {
	Rows            int
	Target          int
	Statuses        map[string]int
	Contexts        map[int]bool
	Seqs            map[int]bool
	FirstStartedAt  time.Time
	LatestFinished  time.Time
	LatestCandidate string
	LatestStatus    string
}

func main() {
	var resultsPath string
	var targetRows int
	flag.StringVar(&resultsPath, "results", "", "path to sweep results JSONL")
	flag.IntVar(&targetRows, "target-rows", 100, "target row count for ETA")
	flag.Parse()
	if strings.TrimSpace(resultsPath) == "" {
		fmt.Fprintln(os.Stderr, "missing --results")
		os.Exit(2)
	}
	file, err := os.Open(resultsPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer file.Close()
	report, err := readProgress(file, targetRows)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	printProgress(report)
}

func readProgress(reader io.Reader, targetRows int) (progress, error) {
	out := progress{
		Target:   targetRows,
		Statuses: map[string]int{},
		Contexts: map[int]bool{},
		Seqs:     map[int]bool{},
	}
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 1024*1024), 64*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var item row
		if err := json.Unmarshal([]byte(line), &item); err != nil {
			return out, err
		}
		out.Rows++
		out.Statuses[item.Status]++
		out.LatestCandidate = item.CandidateID
		out.LatestStatus = item.Status
		if started, err := time.Parse(time.RFC3339, item.StartedAt); err == nil {
			if out.FirstStartedAt.IsZero() || started.Before(out.FirstStartedAt) {
				out.FirstStartedAt = started
			}
		}
		if finished, err := time.Parse(time.RFC3339, item.FinishedAt); err == nil && finished.After(out.LatestFinished) {
			out.LatestFinished = finished
		}
		if context := intField(item.Candidate, "max_model_len"); context > 0 {
			out.Contexts[context] = true
		}
		if seqs := intField(item.Candidate, "max_num_seqs"); seqs > 0 {
			out.Seqs[seqs] = true
		}
	}
	if err := scanner.Err(); err != nil {
		return out, err
	}
	return out, nil
}

func printProgress(report progress) {
	fmt.Printf("rows: %d / %d\n", report.Rows, report.Target)
	fmt.Printf("latest: %s (%s)\n", report.LatestCandidate, report.LatestStatus)
	fmt.Printf("contexts: %s\n", collections.JoinIntKeys(report.Contexts, ", "))
	fmt.Printf("max_num_seqs: %s\n", collections.JoinIntKeys(report.Seqs, ", "))
	fmt.Println("statuses:")
	for _, status := range collections.SortedKeys(report.Statuses) {
		fmt.Printf("  %s: %d\n", status, report.Statuses[status])
	}
	if report.Rows > 1 && !report.FirstStartedAt.IsZero() && !report.LatestFinished.IsZero() {
		elapsed := report.LatestFinished.Sub(report.FirstStartedAt)
		perRow := elapsed / time.Duration(report.Rows)
		fmt.Printf("elapsed: %s\n", elapsed.Round(time.Second))
		fmt.Printf("mean row time: %s\n", perRow.Round(time.Second))
		if report.Rows < report.Target {
			remaining := report.Target - report.Rows
			fmt.Printf("rough ETA to %d rows: %s\n", report.Target, (perRow * time.Duration(remaining)).Round(time.Minute))
		}
	}
}

func intField(row map[string]any, key string) int {
	value, ok := row[key]
	if !ok || value == nil {
		return 0
	}
	number, ok := value.(float64)
	if !ok {
		return 0
	}
	return int(number)
}
