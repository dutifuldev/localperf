package main

import (
	"os"

	"github.com/dutifuldev/localperf/internal/benchcli"
)

func main() {
	benchcli.VLLMBenchMain(os.Args[1:])
}
