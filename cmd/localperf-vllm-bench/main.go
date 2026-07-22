package main

import (
	"os"

	"github.com/osolmaz/localperf/internal/benchcli"
)

func main() {
	benchcli.VLLMBenchMain(os.Args[1:])
}
