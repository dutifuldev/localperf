package main

import (
	"os"

	"github.com/dutifuldev/localperf/internal/benchcli"
)

func main() {
	benchcli.LocalPerfMain(os.Args[1:])
}
