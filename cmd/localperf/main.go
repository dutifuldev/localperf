package main

import (
	"os"

	"github.com/dutifuldev/localperf/internal/benchcli"
)

func main() {
	benchcli.Main(os.Args[1:])
}
