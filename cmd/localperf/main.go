package main

import (
	"os"

	"github.com/osolmaz/localperf/internal/benchcli"
)

func main() {
	benchcli.Main(os.Args[1:])
}
