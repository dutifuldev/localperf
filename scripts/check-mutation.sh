#!/usr/bin/env sh
set -eu

mode="${LOCALPERF_MUTATION_MODE:-scan}"

case "$mode" in
  scan)
    go run github.com/osolmaz/slophammer/go/cmd/slophammer-go@v0.4.1 mutate . --scan
    ;;
  full)
    go run github.com/osolmaz/slophammer/go/cmd/slophammer-go@v0.4.1 mutate .
    ;;
  *)
    echo "unknown LOCALPERF_MUTATION_MODE: $mode" >&2
    echo "expected scan or full" >&2
    exit 2
    ;;
esac
