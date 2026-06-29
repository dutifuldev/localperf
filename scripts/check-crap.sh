#!/usr/bin/env sh
set -eu

go run github.com/dutifuldev/slophammer/go/cmd/slophammer-go@v0.4.0 crap . --max-score 8
