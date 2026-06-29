# AGENTS.md

This is a Go repository for local inference benchmark tooling.

Before finishing code changes, run:

```sh
go test ./...
go vet ./...
npx -y @simpledoc/simpledoc check
go run github.com/dutifuldev/slophammer/go/cmd/slophammer-go@v0.4.0 check .
```

The default CI path is non-mutating. `scripts/check-crap.sh` and
`scripts/check-mutation.sh` are explicit Slophammer debt gates; run them when
working on the core runner or artifact internals and expect them to require
focused cleanup if they fail.

For changes that affect benchmark behavior, artifacts, or reports, also run one
small dry benchmark case and validate the SQLite artifact:

```sh
rm -rf /tmp/localperf-onecase-dry /tmp/localperf-onecase-dry.sqlite
go run ./cmd/localperf bench run \
  --dry-run \
  --spec examples/diffusiongemma-vllm-standard/spec.json \
  --profile 4k-reference \
  --workload claim-repro-1k-out1024 \
  --concurrency 1 \
  --run-dir /tmp/localperf-onecase-dry
go run ./cmd/localperf artifact check /tmp/localperf-onecase-dry.sqlite
```

Keep benchmark safety behavior conservative. Do not lower memory floors or
remove guardrails to make a run pass.

Keep production Go code under `cmd` and `internal`. Treat `examples`, `docs`,
and `runs` as fixtures, documentation, or local run data rather than production
library code.

Slophammer standards are applied through `slophammer.yml`; update that policy
and the matching local scripts/CI together.
