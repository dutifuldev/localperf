package bench

import "context"

type Endpoint struct {
	BaseURL string
	Model   string
}

type RunPaths struct {
	RunDir         string
	Artifact       string
	ResultsDir     string
	LogsDir        string
	NormalizedSpec string
}

type EngineMetadata struct {
	Name    string
	Type    string
	Version string
	Fields  map[string]string
}

type Profile struct {
	Name   string
	Model  string
	Engine string
}

type Workload struct {
	Name        string
	Concurrency int
	Samples     int
	Repeat      int
}

type RawResult struct {
	Path string
}

type RunMetadata struct {
	RunID    string
	Engine   string
	Profile  string
	Workload string
}

type Result struct {
	Completed int
	Failed    int
}

type Engine interface {
	Name() string
	Start(ctx context.Context, profile Profile, paths RunPaths) error
	Stop(ctx context.Context) error
	Health(ctx context.Context) error
	Endpoint(profile Profile) Endpoint
	Metadata(ctx context.Context) (EngineMetadata, error)
}

type SleepCapable interface {
	Sleep(ctx context.Context, level int) error
	Wake(ctx context.Context) error
}

type LoadGenerator interface {
	Run(ctx context.Context, endpoint Endpoint, workload Workload, concurrency int, paths RunPaths) (RawResult, error)
}

type Reporter interface {
	Normalize(raw RawResult, metadata RunMetadata) (Result, error)
	Write(result Result, paths RunPaths) error
}
