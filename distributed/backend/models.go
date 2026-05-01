package main

import "time"

type Status string

const (
	StatusPending Status = "pending"
	StatusRunning Status = "running"
	StatusDone    Status = "done"
	StatusFailed  Status = "failed"
)

type Checkpoint struct {
	Step      int                `bson:"step"      json:"step"`
	ElapsedS  float64            `bson:"elapsed_s" json:"elapsed_s"`
	Metrics   map[string]float64 `bson:"metrics"   json:"metrics"`
	Timestamp time.Time          `bson:"timestamp" json:"timestamp"`
}

type Experiment struct {
	ExpName      string                 `bson:"_id"                     json:"exp_name"`
	Architecture string                 `bson:"architecture,omitempty"  json:"architecture,omitempty"`
	Dataset      string                 `bson:"dataset"                 json:"dataset"`
	Config       map[string]interface{} `bson:"config"                  json:"config"`
	Status       Status                 `bson:"status"                  json:"status"`
	AgentID      string                 `bson:"agent_id,omitempty"      json:"agent_id,omitempty"`
	Device       string                 `bson:"device,omitempty"        json:"device,omitempty"`
	StartedAt    *time.Time             `bson:"started_at,omitempty"    json:"started_at,omitempty"`
	FinishedAt   *time.Time             `bson:"finished_at,omitempty"   json:"finished_at,omitempty"`
	Checkpoints  []Checkpoint           `bson:"checkpoints,omitempty"   json:"checkpoints,omitempty"`
	FinalMetrics map[string]float64     `bson:"final_metrics,omitempty" json:"final_metrics,omitempty"`
	DurationS    float64                `bson:"duration_s,omitempty"    json:"duration_s,omitempty"`
	Error        string                 `bson:"error,omitempty"         json:"error,omitempty"`
	CreatedAt    time.Time              `bson:"created_at"              json:"created_at"`
	UpdatedAt    time.Time              `bson:"updated_at"              json:"updated_at"`
}

type CreateExperimentRequest struct {
	ExpName      string                 `json:"exp_name"`
	Architecture string                 `json:"architecture,omitempty"`
	Config       map[string]interface{} `json:"config"`
}

type ClaimRequest struct {
	AgentID string `json:"agent_id"`
	Device  string `json:"device,omitempty"`
}

type CheckpointRequest struct {
	Step     int                `json:"step"`
	ElapsedS float64            `json:"elapsed_s"`
	Metrics  map[string]float64 `json:"metrics"`
}

type FinishRequest struct {
	FinalMetrics map[string]float64 `json:"final_metrics"`
	DurationS    float64            `json:"duration_s"`
	Checkpoints  []Checkpoint       `json:"checkpoints,omitempty"`
}

type FailRequest struct {
	Error string `json:"error"`
}

type StatsResponse struct {
	Dataset string         `json:"dataset"`
	Total   int64          `json:"total"`
	Counts  map[string]int `json:"counts"`
}

type SummaryResponse struct {
	Dataset        string         `json:"dataset"`
	Total          int64          `json:"total"`
	Counts         map[string]int `json:"counts"`
	DoneCount      int            `json:"done_count"`
	AvgDurationS   float64        `json:"avg_duration_s"`
	TotalDurationS float64        `json:"total_duration_s"`
	EtaS           float64        `json:"eta_s"`
	EtaMethod      string         `json:"eta_method"`
}

type ErrorResponse struct {
	Error   string `json:"error"`
	Code    string `json:"code,omitempty"`
	Details string `json:"details,omitempty"`
}
