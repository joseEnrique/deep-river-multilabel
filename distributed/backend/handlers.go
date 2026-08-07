package main

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"github.com/go-chi/chi/v5"
	"go.mongodb.org/mongo-driver/bson"
)

type Handlers struct {
	Store *Store
}

func writeJSON(w http.ResponseWriter, code int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, ErrorResponse{Error: msg})
}

func mapErr(w http.ResponseWriter, err error) bool {
	switch {
	case errors.Is(err, ErrNotFound):
		writeErr(w, http.StatusNotFound, err.Error())
	case errors.Is(err, ErrAlreadyExists):
		writeErr(w, http.StatusConflict, err.Error())
	case errors.Is(err, ErrInvalidState):
		writeErr(w, http.StatusConflict, err.Error())
	case errors.Is(err, ErrInvalidDataset):
		writeErr(w, http.StatusBadRequest, err.Error())
	default:
		return false
	}
	return true
}

func (h *Handlers) Health(w http.ResponseWriter, r *http.Request) {
	if err := h.Store.client.Ping(r.Context(), nil); err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "down", "error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (h *Handlers) ListDatasets(w http.ResponseWriter, r *http.Request) {
	out, err := h.Store.ListDatasets(r.Context())
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{"datasets": out})
}

func (h *Handlers) Stats(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	stats, err := h.Store.Stats(r.Context(), dataset)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, stats)
}

func (h *Handlers) Summary(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	out, err := h.Store.Summary(r.Context(), dataset)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, out)
}

func (h *Handlers) ListExperiments(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	q := r.URL.Query()
	status := q.Get("status")
	limit, _ := strconv.ParseInt(q.Get("limit"), 10, 64)
	offset, _ := strconv.ParseInt(q.Get("offset"), 10, 64)
	if limit == 0 {
		limit = 1000
	}
	out, err := h.Store.List(r.Context(), dataset, status, limit, offset)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"dataset":     dataset,
		"count":       len(out),
		"experiments": out,
	})
}

func (h *Handlers) CreateExperiment(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	var req CreateExperimentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	if strings.TrimSpace(req.ExpName) == "" {
		writeErr(w, http.StatusBadRequest, "exp_name is required")
		return
	}
	if req.Config == nil {
		writeErr(w, http.StatusBadRequest, "config is required")
		return
	}
	exp := &Experiment{
		ExpName:      req.ExpName,
		Architecture: req.Architecture,
		Config:       req.Config,
		Status:       StatusPending,
		ComputeScore: req.ComputeScore,
		SizeTier:     req.SizeTier,
	}
	if err := h.Store.Create(r.Context(), dataset, exp); err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, exp)
}

func (h *Handlers) BulkCreate(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	var req struct {
		Experiments []CreateExperimentRequest `json:"experiments"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	if len(req.Experiments) == 0 {
		writeErr(w, http.StatusBadRequest, "experiments must be non-empty")
		return
	}
	exps := make([]*Experiment, 0, len(req.Experiments))
	for _, e := range req.Experiments {
		if strings.TrimSpace(e.ExpName) == "" || e.Config == nil {
			writeErr(w, http.StatusBadRequest, "each experiment requires exp_name and config")
			return
		}
		exps = append(exps, &Experiment{
			ExpName:      e.ExpName,
			Architecture: e.Architecture,
			Config:       e.Config,
			Status:       StatusPending,
			ComputeScore: e.ComputeScore,
			SizeTier:     e.SizeTier,
		})
	}
	inserted, skipped, err := h.Store.BulkCreate(r.Context(), dataset, exps)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"inserted": inserted,
		"skipped":  skipped,
		"total":    len(exps),
	})
}

func (h *Handlers) GetExperiment(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	name := chi.URLParam(r, "name")
	exp, err := h.Store.Get(r.Context(), dataset, name)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, exp)
}

func (h *Handlers) ReplaceExperiment(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	name := chi.URLParam(r, "name")
	var exp Experiment
	if err := json.NewDecoder(r.Body).Decode(&exp); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	exp.ExpName = name
	if err := h.Store.Replace(r.Context(), dataset, &exp); err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, exp)
}

func (h *Handlers) PatchExperiment(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	name := chi.URLParam(r, "name")
	var patch map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&patch); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	exp, err := h.Store.Patch(r.Context(), dataset, name, patch)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, exp)
}

func (h *Handlers) DeleteExperiment(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	name := chi.URLParam(r, "name")
	if err := h.Store.Delete(r.Context(), dataset, name); err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *Handlers) ClaimExperiment(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	name := chi.URLParam(r, "name")
	var req ClaimRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	if req.AgentID == "" {
		req.AgentID = r.Header.Get("X-Agent-ID")
	}
	if req.AgentID == "" {
		writeErr(w, http.StatusBadRequest, "agent_id is required")
		return
	}
	exp, err := h.Store.Claim(r.Context(), dataset, name, req.AgentID, req.Device)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, exp)
}

// validSortPath keeps the sort spec to plain BSON paths so a caller cannot
// smuggle operators into the sort document.
func validSortPath(p string) bool {
	if p == "" || len(p) > 128 {
		return false
	}
	for _, r := range p {
		ok := r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' ||
			r >= '0' && r <= '9' || r == '_' || r == '.'
		if !ok {
			return false
		}
	}
	return true
}

// parseSort turns "config.epochs:asc,config.hidden_dim:desc" into a bson.D.
// Direction is optional and defaults to ascending.
func parseSort(s string) (bson.D, error) {
	out := bson.D{}
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		kv := strings.SplitN(part, ":", 2)
		key := strings.TrimSpace(kv[0])
		if !validSortPath(key) {
			return nil, fmt.Errorf("invalid sort key: %q", key)
		}
		dir := 1
		if len(kv) == 2 && strings.EqualFold(strings.TrimSpace(kv[1]), "desc") {
			dir = -1
		}
		out = append(out, bson.E{Key: key, Value: dir})
	}
	return out, nil
}

func (h *Handlers) BackfillScores(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	// ?all=true recomputes every document; the default only fills the gaps.
	onlyMissing := r.URL.Query().Get("all") != "true"

	matched, modified, missing, err := h.Store.BackfillScores(r.Context(), dataset, onlyMissing)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, BackfillResponse{
		Dataset: dataset, Matched: matched, Modified: modified, Missing: missing,
	})
}

func (h *Handlers) ClaimNextExperiment(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	var req ClaimNextRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	if req.AgentID == "" {
		req.AgentID = r.Header.Get("X-Agent-ID")
	}
	if req.AgentID == "" {
		writeErr(w, http.StatusBadRequest, "agent_id is required")
		return
	}

	sortDoc, err := parseSort(req.Sort)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}

	extra := bson.M{}
	if req.PreferDevice != "" {
		extra["config.device"] = req.PreferDevice
	}

	exp, err := h.Store.ClaimNext(r.Context(), dataset, req.AgentID, req.Device, sortDoc, extra)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, exp)
}

func (h *Handlers) AppendCheckpoint(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	name := chi.URLParam(r, "name")
	var req CheckpointRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	cp := Checkpoint{
		Step:     req.Step,
		ElapsedS: req.ElapsedS,
		Metrics:  req.Metrics,
	}
	if err := h.Store.AppendCheckpoint(r.Context(), dataset, name, cp); err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (h *Handlers) FinishExperiment(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	name := chi.URLParam(r, "name")
	var req FinishRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	exp, err := h.Store.Finish(r.Context(), dataset, name, req.FinalMetrics, req.DurationS, req.Checkpoints)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, exp)
}

func (h *Handlers) FailExperiment(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	name := chi.URLParam(r, "name")
	var req FailRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	exp, err := h.Store.Fail(r.Context(), dataset, name, req.Error)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, exp)
}

// flattenJSON walks a value and emits dot-keyed string entries.
//
//	{"loss": {"type":"BCE","alpha":0.25}, "lr": 0.01}
//	→ {"loss.type": "BCE", "loss.alpha": "0.25", "lr": "0.01"}
func flattenJSON(prefix string, v interface{}, out map[string]string) {
	switch t := v.(type) {
	case map[string]interface{}:
		for k, val := range t {
			key := k
			if prefix != "" {
				key = prefix + "." + k
			}
			flattenJSON(key, val, out)
		}
	case []interface{}:
		parts := make([]string, len(t))
		for i, e := range t {
			parts[i] = fmt.Sprintf("%v", e)
		}
		out[prefix] = strings.Join(parts, "|")
	case nil:
		out[prefix] = ""
	default:
		out[prefix] = fmt.Sprintf("%v", t)
	}
}

// ResultsCSV streams a CSV that matches the old `final_results.csv` layout:
// one row per experiment, all config and final-metric fields flattened.
//
// Query params (all optional):
//
//	status         default "done" (or "all" to include every status)
//	architecture   exact-match filter on the architecture field
//	agent_id       exact-match filter
//	device         exact-match filter
//	loss_type      filter on config.loss.type
//	limit, offset  pagination over the result set
func (h *Handlers) ResultsCSV(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	q := r.URL.Query()

	status := q.Get("status")
	if status == "" {
		status = "done"
	}
	filter := bson.M{}
	if status != "all" {
		filter["status"] = status
	}
	if v := q.Get("architecture"); v != "" {
		filter["architecture"] = v
	}
	if v := q.Get("agent_id"); v != "" {
		filter["agent_id"] = v
	}
	if v := q.Get("device"); v != "" {
		filter["device"] = v
	}
	if v := q.Get("loss_type"); v != "" {
		filter["config.loss.type"] = v
	}

	limit, _ := strconv.ParseInt(q.Get("limit"), 10, 64)
	offset, _ := strconv.ParseInt(q.Get("offset"), 10, 64)

	exps, err := h.Store.ListFiltered(r.Context(), dataset, filter, limit, offset)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	// First pass: discover all config + metric keys so the header is stable.
	configKeys := map[string]struct{}{}
	metricKeys := map[string]struct{}{}
	rows := make([]map[string]string, 0, len(exps))
	for _, e := range exps {
		row := map[string]string{
			"exp_name":     e.ExpName,
			"status":       string(e.Status),
			"architecture": e.Architecture,
			"dataset":      e.Dataset,
			"agent_id":     e.AgentID,
			"device":       e.Device,
			"duration_s":   fmt.Sprintf("%g", e.DurationS),
			"error":        e.Error,
		}
		if e.StartedAt != nil {
			row["started_at"] = e.StartedAt.UTC().Format("2006-01-02T15:04:05Z")
		}
		if e.FinishedAt != nil {
			row["finished_at"] = e.FinishedAt.UTC().Format("2006-01-02T15:04:05Z")
		}
		row["created_at"] = e.CreatedAt.UTC().Format("2006-01-02T15:04:05Z")

		flatCfg := map[string]string{}
		flattenJSON("", e.Config, flatCfg)
		for k, v := range flatCfg {
			configKeys[k] = struct{}{}
			row["cfg."+k] = v
		}
		for k, v := range e.FinalMetrics {
			metricKeys[k] = struct{}{}
			row["metric."+k] = fmt.Sprintf("%g", v)
		}
		rows = append(rows, row)
	}

	baseCols := []string{
		"exp_name", "status", "architecture", "dataset",
		"agent_id", "device",
		"started_at", "finished_at", "duration_s", "created_at", "error",
	}
	cfgCols := make([]string, 0, len(configKeys))
	for k := range configKeys {
		cfgCols = append(cfgCols, "cfg."+k)
	}
	sort.Strings(cfgCols)
	metricCols := make([]string, 0, len(metricKeys))
	for k := range metricKeys {
		metricCols = append(metricCols, "metric."+k)
	}
	sort.Strings(metricCols)

	header := append(append(baseCols, cfgCols...), metricCols...)

	filename := fmt.Sprintf("results_%s_%s.csv", dataset, status)
	w.Header().Set("Content-Type", "text/csv; charset=utf-8")
	w.Header().Set("Content-Disposition", `attachment; filename="`+filename+`"`)
	cw := csv.NewWriter(w)
	if err := cw.Write(header); err != nil {
		return
	}
	rec := make([]string, len(header))
	for _, row := range rows {
		for i, col := range header {
			rec[i] = row[col]
		}
		if err := cw.Write(rec); err != nil {
			return
		}
	}
	cw.Flush()
}

func (h *Handlers) ReleaseExperiment(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	name := chi.URLParam(r, "name")
	exp, err := h.Store.Release(r.Context(), dataset, name)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, exp)
}
