package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

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
		ArchRank:     req.ArchRank,
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
			ArchRank:     e.ArchRank,
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

// Only a couple of exports may be in flight at once. Each one walks the
// whole collection; letting an unbounded number of them pile up is how the
// server ends up thrashing instead of answering the agents that are trying
// to claim work.
var csvExportSlots = make(chan struct{}, 2)

const (
	// csvWriteWindow is how long one flush may take before the connection is
	// dropped. The server's global WriteTimeout is 60s counted from the end
	// of the request read, which would cut a multi-minute export in half;
	// this replaces it with a deadline that rolls forward on every flush, so
	// a slow-but-alive client keeps going and a dead one is still dropped.
	csvWriteWindow = 2 * time.Minute
	// Rows written per flush. Bounds how much sits in the bufio writer and
	// keeps bytes arriving steadily, which is what tells the client (and any
	// proxy in between) that the export is alive.
	csvFlushEvery = 200
)

var csvBaseCols = []string{
	"exp_name", "status", "architecture", "dataset",
	"agent_id", "device",
	"started_at", "finished_at", "duration_s", "created_at", "error",
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
//
// It streams. The previous version decoded every matching document into a
// slice, built a second full copy of it as `[]map[string]string`, and only
// then wrote the first byte — so a 60k-row export sent nothing for as long
// as it lived and died of memory before it sent anything at all. Here the
// header is discovered in one key-only pass and the rows go out as the
// cursor yields them, so memory is flat in the number of matches and the
// client sees the header within seconds.
func (h *Handlers) ResultsCSV(w http.ResponseWriter, r *http.Request) {
	select {
	case csvExportSlots <- struct{}{}:
		defer func() { <-csvExportSlots }()
	default:
		w.Header().Set("Retry-After", "60")
		writeErr(w, http.StatusTooManyRequests,
			"another CSV export is already running; retry shortly")
		return
	}

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

	// The route is exempt from the 60s request timeout (see main.go), so the
	// export carries its own deadline. A context deadline takes precedence
	// over the client-level opTimeout, so this is what bounds the cursor.
	ctx, cancel := context.WithTimeout(r.Context(), exportTimeout)
	defer cancel()

	header, err := h.csvHeader(ctx, dataset, filter, limit, offset)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	// `checkpoints` is never used by the export and is the largest field on a
	// finished run — dropping it server-side is most of the I/O saved.
	cur, err := h.Store.StreamFiltered(ctx, dataset, filter,
		bson.M{"checkpoints": 0}, limit, offset)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer cur.Close(ctx)

	filename := fmt.Sprintf("results_%s_%s.csv", dataset, status)
	w.Header().Set("Content-Type", "text/csv; charset=utf-8")
	w.Header().Set("Content-Disposition", `attachment; filename="`+filename+`"`)
	// Tell any reverse proxy not to buffer the body; buffering would undo
	// the streaming and reproduce the "nothing for minutes" symptom.
	w.Header().Set("X-Accel-Buffering", "no")

	rc := http.NewResponseController(w)
	// httptest.ResponseRecorder does not support deadlines. Ignoring the
	// error keeps the handler usable under test.
	extend := func() { _ = rc.SetWriteDeadline(time.Now().Add(csvWriteWindow)) }
	extend()

	cw := csv.NewWriter(w)
	if err := cw.Write(header); err != nil {
		return
	}
	cw.Flush()
	_ = rc.Flush()

	rec := make([]string, len(header))
	row := make(map[string]string, len(header))
	n := 0
	for cur.Next(ctx) {
		var e Experiment
		if err := cur.Decode(&e); err != nil {
			log.Printf("results.csv %s: decode: %v", dataset, err)
			// The header is already on the wire, so there is no status code
			// left to set. Abort the connection rather than closing the
			// chunked body cleanly, which the client would read as a
			// complete file.
			panic(http.ErrAbortHandler)
		}
		clear(row)
		csvRow(&e, row)
		for i, col := range header {
			rec[i] = row[col]
		}
		if err := cw.Write(rec); err != nil {
			return
		}
		n++
		if n%csvFlushEvery == 0 {
			cw.Flush()
			if cw.Error() != nil {
				return // client hung up
			}
			_ = rc.Flush()
			extend()
		}
	}
	if err := cur.Err(); err != nil {
		log.Printf("results.csv %s: cursor: %v", dataset, err)
		panic(http.ErrAbortHandler)
	}
	cw.Flush()
	_ = rc.Flush()
}

// csvHeader walks the matches once keeping only key *names*, so the header can
// be fixed before the first row is written without holding any documents.
// Memory here is the number of distinct columns, not the number of rows.
func (h *Handlers) csvHeader(ctx context.Context, dataset string, filter bson.M, limit, offset int64) ([]string, error) {
	cur, err := h.Store.StreamFiltered(ctx, dataset, filter,
		bson.M{"config": 1, "final_metrics": 1}, limit, offset)
	if err != nil {
		return nil, err
	}
	defer cur.Close(ctx)

	configKeys := map[string]struct{}{}
	metricKeys := map[string]struct{}{}
	flat := map[string]string{}
	for cur.Next(ctx) {
		var e Experiment
		if err := cur.Decode(&e); err != nil {
			return nil, err
		}
		clear(flat)
		flattenJSON("", e.Config, flat)
		for k := range flat {
			configKeys[k] = struct{}{}
		}
		for k := range e.FinalMetrics {
			metricKeys[k] = struct{}{}
		}
	}
	if err := cur.Err(); err != nil {
		return nil, err
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

	header := make([]string, 0, len(csvBaseCols)+len(cfgCols)+len(metricCols))
	header = append(header, csvBaseCols...)
	header = append(header, cfgCols...)
	header = append(header, metricCols...)
	return header, nil
}

// csvRow fills `row` (already cleared) with one experiment's flattened fields.
func csvRow(e *Experiment, row map[string]string) {
	row["exp_name"] = e.ExpName
	row["status"] = string(e.Status)
	row["architecture"] = e.Architecture
	row["dataset"] = e.Dataset
	row["agent_id"] = e.AgentID
	row["device"] = e.Device
	row["duration_s"] = fmt.Sprintf("%g", e.DurationS)
	row["error"] = e.Error
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
		row["cfg."+k] = v
	}
	for k, v := range e.FinalMetrics {
		row["metric."+k] = fmt.Sprintf("%g", v)
	}
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
