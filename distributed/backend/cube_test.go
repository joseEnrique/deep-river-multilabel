package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"
)

// Coverage for the read-only analysis surface (cube, summary, results.csv).
// These are the endpoints used to interrogate the grid after the fact.

func cubeRouter(s *Store) http.Handler {
	h := &Handlers{Store: s}
	r := chi.NewRouter()
	r.Route("/api/v1/datasets/{dataset}", func(r chi.Router) {
		r.Get("/stats", h.Stats)
		r.Get("/summary", h.Summary)
		r.Get("/results.csv", h.ResultsCSV)
		r.Route("/cube", func(r chi.Router) {
			r.Get("/metrics", h.CubeMetrics)
			r.Get("/params", h.CubeParams)
			r.Get("/params/values", h.CubeParamValues)
			r.Get("/top", h.CubeTop)
			r.Get("/groupby", h.CubeGroupBy)
			r.Get("/best-per", h.CubeBestPer)
			r.Get("/distribution", h.CubeDistribution)
		})
	})
	return r
}

func doneExp(name string, epochs, hidden int, arch string, f1 float64) *Experiment {
	return &Experiment{
		ExpName:      name,
		Architecture: arch,
		Status:       StatusDone,
		Config: map[string]interface{}{
			"epochs": epochs, "hidden_dim": hidden, "device": "cuda:0",
			"loss": map[string]interface{}{"type": "BCE"},
		},
		FinalMetrics: map[string]float64{"macro_f1": f1, "subset_acc": f1 + 10},
		DurationS:    float64(epochs) * 10,
	}
}

func seedCube(t *testing.T) (*Store, string, http.Handler) {
	t.Helper()
	s, ds := testStore(t)
	seed(t, s, ds,
		doneExp("a", 1, 32, "LSTM", 10),
		doneExp("b", 3, 64, "LSTM", 30),
		doneExp("c", 10, 128, "CNN", 50),
		doneExp("d", 20, 128, "CNN", 70),
		exp("pending", 1, 32, "cuda:0", StatusPending),
	)
	return s, ds, cubeRouter(s)
}

func getJSON(t *testing.T, srv http.Handler, path string) (int, map[string]interface{}) {
	t.Helper()
	req := httptest.NewRequest("GET", path, nil)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)
	var out map[string]interface{}
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return rec.Code, out
}

func TestCubeMetricsDiscoversMetricNames(t *testing.T) {
	_, ds, srv := seedCube(t)
	code, body := getJSON(t, srv, "/api/v1/datasets/"+ds+"/cube/metrics")
	if code != http.StatusOK {
		t.Fatalf("code = %d", code)
	}
	metrics, _ := body["metrics"].([]interface{})
	names := map[string]bool{}
	for _, m := range metrics {
		mm, _ := m.(map[string]interface{})
		names[mm["name"].(string)] = true
	}
	if !names["macro_f1"] || !names["subset_acc"] {
		t.Errorf("metrics = %v, want macro_f1 and subset_acc", names)
	}
}

func TestCubeParamsDiscoversConfigKeys(t *testing.T) {
	_, ds, srv := seedCube(t)
	code, body := getJSON(t, srv, "/api/v1/datasets/"+ds+"/cube/params")
	if code != http.StatusOK {
		t.Fatalf("code = %d", code)
	}
	params, _ := body["params"].([]interface{})
	keys := map[string]bool{}
	for _, p := range params {
		pm, _ := p.(map[string]interface{})
		keys[pm["key"].(string)] = true
	}
	if !keys["config.epochs"] || !keys["config.hidden_dim"] {
		t.Errorf("params = %v, want config.epochs and config.hidden_dim", keys)
	}
	if !keys["config.loss.type"] {
		t.Errorf("nested key config.loss.type not flattened: %v", keys)
	}
}

func TestCubeParamValuesAggregatesPerValue(t *testing.T) {
	_, ds, srv := seedCube(t)
	code, body := getJSON(t, srv,
		"/api/v1/datasets/"+ds+"/cube/params/values?key=config.epochs&metric=macro_f1")
	if code != http.StatusOK {
		t.Fatalf("code = %d", code)
	}
	values, _ := body["values"].([]interface{})
	if len(values) != 4 {
		t.Fatalf("values = %d, want 4 distinct epochs", len(values))
	}
	for _, v := range values {
		vm, _ := v.(map[string]interface{})
		if vm["value"] == float64(20) && vm["max"] != float64(70) {
			t.Errorf("epochs=20 max = %v, want 70", vm["max"])
		}
	}
}

func TestCubeParamValuesStatusFilter(t *testing.T) {
	_, ds, srv := seedCube(t)
	// Default status=done excludes the pending row.
	_, body := getJSON(t, srv, "/api/v1/datasets/"+ds+"/cube/params/values?key=config.epochs")
	values, _ := body["values"].([]interface{})
	total := 0.0
	for _, v := range values {
		vm, _ := v.(map[string]interface{})
		total += vm["count"].(float64)
	}
	if total != 4 {
		t.Errorf("done rows = %v, want 4", total)
	}

	// status=all includes it.
	_, body = getJSON(t, srv, "/api/v1/datasets/"+ds+"/cube/params/values?key=config.epochs&status=all")
	values, _ = body["values"].([]interface{})
	total = 0
	for _, v := range values {
		vm, _ := v.(map[string]interface{})
		total += vm["count"].(float64)
	}
	if total != 5 {
		t.Errorf("all rows = %v, want 5", total)
	}
}

func TestCubeTopRanksByMetric(t *testing.T) {
	_, ds, srv := seedCube(t)
	code, body := getJSON(t, srv,
		"/api/v1/datasets/"+ds+"/cube/top?metric=macro_f1&limit=2")
	if code != http.StatusOK {
		t.Fatalf("code = %d", code)
	}
	exps, _ := body["experiments"].([]interface{})
	if len(exps) != 2 {
		t.Fatalf("got %d, want 2", len(exps))
	}
	first, _ := exps[0].(map[string]interface{})
	if first["exp_name"] != "d" {
		t.Errorf("top1 = %v, want d (macro_f1=70)", first["exp_name"])
	}

	// Ascending order flips it.
	_, body = getJSON(t, srv,
		"/api/v1/datasets/"+ds+"/cube/top?metric=macro_f1&limit=1&order=asc")
	exps, _ = body["experiments"].([]interface{})
	first, _ = exps[0].(map[string]interface{})
	if first["exp_name"] != "a" {
		t.Errorf("bottom1 = %v, want a", first["exp_name"])
	}
}

func TestCubeTopRequiresMetric(t *testing.T) {
	_, ds, srv := seedCube(t)
	code, _ := getJSON(t, srv, "/api/v1/datasets/"+ds+"/cube/top")
	if code != http.StatusBadRequest {
		t.Errorf("code = %d, want 400 without metric", code)
	}
}

func TestCubeTopWhereFilter(t *testing.T) {
	_, ds, srv := seedCube(t)
	_, body := getJSON(t, srv,
		"/api/v1/datasets/"+ds+"/cube/top?metric=macro_f1&where=architecture=LSTM")
	exps, _ := body["experiments"].([]interface{})
	if len(exps) != 2 {
		t.Fatalf("LSTM rows = %d, want 2", len(exps))
	}
	for _, e := range exps {
		em, _ := e.(map[string]interface{})
		if em["architecture"] != "LSTM" {
			t.Errorf("where filter leaked a %v", em["architecture"])
		}
	}
}

func TestCubeGroupBy(t *testing.T) {
	_, ds, srv := seedCube(t)
	code, body := getJSON(t, srv,
		"/api/v1/datasets/"+ds+"/cube/groupby?by=architecture&metric=macro_f1&agg=max,mean,count")
	if code != http.StatusOK {
		t.Fatalf("code = %d, body=%v", code, body)
	}
	groups, _ := body["groups"].([]interface{})
	if len(groups) != 2 {
		t.Fatalf("groups = %d, want 2 (LSTM, CNN)", len(groups))
	}
	for _, g := range groups {
		gm, _ := g.(map[string]interface{})
		grp, _ := gm["group"].(map[string]interface{})
		met, _ := gm["metrics"].(map[string]interface{})
		if grp["architecture"] == "CNN" {
			if met["max"] != float64(70) || met["count"] != float64(2) {
				t.Errorf("CNN metrics = %v, want max 70 count 2", met)
			}
		}
	}
}

func TestCubeGroupByRequiresParams(t *testing.T) {
	_, ds, srv := seedCube(t)
	if code, _ := getJSON(t, srv, "/api/v1/datasets/"+ds+"/cube/groupby?metric=macro_f1"); code != http.StatusBadRequest {
		t.Errorf("missing by: code = %d, want 400", code)
	}
	if code, _ := getJSON(t, srv, "/api/v1/datasets/"+ds+"/cube/groupby?by=architecture"); code != http.StatusBadRequest {
		t.Errorf("missing metric: code = %d, want 400", code)
	}
}

func TestCubeBestPer(t *testing.T) {
	_, ds, srv := seedCube(t)
	code, body := getJSON(t, srv,
		"/api/v1/datasets/"+ds+"/cube/best-per?by=architecture&metric=macro_f1")
	if code != http.StatusOK {
		t.Fatalf("code = %d", code)
	}
	groups, _ := body["groups"].([]interface{})
	if len(groups) != 2 {
		t.Fatalf("groups = %d, want 2", len(groups))
	}
	for _, g := range groups {
		gm, _ := g.(map[string]interface{})
		best, _ := gm["best"].(map[string]interface{})
		if gm["value"] == "CNN" && best["exp_name"] != "d" {
			t.Errorf("best CNN = %v, want d", best["exp_name"])
		}
		if gm["value"] == "LSTM" && best["exp_name"] != "b" {
			t.Errorf("best LSTM = %v, want b", best["exp_name"])
		}
	}
}

func TestCubeDistribution(t *testing.T) {
	_, ds, srv := seedCube(t)
	code, body := getJSON(t, srv,
		"/api/v1/datasets/"+ds+"/cube/distribution?metric=macro_f1&bins=4")
	if code != http.StatusOK {
		t.Fatalf("code = %d", code)
	}
	if body["count"] != float64(4) {
		t.Errorf("count = %v, want 4", body["count"])
	}
	if body["min"] != float64(10) || body["max"] != float64(70) {
		t.Errorf("min/max = %v/%v, want 10/70", body["min"], body["max"])
	}
	bins, _ := body["bins"].([]interface{})
	if len(bins) != 4 {
		t.Errorf("bins = %d, want 4", len(bins))
	}
	sum := 0.0
	for _, b := range bins {
		bm, _ := b.(map[string]interface{})
		sum += bm["count"].(float64)
	}
	if sum != 4 {
		t.Errorf("bin counts sum to %v, want 4 (no row lost)", sum)
	}
}

func TestSummaryReportsEtaAndRunning(t *testing.T) {
	s, ds, srv := seedCube(t)
	// Put one into running so `running[]` is exercised.
	if _, err := s.Claim(context.Background(), ds, "pending", "ag", "cuda:0"); err != nil {
		t.Fatalf("claim for summary setup: %v", err)
	}

	code, body := getJSON(t, srv, "/api/v1/datasets/"+ds+"/summary")
	if code != http.StatusOK {
		t.Fatalf("code = %d", code)
	}
	if body["done_count"] != float64(4) {
		t.Errorf("done_count = %v, want 4", body["done_count"])
	}
	if body["avg_duration_s"] == nil || body["avg_duration_s"] == float64(0) {
		t.Errorf("avg_duration_s = %v, want > 0", body["avg_duration_s"])
	}
	if _, ok := body["eta_method"].(string); !ok {
		t.Error("eta_method missing")
	}
}

func TestResultsCSVExport(t *testing.T) {
	_, ds, srv := seedCube(t)
	req := httptest.NewRequest("GET", "/api/v1/datasets/"+ds+"/results.csv", nil)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("code = %d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "csv") {
		t.Errorf("content-type = %q, want csv", ct)
	}
	if cd := rec.Header().Get("Content-Disposition"); !strings.Contains(cd, "attachment") {
		t.Errorf("content-disposition = %q", cd)
	}

	rows, err := csv.NewReader(strings.NewReader(rec.Body.String())).ReadAll()
	if err != nil {
		t.Fatalf("not valid CSV: %v", err)
	}
	if len(rows) != 5 {
		t.Fatalf("rows = %d, want 5 (header + 4 done)", len(rows))
	}
	header := strings.Join(rows[0], ",")
	for _, want := range []string{"exp_name", "status", "cfg.epochs", "metric.macro_f1"} {
		if !strings.Contains(header, want) {
			t.Errorf("header missing %q: %s", want, header)
		}
	}
}

func TestResultsCSVStatusAllIncludesPending(t *testing.T) {
	_, ds, srv := seedCube(t)
	req := httptest.NewRequest("GET", "/api/v1/datasets/"+ds+"/results.csv?status=all", nil)
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)
	rows, err := csv.NewReader(strings.NewReader(rec.Body.String())).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 6 {
		t.Errorf("rows = %d, want 6 (header + 5)", len(rows))
	}
}

// The header is now discovered in a separate key-only pass instead of by
// buffering every document, so the risk the rewrite introduces is a header
// that misses a column some later row has — which would silently shift every
// value in that row. Seed documents with deliberately different key sets.
func TestResultsCSVHeaderCoversEveryRow(t *testing.T) {
	s, ds := testStore(t)
	plain := doneExp("a_plain", 1, 32, "LSTM", 10)
	rich := doneExp("b_rich", 3, 64, "CNN", 30)
	rich.Config["loss"] = map[string]interface{}{
		"type": "AdaptiveFocal", "base_alpha": 0.25, "decay": 0.99,
	}
	rich.Config["machine"] = 18
	rich.FinalMetrics["mean_gamma"] = 1.5
	seed(t, s, ds, plain, rich)

	req := httptest.NewRequest("GET", "/api/v1/datasets/"+ds+"/results.csv", nil)
	rec := httptest.NewRecorder()
	cubeRouter(s).ServeHTTP(rec, req)

	rows, err := csv.NewReader(strings.NewReader(rec.Body.String())).ReadAll()
	if err != nil {
		t.Fatalf("not valid CSV: %v", err)
	}
	if len(rows) != 3 {
		t.Fatalf("rows = %d, want 3 (header + 2)", len(rows))
	}
	col := map[string]int{}
	for i, name := range rows[0] {
		col[name] = i
	}
	for _, want := range []string{
		"cfg.loss.type", "cfg.loss.base_alpha", "cfg.loss.decay",
		"cfg.machine", "metric.mean_gamma",
	} {
		if _, ok := col[want]; !ok {
			t.Fatalf("header missing %q: %v", want, rows[0])
		}
	}
	// Rows come back sorted by _id, so a_plain is first.
	if got := rows[1][col["exp_name"]]; got != "a_plain" {
		t.Errorf("first row = %q, want a_plain (export sorts by _id)", got)
	}
	// The document without those keys must carry empty cells, not a shift.
	for _, absent := range []string{"cfg.loss.base_alpha", "cfg.machine", "metric.mean_gamma"} {
		if got := rows[1][col[absent]]; got != "" {
			t.Errorf("a_plain[%s] = %q, want empty", absent, got)
		}
	}
	if got := rows[2][col["cfg.loss.type"]]; got != "AdaptiveFocal" {
		t.Errorf("b_rich[cfg.loss.type] = %q", got)
	}
	if got := rows[2][col["cfg.machine"]]; got != "18" {
		t.Errorf("b_rich[cfg.machine] = %q", got)
	}
}

// checkpoints are projected out of the export: they are the biggest field on a
// finished run and the CSV never reads them. Decoding them anyway is a large
// part of what made the old export a memory event.
func TestResultsCSVIgnoresCheckpoints(t *testing.T) {
	s, ds := testStore(t)
	e := doneExp("with_cp", 1, 32, "LSTM", 10)
	for i := 0; i < 50; i++ {
		e.Checkpoints = append(e.Checkpoints, Checkpoint{
			Step: i * 1000, ElapsedS: float64(i), Metrics: map[string]float64{"loss": 0.5},
		})
	}
	seed(t, s, ds, e)

	req := httptest.NewRequest("GET", "/api/v1/datasets/"+ds+"/results.csv", nil)
	rec := httptest.NewRecorder()
	cubeRouter(s).ServeHTTP(rec, req)

	rows, err := csv.NewReader(strings.NewReader(rec.Body.String())).ReadAll()
	if err != nil {
		t.Fatalf("not valid CSV: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("rows = %d, want 2", len(rows))
	}
	for _, name := range rows[0] {
		if strings.Contains(name, "checkpoint") {
			t.Errorf("header leaks checkpoints: %q", name)
		}
	}
}

// Exports are capped: each one walks the whole collection, and an unbounded
// number of them in flight is how the server stops answering agents.
func TestResultsCSVRejectsConcurrentExports(t *testing.T) {
	s, ds := testStore(t)
	seed(t, s, ds, doneExp("a", 1, 32, "LSTM", 10))

	// Occupy every slot, as in-flight exports would.
	for i := 0; i < cap(csvExportSlots); i++ {
		csvExportSlots <- struct{}{}
	}
	defer func() {
		for i := 0; i < cap(csvExportSlots); i++ {
			<-csvExportSlots
		}
	}()

	req := httptest.NewRequest("GET", "/api/v1/datasets/"+ds+"/results.csv", nil)
	rec := httptest.NewRecorder()
	cubeRouter(s).ServeHTTP(rec, req)

	if rec.Code != http.StatusTooManyRequests {
		t.Errorf("code = %d, want 429", rec.Code)
	}
	if rec.Header().Get("Retry-After") == "" {
		t.Error("no Retry-After on the 429 — the client has nothing to back off on")
	}
}

func TestCubeInvalidDataset(t *testing.T) {
	s, _ := testStore(t)
	srv := cubeRouter(s)
	code, _ := getJSON(t, srv, "/api/v1/datasets/bad!name/cube/metrics")
	if code != http.StatusBadRequest && code != http.StatusNotFound {
		t.Errorf("code = %d, want 400/404", code)
	}
}
