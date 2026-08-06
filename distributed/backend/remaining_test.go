package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/go-chi/chi/v5"
	"go.mongodb.org/mongo-driver/bson"
)

// Coverage for the endpoints and helpers not exercised by the lifecycle or
// cube suites: config loading, auth middleware, health, dataset discovery,
// list/replace/patch, and the percentile helpers.

func TestLoadConfigDefaultsAndEnv(t *testing.T) {
	t.Setenv("BACKEND_ADDR", "")
	t.Setenv("MONGO_URI", "")
	t.Setenv("DB_PREFIX", "")
	t.Setenv("API_KEY", "")
	c := LoadConfig()
	if c.Addr != ":8080" || c.MongoURI != "mongodb://localhost:27017" ||
		c.DBPrefix != "experiments_" || c.APIKey != "" {
		t.Errorf("defaults wrong: %+v", c)
	}

	t.Setenv("BACKEND_ADDR", ":9999")
	t.Setenv("DB_PREFIX", "x_")
	t.Setenv("API_KEY", "secret")
	c = LoadConfig()
	if c.Addr != ":9999" || c.DBPrefix != "x_" || c.APIKey != "secret" {
		t.Errorf("env override wrong: %+v", c)
	}
}

func TestGetEnvFallback(t *testing.T) {
	t.Setenv("SOME_UNSET_KEY_XYZ", "")
	if got := getEnv("SOME_UNSET_KEY_XYZ", "fb"); got != "fb" {
		t.Errorf("got %q, want fb", got)
	}
	t.Setenv("SOME_SET_KEY_XYZ", "v")
	if got := getEnv("SOME_SET_KEY_XYZ", "fb"); got != "v" {
		t.Errorf("got %q, want v", got)
	}
}

func TestAPIKeyAuth(t *testing.T) {
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	// No key configured → everything passes.
	open := APIKeyAuth("")(next)
	rec := httptest.NewRecorder()
	open.ServeHTTP(rec, httptest.NewRequest("GET", "/api/v1/datasets", nil))
	if rec.Code != http.StatusOK {
		t.Errorf("no-auth mode: code = %d, want 200", rec.Code)
	}

	guarded := APIKeyAuth("s3cret")(next)

	// Health must stay reachable without a key (it is the liveness probe).
	rec = httptest.NewRecorder()
	guarded.ServeHTTP(rec, httptest.NewRequest("GET", "/api/v1/health", nil))
	if rec.Code != http.StatusOK {
		t.Errorf("health without key: code = %d, want 200", rec.Code)
	}

	// Missing key → rejected.
	rec = httptest.NewRecorder()
	guarded.ServeHTTP(rec, httptest.NewRequest("GET", "/api/v1/datasets", nil))
	if rec.Code != http.StatusUnauthorized && rec.Code != http.StatusForbidden {
		t.Errorf("missing key: code = %d, want 401/403", rec.Code)
	}

	// X-API-Key header.
	req := httptest.NewRequest("GET", "/api/v1/datasets", nil)
	req.Header.Set("X-API-Key", "s3cret")
	rec = httptest.NewRecorder()
	guarded.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("X-API-Key: code = %d, want 200", rec.Code)
	}

	// Bearer token.
	req = httptest.NewRequest("GET", "/api/v1/datasets", nil)
	req.Header.Set("Authorization", "Bearer s3cret")
	rec = httptest.NewRecorder()
	guarded.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("Bearer: code = %d, want 200", rec.Code)
	}

	// Wrong key.
	req = httptest.NewRequest("GET", "/api/v1/datasets", nil)
	req.Header.Set("X-API-Key", "wrong")
	rec = httptest.NewRecorder()
	guarded.ServeHTTP(rec, req)
	if rec.Code == http.StatusOK {
		t.Error("wrong key was accepted")
	}
}

func TestHealthAndListDatasets(t *testing.T) {
	s, ds := testStore(t)
	seed(t, s, ds, exp("x", 1, 32, "cuda:0", StatusPending))

	h := &Handlers{Store: s}
	r := chi.NewRouter()
	r.Get("/api/v1/health", h.Health)
	r.Get("/api/v1/datasets", h.ListDatasets)

	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, httptest.NewRequest("GET", "/api/v1/health", nil))
	if rec.Code != http.StatusOK {
		t.Errorf("health: code = %d", rec.Code)
	}
	var hb map[string]interface{}
	_ = json.Unmarshal(rec.Body.Bytes(), &hb)
	if hb["status"] != "ok" {
		t.Errorf("health status = %v, want ok", hb["status"])
	}

	rec = httptest.NewRecorder()
	r.ServeHTTP(rec, httptest.NewRequest("GET", "/api/v1/datasets", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("datasets: code = %d", rec.Code)
	}
	var db map[string]interface{}
	_ = json.Unmarshal(rec.Body.Bytes(), &db)
	names, _ := db["datasets"].([]interface{})
	found := false
	for _, n := range names {
		if n == ds {
			found = true
		}
	}
	if !found {
		t.Errorf("datasets = %v, want to contain %q", names, ds)
	}
}

func TestListExperimentsHandler(t *testing.T) {
	s, ds := testStore(t)
	h := &Handlers{Store: s}
	r := chi.NewRouter()
	r.Route("/api/v1/datasets/{dataset}", func(r chi.Router) {
		r.Get("/experiments", h.ListExperiments)
	})
	for i := 0; i < 4; i++ {
		seed(t, s, ds, exp(fmt.Sprintf("p%d", i), 1, 32, "cuda:0", StatusPending))
	}
	seed(t, s, ds, exp("done1", 1, 32, "cuda:0", StatusDone))

	base := "/api/v1/datasets/" + ds + "/experiments"

	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, httptest.NewRequest("GET", base, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("code = %d", rec.Code)
	}
	var body map[string]interface{}
	_ = json.Unmarshal(rec.Body.Bytes(), &body)
	if body["count"] != float64(5) {
		t.Errorf("count = %v, want 5", body["count"])
	}

	rec = httptest.NewRecorder()
	r.ServeHTTP(rec, httptest.NewRequest("GET", base+"?status=pending", nil))
	_ = json.Unmarshal(rec.Body.Bytes(), &body)
	if body["count"] != float64(4) {
		t.Errorf("pending count = %v, want 4", body["count"])
	}

	rec = httptest.NewRecorder()
	r.ServeHTTP(rec, httptest.NewRequest("GET", base+"?status=pending&limit=2&offset=1", nil))
	_ = json.Unmarshal(rec.Body.Bytes(), &body)
	if body["count"] != float64(2) {
		t.Errorf("paginated count = %v, want 2", body["count"])
	}
}

func TestReplaceAndPatch(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	h := &Handlers{Store: s}
	r := chi.NewRouter()
	r.Route("/api/v1/datasets/{dataset}/experiments/{name}", func(r chi.Router) {
		r.Put("/", h.ReplaceExperiment)
		r.Patch("/", h.PatchExperiment)
	})
	seed(t, s, ds, exp("e1", 1, 32, "cuda:0", StatusPending))
	base := "/api/v1/datasets/" + ds + "/experiments/e1"

	// PATCH updates a subset of fields.
	patch := map[string]interface{}{"architecture": "CNN"}
	buf := &bytes.Buffer{}
	_ = json.NewEncoder(buf).Encode(patch)
	req := httptest.NewRequest("PATCH", base, buf)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("patch: code = %d body=%s", rec.Code, rec.Body.String())
	}
	got, _ := s.Get(ctx, ds, "e1")
	if got.Architecture != "CNN" {
		t.Errorf("architecture = %q, want CNN", got.Architecture)
	}
	if got.Status != StatusPending {
		t.Errorf("patch clobbered status: %q", got.Status)
	}

	// PUT replaces the whole document.
	repl := Experiment{
		ExpName: "e1", Architecture: "MLP", Status: StatusPending,
		Config: map[string]interface{}{"epochs": 7},
	}
	buf = &bytes.Buffer{}
	_ = json.NewEncoder(buf).Encode(repl)
	req = httptest.NewRequest("PUT", base, buf)
	rec = httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("put: code = %d body=%s", rec.Code, rec.Body.String())
	}
	got, _ = s.Get(ctx, ds, "e1")
	if got.Architecture != "MLP" {
		t.Errorf("architecture = %q, want MLP", got.Architecture)
	}

	// A done experiment must not be overwritten by PUT.
	if _, err := s.Claim(ctx, ds, "e1", "ag", "cuda:0"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Finish(ctx, ds, "e1", map[string]float64{"macro_f1": 1}, 1, nil); err != nil {
		t.Fatal(err)
	}
	buf = &bytes.Buffer{}
	_ = json.NewEncoder(buf).Encode(repl)
	req = httptest.NewRequest("PUT", base, buf)
	rec = httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	if rec.Code != http.StatusConflict {
		t.Errorf("put on done: code = %d, want 409", rec.Code)
	}
}

func TestPatchStripsProtectedKeys(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	seed(t, s, ds, exp("e1", 1, 32, "cuda:0", StatusPending))
	before, _ := s.Get(ctx, ds, "e1")

	_, err := s.Patch(ctx, ds, "e1", map[string]interface{}{
		"_id":        "hacked",
		"exp_name":   "hacked",
		"dataset":    "other",
		"created_at": "1999-01-01T00:00:00Z",
		"error":      "kept",
	})
	if err != nil {
		t.Fatal(err)
	}
	after, err := s.Get(ctx, ds, "e1")
	if err != nil {
		t.Fatalf("experiment vanished — _id was patched: %v", err)
	}
	if after.ExpName != "e1" || after.Dataset != ds {
		t.Errorf("protected keys were not stripped: %+v", after)
	}
	if !after.CreatedAt.Equal(before.CreatedAt) {
		t.Errorf("created_at was overwritten")
	}
	if after.Error != "kept" {
		t.Errorf("normal field not applied: %q", after.Error)
	}
	if !after.UpdatedAt.After(before.UpdatedAt) && !after.UpdatedAt.Equal(before.UpdatedAt) {
		t.Errorf("updated_at went backwards")
	}
}

func TestPercentileHelpers(t *testing.T) {
	vals := bson.A{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
	for _, c := range []struct{ p, want float64 }{
		{0.5, 5}, {0.9, 9}, {1.0, 10},
	} {
		got := percentile(vals, c.p)
		if got < c.want-1 || got > c.want+1 {
			t.Errorf("percentile(%v) = %v, want ~%v", c.p, got, c.want)
		}
	}
	if got := percentile(bson.A{}, 0.5); got != 0 {
		t.Errorf("percentile(empty) = %v, want 0", got)
	}
	if got := percentile(bson.A{42.0}, 0.9); got != 42 {
		t.Errorf("percentile(single) = %v, want 42", got)
	}
	// Non-numeric entries must be skipped, not crash or poison the result.
	if got := percentile(bson.A{"x", nil, 5.0}, 0.5); got != 5 {
		t.Errorf("percentile(mixed) = %v, want 5", got)
	}
	if got := percentile(bson.A{"x", nil}, 0.5); got != 0 {
		t.Errorf("percentile(no numbers) = %v, want 0", got)
	}
	// Integer-typed values (what Mongo actually returns) must be handled.
	if got := percentile(bson.A{int32(10), int64(20), 30.0}, 1.0); got != 30 {
		t.Errorf("percentile(int types) = %v, want 30", got)
	}
}

func TestPercentileFromName(t *testing.T) {
	for name, want := range map[string]float64{
		"p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99,
	} {
		got := percentileFromName(name)
		if got < want-0.001 || got > want+0.001 {
			t.Errorf("percentileFromName(%q) = %v, want %v", name, got, want)
		}
	}
	// Unknown names fall back to the median rather than blowing up.
	for _, bad := range []string{"max", "mean", "p", "px", ""} {
		if got := percentileFromName(bad); got != 0.5 {
			t.Errorf("percentileFromName(%q) = %v, want 0.5 fallback", bad, got)
		}
	}
}

func TestNewMongoStoreRejectsBadURI(t *testing.T) {
	ctx := context.Background()
	if _, err := NewMongoStore(ctx, "not-a-mongo-uri", "p_"); err == nil {
		t.Error("expected an error for a malformed URI")
	}
}
