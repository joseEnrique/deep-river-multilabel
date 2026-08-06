package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"
	"go.mongodb.org/mongo-driver/bson"
)

// Full experiment lifecycle over the Store: the path every agent walks on
// every experiment. Covers create → claim → checkpoint → finish, plus the
// failure and release branches.

func TestCreateRejectsDuplicates(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	e := exp("dup", 1, 32, "cuda:0", StatusPending)
	if err := s.Create(ctx, ds, e); err != nil {
		t.Fatal(err)
	}
	err := s.Create(ctx, ds, exp("dup", 1, 32, "cuda:0", StatusPending))
	if err != ErrAlreadyExists {
		t.Fatalf("err = %v, want ErrAlreadyExists", err)
	}
}

func TestCreateDefaultsStatusToPending(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	e := &Experiment{ExpName: "nostatus", Config: map[string]interface{}{"epochs": 1}}
	if err := s.Create(ctx, ds, e); err != nil {
		t.Fatal(err)
	}
	got, err := s.Get(ctx, ds, "nostatus")
	if err != nil {
		t.Fatal(err)
	}
	if got.Status != StatusPending {
		t.Errorf("status = %q, want pending", got.Status)
	}
	if got.Dataset != ds {
		t.Errorf("dataset = %q, want %q", got.Dataset, ds)
	}
	if got.CreatedAt.IsZero() || got.UpdatedAt.IsZero() {
		t.Error("created_at/updated_at not stamped")
	}
}

func TestGetMissingReturnsNotFound(t *testing.T) {
	s, ds := testStore(t)
	if _, err := s.Get(context.Background(), ds, "nope"); err != ErrNotFound {
		t.Fatalf("err = %v, want ErrNotFound", err)
	}
}

func TestFullLifecycle(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	seed(t, s, ds, exp("e1", 1, 32, "cuda:0", StatusPending))

	claimed, err := s.Claim(ctx, ds, "e1", "ag", "cuda:0")
	if err != nil {
		t.Fatal(err)
	}
	if claimed.Status != StatusRunning {
		t.Fatalf("status = %s, want running", claimed.Status)
	}

	// Re-claiming a running experiment must fail: that is what stops two
	// agents running the same thing.
	if _, err := s.Claim(ctx, ds, "e1", "other", "cuda:1"); err != ErrInvalidState {
		t.Errorf("second claim err = %v, want ErrInvalidState", err)
	}

	if err := s.AppendCheckpoint(ctx, ds, "e1", Checkpoint{
		Step: 1000, ElapsedS: 12.3, Metrics: map[string]float64{"macro_f1": 0.4},
	}); err != nil {
		t.Fatal(err)
	}
	if err := s.AppendCheckpoint(ctx, ds, "e1", Checkpoint{
		Step: 2000, ElapsedS: 24.6, Metrics: map[string]float64{"macro_f1": 0.5},
	}); err != nil {
		t.Fatal(err)
	}

	got, _ := s.Get(ctx, ds, "e1")
	if len(got.Checkpoints) != 2 {
		t.Fatalf("checkpoints = %d, want 2 (append must not overwrite)", len(got.Checkpoints))
	}

	final := map[string]float64{"macro_f1": 0.78, "subset_acc": 0.91}
	if _, err := s.Finish(ctx, ds, "e1", final, 123.4, nil); err != nil {
		t.Fatal(err)
	}
	got, _ = s.Get(ctx, ds, "e1")
	if got.Status != StatusDone {
		t.Errorf("status = %s, want done", got.Status)
	}
	if got.FinalMetrics["macro_f1"] != 0.78 || got.DurationS != 123.4 {
		t.Errorf("final metrics/duration not stored: %v %v", got.FinalMetrics, got.DurationS)
	}
	if got.FinishedAt == nil {
		t.Error("finished_at not set")
	}

	// Finishing twice must not lower or overwrite results.
	if _, err := s.Finish(ctx, ds, "e1", map[string]float64{"macro_f1": 0.1}, 1, nil); err != ErrInvalidState {
		t.Errorf("second finish err = %v, want ErrInvalidState", err)
	}
	got, _ = s.Get(ctx, ds, "e1")
	if got.FinalMetrics["macro_f1"] != 0.78 {
		t.Errorf("metrics overwritten by second finish: %v", got.FinalMetrics)
	}

	// A done experiment is not claimable.
	if _, err := s.Claim(ctx, ds, "e1", "ag", "cuda:0"); err != ErrInvalidState {
		t.Errorf("claim on done err = %v, want ErrInvalidState", err)
	}
}

func TestFailThenReclaim(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	seed(t, s, ds, exp("e1", 1, 32, "cuda:0", StatusPending))

	if _, err := s.Claim(ctx, ds, "e1", "ag", "cuda:0"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Fail(ctx, ds, "e1", "boom"); err != nil {
		t.Fatal(err)
	}
	got, _ := s.Get(ctx, ds, "e1")
	if got.Status != StatusFailed || got.Error != "boom" {
		t.Fatalf("status=%s error=%q, want failed/boom", got.Status, got.Error)
	}

	// A failed experiment is retryable, and the claim clears the error.
	again, err := s.Claim(ctx, ds, "e1", "ag2", "cuda:1")
	if err != nil {
		t.Fatalf("failed must be re-claimable: %v", err)
	}
	if again.Error != "" {
		t.Errorf("error = %q, want cleared on re-claim", again.Error)
	}
}

func TestFailTruncatesLongError(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	seed(t, s, ds, exp("e1", 1, 32, "cuda:0", StatusPending))
	_, _ = s.Claim(ctx, ds, "e1", "ag", "cuda:0")

	long := strings.Repeat("x", 10000)
	if _, err := s.Fail(ctx, ds, "e1", long); err != nil {
		t.Fatal(err)
	}
	got, _ := s.Get(ctx, ds, "e1")
	if len(got.Error) > 4096 {
		t.Errorf("error length = %d, want truncated", len(got.Error))
	}
}

func TestReleaseReturnsToPending(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	seed(t, s, ds, exp("e1", 1, 32, "cuda:0", StatusPending))
	_, _ = s.Claim(ctx, ds, "e1", "ag", "cuda:0")

	if _, err := s.Release(ctx, ds, "e1"); err != nil {
		t.Fatal(err)
	}
	got, _ := s.Get(ctx, ds, "e1")
	if got.Status != StatusPending {
		t.Fatalf("status = %s, want pending", got.Status)
	}
	if got.AgentID != "" || got.StartedAt != nil {
		t.Errorf("agent_id/started_at not cleared: %q %v", got.AgentID, got.StartedAt)
	}

	// Releasing something that is not running is a state error.
	if _, err := s.Release(ctx, ds, "e1"); err != ErrInvalidState {
		t.Errorf("release on pending err = %v, want ErrInvalidState", err)
	}
}

// This is the exact path the agent takes when a claimed experiment does not
// fit in the free slots: claim-next → release → someone else claims it.
func TestClaimNextThenReleaseIsReclaimable(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	seed(t, s, ds, exp("big", 1, 512, "cuda:0", StatusPending))
	sortDoc, _ := parseSort("config.epochs:asc")

	got, err := s.ClaimNext(ctx, ds, "ag1", "cuda:0", sortDoc, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Release(ctx, ds, got.ExpName); err != nil {
		t.Fatalf("release after claim-next: %v", err)
	}
	// Back in the queue for anybody, including another agent.
	again, err := s.ClaimNext(ctx, ds, "ag2", "cuda:1", sortDoc, nil)
	if err != nil {
		t.Fatalf("released experiment must be claimable again: %v", err)
	}
	if again.ExpName != "big" || again.AgentID != "ag2" {
		t.Errorf("got %s/%s, want big/ag2", again.ExpName, again.AgentID)
	}
}

func TestBulkCreateSkipsDuplicates(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	batch := []*Experiment{
		exp("a", 1, 32, "cuda:0", StatusPending),
		exp("b", 1, 32, "cuda:0", StatusPending),
	}
	ins, skip, err := s.BulkCreate(ctx, ds, batch)
	if err != nil {
		t.Fatal(err)
	}
	if ins != 2 || skip != 0 {
		t.Fatalf("inserted=%d skipped=%d, want 2/0", ins, skip)
	}

	// Re-registering the same grid must be idempotent, not an error.
	batch2 := []*Experiment{
		exp("b", 1, 32, "cuda:0", StatusPending),
		exp("c", 1, 32, "cuda:0", StatusPending),
	}
	ins, skip, err = s.BulkCreate(ctx, ds, batch2)
	if err != nil {
		t.Fatalf("re-register must not fail: %v", err)
	}
	if ins != 1 || skip != 1 {
		t.Errorf("inserted=%d skipped=%d, want 1/1", ins, skip)
	}
}

func TestStatsCountsByStatus(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	seed(t, s, ds,
		exp("p1", 1, 32, "cuda:0", StatusPending),
		exp("p2", 1, 32, "cuda:0", StatusPending),
		exp("d1", 1, 32, "cuda:0", StatusDone),
		exp("f1", 1, 32, "cuda:0", StatusFailed),
	)
	st, err := s.Stats(ctx, ds)
	if err != nil {
		t.Fatal(err)
	}
	if st.Total != 4 {
		t.Errorf("total = %d, want 4", st.Total)
	}
	if st.Counts["pending"] != 2 || st.Counts["done"] != 1 || st.Counts["failed"] != 1 {
		t.Errorf("counts = %v", st.Counts)
	}
}

func TestListFiltersByStatusAndPaginates(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	for i := 0; i < 5; i++ {
		seed(t, s, ds, exp(fmt.Sprintf("p%d", i), 1, 32, "cuda:0", StatusPending))
	}
	seed(t, s, ds, exp("done", 1, 32, "cuda:0", StatusDone))

	all, err := s.List(ctx, ds, string(StatusPending), 100, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 5 {
		t.Fatalf("pending = %d, want 5", len(all))
	}

	page1, _ := s.List(ctx, ds, string(StatusPending), 2, 0)
	page2, _ := s.List(ctx, ds, string(StatusPending), 2, 2)
	if len(page1) != 2 || len(page2) != 2 {
		t.Fatalf("pagination broken: %d %d", len(page1), len(page2))
	}
	if page1[0].ExpName == page2[0].ExpName {
		t.Error("offset did not advance the page")
	}
}

func TestDeleteRemoves(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	seed(t, s, ds, exp("gone", 1, 32, "cuda:0", StatusPending))
	if err := s.Delete(ctx, ds, "gone"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Get(ctx, ds, "gone"); err != ErrNotFound {
		t.Errorf("err = %v, want ErrNotFound after delete", err)
	}
	if err := s.Delete(ctx, ds, "gone"); err != ErrNotFound {
		t.Errorf("double delete err = %v, want ErrNotFound", err)
	}
}

func TestInvalidDatasetNameRejected(t *testing.T) {
	s, _ := testStore(t)
	ctx := context.Background()
	for _, bad := range []string{"", "with space", "semi;colon", strings.Repeat("a", 65)} {
		if _, err := s.Stats(ctx, bad); err != ErrInvalidDataset {
			t.Errorf("dataset %q: err = %v, want ErrInvalidDataset", bad, err)
		}
	}
}

// ── Handler-level coverage of the agent's routes ─────────────────────────────

func fullRouter(s *Store) http.Handler {
	h := &Handlers{Store: s}
	r := chi.NewRouter()
	r.Route("/api/v1/datasets/{dataset}", func(r chi.Router) {
		r.Get("/stats", h.Stats)
		r.Get("/experiments", h.ListExperiments)
		r.Post("/experiments", h.CreateExperiment)
		r.Post("/experiments/bulk", h.BulkCreate)
		r.Post("/claim-next", h.ClaimNextExperiment)
		r.Route("/experiments/{name}", func(r chi.Router) {
			r.Get("/", h.GetExperiment)
			r.Delete("/", h.DeleteExperiment)
			r.Post("/claim", h.ClaimExperiment)
			r.Post("/checkpoints", h.AppendCheckpoint)
			r.Post("/finish", h.FinishExperiment)
			r.Post("/fail", h.FailExperiment)
			r.Post("/release", h.ReleaseExperiment)
		})
	})
	return r
}

func do(t *testing.T, srv http.Handler, method, path string, body interface{}) (int, map[string]interface{}) {
	t.Helper()
	var rdr *bytes.Buffer = &bytes.Buffer{}
	if body != nil {
		if err := json.NewEncoder(rdr).Encode(body); err != nil {
			t.Fatal(err)
		}
	}
	req := httptest.NewRequest(method, path, rdr)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)
	var out map[string]interface{}
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return rec.Code, out
}

func TestHandlerLifecycleOverHTTP(t *testing.T) {
	s, ds := testStore(t)
	srv := fullRouter(s)
	base := "/api/v1/datasets/" + ds

	code, _ := do(t, srv, "POST", base+"/experiments", CreateExperimentRequest{
		ExpName: "e1", Architecture: "LSTM",
		Config: map[string]interface{}{"epochs": 1, "hidden_dim": 32},
	})
	if code != http.StatusCreated {
		t.Fatalf("create: %d, want 201", code)
	}

	// Duplicate → 409.
	code, _ = do(t, srv, "POST", base+"/experiments", CreateExperimentRequest{
		ExpName: "e1", Config: map[string]interface{}{"epochs": 1},
	})
	if code != http.StatusConflict {
		t.Errorf("duplicate create: %d, want 409", code)
	}

	code, body := do(t, srv, "POST", base+"/claim-next",
		ClaimNextRequest{AgentID: "ag", Device: "cuda:0", Sort: "config.epochs:asc"})
	if code != http.StatusOK || body["exp_name"] != "e1" {
		t.Fatalf("claim-next: %d %v", code, body)
	}

	code, _ = do(t, srv, "POST", base+"/experiments/e1/checkpoints",
		CheckpointRequest{Step: 1000, ElapsedS: 5, Metrics: map[string]float64{"macro_f1": 0.3}})
	if code != http.StatusOK {
		t.Errorf("checkpoint: %d, want 200", code)
	}

	code, _ = do(t, srv, "POST", base+"/experiments/e1/finish", FinishRequest{
		FinalMetrics: map[string]float64{"macro_f1": 0.7}, DurationS: 42,
	})
	if code != http.StatusOK {
		t.Errorf("finish: %d, want 200", code)
	}

	code, body = do(t, srv, "GET", base+"/experiments/e1", nil)
	if code != http.StatusOK || body["status"] != "done" {
		t.Errorf("get: %d status=%v", code, body["status"])
	}

	code, body = do(t, srv, "GET", base+"/stats", nil)
	if code != http.StatusOK {
		t.Fatalf("stats: %d", code)
	}
	counts, _ := body["counts"].(map[string]interface{})
	if counts["done"] != float64(1) {
		t.Errorf("stats counts = %v", counts)
	}
}

func TestHandlerReleaseAndFailOverHTTP(t *testing.T) {
	s, ds := testStore(t)
	srv := fullRouter(s)
	base := "/api/v1/datasets/" + ds
	seed(t, s, ds, exp("e1", 1, 32, "cuda:0", StatusPending))

	code, _ := do(t, srv, "POST", base+"/experiments/e1/release", nil)
	if code != http.StatusConflict {
		t.Errorf("release on pending: %d, want 409", code)
	}

	code, _ = do(t, srv, "POST", base+"/experiments/e1/claim", ClaimRequest{AgentID: "ag", Device: "cuda:0"})
	if code != http.StatusOK {
		t.Fatalf("claim: %d", code)
	}
	code, _ = do(t, srv, "POST", base+"/experiments/e1/release", nil)
	if code != http.StatusOK {
		t.Errorf("release on running: %d, want 200", code)
	}

	_, _ = do(t, srv, "POST", base+"/experiments/e1/claim", ClaimRequest{AgentID: "ag", Device: "cuda:0"})
	code, _ = do(t, srv, "POST", base+"/experiments/e1/fail", FailRequest{Error: "kaboom"})
	if code != http.StatusOK {
		t.Errorf("fail: %d, want 200", code)
	}
	code, body := do(t, srv, "GET", base+"/experiments/e1", nil)
	if code != http.StatusOK || body["status"] != "failed" {
		t.Errorf("after fail: %d status=%v", code, body["status"])
	}
}

func TestHandlerNotFoundPaths(t *testing.T) {
	s, ds := testStore(t)
	srv := fullRouter(s)
	base := "/api/v1/datasets/" + ds

	for _, tc := range []struct {
		method, path string
		body         interface{}
	}{
		{"GET", base + "/experiments/ghost", nil},
		{"POST", base + "/experiments/ghost/claim", ClaimRequest{AgentID: "ag"}},
		{"POST", base + "/experiments/ghost/checkpoints", CheckpointRequest{Step: 1}},
		{"DELETE", base + "/experiments/ghost", nil},
	} {
		code, _ := do(t, srv, tc.method, tc.path, tc.body)
		if code != http.StatusNotFound {
			t.Errorf("%s %s: %d, want 404", tc.method, tc.path, code)
		}
	}
}

func TestHandlerBulkCreateOverHTTP(t *testing.T) {
	s, ds := testStore(t)
	srv := fullRouter(s)
	base := "/api/v1/datasets/" + ds

	items := []CreateExperimentRequest{
		{ExpName: "b1", Config: map[string]interface{}{"epochs": 1}},
		{ExpName: "b2", Config: map[string]interface{}{"epochs": 3}},
	}
	code, body := do(t, srv, "POST", base+"/experiments/bulk",
		map[string]interface{}{"experiments": items})
	if code != http.StatusOK {
		t.Fatalf("bulk: %d %v", code, body)
	}
	if body["inserted"] != float64(2) {
		t.Errorf("inserted = %v, want 2", body["inserted"])
	}

	// Idempotent re-registration.
	code, body = do(t, srv, "POST", base+"/experiments/bulk",
		map[string]interface{}{"experiments": items})
	if code != http.StatusOK || body["skipped"] != float64(2) {
		t.Errorf("re-bulk: %d skipped=%v, want 200/2", code, body["skipped"])
	}
}

// The agent's grid registration order must not decide the claim order —
// that is now the sort's job. Registered CNN-first, claimed epochs-first.
func TestClaimOrderIndependentOfRegistrationOrder(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	// Mimic the YAML: the whole CNN block first, then LSTM.
	for i := 0; i < 5; i++ {
		e := exp(fmt.Sprintf("cnn%d", i), 20, 32, "cuda:0", StatusPending)
		e.Architecture = "CNN"
		seed(t, s, ds, e)
	}
	for i := 0; i < 5; i++ {
		e := exp(fmt.Sprintf("lstm%d", i), 1, 32, "cuda:0", StatusPending)
		e.Architecture = "LSTM"
		seed(t, s, ds, e)
	}
	sortDoc, _ := parseSort("config.epochs:asc")

	got, err := s.ClaimNext(ctx, ds, "ag", "cuda:0", sortDoc, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.Architecture != "LSTM" {
		t.Errorf("first claim = %s (%s), want an LSTM: registration order must "+
			"not beat the sort", got.ExpName, got.Architecture)
	}
}

func TestEnsureIndexesOnEmptyCollection(t *testing.T) {
	s, ds := testStore(t)
	if err := s.EnsureIndexes(context.Background(), ds); err != nil {
		t.Fatalf("EnsureIndexes on empty collection: %v", err)
	}
}

func TestEnsureIndexesRejectsBadDataset(t *testing.T) {
	s, _ := testStore(t)
	if err := s.EnsureIndexes(context.Background(), "bad name"); err != ErrInvalidDataset {
		t.Fatalf("err = %v, want ErrInvalidDataset", err)
	}
}

func TestClaimNextRejectsBadDataset(t *testing.T) {
	s, _ := testStore(t)
	_, err := s.ClaimNext(context.Background(), "bad name", "ag", "d", bson.D{}, nil)
	if err != ErrInvalidDataset {
		t.Fatalf("err = %v, want ErrInvalidDataset", err)
	}
}

func TestClaimNextWithEmptySortStillWorks(t *testing.T) {
	s, ds := testStore(t)
	seed(t, s, ds, exp("only", 1, 32, "cuda:0", StatusPending))
	got, err := s.ClaimNext(context.Background(), ds, "ag", "d", bson.D{}, nil)
	if err != nil {
		t.Fatalf("empty sort must be allowed: %v", err)
	}
	if got.ExpName != "only" {
		t.Errorf("got %s", got.ExpName)
	}
}
