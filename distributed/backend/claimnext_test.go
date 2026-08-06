package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

// ── Pure unit tests (no Mongo) ────────────────────────────────────────────────

func TestValidSortPath(t *testing.T) {
	ok := []string{"status", "config.epochs", "config.hidden_dim", "a_b.c1"}
	for _, p := range ok {
		if !validSortPath(p) {
			t.Errorf("validSortPath(%q) = false, want true", p)
		}
	}
	bad := []string{
		"",                      // empty
		"$where",                // operator injection
		"config.epochs; drop",   // separator + space
		"config['epochs']",      // brackets
		"config epochs",         // space
		"cfg\x00",               // NUL
		string(make([]byte, 0)), // empty again
	}
	for _, p := range bad {
		if validSortPath(p) {
			t.Errorf("validSortPath(%q) = true, want false", p)
		}
	}
	long := ""
	for i := 0; i < 129; i++ {
		long += "a"
	}
	if validSortPath(long) {
		t.Error("validSortPath(129 chars) = true, want false (length cap)")
	}
}

func TestParseSort(t *testing.T) {
	cases := []struct {
		in   string
		want bson.D
	}{
		{"", bson.D{}},
		{"config.epochs", bson.D{{Key: "config.epochs", Value: 1}}},
		{"config.epochs:asc", bson.D{{Key: "config.epochs", Value: 1}}},
		{"config.epochs:desc", bson.D{{Key: "config.epochs", Value: -1}}},
		{"config.epochs:DESC", bson.D{{Key: "config.epochs", Value: -1}}},
		{" config.epochs : desc ", bson.D{{Key: "config.epochs", Value: -1}}},
		{"config.epochs:asc,config.hidden_dim:desc", bson.D{
			{Key: "config.epochs", Value: 1},
			{Key: "config.hidden_dim", Value: -1},
		}},
		// Trailing/duplicate commas are skipped, order is preserved.
		{"a,,b", bson.D{{Key: "a", Value: 1}, {Key: "b", Value: 1}}},
	}
	for _, c := range cases {
		got, err := parseSort(c.in)
		if err != nil {
			t.Errorf("parseSort(%q) error: %v", c.in, err)
			continue
		}
		if len(got) != len(c.want) {
			t.Errorf("parseSort(%q) = %v, want %v", c.in, got, c.want)
			continue
		}
		for i := range got {
			if got[i].Key != c.want[i].Key || got[i].Value != c.want[i].Value {
				t.Errorf("parseSort(%q)[%d] = %v, want %v", c.in, i, got[i], c.want[i])
			}
		}
	}

	// An invalid key must be rejected, not silently dropped.
	for _, bad := range []string{"$where:asc", "a b", "config.epochs:asc,$x"} {
		if _, err := parseSort(bad); err == nil {
			t.Errorf("parseSort(%q) = nil error, want rejection", bad)
		}
	}
}

func TestSortIsUnorderedSafe(t *testing.T) {
	// bson.D preserves order — critical, because a sort document is
	// order-sensitive (primary key first). bson.M would not.
	got, err := parseSort("config.epochs:asc,config.hidden_dim:asc,config.num_layers:asc")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"config.epochs", "config.hidden_dim", "config.num_layers"}
	for i, k := range want {
		if got[i].Key != k {
			t.Fatalf("sort key %d = %q, want %q (order must be preserved)", i, got[i].Key, k)
		}
	}
}

// ── Integration tests (need Mongo) ───────────────────────────────────────────

var (
	testDBSeq  int64
	mongoOnce  sync.Once
	mongoURI   string
	mongoAlive bool
)

// probeMongo checks reachability ONCE for the whole package. Without this each
// integration test pays its own connect timeout, so a run with no Mongo takes
// minutes instead of milliseconds.
func probeMongo() (string, bool) {
	mongoOnce.Do(func() {
		mongoURI = os.Getenv("TEST_MONGO_URI")
		if mongoURI == "" {
			mongoURI = "mongodb://localhost:27099"
		}
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		c, err := mongo.Connect(ctx, options.Client().ApplyURI(mongoURI).
			SetServerSelectionTimeout(2*time.Second))
		if err != nil {
			return
		}
		if err := c.Ping(ctx, nil); err != nil {
			_ = c.Disconnect(context.Background())
			return
		}
		_ = c.Disconnect(context.Background())
		mongoAlive = true
	})
	return mongoURI, mongoAlive
}

func testStore(t *testing.T) (*Store, string) {
	t.Helper()
	uri, alive := probeMongo()
	if !alive {
		t.Skipf("no Mongo at %s — start one with: "+
			"docker run -d --name mongo-test -p 27099:27017 mongo:7", uri)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	client, err := mongo.Connect(ctx, options.Client().ApplyURI(uri))
	if err != nil {
		t.Skipf("no Mongo at %s: %v", uri, err)
	}
	if err := client.Ping(ctx, nil); err != nil {
		t.Skipf("no Mongo at %s: %v", uri, err)
	}
	// Unique but SHORT prefix per test: Mongo caps database names at 63 chars,
	// and the name is prefix+dataset.
	prefix := fmt.Sprintf("t%d_%d_", time.Now().UnixNano()%1e9, atomic.AddInt64(&testDBSeq, 1))
	s := &Store{client: client, dbPrefix: prefix}
	dataset := "ds"
	t.Cleanup(func() {
		_ = client.Database(prefix + dataset).Drop(context.Background())
		_ = client.Disconnect(context.Background())
	})
	return s, dataset
}

func seed(t *testing.T, s *Store, dataset string, exps ...*Experiment) {
	t.Helper()
	for _, e := range exps {
		if err := s.Create(context.Background(), dataset, e); err != nil {
			t.Fatalf("seed %s: %v", e.ExpName, err)
		}
	}
}

func exp(name string, epochs, hidden int, device string, status Status) *Experiment {
	return &Experiment{
		ExpName: name,
		Status:  status,
		Config: map[string]interface{}{
			"epochs":     epochs,
			"hidden_dim": hidden,
			"device":     device,
		},
	}
}

func TestClaimNextRespectsSortOrder(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()

	// Insert in an order that is NOT the expected claim order, so a passing
	// test cannot be explained by insertion order.
	seed(t, s, ds,
		exp("ep20", 20, 64, "cuda:0", StatusPending),
		exp("ep1", 1, 64, "cuda:0", StatusPending),
		exp("ep10", 10, 64, "cuda:0", StatusPending),
		exp("ep3", 3, 64, "cuda:0", StatusPending),
	)

	sortDoc, _ := parseSort("config.epochs:asc")
	want := []string{"ep1", "ep3", "ep10", "ep20"}
	for i, w := range want {
		got, err := s.ClaimNext(ctx, ds, "agent", "cuda:0", sortDoc, nil)
		if err != nil {
			t.Fatalf("claim %d: %v", i, err)
		}
		if got.ExpName != w {
			t.Fatalf("claim %d = %s, want %s (ascending epochs)", i, got.ExpName, w)
		}
		if got.Status != StatusRunning {
			t.Errorf("claimed %s status = %s, want running", got.ExpName, got.Status)
		}
		if got.AgentID != "agent" || got.Device != "cuda:0" {
			t.Errorf("claimed %s: agent=%q device=%q, want agent/cuda:0",
				got.ExpName, got.AgentID, got.Device)
		}
		if got.StartedAt == nil {
			t.Errorf("claimed %s: started_at not set", got.ExpName)
		}
	}
}

func TestClaimNextDescendingAndTiebreak(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	seed(t, s, ds,
		exp("a", 10, 128, "cuda:0", StatusPending),
		exp("b", 10, 32, "cuda:0", StatusPending),
		exp("c", 20, 32, "cuda:0", StatusPending),
	)
	// epochs desc, then hidden_dim asc → c (ep20), then b (ep10,hd32), then a.
	sortDoc, _ := parseSort("config.epochs:desc,config.hidden_dim:asc")
	for _, w := range []string{"c", "b", "a"} {
		got, err := s.ClaimNext(ctx, ds, "ag", "d", sortDoc, nil)
		if err != nil {
			t.Fatal(err)
		}
		if got.ExpName != w {
			t.Fatalf("got %s, want %s", got.ExpName, w)
		}
	}
}

func TestClaimNextOnlyPendingAndFailed(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	seed(t, s, ds,
		exp("done", 1, 32, "cuda:0", StatusDone),
		exp("running", 1, 32, "cuda:0", StatusRunning),
		exp("failed", 5, 32, "cuda:0", StatusFailed),
	)
	sortDoc, _ := parseSort("config.epochs:asc")

	// `failed` is claimable (a retry); done/running are not.
	got, err := s.ClaimNext(ctx, ds, "ag", "d", sortDoc, nil)
	if err != nil {
		t.Fatalf("failed row should be claimable: %v", err)
	}
	if got.ExpName != "failed" {
		t.Fatalf("claimed %s, want the failed one", got.ExpName)
	}
	// A claim must clear the previous error message.
	if got.Error != "" {
		t.Errorf("error field = %q, want cleared on claim", got.Error)
	}

	// Nothing else is claimable now.
	if _, err := s.ClaimNext(ctx, ds, "ag", "d", sortDoc, nil); err != ErrNotFound {
		t.Fatalf("err = %v, want ErrNotFound (done/running are not claimable)", err)
	}
}

func TestClaimNextEmptyQueue(t *testing.T) {
	s, ds := testStore(t)
	sortDoc, _ := parseSort("config.epochs:asc")
	_, err := s.ClaimNext(context.Background(), ds, "ag", "d", sortDoc, nil)
	if err != ErrNotFound {
		t.Fatalf("err = %v, want ErrNotFound on empty queue", err)
	}
}

func TestClaimNextDeviceAffinity(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	seed(t, s, ds,
		// The globally-fastest one is registered for a different device.
		exp("fast_other_gpu", 1, 32, "cuda:1", StatusPending),
		exp("slow_my_gpu", 20, 32, "cuda:0", StatusPending),
	)
	sortDoc, _ := parseSort("config.epochs:asc")

	// With affinity, the device filter wins over the sort.
	got, err := s.ClaimNext(ctx, ds, "ag", "cuda:0", sortDoc,
		bson.M{"config.device": "cuda:0"})
	if err != nil {
		t.Fatal(err)
	}
	if got.ExpName != "slow_my_gpu" {
		t.Fatalf("with affinity got %s, want slow_my_gpu", got.ExpName)
	}

	// Affinity exhausted → 404, so the agent can retry without the filter.
	_, err = s.ClaimNext(ctx, ds, "ag", "cuda:0", sortDoc,
		bson.M{"config.device": "cuda:0"})
	if err != ErrNotFound {
		t.Fatalf("err = %v, want ErrNotFound once affinity is exhausted", err)
	}

	// Fallback without filter picks up the foreign-device one.
	got, err = s.ClaimNext(ctx, ds, "ag", "cuda:0", sortDoc, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.ExpName != "fast_other_gpu" {
		t.Fatalf("fallback got %s, want fast_other_gpu", got.ExpName)
	}
}

// The whole point of claim-next: concurrent agents must never get the same
// experiment. Without atomicity this test produces duplicates.
func TestClaimNextIsAtomicUnderConcurrency(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()

	const n = 60
	for i := 0; i < n; i++ {
		seed(t, s, ds, exp(fmt.Sprintf("e%03d", i), i, 32, "cuda:0", StatusPending))
	}
	sortDoc, _ := parseSort("config.epochs:asc")

	const workers = 12
	var mu sync.Mutex
	seen := map[string]int{}
	notFound := 0

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for {
				got, err := s.ClaimNext(ctx, ds, fmt.Sprintf("ag%d", id), "d", sortDoc, nil)
				if err == ErrNotFound {
					mu.Lock()
					notFound++
					mu.Unlock()
					return
				}
				if err != nil {
					t.Errorf("worker %d: %v", id, err)
					return
				}
				mu.Lock()
				seen[got.ExpName]++
				mu.Unlock()
			}
		}(w)
	}
	wg.Wait()

	if len(seen) != n {
		t.Errorf("claimed %d distinct experiments, want %d", len(seen), n)
	}
	for name, count := range seen {
		if count != 1 {
			t.Errorf("%s claimed %d times — claim is NOT atomic", name, count)
		}
	}
	if notFound != workers {
		t.Errorf("%d workers saw an empty queue, want %d", notFound, workers)
	}
}

// The exact index set and its idempotency are asserted in
// TestEnsureIndexesCreatesTheFullSet (indexes_test.go).

// With the index in place the sorted claim must be served by an index scan,
// not an in-memory sort (which Mongo caps at 32 MB and would fail on a large
// pending queue).
func TestClaimNextUsesIndexNotInMemorySort(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	for i := 0; i < 200; i++ {
		seed(t, s, ds, exp(fmt.Sprintf("e%03d", i), i%20, 32, "cuda:0", StatusPending))
	}
	if err := s.EnsureIndexes(ctx, ds); err != nil {
		t.Fatal(err)
	}

	c, _ := s.coll(ds)
	var plan bson.M
	err := c.Database().RunCommand(ctx, bson.D{
		{Key: "explain", Value: bson.D{
			{Key: "find", Value: collectionName},
			{Key: "filter", Value: bson.M{"status": bson.M{"$in": []Status{StatusPending, StatusFailed}}}},
			{Key: "sort", Value: bson.D{{Key: "config.epochs", Value: 1}}},
			{Key: "limit", Value: 1},
		}},
		{Key: "verbosity", Value: "queryPlanner"},
	}).Decode(&plan)
	if err != nil {
		t.Skipf("explain unavailable: %v", err)
	}
	qp, ok := plan["queryPlanner"].(bson.M)
	if !ok {
		t.Skipf("unexpected explain shape: %v", plan)
	}
	winning, ok := qp["winningPlan"].(bson.M)
	if !ok {
		t.Skipf("no winningPlan in explain: %v", qp)
	}
	stages := collectStages(winning)

	// A blocking "SORT" stage means Mongo materialised and sorted the whole
	// result set in memory (capped at 32 MB → fails on a big pending queue).
	// "SORT_MERGE" is fine and expected: the $in on status splits into one
	// index scan per status, merged in streaming order.
	for _, s := range stages {
		if s == "SORT" {
			raw, _ := json.Marshal(winning)
			t.Errorf("winning plan has a blocking SORT stage (stages=%v):\n%s", stages, raw)
		}
	}
	hasIx := false
	for _, s := range stages {
		if s == "IXSCAN" {
			hasIx = true
		}
	}
	if !hasIx {
		raw, _ := json.Marshal(winning)
		t.Errorf("winning plan has no IXSCAN (stages=%v):\n%s", stages, raw)
	}
}

// collectStages walks an explain plan tree and returns every stage name.
func collectStages(node bson.M) []string {
	var out []string
	if s, ok := node["stage"].(string); ok {
		out = append(out, s)
	}
	if child, ok := node["inputStage"].(bson.M); ok {
		out = append(out, collectStages(child)...)
	}
	if children, ok := node["inputStages"].(bson.A); ok {
		for _, c := range children {
			if cm, ok := c.(bson.M); ok {
				out = append(out, collectStages(cm)...)
			}
		}
	}
	return out
}

// ── HTTP handler tests ───────────────────────────────────────────────────────

func testRouter(s *Store) http.Handler {
	h := &Handlers{Store: s}
	r := chi.NewRouter()
	r.Route("/api/v1/datasets/{dataset}", func(r chi.Router) {
		r.Post("/claim-next", h.ClaimNextExperiment)
	})
	return r
}

func postClaimNext(t *testing.T, srv http.Handler, dataset string, body interface{}) (int, map[string]interface{}) {
	t.Helper()
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(body); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest("POST", "/api/v1/datasets/"+dataset+"/claim-next", &buf)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)
	var out map[string]interface{}
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return rec.Code, out
}

func TestClaimNextHandler(t *testing.T) {
	s, ds := testStore(t)
	srv := testRouter(s)
	seed(t, s, ds,
		exp("slow", 20, 32, "cuda:0", StatusPending),
		exp("fast", 1, 32, "cuda:0", StatusPending),
	)

	// Happy path: honours the sort.
	code, body := postClaimNext(t, srv, ds, ClaimNextRequest{
		AgentID: "ag1", Device: "cuda:0", Sort: "config.epochs:asc",
	})
	if code != http.StatusOK {
		t.Fatalf("code = %d, want 200 (body=%v)", code, body)
	}
	if body["exp_name"] != "fast" {
		t.Errorf("exp_name = %v, want fast", body["exp_name"])
	}
	if body["status"] != "running" {
		t.Errorf("status = %v, want running", body["status"])
	}

	// agent_id is mandatory.
	code, _ = postClaimNext(t, srv, ds, ClaimNextRequest{Sort: "config.epochs:asc"})
	if code != http.StatusBadRequest {
		t.Errorf("missing agent_id: code = %d, want 400", code)
	}

	// An invalid sort key is rejected rather than silently ignored.
	code, _ = postClaimNext(t, srv, ds, ClaimNextRequest{
		AgentID: "ag1", Sort: "$where:asc",
	})
	if code != http.StatusBadRequest {
		t.Errorf("invalid sort: code = %d, want 400", code)
	}

	// Drain, then an empty queue must be 404 (the agent's "nothing pending").
	code, _ = postClaimNext(t, srv, ds, ClaimNextRequest{AgentID: "ag1", Sort: "config.epochs:asc"})
	if code != http.StatusOK {
		t.Fatalf("draining claim: code = %d, want 200", code)
	}
	code, _ = postClaimNext(t, srv, ds, ClaimNextRequest{AgentID: "ag1", Sort: "config.epochs:asc"})
	if code != http.StatusNotFound {
		t.Errorf("empty queue: code = %d, want 404", code)
	}
}

func TestClaimNextHandlerAgentIDFromHeader(t *testing.T) {
	s, ds := testStore(t)
	srv := testRouter(s)
	seed(t, s, ds, exp("only", 1, 32, "cuda:0", StatusPending))

	var buf bytes.Buffer
	_ = json.NewEncoder(&buf).Encode(ClaimNextRequest{Sort: "config.epochs:asc"})
	req := httptest.NewRequest("POST", "/api/v1/datasets/"+ds+"/claim-next", &buf)
	req.Header.Set("X-Agent-ID", "from-header")
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("code = %d, want 200: %s", rec.Code, rec.Body.String())
	}
	var out Experiment
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.AgentID != "from-header" {
		t.Errorf("agent_id = %q, want from-header", out.AgentID)
	}
}

func TestClaimNextHandlerPreferDevice(t *testing.T) {
	s, ds := testStore(t)
	srv := testRouter(s)
	seed(t, s, ds,
		exp("fast_gpu1", 1, 32, "cuda:1", StatusPending),
		exp("slow_gpu0", 20, 32, "cuda:0", StatusPending),
	)
	code, body := postClaimNext(t, srv, ds, ClaimNextRequest{
		AgentID: "ag", Device: "cuda:0", Sort: "config.epochs:asc",
		PreferDevice: "cuda:0",
	})
	if code != http.StatusOK {
		t.Fatalf("code = %d", code)
	}
	if body["exp_name"] != "slow_gpu0" {
		t.Errorf("exp_name = %v, want slow_gpu0 (affinity beats sort)", body["exp_name"])
	}
}

func TestClaimNextHandlerBadJSONAndDataset(t *testing.T) {
	s, ds := testStore(t)
	srv := testRouter(s)

	req := httptest.NewRequest("POST", "/api/v1/datasets/"+ds+"/claim-next",
		bytes.NewBufferString("{not json"))
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("bad json: code = %d, want 400", rec.Code)
	}

	code, _ := postClaimNext(t, srv, "bad!dataset", ClaimNextRequest{AgentID: "ag"})
	if code != http.StatusBadRequest && code != http.StatusNotFound {
		t.Errorf("invalid dataset: code = %d, want 400/404", code)
	}
}
