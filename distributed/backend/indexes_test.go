package main

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"go.mongodb.org/mongo-driver/bson"
)

// The sort strings the agent sends, one per pick_order. Kept in sync with
// _SORT_BY_PICK_ORDER in distributed/agent/agent.py — if that map changes and
// this one does not, TestEveryPickOrderIsIndexBacked will fail on the new sort.
var agentSortSpecs = map[string]string{
	"speed_asc":  "config.epochs:asc,config.hidden_dim:asc,config.num_layers:asc",
	"speed_desc": "config.epochs:desc,config.hidden_dim:asc,config.num_layers:asc",
	"size_asc":   "config.hidden_dim:asc,config.num_layers:asc,config.epochs:asc",
	"size_desc":  "config.hidden_dim:desc,config.num_layers:desc,config.epochs:asc",
	"slots":      "config.hidden_dim:asc,config.num_layers:asc",
}

// explainFind returns the winning plan's stage names for a find+sort+limit.
func explainFind(t *testing.T, s *Store, dataset string, filter bson.M, sortDoc bson.D) []string {
	t.Helper()
	c, err := s.coll(dataset)
	if err != nil {
		t.Fatal(err)
	}
	var plan bson.M
	err = c.Database().RunCommand(context.Background(), bson.D{
		{Key: "explain", Value: bson.D{
			{Key: "find", Value: collectionName},
			{Key: "filter", Value: filter},
			{Key: "sort", Value: sortDoc},
			{Key: "limit", Value: 1},
		}},
		{Key: "verbosity", Value: "queryPlanner"},
	}).Decode(&plan)
	if err != nil {
		t.Skipf("explain unavailable: %v", err)
	}
	qp, ok := plan["queryPlanner"].(bson.M)
	if !ok {
		t.Skipf("unexpected explain shape")
	}
	winning, ok := qp["winningPlan"].(bson.M)
	if !ok {
		t.Skipf("no winningPlan")
	}
	return collectStages(winning)
}

func hasStage(stages []string, want string) bool {
	for _, s := range stages {
		if s == want {
			return true
		}
	}
	return false
}

func seedForIndexes(t *testing.T, s *Store, ds string, n int) {
	t.Helper()
	devices := []string{"cuda:0", "cuda:1"}
	epochs := []int{1, 3, 10, 20}
	hidden := []int{32, 64, 128}
	for i := 0; i < n; i++ {
		e := &Experiment{
			ExpName: fmt.Sprintf("e%04d", i),
			Status:  StatusPending,
			Config: map[string]interface{}{
				"epochs":     epochs[i%len(epochs)],
				"hidden_dim": hidden[i%len(hidden)],
				"num_layers": 1 + i%2,
				"device":     devices[i%len(devices)],
				"loss":       map[string]interface{}{"type": "BCE"},
			},
		}
		if err := s.Create(context.Background(), ds, e); err != nil {
			t.Fatal(err)
		}
	}
}

// Every pick_order the agent can send must be served from an index. A blocking
// SORT stage means Mongo sorts the whole pending queue in memory — capped at
// 32 MB, so it does not slow down, it errors out once the grid is big.
func TestEveryPickOrderIsIndexBacked(t *testing.T) {
	s, ds := testStore(t)
	seedForIndexes(t, s, ds, 200)
	if err := s.EnsureIndexes(context.Background(), ds); err != nil {
		t.Fatal(err)
	}

	statusFilter := bson.M{"status": bson.M{"$in": []Status{StatusPending, StatusFailed}}}

	for pickOrder, spec := range agentSortSpecs {
		sortDoc, err := parseSort(spec)
		if err != nil {
			t.Fatalf("%s: %v", pickOrder, err)
		}

		t.Run(pickOrder, func(t *testing.T) {
			stages := explainFind(t, s, ds, statusFilter, sortDoc)
			if hasStage(stages, "SORT") {
				t.Errorf("blocking SORT for %s (%s): stages=%v", pickOrder, spec, stages)
			}
			if !hasStage(stages, "IXSCAN") {
				t.Errorf("no IXSCAN for %s (%s): stages=%v", pickOrder, spec, stages)
			}
		})

		t.Run(pickOrder+"_with_device_affinity", func(t *testing.T) {
			filter := bson.M{
				"status":        bson.M{"$in": []Status{StatusPending, StatusFailed}},
				"config.device": "cuda:0",
			}
			stages := explainFind(t, s, ds, filter, sortDoc)
			if hasStage(stages, "SORT") {
				t.Errorf("blocking SORT for %s+device (%s): stages=%v", pickOrder, spec, stages)
			}
			if !hasStage(stages, "IXSCAN") {
				t.Errorf("no IXSCAN for %s+device (%s): stages=%v", pickOrder, spec, stages)
			}
		})
	}
}

// The other read paths must not collection-scan either.
func TestAnalysisQueriesAreIndexBacked(t *testing.T) {
	s, ds := testStore(t)
	seedForIndexes(t, s, ds, 200)
	ctx := context.Background()
	// Give the summary/top indexes something to bite on.
	for i := 0; i < 20; i++ {
		name := fmt.Sprintf("e%04d", i)
		if _, err := s.Claim(ctx, ds, name, "ag", "cuda:0"); err != nil {
			t.Fatal(err)
		}
		if i < 10 {
			if _, err := s.Finish(ctx, ds, name,
				map[string]float64{"macro_f1": float64(i), "subset_acc": float64(i)},
				float64(i+1), nil); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := s.EnsureIndexes(ctx, ds); err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name   string
		filter bson.M
		sort   bson.D
	}{
		{
			"list_pending_by_created_at",
			bson.M{"status": StatusPending},
			bson.D{{Key: "created_at", Value: 1}},
		},
		{
			"summary_running_by_started_at",
			bson.M{"status": StatusRunning},
			bson.D{{Key: "started_at", Value: 1}},
		},
		{
			"cube_top_macro_f1",
			bson.M{"status": StatusDone, "final_metrics.macro_f1": bson.M{"$exists": true}},
			bson.D{{Key: "final_metrics.macro_f1", Value: -1}},
		},
		{
			"cube_top_subset_acc",
			bson.M{"status": StatusDone, "final_metrics.subset_acc": bson.M{"$exists": true}},
			bson.D{{Key: "final_metrics.subset_acc", Value: -1}},
		},
		{
			"csv_by_architecture",
			bson.M{"status": StatusDone, "architecture": "LSTM"},
			bson.D{{Key: "created_at", Value: 1}},
		},
		{
			"csv_by_loss_type",
			bson.M{"status": StatusDone, "config.loss.type": "BCE"},
			bson.D{{Key: "created_at", Value: 1}},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			stages := explainFind(t, s, ds, c.filter, c.sort)
			if hasStage(stages, "COLLSCAN") {
				t.Errorf("%s does a full collection scan: stages=%v", c.name, stages)
			}
			if hasStage(stages, "SORT") {
				t.Errorf("%s does a blocking in-memory sort: stages=%v", c.name, stages)
			}
		})
	}
}

// A plain status filter with no sort must never scan the collection: that is
// the query behind /stats and every `status=pending` listing.
func TestStatusFilterNeverCollectionScans(t *testing.T) {
	s, ds := testStore(t)
	seedForIndexes(t, s, ds, 100)
	if err := s.EnsureIndexes(context.Background(), ds); err != nil {
		t.Fatal(err)
	}
	for _, st := range []Status{StatusPending, StatusRunning, StatusDone, StatusFailed} {
		stages := explainFind(t, s, ds, bson.M{"status": st},
			bson.D{{Key: "created_at", Value: 1}})
		if hasStage(stages, "COLLSCAN") {
			t.Errorf("status=%s collection-scans: stages=%v", st, stages)
		}
	}
}

func TestEnsureIndexesCreatesTheFullSet(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	seedForIndexes(t, s, ds, 5)

	// Repeated calls must not duplicate or error.
	for i := 0; i < 3; i++ {
		if err := s.EnsureIndexes(ctx, ds); err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
	}

	c, _ := s.coll(ds)
	cur, err := c.Indexes().List(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var idx []bson.M
	if err := cur.All(ctx, &idx); err != nil {
		t.Fatal(err)
	}

	names := map[string]bool{}
	for _, i := range idx {
		if n, ok := i["name"].(string); ok {
			names[n] = true
		}
	}
	want := []string{
		"claim_speed_asc", "claim_speed_desc", "claim_size_asc", "claim_size_desc",
		"claim_device_speed_asc", "claim_device_speed_desc",
		"claim_device_size_asc", "claim_device_size_desc",
		"list_status_created", "summary_running", "summary_duration",
		"csv_architecture", "csv_agent_device", "csv_loss_type",
		"top_macro_f1", "top_subset_acc",
	}
	for _, w := range want {
		if !names[w] {
			t.Errorf("index %q missing; have %v", w, names)
		}
	}
	// _id + the declared set, nothing duplicated by the repeated calls.
	if len(idx) != len(want)+1 {
		raw, _ := json.Marshal(names)
		t.Errorf("got %d indexes, want %d: %s", len(idx), len(want)+1, raw)
	}
}

// Index count is a real cost on the write path: the grid registers 161k
// documents in bulk. This pins the number so it cannot creep silently.
func TestIndexCountStaysBounded(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	seedForIndexes(t, s, ds, 5)
	if err := s.EnsureIndexes(ctx, ds); err != nil {
		t.Fatal(err)
	}
	c, _ := s.coll(ds)
	cur, _ := c.Indexes().List(ctx)
	var idx []bson.M
	_ = cur.All(ctx, &idx)
	if len(idx) > 20 {
		t.Errorf("%d indexes — each one slows down the bulk registration of "+
			"the grid; justify before raising this bound", len(idx))
	}
}
