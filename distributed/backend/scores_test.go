package main

import (
	"context"
	"fmt"
	"net/http"
	"testing"

	"github.com/go-chi/chi/v5"
	"go.mongodb.org/mongo-driver/bson"
)

// compute_score ordering: speed first, then real model size
// (SMALL → MEDIUM → LARGE), and the backfill that makes it work on documents
// registered before the field existed.

func scored(name string, epochs int, score float64, tier, device string) *Experiment {
	return &Experiment{
		ExpName:      name,
		Status:       StatusPending,
		ComputeScore: score,
		SizeTier:     tier,
		Config: map[string]interface{}{
			"epochs": epochs, "device": device,
		},
	}
}

// The requested priority: within each epoch value, SMALL then MEDIUM then
// LARGE — and only then the next epoch value.
func TestSpeedThenSizeOrdering(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()

	// Inserted deliberately scrambled.
	seed(t, s, ds,
		scored("slow_small", 20, 1024, "SMALL", "cuda:0"),
		scored("fast_large", 1, 126024, "LARGE", "cuda:0"),
		scored("fast_small", 1, 1024, "SMALL", "cuda:0"),
		scored("slow_large", 20, 126024, "LARGE", "cuda:0"),
		scored("fast_medium", 1, 32768, "MEDIUM", "cuda:0"),
		scored("slow_medium", 20, 32768, "MEDIUM", "cuda:0"),
	)

	sortDoc, err := parseSort("config.epochs:asc,compute_score:asc")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"fast_small", "fast_medium", "fast_large",
		"slow_small", "slow_medium", "slow_large",
	}
	for i, w := range want {
		got, err := s.ClaimNext(ctx, ds, "ag", "cuda:0", sortDoc, nil)
		if err != nil {
			t.Fatalf("claim %d: %v", i, err)
		}
		if got.ExpName != w {
			t.Fatalf("claim %d = %s, want %s (speed then size)", i, got.ExpName, w)
		}
	}
}

// The reason the size key is compute_score and not config.hidden_dim: for a
// Transformer the ws² attention term dominates, so hidden_dim gets the tier
// ordering backwards.
func TestComputeScoreBeatsHiddenDimForTransformers(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()

	// Transformer hd=32 ws=500 → 32²·1 + 500²·0.5 = 126,024 → LARGE
	tr := &Experiment{
		ExpName: "transformer_small_hd", Status: StatusPending,
		Architecture: "Transformer", ComputeScore: 126024, SizeTier: "LARGE",
		Config: map[string]interface{}{
			"epochs": 1, "hidden_dim": 32, "num_layers": 1, "window_size": 500,
		},
	}
	// LSTM hd=128 nl=2 → 128²·2 = 32,768 → MEDIUM
	lstm := &Experiment{
		ExpName: "lstm_big_hd", Status: StatusPending,
		Architecture: "LSTM", ComputeScore: 32768, SizeTier: "MEDIUM",
		Config: map[string]interface{}{
			"epochs": 1, "hidden_dim": 128, "num_layers": 2, "window_size": 500,
		},
	}
	seed(t, s, ds, tr, lstm)

	// By compute_score the MEDIUM LSTM comes first — correct.
	sortDoc, _ := parseSort("config.epochs:asc,compute_score:asc")
	got, err := s.ClaimNext(ctx, ds, "ag", "d", sortDoc, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.ExpName != "lstm_big_hd" {
		t.Errorf("got %s, want lstm_big_hd (MEDIUM before LARGE)", got.ExpName)
	}

	// By hidden_dim the LARGE Transformer would have come first — the bug
	// this field exists to avoid.
	s2, ds2 := testStore(t)
	seed(t, s2, ds2, tr, lstm)
	byHidden, _ := parseSort("config.epochs:asc,config.hidden_dim:asc")
	wrong, err := s2.ClaimNext(ctx, ds2, "ag", "d", byHidden, nil)
	if err != nil {
		t.Fatal(err)
	}
	if wrong.ExpName != "transformer_small_hd" {
		t.Logf("note: hidden_dim ordering gave %s", wrong.ExpName)
	} else {
		t.Logf("confirmed: ordering by hidden_dim would pick the LARGE " +
			"Transformer first — hence compute_score")
	}
}

// A missing field sorts FIRST ascending in Mongo, so un-backfilled documents
// would jump the whole queue. This is a correctness bug, not a slow path.
func TestMissingScoreJumpsTheQueueUntilBackfilled(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()

	seed(t, s, ds, scored("has_score_fast_small", 1, 1024, "SMALL", "cuda:0"))
	// Registered the old way: no compute_score at all, and expensive.
	seed(t, s, ds, &Experiment{
		ExpName: "no_score_slow_large", Status: StatusPending,
		Architecture: "Transformer",
		Config: map[string]interface{}{
			"epochs": 1, "hidden_dim": 128, "num_layers": 2, "window_size": 500,
		},
	})

	sortDoc, _ := parseSort("config.epochs:asc,compute_score:asc")
	got, err := s.ClaimNext(ctx, ds, "ag", "d", sortDoc, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.ExpName != "no_score_slow_large" {
		t.Fatalf("expected the un-scored doc to jump the queue, got %s", got.ExpName)
	}
	if _, err := s.Release(ctx, ds, got.ExpName); err != nil {
		t.Fatal(err)
	}

	// After the backfill, ordering is correct again.
	if _, _, missing, err := s.BackfillScores(ctx, ds, true); err != nil {
		t.Fatal(err)
	} else if missing != 0 {
		t.Fatalf("%d documents still without a score", missing)
	}
	got, err = s.ClaimNext(ctx, ds, "ag", "d", sortDoc, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.ExpName != "has_score_fast_small" {
		t.Errorf("after backfill got %s, want has_score_fast_small", got.ExpName)
	}
}

// The server-side pipeline must reproduce slots.get_compute_score exactly.
func TestBackfillMatchesAgentFormula(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()

	cases := []struct {
		name      string
		arch      string
		config    map[string]interface{}
		wantScore float64
		wantTier  string
	}{
		{"lstm_small", "LSTM",
			map[string]interface{}{"hidden_dim": 32, "num_layers": 1, "window_size": 500},
			1024, "SMALL"},
		{"lstm_medium", "LSTM",
			map[string]interface{}{"hidden_dim": 128, "num_layers": 2, "window_size": 500},
			32768, "MEDIUM"},
		{"lstm_64_1", "LSTM",
			map[string]interface{}{"hidden_dim": 64, "num_layers": 1, "window_size": 1},
			4096, "SMALL"},
		{"lstm_64_2", "LSTM",
			map[string]interface{}{"hidden_dim": 64, "num_layers": 2, "window_size": 1},
			8192, "MEDIUM"},
		{"transformer_large", "Transformer",
			map[string]interface{}{"hidden_dim": 32, "num_layers": 1, "window_size": 500},
			126024, "LARGE"},
		{"transformer_ws1", "Transformer",
			map[string]interface{}{"hidden_dim": 32, "num_layers": 1, "window_size": 1},
			1024.5, "SMALL"},
		// MLP carries a list; the largest layer wins.
		{"mlp_list", "MLP",
			map[string]interface{}{"hidden_dims": bson.A{128, 64, 32}, "num_layers": 1, "window_size": 1},
			16384, "MEDIUM"},
		// Missing fields fall back to the agent's defaults (hd=32, nl=1, ws=1).
		{"defaults", "LSTM", map[string]interface{}{}, 1024, "SMALL"},
	}

	for _, c := range cases {
		seed(t, s, ds, &Experiment{
			ExpName: c.name, Status: StatusPending,
			Architecture: c.arch, Config: c.config,
		})
	}
	if _, _, missing, err := s.BackfillScores(ctx, ds, true); err != nil {
		t.Fatal(err)
	} else if missing != 0 {
		t.Fatalf("%d without score", missing)
	}

	for _, c := range cases {
		got, err := s.Get(ctx, ds, c.name)
		if err != nil {
			t.Fatal(err)
		}
		if got.ComputeScore != c.wantScore {
			t.Errorf("%s: score = %v, want %v (must match slots.get_compute_score)",
				c.name, got.ComputeScore, c.wantScore)
		}
		if got.SizeTier != c.wantTier {
			t.Errorf("%s: tier = %q, want %q", c.name, got.SizeTier, c.wantTier)
		}
	}
}

func TestBackfillIsIdempotentAndScoped(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()

	seed(t, s, ds, &Experiment{
		ExpName: "a", Status: StatusPending, Architecture: "LSTM",
		Config: map[string]interface{}{"hidden_dim": 64, "num_layers": 1},
	})
	// Already scored, with a deliberately wrong value.
	seed(t, s, ds, &Experiment{
		ExpName: "b", Status: StatusPending, Architecture: "LSTM",
		ComputeScore: 999999, SizeTier: "LARGE",
		Config: map[string]interface{}{"hidden_dim": 64, "num_layers": 1},
	})

	// Default scope: only fill the gaps, leave existing values alone.
	if _, _, _, err := s.BackfillScores(ctx, ds, true); err != nil {
		t.Fatal(err)
	}
	a, _ := s.Get(ctx, ds, "a")
	if a.ComputeScore != 4096 {
		t.Errorf("a score = %v, want 4096", a.ComputeScore)
	}
	b, _ := s.Get(ctx, ds, "b")
	if b.ComputeScore != 999999 {
		t.Errorf("onlyMissing=true overwrote an existing score: %v", b.ComputeScore)
	}

	// all=true recomputes everything, fixing the wrong value.
	if _, _, _, err := s.BackfillScores(ctx, ds, false); err != nil {
		t.Fatal(err)
	}
	b, _ = s.Get(ctx, ds, "b")
	if b.ComputeScore != 4096 {
		t.Errorf("recompute-all did not fix b: %v", b.ComputeScore)
	}

	// Running it repeatedly must not change anything further.
	for i := 0; i < 3; i++ {
		if _, _, missing, err := s.BackfillScores(ctx, ds, false); err != nil {
			t.Fatalf("run %d: %v", i, err)
		} else if missing != 0 {
			t.Errorf("run %d left %d without score", i, missing)
		}
	}
}

func TestRegisterCarriesScoreThroughBulk(t *testing.T) {
	s, ds := testStore(t)
	h := &Handlers{Store: s}
	r := chi.NewRouter()
	r.Route("/api/v1/datasets/{dataset}", func(r chi.Router) {
		r.Post("/experiments", h.CreateExperiment)
		r.Post("/experiments/bulk", h.BulkCreate)
		r.Post("/backfill-scores", h.BackfillScores)
		r.Get("/experiments/{name}", h.GetExperiment)
	})
	base := "/api/v1/datasets/" + ds

	code, _ := do(t, r, "POST", base+"/experiments", CreateExperimentRequest{
		ExpName: "single", Architecture: "LSTM",
		Config:       map[string]interface{}{"epochs": 1, "hidden_dim": 64},
		ComputeScore: 4096, SizeTier: "SMALL",
	})
	if code != http.StatusCreated {
		t.Fatalf("create: %d", code)
	}
	code, body := do(t, r, "GET", base+"/experiments/single", nil)
	if code != http.StatusOK {
		t.Fatalf("get: %d", code)
	}
	if body["compute_score"] != float64(4096) || body["size_tier"] != "SMALL" {
		t.Errorf("single: score=%v tier=%v, want 4096/SMALL",
			body["compute_score"], body["size_tier"])
	}

	items := []CreateExperimentRequest{
		{ExpName: "b1", Config: map[string]interface{}{"epochs": 1},
			ComputeScore: 1024, SizeTier: "SMALL"},
		{ExpName: "b2", Config: map[string]interface{}{"epochs": 1},
			ComputeScore: 126024, SizeTier: "LARGE"},
	}
	code, _ = do(t, r, "POST", base+"/experiments/bulk",
		map[string]interface{}{"experiments": items})
	if code != http.StatusOK {
		t.Fatalf("bulk: %d", code)
	}
	_, body = do(t, r, "GET", base+"/experiments/b2", nil)
	if body["compute_score"] != float64(126024) || body["size_tier"] != "LARGE" {
		t.Errorf("bulk lost the score: %v / %v",
			body["compute_score"], body["size_tier"])
	}
}

func TestBackfillHandlerOverHTTP(t *testing.T) {
	s, ds := testStore(t)
	h := &Handlers{Store: s}
	r := chi.NewRouter()
	r.Route("/api/v1/datasets/{dataset}", func(r chi.Router) {
		r.Post("/backfill-scores", h.BackfillScores)
	})
	for i := 0; i < 10; i++ {
		seed(t, s, ds, &Experiment{
			ExpName: fmt.Sprintf("e%d", i), Status: StatusPending,
			Architecture: "LSTM",
			Config:       map[string]interface{}{"hidden_dim": 64, "num_layers": 1},
		})
	}

	code, body := do(t, r, "POST", "/api/v1/datasets/"+ds+"/backfill-scores", nil)
	if code != http.StatusOK {
		t.Fatalf("code = %d body=%v", code, body)
	}
	if body["modified"] != float64(10) {
		t.Errorf("modified = %v, want 10", body["modified"])
	}
	if body["still_missing"] != float64(0) {
		t.Errorf("still_missing = %v, want 0", body["still_missing"])
	}

	// Second run: nothing left to fill.
	_, body = do(t, r, "POST", "/api/v1/datasets/"+ds+"/backfill-scores", nil)
	if body["matched"] != float64(0) {
		t.Errorf("second run matched = %v, want 0", body["matched"])
	}
}

// Every pick_order must still be index-backed with the new score-based sorts.
func TestScoreSortsAreIndexBacked(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	for i := 0; i < 200; i++ {
		seed(t, s, ds, scored(fmt.Sprintf("e%03d", i), 1+i%4,
			float64((i%6+1)*1024), "SMALL", fmt.Sprintf("cuda:%d", i%2)))
	}
	if err := s.EnsureIndexes(ctx, ds); err != nil {
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
				t.Errorf("blocking SORT for %s (%s): %v", pickOrder, spec, stages)
			}
		})
		t.Run(pickOrder+"_device", func(t *testing.T) {
			f := bson.M{
				"status":        bson.M{"$in": []Status{StatusPending, StatusFailed}},
				"config.device": "cuda:0",
			}
			stages := explainFind(t, s, ds, f, sortDoc)
			if hasStage(stages, "SORT") {
				t.Errorf("blocking SORT for %s+device (%s): %v", pickOrder, spec, stages)
			}
		})
	}
}
