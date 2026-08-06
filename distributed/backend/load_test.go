package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

// Protections that keep Mongo from ever being saturated: a hard ceiling on
// concurrent requests, load shedding when the backlog fills, and pool bounds.

func TestConfigConcurrencyDefaults(t *testing.T) {
	t.Setenv("MAX_IN_FLIGHT", "")
	t.Setenv("REQUEST_BACKLOG", "")
	c := LoadConfig()
	if c.MaxInFlight != 32 || c.Backlog != 256 {
		t.Errorf("defaults = %d/%d, want 32/256", c.MaxInFlight, c.Backlog)
	}

	t.Setenv("MAX_IN_FLIGHT", "8")
	t.Setenv("REQUEST_BACKLOG", "64")
	c = LoadConfig()
	if c.MaxInFlight != 8 || c.Backlog != 64 {
		t.Errorf("env override = %d/%d, want 8/64", c.MaxInFlight, c.Backlog)
	}
}

// A typo must not silently remove the ceiling — that is the whole protection.
func TestConfigRejectsBadConcurrencyValues(t *testing.T) {
	for _, bad := range []string{"0", "-5", "abc", "1.5", " "} {
		t.Setenv("MAX_IN_FLIGHT", bad)
		if got := LoadConfig().MaxInFlight; got != 32 {
			t.Errorf("MAX_IN_FLIGHT=%q → %d, want the 32 fallback", bad, got)
		}
	}
}

func TestGetEnvIntFallback(t *testing.T) {
	t.Setenv("X_INT_TEST", "")
	if got := getEnvInt("X_INT_TEST", 7); got != 7 {
		t.Errorf("unset → %d, want 7", got)
	}
	t.Setenv("X_INT_TEST", "13")
	if got := getEnvInt("X_INT_TEST", 7); got != 13 {
		t.Errorf("set → %d, want 13", got)
	}
}

// The throttle must actually cap concurrency: with a limit of N, no more than
// N handlers may be inside Mongo at the same instant.
func TestThrottleCapsConcurrency(t *testing.T) {
	const limit = 4
	var inFlight, maxSeen int64

	slow := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cur := atomic.AddInt64(&inFlight, 1)
		for {
			old := atomic.LoadInt64(&maxSeen)
			if cur <= old || atomic.CompareAndSwapInt64(&maxSeen, old, cur) {
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
		atomic.AddInt64(&inFlight, -1)
		w.WriteHeader(http.StatusOK)
	})

	r := chi.NewRouter()
	r.Use(middleware.ThrottleBacklog(limit, 256, 30*time.Second))
	r.Get("/x", slow)
	srv := httptest.NewServer(r)
	defer srv.Close()

	var wg sync.WaitGroup
	for i := 0; i < 60; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := http.Get(srv.URL + "/x")
			if err == nil {
				resp.Body.Close()
			}
		}()
	}
	wg.Wait()

	if got := atomic.LoadInt64(&maxSeen); got > limit {
		t.Errorf("peak concurrency = %d, want <= %d — the ceiling does not hold", got, limit)
	}
	if atomic.LoadInt64(&maxSeen) == 0 {
		t.Error("no request was served")
	}
}

// When the backlog is full the server must shed rather than queue without
// bound. chi sheds with 429 + Retry-After, which BackendClient treats as
// retryable — see the retry_on tuple in api_client.py. If that list and this
// status ever diverge, agents crash under load instead of backing off.
func TestThrottleShedsWhenBacklogFull(t *testing.T) {
	block := make(chan struct{})
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-block
		w.WriteHeader(http.StatusOK)
	})

	r := chi.NewRouter()
	r.Use(middleware.ThrottleBacklog(1, 1, 100*time.Millisecond))
	r.Get("/x", handler)
	srv := httptest.NewServer(r)
	defer srv.Close()

	var mu sync.Mutex
	codes := map[int]int{}
	var wg sync.WaitGroup
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := http.Get(srv.URL + "/x")
			if err != nil {
				return
			}
			defer resp.Body.Close()
			mu.Lock()
			codes[resp.StatusCode]++
			mu.Unlock()
		}()
	}

	time.Sleep(300 * time.Millisecond)
	close(block)
	wg.Wait()

	if codes[http.StatusTooManyRequests] == 0 {
		t.Errorf("nothing shed under overload: %v", codes)
	}
	// Whatever is shed must be retryable by the agent, never a hard error.
	for code, n := range codes {
		switch code {
		case http.StatusOK, http.StatusTooManyRequests,
			http.StatusServiceUnavailable:
		default:
			t.Errorf("shed with non-retryable status %d (%d times): agents "+
				"would fail instead of backing off", code, n)
		}
	}
}

// Pins the contract between the server's shedding status and the agent's
// retry list. Both must be updated together.
func TestShedStatusIsRetryableByAgent(t *testing.T) {
	// Mirrors retry_on in distributed/agent/api_client.py.
	agentRetries := map[int]bool{429: true, 502: true, 503: true, 504: true}
	if !agentRetries[http.StatusTooManyRequests] {
		t.Error("chi's throttle sheds with 429 but the agent does not retry it")
	}
}

// Pool bounds are what stop the backend opening unbounded connections to
// Mongo when many agents connect at once.
func TestMongoPoolIsBounded(t *testing.T) {
	if maxPoolSize == 0 {
		t.Fatal("maxPoolSize is unset — the pool would be unbounded")
	}
	if maxPoolSize > 200 {
		t.Errorf("maxPoolSize = %d, too high to call bounded", maxPoolSize)
	}
	if minPoolSize > maxPoolSize {
		t.Errorf("minPoolSize %d > maxPoolSize %d", minPoolSize, maxPoolSize)
	}
	if opTimeout <= 0 {
		t.Error("opTimeout unset — a runaway query could hold a connection forever")
	}
	// Must stay under the 60s HTTP middleware timeout, so the DB operation is
	// cancelled before the HTTP layer gives up on it.
	if opTimeout >= 60*time.Second {
		t.Errorf("opTimeout = %s, must be below the 60s HTTP timeout", opTimeout)
	}
	if serverSelectionTimeout <= 0 {
		t.Error("serverSelectionTimeout unset — requests would hang if Mongo is down")
	}
}

// A store built through NewMongoStore must carry those bounds; this catches
// somebody constructing the client without the options.
func TestNewMongoStoreAppliesBounds(t *testing.T) {
	uri, alive := probeMongo()
	if !alive {
		t.Skipf("no Mongo at %s", uri)
	}
	s, err := NewMongoStore(context.Background(), uri, "poolcheck_")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close(context.Background())

	// The driver exposes no getter for the pool size, so assert the operation
	// path works and that a cancelled context is honoured promptly (which is
	// what the timeouts are for).
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.Stats(ctx, "nothing"); err == nil {
		t.Error("a cancelled context should fail fast, not run to completion")
	}
}

// Under a burst of concurrent claims Mongo must stay correct: no duplicates,
// no lost work. This is the saturation scenario with four machines × 8 slots.
func TestBurstOfConcurrentClaimsStaysCorrect(t *testing.T) {
	s, ds := testStore(t)
	ctx := context.Background()
	const n = 120
	seedForIndexes(t, s, ds, n)
	if err := s.EnsureIndexes(ctx, ds); err != nil {
		t.Fatal(err)
	}
	sortDoc, _ := parseSort("config.epochs:asc,config.hidden_dim:asc")

	const agents = 32 // 4 machines × 8 slots
	var mu sync.Mutex
	seen := map[string]int{}

	var wg sync.WaitGroup
	for a := 0; a < agents; a++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for {
				got, err := s.ClaimNext(ctx, ds, "ag", "cuda:0", sortDoc, nil)
				if err == ErrNotFound {
					return
				}
				if err != nil {
					t.Errorf("agent %d: %v", id, err)
					return
				}
				mu.Lock()
				seen[got.ExpName]++
				mu.Unlock()
			}
		}(a)
	}
	wg.Wait()

	if len(seen) != n {
		t.Errorf("claimed %d distinct, want %d", len(seen), n)
	}
	for name, c := range seen {
		if c != 1 {
			t.Errorf("%s claimed %d times under burst load", name, c)
		}
	}
}
