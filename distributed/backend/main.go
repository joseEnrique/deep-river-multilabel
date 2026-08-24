package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
)

// exceptSuffix applies a middleware to every request except the ones whose
// path ends in `suffix`. Used to keep the blanket request timeout off the
// streaming CSV export without having to hang the middleware on each of the
// two dozen routes that do want it.
func exceptSuffix(suffix string, mw func(http.Handler) http.Handler) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		wrapped := mw(next)
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if strings.HasSuffix(r.URL.Path, suffix) {
				next.ServeHTTP(w, r)
				return
			}
			wrapped.ServeHTTP(w, r)
		})
	}
}

func main() {
	cfg := LoadConfig()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	store, err := NewMongoStore(ctx, cfg.MongoURI, cfg.DBPrefix)
	if err != nil {
		log.Fatalf("mongo connect: %v", err)
	}
	defer store.Close(context.Background())

	// Indexes for the hot paths. Built in the background and one dataset at a
	// time: on a 240k-document collection this is heavy, and doing it inline
	// would both delay startup and hit Mongo with everything at once.
	// A failure is not fatal — the server still serves, just slower.
	go func() {
		datasets, derr := store.ListDatasets(ctx)
		if derr != nil {
			log.Printf("index setup: cannot list datasets: %v", derr)
			return
		}
		for _, ds := range datasets {
			start := time.Now()
			if ierr := store.EnsureIndexes(ctx, ds); ierr != nil {
				log.Printf("index setup: %s: %v", ds, ierr)
			} else {
				log.Printf("index setup: %s ok (%s)", ds, time.Since(start).Round(time.Millisecond))
			}
		}
	}()

	h := &Handlers{Store: store}

	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	// 60s is right for every JSON endpoint and wrong for results.csv, which
	// is a streaming export of the whole collection: the middleware cancels
	// the request context, which would kill the cursor mid-file. The export
	// carries its own (much longer) deadline instead.
	r.Use(exceptSuffix("/results.csv", middleware.Timeout(60*time.Second)))
	// Hard ceiling on in-flight requests. This is what guarantees Mongo is
	// never asked to do more than `cfg.MaxInFlight` things at once, however
	// many agents connect. Excess requests wait in the backlog; if that fills
	// too, the client gets a 503 — which BackendClient already retries with
	// backoff, so the load sheds instead of piling up.
	r.Use(middleware.ThrottleBacklog(cfg.MaxInFlight, cfg.Backlog, 30*time.Second))
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-Agent-ID", "X-API-Key"},
		AllowCredentials: false,
		MaxAge:           300,
	}))
	r.Use(APIKeyAuth(cfg.APIKey))
	if cfg.APIKey == "" {
		log.Println("WARNING: API_KEY is empty — server is running without authentication")
	}

	r.Get("/api/v1/health", h.Health)
	r.Get("/api/v1/datasets", h.ListDatasets)

	r.Route("/api/v1/datasets/{dataset}", func(r chi.Router) {
		r.Get("/stats", h.Stats)
		r.Get("/summary", h.Summary)
		r.Get("/results.csv", h.ResultsCSV)
		r.Get("/experiments", h.ListExperiments)
		r.Post("/experiments", h.CreateExperiment)
		r.Post("/experiments/bulk", h.BulkCreate)
		r.Post("/claim-next", h.ClaimNextExperiment)
		r.Post("/backfill-scores", h.BackfillScores)

		r.Route("/cube", func(r chi.Router) {
			r.Get("/metrics", h.CubeMetrics)
			r.Get("/params", h.CubeParams)
			r.Get("/params/values", h.CubeParamValues)
			r.Get("/top", h.CubeTop)
			r.Get("/groupby", h.CubeGroupBy)
			r.Get("/best-per", h.CubeBestPer)
			r.Get("/distribution", h.CubeDistribution)
		})

		r.Route("/experiments/{name}", func(r chi.Router) {
			r.Get("/", h.GetExperiment)
			r.Put("/", h.ReplaceExperiment)
			r.Patch("/", h.PatchExperiment)
			r.Delete("/", h.DeleteExperiment)
			r.Post("/claim", h.ClaimExperiment)
			r.Post("/checkpoints", h.AppendCheckpoint)
			r.Post("/finish", h.FinishExperiment)
			r.Post("/fail", h.FailExperiment)
			r.Post("/release", h.ReleaseExperiment)
		})
	})

	srv := &http.Server{
		Addr:         cfg.Addr,
		Handler:      r,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	go func() {
		log.Printf("backend listening on %s (db prefix=%s)", cfg.Addr, cfg.DBPrefix)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("listen: %v", err)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	log.Println("shutting down...")

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("shutdown error: %v", err)
	}
}
