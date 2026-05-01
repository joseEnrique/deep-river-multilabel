package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
)

func main() {
	cfg := LoadConfig()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	store, err := NewMongoStore(ctx, cfg.MongoURI, cfg.DBPrefix)
	if err != nil {
		log.Fatalf("mongo connect: %v", err)
	}
	defer store.Close(context.Background())

	h := &Handlers{Store: store}

	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(60 * time.Second))
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
