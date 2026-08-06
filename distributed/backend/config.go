package main

import (
	"os"
	"strconv"
)

type Config struct {
	Addr     string
	MongoURI string
	DBPrefix string
	APIKey   string
	// MaxInFlight caps how many requests are served concurrently, which is
	// what bounds the load reaching Mongo. Backlog is how many may wait
	// before the server starts shedding with 503 (agents retry with backoff).
	MaxInFlight int
	Backlog     int
}

func LoadConfig() Config {
	return Config{
		Addr:        getEnv("BACKEND_ADDR", ":8080"),
		MongoURI:    getEnv("MONGO_URI", "mongodb://localhost:27017"),
		DBPrefix:    getEnv("DB_PREFIX", "experiments_"),
		APIKey:      getEnv("API_KEY", ""),
		MaxInFlight: getEnvInt("MAX_IN_FLIGHT", 32),
		Backlog:     getEnvInt("REQUEST_BACKLOG", 256),
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// getEnvInt falls back on anything unparseable or non-positive, so a typo in
// the environment cannot silently disable the concurrency ceiling.
func getEnvInt(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return fallback
	}
	return n
}
