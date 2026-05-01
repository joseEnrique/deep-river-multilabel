package main

import "os"

type Config struct {
	Addr     string
	MongoURI string
	DBPrefix string
	APIKey   string
}

func LoadConfig() Config {
	return Config{
		Addr:     getEnv("BACKEND_ADDR", ":8080"),
		MongoURI: getEnv("MONGO_URI", "mongodb://localhost:27017"),
		DBPrefix: getEnv("DB_PREFIX", "experiments_"),
		APIKey:   getEnv("API_KEY", ""),
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
