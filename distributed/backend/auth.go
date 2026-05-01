package main

import (
	"crypto/subtle"
	"net/http"
	"strings"
)

// APIKeyAuth returns a middleware that requires `X-API-Key: <key>` (or
// `Authorization: Bearer <key>`). If the configured key is empty, the
// middleware is a no-op (handy for local dev).
//
// The `/api/v1/health` endpoint is always exempt so that liveness probes
// don't need credentials. CORS preflight requests are also exempt.
func APIKeyAuth(key string) func(http.Handler) http.Handler {
	keyBytes := []byte(key)
	exemptPaths := map[string]struct{}{
		"/api/v1/health": {},
	}
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if key == "" {
				next.ServeHTTP(w, r)
				return
			}
			if r.Method == http.MethodOptions {
				next.ServeHTTP(w, r)
				return
			}
			if _, ok := exemptPaths[r.URL.Path]; ok {
				next.ServeHTTP(w, r)
				return
			}

			provided := r.Header.Get("X-API-Key")
			if provided == "" {
				if h := r.Header.Get("Authorization"); strings.HasPrefix(h, "Bearer ") {
					provided = strings.TrimPrefix(h, "Bearer ")
				}
			}

			if provided == "" {
				writeErr(w, http.StatusUnauthorized, "missing API key")
				return
			}
			if subtle.ConstantTimeCompare([]byte(provided), keyBytes) != 1 {
				writeErr(w, http.StatusUnauthorized, "invalid API key")
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}
