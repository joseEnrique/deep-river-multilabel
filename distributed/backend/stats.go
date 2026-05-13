package main

import (
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"github.com/go-chi/chi/v5"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo/options"
)

// metricPath maps a metric query value to its BSON path.
// "exact_match" → "final_metrics.exact_match". A value already prefixed with
// "final_metrics." is returned untouched.
func metricPath(m string) string {
	if m == "" {
		return ""
	}
	if strings.HasPrefix(m, "final_metrics.") {
		return m
	}
	return "final_metrics." + m
}

// parseFilterValue tries float, then bool, then leaves the value as string —
// so a query like `config.lr=0.5` matches numeric BSON values and
// `architecture=LSTM` matches string values.
func parseFilterValue(v string) interface{} {
	if f, err := strconv.ParseFloat(v, 64); err == nil {
		return f
	}
	switch strings.ToLower(v) {
	case "true":
		return true
	case "false":
		return false
	}
	return v
}

// parseWhere reads "k=v,k2=v2" into a bson.M ready for $match.
// Empty / malformed pairs are skipped.
func parseWhere(s string) bson.M {
	out := bson.M{}
	if s == "" {
		return out
	}
	for _, pair := range strings.Split(s, ",") {
		kv := strings.SplitN(pair, "=", 2)
		if len(kv) != 2 {
			continue
		}
		k := strings.TrimSpace(kv[0])
		v := strings.TrimSpace(kv[1])
		if k == "" {
			continue
		}
		out[k] = parseFilterValue(v)
	}
	return out
}

// statusMatch returns the status part of a $match doc.
// Default fallback applies when ?status is missing; "all" disables filtering.
func statusMatch(raw, def string) (string, bool) {
	if raw == "" {
		raw = def
	}
	if raw == "all" {
		return "", false
	}
	return raw, true
}

// escapeKey replaces dots so dotted paths can be used as BSON document field
// names in `$group._id`. We restore the original key when serialising.
func escapeKey(k string) string {
	return strings.ReplaceAll(k, ".", "__")
}

func normalizeAggName(a string) string {
	switch a {
	case "avg":
		return "mean"
	case "stddev":
		return "std"
	}
	return a
}

func percentileFromName(name string) float64 {
	switch name {
	case "p50":
		return 0.5
	case "p90":
		return 0.9
	case "p95":
		return 0.95
	case "p99":
		return 0.99
	}
	return 0.5
}

func percentile(vs bson.A, p float64) float64 {
	if len(vs) == 0 {
		return 0
	}
	nums := make([]float64, 0, len(vs))
	for _, v := range vs {
		if f, ok := toFloat(v); ok {
			nums = append(nums, f)
		}
	}
	if len(nums) == 0 {
		return 0
	}
	sort.Float64s(nums)
	idx := int(math.Floor(p * float64(len(nums)-1)))
	if idx < 0 {
		idx = 0
	}
	if idx >= len(nums) {
		idx = len(nums) - 1
	}
	return nums[idx]
}

func toFloat(v interface{}) (float64, bool) {
	switch t := v.(type) {
	case float64:
		return t, true
	case float32:
		return float64(t), true
	case int:
		return float64(t), true
	case int32:
		return float64(t), true
	case int64:
		return float64(t), true
	}
	return 0, false
}

// CubeMetrics lists every metric key found in final_metrics with
// count / min / max / mean over the filtered set.
func (h *Handlers) CubeMetrics(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	q := r.URL.Query()
	c, err := h.Store.coll(dataset)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	match := bson.M{"final_metrics": bson.M{"$exists": true, "$ne": nil}}
	if s, ok := statusMatch(q.Get("status"), "done"); ok {
		match["status"] = s
	}

	pipeline := []bson.M{
		{"$match": match},
		{"$project": bson.M{"kv": bson.M{"$objectToArray": "$final_metrics"}}},
		{"$unwind": "$kv"},
		{"$group": bson.M{
			"_id":   "$kv.k",
			"count": bson.M{"$sum": 1},
			"min":   bson.M{"$min": "$kv.v"},
			"max":   bson.M{"$max": "$kv.v"},
			"mean":  bson.M{"$avg": "$kv.v"},
		}},
		{"$sort": bson.D{{Key: "_id", Value: 1}}},
	}

	cur, err := c.Aggregate(r.Context(), pipeline)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer cur.Close(r.Context())

	var rows []bson.M
	if err := cur.All(r.Context(), &rows); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	out := make([]map[string]interface{}, 0, len(rows))
	for _, x := range rows {
		out = append(out, map[string]interface{}{
			"name":  x["_id"],
			"count": x["count"],
			"min":   x["min"],
			"max":   x["max"],
			"mean":  x["mean"],
		})
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"dataset": dataset,
		"metrics": out,
	})
}

// CubeParams returns the flattened config keys discovered across a sample of
// experiments. These are the dimensions you can use as `by=` or in `where=`.
func (h *Handlers) CubeParams(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	q := r.URL.Query()
	c, err := h.Store.coll(dataset)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	sample, _ := strconv.ParseInt(q.Get("sample"), 10, 64)
	if sample <= 0 {
		sample = 1000
	}

	filter := bson.M{}
	if s, ok := statusMatch(q.Get("status"), "all"); ok {
		filter["status"] = s
	}
	opts := options.Find().
		SetProjection(bson.M{"config": 1, "architecture": 1, "agent_id": 1, "device": 1}).
		SetLimit(sample)
	cur, err := c.Find(r.Context(), filter, opts)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer cur.Close(r.Context())

	keyCounts := map[string]int{}
	topLevelCounts := map[string]int{}
	for cur.Next(r.Context()) {
		var doc struct {
			Config       map[string]interface{} `bson:"config"`
			Architecture string                 `bson:"architecture"`
			AgentID      string                 `bson:"agent_id"`
			Device       string                 `bson:"device"`
		}
		if err := cur.Decode(&doc); err != nil {
			continue
		}
		if doc.Architecture != "" {
			topLevelCounts["architecture"]++
		}
		if doc.AgentID != "" {
			topLevelCounts["agent_id"]++
		}
		if doc.Device != "" {
			topLevelCounts["device"]++
		}
		flat := map[string]string{}
		flattenJSON("config", doc.Config, flat)
		for k := range flat {
			keyCounts[k]++
		}
	}

	keys := make([]map[string]interface{}, 0, len(keyCounts))
	for k, count := range keyCounts {
		keys = append(keys, map[string]interface{}{"key": k, "count": count})
	}
	sort.Slice(keys, func(i, j int) bool {
		return keys[i]["key"].(string) < keys[j]["key"].(string)
	})
	topLevel := make([]map[string]interface{}, 0, len(topLevelCounts))
	for k, c := range topLevelCounts {
		topLevel = append(topLevel, map[string]interface{}{"key": k, "count": c})
	}
	sort.Slice(topLevel, func(i, j int) bool {
		return topLevel[i]["key"].(string) < topLevel[j]["key"].(string)
	})

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"dataset":   dataset,
		"sample":    sample,
		"params":    keys,
		"top_level": topLevel,
	})
}

// CubeParamValues lists distinct values of a single key with optional
// per-value metric aggregates (max / min / mean / count).
func (h *Handlers) CubeParamValues(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	q := r.URL.Query()
	key := q.Get("key")
	if key == "" {
		writeErr(w, http.StatusBadRequest, "key is required")
		return
	}
	metric := q.Get("metric")
	c, err := h.Store.coll(dataset)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	match := bson.M{}
	for k, v := range parseWhere(q.Get("where")) {
		match[k] = v
	}
	if s, ok := statusMatch(q.Get("status"), "done"); ok {
		match["status"] = s
	}

	group := bson.M{
		"_id":   "$" + key,
		"count": bson.M{"$sum": 1},
	}
	if metric != "" {
		mp := "$" + metricPath(metric)
		group["max"] = bson.M{"$max": mp}
		group["min"] = bson.M{"$min": mp}
		group["mean"] = bson.M{"$avg": mp}
	}
	pipeline := []bson.M{
		{"$match": match},
		{"$group": group},
		{"$sort": bson.D{{Key: "_id", Value: 1}}},
	}
	cur, err := c.Aggregate(r.Context(), pipeline)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer cur.Close(r.Context())

	var rows []bson.M
	if err := cur.All(r.Context(), &rows); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	out := make([]map[string]interface{}, 0, len(rows))
	for _, x := range rows {
		m := map[string]interface{}{
			"value": x["_id"],
			"count": x["count"],
		}
		if metric != "" {
			m["max"] = x["max"]
			m["min"] = x["min"]
			m["mean"] = x["mean"]
		}
		out = append(out, m)
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"dataset": dataset,
		"key":     key,
		"metric":  metric,
		"values":  out,
	})
}

// CubeTop returns the top-N experiments by a final metric.
func (h *Handlers) CubeTop(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	q := r.URL.Query()
	metric := q.Get("metric")
	if metric == "" {
		writeErr(w, http.StatusBadRequest, "metric is required")
		return
	}
	limit, _ := strconv.ParseInt(q.Get("limit"), 10, 64)
	if limit <= 0 {
		limit = 10
	}
	sortDir := -1
	if strings.EqualFold(q.Get("order"), "asc") {
		sortDir = 1
	}

	c, err := h.Store.coll(dataset)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	mp := metricPath(metric)
	filter := bson.M{mp: bson.M{"$exists": true}}
	for k, v := range parseWhere(q.Get("where")) {
		filter[k] = v
	}
	if s, ok := statusMatch(q.Get("status"), "done"); ok {
		filter["status"] = s
	}

	opts := options.Find().
		SetSort(bson.D{{Key: mp, Value: sortDir}}).
		SetLimit(limit)
	cur, err := c.Find(r.Context(), filter, opts)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer cur.Close(r.Context())
	var out []*Experiment
	if err := cur.All(r.Context(), &out); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"dataset":     dataset,
		"metric":      metric,
		"order":       q.Get("order"),
		"count":       len(out),
		"experiments": out,
	})
}

// CubeGroupBy is the OLAP data-cube endpoint. It groups by one or more keys
// (config.* or top-level fields) and computes aggregations over a metric.
func (h *Handlers) CubeGroupBy(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	q := r.URL.Query()
	byParam := q.Get("by")
	if byParam == "" {
		writeErr(w, http.StatusBadRequest, "by is required (comma-separated keys)")
		return
	}
	keys := strings.Split(byParam, ",")
	for i, k := range keys {
		keys[i] = strings.TrimSpace(k)
	}
	metric := q.Get("metric")
	if metric == "" {
		writeErr(w, http.StatusBadRequest, "metric is required")
		return
	}
	aggParam := q.Get("agg")
	if aggParam == "" {
		aggParam = "max,mean,count"
	}
	aggs := strings.Split(aggParam, ",")
	for i, a := range aggs {
		aggs[i] = strings.TrimSpace(a)
	}

	limit, _ := strconv.ParseInt(q.Get("limit"), 10, 64)
	if limit <= 0 {
		limit = 50
	}

	c, err := h.Store.coll(dataset)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	mp := metricPath(metric)
	match := bson.M{mp: bson.M{"$exists": true}}
	for k, v := range parseWhere(q.Get("where")) {
		match[k] = v
	}
	if s, ok := statusMatch(q.Get("status"), "done"); ok {
		match["status"] = s
	}

	idDoc := bson.M{}
	for _, k := range keys {
		idDoc[escapeKey(k)] = "$" + k
	}
	group := bson.M{"_id": idDoc}
	mpRef := "$" + mp
	needPercentile := false
	for _, a := range aggs {
		switch a {
		case "count":
			group["count"] = bson.M{"$sum": 1}
		case "max":
			group["max"] = bson.M{"$max": mpRef}
		case "min":
			group["min"] = bson.M{"$min": mpRef}
		case "mean", "avg":
			group["mean"] = bson.M{"$avg": mpRef}
		case "std", "stddev":
			group["std"] = bson.M{"$stdDevSamp": mpRef}
		case "p50", "p90", "p95", "p99":
			needPercentile = true
			group["_values"] = bson.M{"$push": mpRef}
		default:
			writeErr(w, http.StatusBadRequest, "unknown agg: "+a)
			return
		}
	}

	// Sort field. Default: first non-count agg, descending.
	sortField, sortDir := defaultSort(aggs)
	if v := q.Get("order"); v != "" {
		parts := strings.SplitN(v, ":", 2)
		sortField = normalizeAggName(parts[0])
		if len(parts) == 2 && strings.EqualFold(parts[1], "asc") {
			sortDir = 1
		} else {
			sortDir = -1
		}
	}
	isPercentileSort := strings.HasPrefix(sortField, "p")

	pipeline := []bson.M{
		{"$match": match},
		{"$group": group},
	}
	if !isPercentileSort {
		pipeline = append(pipeline,
			bson.M{"$sort": bson.D{{Key: sortField, Value: sortDir}}},
			bson.M{"$limit": limit},
		)
	}

	cur, err := c.Aggregate(r.Context(), pipeline)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer cur.Close(r.Context())
	var rows []bson.M
	if err := cur.All(r.Context(), &rows); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	type result struct {
		Group   map[string]interface{} `json:"group"`
		Metrics map[string]interface{} `json:"metrics"`
	}
	out := make([]result, 0, len(rows))
	for _, row := range rows {
		idMap, _ := row["_id"].(bson.M)
		grp := map[string]interface{}{}
		for _, k := range keys {
			if idMap != nil {
				grp[k] = idMap[escapeKey(k)]
			}
		}
		mtr := map[string]interface{}{}
		for _, a := range aggs {
			name := normalizeAggName(a)
			switch a {
			case "p50", "p90", "p95", "p99":
				if vs, ok := row["_values"].(bson.A); ok {
					mtr[name] = percentile(vs, percentileFromName(a))
				}
			default:
				if v, ok := row[name]; ok {
					mtr[name] = v
				}
			}
		}
		out = append(out, result{Group: grp, Metrics: mtr})
	}

	if isPercentileSort {
		sort.Slice(out, func(i, j int) bool {
			a, _ := toFloat(out[i].Metrics[sortField])
			b, _ := toFloat(out[j].Metrics[sortField])
			if sortDir == 1 {
				return a < b
			}
			return a > b
		})
		if int64(len(out)) > limit {
			out = out[:limit]
		}
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"dataset": dataset,
		"by":      keys,
		"metric":  metric,
		"agg":     aggs,
		"order":   sortField,
		"count":   len(out),
		"groups":  out,
	})
	_ = needPercentile
}

func defaultSort(aggs []string) (string, int) {
	for _, a := range aggs {
		if a == "count" {
			continue
		}
		return normalizeAggName(a), -1
	}
	return "count", -1
}

// CubeBestPer returns, for each value of `by`, the single best experiment
// (full document) ranked by `metric`.
func (h *Handlers) CubeBestPer(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	q := r.URL.Query()
	byKey := q.Get("by")
	if byKey == "" {
		writeErr(w, http.StatusBadRequest, "by is required")
		return
	}
	metric := q.Get("metric")
	if metric == "" {
		writeErr(w, http.StatusBadRequest, "metric is required")
		return
	}
	sortDir := -1
	if strings.EqualFold(q.Get("order"), "asc") {
		sortDir = 1
	}
	limit, _ := strconv.ParseInt(q.Get("limit"), 10, 64)
	if limit <= 0 {
		limit = 50
	}

	c, err := h.Store.coll(dataset)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	mp := metricPath(metric)
	match := bson.M{mp: bson.M{"$exists": true}}
	for k, v := range parseWhere(q.Get("where")) {
		match[k] = v
	}
	if s, ok := statusMatch(q.Get("status"), "done"); ok {
		match["status"] = s
	}

	pipeline := []bson.M{
		{"$match": match},
		{"$sort": bson.D{{Key: mp, Value: sortDir}}},
		{"$group": bson.M{
			"_id":   "$" + byKey,
			"best":  bson.M{"$first": "$$ROOT"},
			"count": bson.M{"$sum": 1},
		}},
		{"$sort": bson.D{{Key: "best." + mp, Value: sortDir}}},
		{"$limit": limit},
	}
	cur, err := c.Aggregate(r.Context(), pipeline)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer cur.Close(r.Context())
	var rows []bson.M
	if err := cur.All(r.Context(), &rows); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	out := make([]map[string]interface{}, 0, len(rows))
	for _, row := range rows {
		var best *Experiment
		if bm, ok := row["best"].(bson.M); ok {
			if b, err := bson.Marshal(bm); err == nil {
				var e Experiment
				if err := bson.Unmarshal(b, &e); err == nil {
					best = &e
				}
			}
		}
		out = append(out, map[string]interface{}{
			"value": row["_id"],
			"count": row["count"],
			"best":  best,
		})
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"dataset": dataset,
		"by":      byKey,
		"metric":  metric,
		"count":   len(out),
		"groups":  out,
	})
}

// CubeDistribution returns a histogram of a metric over the filtered set.
func (h *Handlers) CubeDistribution(w http.ResponseWriter, r *http.Request) {
	dataset := chi.URLParam(r, "dataset")
	q := r.URL.Query()
	metric := q.Get("metric")
	if metric == "" {
		writeErr(w, http.StatusBadRequest, "metric is required")
		return
	}
	bins, _ := strconv.Atoi(q.Get("bins"))
	if bins <= 0 {
		bins = 10
	}

	c, err := h.Store.coll(dataset)
	if err != nil {
		if mapErr(w, err) {
			return
		}
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	mp := metricPath(metric)
	match := bson.M{mp: bson.M{"$exists": true}}
	for k, v := range parseWhere(q.Get("where")) {
		match[k] = v
	}
	if s, ok := statusMatch(q.Get("status"), "done"); ok {
		match["status"] = s
	}

	// First pass: min / max / count.
	cur, err := c.Aggregate(r.Context(), []bson.M{
		{"$match": match},
		{"$group": bson.M{
			"_id":   nil,
			"min":   bson.M{"$min": "$" + mp},
			"max":   bson.M{"$max": "$" + mp},
			"mean":  bson.M{"$avg": "$" + mp},
			"std":   bson.M{"$stdDevSamp": "$" + mp},
			"count": bson.M{"$sum": 1},
		}},
	})
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer cur.Close(r.Context())
	var stats []struct {
		Min   float64 `bson:"min"`
		Max   float64 `bson:"max"`
		Mean  float64 `bson:"mean"`
		Std   float64 `bson:"std"`
		Count int64   `bson:"count"`
	}
	if err := cur.All(r.Context(), &stats); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if len(stats) == 0 || stats[0].Count == 0 {
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"dataset": dataset, "metric": metric, "count": 0, "bins": []interface{}{},
		})
		return
	}
	mn, mx := stats[0].Min, stats[0].Max
	if mn == mx {
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"dataset": dataset, "metric": metric,
			"min": mn, "max": mx, "mean": stats[0].Mean, "std": stats[0].Std,
			"count": stats[0].Count,
			"bins": []map[string]interface{}{{
				"lo": mn, "hi": mx, "count": stats[0].Count,
			}},
		})
		return
	}

	width := (mx - mn) / float64(bins)
	boundaries := make([]interface{}, 0, bins+1)
	for i := 0; i <= bins; i++ {
		boundaries = append(boundaries, mn+width*float64(i))
	}
	// Extend the last boundary so the max value falls inside the final bucket.
	boundaries[len(boundaries)-1] = mx + width*1e-6

	cur2, err := c.Aggregate(r.Context(), []bson.M{
		{"$match": match},
		{"$bucket": bson.M{
			"groupBy":    "$" + mp,
			"boundaries": boundaries,
			"default":    "other",
			"output":     bson.M{"count": bson.M{"$sum": 1}},
		}},
	})
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer cur2.Close(r.Context())
	var buckets []struct {
		ID    interface{} `bson:"_id"`
		Count int64       `bson:"count"`
	}
	if err := cur2.All(r.Context(), &buckets); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	countByLo := map[int]int64{}
	for _, b := range buckets {
		f, ok := toFloat(b.ID)
		if !ok {
			continue
		}
		idx := int(math.Round((f - mn) / width))
		if idx >= 0 && idx < bins {
			countByLo[idx] = b.Count
		}
	}
	binsOut := make([]map[string]interface{}, 0, bins)
	for i := 0; i < bins; i++ {
		binsOut = append(binsOut, map[string]interface{}{
			"lo":    mn + width*float64(i),
			"hi":    mn + width*float64(i+1),
			"count": countByLo[i],
		})
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"dataset": dataset,
		"metric":  metric,
		"min":     mn,
		"max":     mx,
		"mean":    stats[0].Mean,
		"std":     stats[0].Std,
		"count":   stats[0].Count,
		"bins":    binsOut,
	})
}
