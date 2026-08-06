package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

const collectionName = "experiments"

var (
	ErrNotFound       = errors.New("experiment not found")
	ErrAlreadyExists  = errors.New("experiment already exists")
	ErrInvalidState   = errors.New("invalid state transition")
	ErrInvalidDataset = errors.New("invalid dataset name")
)

type Store struct {
	client   *mongo.Client
	dbPrefix string
}

// Connection-pool bounds. The point is that Mongo can never be asked to do
// more work than this at once, no matter how many agents show up: excess
// requests queue on the pool instead of piling onto the server.
const (
	maxPoolSize            = 50
	minPoolSize            = 5
	maxConnIdleTime        = 5 * time.Minute
	serverSelectionTimeout = 10 * time.Second
	// Hard ceiling per operation. A runaway query gets killed server-side
	// rather than holding a connection (and Mongo's resources) forever.
	opTimeout = 55 * time.Second
)

func NewMongoStore(ctx context.Context, uri, dbPrefix string) (*Store, error) {
	connectCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	opts := options.Client().ApplyURI(uri).
		SetMaxPoolSize(maxPoolSize).
		SetMinPoolSize(minPoolSize).
		SetMaxConnIdleTime(maxConnIdleTime).
		SetServerSelectionTimeout(serverSelectionTimeout).
		SetTimeout(opTimeout).
		SetRetryWrites(true).
		SetRetryReads(true)

	client, err := mongo.Connect(connectCtx, opts)
	if err != nil {
		return nil, fmt.Errorf("connect: %w", err)
	}
	if err := client.Ping(connectCtx, nil); err != nil {
		return nil, fmt.Errorf("ping: %w", err)
	}
	return &Store{client: client, dbPrefix: dbPrefix}, nil
}

func (s *Store) Close(ctx context.Context) error {
	return s.client.Disconnect(ctx)
}

func (s *Store) coll(dataset string) (*mongo.Collection, error) {
	if !validDatasetName(dataset) {
		return nil, ErrInvalidDataset
	}
	return s.client.Database(s.dbPrefix + dataset).Collection(collectionName), nil
}

func validDatasetName(d string) bool {
	if d == "" || len(d) > 64 {
		return false
	}
	for _, r := range d {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_' || r == '-') {
			return false
		}
	}
	return true
}

// EnsureIndexes creates the indexes every hot path depends on. Idempotent:
// CreateMany on an existing index is a no-op.
//
// Two things make these mandatory rather than nice-to-have:
//
//   - Without a `status` prefix, every `status=pending` query is a full
//     collection scan (240k docs on alpi).
//   - A sorted claim-next with no usable index makes Mongo materialise and
//     sort the whole pending queue in memory, which it caps at 32 MB. That
//     does not degrade — it *fails* once the queue is big enough.
//
// Sort direction matters: Mongo serves a sort from an index only when the
// requested sort equals the index key pattern or its EXACT inverse. That is
// why `speed_desc` (epochs desc, hidden_dim asc — mixed directions) needs its
// own index and cannot reuse the `speed_asc` one.
func (s *Store) EnsureIndexes(ctx context.Context, dataset string) error {
	c, err := s.coll(dataset)
	if err != nil {
		return err
	}
	idx := func(name string, keys bson.D) mongo.IndexModel {
		return mongo.IndexModel{Keys: keys, Options: options.Index().SetName(name)}
	}
	models := []mongo.IndexModel{
		// ── claim-next: one index per pick_order sort shape ──────────────
		// speed_asc (the default): epochs → hidden_dim → num_layers, all asc.
		idx("claim_speed_asc", bson.D{
			{Key: "status", Value: 1},
			{Key: "config.epochs", Value: 1},
			{Key: "config.hidden_dim", Value: 1},
			{Key: "config.num_layers", Value: 1},
		}),
		// speed_desc: mixed directions, so not covered by the index above.
		idx("claim_speed_desc", bson.D{
			{Key: "status", Value: 1},
			{Key: "config.epochs", Value: -1},
			{Key: "config.hidden_dim", Value: 1},
			{Key: "config.num_layers", Value: 1},
		}),
		// size_asc, and `slots` as its two-key prefix.
		idx("claim_size_asc", bson.D{
			{Key: "status", Value: 1},
			{Key: "config.hidden_dim", Value: 1},
			{Key: "config.num_layers", Value: 1},
			{Key: "config.epochs", Value: 1},
		}),
		// size_desc: hidden_dim/num_layers desc but epochs asc → mixed again.
		idx("claim_size_desc", bson.D{
			{Key: "status", Value: 1},
			{Key: "config.hidden_dim", Value: -1},
			{Key: "config.num_layers", Value: -1},
			{Key: "config.epochs", Value: 1},
		}),
		// Device-affinity variants: prefer_device adds an equality on
		// config.device, which must sit before the sort keys.
		idx("claim_device_speed_asc", bson.D{
			{Key: "status", Value: 1},
			{Key: "config.device", Value: 1},
			{Key: "config.epochs", Value: 1},
			{Key: "config.hidden_dim", Value: 1},
			{Key: "config.num_layers", Value: 1},
		}),
		idx("claim_device_speed_desc", bson.D{
			{Key: "status", Value: 1},
			{Key: "config.device", Value: 1},
			{Key: "config.epochs", Value: -1},
			{Key: "config.hidden_dim", Value: 1},
			{Key: "config.num_layers", Value: 1},
		}),
		idx("claim_device_size_asc", bson.D{
			{Key: "status", Value: 1},
			{Key: "config.device", Value: 1},
			{Key: "config.hidden_dim", Value: 1},
			{Key: "config.num_layers", Value: 1},
			{Key: "config.epochs", Value: 1},
		}),
		idx("claim_device_size_desc", bson.D{
			{Key: "status", Value: 1},
			{Key: "config.device", Value: 1},
			{Key: "config.hidden_dim", Value: -1},
			{Key: "config.num_layers", Value: -1},
			{Key: "config.epochs", Value: 1},
		}),

		// ── list / stats / summary ───────────────────────────────────────
		// GET /experiments and results.csv: filter by status, sort created_at.
		idx("list_status_created", bson.D{
			{Key: "status", Value: 1},
			{Key: "created_at", Value: 1},
		}),
		// Summary's running table: status=running sorted by started_at.
		idx("summary_running", bson.D{
			{Key: "status", Value: 1},
			{Key: "started_at", Value: 1},
		}),
		// Summary's duration aggregate: status=done, duration_s > 0.
		idx("summary_duration", bson.D{
			{Key: "status", Value: 1},
			{Key: "duration_s", Value: 1},
		}),

		// ── CSV export filters ───────────────────────────────────────────
		idx("csv_architecture", bson.D{
			{Key: "status", Value: 1},
			{Key: "architecture", Value: 1},
			{Key: "created_at", Value: 1},
		}),
		idx("csv_agent_device", bson.D{
			{Key: "status", Value: 1},
			{Key: "agent_id", Value: 1},
			{Key: "device", Value: 1},
		}),
		idx("csv_loss_type", bson.D{
			{Key: "status", Value: 1},
			{Key: "config.loss.type", Value: 1},
		}),

		// ── cube/top on the two metrics actually reported ────────────────
		// Other metric names fall back to a scan; these are the ones the
		// grid is ranked by.
		idx("top_macro_f1", bson.D{
			{Key: "status", Value: 1},
			{Key: "final_metrics.macro_f1", Value: 1},
		}),
		idx("top_subset_acc", bson.D{
			{Key: "status", Value: 1},
			{Key: "final_metrics.subset_acc", Value: 1},
		}),
	}
	idxCtx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()
	_, err = c.Indexes().CreateMany(idxCtx, models)
	return err
}

// ClaimNext atomically picks the best pending/failed experiment according to
// `sort` and flips it to running in a single round trip. Because Mongo's
// FindOneAndUpdate is atomic, two agents can never receive the same document —
// there is no claim race to retry.
//
// `extraFilter` is merged into the status filter (used for soft device
// affinity: try `config.device == cuda:0` first, then fall back to no filter).
func (s *Store) ClaimNext(ctx context.Context, dataset, agentID, device string,
	sort bson.D, extraFilter bson.M) (*Experiment, error) {
	c, err := s.coll(dataset)
	if err != nil {
		return nil, err
	}
	filter := bson.M{
		"status": bson.M{"$in": []Status{StatusPending, StatusFailed}},
	}
	for k, v := range extraFilter {
		filter[k] = v
	}
	now := time.Now().UTC()
	opts := options.FindOneAndUpdate().SetReturnDocument(options.After)
	if len(sort) > 0 {
		opts = opts.SetSort(sort)
	}
	res := c.FindOneAndUpdate(ctx,
		filter,
		bson.M{"$set": bson.M{
			"status":     StatusRunning,
			"agent_id":   agentID,
			"device":     device,
			"started_at": now,
			"updated_at": now,
			"error":      "",
		}},
		opts)
	if err := res.Err(); err != nil {
		if errors.Is(err, mongo.ErrNoDocuments) {
			return nil, ErrNotFound
		}
		return nil, err
	}
	var exp Experiment
	if err := res.Decode(&exp); err != nil {
		return nil, err
	}
	return &exp, nil
}

func (s *Store) Create(ctx context.Context, dataset string, exp *Experiment) error {
	c, err := s.coll(dataset)
	if err != nil {
		return err
	}
	now := time.Now().UTC()
	exp.Dataset = dataset
	exp.CreatedAt = now
	exp.UpdatedAt = now
	if exp.Status == "" {
		exp.Status = StatusPending
	}
	_, err = c.InsertOne(ctx, exp)
	if err != nil {
		if mongo.IsDuplicateKeyError(err) {
			return ErrAlreadyExists
		}
		return err
	}
	return nil
}

func (s *Store) BulkCreate(ctx context.Context, dataset string, exps []*Experiment) (inserted, skipped int, err error) {
	c, err := s.coll(dataset)
	if err != nil {
		return 0, 0, err
	}
	if len(exps) == 0 {
		return 0, 0, nil
	}
	now := time.Now().UTC()
	docs := make([]interface{}, 0, len(exps))
	for _, e := range exps {
		e.Dataset = dataset
		e.CreatedAt = now
		e.UpdatedAt = now
		if e.Status == "" {
			e.Status = StatusPending
		}
		docs = append(docs, e)
	}
	opts := options.InsertMany().SetOrdered(false)
	res, err := c.InsertMany(ctx, docs, opts)
	if res != nil {
		inserted = len(res.InsertedIDs)
	}
	if err != nil {
		var bwe mongo.BulkWriteException
		if errors.As(err, &bwe) {
			for _, we := range bwe.WriteErrors {
				if we.Code == 11000 {
					skipped++
				}
			}
			if inserted+skipped == len(exps) {
				return inserted, skipped, nil
			}
		}
		return inserted, skipped, err
	}
	return inserted, skipped, nil
}

func (s *Store) Get(ctx context.Context, dataset, name string) (*Experiment, error) {
	c, err := s.coll(dataset)
	if err != nil {
		return nil, err
	}
	var exp Experiment
	err = c.FindOne(ctx, bson.M{"_id": name}).Decode(&exp)
	if errors.Is(err, mongo.ErrNoDocuments) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	return &exp, nil
}

func (s *Store) List(ctx context.Context, dataset string, statusFilter string, limit, offset int64) ([]*Experiment, error) {
	filter := bson.M{}
	if statusFilter != "" {
		filter["status"] = statusFilter
	}
	return s.ListFiltered(ctx, dataset, filter, limit, offset)
}

// ListFiltered runs an arbitrary mongo filter. Used by the CSV exporter so
// callers can apply ad-hoc query params like architecture, agent_id, etc.
func (s *Store) ListFiltered(ctx context.Context, dataset string, filter bson.M, limit, offset int64) ([]*Experiment, error) {
	c, err := s.coll(dataset)
	if err != nil {
		return nil, err
	}
	if filter == nil {
		filter = bson.M{}
	}
	opts := options.Find().SetSort(bson.D{{Key: "created_at", Value: 1}})
	if limit > 0 {
		opts = opts.SetLimit(limit)
	}
	if offset > 0 {
		opts = opts.SetSkip(offset)
	}
	cur, err := c.Find(ctx, filter, opts)
	if err != nil {
		return nil, err
	}
	defer cur.Close(ctx)
	var out []*Experiment
	if err := cur.All(ctx, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func (s *Store) Replace(ctx context.Context, dataset string, exp *Experiment) error {
	c, err := s.coll(dataset)
	if err != nil {
		return err
	}
	existing, err := s.Get(ctx, dataset, exp.ExpName)
	if err != nil {
		return err
	}
	if existing.Status == StatusDone {
		return ErrInvalidState
	}
	exp.Dataset = dataset
	exp.CreatedAt = existing.CreatedAt
	exp.UpdatedAt = time.Now().UTC()
	_, err = c.ReplaceOne(ctx, bson.M{"_id": exp.ExpName}, exp)
	return err
}

func (s *Store) Patch(ctx context.Context, dataset, name string, patch map[string]interface{}) (*Experiment, error) {
	c, err := s.coll(dataset)
	if err != nil {
		return nil, err
	}
	delete(patch, "_id")
	delete(patch, "exp_name")
	delete(patch, "dataset")
	delete(patch, "created_at")
	patch["updated_at"] = time.Now().UTC()

	res := c.FindOneAndUpdate(ctx,
		bson.M{"_id": name},
		bson.M{"$set": patch},
		options.FindOneAndUpdate().SetReturnDocument(options.After))
	if err := res.Err(); err != nil {
		if errors.Is(err, mongo.ErrNoDocuments) {
			return nil, ErrNotFound
		}
		return nil, err
	}
	var exp Experiment
	if err := res.Decode(&exp); err != nil {
		return nil, err
	}
	return &exp, nil
}

func (s *Store) Delete(ctx context.Context, dataset, name string) error {
	c, err := s.coll(dataset)
	if err != nil {
		return err
	}
	res, err := c.DeleteOne(ctx, bson.M{"_id": name})
	if err != nil {
		return err
	}
	if res.DeletedCount == 0 {
		return ErrNotFound
	}
	return nil
}

func (s *Store) Claim(ctx context.Context, dataset, name, agentID, device string) (*Experiment, error) {
	c, err := s.coll(dataset)
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC()
	res := c.FindOneAndUpdate(ctx,
		bson.M{
			"_id":    name,
			"status": bson.M{"$in": []Status{StatusPending, StatusFailed}},
		},
		bson.M{"$set": bson.M{
			"status":     StatusRunning,
			"agent_id":   agentID,
			"device":     device,
			"started_at": now,
			"updated_at": now,
			"error":      "",
		}},
		options.FindOneAndUpdate().SetReturnDocument(options.After))
	if err := res.Err(); err != nil {
		if errors.Is(err, mongo.ErrNoDocuments) {
			exists, gerr := s.Get(ctx, dataset, name)
			if gerr != nil {
				return nil, gerr
			}
			_ = exists
			return nil, ErrInvalidState
		}
		return nil, err
	}
	var exp Experiment
	if err := res.Decode(&exp); err != nil {
		return nil, err
	}
	return &exp, nil
}

func (s *Store) AppendCheckpoint(ctx context.Context, dataset, name string, cp Checkpoint) error {
	c, err := s.coll(dataset)
	if err != nil {
		return err
	}
	if cp.Timestamp.IsZero() {
		cp.Timestamp = time.Now().UTC()
	}
	res, err := c.UpdateOne(ctx,
		bson.M{"_id": name},
		bson.M{
			"$push": bson.M{"checkpoints": cp},
			"$set":  bson.M{"updated_at": time.Now().UTC()},
		})
	if err != nil {
		return err
	}
	if res.MatchedCount == 0 {
		return ErrNotFound
	}
	return nil
}

func (s *Store) Finish(ctx context.Context, dataset, name string, finalMetrics map[string]float64, durationS float64, checkpoints []Checkpoint) (*Experiment, error) {
	c, err := s.coll(dataset)
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC()
	for i := range checkpoints {
		if checkpoints[i].Timestamp.IsZero() {
			checkpoints[i].Timestamp = now
		}
	}
	update := bson.M{
		"$set": bson.M{
			"status":        StatusDone,
			"final_metrics": finalMetrics,
			"duration_s":    durationS,
			"finished_at":   now,
			"updated_at":    now,
			"error":         "",
		},
	}
	if len(checkpoints) > 0 {
		update["$push"] = bson.M{
			"checkpoints": bson.M{"$each": checkpoints},
		}
	}
	res := c.FindOneAndUpdate(ctx,
		bson.M{
			"_id":    name,
			"status": bson.M{"$ne": StatusDone},
		},
		update,
		options.FindOneAndUpdate().SetReturnDocument(options.After))
	if err := res.Err(); err != nil {
		if errors.Is(err, mongo.ErrNoDocuments) {
			if _, gerr := s.Get(ctx, dataset, name); gerr != nil {
				return nil, gerr
			}
			return nil, ErrInvalidState
		}
		return nil, err
	}
	var exp Experiment
	if err := res.Decode(&exp); err != nil {
		return nil, err
	}
	return &exp, nil
}

func (s *Store) Fail(ctx context.Context, dataset, name, errMsg string) (*Experiment, error) {
	c, err := s.coll(dataset)
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC()
	if len(errMsg) > 4000 {
		errMsg = errMsg[:4000]
	}
	res := c.FindOneAndUpdate(ctx,
		bson.M{"_id": name},
		bson.M{"$set": bson.M{
			"status":      StatusFailed,
			"error":       errMsg,
			"finished_at": now,
			"updated_at":  now,
		}},
		options.FindOneAndUpdate().SetReturnDocument(options.After))
	if err := res.Err(); err != nil {
		if errors.Is(err, mongo.ErrNoDocuments) {
			return nil, ErrNotFound
		}
		return nil, err
	}
	var exp Experiment
	if err := res.Decode(&exp); err != nil {
		return nil, err
	}
	return &exp, nil
}

func (s *Store) Release(ctx context.Context, dataset, name string) (*Experiment, error) {
	c, err := s.coll(dataset)
	if err != nil {
		return nil, err
	}
	res := c.FindOneAndUpdate(ctx,
		bson.M{"_id": name, "status": StatusRunning},
		bson.M{"$set": bson.M{
			"status":     StatusPending,
			"updated_at": time.Now().UTC(),
		}, "$unset": bson.M{
			"agent_id":   "",
			"started_at": "",
		}},
		options.FindOneAndUpdate().SetReturnDocument(options.After))
	if err := res.Err(); err != nil {
		if errors.Is(err, mongo.ErrNoDocuments) {
			return nil, ErrInvalidState
		}
		return nil, err
	}
	var exp Experiment
	if err := res.Decode(&exp); err != nil {
		return nil, err
	}
	return &exp, nil
}

// Summary returns counts per status + duration stats for `done` experiments
// + a rough ETA based on the current number of `running` experiments.
func (s *Store) Summary(ctx context.Context, dataset string) (*SummaryResponse, error) {
	c, err := s.coll(dataset)
	if err != nil {
		return nil, err
	}

	pipeline := []bson.M{
		{"$facet": bson.M{
			"by_status": []bson.M{
				{"$group": bson.M{"_id": "$status", "n": bson.M{"$sum": 1}}},
			},
			"done_dur": []bson.M{
				{"$match": bson.M{"status": StatusDone, "duration_s": bson.M{"$gt": 0}}},
				{"$group": bson.M{
					"_id":   nil,
					"count": bson.M{"$sum": 1},
					"avg":   bson.M{"$avg": "$duration_s"},
					"total": bson.M{"$sum": "$duration_s"},
				}},
			},
		}},
	}

	cur, err := c.Aggregate(ctx, pipeline)
	if err != nil {
		return nil, err
	}
	defer cur.Close(ctx)

	var raw []struct {
		ByStatus []struct {
			ID string `bson:"_id"`
			N  int    `bson:"n"`
		} `bson:"by_status"`
		DoneDur []struct {
			Count int     `bson:"count"`
			Avg   float64 `bson:"avg"`
			Total float64 `bson:"total"`
		} `bson:"done_dur"`
	}
	if err := cur.All(ctx, &raw); err != nil {
		return nil, err
	}

	out := &SummaryResponse{Dataset: dataset, Counts: map[string]int{}}
	if len(raw) == 0 {
		return out, nil
	}
	row := raw[0]

	var total int64
	for _, s := range row.ByStatus {
		out.Counts[s.ID] = s.N
		total += int64(s.N)
	}
	out.Total = total

	if len(row.DoneDur) == 1 {
		out.DoneCount = row.DoneDur[0].Count
		out.AvgDurationS = row.DoneDur[0].Avg
		out.TotalDurationS = row.DoneDur[0].Total
	}

	pending := out.Counts[string(StatusPending)] + out.Counts[string(StatusFailed)]
	running := out.Counts[string(StatusRunning)]
	if running < 1 {
		running = 1
	}
	if out.AvgDurationS > 0 && pending > 0 {
		out.EtaS = out.AvgDurationS * float64(pending) / float64(running)
		out.EtaMethod = "avg_done_duration * (pending+failed) / max(running, 1)"
	} else {
		out.EtaMethod = "n/a (no done duration data or no pending)"
	}

	// Lista de experimentos actualmente corriendo (para el resumen).
	runOpts := options.Find().
		SetProjection(bson.M{
			"_id":          1,
			"architecture": 1,
			"agent_id":     1,
			"device":       1,
			"started_at":   1,
		}).
		SetSort(bson.D{{Key: "started_at", Value: 1}})
	runCur, err := c.Find(ctx, bson.M{"status": StatusRunning}, runOpts)
	if err != nil {
		return nil, err
	}
	defer runCur.Close(ctx)
	out.Running = []RunningExperiment{}
	now := time.Now().UTC()
	for runCur.Next(ctx) {
		var r struct {
			ID           string     `bson:"_id"`
			Architecture string     `bson:"architecture"`
			AgentID      string     `bson:"agent_id"`
			Device       string     `bson:"device"`
			StartedAt    *time.Time `bson:"started_at"`
		}
		if err := runCur.Decode(&r); err != nil {
			return nil, err
		}
		elapsed := 0.0
		if r.StartedAt != nil {
			elapsed = now.Sub(*r.StartedAt).Seconds()
		}
		out.Running = append(out.Running, RunningExperiment{
			ExpName:      r.ID,
			Architecture: r.Architecture,
			AgentID:      r.AgentID,
			Device:       r.Device,
			StartedAt:    r.StartedAt,
			ElapsedS:     elapsed,
		})
	}
	return out, nil
}

func (s *Store) Stats(ctx context.Context, dataset string) (*StatsResponse, error) {
	c, err := s.coll(dataset)
	if err != nil {
		return nil, err
	}
	pipeline := []bson.M{
		{"$group": bson.M{"_id": "$status", "n": bson.M{"$sum": 1}}},
	}
	cur, err := c.Aggregate(ctx, pipeline)
	if err != nil {
		return nil, err
	}
	defer cur.Close(ctx)

	stats := &StatsResponse{Dataset: dataset, Counts: map[string]int{}}
	var total int64
	for cur.Next(ctx) {
		var row struct {
			ID string `bson:"_id"`
			N  int    `bson:"n"`
		}
		if err := cur.Decode(&row); err != nil {
			return nil, err
		}
		stats.Counts[row.ID] = row.N
		total += int64(row.N)
	}
	stats.Total = total
	return stats, nil
}

func (s *Store) ListDatasets(ctx context.Context) ([]string, error) {
	dbs, err := s.client.ListDatabaseNames(ctx, bson.M{})
	if err != nil {
		return nil, err
	}
	var out []string
	for _, d := range dbs {
		if strings.HasPrefix(d, s.dbPrefix) {
			out = append(out, strings.TrimPrefix(d, s.dbPrefix))
		}
	}
	return out, nil
}
