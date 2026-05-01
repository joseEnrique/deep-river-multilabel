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

func NewMongoStore(ctx context.Context, uri, dbPrefix string) (*Store, error) {
	connectCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	client, err := mongo.Connect(connectCtx, options.Client().ApplyURI(uri))
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
