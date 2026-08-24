# Backend API Reference

Go + chi + MongoDB. One database per dataset (`<DB_PREFIX><dataset>`), one
collection `experiments`. Document `_id` is the experiment name.

All paths are prefixed with `/api/v1`. If `API_KEY` is set on the server,
every route except `/health` requires `X-API-Key: <key>` or
`Authorization: Bearer <key>`.

## Conventions

- Path param `{dataset}` is `[A-Za-z0-9_-]{1,64}`. Invalid names → 400.
- All responses are JSON (UTF-8) unless noted.
- Errors: `{ "error": "<msg>", "code": "<optional>", "details": "<optional>" }`.
- Status codes:
  - `404` — experiment / resource not found
  - `409` — duplicate, invalid state transition (e.g. claim on `done`)
  - `400` — bad input
  - `503` — Mongo down (health only)

## Experiment document shape

```jsonc
{
  "exp_name": "lstm_alpi_ws1_ph10_lr0.5",
  "architecture": "LSTM",          // optional
  "dataset": "ALPI",
  "config": { /* arbitrary nested JSON */ },
  "status": "pending|running|done|failed",
  "agent_id": "node-3",            // when claimed
  "device": "cuda:0",              // when claimed
  "started_at": "2026-05-01T12:00:00Z",
  "finished_at": "2026-05-01T12:34:56Z",
  "checkpoints": [
    { "step": 1000, "elapsed_s": 12.3, "metrics": { "loss": 0.42 },
      "timestamp": "..." }
  ],
  "final_metrics": { "exact_match": 0.7978, "macro_f1": 0.7218 },
  "duration_s": 2096.4,
  "error": "",
  "created_at": "...",
  "updated_at": "..."
}
```

---

# Core endpoints

## `GET /api/v1/health`

Ping Mongo.

- `200` → `{ "status": "ok" }`
- `503` → `{ "status": "down", "error": "..." }`

## `GET /api/v1/datasets`

List dataset names discovered by scanning DBs that match the configured
prefix.

- `200` → `{ "datasets": ["ALPI", "NPS", "AI4I"] }`

---

# Per-dataset endpoints

Base: `/api/v1/datasets/{dataset}`

## `GET .../stats`

Counts per status.

- `200` → `{ "dataset": "ALPI", "total": 1234, "counts": { "done": 1000, "running": 4, "pending": 230 } }`

## `GET .../summary`

Counts + duration stats over `done` experiments + a rough ETA + the
list of currently `running` experiments.

- `200` →
  ```jsonc
  {
    "dataset": "ALPI",
    "total": 1234,
    "counts": { "...": 0 },
    "done_count": 1000,
    "avg_duration_s": 1800.5,
    "total_duration_s": 1800500,
    "eta_s": 12345,
    "eta_method": "avg_done_duration * (pending+failed) / max(running, 1)",
    "running": [
      { "exp_name": "...", "architecture": "...", "agent_id": "...",
        "device": "...", "started_at": "...", "elapsed_s": 12.3 }
    ]
  }
  ```

## `GET .../results.csv`

Flat CSV export, one row per experiment with `config` and `final_metrics`
flattened to dotted columns (`cfg.lr`, `metric.exact_match`, ...).

Query params (all optional):

| Param | Default | Notes |
|---|---|---|
| `status` | `done` | `all` disables the filter |
| `architecture` | — | exact match |
| `agent_id` | — | exact match |
| `device` | — | exact match |
| `loss_type` | — | matches `config.loss.type` |
| `limit` / `offset` | unbounded | pagination |

Returns `text/csv` with `Content-Disposition: attachment; filename="results_<dataset>_<status>.csv"`.

The export **streams**: rows are written as the cursor yields them (sorted by
`exp_name`), `checkpoints` are projected out server-side, and the response
starts within seconds regardless of size. It is exempt from the global 60s
request timeout and carries its own 30-minute deadline instead. At most **2**
exports run concurrently; a third gets `429` with `Retry-After` rather than
queueing behind them.

## `GET .../experiments`

List experiments.

| Param | Default |
|---|---|
| `status` | — (all statuses) |
| `limit` | `1000` |
| `offset` | `0` |

- `200` → `{ "dataset": "...", "count": N, "experiments": [ ... ] }`

## `POST .../experiments`

Create one experiment. Body:

```json
{ "exp_name": "...", "architecture": "...", "config": { ... } }
```

- `201` → the created document
- `409` if `exp_name` already exists

## `POST .../claim-next`

Pick **and** claim the best pending/failed experiment in a single atomic
operation. This is what agents use to pull work: one small request per
launch instead of downloading the whole pending queue to sort it locally.

Because it is a `findOneAndUpdate`, two agents can never receive the same
document — there is no claim race and nothing to retry.

Body:

```jsonc
{
  "agent_id": "megatron",              // or the X-Agent-ID header
  "device": "cuda:0",                  // recorded on the claimed experiment
  "sort": "config.epochs:asc,config.hidden_dim:asc",   // optional
  "prefer_device": "cuda:0"            // optional soft affinity
}
```

- `sort` is a comma-separated list of `path[:asc|desc]`. Direction defaults
  to `asc`. Paths are restricted to `[A-Za-z0-9_.]` — anything else is a
  `400`, so an operator cannot be smuggled into the sort document.
- The default agent sort is `config.epochs:asc,compute_score:asc`, which
  yields **fast+SMALL → fast+MEDIUM → fast+LARGE → slow+SMALL → …**.
  The size key is `compute_score`, not `config.hidden_dim`: for a Transformer
  the attention term dominates, so `hd=32, ws=500` scores 126,024 (LARGE)
  while an LSTM `hd=128, nl=2` scores 32,768 (MEDIUM) — ordering by
  `hidden_dim` would put the LARGE one first.
- `prefer_device` restricts the claim to experiments whose registered
  `config.device` matches. The agent calls with it first and retries without
  it on `404`, which reproduces "prefer this GPU, but take anything".

Responses:

- `200` → the claimed experiment, already in `running`
- `404` → nothing claimable (the agent's "queue empty" signal)
- `400` → missing `agent_id`, malformed JSON, or an invalid sort key

**Indexes.** The sorted claim needs `{status:1, "config.epochs":1, ...}`,
otherwise Mongo does a blocking in-memory sort (capped at 32 MB) over the
whole pending queue. The server creates the indexes it needs at startup.

## `POST .../backfill-scores`

Computes `compute_score` / `size_tier` for documents registered before those
fields existed. One server-side pipeline update — no per-document round
trips — and idempotent.

| Param | Default | Notes |
|---|---|---|
| `all` | `false` | `true` recomputes every document instead of only the ones missing a score |

- `200` → `{ "dataset": "...", "matched": N, "modified": M, "still_missing": 0 }`

**Run this once after upgrading.** A missing field sorts *first* in an
ascending Mongo sort, so any document without a `compute_score` jumps to the
head of the queue regardless of how big or slow it is.

From the agent: `python agent.py --config <cfg> --backfill-scores`

## `POST .../experiments/bulk`

Insert many. Body: `{ "experiments": [ { exp_name, architecture, config }, ... ] }`.
Uses unordered `insertMany` — duplicates are counted as skipped, not fatal.

- `200` → `{ "inserted": N, "skipped": M, "total": N+M }`

---

# Per-experiment endpoints

Base: `/api/v1/datasets/{dataset}/experiments/{name}`

## `GET .../`

Fetch a single experiment. `404` if missing.

## `PUT .../`

Replace the whole document. `409` if the experiment is already `done`
(we never overwrite final results).

## `PATCH .../`

Partial update. The keys `_id`, `exp_name`, `dataset`, `created_at` are
stripped from the patch automatically. `updated_at` is refreshed.

## `DELETE .../`

Hard delete. `204` on success, `404` if missing.

## `POST .../claim`

Atomic claim: only moves `pending|failed → running`. Two agents cannot
claim the same experiment.

Body: `{ "agent_id": "...", "device": "cuda:0" }` (or pass `X-Agent-ID` header).

- `200` → updated experiment
- `409` if not in a claimable state

## `POST .../checkpoints`

Append a checkpoint (`$push`, never overwrites). Body:

```json
{ "step": 1000, "elapsed_s": 12.3, "metrics": { "loss": 0.42 } }
```

## `POST .../finish`

Mark as `done`, set `final_metrics`, `duration_s`, `finished_at`. Optionally
push extra checkpoints in the same call.

Body:
```json
{ "final_metrics": { "exact_match": 0.79 }, "duration_s": 1234.5,
  "checkpoints": [ ... ] }
```

- `409` if already `done` (final metrics are never lowered).

## `POST .../fail`

Mark as `failed` with an error message (truncated to 4000 chars).

Body: `{ "error": "..." }`

## `POST .../release`

Move `running → pending` and unset `agent_id` / `started_at`. Used when
an agent crashes or hands the experiment back.

- `409` if not currently `running`.

---

# Statistics endpoints (cube)

All read-only. They aggregate over the existing schema (no writes, no
new fields). Useful for asking the database "what's the best model
according to X parameters?" without dumping the full CSV.

Base: `/api/v1/datasets/{dataset}/cube`

## Shared conventions for cube endpoints

- `metric=<name>` resolves to `final_metrics.<name>`. If you pass a value
  already prefixed with `final_metrics.`, it is used as-is.
- `by=k1,k2` is a comma-separated list of BSON paths. Anything goes:
  `config.lr`, `config.loss.type`, `architecture`, `agent_id`, `device`.
- `where=k=v,k2=v2` is a comma-separated equality filter. Values are
  parsed as `float64` if they parse, then `bool` (`true`/`false`),
  otherwise kept as string. Applied as `$match` before grouping.
- `status` defaults to `done`. Use `status=all` to disable the filter.
- `order=field:asc` / `order=field:desc` — direction is optional, default
  is `desc`. For `groupby`, `field` can be `max`, `min`, `mean`, `std`,
  `count`, or `p50` / `p90` / `p95` / `p99`.
- Aggregations available: `max`, `min`, `mean` (alias `avg`), `std`
  (alias `stddev`), `count`, `p50`, `p90`, `p95`, `p99`. Percentiles are
  computed in the backend (Mongo collects values with `$push`).

## `GET .../cube/metrics`

Discover available metric names.

| Param | Default |
|---|---|
| `status` | `done` |

Returns one row per key found in `final_metrics`:

```json
{
  "dataset": "ALPI",
  "metrics": [
    { "name": "exact_match", "count": 1000, "min": 0.51, "max": 0.83, "mean": 0.72 },
    { "name": "macro_f1",   "count": 1000, "min": 0.30, "max": 0.78, "mean": 0.61 }
  ]
}
```

## `GET .../cube/params`

Discover available config keys (dotted, flattened) plus useful top-level
fields. Uses a sample of documents (cheap, no full scan).

| Param | Default |
|---|---|
| `status` | `all` |
| `sample` | `1000` (max docs to scan) |

```json
{
  "dataset": "ALPI",
  "sample": 1000,
  "params": [
    { "key": "config.lr", "count": 980 },
    { "key": "config.loss.type", "count": 1000 },
    { "key": "config.window_size", "count": 1000 }
  ],
  "top_level": [
    { "key": "architecture", "count": 1000 },
    { "key": "device", "count": 1000 }
  ]
}
```

## `GET .../cube/params/values`

Distinct values of one parameter with optional per-value metric aggregates.

| Param | Required | Notes |
|---|---|---|
| `key` | yes | any BSON path, e.g. `config.lr` |
| `metric` | no | if set, returns `max` / `min` / `mean` per value |
| `where` | no | extra filter |
| `status` | no, `done` | |

```json
{
  "dataset": "ALPI",
  "key": "config.window_size",
  "metric": "exact_match",
  "values": [
    { "value": 1, "count": 250, "max": 0.83, "min": 0.61, "mean": 0.72 },
    { "value": 4, "count": 250, "max": 0.78, "min": 0.55, "mean": 0.68 }
  ]
}
```

## `GET .../cube/top`

Top-N experiments by a final metric.

| Param | Required | Default |
|---|---|---|
| `metric` | yes | — |
| `limit` | no | `10` |
| `order` | no | `desc` |
| `where` | no | — |
| `status` | no | `done` |

Example: top 5 by `exact_match` with `window_size=1` and `lr=0.5`:

```
GET .../cube/top?metric=exact_match&limit=5&where=config.window_size=1,config.lr=0.5
```

```jsonc
{
  "dataset": "ALPI",
  "metric": "exact_match",
  "order": "desc",
  "count": 5,
  "experiments": [ /* full Experiment documents */ ]
}
```

## `GET .../cube/groupby`

The OLAP-style data cube. Group by N keys, aggregate over a metric.

| Param | Required | Default |
|---|---|---|
| `by` | yes | comma-separated paths |
| `metric` | yes | — |
| `agg` | no | `max,mean,count` |
| `order` | no | first non-count agg, desc |
| `limit` | no | `50` |
| `where` | no | — |
| `status` | no | `done` |

Example: best `exact_match` per `(window_size, lr)` combination, ordered
by max desc, top 20:

```
GET .../cube/groupby?by=config.window_size,config.lr&metric=exact_match
                   &agg=max,mean,std,count&order=max:desc&limit=20
```

```jsonc
{
  "dataset": "ALPI",
  "by": ["config.window_size", "config.lr"],
  "metric": "exact_match",
  "agg": ["max", "mean", "std", "count"],
  "order": "max",
  "count": 20,
  "groups": [
    {
      "group":   { "config.window_size": 1, "config.lr": 0.5 },
      "metrics": { "max": 0.8312, "mean": 0.7891, "std": 0.012, "count": 4 }
    }
  ]
}
```

Sorting by percentile (`order=p95:desc`) is supported but ranks
application-side after the group reduction.

## `GET .../cube/best-per`

For each distinct value of one key, return the single best full experiment
ranked by `metric`.

| Param | Required | Default |
|---|---|---|
| `by` | yes | one path (no commas) |
| `metric` | yes | — |
| `order` | no | `desc` |
| `limit` | no | `50` |
| `where` | no | — |
| `status` | no | `done` |

Example: best LSTM/MLP/etc. by `exact_match`:

```
GET .../cube/best-per?by=architecture&metric=exact_match
```

```jsonc
{
  "dataset": "ALPI",
  "by": "architecture",
  "metric": "exact_match",
  "count": 3,
  "groups": [
    { "value": "LSTM", "count": 412, "best": { /* full Experiment */ } },
    { "value": "MLP",  "count": 200, "best": { /* ... */ } }
  ]
}
```

## `GET .../cube/distribution`

Histogram of a metric over the filtered set. Boundaries are computed from
the observed min/max and split into equal-width bins.

| Param | Required | Default |
|---|---|---|
| `metric` | yes | — |
| `bins` | no | `10` |
| `where` | no | — |
| `status` | no | `done` |

```jsonc
{
  "dataset": "ALPI",
  "metric": "exact_match",
  "min": 0.51,
  "max": 0.83,
  "mean": 0.72,
  "std": 0.05,
  "count": 1000,
  "bins": [
    { "lo": 0.51, "hi": 0.54, "count": 12 },
    { "lo": 0.54, "hi": 0.57, "count": 45 }
  ]
}
```

---

# Quick-start examples

```bash
BASE=http://localhost:8080/api/v1
DS=ALPI

# What metrics exist on this dataset?
curl "$BASE/datasets/$DS/cube/metrics"

# What config keys are tunable?
curl "$BASE/datasets/$DS/cube/params"

# Top 10 by exact_match
curl "$BASE/datasets/$DS/cube/top?metric=exact_match&limit=10"

# Best macro_f1 for each loss type
curl "$BASE/datasets/$DS/cube/best-per?by=config.loss.type&metric=macro_f1"

# Heat-map: max exact_match per (window_size, lr)
curl "$BASE/datasets/$DS/cube/groupby?by=config.window_size,config.lr&metric=exact_match&agg=max,count&order=max:desc"

# Distribution of exact_match restricted to window_size=1
curl "$BASE/datasets/$DS/cube/distribution?metric=exact_match&where=config.window_size=1&bins=20"
```
