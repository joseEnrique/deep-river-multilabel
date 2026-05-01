# Backend (Go + MongoDB)

REST API que centraliza los experimentos. Una base de datos por dataset,
colección `experiments`, `_id` = nombre exhaustivo del experimento.

## Variables de entorno

| Variable      | Defecto                          |
|---------------|----------------------------------|
| `BACKEND_ADDR`| `:8080`                          |
| `MONGO_URI`   | `mongodb://localhost:27017`      |
| `DB_PREFIX`   | `experiments_` (DB = `<prefix><dataset>`) |
| `API_KEY`     | vacío (dev sin auth) — si lo pones, todas las rutas exigen `X-API-Key` o `Authorization: Bearer <key>`, salvo `/health` |

## Desarrollo

```bash
go mod tidy
go run .
```

## Docker

```bash
docker compose up -d
```

Levanta MongoDB (volumen persistente `mongo_data`) y el backend en `:8080`.

## Garantías de no-sobrescritura

- `POST /experiments` → 409 si el `exp_name` ya existe.
- `PUT  /experiments/{name}` → 409 si el experimento está `done`.
- `POST /experiments/{name}/finish` → 409 si ya estaba `done` (no rebaja
  métricas finales).
- `POST /experiments/{name}/claim` → solo cambia `pending|failed → running`,
  atómico con `findOneAndUpdate`. Nunca dos agentes reclaman el mismo experimento.
- Los checkpoints siempre se **añaden** (`$push`), nunca se reemplazan.

## Exportación CSV

`GET /datasets/{dataset}/results.csv` devuelve una fila por experimento,
**siempre por dataset** (la URL distingue el dataset y el filename incluye
también el dataset y el status). Soporta filtros por query string:
`status`, `architecture`, `agent_id`, `device`, `loss_type`, `limit`, `offset`.
