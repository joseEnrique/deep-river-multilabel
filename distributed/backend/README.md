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
| `MAX_IN_FLIGHT`   | `32` — tope de peticiones servidas a la vez |
| `REQUEST_BACKLOG` | `256` — cuántas esperan antes de descargar carga con 429 |

## Protección contra saturación

Mongo nunca recibe más carga de la que se le fija, vengan los agentes que
vengan. Tres barreras encadenadas:

1. **Tope de concurrencia HTTP** (`MAX_IN_FLIGHT`, por defecto 32). Nunca hay
   más de N peticiones dentro a la vez. Las demás esperan en el backlog; si
   este se llena, el servidor responde **429 + `Retry-After`**.
2. **El agente reintenta el 429** con retroceso, respetando `Retry-After`. La
   carga se descarga en vez de acumularse. *(Si tocas una de las dos partes,
   toca la otra: `TestShedStatusIsRetryableByAgent` fija ese contrato.)*
3. **Pool de conexiones acotado** (50 máx., 5 mín.) y `opTimeout` de 55 s por
   operación, por debajo del timeout HTTP de 60 s, para que una consulta
   desbocada muera server-side en lugar de retener una conexión.

Del lado del agente, además, hay **retroceso exponencial cuando no hay
trabajo** (hasta 120 s): con el grid terminado, los workers ociosos dejan de
preguntar cada `poll_interval` en vez de machacar el backend sin motivo.

## Índices

Se crean al arrancar, **en segundo plano** (el servidor atiende desde el
primer segundo) y un dataset cada vez. Son 16, uno por forma de consulta:
las cuatro variantes de `pick_order` de `claim-next` —con y sin afinidad de
device—, más listado/stats/summary, filtros del CSV y `cube/top`.

La dirección importa: Mongo solo sirve un sort desde un índice si coincide con
el patrón **o es su inverso exacto**, por eso `speed_desc` (`epochs` desc pero
`hidden_dim` asc) necesita índice propio. `TestEveryPickOrderIsIndexBacked`
comprueba con `explain` que ninguna combinación cae en un sort en memoria —
que no es solo lento: Mongo lo corta a 32 MB y **falla** cuando la cola crece.

Medido sobre 60.000 documentos: **2,7 s de construcción y 6,4 MB de índices**
para 12,3 MB de datos (0,52×). Extrapolado a los 240.000 de `alpi`, unos 11 s.

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

## Tests

Los tests de integración necesitan un Mongo desechable. Sin él **se saltan**
(no fallan), así que `go test ./...` siempre es seguro de ejecutar:

```bash
docker run -d --name mongo-test -p 27099:27017 mongo:7
go test ./... -count=1
docker rm -f mongo-test
```

Otro puerto/host: `TEST_MONGO_URI=mongodb://host:port go test ./...`

Cobertura: `go test ./... -coverprofile=c.out && go tool cover -func=c.out`

Del lado del agente: `python -m pytest distributed/agent/test_claim_next.py`
(sin dependencias externas — levanta un backend HTTP de mentira).

## Garantías de no-sobrescritura

- `POST /experiments` → 409 si el `exp_name` ya existe.
- `PUT  /experiments/{name}` → 409 si el experimento está `done`.
- `POST /experiments/{name}/finish` → 409 si ya estaba `done` (no rebaja
  métricas finales).
- `POST /experiments/{name}/claim` → solo cambia `pending|failed → running`,
  atómico con `findOneAndUpdate`. Nunca dos agentes reclaman el mismo experimento.
- `POST /claim-next` → elige **y** reclama el mejor pendiente según `sort`, en
  una sola operación atómica. Es lo que usan los agentes para pedir trabajo:
  una petición pequeña por lanzamiento en vez de descargarse la cola entera
  para ordenarla en local. Los índices que necesita se crean al arrancar.
- Los checkpoints siempre se **añaden** (`$push`), nunca se reemplazan.

## Exportación CSV

`GET /datasets/{dataset}/results.csv` devuelve una fila por experimento,
**siempre por dataset** (la URL distingue el dataset y el filename incluye
también el dataset y el status). Soporta filtros por query string:
`status`, `architecture`, `agent_id`, `device`, `loss_type`, `limit`, `offset`.
