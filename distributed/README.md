# Sistema distribuido de experimentos

Centraliza las métricas de los experimentos de la tesis. Un **backend en Go** con
**MongoDB** acepta resultados vía REST; **agentes en Python** corren en cada
servidor con GPU, ejecutan los experimentos y envían las métricas.

```
┌────────────┐ HTTP  ┌─────────────────┐  ┌─────────────────┐
│  agent-A   ├──────▶│  backend (Go)   ├─▶│  MongoDB        │
│  cuda:0..N │       │  CRUD + claim   │  │  experiments_*  │
└────────────┘       └─────────────────┘  └─────────────────┘
       ▲                      ▲
       │                      │
┌────────────┐                │
│  agent-B   ├────────────────┘
│  cuda:0..N │
└────────────┘
```

## Estructura

```
distributed/
  backend/    Backend Go con CRUD sobre MongoDB
  agent/      Agente Python que ejecuta experimentos y reporta métricas
```

## Modelo de datos

- Una **base de datos por dataset** (`experiments_ai4i`, `experiments_nps`, ...).
- Una colección `experiments` por base de datos.
- **Primary key** (`_id`) = nombre canónico exhaustivo del experimento.
  Dos configuraciones que difieran en cualquier hiperparámetro generan nombres
  distintos, por lo que **nunca se sobrescribe nada**.

Ejemplo de PK:
```
Transformer_ai4i_ph10_h64_nl1_w200_lr0.0001_d0.2_s42_ep1_sgd_layernorm_StaticFocal_a0.25_g2.0
```

Campos codificados en la PK: `architecture`, `dataset`, `past_history`,
`hidden_dim`/`hidden_dims`, `num_layers`, `window_size`, `lr`, `dropout`,
`seed`, `epochs`, `optimizer`, `normalization`, tipo de pérdida + sus
hiperparámetros, y sufijo `_bidir` si procede.

## Ciclo de vida de un experimento

```
pending ──claim──▶ running ──finish──▶ done
   ▲                  │
   └────release───────┤
                      └──fail──▶ failed ──claim──▶ running ...
```

- `done` es **terminal**: el endpoint `finish` rechaza con 409 si ya estaba `done`.
- `failed` es reintentar­able (`claim` lo vuelve a poner en `running`).
- `release` devuelve a `pending` un experimento `running` huérfano.

## Arrancar el backend

```bash
cd distributed/backend
# Genera una API key y pásala al backend (y al agente).
export API_KEY="$(openssl rand -hex 32)"

docker compose up -d        # arranca MongoDB + backend en :8080
# o, sin Docker:
API_KEY=$API_KEY MONGO_URI=mongodb://localhost:27017 go run .
```

Verificación:
```bash
curl http://localhost:8080/api/v1/health
# {"status":"ok"}    (health no requiere auth)

curl -H "X-API-Key: $API_KEY" http://localhost:8080/api/v1/datasets
```

Si `API_KEY` queda vacío el servidor arranca sin autenticación (modo dev),
imprimiendo un aviso en logs.

## Arrancar un agente

```bash
cd distributed/agent
pip install -r requirements.txt
cp config.example.yaml config.yaml      # editar backend_url, devices, grid
python agent.py --config config.yaml
```

Comandos útiles del agente:
```bash
python agent.py --register-only   # solo registra el grid en el backend
python agent.py --status          # imprime stats del backend para el dataset
python agent.py --no-register     # corre directamente sin volver a registrar
```

## Endpoints (resumen)

Todas las rutas (excepto `/health`) requieren `X-API-Key: <key>` si el backend
arrancó con `API_KEY` no vacío. También se acepta `Authorization: Bearer <key>`.

| Método  | Ruta                                                              | Uso |
|---------|-------------------------------------------------------------------|-----|
| GET     | `/api/v1/health`                                                  | Health (sin auth) |
| GET     | `/api/v1/datasets`                                                | Lista DBs detectadas |
| GET     | `/api/v1/datasets/{ds}/stats`                                     | Conteos por estado |
| GET     | `/api/v1/datasets/{ds}/results.csv`                               | **CSV equivalente al `final_results.csv` antiguo, por dataset** |
| GET     | `/api/v1/datasets/{ds}/experiments?status=pending&limit=...`      | Listar |
| POST    | `/api/v1/datasets/{ds}/experiments`                               | Crear (409 si existe) |
| POST    | `/api/v1/datasets/{ds}/experiments/bulk`                          | Crear N (idempotente) |
| GET     | `/api/v1/datasets/{ds}/experiments/{name}`                        | Obtener uno |
| PUT     | `/api/v1/datasets/{ds}/experiments/{name}`                        | Reemplazar (409 si done) |
| PATCH   | `/api/v1/datasets/{ds}/experiments/{name}`                        | Actualización parcial |
| DELETE  | `/api/v1/datasets/{ds}/experiments/{name}`                        | Borrar |
| POST    | `/api/v1/datasets/{ds}/experiments/{name}/claim`                  | Reservar (atómico) |
| POST    | `/api/v1/datasets/{ds}/experiments/{name}/checkpoints`            | Añadir checkpoint |
| POST    | `/api/v1/datasets/{ds}/experiments/{name}/finish`                 | Marcar `done` |
| POST    | `/api/v1/datasets/{ds}/experiments/{name}/fail`                   | Marcar `failed` |
| POST    | `/api/v1/datasets/{ds}/experiments/{name}/release`                | Devolver a `pending` |

### Exportación CSV (por dataset, con filtros)

```
GET /api/v1/datasets/{dataset}/results.csv
    ?status=done|running|failed|pending|all   (default: done)
    &architecture=Transformer
    &agent_id=gpu-host-1
    &device=cuda:1
    &loss_type=StaticFocal
    &limit=1000&offset=0
```

Una fila por experimento. Columnas base + `cfg.<...>` (config aplanado) +
`metric.<...>` (métricas finales). El archivo se llama
`results_<dataset>_<status>.csv` para que sea trivial distinguir qué dataset
se descargó.

Desde el agente (CLI):

```bash
python distributed/agent/download_results.py \
    --config distributed/agent/config.yaml \
    --dataset ai4i \
    --status done \
    --architecture Transformer \
    --out results_ai4i_transformer.csv
```

Con `curl`:

```bash
curl -H "X-API-Key: $API_KEY" \
     "http://backend:8080/api/v1/datasets/ai4i/results.csv?architecture=LSTM" \
     -o results_ai4i_lstm.csv
```

## Multi-agente

Varios agentes pueden apuntar al mismo backend. Cada agente puede ser:

- **Productor + consumidor** (defecto): si trae `model:`/`loss:`, los
  registra (idempotente) y a la vez consume cualquier pending del dataset.
- **Consumidor puro**: sin `model:`, solo reclama trabajo que registraron
  otros agentes.

El endpoint `claim` es atómico (`findOneAndUpdate` en MongoDB), así que dos
agentes nunca procesan el mismo experimento. El campo `device` en el config
es una **preferencia suave**: el worker de `cuda:1` toma primero los configs
etiquetados con `cuda:1`, pero si se queda sin esos también consume otros.

Cada agente mantiene además un **espejo SQLite local** con todo lo que ha
visto (registros, claims, checkpoints, finales, errores) — no depende del
backend para su propia trazabilidad.
