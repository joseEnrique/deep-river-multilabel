# Agent

Cualquier agente combina dos roles a la vez:

- **Productor**: si tiene un bloque `model:`/`loss:`, lo expande y lo registra
  en el backend de forma **idempotente** (los duplicados se ignoran vía 409 y
  vía `INSERT OR IGNORE` en el SQLite local — nunca se sobrescribe nada).
- **Consumidor**: cada GPU configurada arranca un dispatcher que reclama
  cualquier `pending`/`failed` del dataset. Por defecto reclama también
  trabajo registrado por **otros agentes**. Pon `consume_any: false` para
  restringirlo a su propio grid.


Un agente sin `model:` arranca como **consumidor puro** y solo procesa lo que
ya hay en el backend.

Cada checkpoint y métrica final se envía al backend vía HTTP **y** se escribe
en una réplica local SQLite (`local_<dataset>.db`) por si el backend cae.

## Instalación

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Edita `config.yaml`:
- `backend_url`: URL del backend Go.
- `agent_id`: identificador único (deja `null` para usar el hostname).
- `dataset`: dataset que ejecutará este agente (la DB destino será
  `experiments_<dataset>`).
- `devices`: lista de GPUs que esta máquina/agente debe usar (un proceso por GPU).
- Bloque `model:` y `loss:`: mismo formato que `experiment_system/config.yaml`.

## Uso

```bash
python agent.py --config config.yaml
python agent.py --register-only      # registra grid en backend y sale
python agent.py --consume-only       # no registra, solo consume
python agent.py --status             # stats backend + SQLite local

# Descargar CSV con resultados (filtrable):
python download_results.py --config config.yaml --status done
python download_results.py --config config.yaml --architecture Transformer
python download_results.py --config config.yaml --dataset ai4i --status all
```

## Migrar datos del experiment_system antiguo

Para volcar el SQLite legacy + (opcionalmente) los CSV de checkpoints al
backend nuevo:

```bash
# Solo métricas finales (no hay checkpoints — el SQLite antiguo no los tenía)
python migrate_old.py \
    --config config.yaml \
    --source-db ../../experiment_system/experiments.db

# Métricas finales + checkpoints leídos de los CSV de results/
python migrate_old.py \
    --config config.yaml \
    --source-db ../../experiment_system/experiments.db \
    --checkpoints-dir ../../experiment_system/results

# Pre-visualizar sin tocar el backend
python migrate_old.py --source-db ... --dry-run
```

Es **idempotente**: si ya importaste, las filas con `status=done` en el
SQLite local se saltan automáticamente. El backend también devuelve 409 para
duplicados, así que es seguro relanzarlo. Los experimentos sin checkpoints
(o con su CSV ausente) se importan igual — quedan como `done` con
`final_metrics` y array de checkpoints vacío.

## Autenticación

Pon `api_key:` en el config (o exporta `BACKEND_API_KEY`) y todos los HTTP
salen con `X-API-Key: <key>`. Si el backend está sin clave (dev), deja
`api_key: null` y se omite el header.

## Reparto multi-agente

- **Quién popula la DB**: cualquier agente con grid hace `bulk_create` al
  arrancar. Es idempotente (409 = ya existía → se salta).
- **Quién ejecuta**: por defecto (`consume_any: true`) cualquier agente
  procesa cualquier `pending` del dataset, sin importar quién lo registró.
- **Routing por GPU**: si un config tiene `device: "cuda:1"`, el worker de
  `cuda:1` lo prefiere primero. Es preferencia suave: si un worker de
  `cuda:0` se queda sin trabajo etiquetado, también puede tomarlo.
- **Atomicidad**: el endpoint `claim` usa `findOneAndUpdate` en MongoDB, así
  que dos workers nunca pillan el mismo experimento.

## Tolerancia a fallos

- **SQLite local** (`local_<dataset>.db`): cada upsert/claim/checkpoint/
  finish/fail se persiste en local. Si el backend cae, los checkpoints
  siguen registrándose en local; el siguiente push se reintenta. Permite
  además análisis offline igual que `experiment_system/db.py`.
- **Backend autoritativo**: el `claim` y la lista de `pending` siempre
  vienen del backend, así que el orden global se mantiene aunque un agente
  se quede aislado un rato.
- **Si un worker muere a mitad** → estado queda `running` en backend. Usa
  `POST /experiments/{name}/release` para devolverlo a `pending`. Los
  `failed` se reintentan automáticamente en el siguiente ciclo.
- **`done` es terminal**: `claim` solo acepta `pending|failed`.
- **Reintentos HTTP**: back-off exponencial sobre 502/503/504.
