# Backend — catálogo de endpoints GET (consulta)

Solo lectura. Para cada endpoint: **qué hace** y **cuándo usarlo**. Detalles de params/respuesta en `API.md`.

Prefijo común: `/api/v1`. Si `API_KEY` está seteada, mandar header `X-API-Key: <key>`.

---

## Salud y descubrimiento

### `GET /health`
Comprueba si el backend está vivo y Mongo responde.
**Cuándo**: monitorización, smoke tests.

### `GET /datasets`
Lista los datasets disponibles (ALPI, NPS, AI4I, ...).
**Cuándo**: para saber sobre qué datasets puedes consultar sin tener que mirar la config.

---

## Estado global del dataset

### `GET /datasets/{ds}/stats`
Cuenta cuántos experimentos hay en cada estado (`pending`/`running`/`done`/`failed`).
**Cuándo**: el "¿cómo va lo mío?" rápido — un solo número por estado.

### `GET /datasets/{ds}/summary`
Lo de `/stats` + duración media/total + ETA estimado + lista de experimentos corriendo ahora mismo (con agent, device, tiempo transcurrido).
**Cuándo**: dashboard de progreso. Para saber cuánto queda y qué está haciendo cada GPU.

### `GET /datasets/{ds}/results.csv`
Descarga TODO el dataset como CSV plano, una fila por experimento, con la `config` y las `final_metrics` ya aplanadas a columnas.
**Cuándo**: análisis offline en pandas / Excel / R. Filtros opcionales por arquitectura, agente, device, tipo de loss.

---

## Lectura de experimentos

### `GET /datasets/{ds}/experiments`
Lista experimentos del dataset (paginable, filtrable por estado).
**Cuándo**: explorar qué hay. Si solo quieres métricas/parámetros, los endpoints de cubo son más eficientes.

### `GET /datasets/{ds}/experiments/{name}`
Trae el documento entero de un experimento (config, status, checkpoints, métricas, etc.).
**Cuándo**: debug, inspeccionar uno concreto.

---

## Cubo estadístico (todos GET, solo lectura)

El "data cube": preguntas tipo OLAP sobre los experimentos `done` sin descargar el CSV. Todos bajo `/datasets/{ds}/cube/...`.

### `GET /cube/metrics`
Lista qué métricas existen en `final_metrics` (nombres + count + min/max/media).
**Cuándo**: lo primero que llamas en un dataset nuevo — para saber qué métricas hay y cómo se llaman. Sin esto no sabes qué meter en `?metric=`.

### `GET /cube/params`
Lista qué claves de `config` existen (aplanadas, p. ej. `config.lr`, `config.loss.type`). También expone los top-level útiles (`architecture`, `agent_id`, `device`).
**Cuándo**: descubrir qué dimensiones puedes usar en `by=` o `where=`.

### `GET /cube/params/values?key=...&metric=...`
Valores distintos de un parámetro con cuántos experimentos por valor. Si pasas `metric`, también te da max/mean/min de la métrica por valor.
**Cuándo**: "¿qué valores de `lr` se han probado y cuál da mejor `subset_acc`?".

### `GET /cube/top?metric=...&limit=...`
Top-N experimentos completos ordenados por una métrica (asc o desc). Con filtros `where=` para acotar.
**Cuándo**: "¿cuáles son los 10 mejores modelos?". Devuelve documentos enteros, no agregados.

### `GET /cube/groupby?by=k1,k2&metric=...&agg=...`
**El cubo OLAP.** Agrupa por una o varias claves de `config`/top-level y agrega una métrica (max/min/mean/std/count/p50/p90/p95/p99).
**Cuándo**: "¿qué combinación de `(window_size, lr)` da la media más alta de `subset_acc`?", heatmaps, comparar familias de hiperparámetros.

### `GET /cube/best-per?by=key&metric=...`
Por cada valor distinto de `key`, devuelve el **mejor experimento completo** según `metric`.
**Cuándo**: "el mejor LSTM vs el mejor MLP", "el mejor por tipo de loss". Quieres el documento entero, no un agregado.

### `GET /cube/distribution?metric=...&bins=...`
Histograma de una métrica (min/max/media/std + buckets equiespaciados).
**Cuándo**: ver la forma de la distribución, detectar bimodalidades, comprobar si una métrica está saturada o muy dispersa.

---

## Convenciones del cubo

- `metric=X` → resuelve a `final_metrics.X`.
- `by=k1,k2` → lista coma-separada de paths (`config.lr`, `config.loss.type`, `architecture`, ...).
- `where=k=v,k2=v2` → filtros igualdad (parseo automático a float/bool/string).
- `status` → por defecto `done`. `status=all` para no filtrar.
- `order=field:asc|desc` → en `groupby` puedes ordenar por `max`/`min`/`mean`/`std`/`count`/`p50`/`p90`/`p95`/`p99`.

---

## Tabla resumen

| Endpoint | Para qué |
|---|---|
| `GET /health` | ¿Vivo? |
| `GET /datasets` | ¿Qué datasets hay? |
| `GET /stats` | Counts por estado |
| `GET /summary` | Counts + ETA + running |
| `GET /results.csv` | Dump CSV completo |
| `GET /experiments` | Lista bruta |
| `GET /experiments/{n}` | Inspeccionar uno |
| `GET /cube/metrics` | ¿Qué métricas hay? |
| `GET /cube/params` | ¿Qué hiperparámetros hay? |
| `GET /cube/params/values` | Valores de un hiperparam |
| `GET /cube/top` | Los N mejores |
| `GET /cube/groupby` | Cubo OLAP |
| `GET /cube/best-per` | Mejor por valor de X |
| `GET /cube/distribution` | Histograma |
