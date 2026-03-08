# Sistema de Experimentos

Sistema de ejecución de experimentos tolerante a fallos, con seguimiento en SQLite.
Si el proceso se interrumpe (apagado, error), los experimentos completados se saltan
automáticamente al reanudar; los interrumpidos se reintentarán.

## Estructura

```
experiment_system/
  config.yaml              ← configuración del grid de experimentos
  config_test.yaml         ← config mínima para pruebas (1 experimento)
  run_experiments.py       ← punto de entrada principal
  runner.py                ← ejecuta un único experimento
  db.py                    ← gestión de estado en SQLite
  experiments.db           ← base de datos de estado (se crea automáticamente)
  results/
    <nombre>_checkpoints.csv  ← métricas cada 500 pasos por experimento
    final_results.csv         ← una fila por experimento completado
```

## Uso

```bash
# Ejecutar todos los experimentos pendientes
python run_experiments.py

# Ver estado sin ejecutar nada
python run_experiments.py --status

# Usar una config diferente
python run_experiments.py --config mi_config.yaml
```

Si el proceso se cae a mitad: vuelve a ejecutar el mismo comando.
Los experimentos completados (`done`) se saltan, los interrumpidos se reintentarán.

## Configuración (`config.yaml`)

```yaml
dataset: ai4i

model:                    # producto cartesiano de todas las combinaciones
  past_history: [1, 2, 5]
  window_size: [200]
  hidden_dim: [128, 256]
  num_layers: [2, 3]
  lr: [1e-3]
  dropout: [0.3]
  bidirectional: [false]
  output_dim: [5]
  seed: [42]
  epochs: [1]

loss:                     # lista de funciones de pérdida a comparar
  - type: BCE
  - type: FullAdaptive
    base_gamma: 2.0
    base_alpha: 0.25
    decay: 0.999
    alpha_gain: 1.0
    gamma_gain: 2.0
  - type: ImprovedAdaptive
    base_gamma: 2.0
    base_alpha: 0.25
    decay: 0.999

checkpoint_every: 500           # guardar métricas cada N instancias
results_dir: results            # carpeta de resultados
db_path: experiments.db         # fichero SQLite
final_results_file: results/final_results.csv
```

En el ejemplo de arriba se generan `3 × 2 × 2 × 3 = 36` experimentos.

## Nombres de experimentos

Cada experimento recibe un nombre legible automáticamente, p.ej.:

```
LSTM_ph1_h128_nl2_w200_lr0.001_BCE
LSTM_ph2_h256_nl3_w200_lr0.001_FullAdaptive
```

Los ficheros de checkpoints y el CSV final usan este nombre, no un hash.

## Métricas guardadas (11 en total)

| Columna | Descripción |
|---|---|
| `subset_acc` | Subset Accuracy (ExactMatch) × 100 |
| `hamm_loss` | Hamming Loss × 100 |
| `examp_f1/prec/rec` | Example-based F1, Precision, Recall |
| `micro_f1/prec/rec` | Micro-averaged F1, Precision, Recall |
| `macro_f1/prec/rec` | Macro-averaged F1, Precision, Recall |

## Funciones de pérdida disponibles

| `type` | Descripción |
|---|---|
| `BCE` | BCEWithLogitsLoss estándar |
| `FullAdaptive` | Focal adaptativo vía EMA de recall/accuracy |
| `ImprovedAdaptive` | Focal adaptativo vía EMA de F1 por clase |
