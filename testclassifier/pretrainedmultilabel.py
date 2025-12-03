"""
Pretrained multi-label classifier using PretrainedRollingMultiLabelClassifier.

- Carga modelo preentrenado desde archivo
- window_size = 10 (según checkpoint)
- BCEWithLogitsLoss
- Métricas: MicroF1 y MacroF1 únicamente
"""

import csv
import sys
from typing import Dict
from pathlib import Path
import torch

# Rutas locales
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Importa modelo y PretrainedRollingMultiLabelClassifier
from model import LSTM_MultiLabel
from classes.pretrained_rolling_classifier import PretrainedRollingMultiLabelClassifier

# Métricas incrementales personalizadas
testforecaster_path = ROOT / "testforecaster"
sys.path.insert(0, str(testforecaster_path))
from incrementalmetrics import IncrementalMultiLabelMetrics  # type: ignore


def stream_csv_to_dicts(filepath: str, feature_cols, target_names):
    """Generador que convierte el CSV en un flujo de (x, y) para aprendizaje online."""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = {k: float(row[k]) for k in feature_cols}
                y = {t: int(float(row.get(t, 0))) for t in target_names}
                yield x, y
            except ValueError:
                # Saltar filas corruptas
                continue


if __name__ == "__main__":
    # === Dataset y configuración ===
    dataset_path = "/home/quique/tesis/OEMLHAT4PdM/datasets/ai4i2020formatted.csv"
    feature_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type_H",
        "Type_L",
        "Type_M",
    ]
    target_names = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Rutas al modelo preentrenado
    checkpoint_path = ROOT / "testbatch" / "lstm_multilabel_ai4i_complete.pt"
    scaler_path = ROOT / "testbatch" / "scaler_ai4i.pkl"

    # === Pretrained Rolling multi-label classifier ===
    clf = PretrainedRollingMultiLabelClassifier(
        module=LSTM_MultiLabel,
        label_names=target_names,
        checkpoint_path=str(checkpoint_path),
        scaler_path=str(scaler_path),
        optimizer_fn="adam",
        lr=1e-4,  # Learning rate más bajo para fine-tuning
        device=device,
        window_size=10,  # Según el modelo preentrenado
        append_predict=False,
        freeze_pretrained=False,  # Permitir fine-tuning
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        epochs=1,
    )

    # === Calcular índice de inicio del 20% test (último 20%) ===
    import pandas as pd
    df_full = pd.read_csv(dataset_path)
    total_rows = len(df_full)
    train_size = int(total_rows * 0.8)  # 80% train
    test_start_idx = train_size  # Empezar desde aquí

    print("="*60)
    print("CONFIGURACIÓN DE DATOS")
    print("="*60)
    print(f"Total de filas en dataset: {total_rows}")
    print(f"Índice de inicio del test (20%): {test_start_idx}")
    print(f"Muestras a procesar: {total_rows - test_start_idx}")

    # === Métricas incrementales (solo MicroF1 y MacroF1) ===
    metrics_agg = IncrementalMultiLabelMetrics(label_names=target_names)
    total_samples = 0
    skipped_samples = 0

    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO ONLINE CON MODELO PREENTRENADO")
    print("="*60)
    print("IMPORTANTE: El modelo usa el scaler del batch internamente")
    print("Evaluando SOLO en el último 20% de los datos (test set)")
    print("="*60)

    # === Bucle principal de entrenamiento ===
    for x, y in stream_csv_to_dicts(dataset_path, feature_cols, target_names):
        # Saltar las primeras 80% muestras (train set)
        if skipped_samples < test_start_idx:
            skipped_samples += 1
            continue

        total_samples += 1

        # NO aplicar scaler aquí - el PretrainedRollingMultiLabelClassifier
        # ya aplica el scaler del batch internamente

        # Predicción incremental
        y_pred = clf.predict_one(x)

        # Actualiza métricas desde el inicio (sin warm-up)
        y_true_vec = [y[t] for t in target_names]
        y_pred_vec = [y_pred[t] for t in target_names]
        metrics_agg.update(y_true_vec, y_pred_vec)

        # Aprendizaje online (fine-tuning)
        clf.learn_one(x, y)

        # Progreso cada 200 muestras
        if total_samples % 200 == 0:
            print(f"Procesadas {total_samples} muestras del test set...")

    # === Reporte final ===
    print("\n" + "="*60)
    print("RESULTADOS FINALES")
    print("="*60)
    print(f"Total de muestras procesadas: {total_samples}")
    results = metrics_agg.pretty_print()
