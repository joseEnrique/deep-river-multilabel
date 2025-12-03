"""
Baseline multi-label classifier (single RollingClassifier, no pos_weight, no warm-up).

- window_size = 50
- BCEWithLogitsLoss (sin pos_weight)
- Métricas: MicroF1 y MacroF1 únicamente
"""

import csv
import sys
from typing import Dict
from pathlib import Path
import torch
from river import preprocessing

# Rutas locales
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Importa modelo y RollingClassifier
from model import LSTM_MultiLabel
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier

# Métricas incrementales personalizadas

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
    window_size = 100

    # === Rolling multi-label classifier ===
    clf = RollingMultiLabelClassifier(
        module=LSTM_MultiLabel,
        label_names=target_names,
        optimizer_fn="adam",
        lr=1e-3,
        device=device,
        window_size=window_size,
        append_predict=False,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        output_dim=len(target_names),
    )

    # Escalador online (normalización incremental)
    scaler = preprocessing.StandardScaler()

    # === Métricas incrementales (solo MicroF1 y MacroF1) ===
    metrics_agg = IncrementalMultiLabelMetrics(label_names=target_names)
    total_samples = 0

    # === Bucle principal de entrenamiento ===
    for x, y in stream_csv_to_dicts(dataset_path, feature_cols, target_names):
        total_samples += 1
        # Transformación online antes de predecir/aprender
        x_s = scaler.transform_one(x)
        # Predicción incremental
        y_pred = clf.predict_one(x_s)

        # Actualiza métricas desde el inicio (sin warm-up)
        y_true_vec = [y[t] for t in target_names]
        y_pred_vec = [y_pred[t] for t in target_names]
        metrics_agg.update(y_true_vec, y_pred_vec)

        # Aprendizaje online
        clf.learn_one(x_s, y)

        # Actualizar el escalador con la muestra actual
        scaler.learn_one(x)

    # === Reporte final ===
    results = metrics_agg.pretty_print()
