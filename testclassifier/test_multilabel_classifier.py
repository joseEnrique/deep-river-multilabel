"""
Test script for MultiLabelClassifier (without rolling window).
Uses LSTM_MultiLabel for multi-label classification.

NOTE: MultiLabelClassifier does NOT use rolling windows, so it processes
each sample independently. For temporal modeling, use RollingMultiLabelClassifier.
"""

import csv
import sys
from pathlib import Path
import torch
from river import preprocessing

# Rutas locales
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from classes.multilabel_classifier import MultiLabelClassifier
from testclassifier.model import LSTM_MultiLabel
from incrementalmetrics import IncrementalMultiLabelMetrics
import torch.nn as nn


# Wrapper to adapt LSTM_MultiLabel for non-rolling use
class LSTM_MultiLabel_SingleStep(nn.Module):
    """
    Wrapper around LSTM_MultiLabel to handle single timestep inputs.

    MultiLabelClassifier feeds single samples [batch, features], but LSTM expects
    [batch, seq_len, features]. This wrapper adds a sequence dimension.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.lstm_model = LSTM_MultiLabel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

    def forward(self, x):
        # x is [batch, features] from MultiLabelClassifier
        # Add sequence dimension: [batch, 1, features]
        x = x.unsqueeze(1)
        return self.lstm_model(x)


def stream_csv_to_dicts(filepath: str, feature_cols, target_names):
    """Generator that converts CSV to a stream of (x, y) for online learning."""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = {k: float(row[k]) for k in feature_cols}
                y = {t: int(float(row.get(t, 0))) for t in target_names}
                yield x, y
            except ValueError:
                # Skip corrupted rows
                continue


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Label Classifier Test (No Rolling Window)")
    print("Using LSTM_MultiLabel with single-step wrapper")
    print("=" * 60)

    # Dataset configuration
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
    print(f"\nDevice: {device}")

    # Create multi-label classifier with LSTM wrapper
    clf = MultiLabelClassifier(
        module=LSTM_MultiLabel_SingleStep,
        label_names=target_names,
        optimizer_fn="adam",
        lr=1e-3,
        device=device,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
    )

    # Online scaler
    scaler = preprocessing.StandardScaler()

    # Incremental custommetrics
    metrics_agg = IncrementalMultiLabelMetrics(label_names=target_names)
    total_samples = 0

    print("\nStarting online training...")
    print("=" * 60)

    # Main training loop
    for x, y in stream_csv_to_dicts(dataset_path, feature_cols, target_names):
        total_samples += 1

        # Online transformation before predict/learn
        x_s = scaler.transform_one(x)

        # Incremental prediction
        y_pred = clf.predict_one(x_s)

        # Update custommetrics
        y_true_vec = [y[t] for t in target_names]
        y_pred_vec = [y_pred[t] for t in target_names]
        metrics_agg.update(y_true_vec, y_pred_vec)

        # Online learning
        clf.learn_one(x_s, y)

        # Update scaler with current sample
        scaler.learn_one(x)

        # Progress update
        if total_samples % 2000 == 0:
            print(f"Processed {total_samples} samples...")

    # Final report
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    results = metrics_agg.pretty_print()
