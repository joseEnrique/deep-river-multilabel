"""
Test script to validate warm-up removal from DirectMultiLabelForecaster.

Compares a single configuration to verify that removing the warm-up period
improves performance and brings results closer to RollingMultiLabelClassifier.
"""

import torch
import pandas as pd
from datetime import datetime
from testclassifier.model import LSTM_MultiLabel
from classes.direct_multilabel_forecaster import DirectMultiLabelForecaster
from datasets.multioutput import RollingAi4i
from river import compose, preprocessing
from river.metrics import F1
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch
from custommetrics.multioutput import MacroAverage, MicroAverage
from evaluate.progressive_validation import progressive_val_score_sequence

# Configuration to test (Medium-LR5e4-W200 from comparison)
config = {
    "name": "LSTM-Medium-LR5e4-W200-NoWarmup",
    "hidden_dim": 128,
    "num_layers": 2,
    "lr": 5e-4,
    "dropout": 0.3,
    "bidirectional": True,
    "window_size": 200,
    "loss": "BCEWithLogits",
    "past_size": 1,
    "epochs": 10
}

# Target names
target_names = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("="*80)
print("TEST: DirectMultiLabelForecaster WITHOUT Warm-up Period")
print("="*80)
print(f"Configuration: {config['name']}")
print(f"Device: {device}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

print("\nExpected improvements:")
print("  - LSTM Normal (lstm_generico.csv): Micro F1 = 29.97%")
print("  - Direct WITH warm-up: Micro F1 = 28.99%")
print("  - Direct WITHOUT warm-up: Expected ≈ 30.5-31.0%")
print("\n" + "="*80)

# Setup numeric columns
numeric_cols = ['Air temperature [K]', 'Process temperature [K]',
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# Create loss function
loss_fn = torch.nn.BCEWithLogitsLoss()

# Create forecaster
forecaster = DirectMultiLabelForecaster(
    window_size=config['window_size'],
    label_names=target_names,
    module=LSTM_MultiLabel,
    loss_fn=loss_fn,
    optimizer_fn='adam',
    lr=config['lr'],
    device=device,
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers'],
    output_dim=len(target_names),
    epochs=config['epochs'],
    seed=42,
    threshold=0.5,
    dropout=config['dropout'],
    bidirectional=config['bidirectional']
)

# Pipeline setup
p_numeric = compose.Select(*numeric_cols) | preprocessing.StandardScaler()
p_categorical = compose.Select('Type') | preprocessing.OneHotEncoder()
pipeline = (p_numeric + p_categorical) | forecaster

# Create stream
stream = RollingAi4i(past_size=config['past_size'], n_instances=10000, include_targets=False)

print("\nStarting evaluation...")
print("-" * 80)

# Evaluate
metrics_result = progressive_val_score_sequence(
    dataset=stream,
    model=pipeline,
    metric=Metrics([
        ExactMatch(),
        MacroAverage(F1()),
        MicroAverage(F1())
    ]),
    print_every=2000
)

# Extract metrics
exact_match = metrics_result[0].get()
macro_f1 = metrics_result[1].get()
micro_f1 = metrics_result[2].get()

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"ExactMatch: {exact_match*100:.2f}%")
print(f"Macro F1:   {macro_f1*100:.2f}%")
print(f"Micro F1:   {micro_f1*100:.2f}%")
print("="*80)

# Comparison
print("\nCOMPARISON:")
print(f"  LSTM Normal (no warm-up):     Micro F1 = 29.97%")
print(f"  Direct WITH warm-up:          Micro F1 = 28.99%")
print(f"  Direct WITHOUT warm-up (NEW): Micro F1 = {micro_f1*100:.2f}%")

improvement_vs_warmup = micro_f1*100 - 28.99
improvement_vs_lstm = micro_f1*100 - 29.97

print(f"\n  Improvement vs Direct with warm-up: {improvement_vs_warmup:+.2f}pp")
print(f"  Difference vs LSTM Normal:          {improvement_vs_lstm:+.2f}pp")

if improvement_vs_warmup > 0:
    print("\n✅ SUCCESS: Removing warm-up IMPROVED performance!")
else:
    print("\n⚠️  WARNING: No improvement detected. Investigate further.")

print("\n" + "="*80)
print(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
