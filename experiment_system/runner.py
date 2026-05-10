"""
runner.py — Single experiment executor.

Supports:
  - Architectures: LSTM, MLP, CNN, Transformer
  - Datasets:      ai4i, nps, alpi
  - Optimizers:    adam, sgd
  - Losses:        BCE, StaticFocal, FullAdaptive, ImprovedAdaptive
"""

import sys
import time
import numbers
import json
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from river.compose import SelectType
from river.metrics import F1, Precision, Recall
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch, MicroAverage, MacroAverage
from river import preprocessing

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datasets.multioutput import Ai4i
from datasets.multioutput.nps import NPS
from datasets.multioutput.newalpi import NewAlpi
from classes.rolling_multilabel_classifier_sequences import RollingMultiLabelClassifierSequences
from testclassifier.model import (
    LSTM_MultiLabel, MLP_MultiLabel, CNN_MultiLabel, Transformer_MultiLabel,
    AlpiLSTM, AlpiMLP, AlpiCNN, AlpiTransformer,
)
from testclassifier.loss import FocalLoss, AdaptiveFocalLoss
import evaluate as _evaluate
from metrics import HammingLoss, ExampleF1, ExamplePrecision, ExampleRecall

# ── Dataset registry ──────────────────────────────────────────────────────────
# For datasets with fixed schema (ai4i, nps) we declare label_names/output_dim.
# For alpi the output schema depends on (machine, input_win, output_win, ...)
# so we mark it `dynamic=True` and resolve labels by instantiating the dataset.

ALPI_PARAM_KEYS = ("machine", "input_win", "output_win", "delta", "sigma", "min_count")

DATASETS = {
    "ai4i": {
        "cls":         Ai4i,
        "label_names": ['TWF', 'HDF', 'PWF', 'OSF', 'RNF'],
        "output_dim":  5,
    },
    "nps": {
        "cls":         NPS,
        "label_names": ['PRP', 'HLL', 'GTC', 'GT'],
        "output_dim":  4,
    },
    "alpi": {
        "cls":     NewAlpi,
        "dynamic": True,
        "param_keys": ALPI_PARAM_KEYS,
    },
}

# ── Architecture registry ─────────────────────────────────────────────────────
# Alpi datasets need the embedding-based variants because inputs are
# alarm IDs (categorical) instead of floats.

ARCHITECTURES = {
    "LSTM": LSTM_MultiLabel,
    "MLP":  MLP_MultiLabel,
    "CNN":  CNN_MultiLabel,
    "Transformer": Transformer_MultiLabel,
}

ALPI_ARCHITECTURES = {
    "LSTM": AlpiLSTM,
    "MLP":  AlpiMLP,
    "CNN":  AlpiCNN,
    "Transformer": AlpiTransformer,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_loss(loss_cfg: dict):
    ltype = loss_cfg["type"]
    if ltype == "BCE":
        return torch.nn.BCEWithLogitsLoss()
    elif ltype == "StaticFocal":
        return FocalLoss(
            alpha=loss_cfg.get("alpha", 0.25),
            gamma=loss_cfg.get("gamma", 2.0),
        )
    elif ltype in ("AdaptiveFocal", "ImprovedAdaptive"):
        return AdaptiveFocalLoss(
            base_gamma=loss_cfg.get("base_gamma", 2.0),
            base_alpha=loss_cfg.get("base_alpha", 0.25),
            decay=loss_cfg.get("decay", 0.999),
            alpha_gain=loss_cfg.get("alpha_gain", 1.0),
            gamma_gain=loss_cfg.get("gamma_gain", 2.0),
        )
    else:
        raise ValueError(f"Unknown loss type: {ltype!r}")


def build_metrics():
    return Metrics([
        ExactMatch(),               # 0
        HammingLoss(),              # 1
        ExampleF1(),                # 2
        ExamplePrecision(),         # 3
        ExampleRecall(),            # 4
        MicroAverage(F1()),         # 5
        MicroAverage(Precision()),  # 6
        MicroAverage(Recall()),     # 7
        MacroAverage(F1()),         # 8
        MacroAverage(Precision()),  # 9
        MacroAverage(Recall()),     # 10
    ])


def extract_metrics(m) -> dict:
    return {
        "subset_acc": m[0].get() * 100,
        "hamm_loss":  m[1].get() * 100,
        "examp_f1":   m[2].get() * 100,
        "examp_prec": m[3].get() * 100,
        "examp_rec":  m[4].get() * 100,
        "micro_f1":   m[5].get() * 100,
        "micro_prec": m[6].get() * 100,
        "micro_rec":  m[7].get() * 100,
        "macro_f1":   m[8].get() * 100,
        "macro_prec": m[9].get() * 100,
        "macro_rec":  m[10].get() * 100,
    }


def coerce_types(cfg: dict) -> dict:
    """Coerce config values to proper Python types after JSON round-trip."""
    float_keys = {"lr", "dropout"}
    int_keys   = {"past_history", "window_size", "hidden_dim", "num_layers",
                  "output_dim", "seed", "epochs",
                  "embedding_dim", "num_alarms",
                  "machine", "input_win", "output_win", "delta", "sigma", "min_count"}
    bool_keys  = {"bidirectional"}
    out = {}
    for k, v in cfg.items():
        if k in float_keys:
            out[k] = float(v)
        elif k in int_keys:
            out[k] = int(v)
        elif k in bool_keys:
            out[k] = bool(v)
        else:
            out[k] = v
    return out


# ── Main run function ─────────────────────────────────────────────────────────

def run(exp_id: str, exp_name: str, config: dict, results_dir: Path,
        checkpoint_every: int = 500, device_str: str = "cuda") -> dict:
    """
    Run a single experiment.

    Config keys (model):
      architecture : 'LSTM' | 'MLP'        (default: LSTM)
      dataset      : 'ai4i' | 'nps'        (default: ai4i)
      optimizer    : 'adam' | 'sgd'        (default: adam)
      past_history : int                   (seq length; LSTM only meaningful >1)
      window_size  : int
      hidden_dim   : int
      num_layers   : int
      lr           : float
      dropout      : float
      bidirectional: bool                  (LSTM only)
      output_dim   : int  (auto-set from dataset if omitted)
      seed         : int
      epochs       : int
      loss         : dict  (type + params)
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = coerce_types({k: v for k, v in config.items() if k != "loss"})
    loss_cfg  = config["loss"]

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset_name = model_cfg.get("dataset", "ai4i")
    ds_info      = DATASETS[dataset_name]

    if ds_info.get("dynamic"):
        ds_kwargs = {k: model_cfg[k] for k in ds_info["param_keys"] if k in model_cfg}
        stream = ds_info["cls"](**ds_kwargs)
        # Pandas may load label columns as non-strings; river expects str names.
        stream.Y.columns = stream.Y.columns.astype(str)
        label_names = list(stream.Y.columns)
        output_dim  = model_cfg.get("output_dim", len(label_names))
    else:
        ds_kwargs   = {}
        stream      = ds_info["cls"]()
        label_names = ds_info["label_names"]
        output_dim  = model_cfg.get("output_dim", ds_info["output_dim"])

    # ── Architecture ─────────────────────────────────────────────────────────
    arch_name = model_cfg.get("architecture", "LSTM")
    arch_table = ALPI_ARCHITECTURES if dataset_name == "alpi" else ARCHITECTURES
    module_cls = arch_table[arch_name]

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer_fn = model_cfg.get("optimizer", "adam")

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_fn = build_loss(loss_cfg)

    excluded_keys = {"dataset", "architecture", "optimizer", "output_dim", "device"}
    excluded_keys.update(ds_kwargs.keys())
    kwargs_for_clf = {k: v for k, v in model_cfg.items() if k not in excluded_keys}

    clf = RollingMultiLabelClassifierSequences(
        module=module_cls,
        label_names=label_names,
        optimizer_fn=optimizer_fn,
        device=device_str,
        output_dim=output_dim,
        loss_fn=loss_fn,
        **kwargs_for_clf
    )

    if dataset_name == "alpi":
        # Alarm IDs are categorical: skip StandardScaler/OneHotEncoder.
        pipeline = SelectType(numbers.Number) | clf
    else:
        pr = SelectType(numbers.Number) | preprocessing.StandardScaler()
        pr += SelectType(str) | preprocessing.OneHotEncoder()
        pipeline = pr | clf

    all_metrics = build_metrics()
    checkpoint_rows = []
    start = time.time()

    # ── Evaluation ───────────────────────────────────────────────────────────
    for cp in _evaluate.iter_progressive_val_score(
        dataset=stream,
        model=pipeline,
        metric=all_metrics,
        step=checkpoint_every,
        measure_time=True,
    ):
        step_n  = cp["Step"]
        elapsed = cp["Time"].total_seconds()
        mv = extract_metrics(all_metrics)

        print(f"  [{step_n:,d}] "
              f"SubsetAcc={mv['subset_acc']:.2f}% | "
              f"MicroF1={mv['micro_f1']:.2f}% | "
              f"MacroF1={mv['macro_f1']:.2f}% | "
              f"t={elapsed:.1f}s")

        checkpoint_rows.append(
            {"exp_id": exp_id, "exp_name": exp_name,
             "step": step_n, "elapsed_s": round(elapsed, 2), **mv}
        )

    # ── Save checkpoint CSV ───────────────────────────────────────────────────
    safe_name = exp_name.replace("/", "-").replace(" ", "_")
    ckpt_path = results_dir / f"{safe_name}_checkpoints.csv"
    pd.DataFrame(checkpoint_rows).to_csv(ckpt_path, index=False)
    print(f"  📄 Checkpoints → {ckpt_path.name}")

    duration = time.time() - start
    final_metrics = extract_metrics(all_metrics)

    result = {
        "exp_name":    exp_name,
        "architecture": arch_name,
        "dataset":     dataset_name,
        "optimizer":   optimizer_fn,
        **model_cfg,
        "loss_type":   loss_cfg["type"],
        "loss_config": json.dumps({k: v for k, v in loss_cfg.items() if k != "type"}),
        **final_metrics,
        "duration_s":  round(duration, 1),
        "finished_at": datetime.now().isoformat(),
    }
    return result
