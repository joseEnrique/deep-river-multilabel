"""
runner_adapter.py — wraps experiment_system/runner.py and streams checkpoints
to the backend instead of writing local CSVs.

Reuses the existing pipeline construction, evaluation loop and metric extraction
so behaviour stays consistent with the pre-distributed system.
"""

from __future__ import annotations
import sys
import time
import numbers
from pathlib import Path

import torch
import pandas as pd  # noqa: F401  (kept for parity with the original runner)
from river.compose import SelectType
from river.metrics import F1, Precision, Recall
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch, MicroAverage, MacroAverage
from river import preprocessing

# Project root sits two levels up: deep-river-tesis/distributed/agent/runner_adapter.py
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
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

# alpi has dynamic schema (labels/output_dim depend on dataset kwargs)
ALPI_PARAM_KEYS = ("machine", "input_win", "output_win", "delta", "sigma", "min_count")

DATASETS = {
    "ai4i": {"cls": Ai4i,  "label_names": ['TWF','HDF','PWF','OSF','RNF'], "output_dim": 5},
    "nps":  {"cls": NPS,   "label_names": ['PRP','HLL','GTC','GT'],         "output_dim": 4},
    "alpi": {"cls": NewAlpi, "dynamic": True, "param_keys": ALPI_PARAM_KEYS},
}

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


def _build_loss(loss_cfg: dict):
    ltype = loss_cfg["type"]
    if ltype == "BCE":
        return torch.nn.BCEWithLogitsLoss()
    if ltype == "StaticFocal":
        return FocalLoss(
            alpha=loss_cfg.get("alpha", 0.25),
            gamma=loss_cfg.get("gamma", 2.0),
        )
    if ltype in ("AdaptiveFocal", "ImprovedAdaptive", "FullAdaptive"):
        return AdaptiveFocalLoss(
            base_gamma=loss_cfg.get("base_gamma", 2.0),
            base_alpha=loss_cfg.get("base_alpha", 0.25),
            decay=loss_cfg.get("decay", 0.999),
            alpha_gain=loss_cfg.get("alpha_gain", 1.0),
            gamma_gain=loss_cfg.get("gamma_gain", 2.0),
        )
    raise ValueError(f"Unknown loss type: {ltype!r}")


def _build_metrics() -> Metrics:
    return Metrics([
        ExactMatch(), HammingLoss(),
        ExampleF1(), ExamplePrecision(), ExampleRecall(),
        MicroAverage(F1()), MicroAverage(Precision()), MicroAverage(Recall()),
        MacroAverage(F1()), MacroAverage(Precision()), MacroAverage(Recall()),
    ])


def _extract(m: Metrics) -> dict[str, float]:
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


def _loss_logs(loss_fn) -> dict[str, float]:
    """Estado interno de las losses adaptativas (alpha/gamma por clase, F1 EMA).

    Devuelve {} para losses sin get_logs (BCE, StaticFocal) o si la loss aún no
    se ha inicializado (num_classes se fija en el primer forward)."""
    if hasattr(loss_fn, "get_logs") and getattr(loss_fn, "num_classes", None):
        return loss_fn.get_logs()
    return {}


def _coerce_types(cfg: dict) -> dict:
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


def run_with_backend(exp_name: str, config: dict, device_str: str,
                     checkpoint_every: int, on_checkpoint, on_finish) -> None:
    """
    Run a single experiment.

    on_checkpoint(step, elapsed_s, metrics_dict) is called every N instances.
    on_finish(final_metrics_dict, duration_s) is called once at the end.

    Both callbacks may raise to abort early.
    """
    model_cfg = _coerce_types({k: v for k, v in config.items() if k != "loss"})
    loss_cfg  = config["loss"]

    dataset_name = model_cfg.get("dataset", "ai4i")
    ds_info      = DATASETS[dataset_name]

    if ds_info.get("dynamic"):
        ds_kwargs = {k: model_cfg[k] for k in ds_info["param_keys"] if k in model_cfg}
        stream = ds_info["cls"](**ds_kwargs)
        stream.Y.columns = stream.Y.columns.astype(str)
        label_names = list(stream.Y.columns)
        output_dim  = model_cfg.get("output_dim", len(label_names))
    else:
        ds_kwargs   = {}
        stream      = ds_info["cls"]()
        label_names = ds_info["label_names"]
        output_dim  = model_cfg.get("output_dim", ds_info["output_dim"])

    arch_name    = model_cfg.get("architecture", "LSTM")
    arch_table   = ALPI_ARCHITECTURES if dataset_name == "alpi" else ARCHITECTURES
    module_cls   = arch_table[arch_name]
    optimizer_fn = model_cfg.get("optimizer", "adam")
    loss_fn      = _build_loss(loss_cfg)

    excluded = {"dataset", "architecture", "optimizer", "output_dim", "device"}
    excluded.update(ds_kwargs.keys())
    if dataset_name != "alpi":
        # alpi-only keys: si se cuelan en un config nps/ai4i, ignorarlas en lugar
        # de dejar que viajen al classifier vía **kwargs.
        excluded |= {"machine", "input_win", "output_win", "delta", "sigma",
                     "min_count", "embedding_dim", "num_alarms"}
    kwargs_for_clf = {k: v for k, v in model_cfg.items() if k not in excluded}

    clf = RollingMultiLabelClassifierSequences(
        module=module_cls,
        label_names=label_names,
        optimizer_fn=optimizer_fn,
        device=device_str,
        output_dim=output_dim,
        loss_fn=loss_fn,
        **kwargs_for_clf,
    )

    if dataset_name == "alpi":
        # Alarm IDs are categorical: skip StandardScaler/OneHotEncoder.
        pipeline = SelectType(numbers.Number) | clf
    else:
        pr = SelectType(numbers.Number) | preprocessing.StandardScaler()
        pr += SelectType(str) | preprocessing.OneHotEncoder()
        pipeline = pr | clf

    metrics = _build_metrics()
    start = time.time()

    for cp in _evaluate.iter_progressive_val_score(
        dataset=stream,
        model=pipeline,
        metric=metrics,
        step=checkpoint_every,
        measure_time=True,
    ):
        step_n  = cp["Step"]
        elapsed = cp["Time"].total_seconds()
        mv = _extract(metrics)
        mv.update(_loss_logs(loss_fn))
        on_checkpoint(step_n, elapsed, mv)

    duration = time.time() - start
    final = _extract(metrics)
    final.update(_loss_logs(loss_fn))
    on_finish(final, duration)
