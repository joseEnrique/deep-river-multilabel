"""
runner.py — Single experiment executor.

Supports:
  - Architectures: LSTM, MLP
  - Datasets:      ai4i, nps
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
from classes.rolling_multilabel_classifier_sequences import RollingMultiLabelClassifierSequences
from testclassifier.model import (
    LSTM_MultiLabel, MLP_MultiLabel, CNN_MultiLabel, Transformer_MultiLabel,
    FocalLoss, AdaptiveFocalLoss,
)
import evaluate as _evaluate
from metrics import HammingLoss, ExampleF1, ExamplePrecision, ExampleRecall

# ── Dataset registry ──────────────────────────────────────────────────────────

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
}

# ── Architecture registry ─────────────────────────────────────────────────────

ARCHITECTURES = {
    "LSTM": LSTM_MultiLabel,
    "MLP":  MLP_MultiLabel,
    "CNN":  CNN_MultiLabel,
    "Transformer": Transformer_MultiLabel,
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
                  "output_dim", "seed", "epochs"}
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
    label_names  = ds_info["label_names"]
    # Allow config to override output_dim; default from dataset
    output_dim   = model_cfg.get("output_dim", ds_info["output_dim"])

    # ── Architecture ─────────────────────────────────────────────────────────
    arch_name = model_cfg.get("architecture", "LSTM")
    module_cls = ARCHITECTURES[arch_name]

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer_fn = model_cfg.get("optimizer", "adam")

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_fn = build_loss(loss_cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    clf = RollingMultiLabelClassifierSequences(
        module=module_cls,
        label_names=label_names,
        optimizer_fn=optimizer_fn,
        lr=model_cfg["lr"],
        device=device_str,
        window_size=model_cfg["window_size"],
        past_history=model_cfg["past_history"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        bidirectional=model_cfg.get("bidirectional", False),
        output_dim=output_dim,
        seed=model_cfg.get("seed", 42),
        epochs=model_cfg.get("epochs", 1),
        loss_fn=loss_fn,
    )

    pr = SelectType(numbers.Number) | preprocessing.StandardScaler()
    pr += SelectType(str) | preprocessing.OneHotEncoder()
    pipeline = pr | clf

    all_metrics = build_metrics()
    checkpoint_rows = []
    start = time.time()

    # ── Evaluation ───────────────────────────────────────────────────────────
    for cp in _evaluate.iter_progressive_val_score(
        dataset=ds_info["cls"](),
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
