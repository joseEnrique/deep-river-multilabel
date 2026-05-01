"""
Sweep NPS: hiperparámetros de AdaptiveFocalLoss vs BCE/Focal de referencia.
Arquitectura LSTM fija. Solo se barren los hiperparámetros del Adaptive.
"""
import torch
import pandas as pd
import numbers
import time
import multiprocessing
from testclassifier.model import LSTM_MultiLabel
from testclassifier.loss import AdaptiveFocalLoss, FocalLoss
from classes.rolling_multilabel_classifier_sequences import RollingMultiLabelClassifierSequences
from datasets.multioutput.nps import NPS
from river.compose import SelectType
from river.metrics import F1
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch
from river import preprocessing
from custommetrics.multioutput import MicroAverage, MacroAverage
import evaluate


TARGET_NAMES = ['PRP', 'HLL', 'GTC', 'GT']

# ── Arquitectura y entrenamiento fijos ────────────────────────────────────────
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = True
LR = 1e-3
WS = 1
PH = 1
EPOCHS = 1
THRESHOLD = 0.5
GRADIENT_SCALE = 1.0


# ── Configuraciones de loss ───────────────────────────────────────────────────
CONFIGS = []

# 1) BCE de referencia
CONFIGS.append({"loss": "bce", "params": {}, "label": "bce"})

# 2) Focal estático de referencia
for alpha in [0.25, 0.5, 0.75]:
    for gamma in [2.0, 3.0]:
        CONFIGS.append({
            "loss": "focal",
            "params": {"alpha": alpha, "gamma": gamma},
            "label": f"focal_a{alpha}_g{gamma}",
        })

# 3) AdaptiveFocal — selección manual (máx 10) variando un eje cada vez
ADAPTIVE_VARIANTS = [
    # baseline
    {"base_gamma": 2.0, "decay": 0.99,  "alpha_gain": 1.0,  "gamma_gain": 2.0},
    # variar decay
    {"base_gamma": 2.0, "decay": 0.9,   "alpha_gain": 1.0,  "gamma_gain": 2.0},
    {"base_gamma": 2.0, "decay": 0.999, "alpha_gain": 1.0,  "gamma_gain": 2.0},
    # variar base_gamma
    {"base_gamma": 1.0, "decay": 0.99,  "alpha_gain": 1.0,  "gamma_gain": 2.0},
    {"base_gamma": 3.0, "decay": 0.99,  "alpha_gain": 1.0,  "gamma_gain": 2.0},
    # variar alpha_gain
    {"base_gamma": 2.0, "decay": 0.99,  "alpha_gain": 5.0,  "gamma_gain": 2.0},
    {"base_gamma": 2.0, "decay": 0.99,  "alpha_gain": 10.0, "gamma_gain": 2.0},
    # variar gamma_gain
    {"base_gamma": 2.0, "decay": 0.99,  "alpha_gain": 1.0,  "gamma_gain": 5.0},
    # combos agresivos
    {"base_gamma": 2.0, "decay": 0.99,  "alpha_gain": 5.0,  "gamma_gain": 5.0},
    {"base_gamma": 2.0, "decay": 0.999, "alpha_gain": 5.0,  "gamma_gain": 2.0},
]

for v in ADAPTIVE_VARIANTS:
    CONFIGS.append({
        "loss": "adaptive_focal",
        "params": {"base_alpha": 0.25, **v},
        "label": f"adapt_g{v['base_gamma']}_d{v['decay']}_ag{v['alpha_gain']}_gg{v['gamma_gain']}",
    })


def build_loss(kind, params):
    if kind == "bce":
        return torch.nn.BCEWithLogitsLoss(reduction='mean')
    if kind == "focal":
        return FocalLoss(alpha=params["alpha"], gamma=params["gamma"], reduction='mean')
    if kind == "adaptive_focal":
        return AdaptiveFocalLoss(reduction='mean', **params)
    raise ValueError(f"Loss desconocida: {kind}")


def run_experiment(cfg, device, queue):
    try:
        label = cfg["label"]
        stream = NPS()

        loss_fn = build_loss(cfg["loss"], cfg["params"])

        clf = RollingMultiLabelClassifierSequences(
            window_size=WS,
            past_history=PH,
            label_names=TARGET_NAMES,
            module=LSTM_MultiLabel,
            optimizer_fn="adam",
            lr=LR,
            device=device,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            bidirectional=BIDIRECTIONAL,
            output_dim=len(TARGET_NAMES),
            seed=42,
            epochs=EPOCHS,
            loss_fn=loss_fn,
            threshold=THRESHOLD,
            gradient_scale=GRADIENT_SCALE,
        )

        pipeliner = SelectType(numbers.Number) | preprocessing.StandardScaler() | clf
        all_metrics = Metrics([ExactMatch(), MicroAverage(F1()), MacroAverage(F1())])

        start = time.time()
        m = evaluate.progressive_val_score(
            dataset=stream, model=pipeliner, metric=all_metrics,
            show_memory=False, print_every=5000,
        )
        duration = time.time() - start

        result = {
            "Config":     label,
            "Loss":       cfg["loss"],
            "base_gamma": cfg["params"].get("base_gamma", cfg["params"].get("gamma", "-")),
            "base_alpha": cfg["params"].get("base_alpha", cfg["params"].get("alpha", "-")),
            "decay":      cfg["params"].get("decay", "-"),
            "alpha_gain": cfg["params"].get("alpha_gain", "-"),
            "gamma_gain": cfg["params"].get("gamma_gain", "-"),
            "ExactMatch": round(m[0].get() * 100, 2),
            "Micro F1":   round(m[1].get() * 100, 2),
            "Macro F1":   round(m[2].get() * 100, 2),
            "Time (s)":   round(duration, 1),
        }
        print(f"  ✓ {label}: Micro={result['Micro F1']}  Macro={result['Macro F1']}  ({duration:.0f}s) [{device}]")
        queue.put(result)
    except Exception as e:
        import traceback
        print(f"  ✗ {cfg['label']}: {e}")
        traceback.print_exc()
        queue.put(None)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    gpus = ["cuda:0"]

    print(f"GPUs: {gpus}")
    print(f"Total configs: {len(CONFIGS)}")
    print(f"Dataset: NPS  |  Targets: {TARGET_NAMES}")
    print(f"Arquitectura fija: hid={HIDDEN_DIM} nl={NUM_LAYERS} drop={DROPOUT} bidir={BIDIRECTIONAL} lr={LR}")
    print(f"Entrenamiento fijo: ws={WS} ph={PH} ep={EPOCHS} th={THRESHOLD}\n")

    queue = multiprocessing.Queue()
    results = []
    active = {}
    free_gpus = list(gpus)
    exp_iter = iter(CONFIGS)

    while True:
        while free_gpus:
            try:
                cfg = next(exp_iter)
                gpu = free_gpus.pop(0)
                print(f"🚀 {cfg['label']} on {gpu}")
                p = multiprocessing.Process(target=run_experiment, args=(cfg, gpu, queue))
                p.start()
                active[p] = gpu
            except StopIteration:
                break

        if not active:
            break

        while not queue.empty():
            res = queue.get()
            if res:
                results.append(res)

        finished = [p for p in active if not p.is_alive()]
        for p in finished:
            gpu = active.pop(p)
            free_gpus.append(gpu)
            p.join()

        time.sleep(0.5)

    while not queue.empty():
        res = queue.get()
        if res:
            results.append(res)

    df = pd.DataFrame(results)
    df = df.sort_values("Micro F1", ascending=False)

    print("\n" + "=" * 120)
    print("FULL RESULTS — NPS BCE vs Focal vs AdaptiveFocal — sorted by Micro F1")
    print("=" * 120)
    print(df.to_string(index=False))

    print(f"\n{'='*60}")
    if len(df) > 0:
        best = df.iloc[0]
        print(f"BEST: {best['Config']}")
        print(f"  Micro F1   = {best['Micro F1']}")
        print(f"  Macro F1   = {best['Macro F1']}")
        print(f"  ExactMatch = {best['ExactMatch']}")
    print(f"{'='*60}")

    import os
    os.makedirs('results/lstmnps', exist_ok=True)
    df.to_csv("results/lstmnps/nps_adaptive_sweep.csv", index=False)
