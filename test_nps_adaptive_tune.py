"""
Tuning fino de AdaptiveFocalLoss para batir Focal(α=0.25, γ=2.0) = 99.77 en NPS.
Arquitectura/training fijos = baseline del sweep anterior.
Solo se varían los parámetros del Adaptive.
"""
import torch
import pandas as pd
import numbers
import time
import multiprocessing
from testclassifier.model import LSTM_MultiLabel
from testclassifier.loss import AdaptiveFocalLoss
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

# ── Arquitectura / training FIJOS (baseline ganador anterior) ─────────────────
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = True
LR = 1e-3
WS = 200
PH = 1
EPOCHS = 1
THRESHOLD = 0.5
GRADIENT_SCALE = 1.0


# ── Sweep manual del Adaptive (10 configs) ────────────────────────────────────
# Partimos del Focal ganador: base_alpha=0.25, base_gamma=2.0
# Gains conservadores → Adaptive ≈ Focal cuando F1 es alto (NPS está saturado).
ADAPTIVE_VARIANTS = [
    # Adaptive ≈ Focal estático (gains casi nulos, decay alto)
    {"base_gamma": 2.0, "base_alpha": 0.25, "decay": 0.999,  "alpha_gain": 0.0, "gamma_gain": 0.0},
    # Solo modula alpha
    {"base_gamma": 2.0, "base_alpha": 0.25, "decay": 0.999,  "alpha_gain": 0.5, "gamma_gain": 0.0},
    {"base_gamma": 2.0, "base_alpha": 0.25, "decay": 0.999,  "alpha_gain": 1.0, "gamma_gain": 0.0},
    # Solo modula gamma
    {"base_gamma": 2.0, "base_alpha": 0.25, "decay": 0.999,  "alpha_gain": 0.0, "gamma_gain": 0.5},
    {"base_gamma": 2.0, "base_alpha": 0.25, "decay": 0.999,  "alpha_gain": 0.0, "gamma_gain": 1.0},
    # Modula ambos suave
    {"base_gamma": 2.0, "base_alpha": 0.25, "decay": 0.999,  "alpha_gain": 0.5, "gamma_gain": 0.5},
    {"base_gamma": 2.0, "base_alpha": 0.25, "decay": 0.999,  "alpha_gain": 1.0, "gamma_gain": 1.0},
    # Variar decay (ventana del EMA)
    {"base_gamma": 2.0, "base_alpha": 0.25, "decay": 0.99,   "alpha_gain": 0.5, "gamma_gain": 0.5},
    {"base_gamma": 2.0, "base_alpha": 0.25, "decay": 0.9999, "alpha_gain": 0.5, "gamma_gain": 0.5},
    # Empujar gamma base ligeramente arriba
    {"base_gamma": 2.5, "base_alpha": 0.25, "decay": 0.999,  "alpha_gain": 0.5, "gamma_gain": 0.5},
]


CONFIGS = [
    {
        "loss": "adaptive_focal",
        "params": v,
        "label": (
            f"adapt_g{v['base_gamma']}_a{v['base_alpha']}"
            f"_d{v['decay']}_ag{v['alpha_gain']}_gg{v['gamma_gain']}"
        ),
    }
    for v in ADAPTIVE_VARIANTS
]


def run_experiment(cfg, device, queue):
    try:
        label = cfg["label"]
        stream = NPS()

        loss_fn = AdaptiveFocalLoss(reduction='mean', **cfg["params"])

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
            "base_gamma": cfg["params"]["base_gamma"],
            "base_alpha": cfg["params"]["base_alpha"],
            "decay":      cfg["params"]["decay"],
            "alpha_gain": cfg["params"]["alpha_gain"],
            "gamma_gain": cfg["params"]["gamma_gain"],
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
    print(f"Arquitectura/training fijos: hid={HIDDEN_DIM} nl={NUM_LAYERS} drop={DROPOUT} bidir={BIDIRECTIONAL}")
    print(f"  lr={LR}  ws={WS}  ph={PH}  ep={EPOCHS}  th={THRESHOLD}")
    print(f"Baseline a batir: Focal(α=0.25, γ=2.0) → Micro/Macro F1 = 99.77\n")

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
    print("FULL RESULTS — NPS AdaptiveFocal tune — sorted by Micro F1")
    print("=" * 120)
    print(df.to_string(index=False))

    print(f"\n{'='*60}")
    if len(df) > 0:
        best = df.iloc[0]
        beat = "✓ BEATS" if best['Micro F1'] > 99.77 else "✗ no supera"
        print(f"BEST: {best['Config']}")
        print(f"  Micro F1   = {best['Micro F1']}  (Focal=99.77) {beat}")
        print(f"  Macro F1   = {best['Macro F1']}")
        print(f"  ExactMatch = {best['ExactMatch']}")
    print(f"{'='*60}")

    print(f"\n{'='*60}")
    print(f"BEST: focal_a0.25_g2.0")
    print(f"  Micro F1   = 99.77")
    print(f"  Macro F1   = 99.77")
    print(f"  ExactMatch = 99.16")
    print(f"{'='*60}")

    import os
    os.makedirs('results/lstmnps', exist_ok=True)
    df.to_csv("results/lstmnps/nps_adaptive_tune.csv", index=False)
