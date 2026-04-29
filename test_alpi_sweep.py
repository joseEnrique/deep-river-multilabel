"""
Systematic sweep: Adam + BCE(mean) to beat OEMLHAT Machine 2.
OEMLHAT: Micro F1=67.59, Macro F1=58.02
Uses cuda:0 and cuda:1 in parallel.
"""
import torch
import pandas as pd
import numbers
import time
import multiprocessing
from stateful_multilabel_classifier import OnlineLSTM
from classes.rolling_multilabel_classifier_sequences import RollingMultiLabelClassifierSequences
from datasets.multioutput.newalpi import NewAlpi
from river.compose import SelectType
from river.metrics import F1
from river.metrics.base import Metrics
from custommetrics.multioutput import MicroAverage, MacroAverage
import evaluate


CONFIGS = [
    # --- Gradient scale: the key insight. Adam + mean needs gs to compensate ---
    # With sum+gs=10 we get ~79/71. Mean divides by n_labels(54), so gs needs to be higher.
    {"ws": 1,  "lr": 1e-3, "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.5, "gs": 50.0,  "label": "gs50_lr1e-3"},
    {"ws": 1,  "lr": 1e-3, "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.5, "gs": 100.0, "label": "gs100_lr1e-3"},
    {"ws": 1,  "lr": 1e-3, "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.5, "gs": 500.0, "label": "gs500_lr1e-3"},
    {"ws": 1,  "lr": 1e-2, "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.5, "gs": 50.0,  "label": "gs50_lr1e-2"},
    {"ws": 1,  "lr": 1e-2, "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.5, "gs": 100.0, "label": "gs100_lr1e-2"},
    {"ws": 1,  "lr": 1e-1, "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.5, "gs": 10.0,  "label": "gs10_lr1e-1"},
    {"ws": 1,  "lr": 1e-1, "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.5, "gs": 50.0,  "label": "gs50_lr1e-1"},

    # --- No gradient scale, just high LR ---
    {"ws": 1,  "lr": 1.0,  "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.5, "gs": 1.0,   "label": "gs1_lr1"},
    {"ws": 1,  "lr": 5.0,  "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.5, "gs": 1.0,   "label": "gs1_lr5"},
    {"ws": 1,  "lr": 10.0, "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.5, "gs": 1.0,   "label": "gs1_lr10"},

    # --- Window size with moderate lr ---
    {"ws": 10, "lr": 1e-2, "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.5, "gs": 1.0,   "label": "ws10_lr1e-2"},
    {"ws": 10, "lr": 1e-1, "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.5, "gs": 1.0,   "label": "ws10_lr1e-1"},
    {"ws": 50, "lr": 1e-2, "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.5, "gs": 1.0,   "label": "ws50_lr1e-2"},
    {"ws": 50, "lr": 1e-1, "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.5, "gs": 1.0,   "label": "ws50_lr1e-1"},
    {"ws": 100,"lr": 1e-2, "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.5, "gs": 1.0,   "label": "ws100_lr1e-2"},
    {"ws": 100,"lr": 1e-1, "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.5, "gs": 1.0,   "label": "ws100_lr1e-1"},

    # --- Bigger model ---
    {"ws": 1,  "lr": 1e-1, "emb": 32, "hid": 64,  "ep": 1,  "th": 0.5, "gs": 1.0,   "label": "big_gs1_lr0.1"},
    {"ws": 1,  "lr": 1.0,  "emb": 32, "hid": 64,  "ep": 1,  "th": 0.5, "gs": 1.0,   "label": "big_gs1_lr1"},
    {"ws": 1,  "lr": 5.0,  "emb": 32, "hid": 64,  "ep": 1,  "th": 0.5, "gs": 1.0,   "label": "big_gs1_lr5"},
    {"ws": 10, "lr": 1e-1, "emb": 32, "hid": 64,  "ep": 1,  "th": 0.5, "gs": 1.0,   "label": "big_ws10_lr0.1"},
    {"ws": 10, "lr": 1e-2, "emb": 32, "hid": 64,  "ep": 1,  "th": 0.5, "gs": 1.0,   "label": "big_ws10_lr0.01"},
    {"ws": 50, "lr": 1e-2, "emb": 32, "hid": 64,  "ep": 1,  "th": 0.5, "gs": 1.0,   "label": "big_ws50_lr0.01"},

    # --- More epochs ---
    {"ws": 1,  "lr": 1.0,  "emb": 8,  "hid": 10,  "ep": 5,  "th": 0.5, "gs": 1.0,   "label": "ep5_lr1"},
    {"ws": 1,  "lr": 5.0,  "emb": 8,  "hid": 10,  "ep": 5,  "th": 0.5, "gs": 1.0,   "label": "ep5_lr5"},
    {"ws": 10, "lr": 1e-1, "emb": 8,  "hid": 10,  "ep": 5,  "th": 0.5, "gs": 1.0,   "label": "ep5_ws10_lr0.1"},

    # --- Threshold sweep on best baseline ---
    {"ws": 1,  "lr": 5.0,  "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.3, "gs": 1.0,   "label": "th0.3_lr5"},
    {"ws": 1,  "lr": 5.0,  "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.25,"gs": 1.0,   "label": "th0.25_lr5"},
    {"ws": 1,  "lr": 5.0,  "emb": 8,  "hid": 10,  "ep": 1,  "th": 0.4, "gs": 1.0,   "label": "th0.4_lr5"},
]


def run_experiment(cfg, device, queue):
    try:
        label = cfg["label"]
        stream = NewAlpi(machine=2, input_win=1720, output_win=480, delta=0, sigma=120)
        stream.Y.columns = stream.Y.columns.astype(str)
        label_names = list(stream.Y.columns)

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

        clf = RollingMultiLabelClassifierSequences(
            window_size=cfg["ws"],
            past_history=1,
            label_names=label_names,
            module=OnlineLSTM,
            optimizer_fn="adam",
            lr=cfg["lr"],
            device=device,
            embedding_dim=cfg["emb"],
            hidden_size=cfg["hid"],
            output_dim=len(label_names),
            seed=42,
            epochs=cfg["ep"],
            loss_fn=loss_fn,
            threshold=cfg["th"],
            gradient_scale=cfg["gs"],
        )

        pipeline = SelectType(numbers.Number) | clf
        all_metrics = Metrics([MicroAverage(F1()), MacroAverage(F1())])

        start = time.time()
        m = evaluate.progressive_val_score(
            dataset=stream, model=pipeline, metric=all_metrics,
            show_memory=False, print_every=5000,
        )
        duration = time.time() - start

        result = {
            "Config":   label,
            "WS":       cfg["ws"],
            "LR":       cfg["lr"],
            "Emb":      cfg["emb"],
            "Hid":      cfg["hid"],
            "Ep":       cfg["ep"],
            "Th":       cfg["th"],
            "GS":       cfg["gs"],
            "Micro F1": round(m[0].get() * 100, 2),
            "Macro F1": round(m[1].get() * 100, 2),
            "Time (s)": round(duration, 1),
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

    gpus = []
    if torch.cuda.device_count() >= 2:
        gpus = ["cuda:0", "cuda:1"]
    elif torch.cuda.device_count() == 1:
        gpus = ["cuda:0"]
    else:
        gpus = ["cpu"]

    print(f"GPUs: {gpus}")
    print(f"Total configs: {len(CONFIGS)}")
    print(f"Target: OEMLHAT Machine 2 → Micro F1=67.59, Macro F1=58.02\n")

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

    # Drain queue
    while not queue.empty():
        res = queue.get()
        if res:
            results.append(res)

    df = pd.DataFrame(results)
    df = df.sort_values("Micro F1", ascending=False)

    print("\n" + "=" * 120)
    print("FULL RESULTS — Adam + BCE(mean) — sorted by Micro F1")
    print("=" * 120)
    print(df.to_string(index=False))

    print(f"\n{'='*60}")
    if len(df) > 0:
        best = df.iloc[0]
        beat_micro = "✓ BEATS" if best['Micro F1'] > 67.59 else "✗ below"
        beat_macro = "✓ BEATS" if best['Macro F1'] > 58.02 else "✗ below"
        print(f"BEST: {best['Config']}")
        print(f"  Micro F1 = {best['Micro F1']}  (OEMLHAT=67.59) {beat_micro}")
        print(f"  Macro F1 = {best['Macro F1']}  (OEMLHAT=58.02) {beat_macro}")
    print(f"{'='*60}")

    import os
    os.makedirs('results', exist_ok=True)
    df.to_csv("results/alpi_adam_mean_sweep_machine2.csv", index=False)
    print("Saved to results/alpi_adam_mean_sweep_machine2.csv")
