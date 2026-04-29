"""
Sweep 3: Fine-tuning alrededor de best config con BCE(mean) y AdaptiveFocal.
Adam + loss ∈ {bce, adaptive_focal}. Target: OEMLHAT Micro=67.59, Macro=58.02
"""
import torch
import pandas as pd
import numbers
import time
import multiprocessing
from stateful_multilabel_classifier import OnlineLSTM
from classes.rolling_multilabel_classifier_sequences import RollingMultiLabelClassifierSequences
from datasets.multioutput.newalpi import NewAlpi
from testclassifier.loss import AdaptiveFocalLoss
from river.compose import SelectType
from river.metrics import F1
from river.metrics.base import Metrics
from custommetrics.multioutput import MicroAverage, MacroAverage
import evaluate


CONFIGS = [
    {"ws": ws, "lr": 0.5, "emb": 8, "hid": 10, "ep": ep, "th": 0.5, "gs": 1.0, "ph": ph,
     "loss": loss,
     "label": f"ws{ws}_ph{ph}_ep{ep}_lr0.5_{loss}"}
    for ws in [1, 10, 50, 200, 500]
    for ph in [1, 5, 10]
    for ep in [1, 5, 10, 20]
    for loss in ["bce", "adaptive_focal"]
]


def build_loss(kind):
    if kind == "bce":
        return torch.nn.BCEWithLogitsLoss(reduction='mean')
    if kind == "adaptive_focal":
        return AdaptiveFocalLoss(
            base_gamma=2.0, base_alpha=0.25, decay=0.999,
            alpha_gain=1.0, gamma_gain=2.0, reduction='mean',
        )
    raise ValueError(f"Loss desconocida: {kind}")


def run_experiment(cfg, device, queue):
    try:
        label = cfg["label"]
        stream = NewAlpi(machine=2, input_win=1720, output_win=480, delta=0, sigma=120)
        stream.Y.columns = stream.Y.columns.astype(str)
        label_names = list(stream.Y.columns)

        loss_fn = build_loss(cfg["loss"])

        clf = RollingMultiLabelClassifierSequences(
            window_size=cfg["ws"],
            past_history=cfg["ph"],
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
            "Loss":     cfg["loss"],
            "PH":       cfg["ph"],
            "WS":       cfg["ws"],
            "LR":       cfg["lr"],
            "Emb":      cfg["emb"],
            "Hid":      cfg["hid"],
            "Ep":       cfg["ep"],
            "Th":       cfg["th"],
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
    df.to_csv("results/alpi_adam_mean_sweep3_machine2.csv", index=False)
