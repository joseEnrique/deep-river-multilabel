import torch
import pandas as pd
import numbers
import time
from testclassifier.model import LSTM_MultiLabel, FullAdaptiveFocalLoss
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier
from classes.rolling_multilabel_classifier_sequences import RollingMultiLabelClassifierSequences
from datasets.multioutput import Ai4i
import evaluate
from river.compose import SelectType
from river.metrics import F1, Precision, Recall
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch, MicroAverage, MacroAverage
from metrics import HammingLoss, ExampleF1, ExamplePrecision, ExampleRecall
from river import preprocessing

target_names = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
window_size = 200

def run_experiment(model_type, past_history=1, device_str="cpu"):
    name = f"{model_type} (past={past_history})" if model_type == "RollingMultiLabelClassifierSequences" else model_type
    print(f"\\n▶ STARTING: {name}")
    start_time = time.time()

    stream = Ai4i()
    loss_fn = FullAdaptiveFocalLoss()

    if model_type == "RollingMultiLabelClassifier":
        clf = RollingMultiLabelClassifier(
            module=LSTM_MultiLabel,
            label_names=target_names,
            optimizer_fn="adam",
            lr=1e-3,
            device=device_str,
            window_size=window_size,
            append_predict=False,
            hidden_dim=256,
            num_layers=3,
            dropout=0.3,
            bidirectional=False,
            output_dim=len(target_names),
            seed=42,
            epochs=1,
            loss_fn=loss_fn
        )
    elif model_type == "RollingMultiLabelClassifierSequences":
        clf = RollingMultiLabelClassifierSequences(
            window_size=window_size,
            past_history=past_history,
            label_names=target_names,
            module=LSTM_MultiLabel,
            optimizer_fn="adam",
            lr=1e-3,
            device=device_str,
            hidden_dim=256,
            num_layers=3,
            dropout=0.3,
            bidirectional=False,
            output_dim=len(target_names),
            seed=42,
            epochs=1,
            loss_fn=loss_fn
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pr = SelectType(numbers.Number) | preprocessing.StandardScaler()
    pr += SelectType(str) | preprocessing.OneHotEncoder()
    pipeliner = pr | clf

    # All 11 metrics from the paper table
    all_metrics = Metrics([
        ExactMatch(),           # 0  - Subset acc
        HammingLoss(),          # 1  - Hamming loss
        ExampleF1(),            # 2  - Examp F1
        ExamplePrecision(),     # 3  - Examp prec
        ExampleRecall(),        # 4  - Examp rec
        MicroAverage(F1()),     # 5  - Micro F1
        MicroAverage(Precision()),  # 6  - Micro prec
        MicroAverage(Recall()),     # 7  - Micro rec
        MacroAverage(F1()),         # 8  - Macro F1
        MacroAverage(Precision()),  # 9  - Macro prec
        MacroAverage(Recall()),     # 10 - Macro rec
    ])

    metrics_result = evaluate.progressive_val_score(
        dataset=stream,
        model=pipeliner,
        metric=all_metrics,
        show_memory=False,
        print_every=2000
    )

    m = metrics_result
    duration = time.time() - start_time

    result = {
        "Model":        name,
        "Subset acc":   round(m[0].get() * 100, 2),
        "Hamm loss":    round(m[1].get() * 100, 2),
        "Examp F1":     round(m[2].get() * 100, 2),
        "Examp prec":   round(m[3].get() * 100, 2),
        "Examp rec":    round(m[4].get() * 100, 2),
        "Micro F1":     round(m[5].get() * 100, 2),
        "Micro prec":   round(m[6].get() * 100, 2),
        "Micro rec":    round(m[7].get() * 100, 2),
        "Macro F1":     round(m[8].get() * 100, 2),
        "Macro prec":   round(m[9].get() * 100, 2),
        "Macro rec":    round(m[10].get() * 100, 2),
        "Time (s)":     round(duration, 1),
    }

    print(f"✅ FINISHED: {name}  ({duration:.1f}s)")
    for k, v in result.items():
        if k != "Model":
            print(f"   {k}: {v}")
    return result


if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str}")

    results = []

    res = run_experiment("RollingMultiLabelClassifier", device_str=device_str)
    results.append(res)

    for ph in [1]:
        res = run_experiment("RollingMultiLabelClassifierSequences", past_history=ph, device_str=device_str)
        results.append(res)

    print("\n" + "="*150)
    print("COMPARATIVE RESULTS TABLE")
    print("="*150)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("="*150)
