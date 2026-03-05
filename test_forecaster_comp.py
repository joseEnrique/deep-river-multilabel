import torch
import pandas as pd
import numbers
import os
import time
from testclassifier.model import LSTM_MultiLabel, FocalLoss
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier
from classes.direct_multilabel_forecaster import DirectMultiLabelForecaster
from datasets.multioutput import Ai4i
import evaluate
from river.compose import SelectType
from river.metrics import F1
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch, MicroAverage, MacroAverage
from river import preprocessing

target_names = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
window_size = 50

def run_experiment(model_type, past_history=1, device_str="cpu"):
    name = f"{model_type} (past={past_history})" if model_type == "DirectMultiLabelForecaster" else model_type
    print(f"\\n▶ STARTING: {name}")
    start_time = time.time()
    
    stream = Ai4i()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    if model_type == "RollingMultiLabelClassifier":
        clf = RollingMultiLabelClassifier(
            module=LSTM_MultiLabel,
            label_names=target_names,
            optimizer_fn="adam",
            lr=1e-3,
            device=device_str,
            window_size=window_size,
            append_predict=False,
            hidden_dim=64,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            output_dim=len(target_names),
            seed=42,
            epochs=1,
            loss_fn=loss_fn
        )
    elif model_type == "DirectMultiLabelForecaster":
        clf = DirectMultiLabelForecaster(
            window_size=window_size,
            past_history=past_history,
            label_names=target_names,
            module=LSTM_MultiLabel,
            optimizer_fn="adam",
            lr=1e-3,
            device=device_str,
            hidden_dim=64,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            output_dim=len(target_names),
            seed=42,
            epochs=1,
            loss_fn=loss_fn,
            shift=0
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pr = SelectType(numbers.Number) | preprocessing.StandardScaler()
    pr += SelectType(str) | preprocessing.OneHotEncoder()
    pipeliner = pr | clf

    metrics_result = evaluate.progressive_val_score(
        dataset=stream,
        model=pipeliner,
        metric=Metrics([ExactMatch(), MacroAverage(F1()), MicroAverage(F1())]),
        show_memory=False,
        print_every=2000 
    )

    exact_match = metrics_result[0].get() * 100
    macro_f1 = metrics_result[1].get() * 100
    micro_f1 = metrics_result[2].get() * 100
    duration = time.time() - start_time
    
    print(f"✅ FINISHED: {name}")
    print(f"   Exact Match: {exact_match:.2f}%")
    print(f"   Macro F1:    {macro_f1:.2f}%")
    print(f"   Micro F1:    {micro_f1:.2f}%")
    print(f"   Time:        {duration:.1f}s")
    
    return {
        "Model": name,
        "Exact Match (%)": round(exact_match, 2),
        "Macro F1 (%)": round(macro_f1, 2),
        "Micro F1 (%)": round(micro_f1, 2),
        "Time (s)": round(duration, 1)
    }

if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str}")
    
    results = []
    
    # Run Rolling
    res = run_experiment("RollingMultiLabelClassifier", device_str=device_str)
    results.append(res)
    
    # Run Direct with different past_histories
    for ph in [1, 2, 3]:
        res = run_experiment("DirectMultiLabelForecaster", past_history=ph, device_str=device_str)
        results.append(res)
        
    # Print Comparative Table
    print("\\n" + "="*70)
    print("COMPARATIVE RESULTS TABLE")
    print("="*70)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("="*70)
