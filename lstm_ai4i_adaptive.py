
import torch
import torch.multiprocessing as mp
import pandas as pd
import numbers
import os
import time
from datetime import datetime
from testclassifier.model import LSTM_MultiLabel, FocalLoss, AdaptiveWeightedFocalLoss, FullAdaptiveFocalLoss, BidirectionalAdaptiveFocalLoss
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier
from datasets.multioutput import Ai4i
import evaluate
from river.compose import SelectType
from river.metrics import F1
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch, MicroAverage, MacroAverage
from river import preprocessing

# FORCE SPAWN for CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# GLOBAL VARIABLE for Worker Process
worker_device = None

def init_worker(device_queue):
    global worker_device
    worker_device = device_queue.get()
    torch.cuda.set_device(worker_device)
    pid = os.getpid()
    print(f"🔧 Worker Initialized: PID {pid} assigned to cuda:{worker_device}")

target_names = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Lighter model for speed
lstm_config = {
    "name": "LSTM-Tiny-LR1e3-W100", 
    "hidden_dim": 32, 
    "num_layers": 1, 
    "lr": 1e-3, 
    "dropout": 0.0, 
    "bidirectional": True, 
    "window_size": 100
}

# Adaptive Loss Configurations
# Compares standard Focal with Adaptive variations
# We can vary Gamma and Decay
adaptive_configs = [
    # Baseline Focal (Static) - as reference
    # {"type": "Static", "alpha": 0.25, "gamma": 2.0},
    
    # Adaptive
    {"type": "Adaptive", "gamma": 2.0, "decay": 0.99},
    {"type": "Adaptive", "gamma": 2.0, "decay": 0.999},
    {"type": "Adaptive", "gamma": 3.0, "decay": 0.99},
    {"type": "Adaptive", "gamma": 5.0, "decay": 0.99},
    {"type": "Adaptive", "gamma": 1.0, "decay": 0.99},
]

def run_single_experiment(args):
    """
    args is a tuple: (lstm_config_dict, adaptive_conf_dict)
    """
    config, loss_conf = args
    
    global worker_device
    device_id = worker_device
    device_str = f"cuda:{device_id}"
    pid = os.getpid()
    
    start_time = time.time()
    
    loss_type = loss_conf["type"]
    if loss_type == "Static":
        loss_fn = FocalLoss(alpha=loss_conf["alpha"], gamma=loss_conf["gamma"])
        task_id = f"{config['name']} | Static A={loss_conf['alpha']} G={loss_conf['gamma']}"
    elif loss_type == "FullAdaptive":
        loss_fn = FullAdaptiveFocalLoss(
            base_gamma=loss_conf["gamma"], 
            base_alpha=loss_conf.get("base_alpha", 0.25), 
            decay=loss_conf["decay"],
            alpha_gain=loss_conf.get("alpha_gain", 1.0),
            gamma_gain=loss_conf.get("gamma_gain", 2.0)
        )
        task_id = f"{config['name']} | FullAdaptive D={loss_conf['decay']} AG={loss_conf.get('alpha_gain', 1.0)}"
    elif loss_type == "Bidirectional":
        loss_fn = BidirectionalAdaptiveFocalLoss(
            base_gamma=loss_conf["gamma"], 
            base_alpha=loss_conf.get("base_alpha", 0.25), 
            decay=loss_conf["decay"],
            target_recall=loss_conf.get("target_recall", 0.75),
            target_accuracy=loss_conf.get("target_accuracy", 0.95),
            alpha_gain=loss_conf.get("alpha_gain", 1.0),
            gamma_gain=loss_conf.get("gamma_gain", 2.0)
        )
        task_id = f"{config['name']} | Bidirectional TR={loss_conf.get('target_recall')} TA={loss_conf.get('target_accuracy')}"
    else:
        base_beta = loss_conf.get("base_beta", 0.25)
        loss_fn = AdaptiveWeightedFocalLoss(gamma=loss_conf["gamma"], decay=loss_conf["decay"], base_beta=base_beta)
        task_id = f"{config['name']} | Adaptive G={loss_conf['gamma']} D={loss_conf['decay']} B={base_beta}"

    print(f"\n▶ [PID {pid} | {device_str}] STARTING: {task_id}")

    try:
        stream = Ai4i()
        
        clf = RollingMultiLabelClassifier(
            module=LSTM_MultiLabel,
            label_names=target_names,
            optimizer_fn="adam",
            lr=config['lr'],
            device=device_str,
            window_size=config['window_size'],
            append_predict=False,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            bidirectional=config['bidirectional'],
            output_dim=len(target_names),
            seed=42,
            epochs=1, # FAST
            loss_fn=loss_fn
        )

        pr = SelectType(numbers.Number) | preprocessing.StandardScaler()
        pr += SelectType(str) | preprocessing.OneHotEncoder()
        pipeliner = pr | clf

        metrics_result = evaluate.progressive_val_score(
            dataset=stream,
            model=pipeliner,
            metric=Metrics([ExactMatch(), MacroAverage(F1()), MicroAverage(F1())]),
            show_memory=False,
            print_every=1000 
        )

        micro_f1 = metrics_result[2].get() * 100
        duration = time.time() - start_time

        # Check for adaptive logs
        if hasattr(loss_fn, "get_logs"):
            logs = loss_fn.get_logs()
            print(f"📊 Adaptive Params: {logs}")

        print(f"✅ [PID {pid} | {device_str}] FINISHED: {task_id} | F1={micro_f1:.2f}% | {duration:.1f}s")
        
        # Result Dictionary
        result = config.copy()
        
        # Default values from config
        final_alpha = loss_conf.get("alpha", loss_conf.get("base_alpha", loss_conf.get("base_beta", 0.25)))
        final_gamma = loss_conf.get("gamma", 2.0)
        mean_recall = None
        mean_acc = None
        
        # Override with adaptive logs if available
        if hasattr(loss_fn, "get_logs"):
            logs = loss_fn.get_logs()
            if "mean_alpha" in logs: final_alpha = logs["mean_alpha"]
            if "mean_gamma" in logs: final_gamma = logs["mean_gamma"]
            if "mean_recall" in logs: mean_recall = logs["mean_recall"]
            if "mean_acc" in logs: mean_acc = logs["mean_acc"]
            
        result.update({
            "Loss_Type": loss_type,
            "Gamma": final_gamma, 
            "Alpha": final_alpha, 
            "Target Recall": loss_conf.get("target_recall", "-"),
            "Target Acc": loss_conf.get("target_accuracy", "-"),
            "Mean Recall": mean_recall,
            "Mean Acc": mean_acc,
            "Alpha_Decay": loss_conf.get("decay", "Fixed"),
            "Alpha_Gain": loss_conf.get("alpha_gain", 1.0),
            "Micro F1": micro_f1,
            "ExactMatch": metrics_result[0].get() * 100,
            "Duration": duration
        })
        
        return result
        
    except Exception as e:
        print(f"❌ [PID {pid}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Define configurations to compare
    tasks = []
    
    # 1. Static Focal Loss (Baseline)
    tasks.append((lstm_config, {"type": "Static", "alpha": 0.25, "gamma": 2.0}))
    
    # 2. Adaptive Focal Loss (Decay Search)
    decays = [0.9, 0.95, 0.99, 0.999]
    for d in decays:
        tasks.append((lstm_config, {"type": "Adaptive", "gamma": 2.0, "decay": d, "base_beta": 0.25}))
        
    # 3. Full Adaptive (Alpha + Gamma) - Gain Sweep
    # Using best decay (0.99) found previously
    # Default Gains: alpha_gain=1.0, gamma_gain=2.0
    
    # Sweep Alpha Gain: 1.0 (Standard), 5.0 (Aggressive), 10.0 (Very Aggressive)
    alpha_gains = [1.0, 5.0, 10.0]
    
    for ag in alpha_gains:
        tasks.append((lstm_config, {
            "type": "FullAdaptive", 
            "gamma": 2.0, 
            "decay": 0.99, 
            "base_alpha": 0.25,
            "alpha_gain": ag,
            "gamma_gain": 2.0 # Keep gamma gain standard for now since effect is small
        }))
            
    print(f"\n{'='*80}")
    # 4. Bidirectional Adaptive (NEW)
    # Test different target recalls and accuracy combinations
    # High Recall Target / Maintain Accuracy
    tasks.append((lstm_config, {
        "type": "Bidirectional", "gamma": 2.0, "base_alpha": 0.25, "decay": 0.99,
        "target_recall": 0.60, "target_accuracy": 0.95, "alpha_gain": 2.0, "gamma_gain": 2.0
    }))
    tasks.append((lstm_config, {
        "type": "Bidirectional", "gamma": 2.0, "base_alpha": 0.25, "decay": 0.99,
        "target_recall": 0.75, "target_accuracy": 0.95, "alpha_gain": 2.0, "gamma_gain": 2.0
    }))
    tasks.append((lstm_config, {
        "type": "Bidirectional", "gamma": 2.0, "base_alpha": 0.25, "decay": 0.99,
        "target_recall": 0.90, "target_accuracy": 0.95, "alpha_gain": 2.0, "gamma_gain": 2.0
    }))

    # High Accuracy Target vs Lower Accuracy Target 
    tasks.append((lstm_config, {
        "type": "Bidirectional", "gamma": 2.0, "base_alpha": 0.25, "decay": 0.99,
        "target_recall": 0.75, "target_accuracy": 0.90, "alpha_gain": 2.0, "gamma_gain": 2.0
    }))
    tasks.append((lstm_config, {
        "type": "Bidirectional", "gamma": 2.0, "base_alpha": 0.25, "decay": 0.99,
        "target_recall": 0.75, "target_accuracy": 0.99, "alpha_gain": 2.0, "gamma_gain": 2.0
    }))

    # Aggressive Gain
    tasks.append((lstm_config, {
        "type": "Bidirectional", "gamma": 2.0, "base_alpha": 0.25, "decay": 0.99,
        "target_recall": 0.75, "target_accuracy": 0.95, "alpha_gain": 5.0, "gamma_gain": 5.0
    }))

    # Also include best FullAdaptive for direct comparison
    tasks.append((lstm_config, {
        "type": "FullAdaptive", 
        "gamma": 2.0, 
        "decay": 0.99, 
        "base_alpha": 0.25,
        "alpha_gain": 5.0,
        "gamma_gain": 2.0 
    }))
            
    print(f"\n{'='*80}")
    print(f"AI4I LSTM (Tiny): Bidirectional Adaptive Comparison")
    # print(f"Target Recalls: {target_recalls}")
    print(f"Total Experiments: {len(tasks)}")
    print(f"{'='*80}\n")
    
    # Run sequentially
    results = []
    
    # Init worker for single process (set device)
    if worker_device is None:
        worker_device = 0 # Default to cuda:0
        torch.cuda.set_device(worker_device)
        print(f"🔧 Running on cuda:{worker_device}")
        
    for task in tasks:
        res = run_single_experiment(task)
        results.append(res)
    
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs("results/lstm_ai4i_adaptive", exist_ok=True)
    filename = f"results/lstm_ai4i_adaptive/comparison_{timestamp}.csv"
    df.to_csv(filename, index=False)
    # Print Summary
    print("\n" + "="*80)
    print("RESULTS Comparison")
    print("="*80)
    # Select specific columns
    # Select specific columns
    cols = ["Loss_Type", "Target Recall", "Gamma", "Alpha", "Mean Recall", "Micro F1", "ExactMatch"]
    # Add Alpha_Gain column if not present (fill with 1.0 for others)
    if "Alpha_Gain" not in df.columns:
        df["Alpha_Gain"] = 1.0
        
    print(df[cols].to_string(index=False))
    print("\n")    
    print(f"\nSaved to: {filename}")
