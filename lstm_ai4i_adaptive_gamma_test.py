
import torch
import torch.multiprocessing as mp
import pandas as pd
import numbers
import os
import time
from datetime import datetime
from testclassifier.model import LSTM_MultiLabel, FocalLoss, AdaptiveWeightedFocalLoss, AdaptiveFocalLoss, BidirectionalAdaptiveFocalLoss
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

# Medium model for comparison (Upgrade from Tiny)
lstm_config = {
    "name": "LSTM-Medium-LR1e3-W100", 
    "hidden_dim": 64,  # Increased from 32
    "num_layers": 2,   # Increased from 1
    "lr": 1e-3, 
    "dropout": 0.2,    # Increased from 0.0
    "bidirectional": True, 
    "window_size": 100
}

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
        task_id = f"{config['name']} | Bidirectional G_init={loss_conf['gamma']} TR={loss_conf.get('target_recall')}"
    else:
        # Fallback
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        task_id = "Unknown"

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
            
        # Result Dictionary
        result = config.copy()
        
        # Default values from config
        final_alpha = loss_conf.get("alpha", loss_conf.get("base_alpha", 0.25))
        final_gamma = loss_conf.get("gamma", 2.0) # Initial gamma
        mean_recall = None
        
        # Override with adaptive logs if available
        if hasattr(loss_fn, "get_logs"):
            logs = loss_fn.get_logs()
            if "mean_alpha" in logs: final_alpha = logs["mean_alpha"]
            if "mean_gamma" in logs: final_gamma = logs["mean_gamma"] # Final gamma
            if "mean_recall" in logs: mean_recall = logs["mean_recall"]
            
        result.update({
            "Loss_Type": loss_type,
            "Init Gamma": loss_conf.get("gamma", 2.0),
            "Final Gamma": final_gamma, 
            "Final Alpha": final_alpha, 
            "Target Recall": loss_conf.get("target_recall", "-"),
            "Micro F1": micro_f1,
            "Duration": duration
        })
        
        print(f"✅ [PID {pid} | {device_str}] FINISHED: {task_id} | F1={micro_f1:.2f}%")
        return result
        
    except Exception as e:
        print(f"❌ [PID {pid}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Define configurations to compare
    tasks = []
    
    # Sensitivity Test for Initial Gamma
    # Using best config: Target Recall=0.9, Target Acc=0.95
    
    initial_gammas = [2.0, 3.0, 5.0, 1.0]
    
    for g in initial_gammas:
        tasks.append((lstm_config, {
            "type": "Bidirectional",
            "gamma": g,          # Varying Initial Gamma
            "base_alpha": 0.25,
            "decay": 0.99,
            "target_recall": 0.90, # Best performer
            "target_accuracy": 0.95,
            "alpha_gain": 2.0,
            "gamma_gain": 2.0
        }))

    # Baseline Static for reference
    tasks.append((lstm_config, {"type": "Static", "alpha": 0.25, "gamma": 2.0}))
            
    print(f"\n{'='*80}")
    print(f"AI4I LSTM (v2 - Medium): Gamma Sensitivity Test")
    print(f"Initial Gammas: {initial_gammas}")
    print(f"Target Recall: 0.90")
    print(f"{'='*80}\n")
    
    results = []
    if worker_device is None:
        worker_device = 0
        torch.cuda.set_device(worker_device)
        
    for task in tasks:
        res = run_single_experiment(task)
        if res: results.append(res)
    
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs("results/gamma_test", exist_ok=True)
    filename = f"results/gamma_test/gamma_sensitivity_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    print("\n" + "="*80)
    print("RESULTS: Gamma Sensitivity")
    print("="*80)
    cols = ["Loss_Type", "Init Gamma", "Final Gamma", "Final Alpha", "Micro F1"]
    print(df[cols].to_string(index=False))
    print(f"\nSaved to: {filename}")
