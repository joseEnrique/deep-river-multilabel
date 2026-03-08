
import torch
import torch.multiprocessing as mp
import pandas as pd
import numbers
import os
import time
from datetime import datetime
from testclassifier.model import LSTM_MultiLabel, FocalLoss, AdaptiveWeightedFocalLoss, AdaptiveFocalLoss, BidirectionalAdaptiveFocalLoss, LearnableFocalLoss, RobustFocalLoss
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier
from datasets.multioutput import Ai4i
import evaluate
from river.compose import SelectType
from river.metrics import F1
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch, MicroAverage, MacroAverage
from custommetrics.multioutput import HammingLoss
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
    if loss_type == "BCE":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        task_id = f"{config['name']} | BCE"
    elif loss_type == "WCE":
        # Weight for positive class. 
        # Calculate pos_weight from alpha if provided or use direct pos_weight
        if "pos_weight" in loss_conf:
            pw = loss_conf["pos_weight"]
        elif "alpha" in loss_conf:
            # Convert alpha to pos_weight: w = alpha / (1 - alpha)
            # e.g alpha=0.75 -> 0.75/0.25 = 3.0
            alpha = loss_conf["alpha"]
            pw = alpha / (1.0 - alpha + 1e-6) # Avoid div by zero
        else:
            pw = 1.0
            
        # Ensure tensor is on correct device
        # We need a tensor of shape [num_classes]
        pos_weight_tensor = torch.full((len(target_names),), pw, device=device_str)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        task_id = f"{config['name']} | WCE PW={pw:.2f}"
    elif loss_type == "Static":
        loss_fn = FocalLoss(alpha=loss_conf["alpha"], gamma=loss_conf["gamma"])
        task_id = f"{config['name']} | Static A={loss_conf['alpha']} G={loss_conf['gamma']}"
    elif loss_type == "FullAdaptive":
        loss_fn = AdaptiveFocalLoss(
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
    elif loss_type == "LearnableFocalLoss":
        loss_fn = LearnableFocalLoss(
            init_gamma=loss_conf.get("init_gamma", 2.0),
            init_alpha=loss_conf.get("init_alpha", 0.25),
            reduction="mean"
        )
        task_id = f"{config['name']} | Learnable InitAlpha={loss_conf.get('init_alpha', 0.25)}"
    elif loss_type == "RobustFocal":
        loss_fn = RobustFocalLoss(
            base_gamma=loss_conf.get("gamma", 2.0),
            base_alpha=loss_conf.get("base_alpha", 0.25),
            momentum=loss_conf.get("momentum", 0.9),
            max_gain=loss_conf.get("max_gain", 2.0),
            anchor_weight=loss_conf.get("anchor_weight", 0.1)
        )
        task_id = f"{config['name']} | RobustFocal M={loss_conf.get('momentum', 0.9)} AW={loss_conf.get('anchor_weight', 0.1)}"
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

        metric = Metrics([ExactMatch(), MacroAverage(F1()), MicroAverage(F1()), HammingLoss()])
        
        # Use iter_progressive_val_score to capture metrics every 500 steps
        checkpoints_data = []
        for checkpoint in evaluate.iter_progressive_val_score(
            dataset=stream,
            model=pipeliner,
            metric=metric,
            step=500,
            measure_time=True
        ):
            step_n = checkpoint['Step']
            elapsed = checkpoint['Time'].total_seconds()
            exact_match_val = metric[0].get() * 100
            macro_f1_val = metric[1].get() * 100
            micro_f1_val = metric[2].get() * 100
            hamming_val = metric[3].get()
            
            acc_str = ""
            if hasattr(loss_fn, "get_logs"):
                logs_tmp = loss_fn.get_logs()
                if "mean_acc" in logs_tmp:
                    acc_str = f" | MeanAcc={logs_tmp['mean_acc']:.4f}"
            print(f"  [{step_n:,d}] ExactMatch={exact_match_val:.2f}% | MacroF1={macro_f1_val:.2f}% | MicroF1={micro_f1_val:.2f}% | HammingLoss={hamming_val:.4f}{acc_str} | Time={elapsed:.1f}s")
            
            checkpoint_row = {
                "Experiment_Name": task_id,
                "Step": step_n,
                "Loss_Type": loss_type,
                "Target_Recall": loss_conf.get("target_recall", "-"),
                "Target_Accuracy": loss_conf.get("target_accuracy", "-"),
                "Elapsed_s": elapsed,
                "ExactMatch": exact_match_val,
                "MacroF1": macro_f1_val,
                "MicroF1": micro_f1_val,
                "HammingLoss": hamming_val,
            }
            # Include adaptive params if available
            if hasattr(loss_fn, "get_logs"):
                logs = loss_fn.get_logs()
                checkpoint_row["Alpha"] = logs.get("mean_alpha", None)
                checkpoint_row["Gamma"] = logs.get("mean_gamma", None)
                checkpoint_row["Recall"] = logs.get("mean_recall", None)
                checkpoint_row["Acc"] = logs.get("mean_acc", None)
            
            checkpoints_data.append(checkpoint_row)
        
        metrics_result = metric
        
        # Save checkpoint history to CSV
        os.makedirs("results/lstm_ai4i_charts", exist_ok=True)
        safe_task_id = task_id.replace(" | ", "_").replace("=", "").replace(" ", "_")
        
        # Include hidden_dim and lr in filename if not already present in safe_task_id 
        # (safe_task_id comes from task_id which comes from config['name'])
        # Current config['name'] is f"LSTM-H{h_dim}-LR{lr}-W{w_size}"
        
        checkpoint_filename = f"results/lstm_ai4i_charts/metrics_every500_{safe_task_id}.csv"
        pd.DataFrame(checkpoints_data).to_csv(checkpoint_filename, index=False)
        print(f"📄 Metrics history saved to: {checkpoint_filename}")

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
            "Macro F1": metrics_result[1].get() * 100,
            "Micro F1": micro_f1,
            "ExactMatch": metrics_result[0].get() * 100,
            "HammingLoss": metrics_result[3].get(),
            "Duration": duration
        })
        
        return result
        
    except Exception as e:
        print(f"❌ [PID {pid}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    window_sizes = [1, 50, 100, 200]
    learning_rates = [1e-2, 1e-3, 1e-4]
    hidden_dims = [32, 64, 128]

    # Init worker for single process (set device)
    if worker_device is None:
        worker_device = 0 # Default to cuda:0
        torch.cuda.set_device(worker_device)
        print(f"🔧 Running on cuda:{worker_device}")
    
    for w_size in window_sizes:
        for lr in learning_rates:
            for h_dim in hidden_dims:
                print(f"\n{'#'*80}")
                print(f"Running Experiments | W={w_size} | LR={lr} | H={h_dim}")
                print(f"{'#'*80}\n")
                
                # Update config for this combination
                current_config = lstm_config.copy()
                current_config["window_size"] = w_size
                current_config["lr"] = lr
                current_config["hidden_dim"] = h_dim
                
                # Format LR for filename to avoid scientific notation if possible or clean it
                lr_str = str(lr).replace(".", "") if "e" not in str(lr) else str(lr).replace(".", "").replace("-", "")
                # Better formatting: 1e-3 -> 1e3, 0.01 -> 1e2
                if lr == 1e-2: lr_str = "1e2"
                elif lr == 1e-3: lr_str = "1e3"
                elif lr == 1e-4: lr_str = "1e4"
                else: lr_str = str(lr)
                
                current_config["name"] = f"LSTM-H{h_dim}-LR{lr_str}-W{w_size}"
                
                # Define configurations to compare
                tasks = []
                
                # 0. BCE (Standard Baseline)
                tasks.append((current_config, {"type": "BCE"}))

                # 0.5 WCE (Weighted Baseline)
                # Alpha 0.75 equivalent -> pos_weight=3.0
                tasks.append((current_config, {"type": "WCE", "pos_weight": 3.0}))

                # 1. Static Focal Loss (Baseline)
                tasks.append((current_config, {"type": "Static", "alpha": 0.25, "gamma": 2.0}))
                
                # 4. Bidirectional Adaptive
                # High Recall Target / Maintain Accuracy
                tasks.append((current_config, {
                    "type": "Bidirectional", "gamma": 2.0, "base_alpha": 0.25, "decay": 0.99,
                    "target_recall": 0.60, "target_accuracy": 0.95, "alpha_gain": 2.0, "gamma_gain": 2.0
                }))
                tasks.append((current_config, {
                    "type": "Bidirectional", "gamma": 2.0, "base_alpha": 0.25, "decay": 0.99,
                    "target_recall": 0.75, "target_accuracy": 0.95, "alpha_gain": 2.0, "gamma_gain": 2.0
                }))
                tasks.append((current_config, {
                    "type": "Bidirectional", "gamma": 2.0, "base_alpha": 0.25, "decay": 0.99,
                    "target_recall": 0.90, "target_accuracy": 0.95, "alpha_gain": 2.0, "gamma_gain": 2.0
                }))

                # High Accuracy Target vs Lower Accuracy Target 
                tasks.append((current_config, {
                    "type": "Bidirectional", "gamma": 2.0, "base_alpha": 0.25, "decay": 0.99,
                    "target_recall": 0.75, "target_accuracy": 0.90, "alpha_gain": 2.0, "gamma_gain": 2.0
                }))
                tasks.append((current_config, {
                    "type": "Bidirectional", "gamma": 2.0, "base_alpha": 0.25, "decay": 0.99,
                    "target_recall": 0.75, "target_accuracy": 0.99, "alpha_gain": 2.0, "gamma_gain": 2.0
                }))

                # Aggressive Gain
                tasks.append((current_config, {
                    "type": "Bidirectional", "gamma": 2.0, "base_alpha": 0.25, "decay": 0.99,
                    "target_recall": 0.75, "target_accuracy": 0.95, "alpha_gain": 5.0, "gamma_gain": 5.0
                }))

                # Also include best FullAdaptive for direct comparison
                tasks.append((current_config, {
                    "type": "FullAdaptive", 
                    "gamma": 2.0, 
                    "decay": 0.99, 
                    "base_alpha": 0.25,
                    "alpha_gain": 5.0,
                    "gamma_gain": 2.0 
                }))

                # 6. FullAdaptive Optimal (Base Alpha=0.5)
                tasks.append((current_config, {
                    "type": "FullAdaptive", 
                    "gamma": 2.0, 
                    "decay": 0.99, 
                    "base_alpha": 0.5,
                    "alpha_gain": 2.0,
                    "gamma_gain": 2.0 
                }))
                
                # 7. Learnable Focal Loss (Winnder from Optimality Analysis)
                tasks.append((current_config, {
                    "type": "LearnableFocalLoss",
                    "init_gamma": 2.0,
                    "init_alpha": 0.5,
                    "reduction": "mean"
                }))

                # 8. Bidirectional High Acc (from Optimality Analysis)
                tasks.append((current_config, {
                    "type": "Bidirectional", "gamma": 2.0, "base_alpha": 0.5, "decay": 0.99,
                    "target_recall": 0.75, "target_accuracy": 0.98, "alpha_gain": 1.0, "gamma_gain": 2.0
                }))

                # 9. RobustFocalLoss (The Champion)
                tasks.append((current_config, {
                    "type": "RobustFocal",
                    "gamma": 2.0, "base_alpha": 0.25,
                    "momentum": 0.9, "max_gain": 2.0, "anchor_weight": 0.1
                }))

                        
                print(f"\nExperiments | H={h_dim} | LR={lr_str} | W={w_size}")
                print(f"Total Tasks: {len(tasks)}")
                print(f"{'='*80}\n")
                
                # Run sequentially
                results = []
                
                for task in tasks:
                    res = run_single_experiment(task)
                    if res: results.append(res)
                
                df = pd.DataFrame(results)
                os.makedirs("results/lstm_ai4i_charts", exist_ok=True)
                filename = f"results/lstm_ai4i_charts/comparison_bidirectional_W{w_size}_LR{lr_str}_H{h_dim}.csv"
                df.to_csv(filename, index=False)
                # Print Summary (brief)
                print(f"Saved summary to: {filename}")

