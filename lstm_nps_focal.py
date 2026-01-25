
import torch
import torch.multiprocessing as mp
import pandas as pd
import numbers
import os
import time
from datetime import datetime
from testclassifier.model import LSTM_MultiLabel, FocalLoss
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier
from datasets.multioutput.nps import NPS
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

target_names = ['PRP', 'HLL', 'GTC', 'GT']

# 1. Base LSTM Configurations (from lstm_window_size_search.py)
lstm_configs = [
    # SMALL
    {"name": "LSTM-Small-LR1e3-W100", "hidden_dim": 64, "num_layers": 1, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-Small-LR1e3-W200", "hidden_dim": 64, "num_layers": 1, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-Small-LR1e3-W500", "hidden_dim": 64, "num_layers": 1, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 500},
    {"name": "LSTM-Small-LR5e4-W100", "hidden_dim": 64, "num_layers": 1, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-Small-LR5e4-W200", "hidden_dim": 64, "num_layers": 1, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-Small-LR5e4-W500", "hidden_dim": 64, "num_layers": 1, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500},
    {"name": "LSTM-Small-LR1e4-W100", "hidden_dim": 64, "num_layers": 1, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-Small-LR1e4-W200", "hidden_dim": 64, "num_layers": 1, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-Small-LR1e4-W500", "hidden_dim": 64, "num_layers": 1, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500},
    
    # MEDIUM
    {"name": "LSTM-Medium-LR1e3-W100", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-Medium-LR1e3-W200", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-Medium-LR1e3-W500", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 500},
    {"name": "LSTM-Medium-LR5e4-W100", "hidden_dim": 128, "num_layers": 2, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-Medium-LR5e4-W200", "hidden_dim": 128, "num_layers": 2, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-Medium-LR5e4-W500", "hidden_dim": 128, "num_layers": 2, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500},
    {"name": "LSTM-Medium-LR1e4-W100", "hidden_dim": 128, "num_layers": 2, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-Medium-LR1e4-W200", "hidden_dim": 128, "num_layers": 2, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-Medium-LR1e4-W500", "hidden_dim": 128, "num_layers": 2, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500},
    {"name": "LSTM-Medium-LR5e5-W100", "hidden_dim": 128, "num_layers": 2, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-Medium-LR5e5-W200", "hidden_dim": 128, "num_layers": 2, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-Medium-LR5e5-W500", "hidden_dim": 128, "num_layers": 2, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 500},

    # LARGE
    {"name": "LSTM-Large-LR1e3-W100", "hidden_dim": 256, "num_layers": 2, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-Large-LR1e3-W200", "hidden_dim": 256, "num_layers": 2, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-Large-LR1e3-W500", "hidden_dim": 256, "num_layers": 2, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 500},
    {"name": "LSTM-Large-LR5e4-W100", "hidden_dim": 256, "num_layers": 2, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-Large-LR5e4-W200", "hidden_dim": 256, "num_layers": 2, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-Large-LR5e4-W500", "hidden_dim": 256, "num_layers": 2, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500},
    {"name": "LSTM-Large-LR1e4-W100", "hidden_dim": 256, "num_layers": 2, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-Large-LR1e4-W200", "hidden_dim": 256, "num_layers": 2, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-Large-LR1e4-W500", "hidden_dim": 256, "num_layers": 2, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500},
    {"name": "LSTM-Large-LR5e5-W100", "hidden_dim": 256, "num_layers": 2, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-Large-LR5e5-W200", "hidden_dim": 256, "num_layers": 2, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-Large-LR5e5-W500", "hidden_dim": 256, "num_layers": 2, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 500},

    # XLARGE
    {"name": "LSTM-XLarge-LR5e4-W100", "hidden_dim": 256, "num_layers": 3, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-XLarge-LR5e4-W200", "hidden_dim": 256, "num_layers": 3, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-XLarge-LR5e4-W500", "hidden_dim": 256, "num_layers": 3, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500},
    {"name": "LSTM-XLarge-LR1e4-W100", "hidden_dim": 256, "num_layers": 3, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-XLarge-LR1e4-W200", "hidden_dim": 256, "num_layers": 3, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-XLarge-LR1e4-W500", "hidden_dim": 256, "num_layers": 3, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500},
    {"name": "LSTM-XLarge-LR5e5-W100", "hidden_dim": 256, "num_layers": 3, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-XLarge-LR5e5-W200", "hidden_dim": 256, "num_layers": 3, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-XLarge-LR5e5-W500", "hidden_dim": 256, "num_layers": 3, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 500},
    
    # UNI
    {"name": "LSTM-Medium-Uni-LR1e3-W200", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "dropout": 0.3, "bidirectional": False, "window_size": 200},
    {"name": "LSTM-Medium-Uni-LR5e4-W200", "hidden_dim": 128, "num_layers": 2, "lr": 5e-4, "dropout": 0.3, "bidirectional": False, "window_size": 200},
    {"name": "LSTM-Medium-Uni-LR1e4-W200", "hidden_dim": 128, "num_layers": 2, "lr": 1e-4, "dropout": 0.3, "bidirectional": False, "window_size": 200},
]

# 2. Focal Loss Configurations (User Provided)
focal_configs = [
    (0.75, 5.0),
    (0.75, 2.0),
    (0.75, 3.0),
    (0.75, 1.0),
    (0.5, 5.0),
    (0.5, 2.0),
    (0.5, 3.0),
]

def run_single_experiment(args):
    """
    args is a tuple: (lstm_config_dict, alpha, gamma)
    """
    config, alpha, gamma = args
    
    global worker_device
    device_id = worker_device
    device_str = f"cuda:{device_id}"
    pid = os.getpid()
    
    start_time = time.time()
    
    # Construct task name for logging
    task_id = f"{config['name']} | A={alpha} G={gamma}"
    print(f"\n▶ [PID {pid} | {device_str}] STARTING: {task_id}")

    try:
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        stream = NPS()
        
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
            # removed output_dim as it is handled by the wrapper
            seed=42,
            epochs=10,
            loss_fn=loss_fn
        )

        # Pipeline: Solo scaling numérico (NPS es todo numérico)
        pipeliner = SelectType(numbers.Number) | preprocessing.StandardScaler() | clf

        metrics_result = evaluate.progressive_val_score(
            dataset=stream,
            model=pipeliner,
            metric=Metrics([ExactMatch(), MacroAverage(F1()), MicroAverage(F1())]),
            show_memory=False,
            print_every=50000 
        )

        micro_f1 = metrics_result[2].get() * 100
        duration = time.time() - start_time
        duration_str = f"{duration:.2f}s"
        
        print(f"✅ [PID {pid} | {device_str}] FINISHED: {task_id} | F1={micro_f1:.2f}% | {duration_str}")
        
        # Result Dictionary - Must match structure for easy joining
        # Keeping 'name' exactly as requested
        result = config.copy()
        result.update({
            "ExactMatch": metrics_result[0].get() * 100,
            "Macro F1": metrics_result[1].get() * 100,
            "Micro F1": micro_f1,
            "Duration": duration_str,
            "Device": device_str,
            "Status": "Success",
            "Focal_Alpha": alpha,
            "Focal_Gamma": gamma,
            "Loss_Type": "Focal"
        })
        return result

    except Exception as e:
        print(f"❌ [PID {pid} | {device_str}] ERROR in {task_id}: {e}")
        import traceback
        traceback.print_exc()
        result = config.copy()
        result.update({
            "Status": f"Error: {e}",
            "Focal_Alpha": alpha,
            "Focal_Gamma": gamma,
            "Loss_Type": "Focal",
            "Duration": "N/A"
        })
        return result

if __name__ == "__main__":
    # Generate Cartesian Product of Tasks
    # (LSTM Config + Focal Params)
    tasks = []
    for l_conf in lstm_configs:
        for (a, g) in focal_configs:
            tasks.append((l_conf, a, g))
            
    print(f"\n{'='*80}")
    print(f"FULL PARALLEL LSTM x FOCAL SEARCH (NPS)")
    print(f"LSTM Models: {len(lstm_configs)}")
    print(f"Focal Configs: {len(focal_configs)}")
    print(f"Total Experiments: {len(tasks)}")
    print(f"Worker Pool: 2 Processes (1x cuda:0, 1x cuda:1)")
    
    # Estimate time
    est_per_exp = 200 # seconds
    total_est_seconds = (len(tasks) * est_per_exp) / 2 # divided by workers
    total_est_hours = total_est_seconds / 3600
    print(f"Estimated Time: ~{total_est_hours:.2f} hours (assuming {est_per_exp}s per exp)")
    print(f"{'='*80}\n")
    
    m = mp.Manager()
    device_queue = m.Queue()
    # 1 process for GPU 0
    device_queue.put(0)
    # 1 process for GPU 1
    device_queue.put(1)
    
    with mp.Pool(processes=2, initializer=init_worker, initargs=(device_queue,)) as pool:
        results = pool.map(run_single_experiment, tasks)
    
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save to the requested folder
    SAVE_DIR = "results/lstmnps"
    os.makedirs(SAVE_DIR, exist_ok=True)
    filename = f"{SAVE_DIR}/lstm_nps_focal_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    print("\n" + "="*80)
    print("TOP 20 LSTM + FOCAL CONFIGURATIONS")
    print("="*80)
    df_sorted = df[df['Status'] == 'Success'].sort_values("Micro F1", ascending=False)
    cols = ['name', 'Focal_Alpha', 'Focal_Gamma', 'Micro F1', 'Duration']
    print(df_sorted[cols].head(20).to_string(index=False))
    print(f"\nSaved to: {filename}")
