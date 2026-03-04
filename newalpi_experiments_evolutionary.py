"""
NewAlpi Experiments: Evolutionary vs Baseline vs Adaptive with LSTM_MultiLabel
"""
import torch
import numpy as np
import random
import numbers
import pandas as pd
import time
import evaluate
import multiprocessing
import os
import datetime

# Enforce determinism
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

from river.compose import SelectType
from river import preprocessing
from river.metrics import F1
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch
from custommetrics.multioutput import MacroAverage, MicroAverage

from datasets.multioutput.newalpi import NewAlpi
from classes.evolutionary_rolling_classifier import EvolutionaryRollingClassifier
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier
from classes.adaptive_rolling_classifier import AdaptiveRollingMultiLabelClassifier
from testclassifier.model import LSTM_MultiLabel  # Explicitly requested model

def run_evaluation_process(exp_name, loss_fn, config, device, queue):
    try:
        print(f"\n{'='*50}")
        print(f"EXPERIMENTO: {exp_name} on {device}")
        strategy_type = config.get('strategy', 'baseline')
        print(f"Strategy: {strategy_type}")
        if strategy_type == 'evolutionary':
            print(f"Mutation Factor: {config.get('mutation_factor', 'N/A')}")
        elif strategy_type == 'adaptive':
            print(f"Scheduler: Patience={config.get('scheduler_patience')}, Factor={config.get('scheduler_factor')}")
        print(f"{'='*50}")

        dataset_params = config['dataset_params']
        # Machine 1 (Long)
        stream = NewAlpi(machine=1, **dataset_params) 
        stream.Y.columns = stream.Y.columns.astype(str)
        label_names = list(stream.Y.columns)
        
        thresholds = {t: config['threshold'] for t in label_names}

        # Classifier instantiation
        if strategy_type == 'evolutionary':
            clf = EvolutionaryRollingClassifier(
                module=LSTM_MultiLabel,
                label_names=label_names,
                optimizer_fn="adam",
                lr=config['lr'],
                device=device,
                gradient_scale=config['gradient_scale'],
                window_size=config['window_size'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                bidirectional=config['bidirectional'],
                output_dim=len(label_names),
                seed=42,
                loss_fn=loss_fn,
                thresholds=thresholds,
                epochs=config['epochs'],
                mutation_factor=config['mutation_factor']
            )
        elif strategy_type == 'adaptive':
             clf = AdaptiveRollingMultiLabelClassifier(
                module=LSTM_MultiLabel,
                label_names=label_names,
                optimizer_fn="adam",
                lr=config['lr'],
                device=device,
                gradient_scale=config['gradient_scale'],
                window_size=config['window_size'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                bidirectional=config['bidirectional'],
                output_dim=len(label_names),
                seed=42,
                loss_fn=loss_fn,
                thresholds=thresholds,
                epochs=config['epochs'],
                # Adaptive Scheduler Params
                scheduler_patience=config.get('scheduler_patience', 1000),
                scheduler_factor=config.get('scheduler_factor', 0.5),
                scheduler_min_lr=config.get('scheduler_min_lr', 1e-6),
                scheduler_smoothing=config.get('scheduler_smoothing', 0.98)
            )
        else: # Baseline
            clf = RollingMultiLabelClassifier(
                module=LSTM_MultiLabel,
                label_names=label_names,
                optimizer_fn="adam",
                lr=config['lr'],
                device=device,
                window_size=config['window_size'],
                append_predict=False,
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                bidirectional=config['bidirectional'],
                output_dim=len(label_names),
                seed=42,
                epochs=config['epochs'],
                loss_fn=loss_fn # Can be None (defaults to BCE) or custom
            )

        # Pipeline: Treat features as numbers for LSTM_MultiLabel
        # Note: NewAlpi outputs sequence indices, StandardScaler will treat them as magnitudes.
        # This is the "naive" integration requested to preserve the Neural Network structure.
        pr = SelectType(numbers.Number) | preprocessing.StandardScaler()
        # pr += SelectType(str) | preprocessing.OneHotEncoder() # NewAlpi has no string features
        pipeline = pr | clf
        
        start_time = time.time()
        metrics_result = evaluate.progressive_val_score(
            dataset=stream,
            model=pipeline,
            metric=Metrics([ExactMatch(), MacroAverage(F1()), MicroAverage(F1())]),
            show_memory=False,
            print_every=2000
        )
        duration = time.time() - start_time
        
        # Determine Final LR
        if hasattr(clf, 'current_base_lr'):
            final_lr = clf.current_base_lr 
        elif hasattr(clf, 'scheduler') and clf.scheduler:
             final_lr = clf.scheduler._get_lr()
        else:
            final_lr = config['lr']
        
        result = {
            "Name": exp_name,
            "Strategy": strategy_type,
            "Mutation": config.get('mutation_factor', 0.0),
            "Start LR": config['lr'],
            "Final LR": final_lr,
            "Exact Match": metrics_result[0].get()*100,
            "Macro F1": metrics_result[1].get()*100,
            "Micro F1": metrics_result[2].get()*100,
            "Time": duration
        }
        queue.put(result)
        print(f"✅ Finished {exp_name}: MicroF1={result['Micro F1']:.2f}%, FinalLR={final_lr:.6f}")

    except Exception as e:
        import traceback
        print(f"Error en experimento {exp_name}: {e}")
        traceback.print_exc()
        queue.put(None)


def main():
    # Common Configuration (LARGE LSTM)
    base_config = {
        'hidden_dim': 256, # Increased from 128
        'num_layers': 2,
        'bidirectional': True,
        'dropout': 0.3,
        'window_size': 200, 
        'gradient_scale': 1.0,
        'threshold': 0.5,
        'epochs': 10,
        'dataset_params': {'input_win': 1720, 'output_win': 480, 'delta': 0, 'sigma': 120, 'min_count': 0},
    }
    
    loss_config = ('BCE', None)
    loss_name, loss_fn = loss_config
    
    experiments_list = []
    
    # 1. Baseline Experiment (Standard LR)
    cfg_baseline = base_config.copy()
    cfg_baseline['lr'] = 1e-3
    cfg_baseline['strategy'] = 'baseline'
    experiments_list.append((f"Baseline_LSTM_Large", loss_fn, cfg_baseline))
    
    # 2. Evolutionary Experiment
    cfg_evo = base_config.copy()
    cfg_evo['lr'] = 1e-3
    cfg_evo['strategy'] = 'evolutionary'
    cfg_evo['mutation_factor'] = 1.5
    experiments_list.append((f"Evo_LSTM_Large_Mut1.5", loss_fn, cfg_evo))
    
    # 3. Adaptive Experiment
    cfg_adap = base_config.copy()
    cfg_adap['lr'] = 1e-2 
    cfg_adap['strategy'] = 'adaptive'
    cfg_adap['scheduler_patience'] = 1000
    cfg_adap['scheduler_factor'] = 0.5
    experiments_list.append((f"Adaptive_LSTM_Large_Start1e-2", loss_fn, cfg_adap))

    # Queue for results
    queue = multiprocessing.Queue()
    
    print(f"Running {len(experiments_list)} experiments (Large Model)...")
    
    # Run Loop
    run_experiments_loop(experiments_list, queue)


def run_experiments_loop(experiments_list, queue):
    results = []
    active_processes = {}
    
    # GPU Management - SEQUENTIAL EXECUTION ENFORCED
    # User requested running one after another on the same GPU.
    free_gpus = [0] # Only use GPU 0
    if not torch.cuda.is_available(): free_gpus = ['cpu']
    
    # Max workers = 1 ensures sequential execution
    max_workers = 1
    
    exp_iter = iter(experiments_list)
    
    while True:
        # Launch new processes if GPUs available
        while free_gpus and len(active_processes) < max_workers:
            try:
                name, loss_fn, config = next(exp_iter)
            except StopIteration:
                break
                
            gpu_id = free_gpus.pop(0)
            device = f"cuda:{gpu_id}" if isinstance(gpu_id, int) else "cpu"
            
            p = multiprocessing.Process(target=run_evaluation_process, args=(name, loss_fn, config, device, queue))
            p.start()
            active_processes[p] = gpu_id
        
        if not active_processes and not free_gpus and len(results) == 0: 
             # Edge case if no experiments run
             break
        if not active_processes and len(free_gpus) == 1: 
             # All done
             break
            
        # Collect results
        while not queue.empty():
            res = queue.get()
            if res:
                results.append(res)
        
        # Cleanup finished processes
        finished = [p for p in active_processes if not p.is_alive()]
        for p in finished:
            gpu_id = active_processes.pop(p)
            free_gpus.append(gpu_id)
            p.join()
            
        time.sleep(1)
    
    print("\n\nFINAL RESULTS: LSTM Baseline vs Evolutionary vs Adaptive")
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by=["Micro F1"], ascending=False)
        print(df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x) if isinstance(x, float) else str(x)))
    else:
        print("No results returned.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()

