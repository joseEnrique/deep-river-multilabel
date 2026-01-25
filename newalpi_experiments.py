"""
LSTM Experiments with WCE and WFL on NewAlpi Dataset

Based on the user's reference table, runs experiments with:
- WCE (Weighted Cross Entropy): gamma=0
- WFL (Weighted Focal Loss): gamma=2.0

Using normal LSTM model on NewAlpi dataset (Machine 4).
"""

import torch
import numbers
import pandas as pd
import time
import evaluate
import multiprocessing
import os
import datetime
from river.compose import SelectType
from river.metrics import F1
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch
from custommetrics.multioutput import MacroAverage, MicroAverage

from datasets.multioutput.newalpi import NewAlpi
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier
from testclassifier.model import AlpiOneHotLSTM, WeightedFocalLoss


def run_evaluation_process(exp_name, loss_fn, config, device, queue):
    """Ejecuta la evaluación en un proceso separado."""
    try:
        print(f"\n{'='*50}")
        print(f"EXPERIMENTO: {exp_name} on {device}")
        print(f"Loss: {loss_fn}")
        print(f"Dataset Params: {config.get('dataset_params', {})}")
        print(f"{'='*50}")
        
        dataset_params = config['dataset_params']
        stream = NewAlpi(machine=4, **dataset_params)
        stream.Y.columns = stream.Y.columns.astype(str)
        label_names = list(stream.Y.columns)
        
        # Umbrales fijos
        thresholds = {t: config['threshold'] for t in label_names}
        
        clf = RollingMultiLabelClassifier(
            module=AlpiOneHotLSTM,
            label_names=label_names,
            optimizer_fn="adam",
            lr=config['lr'],
            device=device,
            window_size=config['window_size'],
            append_predict=False,
            hidden_dim=config['hidden_dim'],
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            bidirectional=True,
            seed=42,
            epochs=config['epochs'],
            loss_fn=loss_fn,
            thresholds=thresholds,
        )
        
        pipeline = SelectType(numbers.Number) | clf
        
        start_time = time.time()
        metrics_result = evaluate.progressive_val_score(
            dataset=stream,
            model=pipeline,
            metric=Metrics([ExactMatch(), MacroAverage(F1()), MicroAverage(F1())]),
            show_memory=False,
            print_every=500
        )
        duration = time.time() - start_time
        
        result = {
            "Name": exp_name,
            "Loss": config['loss_name'],
            "Input_win": dataset_params['input_win'],
            "Output_win": dataset_params['output_win'],
            "Delta": dataset_params['delta'],
            "Sigma": dataset_params['sigma'],
            "Exact Match": metrics_result[0].get()*100,
            "Macro F1": metrics_result[1].get()*100,
            "Micro F1": metrics_result[2].get()*100,
            "Time": duration
        }
        queue.put(result)
        print(f"✅ Finished {exp_name}: MicroF1={result['Micro F1']:.2f}%, MacroF1={result['Macro F1']:.2f}%")

    except Exception as e:
        import traceback
        print(f"Error en experimento {exp_name}: {e}")
        traceback.print_exc()
        queue.put(None)


def main():
    # Base configuration (model hyperparameters)
    base_config = {
        'hidden_dim': 256,
        'embedding_dim': 128,
        'num_layers': 2,
        'lr': 1e-3,
        'dropout': 0.2,
        'window_size': 100,
        'threshold': 0.25,
        'epochs': 10,
    }
    
    # Dataset parameter combinations from the user's table
    # Left side of table (output_win = 60, 90)
    dataset_configs_left = [
        {'input_win': 1720, 'output_win': 60, 'delta': 0, 'sigma': 60},
        {'input_win': 1720, 'output_win': 60, 'delta': 0, 'sigma': 30},
        {'input_win': 1720, 'output_win': 60, 'delta': 0, 'sigma': 90},
        {'input_win': 1720, 'output_win': 90, 'delta': 0, 'sigma': 90},
        {'input_win': 1720, 'output_win': 90, 'delta': 0, 'sigma': 70},
        {'input_win': 1720, 'output_win': 90, 'delta': 0, 'sigma': 60},
    ]
    
    # Right side of table (output_win = 480)
    dataset_configs_right = [
        {'input_win': 1720, 'output_win': 480, 'delta': 0, 'sigma': 120},
        {'input_win': 1720, 'output_win': 480, 'delta': 0, 'sigma': 60},
        {'input_win': 1720, 'output_win': 480, 'delta': 0, 'sigma': 480},
        {'input_win': 1720, 'output_win': 480, 'delta': 0, 'sigma': 360},
        {'input_win': 1720, 'output_win': 480, 'delta': 0, 'sigma': 240},
    ]
    
    # Combine all dataset configurations
    all_dataset_configs = dataset_configs_left + dataset_configs_right
    
    # Loss functions
    beta_opt = 0.8  # From previous experiments
    loss_configs = [
        ('WCE', WeightedFocalLoss(beta=beta_opt, gamma=0)),   # WCE = WFL with gamma=0
        ('WFL', WeightedFocalLoss(beta=beta_opt, gamma=2.0)), # WFL with gamma=2.0
    ]
    
    # Build experiments list
    experiments_list = []
    for loss_name, loss_fn in loss_configs:
        for ds_params in all_dataset_configs:
            exp_name = f"M4_{loss_name}_iw{ds_params['input_win']}_ow{ds_params['output_win']}_s{ds_params['sigma']}"
            config = base_config.copy()
            config['dataset_params'] = ds_params
            config['loss_name'] = loss_name
            experiments_list.append((exp_name, loss_fn, config))
    
    print(f"\n🚀 Total experiments: {len(experiments_list)}")
    
    queue = multiprocessing.Queue()
    results = []
    
    # Process management
    active_processes = {}
    if torch.cuda.device_count() >= 2:
        free_gpus = [0]
    elif torch.cuda.device_count() == 1:
        free_gpus = [0]
    else:
        free_gpus = ['cpu']
    
    exp_iter = iter(experiments_list)
    
    print("\nStarting NewAlpi LSTM experiments (WCE vs WFL)...\n")
    
    while True:
        # Launch new processes if GPU available
        while free_gpus:
            try:
                name, loss_fn, config = next(exp_iter)
                gpu_id = free_gpus.pop(0)
                device = f"cuda:{gpu_id}" if isinstance(gpu_id, int) else gpu_id
                
                print(f"🚀 Launching {name} on {device}")
                p = multiprocessing.Process(
                    target=run_evaluation_process,
                    args=(name, loss_fn, config, device, queue)
                )
                p.start()
                active_processes[p] = gpu_id
                
            except StopIteration:
                break
        
        if not active_processes:
            break
        
        # Check results
        while not queue.empty():
            res = queue.get()
            if res:
                results.append(res)
                print(f"✅ Received result for {res['Name']}")
        
        # Cleanup finished processes
        finished = [p for p in active_processes if not p.is_alive()]
        for p in finished:
            gpu_id = active_processes.pop(p)
            free_gpus.append(gpu_id)
            p.join()
        
        time.sleep(1)
    
    # Save results
    if results:
        print("\n\n" + "="*80)
        print("FINAL RESULTS: NewAlpi LSTM Experiments (WCE vs WFL)")
        print("="*80)
        
        df = pd.DataFrame(results)
        df = df.sort_values(by=["Loss", "Sigma", "Output_win"], ascending=[True, True, True])
        
        os.makedirs('results/newalpi_experiments', exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/newalpi_experiments/onehot_lstm_wce_wfl_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        print(f"Resultados guardados en: {filename}")
        print(df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
        print("="*80)
    else:
        print("No results collected!")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
