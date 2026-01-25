"""
LSTM Experiments with WCE and WFL on NewAlpi Dataset - Using AlpiEmbeddingLSTM

Based on the user's reference table, runs experiments with:
- WCE (Weighted Cross Entropy): gamma=0
- WFL (Weighted Focal Loss): gamma=2.0

Using AlpiEmbeddingLSTM model on NewAlpi dataset (Machine 4).
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
from river.metrics import F1
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch
from custommetrics.multioutput import MacroAverage, MicroAverage

from datasets.multioutput.newalpi import NewAlpi
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier
from testclassifier.model import WeightedFocalLoss

class ExactOnlineLSTM(torch.nn.Module):
    """
    Exact copy of OnlineLSTM from stateful_multilabel_classifier.py
    Adapted to accept input_dim/output_dim for Rolling compatibility.
    """
    def __init__(self, input_dim, output_dim, embedding_dim=8, hidden_size=10, num_embeddings=155):
        super().__init__()
        
        # HACK: strict reproduction of manual script behavior
        # Manual script re-initializes model for every new label found in first sample
        # It creates models with output_dim = 1, 2, ..., N-1 before the final N.
        # We must burn the RNG for these intermediate initializations to match weights.
        with torch.no_grad():
            for i in range(1, output_dim):
                torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
                torch.nn.LSTM(embedding_dim, hidden_size, batch_first=True)
                torch.nn.Linear(hidden_size, i)

        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.head = torch.nn.Linear(hidden_size, output_dim)
    
    def forward(self, x):
        # x: (batch, seq_len) - alarm IDs
        # Ensure x is LongTensor
        if x.dtype != torch.long:
            x = x.long()
        
        # In RollingClassifier, x comes as (batch, 1, n_features) if window_size=1
        # We need to squeeze the middle dim if present
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        # EXACT MATCH: Manual script clips values to NUM_EMBEDDINGS - 1
        # alarm_ids = [min(int(x.get(i, 0)), self.NUM_EMBEDDINGS - 1) ...]
        # Here we do it tensor-wise
        x = torch.clamp(x, max=154) # self.embedding.num_embeddings - 1 = 155 - 1 = 154
            
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        _, (h, _) = self.lstm(embedded)
        return self.head(h[-1])


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
            module=ExactOnlineLSTM,
            label_names=label_names,
            optimizer_fn="sgd",
            lr=config['lr'],
            device=device,
            gradient_scale=config['gradient_scale'],
            window_size=config['window_size'],
            # Model params
            hidden_size=config['hidden_dim'],
            embedding_dim=config['embedding_dim'],
            num_embeddings=155,
            seed=42,
            loss_fn=loss_fn,
            thresholds=thresholds,
            epochs=config['epochs'],
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
            "Window Size": config['window_size'],
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


def run_experiments(experiments_list, queue=None):
    """
    Run experiments from the list.
    If queue is provided, put results there.
    Returns list of results.
    """
    if queue is None:
        queue = multiprocessing.Queue()
    
    results = []
    
    # Process management - single GPU
    active_processes = {}
    free_gpus = [0]
    
    exp_iter = iter(experiments_list)
    
    print("\nStarting NewAlpi AlpiEmbeddingLSTM experiments...\n")
    
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
        
    return results


def main():
    # Base configuration (model hyperparameters)
    base_config = {
        'hidden_dim': 10,
        'embedding_dim': 8,
        'num_layers': 1,
        'lr': 1.0,
        'gradient_scale': 10.0,
        'dropout': 0.0,
        'window_size': 1,
        'threshold': 0.5,
        'epochs': 1,
        'bidirectional': False,
    }
    
    # Dataset parameter combinations from the user's table
    # Only one specific configuration as requested
    all_dataset_configs = [
        {'input_win': 1720, 'output_win': 480, 'delta': 0, 'sigma': 120, 'min_count': 0},
    ]
    
    # Loss functions
    beta_opt = 0.8  # From previous experiments
    loss_configs = [
        ('BCE', torch.nn.BCEWithLogitsLoss(reduction='sum')), # Exact match to stateful script
    ]
    
    # Window sizes to experiment with
    window_sizes = [1, 10, 50, 100]
    
    # Build experiments list
    experiments_list = []
    for loss_name, loss_fn in loss_configs:
        for ds_params in all_dataset_configs:
            for win_size in window_sizes:
                exp_name = f"M4_Embed_{loss_name}_iw{ds_params['input_win']}_ow{ds_params['output_win']}_ws{win_size}"
                config = base_config.copy()
                config['dataset_params'] = ds_params
                config['loss_name'] = loss_name
                config['window_size'] = win_size
                experiments_list.append((exp_name, loss_fn, config))
    
    print(f"\n🚀 Total experiments: {len(experiments_list)}")
    
    results = run_experiments(experiments_list)
    
    # Save results
    if results:
        print("\n\n" + "="*80)
        print("FINAL RESULTS: NewAlpi StatefulAlpiEmbeddingLSTM Experiments (BCE with varying Window Sizes)")
        print("="*80)
        
        df = pd.DataFrame(results)
        df = df.sort_values(by=["Loss", "Sigma", "Output_win", "Window Size"], ascending=[True, True, True, True])
        
        os.makedirs('results/newalpi_experiments', exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/newalpi_experiments/stateful_embedding_lstm_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        print(f"Resultados guardados en: {filename}")
        print(df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
        print("="*80)
    else:
        print("No results collected!")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
