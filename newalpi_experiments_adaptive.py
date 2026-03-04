"""
LSTM Experiments with Adaptive LR on NewAlpi Dataset
Based on newalpi_experiments_embedding.py, but using AdaptiveRollingMultiLabelClassifier to test online LR adaptation.
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
import torch.nn as nn
import torch.nn.functional as F

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

class AdaptiveWeightedFocalLoss(nn.Module):
    """
    Adaptive Weighted Focal Loss.
    
    Adaptively balances the positive class weight (beta) based on the 
    stream of data observed so far using exponential moving averages.
    
    beta_t = n_neg / (n_pos + n_neg + epsilon)
    
    Ideally, if n_pos is low, beta should be high? 
    Wait, standard Weighted Cross Entropy (WCE) usually uses pos_weight = n_neg / n_pos.
    Or if we use the alpha formulation in Focal Loss:
    FL = -alpha * (1-pt)^gamma * log(pt)
    
    Here we utilize a similar logic to `WeightedFocalLoss` where we have a `beta` argument.
    If `beta` acts like `alpha` (weight for positive class), it typically should be inverse class frequency or similar.
    
    In the user's `WeightedFocalLoss`:
    beta_t = self.beta * targets + (1 - self.beta) * (1 - targets)
    loss = beta_t * focal_term * bce_loss
    
    So if target=1, we multiply by `beta`.
    If we want to upweight minority positives, `beta` should be > 0.5.
    Specifically, balanced weight would be `n_neg / N_total`.
    """
    def __init__(self, gamma=2.0, decay=0.99, epsilon=1e-6, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.decay = decay
        self.epsilon = epsilon
        self.reduction = reduction
        
        # Register buffers for running stats
        # Assuming single scalar for binary classification or broadcastable for multilabel?
        # Multilabel: we need one tracker per class.
        # Initialize with balanced assumption or 0
        self.register_buffer('running_pos', torch.tensor(1.0)) # Initialized to 1 to avoid div0 initially
        self.register_buffer('running_neg', torch.tensor(1.0))
        
        # We will support multi-label, so we might need to adjust shape dynamically or init with correct size if known.
        # But `forward` receives batch logits, so we can infer size there or just update scalars if we Average over classes?
        # Better to track per-class stats. We'll init as scalar 0 and broadcast/expand on first forward.
        self.num_classes = None

    def _update_stats(self, targets):
        # targets: (batch, num_classes)
        if self.num_classes is None:
            self.num_classes = targets.shape[1]
            self.running_pos = torch.ones(self.num_classes, device=targets.device)
            self.running_neg = torch.ones(self.num_classes, device=targets.device)
            
        # Update proper device if needed (in case module moved)
        if self.running_pos.device != targets.device:
            self.running_pos = self.running_pos.to(targets.device)
            self.running_neg = self.running_neg.to(targets.device)
            
        # Calculate current batch stats
        batch_pos = targets.sum(dim=0)
        batch_neg = (1 - targets).sum(dim=0)
        
        # Update EMAs
        self.running_pos = self.decay * self.running_pos + (1 - self.decay) * batch_pos
        self.running_neg = self.decay * self.running_neg + (1 - self.decay) * batch_neg

    def forward(self, logits, targets):
        if self.training:
            with torch.no_grad():
                self._update_stats(targets)
                
        # Calculate dynamic beta (weight for positive class)
        # Weight for P = N_neg / (N_pos + N_neg) -> Higher if N_pos is small
        
        total = self.running_pos + self.running_neg + self.epsilon
        beta = self.running_neg / total
        
        # Clamp beta to avoid extremes? Maybe [0.1, 0.9]? 
        # User requested "Adaptive", let's trust the math.
        
        probs = torch.sigmoid(logits)
        
        # p_t: probability of the true class
        # If target=1, p_t = probs
        # If target=0, p_t = 1 - probs
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # beta_t: weight for the true class
        # If target=1, beta_t = beta
        # If target=0, beta_t = 1 - beta
        # Note: 1 - beta = 1 - (N_neg / Total) = (Total - N_neg) / Total = N_pos / Total
        # This gives high weight to rare classes (if N_pos small -> beta high; if N_neg small -> 1-beta high)
        beta_t = beta * targets + (1 - beta) * (1 - targets)
        
        # Focal Term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # WFL
        loss = beta_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def run_evaluation_process(exp_name, loss_fn, config, device, queue):
    """Ejecuta la evaluación en un proceso separado."""
    try:
        print(f"\n{'='*50}")
        print(f"EXPERIMENTO: {exp_name} on {device}")
        print(f"Loss: {loss_fn}")
        print(f"Dataset Params: {config.get('dataset_params', {})}")
        print(f"{'='*50}")

        dataset_params = config['dataset_params']
        stream = NewAlpi(machine=1, **dataset_params)
        stream.Y.columns = stream.Y.columns.astype(str)
        label_names = list(stream.Y.columns)

        # Umbrales fijos
        thresholds = {t: config['threshold'] for t in label_names}

        # USE NORMAL ROLLING CLASSIFIER
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
            "Final LR": config['lr'], # Constant LR now
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
    
    print("\nStarting NewAlpi ADAPTIVE LOSS Experiments...\n")
    
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
    
    # Dataset parameter combinations
    all_dataset_configs = [
        {'input_win': 1720, 'output_win': 480, 'delta': 0, 'sigma': 120, 'min_count': 0},
    ]
    
    # Loss functions - Compare Baseline (BCE) vs Adaptive Focal
    loss_configs = [
        ('BCE', torch.nn.BCEWithLogitsLoss(reduction='sum')), 
        ('AdaptiveFocal', AdaptiveWeightedFocalLoss(gamma=2.0, reduction='sum')),
    ]
    
    # Experiment Variances
    window_sizes = [1]
    
    # Build experiments list
    experiments_list = []
    for loss_name, loss_fn in loss_configs:
        for ds_params in all_dataset_configs:
            for win_size in window_sizes:
                exp_name = f"ExactLSTM_{loss_name}_ws{win_size}"
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
        print("FINAL RESULTS: NewAlpi Adaptive Experiments")
        print("="*80)
        
        df = pd.DataFrame(results)
        df = df.sort_values(by=["Loss", "Window Size"], ascending=[True, True])
        
        os.makedirs('results/newalpi_experiments', exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/newalpi_experiments/adaptive_embedding_lstm_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        print(f"Resultados guardados en: {filename}")
        print(df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
        print("="*80)
    else:
        print("No results collected!")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
