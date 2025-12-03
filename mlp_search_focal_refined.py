"""
Búsqueda REFINADA de hiperparámetros para MLP con FOCAL LOSS.

Objetivo: Superar el 50% de Micro F1.
Basado en hallazgos previos:
- Mejor Window Size: 1000 (probaremos 1000 y 1500)
- Mejor LR: 5e-5 (probaremos 5e-5 y 1e-4)
- Dropout fijo: 0.1
- Focal Loss es superior a BCE.

Variables a explorar:
- Focal Loss Alpha: [0.25, 0.5, 0.75] (Balance de clases)
- Focal Loss Gamma: [1.0, 2.0, 3.0] (Enfoque en difíciles)
- Threshold: [0.20, 0.25, 0.30] (Ajuste de decisión)

EJECUCIÓN EN PARALELO: Usa 2 GPUs (cuda:0 y cuda:1).
"""

import torch
import pandas as pd
import numbers
from datetime import datetime
from testclassifier.model import MLP_MultiLabel, FocalLoss
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier
from datasets.multioutput import Ai4i
import evaluate
from river.compose import SelectType
from river.metrics import F1
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch
from river import preprocessing
from custommetrics.multioutput import *
from concurrent.futures import ProcessPoolExecutor
import os

# Target names
target_names = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Generación dinámica de configuraciones
configs = []

# Fixed Architecture based on best results
hidden_dims = [512, 256, 128]
dropout = 0.1

# Search Space
window_sizes = [1000, 1500]
learning_rates = [5e-5, 1e-4]
alphas = [0.25, 0.5, 0.75]
gammas = [1.0, 2.0, 3.0]
thresholds = [0.20, 0.25, 0.30]

for ws in window_sizes:
    for lr in learning_rates:
        for alpha in alphas:
            for gamma in gammas:
                for th in thresholds:
                    configs.append({
                        "name": f"MLP-W{ws}-LR{lr}-A{alpha}-G{gamma}-TH{th}",
                        "hidden_dims": hidden_dims,
                        "window_size": ws,
                        "lr": lr,
                        "dropout": dropout,
                        "alpha": alpha,
                        "gamma": gamma,
                        "threshold": th,
                        "epochs": 10
                    })

def run_single_experiment(args):
    """
    Run a single experiment on the assigned GPU.
    """
    i, config, device_id = args
    device = f"cuda:{device_id}"
    
    print(f"\n{'='*80}")
    print(f"[GPU {device_id}] [{i+1}/{len(configs)}] Probando: {config['name']}")
    print(f"  - Window: {config['window_size']}, LR: {config['lr']}")
    print(f"  - Alpha: {config['alpha']}, Gamma: {config['gamma']}")
    print(f"  - Threshold: {config['threshold']}")
    print(f"{'='*80}\n")

    try:
        # Reiniciar stream
        stream = Ai4i()

        # Instanciar Focal Loss
        loss_fn = FocalLoss(alpha=config['alpha'], gamma=config['gamma'], reduction='mean')

        # Crear modelo MLP con rolling
        # Pasamos thresholds específicos para cada label
        thresholds_dict = {t: config['threshold'] for t in target_names}

        clf_mlp = RollingMultiLabelClassifier(
            module=MLP_MultiLabel,
            label_names=target_names,
            optimizer_fn="adam",
            lr=config['lr'],
            device=device,
            window_size=config['window_size'],
            append_predict=False,
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout'],
            output_dim=len(target_names),
            seed=42,
            epochs=config['epochs'],
            loss_fn=loss_fn,
            thresholds=thresholds_dict
        )

        # Pipeline
        pr_mlp = SelectType(numbers.Number) | preprocessing.StandardScaler()
        pr_mlp += SelectType(str) | preprocessing.OneHotEncoder()
        pipeliner_mlp = pr_mlp | clf_mlp

        # Evaluar
        metrics_result = evaluate.progressive_val_score(
            dataset=stream,
            model=pipeliner_mlp,
            metric=Metrics([ExactMatch(), MacroAverage(F1()), MicroAverage(F1())]),
            show_memory=False,
            print_every=5000 # Menos verboso
        )

        # Extraer métricas
        exact_match = metrics_result[0].get()
        macro_f1 = metrics_result[1].get()
        micro_f1 = metrics_result[2].get()

        result = {
            "Model": config['name'],
            "Window Size": config['window_size'],
            "LR": config['lr'],
            "Alpha": config['alpha'],
            "Gamma": config['gamma'],
            "Threshold": config['threshold'],
            "ExactMatch": exact_match * 100,
            "Macro F1": macro_f1 * 100,
            "Micro F1": micro_f1 * 100,
            "Status": "Success",
            "GPU": device_id
        }

        print(f"\n[GPU {device_id}] ✅ {config['name']} - Micro F1: {micro_f1*100:.2f}%\n")
        return result

    except Exception as e:
        print(f"\n[GPU {device_id}] ❌ ERROR en {config['name']}: {str(e)}\n")
        return {
            "Model": config['name'],
            "Window Size": config['window_size'],
            "LR": config['lr'],
            "Alpha": config['alpha'],
            "Gamma": config['gamma'],
            "Threshold": config['threshold'],
            "ExactMatch": 0.0,
            "Macro F1": 0.0,
            "Micro F1": 0.0,
            "Status": f"Error: {str(e)}",
            "GPU": device_id
        }


if __name__ == '__main__':
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    print("="*80)
    print("BÚSQUEDA REFINADA - MLP Focal Loss")
    print("="*80)
    print(f"Total de configuraciones: {len(configs)}")
    print(f"Modo paralelo: 2 GPUs")
    
    experiment_args = []
    for idx, config in enumerate(configs):
        gpu_id = idx % 2
        experiment_args.append((idx, config, gpu_id))
    
    results = []
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_single_experiment, args) for args in experiment_args]
        
        for future in futures:
            result = future.result()
            results.append(result)
            
            if len(results) % 5 == 0:
                df_temp = pd.DataFrame(results)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                df_temp.to_csv(f'mlp_focal_refined_partial.csv', index=False)
                print(f"\n💾 Guardado parcial ({len(results)}/{len(configs)})")
    
    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'mlp_focal_refined_results_{timestamp}.csv'
    df_results.to_csv(csv_filename, index=False)
    
    print("\n" + "="*80)
    print("TOP 10 MODELOS POR MICRO F1")
    print("="*80)
    if len(df_results) > 0:
        print(df_results.sort_values('Micro F1', ascending=False).head(10).to_string(index=False))
