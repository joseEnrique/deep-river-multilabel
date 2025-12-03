"""
Búsqueda de hiperparámetros para MLP con FOCAL LOSS.

Enfoque:
- Arquitectura Fija: [512, 256, 128]
- Window Sizes: [500, 1000, 1500]
- Learning Rates: [1e-4, 5e-5]
- Dropout: [0.1]
- Loss: Focal Loss
- Thresholds: [0.3, 0.4, 0.5]

EJECUCIÓN EN PARALELO: Usa 2 GPUs (cuda:0 y cuda:1) simultáneamente.
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

# Fixed Architecture
hidden_dims = [512, 256, 128]

# Search Space
window_sizes = [500, 1000, 1500]
learning_rates = [1e-4, 5e-5]
dropouts = [0.1]
losses = ["focal"]
thresholds = [0.3, 0.4, 0.5]

for ws in window_sizes:
    for lr in learning_rates:
        for do in dropouts:
            for loss_name in losses:
                for th in thresholds:
                    configs.append({
                        "name": f"MLP-XL-W{ws}-LR{lr}-DO{do}-{loss_name}-TH{th}",
                        "hidden_dims": hidden_dims,
                        "window_size": ws,
                        "lr": lr,
                        "dropout": do,
                        "loss": loss_name,
                        "threshold": th,
                        "epochs": 20
                    })


def run_single_experiment(args):
    """
    Run a single experiment on the assigned GPU.
    
    Args:
        args: tuple of (config_index, config_dict, device_id)
    
    Returns:
        dict with results
    """
    i, config, device_id = args
    device = f"cuda:{device_id}"
    
    print(f"\n{'='*80}")
    print(f"[GPU {device_id}] [{i+1}/{len(configs)}] Probando: {config['name']}")
    print(f"  - Device: {device}")
    print(f"  - Window Size: {config['window_size']}")
    print(f"  - LR: {config['lr']}")
    print(f"  - Dropout: {config['dropout']}")
    print(f"  - Loss: {config['loss']}")
    print(f"  - Threshold: {config['threshold']}")
    print(f"{'='*80}\n")

    try:
        # Reiniciar stream
        stream = Ai4i()

        # Configurar Loss
        loss_fn = None
        if config['loss'] == 'focal':
            loss_fn = FocalLoss(gamma=2.0)  # Default gamma

        # Configurar Thresholds
        thresholds_dict = {t: config['threshold'] for t in target_names}

        # Crear modelo MLP con rolling
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
            print_every=2000
        )

        # Extraer métricas
        # metrics_result es un objeto Metrics, que actúa como una lista de métricas
        exact_match = metrics_result[0].get()
        macro_f1 = metrics_result[1].get()
        micro_f1 = metrics_result[2].get()

        result = {
            "Model": config['name'],
            "Architecture": str(config['hidden_dims']),
            "Window Size": config['window_size'],
            "LR": config['lr'],
            "Dropout": config['dropout'],
            "Loss": config['loss'],
            "Threshold": config['threshold'],
            "ExactMatch": exact_match * 100,
            "Macro F1": macro_f1 * 100,
            "Micro F1": micro_f1 * 100,
            "Status": "Success",
            "GPU": device_id
        }

        print(f"\n[GPU {device_id}] ✅ {config['name']} - ExactMatch: {exact_match*100:.2f}%, Macro F1: {macro_f1*100:.2f}%, Micro F1: {micro_f1*100:.2f}%\n")
        return result

    except Exception as e:
        print(f"\n[GPU {device_id}] ❌ ERROR en {config['name']}: {str(e)}\n")
        import traceback
        traceback.print_exc()
        
        return {
            "Model": config['name'],
            "Architecture": str(config['hidden_dims']),
            "Window Size": config['window_size'],
            "LR": config['lr'],
            "Dropout": config['dropout'],
            "Loss": config['loss'],
            "Threshold": config['threshold'],
            "ExactMatch": 0.0,
            "Macro F1": 0.0,
            "Micro F1": 0.0,
            "Status": f"Error: {str(e)}",
            "GPU": device_id
        }


if __name__ == '__main__':
    # CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    print("="*80)
    print("BÚSQUEDA FOCAL LOSS - MLP Rolling Classifier")
    print("="*80)
    print(f"Inicio del experimento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total de configuraciones: {len(configs)}")
    print(f"Modo paralelo: 2 GPUs (cuda:0 y cuda:1)")
    print("="*80)
    
    # Prepare arguments for parallel execution
    experiment_args = []
    for idx, config in enumerate(configs):
        gpu_id = idx % 2
        experiment_args.append((idx, config, gpu_id))
    
    results = []
    
    # Run experiments in parallel using 2 workers (one per GPU)
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_single_experiment, args) for args in experiment_args]
        
        for future in futures:
            result = future.result()
            results.append(result)
            
            # Save partial results every time we get a new result
            if len(results) % 1 == 0:  # Save every 1 result
                df_temp = pd.DataFrame(results)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                df_temp.to_csv(f'mlp_focal_partial_{timestamp}.csv', index=False)
                print(f"\n💾 Guardados resultados intermedios ({len(results)}/{len(configs)} completados)\n")
    
    # Crear DataFrame con resultados finales
    df_results = pd.DataFrame(results)
    
    # Guardar resultados completos
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'mlp_focal_results_{timestamp}.csv'
    df_results.to_csv(csv_filename, index=False)
    
    print("\n" + "="*80)
    print("RESULTADOS FINALES - Todos los experimentos")
    print("="*80)
    print(df_results.to_string(index=False))
    print("\n")
    
    # Análisis de resultados
    df_success = df_results[df_results['Status'] == 'Success'].copy()
    
    if len(df_success) > 0:
        df_success_sorted = df_success.sort_values('Micro F1', ascending=False)
    
        print("\n" + "="*80)
        print("TOP 10 MODELOS POR MICRO F1")
        print("="*80)
        print(df_success_sorted.head(10).to_string(index=False))
    
        print("\n" + "="*80)
        print("MEJOR CONFIGURACIÓN")
        print("="*80)
        best = df_success_sorted.iloc[0]
        print(f"🏆 Modelo: {best['Model']}")
        print(f"   - Window Size: {best['Window Size']}")
        print(f"   - LR: {best['LR']}")
        print(f"   - Dropout: {best['Dropout']}")
        print(f"   - Loss: {best['Loss']}")
        print(f"   - Threshold: {best['Threshold']}")
        print(f"   - Micro F1: {best['Micro F1']:.2f}%")
    
        # Análisis por Window Size
        print("\n" + "="*80)
        print("ANÁLISIS POR WINDOW SIZE (promedio Micro F1)")
        print("="*80)
        ws_analysis = df_success.groupby('Window Size')['Micro F1'].agg(['mean', 'std', 'max'])
        print(ws_analysis.to_string())

        # Análisis por LR
        print("\n" + "="*80)
        print("ANÁLISIS POR LR (promedio Micro F1)")
        print("="*80)
        lr_analysis = df_success.groupby('LR')['Micro F1'].agg(['mean', 'std', 'max'])
        print(lr_analysis.to_string())

        # Análisis por Dropout
        print("\n" + "="*80)
        print("ANÁLISIS POR DROPOUT (promedio Micro F1)")
        print("="*80)
        do_analysis = df_success.groupby('Dropout')['Micro F1'].agg(['mean', 'std', 'max'])
        print(do_analysis.to_string())

        # Análisis por Loss
        print("\n" + "="*80)
        print("ANÁLISIS POR LOSS (promedio Micro F1)")
        print("="*80)
        loss_analysis = df_success.groupby('Loss')['Micro F1'].agg(['mean', 'std', 'max'])
        print(loss_analysis.to_string())

        # Análisis por Threshold
        print("\n" + "="*80)
        print("ANÁLISIS POR THRESHOLD (promedio Micro F1)")
        print("="*80)
        th_analysis = df_success.groupby('Threshold')['Micro F1'].agg(['mean', 'std', 'max'])
        print(th_analysis.to_string())
    
    print("\n" + "="*80)
    print(f"✅ Resultados guardados en: {csv_filename}")
    print(f"Fin del experimento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
