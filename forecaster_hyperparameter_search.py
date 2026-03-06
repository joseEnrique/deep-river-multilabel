"""
Búsqueda exhaustiva de hiperparámetros para RollingMultiLabelClassifierSequences.

Este script prueba diferentes configuraciones de:
- Learning rates
- Hidden dimensions
- Number of layers
- Window sizes
- Epochs
- Prediction thresholds

Los resultados se guardan en CSV para análisis posterior.

EJECUCIÓN EN PARALELO: Usa 2 GPUs (cuda:0 y cuda:1) simultáneamente.
"""

import torch
import pandas as pd
from datetime import datetime
from testclassifier.model import LSTM_MultiLabel, FocalLoss
from classes.rolling_multilabel_classifier_sequences import RollingMultiLabelClassifierSequences
from datasets.multioutput import RollingAi4i
from river import compose, preprocessing
from river.metrics import F1
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch
from custommetrics.multioutput import *
from evaluate.progressive_validation import progressive_val_score_sequence
from concurrent.futures import ProcessPoolExecutor
import os

# Target names
target_names = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
configs = [
# Configuraciones a pr    # ============================================================================
    # 1. FOCAL LOSS (Baseline with higher thresholds)
    # ============================================================================
    {"name": "Focal-T35", "loss": "FocalLoss", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "past_size": 100, "window_size": 100, "epochs": 5, "threshold": 0.35, "alpha": 0.25, "gamma": 2.0},
    {"name": "Focal-T40", "loss": "FocalLoss", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "past_size": 100, "window_size": 100, "epochs": 5, "threshold": 0.40, "alpha": 0.25, "gamma": 2.0},
    {"name": "Focal-T45", "loss": "FocalLoss", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "past_size": 100, "window_size": 100, "epochs": 5, "threshold": 0.45, "alpha": 0.25, "gamma": 2.0},
    {"name": "Focal-T50", "loss": "FocalLoss", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "past_size": 100, "window_size": 100, "epochs": 5, "threshold": 0.50, "alpha": 0.25, "gamma": 2.0},

    # ============================================================================
    # 2. WEIGHTED BCE (Weights 10, 50, 100)
    # ============================================================================
    {"name": "Weighted10-T40", "loss": "WeightedBCE", "pos_weight": 10.0, "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "past_size": 100, "window_size": 100, "epochs": 5, "threshold": 0.40},
    {"name": "Weighted10-T50", "loss": "WeightedBCE", "pos_weight": 10.0, "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "past_size": 100, "window_size": 100, "epochs": 5, "threshold": 0.50},
    
    {"name": "Weighted50-T40", "loss": "WeightedBCE", "pos_weight": 50.0, "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "past_size": 100, "window_size": 100, "epochs": 5, "threshold": 0.40},
    {"name": "Weighted50-T50", "loss": "WeightedBCE", "pos_weight": 50.0, "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "past_size": 100, "window_size": 100, "epochs": 5, "threshold": 0.50},

    {"name": "Weighted100-T40", "loss": "WeightedBCE", "pos_weight": 100.0, "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "past_size": 100, "window_size": 100, "epochs": 5, "threshold": 0.40},
    {"name": "Weighted100-T50", "loss": "WeightedBCE", "pos_weight": 100.0, "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "past_size": 100, "window_size": 100, "epochs": 5, "threshold": 0.50},

    # ============================================================================
    # 3. STANDARD BCE WITH LOGITS
    # ============================================================================
    {"name": "BCE-T35", "loss": "BCEWithLogits", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "past_size": 100, "window_size": 100, "epochs": 5, "threshold": 0.35},
    {"name": "BCE-T40", "loss": "BCEWithLogits", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "past_size": 100, "window_size": 100, "epochs": 5, "threshold": 0.40},
    {"name": "BCE-T45", "loss": "BCEWithLogits", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "past_size": 100, "window_size": 100, "epochs": 5, "threshold": 0.45},
    {"name": "BCE-T50", "loss": "BCEWithLogits", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "past_size": 100, "window_size": 100, "epochs": 5, "threshold": 0.50},
    
    # ============================================================================
    # 4. MORE EPOCHS (Focal Loss)
    # ============================================================================
    {"name": "Focal-E10-T40", "loss": "FocalLoss", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "past_size": 100, "window_size": 100, "epochs": 10, "threshold": 0.40, "alpha": 0.25, "gamma": 2.0},
    {"name": "Focal-E10-T50", "loss": "FocalLoss", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "past_size": 100, "window_size": 100, "epochs": 10, "threshold": 0.50, "alpha": 0.25, "gamma": 2.0},
]


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
    print(f"  - Loss: {config['loss']}")
    print(f"  - hidden_dim: {config['hidden_dim']}")
    print(f"  - num_layers: {config['num_layers']}")
    print(f"  - lr: {config['lr']}")
    print(f"  - past_size: {config['past_size']}")
    print(f"  - window_size: {config['window_size']}")
    print(f"  - epochs: {config['epochs']}")
    print(f"  - threshold: {config['threshold']}")
    if config['loss'] == 'FocalLoss':
        print(f"  - FocalLoss alpha: {config['alpha']}")
        print(f"  - FocalLoss gamma: {config['gamma']}")
    elif config['loss'] == 'WeightedBCE':
        print(f"  - Pos Weight: {config['pos_weight']}")
    print(f"{'='*80}\n")

    try:
        # Setup numeric and target columns for pipeline
        numeric_cols = ['Air temperature [K]', 'Process temperature [K]',
                        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        target_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        
        # Instantiate Loss Function
        if config['loss'] == 'FocalLoss':
            loss_fn = FocalLoss(alpha=config['alpha'], gamma=config['gamma'], reduction='mean')
        elif config['loss'] == 'WeightedBCE':
            # Create tensor of weights for each class
            pos_weight = torch.tensor([config['pos_weight']] * len(target_names), device=device)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif config['loss'] == 'BCEWithLogits':
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss function: {config['loss']}")

        # Create forecaster with FocalLoss params from config
        forecaster = RollingMultiLabelClassifierSequences(
            window_size=config['window_size'],
            label_names=target_names,
            module=LSTM_MultiLabel,
            loss_fn=loss_fn,
            optimizer_fn='adam',
            lr=config['lr'],
            device=device,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=len(target_names),
            epochs=config['epochs'],
            seed=42,
            threshold=config['threshold']
        )

        # Pipeline setup
        p_numeric = compose.Select(*numeric_cols) | preprocessing.StandardScaler()
        p_categorical = compose.Select('Type') | preprocessing.OneHotEncoder()
        p_targets = compose.Select(*target_cols)
        pipeline = (p_numeric + p_categorical + p_targets) | forecaster

        # Create stream
        # past_size: historical data provided by RollingAi4i
        # window_size: timesteps used by the LSTM model
        stream = RollingAi4i(past_size=config['past_size'], n_instances=10000, include_targets=True)

        # Evaluate
        metrics_result = progressive_val_score_sequence(
            dataset=stream,
            model=pipeline,
            metric=Metrics([
                ExactMatch(),
                MacroAverage(F1()),
                MicroAverage(F1())
            ]),
            print_every=2000
        )

        # Extraer métricas
        exact_match = metrics_result[0].get()
        macro_f1 = metrics_result[1].get()
        micro_f1 = metrics_result[2].get()

        result = {
            "Model": config['name'],
            "Loss": config['loss'],
            "Hidden Dim": config['hidden_dim'],
            "Num Layers": config['num_layers'],
            "LR": config['lr'],
            "Past Size": config['past_size'],
            "Window Size": config['window_size'],
            "Epochs": config['epochs'],
            "Threshold": config['threshold'],
            "Alpha": config.get('alpha', 'N/A'),
            "Gamma": config.get('gamma', 'N/A'),
            "Pos Weight": config.get('pos_weight', 'N/A'),
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
            "Loss": config['loss'],
            "Hidden Dim": config['hidden_dim'],
            "Num Layers": config['num_layers'],
            "LR": config['lr'],
            "Past Size": config['past_size'],
            "Window Size": config['window_size'],
            "Epochs": config['epochs'],
            "Threshold": config['threshold'],
            "Alpha": config.get('alpha', 'N/A'),
            "Gamma": config.get('gamma', 'N/A'),
            "Pos Weight": config.get('pos_weight', 'N/A'),
            "ExactMatch": 0.0,
            "Macro F1": 0.0,
            "Micro F1": 0.0,
            "Status": f"Error: {str(e)}",
            "GPU": device_id
        }


if __name__ == '__main__':
    # CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
    # 'fork' (Linux default) doesn't work with CUDA
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    print("="*80)
    print("BÚSQUEDA DE HIPERPARÁMETROS - RollingMultiLabelClassifierSequences")
    print("="*80)
    print(f"Inicio del experimento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total de configuraciones: {len(configs)}")
    print(f"Modo paralelo: 2 GPUs (cuda:0 y cuda:1)")
    print("="*80)
    
    # Prepare arguments for parallel execution
    # Alternate configs between GPU 0 and GPU 1
    experiment_args = []
    for idx, config in enumerate(configs):
        # Assign GPU: even indices to cuda:0, odd indices to cuda:1
        gpu_id = idx % 2
        experiment_args.append((idx, config, gpu_id))
    
    results = []
    
    # Run experiments in parallel using 2 workers (one per GPU)
    with ProcessPoolExecutor(max_workers=2) as executor:
        # Submit all experiments
        futures = [executor.submit(run_single_experiment, args) for args in experiment_args]
        
        # Collect results as they complete
        for future in futures:
            result = future.result()
            results.append(result)
            
            # Save partial results every time we get a new result
            if len(results) % 2 == 0:  # Save every 2 results
                df_temp = pd.DataFrame(results)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                df_temp.to_csv(f'forecaster_search_partial_{timestamp}.csv', index=False)
                print(f"\n💾 Guardados resultados intermedios ({len(results)}/{len(configs)} completados)\n")
    
    # Crear DataFrame con resultados finales
    df_results = pd.DataFrame(results)
    
    # Guardar resultados completos
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'forecaster_search_results_{timestamp}.csv'
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
        print(f"   - Loss: {best['Loss']}")
        print(f"   - Hidden Dim: {best['Hidden Dim']}")
        print(f"   - Num Layers: {best['Num Layers']}")
        print(f"   - LR: {best['LR']}")
        print(f"   - Past Size: {best['Past Size']}")
        print(f"   - Window Size: {best['Window Size']}")
        print(f"   - Epochs: {best['Epochs']}")
        print(f"   - Threshold: {best['Threshold']}")
        print(f"   - FocalLoss Alpha: {best['Alpha']}")
        print(f"   - FocalLoss Gamma: {best['Gamma']}")
        print(f"   - Pos Weight: {best['Pos Weight']}")
        print(f"   - GPU: {best['GPU']}")
        print(f"   - ExactMatch: {best['ExactMatch']:.2f}%")
        print(f"   - Macro F1: {best['Macro F1']:.2f}%")
        print(f"   - Micro F1: {best['Micro F1']:.2f}%")
    
        # Análisis por window size
        print("\n" + "="*80)
        print("ANÁLISIS POR WINDOW SIZE (promedio Micro F1)")
        print("="*80)
        window_analysis = df_success.groupby('Window Size')['Micro F1'].agg(['mean', 'std', 'max'])
        print(window_analysis.to_string())
    
        # Análisis por threshold
        print("\n" + "="*80)
        print("ANÁLISIS POR THRESHOLD (promedio Micro F1)")
        print("="*80)
        threshold_analysis = df_success.groupby('Threshold')['Micro F1'].agg(['mean', 'std', 'max'])
        print(threshold_analysis.to_string())
        
        # Análisis por FocalLoss Alpha
        print("\n" + "="*80)
        print("ANÁLISIS POR FOCAL LOSS ALPHA (promedio Micro F1)")
        print("="*80)
        alpha_analysis = df_success.groupby('Alpha')['Micro F1'].agg(['mean', 'std', 'max'])
        print(alpha_analysis.to_string())
        
        # Análisis por FocalLoss Gamma
        print("\n" + "="*80)
        print("ANÁLISIS POR FOCAL LOSS GAMMA (promedio Micro F1)")
        print("="*80)
        gamma_analysis = df_success.groupby('Gamma')['Micro F1'].agg(['mean', 'std', 'max'])
        print(gamma_analysis.to_string())
    
        # Análisis por epochs
        print("\n" + "="*80)
        print("ANÁLISIS POR EPOCHS (promedio Micro F1)")
        print("="*80)
        epochs_analysis = df_success.groupby('Epochs')['Micro F1'].agg(['mean', 'std', 'max'])
        print(epochs_analysis.to_string())
        
        # Análisis por Loss
        print("\n" + "="*80)
        print("ANÁLISIS POR LOSS FUNCTION (promedio Micro F1)")
        print("="*80)
        loss_analysis = df_success.groupby('Loss')['Micro F1'].agg(['count', 'mean', 'std', 'max'])
        print(loss_analysis.to_string())
        
        # Análisis por GPU
        print("\n" + "="*80)
        print("ANÁLISIS POR GPU (promedio Micro F1)")
        print("="*80)
        gpu_analysis = df_success.groupby('GPU')['Micro F1'].agg(['count', 'mean', 'std', 'max'])
        print(gpu_analysis.to_string())
    
    print("\n" + "="*80)
    print(f"✅ Resultados guardados en: {csv_filename}")
    print(f"Fin del experimento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
