"""
Búsqueda FOCALIZADA de hiperparámetros para DirectMultiLabelForecaster.

Enfoque:
- Loss: WeightedBCE (optimización de pesos)
- Arquitectura: Exploración de capacidad (Hidden Dim y Layers)
- Thresholds: Fine-tuning

EJECUCIÓN EN PARALELO: Usa 2 GPUs (cuda:0 y cuda:1) simultáneamente.
"""

import torch
import pandas as pd
from datetime import datetime
from testclassifier.model import LSTM_MultiLabel, FocalLoss, AdaptiveWeightedBCE
from classes.direct_multilabel_forecaster import DirectMultiLabelForecaster
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

# Generación dinámica de configuraciones
configs = []

# 1. Adaptive Weighted BCE (New Proposal)
decays = [0.99, 0.999]
thresholds = [0.40,  0.50]
hidden_dims = [128, 256]
num_layers_list = [2, 3]
for decay in decays:
    for th in thresholds:
        for hd in hidden_dims:
            for nl in num_layers_list:
                configs.append({
                    "name": f"Adapt-D{decay}-T{int(th*100)}-H{hd}-L{nl}",
                    "loss": "AdaptiveWeightedBCE",
                    "decay": decay,
                    "hidden_dim": hd,
                    "num_layers": nl,
                    "lr": 1e-3,
                    "past_size": 100,
                    "window_size": 100,
                    "epochs": 5,
                    "threshold": th
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
    print(f"  - Loss: {config['loss']}")
    if config['loss'] == 'AdaptiveWeightedBCE':
        print(f"  - Decay: {config['decay']}")
    elif config['loss'] == 'WeightedBCE':
        print(f"  - Pos Weight: {config['pos_weight']}")
    print(f"  - hidden_dim: {config['hidden_dim']}")
    print(f"  - num_layers: {config['num_layers']}")
    print(f"  - lr: {config['lr']}")
    print(f"  - past_size: {config['past_size']}")
    print(f"  - window_size: {config['window_size']}")
    print(f"  - epochs: {config['epochs']}")
    print(f"  - threshold: {config['threshold']}")
    print(f"{'='*80}\n")

    try:
        # Setup numeric and target columns for pipeline
        numeric_cols = ['Air temperature [K]', 'Process temperature [K]',
                        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        target_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        
        # Instantiate Loss Function
        if config['loss'] == 'AdaptiveWeightedBCE':
            loss_fn = AdaptiveWeightedBCE(num_classes=len(target_names), decay=config['decay'])
            # Mover buffers al dispositivo correcto
            loss_fn.to(device)
        elif config['loss'] == 'WeightedBCE':
            # Create tensor of weights for each class
            pos_weight = torch.tensor([config['pos_weight']] * len(target_names), device=device)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Create forecaster
        forecaster = DirectMultiLabelForecaster(
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
            "Pos Weight": config.get('pos_weight', 'N/A'),
            "Decay": config.get('decay', 'N/A'),
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
            "Pos Weight": config.get('pos_weight', 'N/A'),
            "Decay": config.get('decay', 'N/A'),
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
    print("BÚSQUEDA FOCALIZADA - DirectMultiLabelForecaster")
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
                df_temp.to_csv(f'forecaster_focused_partial_{timestamp}.csv', index=False)
                print(f"\n💾 Guardados resultados intermedios ({len(results)}/{len(configs)} completados)\n")
    
    # Crear DataFrame con resultados finales
    df_results = pd.DataFrame(results)
    
    # Guardar resultados completos
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'forecaster_focused_results_{timestamp}.csv'
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
        print(f"   - Pos Weight: {best['Pos Weight']}")
        print(f"   - Decay: {best['Decay']}")
        print(f"   - Hidden Dim: {best['Hidden Dim']}")
        print(f"   - Num Layers: {best['Num Layers']}")
        print(f"   - Threshold: {best['Threshold']}")
        print(f"   - Micro F1: {best['Micro F1']:.2f}%")
    
        # Análisis por Loss
        print("\n" + "="*80)
        print("ANÁLISIS POR LOSS (promedio Micro F1)")
        print("="*80)
        loss_analysis = df_success.groupby('Loss')['Micro F1'].agg(['mean', 'std', 'max'])
        print(loss_analysis.to_string())

        # Análisis por Decay (solo para Adaptive)
        if 'Decay' in df_success.columns and df_success['Decay'].notna().any():
            print("\n" + "="*80)
            print("ANÁLISIS POR DECAY (promedio Micro F1)")
            print("="*80)
            decay_analysis = df_success[df_success['Loss'] == 'AdaptiveWeightedBCE'].groupby('Decay')['Micro F1'].agg(['mean', 'std', 'max'])
            print(decay_analysis.to_string())

        # Análisis por Pos Weight
        print("\n" + "="*80)
        print("ANÁLISIS POR POS WEIGHT (promedio Micro F1)")
        print("="*80)
        weight_analysis = df_success.groupby('Pos Weight')['Micro F1'].agg(['mean', 'std', 'max'])
        print(weight_analysis.to_string())

        # Análisis por Hidden Dim
        print("\n" + "="*80)
        print("ANÁLISIS POR HIDDEN DIM (promedio Micro F1)")
        print("="*80)
        hidden_analysis = df_success.groupby('Hidden Dim')['Micro F1'].agg(['mean', 'std', 'max'])
        print(hidden_analysis.to_string())

        # Análisis por Num Layers
        print("\n" + "="*80)
        print("ANÁLISIS POR NUM LAYERS (promedio Micro F1)")
        print("="*80)
        layers_analysis = df_success.groupby('Num Layers')['Micro F1'].agg(['mean', 'std', 'max'])
        print(layers_analysis.to_string())
    
    print("\n" + "="*80)
    print(f"✅ Resultados guardados en: {csv_filename}")
    print(f"Fin del experimento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
