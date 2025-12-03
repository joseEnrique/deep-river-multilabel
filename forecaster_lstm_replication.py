"""
Replication of LSTM Window Search with DirectMultiLabelForecaster and varying past_size.

This script replicates the configurations from lstm_window_size_search.py but uses
DirectMultiLabelForecaster and tests with past_size=[1, 5, 10] to evaluate the impact
of historical context from the dataset stream on model performance.

EJECUCIÓN EN PARALELO: Usa 2 GPUs (cuda:0 y cuda:1) simultáneamente.
"""

import torch
import pandas as pd
from datetime import datetime
from testclassifier.model import LSTM_MultiLabel
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

# Base configurations (without past_size - it will be added programmatically)
# These are replicated from lstm_window_size_search.py
base_configs = [
    # ============================================================================
    # SMALL LSTM [hidden_dim=64, 1 layer] - Window size sweep
    # ============================================================================
    {"name": "LSTM-Small-LR1e3-W100", "hidden_dim": 64, "num_layers": 1, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 100, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Small-LR1e3-W200", "hidden_dim": 64, "num_layers": 1, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Small-LR1e3-W500", "hidden_dim": 64, "num_layers": 1, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 500, "loss": "BCEWithLogits", "epochs": 10},

    {"name": "LSTM-Small-LR5e4-W100", "hidden_dim": 64, "num_layers": 1, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Small-LR5e4-W200", "hidden_dim": 64, "num_layers": 1, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Small-LR5e4-W500", "hidden_dim": 64, "num_layers": 1, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500, "loss": "BCEWithLogits", "epochs": 10},

    {"name": "LSTM-Small-LR1e4-W100", "hidden_dim": 64, "num_layers": 1, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Small-LR1e4-W200", "hidden_dim": 64, "num_layers": 1, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Small-LR1e4-W500", "hidden_dim": 64, "num_layers": 1, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500, "loss": "BCEWithLogits", "epochs": 10},

    # ============================================================================
    # MEDIUM LSTM [hidden_dim=128, 2 layers] - Window size sweep
    # ============================================================================
    {"name": "LSTM-Medium-LR1e3-W100", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 100, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Medium-LR1e3-W200", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Medium-LR1e3-W500", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 500, "loss": "BCEWithLogits", "epochs": 10},

    {"name": "LSTM-Medium-LR5e4-W100", "hidden_dim": 128, "num_layers": 2, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Medium-LR5e4-W200", "hidden_dim": 128, "num_layers": 2, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Medium-LR5e4-W500", "hidden_dim": 128, "num_layers": 2, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500, "loss": "BCEWithLogits", "epochs": 10},

    {"name": "LSTM-Medium-LR1e4-W100", "hidden_dim": 128, "num_layers": 2, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Medium-LR1e4-W200", "hidden_dim": 128, "num_layers": 2, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Medium-LR1e4-W500", "hidden_dim": 128, "num_layers": 2, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500, "loss": "BCEWithLogits", "epochs": 10},

    {"name": "LSTM-Medium-LR5e5-W100", "hidden_dim": 128, "num_layers": 2, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 100, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Medium-LR5e5-W200", "hidden_dim": 128, "num_layers": 2, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Medium-LR5e5-W500", "hidden_dim": 128, "num_layers": 2, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 500, "loss": "BCEWithLogits", "epochs": 10},

    # ============================================================================
    # LARGE LSTM [hidden_dim=256, 2 layers] - Window size sweep
    # ============================================================================
    {"name": "LSTM-Large-LR1e3-W100", "hidden_dim": 256, "num_layers": 2, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 100, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Large-LR1e3-W200", "hidden_dim": 256, "num_layers": 2, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Large-LR1e3-W500", "hidden_dim": 256, "num_layers": 2, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 500, "loss": "BCEWithLogits", "epochs": 10},

    {"name": "LSTM-Large-LR5e4-W100", "hidden_dim": 256, "num_layers": 2, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Large-LR5e4-W200", "hidden_dim": 256, "num_layers": 2, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Large-LR5e4-W500", "hidden_dim": 256, "num_layers": 2, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500, "loss": "BCEWithLogits", "epochs": 10},

    {"name": "LSTM-Large-LR1e4-W100", "hidden_dim": 256, "num_layers": 2, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Large-LR1e4-W200", "hidden_dim": 256, "num_layers": 2, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Large-LR1e4-W500", "hidden_dim": 256, "num_layers": 2, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500, "loss": "BCEWithLogits", "epochs": 10},

    {"name": "LSTM-Large-LR5e5-W100", "hidden_dim": 256, "num_layers": 2, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 100, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Large-LR5e5-W200", "hidden_dim": 256, "num_layers": 2, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Large-LR5e5-W500", "hidden_dim": 256, "num_layers": 2, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 500, "loss": "BCEWithLogits", "epochs": 10},

    # ============================================================================
    # XLARGE LSTM [hidden_dim=256, 3 layers] - Window size sweep
    # ============================================================================
    {"name": "LSTM-XLarge-LR5e4-W100", "hidden_dim": 256, "num_layers": 3, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-XLarge-LR5e4-W200", "hidden_dim": 256, "num_layers": 3, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-XLarge-LR5e4-W500", "hidden_dim": 256, "num_layers": 3, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500, "loss": "BCEWithLogits", "epochs": 10},

    {"name": "LSTM-XLarge-LR1e4-W100", "hidden_dim": 256, "num_layers": 3, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-XLarge-LR1e4-W200", "hidden_dim": 256, "num_layers": 3, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-XLarge-LR1e4-W500", "hidden_dim": 256, "num_layers": 3, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500, "loss": "BCEWithLogits", "epochs": 10},

    {"name": "LSTM-XLarge-LR5e5-W100", "hidden_dim": 256, "num_layers": 3, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 100, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-XLarge-LR5e5-W200", "hidden_dim": 256, "num_layers": 3, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-XLarge-LR5e5-W500", "hidden_dim": 256, "num_layers": 3, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 500, "loss": "BCEWithLogits", "epochs": 10},

    # ============================================================================
    # UNIDIRECTIONAL LSTM comparisons (Medium architecture)
    # ============================================================================
    {"name": "LSTM-Medium-Uni-LR1e3-W200", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "dropout": 0.3, "bidirectional": False, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Medium-Uni-LR5e4-W200", "hidden_dim": 128, "num_layers": 2, "lr": 5e-4, "dropout": 0.3, "bidirectional": False, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
    {"name": "LSTM-Medium-Uni-LR1e4-W200", "hidden_dim": 128, "num_layers": 2, "lr": 1e-4, "dropout": 0.3, "bidirectional": False, "window_size": 200, "loss": "BCEWithLogits", "epochs": 10},
]

# Generate configs for past_size values: 1, 5, 10
configs = []
for past_size in [1, 5, 10]:
    for base_config in base_configs:
        # Create a copy with past_size added
        config = base_config.copy()
        config['past_size'] = past_size
        # Append past_size to the name for clarity
        config['name'] = f"{base_config['name']}-P{past_size}"
        configs.append(config)


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
    
    print(f"\\n{'='*80}")
    print(f"[GPU {device_id}] [{i+1}/{len(configs)}] Probando: {config['name']}")
    print(f"  - Device: {device}")
    print(f"  - Loss: {config['loss']}")
    print(f"  - hidden_dim: {config['hidden_dim']}")
    print(f"  - num_layers: {config['num_layers']}")
    print(f"  - lr: {config['lr']}")
    print(f"  - dropout: {config['dropout']}")
    print(f"  - bidirectional: {config['bidirectional']}")
    print(f"  - past_size: {config['past_size']}")
    print(f"  - window_size: {config['window_size']}")
    print(f"  - epochs: {config['epochs']}")
    print(f"{'='*80}\\n")

    try:
        # Setup numeric and target columns for pipeline
        numeric_cols = ['Air temperature [K]', 'Process temperature [K]',
                        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        target_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        
        # Instantiate Loss Function
        if config['loss'] == 'BCEWithLogits':
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss function: {config['loss']}")

        # Create forecaster
        # Note: Passing dropout and bidirectional via kwargs
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
            threshold=0.5, # Default threshold
            dropout=config['dropout'],
            bidirectional=config['bidirectional']
        )

        # Pipeline setup
        p_numeric = compose.Select(*numeric_cols) | preprocessing.StandardScaler()
        p_categorical = compose.Select('Type') | preprocessing.OneHotEncoder()
        # Note: No p_targets since include_targets=False (targets not in DataFrame)
        pipeline = (p_numeric + p_categorical) | forecaster

        # Create stream
        # past_size=1: This is the key change requested
        # include_targets=False: Critical fix to prevent data leakage (targets should NOT be in input)
        stream = RollingAi4i(past_size=config['past_size'], n_instances=10000, include_targets=False)

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
            "Dropout": config['dropout'],
            "Bidirectional": config['bidirectional'],
            "Past Size": config['past_size'],
            "Window Size": config['window_size'],
            "Epochs": config['epochs'],
            "ExactMatch": exact_match * 100,
            "Macro F1": macro_f1 * 100,
            "Micro F1": micro_f1 * 100,
            "Status": "Success",
            "GPU": device_id
        }

        print(f"\\n[GPU {device_id}] ✅ {config['name']} - ExactMatch: {exact_match*100:.2f}%, Macro F1: {macro_f1*100:.2f}%, Micro F1: {micro_f1*100:.2f}%\\n")
        return result

    except Exception as e:
        print(f"\\n[GPU {device_id}] ❌ ERROR en {config['name']}: {str(e)}\\n")
        import traceback
        traceback.print_exc()
        
        return {
            "Model": config['name'],
            "Loss": config['loss'],
            "Hidden Dim": config['hidden_dim'],
            "Num Layers": config['num_layers'],
            "LR": config['lr'],
            "Dropout": config['dropout'],
            "Bidirectional": config['bidirectional'],
            "Past Size": config['past_size'],
            "Window Size": config['window_size'],
            "Epochs": config['epochs'],
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
    
    # Create output directory
    output_dir = 'results/forecasterlstmbce'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("REPLICACIÓN DE LSTM WINDOW SEARCH - DirectMultiLabelForecaster")
    print(f"Testing past_size values: 1, 5, 10")
    print("="*80)
    print(f"Inicio del experimento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total de configuraciones: {len(configs)}")
    print(f"Modo paralelo: 2 GPUs (cuda:0 y cuda:1)")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Prepare arguments for parallel execution
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
                partial_file = os.path.join(output_dir, f'lstm_replication_partial_{timestamp}.csv')
                df_temp.to_csv(partial_file, index=False)
                print(f"\n💾 Guardados resultados intermedios ({len(results)}/{len(configs)} completados)\n")
    
    # Crear DataFrame con resultados finales
    df_results = pd.DataFrame(results)
    
    # Guardar resultados completos
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(output_dir, f'lstm_replication_results_{timestamp}.csv')
    df_results.to_csv(csv_filename, index=False)
    
    print("\\n" + "="*80)
    print("RESULTADOS FINALES - Todos los experimentos")
    print("="*80)
    print(df_results.to_string(index=False))
    print("\\n")
    
    # Análisis de resultados
    df_success = df_results[df_results['Status'] == 'Success'].copy()
    
    if len(df_success) > 0:
        df_success_sorted = df_success.sort_values('Macro F1', ascending=False)
    
        print("\\n" + "="*80)
        print("TOP 10 MODELOS POR MACRO F1")
        print("="*80)
        print(df_success_sorted.head(10).to_string(index=False))
    
        print("\\n" + "="*80)
        print("MEJOR CONFIGURACIÓN")
        print("="*80)
        best = df_success_sorted.iloc[0]
        print(f"🏆 Modelo: {best['Model']}")
        print(f"   - Hidden Dim: {best['Hidden Dim']}")
        print(f"   - Num Layers: {best['Num Layers']}")
        print(f"   - Bidirectional: {best['Bidirectional']}")
        print(f"   - LR: {best['LR']}")
        print(f"   - Dropout: {best['Dropout']}")
        print(f"   - Window Size: {best['Window Size']}")
        print(f"   - ExactMatch: {best['ExactMatch']:.2f}%")
        print(f"   - Macro F1: {best['Macro F1']:.2f}%")
        print(f"   - Micro F1: {best['Micro F1']:.2f}%")
    
    print("\\n" + "="*80)
    print(f"✅ Resultados guardados en: {csv_filename}")
    print(f"Fin del experimento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
