"""
Búsqueda exhaustiva de hiperparámetros para LSTM con Rolling Window (Dataset ALPI - Máquina 2).

Este script usa la clase datasets.multioutput.alpi.Alpi estándar SIN MODIFICACIONES.
NOTA: El rendimiento puede ser lento debido a la implementación original de process_raw.

EJECUCIÓN EN PARALELO: Usa 2 GPUs (cuda:0 y cuda:1) simultáneamente.
"""

import torch
import pandas as pd
import numbers
import numpy as np
import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import time

# Importar Alpi estándar
from datasets.multioutput.alpi import Alpi
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier 
from testclassifier.model import LSTM_MultiLabel
import evaluate
from river.compose import SelectType
from river.metrics import F1
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch
from river import preprocessing
from custommetrics.multioutput import *
import bisect

# ============================================================================
# Configuraciones a probar - COPIADO DE lstm_nps_bce.py
# ============================================================================
configs = [
    # ============================================================================
    # SMALL LSTM [hidden_dim=64, 1 layer] - Window size sweep
    # ============================================================================
    {"name": "LSTM-Small-LR1e3-W100", "hidden_dim": 64, "num_layers": 1, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-Small-LR1e3-W200", "hidden_dim": 64, "num_layers": 1, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-Small-LR1e3-W500", "hidden_dim": 64, "num_layers": 1, "lr": 1e-3, "dropout": 0.3, "bidirectional": True, "window_size": 500},

    {"name": "LSTM-Small-LR5e4-W100", "hidden_dim": 64, "num_layers": 1, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-Small-LR5e4-W200", "hidden_dim": 64, "num_layers": 1, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-Small-LR5e4-W500", "hidden_dim": 64, "num_layers": 1, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500},

    {"name": "LSTM-Small-LR1e4-W100", "hidden_dim": 64, "num_layers": 1, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-Small-LR1e4-W200", "hidden_dim": 64, "num_layers": 1, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-Small-LR1e4-W500", "hidden_dim": 64, "num_layers": 1, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500},

    # ============================================================================
    # MEDIUM LSTM [hidden_dim=128, 2 layers] - Window size sweep
    # ============================================================================
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

    # ============================================================================
    # LARGE LSTM [hidden_dim=256, 2 layers] - Window size sweep
    # ============================================================================
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

    # ============================================================================
    # XLARGE LSTM [hidden_dim=256, 3 layers] - Window size sweep
    # ============================================================================
    {"name": "LSTM-XLarge-LR5e4-W100", "hidden_dim": 256, "num_layers": 3, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-XLarge-LR5e4-W200", "hidden_dim": 256, "num_layers": 3, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-XLarge-LR5e4-W500", "hidden_dim": 256, "num_layers": 3, "lr": 5e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500},

    {"name": "LSTM-XLarge-LR1e4-W100", "hidden_dim": 256, "num_layers": 3, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-XLarge-LR1e4-W200", "hidden_dim": 256, "num_layers": 3, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-XLarge-LR1e4-W500", "hidden_dim": 256, "num_layers": 3, "lr": 1e-4, "dropout": 0.3, "bidirectional": True, "window_size": 500},

    {"name": "LSTM-XLarge-LR5e5-W100", "hidden_dim": 256, "num_layers": 3, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 100},
    {"name": "LSTM-XLarge-LR5e5-W200", "hidden_dim": 256, "num_layers": 3, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 200},
    {"name": "LSTM-XLarge-LR5e5-W500", "hidden_dim": 256, "num_layers": 3, "lr": 5e-5, "dropout": 0.3, "bidirectional": True, "window_size": 500},

    # ============================================================================
    # UNIDIRECTIONAL LSTM comparisons (Medium architecture)
    # ============================================================================
    {"name": "LSTM-Medium-Uni-LR1e3-W200", "hidden_dim": 128, "num_layers": 2, "lr": 1e-3, "dropout": 0.3, "bidirectional": False, "window_size": 200},
    {"name": "LSTM-Medium-Uni-LR5e4-W200", "hidden_dim": 128, "num_layers": 2, "lr": 5e-4, "dropout": 0.3, "bidirectional": False, "window_size": 200},
    {"name": "LSTM-Medium-Uni-LR1e4-W200", "hidden_dim": 128, "num_layers": 2, "lr": 1e-4, "dropout": 0.3, "bidirectional": False, "window_size": 200},
]


def run_single_experiment(args):
    """
    Ejecuta un experimento en una GPU específica.
    """
    i, config, device_id, label_names = args
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*80}")
    print(f"[GPU {device_id}] [{i+1}/{len(configs)}] Probando: {config['name']}")
    print(f"  - Device: {device}")
    print(f"  - hidden_dim: {config['hidden_dim']}")
    print(f"  - window_size: {config['window_size']}")
    print(f"{'='*80}\n")

    try:
        start_time = time.time()
        # Cargar dataset estándar Alpi (Máquina 2)
        stream_obj = Alpi(machine=2)
        # Asegurar keys string para consistencia con RollingMultiLabelClassifier
        stream_obj.Y.columns = stream_obj.Y.columns.astype(str)
        
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
            seed=42,
            epochs=10
        )

        pipeliner = SelectType(numbers.Number) | preprocessing.StandardScaler() | clf

        metrics_result = evaluate.progressive_val_score(
            dataset=stream_obj,
            model=pipeliner,
            metric=Metrics([ExactMatch(), MacroAverage(F1()), MicroAverage(F1())]),
            show_memory=True,
            print_every=2000
        )

        exact_match = metrics_result[0].get()
        macro_f1 = metrics_result[1].get()
        micro_f1 = metrics_result[2].get()

        duration = time.time() - start_time
        duration_str = f"{duration:.2f}s"

        result = {
            "Model": config['name'],
            "Hidden Dim": config['hidden_dim'],
            "Num Layers": config['num_layers'],
            "Bidirectional": config['bidirectional'],
            "LR": config['lr'],
            "Dropout": config['dropout'],
            "Window Size": config['window_size'],
            "ExactMatch": exact_match * 100,
            "Macro F1": macro_f1 * 100,
            "Micro F1": micro_f1 * 100,
            "Status": "Success",
            "GPU": device_id,
            "Duration": duration_str
        }

        print(f"\n[GPU {device_id}] ✅ Resultados - Macro F1: {macro_f1*100:.2f}%\n")
        return result

    except Exception as e:
        print(f"\n[GPU {device_id}] ❌ ERROR en {config['name']}: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return {
            "Model": config['name'],
            "Status": f"Error: {str(e)}",
            "Macro F1": 0.0,
            "Micro F1": 0.0
        }

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print("="*80)
    print("BÚSQUEDA DE HIPERPARÁMETROS LSTM - ALPI Dataset (Máquina 2) [Standard Alpi Class]")
    print("="*80)
    print(f"Inicio del experimento: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("WARMUP: Cargando Dataset Alpi(2)...")
    
    # Warmup uses Standard Alpi
    warmup_stream = Alpi(machine=2)
    warmup_stream.Y.columns = warmup_stream.Y.columns.astype(str)
    target_names = list(warmup_stream.Y.columns)
    
    print(f"Dataset listo. Muestras: {warmup_stream.n_samples}, Etiquetas: {warmup_stream.n_outputs}")
    print(f"Etiquetas (Target Names): {target_names[:5]}...")
    
    experiment_args = []
    for idx, config in enumerate(configs):
        gpu_id = idx % 2 
        experiment_args.append((idx, config, gpu_id, target_names))

    results = []
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_single_experiment, args) for args in experiment_args]
        
        for future in futures:
            result = future.result()
            results.append(result)
            
            # Guardar resultados intermedios
            if len(results) % 2 == 0:
                df_temp = pd.DataFrame(results)
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                df_temp.to_csv(f'results/lstmalpi/lstm_alpi_partial_{timestamp}.csv', index=False)
                print(f"💾 Guardados resultados intermedios ({len(results)}/{len(configs)})")

    
    df_results = pd.DataFrame(results)
    print(f"Fin del experimento: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('results/lstmalpi', exist_ok=True)
    csv_filename = f'results/lstmalpi/lstm_alpi_m2_std_results_{timestamp}.csv'
    df_results.to_csv(csv_filename, index=False)
    
    print("\n" + "="*80)
    print(df_results.to_string(index=False))
    print(f"\nResultados guardados en {csv_filename}")

    # Análisis de resultados
    df_success = df_results[df_results['Status'] == 'Success'].copy()

    if len(df_success) > 0:
        df_success_sorted = df_success.sort_values('Macro F1', ascending=False)

        print("\n" + "="*80)
        print("TOP 10 MODELOS POR MACRO F1")
        print("="*80)
        print(df_success_sorted.head(10).to_string(index=False))

        print("\n" + "="*80)
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

        # Análisis por window size
        print("\n" + "="*80)
        print("ANÁLISIS POR WINDOW SIZE (promedio Macro F1)")
        print("="*80)
        window_analysis = df_success.groupby('Window Size')['Macro F1'].agg(['mean', 'std', 'max'])
        print(window_analysis.to_string())

        # Análisis por LR
        print("\n" + "="*80)
        print("ANÁLISIS POR LEARNING RATE (promedio Macro F1)")
        print("="*80)
        lr_analysis = df_success.groupby('LR')['Macro F1'].agg(['mean', 'std', 'max'])
        print(lr_analysis.to_string())

        # Análisis por hidden_dim
        print("\n" + "="*80)
        print("ANÁLISIS POR HIDDEN DIM (promedio Macro F1)")
        print("="*80)
        hidden_analysis = df_success.groupby('Hidden Dim')['Macro F1'].agg(['mean', 'std', 'max'])
        print(hidden_analysis.to_string())

        # Análisis por num_layers
        print("\n" + "="*80)
        print("ANÁLISIS POR NUM LAYERS (promedio Macro F1)")
        print("="*80)
        layers_analysis = df_success.groupby('Num Layers')['Macro F1'].agg(['mean', 'std', 'max'])
        print(layers_analysis.to_string())

        # Análisis bidirectional vs unidirectional
        print("\n" + "="*80)
        print("ANÁLISIS BIDIRECTIONAL vs UNIDIRECTIONAL (promedio Macro F1)")
        print("="*80)
        bi_analysis = df_success.groupby('Bidirectional')['Macro F1'].agg(['mean', 'std', 'max'])
        print(bi_analysis.to_string())

    print("\n" + "="*80)
    print(f"✅ Resultados guardados en: {csv_filename}")
    print(f"Fin del experimento: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
