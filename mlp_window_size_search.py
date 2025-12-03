"""
Búsqueda exhaustiva de hiperparámetros para MLP con Rolling Window.

Este script prueba diferentes configuraciones de:
- Learning rates
- Arquitecturas (hidden_dims)
- Window sizes

Los resultados se guardan en CSV para análisis posterior.
"""

import torch
import pandas as pd
import numbers
from datetime import datetime
from testclassifier.model import MLP_MultiLabel
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier
from datasets.multioutput import Ai4i
import evaluate
import numbers
import torch
import sys
from river.compose import SelectType
from river.metrics import F1
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch
from river import preprocessing
from custommetrics.multioutput import *

# Configuración del dispositivo
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Inicio del experimento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Target names
target_names = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Configuraciones a probar - ENFOQUE EN WINDOW SIZE Y LEARNING RATE
configs = [
    # ============================================================================
    # SMALL ARCHITECTURE [64, 32] - Window size sweep
    # ============================================================================
    {"name": "MLP-Small-LR1e3-W100", "hidden_dims": [64, 32], "lr": 1e-3, "dropout": 0.3, "window_size": 100},
    {"name": "MLP-Small-LR1e3-W200", "hidden_dims": [64, 32], "lr": 1e-3, "dropout": 0.3, "window_size": 200},
    {"name": "MLP-Small-LR1e3-W500", "hidden_dims": [64, 32], "lr": 1e-3, "dropout": 0.3, "window_size": 500},

    {"name": "MLP-Small-LR5e4-W100", "hidden_dims": [64, 32], "lr": 5e-4, "dropout": 0.3, "window_size": 100},
    {"name": "MLP-Small-LR5e4-W200", "hidden_dims": [64, 32], "lr": 5e-4, "dropout": 0.3, "window_size": 200},
    {"name": "MLP-Small-LR5e4-W500", "hidden_dims": [64, 32], "lr": 5e-4, "dropout": 0.3, "window_size": 500},

    {"name": "MLP-Small-LR1e4-W100", "hidden_dims": [64, 32], "lr": 1e-4, "dropout": 0.3, "window_size": 100},
    {"name": "MLP-Small-LR1e4-W200", "hidden_dims": [64, 32], "lr": 1e-4, "dropout": 0.3, "window_size": 200},
    {"name": "MLP-Small-LR1e4-W500", "hidden_dims": [64, 32], "lr": 1e-4, "dropout": 0.3, "window_size": 500},

    # ============================================================================
    # MEDIUM ARCHITECTURE [128, 64, 32] - Window size sweep
    # ============================================================================
    {"name": "MLP-Medium-LR1e3-W100", "hidden_dims": [128, 64, 32], "lr": 1e-3, "dropout": 0.3, "window_size": 100},
    {"name": "MLP-Medium-LR1e3-W200", "hidden_dims": [128, 64, 32], "lr": 1e-3, "dropout": 0.3, "window_size": 200},
    {"name": "MLP-Medium-LR1e3-W500", "hidden_dims": [128, 64, 32], "lr": 1e-3, "dropout": 0.3, "window_size": 500},

    {"name": "MLP-Medium-LR5e4-W100", "hidden_dims": [128, 64, 32], "lr": 5e-4, "dropout": 0.3, "window_size": 100},
    {"name": "MLP-Medium-LR5e4-W200", "hidden_dims": [128, 64, 32], "lr": 5e-4, "dropout": 0.3, "window_size": 200},
    {"name": "MLP-Medium-LR5e4-W500", "hidden_dims": [128, 64, 32], "lr": 5e-4, "dropout": 0.3, "window_size": 500},

    {"name": "MLP-Medium-LR1e4-W100", "hidden_dims": [128, 64, 32], "lr": 1e-4, "dropout": 0.3, "window_size": 100},
    {"name": "MLP-Medium-LR1e4-W200", "hidden_dims": [128, 64, 32], "lr": 1e-4, "dropout": 0.3, "window_size": 200},
    {"name": "MLP-Medium-LR1e4-W500", "hidden_dims": [128, 64, 32], "lr": 1e-4, "dropout": 0.3, "window_size": 500},

    {"name": "MLP-Medium-LR5e5-W100", "hidden_dims": [128, 64, 32], "lr": 5e-5, "dropout": 0.3, "window_size": 100},
    {"name": "MLP-Medium-LR5e5-W200", "hidden_dims": [128, 64, 32], "lr": 5e-5, "dropout": 0.3, "window_size": 200},
    {"name": "MLP-Medium-LR5e5-W500", "hidden_dims": [128, 64, 32], "lr": 5e-5, "dropout": 0.3, "window_size": 500},

    # ============================================================================
    # LARGE ARCHITECTURE [256, 128, 64] - Window size sweep
    # ============================================================================
    {"name": "MLP-Large-LR1e3-W100", "hidden_dims": [256, 128, 64], "lr": 1e-3, "dropout": 0.3, "window_size": 100},
    {"name": "MLP-Large-LR1e3-W200", "hidden_dims": [256, 128, 64], "lr": 1e-3, "dropout": 0.3, "window_size": 200},
    {"name": "MLP-Large-LR1e3-W500", "hidden_dims": [256, 128, 64], "lr": 1e-3, "dropout": 0.3, "window_size": 500},

    {"name": "MLP-Large-LR5e4-W100", "hidden_dims": [256, 128, 64], "lr": 5e-4, "dropout": 0.3, "window_size": 100},
    {"name": "MLP-Large-LR5e4-W200", "hidden_dims": [256, 128, 64], "lr": 5e-4, "dropout": 0.3, "window_size": 200},
    {"name": "MLP-Large-LR5e4-W500", "hidden_dims": [256, 128, 64], "lr": 5e-4, "dropout": 0.3, "window_size": 500},

    {"name": "MLP-Large-LR1e4-W100", "hidden_dims": [256, 128, 64], "lr": 1e-4, "dropout": 0.3, "window_size": 100},
    {"name": "MLP-Large-LR1e4-W200", "hidden_dims": [256, 128, 64], "lr": 1e-4, "dropout": 0.3, "window_size": 200},
    {"name": "MLP-Large-LR1e4-W500", "hidden_dims": [256, 128, 64], "lr": 1e-4, "dropout": 0.3, "window_size": 500},

    {"name": "MLP-Large-LR5e5-W100", "hidden_dims": [256, 128, 64], "lr": 5e-5, "dropout": 0.3, "window_size": 100},
    {"name": "MLP-Large-LR5e5-W200", "hidden_dims": [256, 128, 64], "lr": 5e-5, "dropout": 0.3, "window_size": 200},
    {"name": "MLP-Large-LR5e5-W500", "hidden_dims": [256, 128, 64], "lr": 5e-5, "dropout": 0.3, "window_size": 500},

    # ============================================================================
    # XLARGE ARCHITECTURE [512, 256, 128] - Window size sweep (LR conservadores)
    # ============================================================================
    {"name": "MLP-XLarge-LR5e4-W100", "hidden_dims": [512, 256, 128], "lr": 5e-4, "dropout": 0.3, "window_size": 100},
    {"name": "MLP-XLarge-LR5e4-W200", "hidden_dims": [512, 256, 128], "lr": 5e-4, "dropout": 0.3, "window_size": 200},
    {"name": "MLP-XLarge-LR5e4-W500", "hidden_dims": [512, 256, 128], "lr": 5e-4, "dropout": 0.3, "window_size": 500},

    {"name": "MLP-XLarge-LR1e4-W100", "hidden_dims": [512, 256, 128], "lr": 1e-4, "dropout": 0.3, "window_size": 100},
    {"name": "MLP-XLarge-LR1e4-W200", "hidden_dims": [512, 256, 128], "lr": 1e-4, "dropout": 0.3, "window_size": 200},
    {"name": "MLP-XLarge-LR1e4-W500", "hidden_dims": [512, 256, 128], "lr": 1e-4, "dropout": 0.3, "window_size": 500},

    {"name": "MLP-XLarge-LR5e5-W100", "hidden_dims": [512, 256, 128], "lr": 5e-5, "dropout": 0.3, "window_size": 100},
    {"name": "MLP-XLarge-LR5e5-W200", "hidden_dims": [512, 256, 128], "lr": 5e-5, "dropout": 0.3, "window_size": 200},
    {"name": "MLP-XLarge-LR5e5-W500", "hidden_dims": [512, 256, 128], "lr": 5e-5, "dropout": 0.3, "window_size": 500},
]

print(f"\nTotal de configuraciones a probar: {len(configs)}")
print(f"{'='*80}\n")

results = []

for i, config in enumerate(configs, 1):
    print(f"\n{'='*80}")
    print(f"[{i}/{len(configs)}] Probando: {config['name']}")
    print(f"  - hidden_dims: {config['hidden_dims']}")
    print(f"  - lr: {config['lr']}")
    print(f"  - dropout: {config['dropout']}")
    print(f"  - window_size: {config['window_size']}")
    print(f"{'='*80}\n")

    try:
        # Reiniciar stream
        stream = Ai4i()

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
            epochs=10
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
        exact_match = metrics_result[0].get()
        macro_f1 = metrics_result[1].get()
        micro_f1 = metrics_result[2].get()

        # Guardar resultados
        results.append({
            "Model": config['name'],
            "Architecture": str(config['hidden_dims']),
            "LR": config['lr'],
            "Dropout": config['dropout'],
            "Window Size": config['window_size'],
            "ExactMatch": exact_match * 100,
            "Macro F1": macro_f1 * 100,
            "Micro F1": micro_f1 * 100,
            "Status": "Success"
        })

        print(f"\n✅ Resultados - ExactMatch: {exact_match*100:.2f}%, Macro F1: {macro_f1*100:.2f}%, Micro F1: {micro_f1*100:.2f}%\n")

        # Guardar resultados intermedios cada 5 configuraciones
        if i % 5 == 0:
            df_temp = pd.DataFrame(results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            df_temp.to_csv(f'mlp_window_search_partial_{timestamp}.csv', index=False)
            print(f"💾 Guardados resultados intermedios ({i}/{len(configs)} completados)")

    except Exception as e:
        print(f"\n❌ ERROR en {config['name']}: {str(e)}\n")
        results.append({
            "Model": config['name'],
            "Architecture": str(config['hidden_dims']),
            "LR": config['lr'],
            "Dropout": config['dropout'],
            "Window Size": config['window_size'],
            "ExactMatch": 0.0,
            "Macro F1": 0.0,
            "Micro F1": 0.0,
            "Status": f"Error: {str(e)}"
        })
        continue

# Crear DataFrame con resultados finales
df_results = pd.DataFrame(results)

# Guardar resultados completos
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_filename = f'mlp_window_search_results_{timestamp}.csv'
df_results.to_csv(csv_filename, index=False)

print("\n" + "="*80)
print("RESULTADOS FINALES - Todos los experimentos")
print("="*80)
print(df_results.to_string(index=False))
print("\n")

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
    print(f"   - Architecture: {best['Architecture']}")
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

    # Análisis por arquitectura
    print("\n" + "="*80)
    print("ANÁLISIS POR ARQUITECTURA (promedio Macro F1)")
    print("="*80)
    arch_analysis = df_success.groupby('Architecture')['Macro F1'].agg(['mean', 'std', 'max'])
    print(arch_analysis.to_string())

print("\n" + "="*80)
print(f"✅ Resultados guardados en: {csv_filename}")
print(f"Fin del experimento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
