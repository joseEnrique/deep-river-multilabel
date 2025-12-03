"""
Búsqueda de hiperparámetros para RollingMultiLabelClassifier y PretrainedRollingMultiLabelClassifier
Prueba diferentes combinaciones de window_size y epochs
Guarda resultados en CSV para análisis
"""

import csv
import sys
from typing import Dict
from pathlib import Path
import torch
import time
from datetime import datetime

# Rutas locales
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Importa modelo y clasificadores
from model import LSTM_MultiLabel
from deep_river.classification import (
    RollingMultiLabelClassifier,
    PretrainedRollingMultiLabelClassifier
)

# Métricas incrementales personalizadas
testforecaster_path = ROOT / "testforecaster"
sys.path.insert(0, str(testforecaster_path))
from incrementalmetrics import IncrementalMultiLabelMetrics  # type: ignore


def stream_csv_to_dicts(filepath: str, feature_cols, target_names):
    """Generador que convierte el CSV en un flujo de (x, y) para aprendizaje online."""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = {k: float(row[k]) for k in feature_cols}
                y = {t: int(float(row.get(t, 0))) for t in target_names}
                yield x, y
            except ValueError:
                continue


def test_configuration(clf, dataset_path, feature_cols, target_names,
                       classifier_type, window_size, epochs, results_file, use_scaler=True):
    """
    Prueba una configuración específica y guarda resultados.

    Args:
        use_scaler: Si True, usa StandardScaler. False para modelos pretrained.
    """
    print(f"\n{'='*60}")
    print(f"Probando: {classifier_type}")
    print(f"  window_size={window_size}, epochs={epochs}")
    print(f"  use_scaler={use_scaler}")
    print(f"{'='*60}")

    # Métricas incrementales
    metrics_agg = IncrementalMultiLabelMetrics(label_names=target_names)
    total_samples = 0

    # Scaler solo para modelos NO pretrained
    from river import preprocessing
    scaler = preprocessing.StandardScaler() if use_scaler else None

    start_time = time.time()

    # Bucle principal - usar TODOS los datos
    for x, y in stream_csv_to_dicts(dataset_path, feature_cols, target_names):
        total_samples += 1

        # Aplicar scaler si está habilitado
        if scaler is not None:
            x_scaled = scaler.transform_one(x)
        else:
            x_scaled = x

        # Predicción incremental
        y_pred = clf.predict_one(x_scaled)

        # Actualizar métricas
        y_true_vec = [y[t] for t in target_names]
        y_pred_vec = [y_pred[t] for t in target_names]
        metrics_agg.update(y_true_vec, y_pred_vec)

        # Aprendizaje online
        clf.learn_one(x_scaled, y)

        # Actualizar scaler DESPUÉS de usar
        if scaler is not None:
            scaler.learn_one(x)

        # Progreso
        if total_samples % 500 == 0:
            print(f"  Procesadas {total_samples} muestras...")

    elapsed_time = time.time() - start_time

    # Obtener métricas finales
    final_metrics = metrics_agg.results()

    # Escribir resultados en CSV
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            classifier_type,
            window_size,
            epochs,
            total_samples,
            f"{final_metrics['accuracy']:.6f}",
            f"{final_metrics['micro_f1']:.6f}",
            f"{final_metrics['macro_f1']:.6f}",
            f"{final_metrics['subset_acc']:.6f}",
            f"{final_metrics['hamming_loss']:.6f}",
            f"{final_metrics['example_f1']:.6f}",
            f"{final_metrics['micro_prec']:.6f}",
            f"{final_metrics['micro_rec']:.6f}",
            f"{final_metrics['macro_prec']:.6f}",
            f"{final_metrics['macro_rec']:.6f}",
            f"{elapsed_time:.2f}"
        ])

    print(f"\n  Resultados:")
    print(f"    MicroF1: {final_metrics['micro_f1']:.6f}")
    print(f"    MacroF1: {final_metrics['macro_f1']:.6f}")
    print(f"    Accuracy: {final_metrics['accuracy']:.6f}")
    print(f"    Tiempo: {elapsed_time:.2f}s")

    return final_metrics


if __name__ == "__main__":
    # === Configuración general ===
    dataset_path = "/home/quique/tesis/OEMLHAT4PdM/datasets/ai4i2020formatted.csv"
    feature_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type_H",
        "Type_L",
        "Type_M",
    ]
    target_names = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Contar total de filas
    import pandas as pd
    df_full = pd.read_csv(dataset_path)
    total_rows = len(df_full)

    print("="*60)
    print("BÚSQUEDA DE HIPERPARÁMETROS")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Total filas: {total_rows}")
    print(f"Usando TODOS los datos (aprendizaje incremental)")
    print(f"Dispositivo: {device}")

    # === Definir combinaciones de hiperparámetros ===
    window_sizes = [50, 100, 500,1000,2000]
    epochs_list = [1, 3, 10,50,100,150]

    # Archivo de resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = ROOT / "testclassifier" / f"hyperparameter_results_{timestamp}.csv"

    # Crear archivo CSV con encabezados
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'classifier_type',
            'window_size',
            'epochs',
            'total_samples',
            'accuracy',
            'micro_f1',
            'macro_f1',
            'subset_accuracy',
            'hamming_loss',
            'example_f1',
            'micro_precision',
            'micro_recall',
            'macro_precision',
            'macro_recall',
            'elapsed_time_seconds'
        ])

    print(f"\nResultados se guardarán en: {results_file}")
    print(f"\nCombinaciones a probar: {len(window_sizes) * len(epochs_list) * 2}")

    # === 1. Probar RollingMultiLabelClassifier (desde cero) ===
    print("\n" + "="*60)
    print("PARTE 1: RollingMultiLabelClassifier (entrenamiento desde cero)")
    print("="*60)

    for window_size in window_sizes:
        for epochs in epochs_list:
            try:
                clf = RollingMultiLabelClassifier(
                    module=LSTM_MultiLabel,
                    label_names=target_names,
                    optimizer_fn="adam",
                    lr=1e-3,
                    device=device,
                    window_size=window_size,
                    append_predict=False,
                    hidden_dim=64,
                    num_layers=2,
                    dropout=0.2,
                    bidirectional=True,
                    epochs=epochs,
                    output_dim=len(target_names),
                )

                test_configuration(
                    clf=clf,
                    dataset_path=dataset_path,
                    feature_cols=feature_cols,
                    target_names=target_names,
                    classifier_type="RollingMultiLabel",
                    window_size=window_size,
                    epochs=epochs,
                    results_file=results_file,
                    use_scaler=True  # RollingMultiLabel SÍ usa scaler
                )

            except Exception as e:
                print(f"ERROR en RollingMultiLabel (ws={window_size}, ep={epochs}): {e}")
                # Guardar error en el CSV
                with open(results_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "RollingMultiLabel",
                        window_size,
                        epochs,
                        0,
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        0
                    ])

    # === 2. Probar PretrainedRollingMultiLabelClassifier ===
    print("\n" + "="*60)
    print("PARTE 2: PretrainedRollingMultiLabelClassifier (con pesos preentrenados)")
    print("="*60)

    checkpoint_path = ROOT / "testbatch" / "lstm_multilabel_ai4i_complete.pt"
    scaler_path = ROOT / "testbatch" / "scaler_ai4i.pkl"

    for window_size in window_sizes:
        for epochs in epochs_list:
            try:
                clf = PretrainedRollingMultiLabelClassifier(
                    module=LSTM_MultiLabel,
                    label_names=target_names,
                    checkpoint_path=str(checkpoint_path),
                    scaler_path=str(scaler_path),
                    optimizer_fn="adam",
                    lr=1e-4,  # LR más bajo para fine-tuning
                    device=device,
                    window_size=window_size,
                    append_predict=False,
                    freeze_pretrained=False,
                    hidden_dim=64,
                    num_layers=2,
                    dropout=0.2,
                    bidirectional=True,
                    epochs=epochs,
                )

                test_configuration(
                    clf=clf,
                    dataset_path=dataset_path,
                    feature_cols=feature_cols,
                    target_names=target_names,
                    classifier_type="PretrainedRollingMultiLabel",
                    window_size=window_size,
                    epochs=epochs,
                    results_file=results_file,
                    use_scaler=False  # Pretrained NO usa scaler (ya normalizado)
                )

            except Exception as e:
                print(f"ERROR en PretrainedRollingMultiLabel (ws={window_size}, ep={epochs}): {e}")
                # Guardar error en el CSV
                with open(results_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "PretrainedRollingMultiLabel",
                        window_size,
                        epochs,
                        0,
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        0
                    ])

    # === Resumen final ===
    print("\n" + "="*60)
    print("BÚSQUEDA COMPLETADA")
    print("="*60)
    print(f"Resultados guardados en: {results_file}")
    print("\nPuedes analizar los resultados con:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_csv('{results_file}')")
    print(f"  print(df.sort_values('micro_f1', ascending=False))")
