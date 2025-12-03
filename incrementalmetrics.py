from __future__ import annotations
from typing import Dict, Sequence, List
import numpy as np


class IncrementalMultiLabelMetrics:
    """
    Incremental multi-label custommetrics (no batch storage).

    Compatible with River-style interface:
        m = IncrementalMultiLabelMetrics(["A","B","C"])
        for yt, yp in zip(y_true, y_pred):
            m.update(yt, yp)
        print(m.results())
    """

    def __init__(self, label_names: Sequence[str]):
        self.label_names = list[str](label_names)
        self.num_labels = len(self.label_names)

        # Global counts (micro)
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.total_bits = 0
        self.correct_bits = 0

        # Per-label (macro)
        self.tp_l = np.zeros(self.num_labels, dtype=int)
        self.fp_l = np.zeros(self.num_labels, dtype=int)
        self.fn_l = np.zeros(self.num_labels, dtype=int)

        # Example-based (averaged)
        self.sum_example_prec = 0.0
        self.sum_example_rec = 0.0
        self.sum_example_f1 = 0.0

        # Subset accuracy (exact match)
        self.total_samples = 0
        self.subset_correct = 0

        # Perfect predictions on positive cases
        self.total_positive_cases = 0
        self.perfect_predictions = 0

    def _to_vec(self, y: Dict[str, int] | Sequence[int]) -> np.ndarray:
        if isinstance(y, dict):
            return np.array([int(bool(y.get(name, 0))) for name in self.label_names], dtype=int)
        else:
            v = np.array(y, dtype=int)
            if len(v) != self.num_labels:
                raise ValueError(f"Expected {self.num_labels} labels, got {len(v)}")
            return (v > 0).astype(int)

    def update(self, y_true, y_pred):
        yt = self._to_vec(y_true)
        yp = self._to_vec(y_pred)
        self.total_samples += 1

        # Bit-level (for accuracy and hamming)
        self.correct_bits += int(np.sum(yt == yp))
        self.total_bits += self.num_labels

        # Exact match
        if np.array_equal(yt, yp):
            self.subset_correct += 1

        # Perfect on positive cases
        if np.sum(yt) > 0:
            self.total_positive_cases += 1
            if np.array_equal(yt, yp):
                self.perfect_predictions += 1

        # TP, FP, FN
        self.tp += np.sum((yt == 1) & (yp == 1))
        self.fp += np.sum((yt == 0) & (yp == 1))
        self.fn += np.sum((yt == 1) & (yp == 0))

        self.tp_l += ((yt == 1) & (yp == 1)).astype(int)
        self.fp_l += ((yt == 0) & (yp == 1)).astype(int)
        self.fn_l += ((yt == 1) & (yp == 0)).astype(int)

        # Example-based custommetrics
        inter = np.sum((yt == 1) & (yp == 1))
        pred_sum = np.sum(yp)
        true_sum = np.sum(yt)
        if pred_sum > 0:
            prec = inter / pred_sum
        else:
            prec = 0.0
        if true_sum > 0:
            rec = inter / true_sum
        else:
            rec = 0.0
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        self.sum_example_prec += prec
        self.sum_example_rec += rec
        self.sum_example_f1 += f1

    def results(self):
        if self.total_samples == 0:
            return {}

        micro_prec = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        micro_rec = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        if micro_prec + micro_rec > 0:
            micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)
        else:
            micro_f1 = 0.0

        # Macro average (mean of per-label)
        prec_l = np.divide(self.tp_l, (self.tp_l + self.fp_l), out=np.zeros_like(self.tp_l, dtype=float), where=(self.tp_l + self.fp_l) > 0)
        rec_l = np.divide(self.tp_l, (self.tp_l + self.fn_l), out=np.zeros_like(self.tp_l, dtype=float), where=(self.tp_l + self.fn_l) > 0)
        f1_l = np.divide(2 * prec_l * rec_l, (prec_l + rec_l), out=np.zeros_like(prec_l, dtype=float), where=(prec_l + rec_l) > 0)
        macro_prec = float(np.mean(prec_l))
        macro_rec = float(np.mean(rec_l))
        macro_f1 = float(np.mean(f1_l))

        # Example-based averages
        example_prec = self.sum_example_prec / self.total_samples
        example_rec = self.sum_example_rec / self.total_samples
        example_f1 = self.sum_example_f1 / self.total_samples

        # Accuracy / hamming
        hamming_loss = 1 - (self.correct_bits / self.total_bits)
        accuracy = self.correct_bits / self.total_bits
        subset_acc = self.subset_correct / self.total_samples

        # Perfect rate
        perfect_rate = (
            (self.perfect_predictions / self.total_positive_cases) * 100.0
            if self.total_positive_cases > 0 else 0.0
        )

        return {
            "total_samples": self.total_samples,
            "accuracy": accuracy,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "subset_acc": subset_acc,
            "hamming_loss": hamming_loss,
            "example_f1": example_f1,
            "example_prec": example_prec,
            "example_rec": example_rec,
            "micro_prec": micro_prec,
            "micro_rec": micro_rec,
            "macro_prec": macro_prec,
            "macro_rec": macro_rec,
            "perfect_overall": self.subset_correct,
            "total_positive_cases": self.total_positive_cases,
            "perfect_predictions": self.perfect_predictions,
            "perfect_rate": perfect_rate,
        }

    def pretty_print(self):
        r = self.results()
        print("\n" + "=" * 60)
        print("RESULTADOS FINALES (Incremental):")
        print("=" * 60)
        print(f"Total de muestras procesadas: {r['total_samples']}")
        print(f"  Accuracy:  {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%)")
        print(f"  MicroF1:   {r['micro_f1']:.4f}")
        print(f"  MacroF1:   {r['macro_f1']:.4f}")
        print(f"  SubsetAcc: {r['subset_acc']:.4f}")
        print(f"  Hamm loss: {r['hamming_loss']:.4f}")
        print(f"  Examp F1:  {r['example_f1']:.4f}")
        print(f"  Examp prec:{r['example_prec']:.4f}")
        print(f"  Examp rec: {r['example_rec']:.4f}")
        print(f"  Micro prec:{r['micro_prec']:.4f}")
        print(f"  Micro rec: {r['micro_rec']:.4f}")
        print(f"  Macro prec:{r['macro_prec']:.4f}")
        print(f"  Macro rec: {r['macro_rec']:.4f}")
        print(f"\nPredicciones Perfectas (casos con fallos):")
        print(f"  Total casos con fallos: {r['total_positive_cases']}")
        print(f"  Predicciones perfectas: {r['perfect_predictions']}")
        print(f"  Tasa de acierto en fallos: {r['perfect_rate']:.2f}%")
        print(f"\nPredicciones perfectas totales (todas las muestras): {r['perfect_overall']}")
        print("=" * 60)
