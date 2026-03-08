"""
Custom multi-label metrics compatible with River's Metrics system.

These implement the metrics from the paper table:
  - HammingLoss
  - ExampleF1, ExamplePrecision, ExampleRecall (sample-averaged)

All classes extend river.metrics.base.MultiOutputClassificationMetric
so they plug directly into river's Metrics([...]) container and 
evaluate.progressive_val_score.
"""

from __future__ import annotations
from river.metrics.base import Metric


class HammingLoss(Metric):
    """
    Hamming Loss for multi-label classification.

    The fraction of labels that are incorrectly predicted,
    averaged across all samples and labels.

    Lower is better (0 = perfect).
    """

    def __init__(self):
        self._n_errors = 0
        self._n_total = 0

    def update(self, y_true: dict, y_pred: dict):
        for label in y_true:
            t = int(y_true.get(label, 0))
            p = int(y_pred.get(label, 0))
            if t != p:
                self._n_errors += 1
            self._n_total += 1
        return self

    def revert(self, y_true: dict, y_pred: dict):
        for label in y_true:
            t = int(y_true.get(label, 0))
            p = int(y_pred.get(label, 0))
            if t != p:
                self._n_errors -= 1
            self._n_total -= 1
        return self

    def get(self) -> float:
        return self._n_errors / self._n_total if self._n_total > 0 else 0.0

    @property
    def bigger_is_better(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"HammingLoss: {self.get() * 100:.2f}%"

    def works_with(self, model) -> bool:
        return True


class _SampleAveragedBinaryMetric(Metric):
    """
    Base class for example-based (sample-averaged) metrics.

    For each sample, computes a per-sample value based on TP, FP, FN,
    then averages across all samples.
    """

    def __init__(self):
        self._sum = 0.0
        self._n = 0

    def _per_sample(self, tp: int, fp: int, fn: int) -> float:
        raise NotImplementedError

    def update(self, y_true: dict, y_pred: dict):
        tp = sum(
            1 for l in y_true
            if int(y_true.get(l, 0)) == 1 and int(y_pred.get(l, 0)) == 1
        )
        fp = sum(
            1 for l in y_true
            if int(y_true.get(l, 0)) == 0 and int(y_pred.get(l, 0)) == 1
        )
        fn = sum(
            1 for l in y_true
            if int(y_true.get(l, 0)) == 1 and int(y_pred.get(l, 0)) == 0
        )
        self._sum += self._per_sample(tp, fp, fn)
        self._n += 1
        return self

    def revert(self, y_true: dict, y_pred: dict):
        tp = sum(
            1 for l in y_true
            if int(y_true.get(l, 0)) == 1 and int(y_pred.get(l, 0)) == 1
        )
        fp = sum(
            1 for l in y_true
            if int(y_true.get(l, 0)) == 0 and int(y_pred.get(l, 0)) == 1
        )
        fn = sum(
            1 for l in y_true
            if int(y_true.get(l, 0)) == 1 and int(y_pred.get(l, 0)) == 0
        )
        self._sum -= self._per_sample(tp, fp, fn)
        self._n -= 1
        return self

    def get(self) -> float:
        return self._sum / self._n if self._n > 0 else 0.0

    @property
    def bigger_is_better(self) -> bool:
        return True

    def works_with(self, model) -> bool:
        return True


class ExamplePrecision(_SampleAveragedBinaryMetric):
    """Sample-averaged (example-based) Precision for multi-label classification."""

    def _per_sample(self, tp: int, fp: int, fn: int) -> float:
        denom = tp + fp
        if denom == 0:
            # Both y_true and y_pred have no positive labels → perfect prediction
            return 1.0 if fn == 0 else 0.0
        return tp / denom

    def __repr__(self) -> str:
        return f"ExamplePrecision: {self.get() * 100:.2f}%"


class ExampleRecall(_SampleAveragedBinaryMetric):
    """Sample-averaged (example-based) Recall for multi-label classification."""

    def _per_sample(self, tp: int, fp: int, fn: int) -> float:
        denom = tp + fn
        if denom == 0:
            # Both y_true and y_pred have no positive labels → perfect prediction
            return 1.0 if fp == 0 else 0.0
        return tp / denom

    def __repr__(self) -> str:
        return f"ExampleRecall: {self.get() * 100:.2f}%"


class ExampleF1(_SampleAveragedBinaryMetric):
    """Sample-averaged (example-based) F1 for multi-label classification."""

    def _per_sample(self, tp: int, fp: int, fn: int) -> float:
        denom = 2 * tp + fp + fn
        if denom == 0:
            # Both y_true and y_pred have no positive labels → perfect prediction
            return 1.0
        return (2 * tp) / denom

    def __repr__(self) -> str:
        return f"ExampleF1: {self.get() * 100:.2f}%"
