from __future__ import annotations

from river import metrics, utils
from river.metrics.multioutput.base import MultiOutputMetric

__all__ = ["MicroAverage"]


class MicroAverage(MultiOutputMetric, metrics.base.WrapperMetric):
    """Micro-average wrapper.

    The provided metric is updated with the value of each output.

    Parameters
    ----------
    metric
        A classification or a regression metric.

    """

    def __init__(self, metric):
        self._metric = metric

    @property
    def metric(self):
        return self._metric

    @property
    def requires_labels(self):
        """Inherit requires_labels from the wrapped metric."""
        return getattr(self._metric, 'requires_labels', True)

    def works_with(self, model) -> bool:
        # Check if it's a multi-label/multi-output classifier
        # This includes river's multi-output classifiers and custom implementations
        if isinstance(self.metric, metrics.base.ClassificationMetric):
            # Check for river's multi-output classifier or custom multi-label classifiers
            has_predict_one = hasattr(model, 'predict_one')
            has_learn_one = hasattr(model, 'learn_one')
            return has_predict_one and has_learn_one
        return utils.inspect.ismoregressor(model)

    def update(self, y_true, y_pred, sample_weight=1.0):
        for i in y_true:
            self.metric.update(y_true[i], y_pred[i], sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        for i in y_true:
            self.metric.revert(y_true[i], y_pred[i], sample_weight)
        return self

    def get(self):
        return self.metric.get()
