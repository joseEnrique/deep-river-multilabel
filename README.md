# Rolling Multi-Label Classifiers for Online Learning

This repository implements two online multi-label classifiers built on top of [River](https://riverml.xyz/) and [PyTorch](https://pytorch.org/), designed for streaming data scenarios such as predictive maintenance.

---

## Classes

### `RollingMultiLabelClassifier`

**File:** `classes/rolling_multilabel_classifier.py`

An online multi-label classifier that uses an LSTM network with **Experience Replay** via a rolling window of fixed size. At each step, it predicts the current sample and then trains on the entire window (a batch of the last `window_size` samples), each treated as an independent sequence of length 1.

**Architecture:** LSTM operating on batches of shape `[window_size, 1, features]` — effectively an MLP with replay, where the LSTM's temporal memory is not exploited.

**Best suited for:** datasets where consecutive samples are not temporally related (e.g., interleaved records from different machines/entities).

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `module` | `Type[nn.Module]` | — | PyTorch module class (e.g. `LSTM_MultiLabel`) |
| `label_names` | `List[str]` | — | Names of the output labels |
| `optimizer_fn` | `str \| Optimizer` | `"adam"` | Optimizer |
| `lr` | `float` | `1e-3` | Learning rate |
| `device` | `str` | `"cpu"` | `"cpu"` or `"cuda"` |
| `seed` | `int` | `42` | Random seed |
| `window_size` | `int` | `100` | Number of samples kept in the replay buffer |
| `append_predict` | `bool` | `False` | Whether to include the prediction step's sample in the window |
| `thresholds` | `Dict[str, float]` | `0.5` for all | Per-label decision thresholds |
| `epochs` | `int` | `1` | Number of gradient steps per sample |
| `loss_fn` | `Callable` | `None` | Custom loss function (e.g. `FullAdaptiveFocalLoss()`) |
| `**kwargs` | | | Extra arguments forwarded to the module (e.g. `hidden_dim`, `num_layers`, `dropout`) |

#### Example

```python
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier
from testclassifier.model import LSTM_MultiLabel, FullAdaptiveFocalLoss

clf = RollingMultiLabelClassifier(
    module=LSTM_MultiLabel,
    label_names=['TWF', 'HDF', 'PWF', 'OSF', 'RNF'],
    optimizer_fn="adam",
    lr=1e-3,
    device="cuda",
    window_size=200,
    hidden_dim=256,
    num_layers=3,
    dropout=0.3,
    output_dim=5,
    seed=42,
    epochs=1,
    loss_fn=FullAdaptiveFocalLoss()
)
```

---

### `RollingMultiLabelClassifierSequences`

**File:** `classes/rolling_multilabel_classifier_sequences.py`

An online multi-label classifier that extends the rolling window approach to support **true temporal sequences**. Instead of treating each sample independently, it builds sequences of length `past_history` from consecutive samples and trains the LSTM on them.

**Architecture:** LSTM operating on batches of shape `[window_size, past_history, features]`, where each item in the batch is a genuine sequence of the last `past_history` observations.

**Best suited for:** datasets where consecutive samples come from the same entity and exhibit temporal dependencies (sensor streams, time-series from a single machine).

> **Note:** With `past_history=1`, this class is mathematically equivalent to `RollingMultiLabelClassifier` and produces identical metrics (within GPU floating-point noise).

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `module` | `Type[nn.Module]` | — | PyTorch module class (e.g. `LSTM_MultiLabel`) |
| `label_names` | `List[str]` | — | Names of the output labels |
| `window_size` | `int` | — | Number of sequences kept in the replay buffer |
| `past_history` | `int` | `1` | Length of each input sequence (temporal context) |
| `optimizer_fn` | `str \| Optimizer` | `"adam"` | Optimizer |
| `lr` | `float` | `1e-3` | Learning rate |
| `device` | `str` | `"cuda"` | `"cpu"` or `"cuda"` |
| `seed` | `int` | `42` | Random seed |
| `epochs` | `int` | `1` | Number of gradient steps per sample |
| `threshold` | `float` | `0.5` | Global decision threshold for all labels |
| `loss_fn` | `Callable` | `None` | Custom loss function (e.g. `FullAdaptiveFocalLoss()`) |
| `**kwargs` | | | Extra arguments forwarded to the module (e.g. `hidden_dim`, `num_layers`, `dropout`) |

#### Example

```python
from classes.rolling_multilabel_classifier_sequences import RollingMultiLabelClassifierSequences
from testclassifier.model import LSTM_MultiLabel, FullAdaptiveFocalLoss

clf = RollingMultiLabelClassifierSequences(
    module=LSTM_MultiLabel,
    label_names=['TWF', 'HDF', 'PWF', 'OSF', 'RNF'],
    window_size=200,
    past_history=5,          # uses last 5 samples as context
    optimizer_fn="adam",
    lr=1e-3,
    device="cuda",
    hidden_dim=256,
    num_layers=3,
    dropout=0.3,
    output_dim=5,
    seed=42,
    epochs=1,
    loss_fn=FullAdaptiveFocalLoss()
)
```

---

## Comparison

| Feature | `RollingMultiLabelClassifier` | `RollingMultiLabelClassifierSequences` |
|---|---|---|
| Temporal context | ❌ None (seq_len=1) | ✅ `past_history` steps |
| Training batch shape | `[W, 1, F]` | `[W, past_history, F]` |
| River compatible | ✅ | ✅ |
| Best for IID streams | ✅ | ✅ (with `past_history=1`) |
| Best for temporal streams | ❌ | ✅ |

---

## Evaluation Metrics

Custom River-compatible multi-label metrics are provided in `metrics/multilabel.py`:

| Metric | Class | Formula |
|---|---|---|
| Hamming Loss | `HammingLoss` | `(1/nL) Σ 1[yᵢₗ ≠ ẑᵢₗ]` |
| Example-based F1 | `ExampleF1` | `(1/n) Σ 2TP / (2TP+FP+FN)` |
| Example-based Precision | `ExamplePrecision` | `(1/n) Σ TP / (TP+FP)` |
| Example-based Recall | `ExampleRecall` | `(1/n) Σ TP / (TP+FN)` |

All follow the convention that a sample with no active labels predicted correctly contributes `1.0` (perfect prediction), consistent with scikit-learn.

Use alongside River's built-in `MicroAverage(F1())`, `MacroAverage(F1())`, `ExactMatch()`, etc.

### Usage

```python
from river.metrics import F1, Precision, Recall
from river.metrics.base import Metrics
from river.metrics.multioutput import ExactMatch, MicroAverage, MacroAverage
from metrics import HammingLoss, ExampleF1, ExamplePrecision, ExampleRecall

all_metrics = Metrics([
    ExactMatch(),
    HammingLoss(),
    ExampleF1(), ExamplePrecision(), ExampleRecall(),
    MicroAverage(F1()), MicroAverage(Precision()), MicroAverage(Recall()),
    MacroAverage(F1()), MacroAverage(Precision()), MacroAverage(Recall()),
])
```

---

## Running the Comparison Test

```bash
python test_forecaster_comp.py
```

This script evaluates `RollingMultiLabelClassifier` and `RollingMultiLabelClassifierSequences` (with different `past_history` values) on the Ai4i dataset using all 11 metrics from the paper table, and prints a comparative summary table.
