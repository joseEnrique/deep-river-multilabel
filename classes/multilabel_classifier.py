from typing import Callable, Dict, List, Type, Union

import torch
import torch.nn.functional as F
from sortedcontainers import SortedSet

from deep_river.base import DeepEstimator
from river import base as river_base


class MultiLabelClassifier(DeepEstimator, river_base.MultiLabelClassifier):
    """
    Multi-label classifier for PyTorch modules using online learning.

    This class wraps a PyTorch module for multi-label classification where each
    instance can have multiple binary labels. Unlike multi-class classification,
    each label is treated as an independent binary classification task.

    What it does
    ------------
    - Wraps a single module that outputs a logit per label (e.g., 5 outputs for 5 labels)
    - Trains online using Binary Cross-Entropy with logits
    - Produces per-label probabilities via sigmoid and thresholds them into 0/1 decisions
    - Supports incremental learning on streaming data

    Key Features
    ------------
    - **Multi-label Support**: Handle multiple binary classification tasks simultaneously
    - **Online Learning**: Update model incrementally with each new example
    - **Flexible Thresholds**: Customize decision thresholds per label
    - **Class Imbalance**: Support for pos_weight to handle imbalanced labels

    Prediction
    ----------
    - `predict_proba_one(x)` returns a dict {label: probability} for each label
    - `predict_one(x)` applies per-label thresholds (default 0.5) on those probabilities
    - Probabilities are independent (don't sum to 1)

    Learning
    --------
    - `learn_one(x, y)` updates the model with one example
    - Uses BCE-with-logits loss for training
    - Optional pos_weight parameter to handle class imbalance

    Requirements & Notes
    --------------------
    - The wrapped `module` must accept `input_dim` and `output_dim` at construction
    - `output_dim` must equal `len(label_names)`
    - Inputs are provided as dicts; unseen features are filled with 0
    - Use custom thresholds if your labels are imbalanced (e.g., lower than 0.5)

    Parameters
    ----------
    module : Type[torch.nn.Module]
        Torch Module class. Must accept `input_dim` and `output_dim` in the constructor.
    label_names : List[str]
        Ordered list of label keys. The module's `output_dim` should equal len(label_names).
    optimizer_fn : Union[str, Type[torch.optim.Optimizer]]
        Optimizer class or alias (e.g., "adam", "sgd").
    lr : float
        Learning rate.
    device : str
        Device string ("cpu" or "cuda:0").
    seed : int
        Random seed.
    thresholds : Dict[str, float] | None
        Optional decision thresholds per label; default 0.5.
    pos_weight : float | None
        Positive class weight for handling class imbalance. Applied to all labels.
    **kwargs
        Extra kwargs forwarded to the module constructor (e.g., hidden sizes, dropout).

    Example
    -------
    >>> import torch.nn as nn
    >>> from classes.multilabel_classifier import MultiLabelClassifier
    >>>
    >>> # Define a simple feedforward network
    >>> class SimpleNet(nn.Module):
    ...     def __init__(self, input_dim, output_dim, hidden_dim=64):
    ...         super().__init__()
    ...         self.fc1 = nn.Linear(input_dim, hidden_dim)
    ...         self.fc2 = nn.Linear(hidden_dim, output_dim)
    ...     def forward(self, x):
    ...         x = torch.relu(self.fc1(x))
    ...         return self.fc2(x)  # logits
    >>>
    >>> labels = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    >>> clf = MultiLabelClassifier(
    ...     module=SimpleNet,
    ...     label_names=labels,
    ...     optimizer_fn="adam",
    ...     lr=1e-3,
    ...     device="cpu",
    ...     hidden_dim=128,
    ... )
    >>>
    >>> # Learn from streaming data
    >>> x = {"Torque [Nm]": 35.0, "Type_H": 0.0, "Speed": 1500.0}
    >>> y = {"TWF": 0, "HDF": 1, "PWF": 0, "OSF": 0, "RNF": 0}
    >>> clf.learn_one(x, y)
    >>>
    >>> # Make predictions
    >>> probs = clf.predict_proba_one(x)
    >>> preds = clf.predict_one(x)
    """

    def __init__(
        self,
        module: Type[torch.nn.Module],
        label_names: List[str],
        optimizer_fn: Union[str, Type[torch.optim.Optimizer]] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        thresholds: Dict[str, float] | None = None,
        pos_weight: float | None = None,
        **kwargs,
    ):
        # Store module class and params for deferred initialization
        self.module_cls = module
        self.module_kwargs = kwargs

        # Multi-label specific attributes
        self.label_names = label_names
        self.thresholds = thresholds or {t: 0.5 for t in label_names}
        self.pos_weight = pos_weight

        # Instantiate module with temporary input_dim=1
        # Will be re-initialized when first data arrives
        torch.manual_seed(seed)
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"n_features", "input_dim", "output_dim"}
        }
        temp_module = module(
            input_dim=1,
            output_dim=len(label_names),
            **filtered_kwargs,
        )

        # Use DeepEstimator init to build base functionality
        DeepEstimator.__init__(
            self,
            module=temp_module,
            loss_fn="binary_cross_entropy_with_logits",
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
        )

        # Note: _supervised property is inherited from river.base.Estimator and returns True by default

        # Mark that we need to re-initialize with correct input_dim
        self.module_initialized = False

    def initialize_module(self, x: Dict, **kwargs):
        """Initialize the actual model with the correct input dimension."""
        import gc

        # Clean up previous module if it exists
        if hasattr(self, 'module') and self.module is not None:
            # Delete optimizer first
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                del self.optimizer
            # Move module to CPU and delete
            self.module.cpu()
            del self.module
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.manual_seed(self.seed)
        n_features = len(x)

        # Instantiate the real module
        if not isinstance(self.module_cls, torch.nn.Module):
            # Filter out parameters that shouldn't be passed
            filtered_kwargs = {
                k: v
                for k, v in self.module_kwargs.items()
                if k not in {"n_features", "input_dim", "output_dim"}
            }
            self.module = self.module_cls(
                input_dim=n_features,
                output_dim=len(self.label_names),
                **filtered_kwargs,
            )
        else:
            self.module = self.module_cls

        self.module.to(self.device)

        # Rebuild optimizer with the real module's parameters
        from deep_river.utils import get_optim_fn
        optimizer_func = get_optim_fn(self.optimizer_fn)
        self.optimizer = optimizer_func(self.module.parameters(), lr=self.lr)

        self.module_initialized = True

    def learn_one(self, x: Dict, y: Dict, **kwargs) -> None:
        """Learn from one example.

        Parameters
        ----------
        x : Dict
            Feature dictionary with feature names as keys and values as floats.
        y : Dict
            Label dictionary with label names as keys and binary values (0/1) as values.
        """
        # Initialize module if needed
        if not self.module_initialized:
            self._update_observed_features(x)
            self.initialize_module(x=x, **self.module_kwargs)

        # Check if new features appeared - if so, reinitialize
        prev_num_features = len(self.observed_features)
        self._update_observed_features(x)
        if len(self.observed_features) > prev_num_features:
            # New features detected - need to reinitialize module with new size
            self.initialize_module(x=x, **self.module_kwargs)

        # Convert input dict to tensor
        x_t = self._dict2tensor(x)

        # Target tensor [1, num_labels]
        y_vec = torch.tensor(
            [[float(y.get(label, 0)) for label in self.label_names]],
            dtype=torch.float32,
            device=self.device,
        )

        # Training step
        self.module.train()
        self.optimizer.zero_grad()
        logits = self.module(x_t)

        # Apply pos_weight if configured
        if self.pos_weight is not None:
            pos_weight_tensor = torch.full(
                (len(self.label_names),),
                self.pos_weight,
                device=self.device
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, y_vec, pos_weight=pos_weight_tensor
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y_vec)

        loss.backward()
        self.optimizer.step()

    def predict_proba_one(self, x: Dict) -> Dict[str, float]:
        """Predict probabilities for each label.

        Parameters
        ----------
        x : Dict
            Feature dictionary with feature names as keys and values as floats.

        Returns
        -------
        Dict[str, float]
            Dictionary with label names as keys and probabilities as values.
        """
        # Initialize if needed
        if not self.module_initialized:
            self._update_observed_features(x)
            self.initialize_module(x=x, **self.module_kwargs)

        # Check if new features appeared - if so, reinitialize
        prev_num_features = len(self.observed_features)
        self._update_observed_features(x)
        if len(self.observed_features) > prev_num_features:
            # New features detected - need to reinitialize module with new size
            self.initialize_module(x=x, **self.module_kwargs)

        # Convert input to tensor
        x_t = self._dict2tensor(x)

        # Predict
        self.module.eval()
        with torch.inference_mode():
            logits = self.module(x_t)
            probs = torch.sigmoid(logits).squeeze(0).detach().cpu().tolist()

        return {label: float(probs[i]) for i, label in enumerate(self.label_names)}

    def predict_one(self, x: Dict) -> Dict[str, int]:
        """Predict binary labels for each label.

        Parameters
        ----------
        x : Dict
            Feature dictionary with feature names as keys and values as floats.

        Returns
        -------
        Dict[str, int]
            Dictionary with label names as keys and binary predictions (0/1) as values.
        """
        probas = self.predict_proba_one(x)
        return {label: int(probas[label] >= self.thresholds.get(label, 0.5))
                for label in self.label_names}
