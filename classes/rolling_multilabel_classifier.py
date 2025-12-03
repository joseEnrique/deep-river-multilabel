from typing import Callable, Dict, List, Type, Union

import torch
import torch.nn.functional as F
from sortedcontainers import SortedSet

from deep_river.base import RollingDeepEstimator
from deep_river.utils.tensor_conversion import deque2rolling_tensor
from river import base as river_base


class RollingMultiLabelClassifier(RollingDeepEstimator, river_base.MultiLabelClassifier):
    """
    Multi-label rolling window wrapper for a single PyTorch module.

    What it does
    ------------
    - Maintains a fixed-size sliding window of the most recent examples.
    - Wraps a single module that outputs a logit per label (e.g., 5 outputs for 5 labels).
    - Trains online using Binary Cross-Entropy with logits over the current window.
    - Produces per-label probabilities via sigmoid and thresholds them into 0/1 decisions.

    Prediction
    ----------
    - `predict_proba_one(x)` returns a dict {label: probability} using the window
      that includes `x` (and optionally appends it if `append_predict=True`).
    - `predict_one(x)` applies per-label thresholds (default 0.5) on those probabilities.
    - If the window is not yet full, returns zeros to avoid using insufficient context.

    Learning
    --------
    - `learn_one(x, y)` appends the current features to the rolling window.
    - When the window reaches `window_size`, runs one optimization step with
      BCE-with-logits on the module outputs vs. the multi-hot targets.

    Requirements & Notes
    --------------------
    - The wrapped `module` must accept `input_dim` and `output_dim` at construction.
      Here, `output_dim` must equal `len(label_names)`.
    - Inputs are provided as dicts; unseen features are filled with 0.
    - Set `append_predict=True` if you want predictions to also extend the window.
    - Use custom thresholds if your labels are imbalanced (e.g., lower than 0.5).

    Parameters
    ----------
    module : Type[torch.nn.Module]
        Torch Module class. Must accept `input_dim` and `output_dim` in the constructor.
    label_names : List[str]
        Ordered list of label keys. The module's `output_dim` should equal len(label_names).
    optimizer_fn : Union[str, Type[torch.optim.Optimizer]]
        Optimizer class or alias (e.g., "adam").
    lr : float
        Learning rate.
    device : str
        Device string ("cpu" or "cuda:0").
    seed : int
        Random seed.
    window_size : int
        Rolling window size.
    append_predict : bool
        If True, appends inputs used for prediction to the rolling window.
    thresholds : Dict[str, float] | None
        Optional decision thresholds per label; default 0.5.
    **kwargs
        Extra kwargs forwarded to the module constructor (e.g., hidden sizes, dropout).

    Example
    -------
    >>> from model import LSTM_MultiLabel
    >>> labels = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    >>> clf = RollingMultiLabelClassifier(
    ...     module=LSTM_MultiLabel,
    ...     label_names=labels,
    ...     window_size=50,
    ...     device="cuda:0",
    ...     lr=1e-3,
    ...     hidden_dim=128,
    ...     num_layers=2,
    ...     bidirectional=True,
    ...     dropout=0.3,
    ...     output_dim=len(labels),
    ... )
    >>> x = {"Torque [Nm]": 35.0, "Type_H": 0.0}
    >>> y = {"TWF": 0, "HDF": 1, "PWF": 0, "OSF": 0, "RNF": 0}
    >>> _ = clf.learn_one(x, y)
    >>> preds = clf.predict_one(x)
    """

    def __init__(
        self,
        module: Type[torch.nn.Module],
        label_names: List[str],
        optimizer_fn: Union[str, Type[torch.optim.Optimizer]] = "adam",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        window_size: int = 100,
        append_predict: bool = False,
        thresholds: Dict[str, float] | None = None,
        epochs: int = 1,
        pos_weight: float | None = None,
        loss_fn: Callable | None = None,
        **kwargs,
    ):
        # Store module class and params for deferred initialization
        self.module_cls = module
        self.module_kwargs = kwargs

        # Multi-label specific attributes
        self.label_names = label_names
        self.thresholds = thresholds or {t: 0.5 for t in label_names}
        self.epochs = max(1, int(epochs))
        self.pos_weight = pos_weight
        # Save the user-provided loss_fn before calling RollingDeepEstimator.__init__
        user_loss_fn = loss_fn

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

        # Use RollingDeepEstimator init to build window + base functionality
        RollingDeepEstimator.__init__(
            self,
            module=temp_module,
            loss_fn="binary_cross_entropy_with_logits",
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
            window_size=window_size,
            append_predict=append_predict,
        )
        
        # Always restore the user-provided loss_fn (None or custom function)
        # This prevents the string "binary_cross_entropy_with_logits" from RollingDeepEstimator
        # from being used as self.loss_fn
        self.loss_fn = user_loss_fn

        # Add target buffer to store targets for each timestep
        from collections import deque
        self._y_window = deque(maxlen=window_size)

        # Mark that we need to re-initialize with correct input_dim
        self.module_initialized = False

    def _adapt_input_dim(self, x: Dict):
        """Check if new features appear and update observed_features if needed."""
        self._update_observed_features(x)

    def initialize_module(self, x: Dict, **kwargs):
        """Initialize the actual model with the correct input dimension."""
        import gc

        # Clean up previous module if it exists (for reinitialization)
        if hasattr(self, 'module') and self.module is not None and self.module_initialized:
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
        n_features = len(self.observed_features)  # Use observed_features for consistent counting

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

    def _to_window_tensor(self, x_window) -> torch.Tensor:
        # Each timestep becomes a separate batch item with seq_len=1
        # Shape: [batch=len(window), seq_len=1, n_features]
        batch_list = []
        for item in x_window:
            # item is a list of features
            # Add sequence dimension: [[features]]
            batch_list.append([item])
        
        x_t = torch.tensor(batch_list, dtype=torch.float32, device=self.device)
        return x_t

    def learn_one(self, x: Dict, y: Dict, **kwargs) -> None:
        # Initialize module if needed
        if not self.module_initialized:
            self._update_observed_features(x)
            self.initialize_module(x=x, **self.module_kwargs)

        # Check if new features appeared and reinitialize if needed
        prev_num_features = len(self.observed_features)
        self._adapt_input_dim(x)
        if len(self.observed_features) > prev_num_features:
            # New features detected - need to reinitialize module with new size
            print(f"⚠ Warning: New features detected ({prev_num_features} -> {len(self.observed_features)}). Reinitializing model...")
            self.initialize_module(x=x, **self.module_kwargs)
            # Clear the window since the feature dimensions have changed
            self._x_window.clear()
            self._y_window.clear()

        # Append to window
        self._x_window.append([x.get(feature, 0) for feature in self.observed_features])
        # Append target to target window
        self._y_window.append([float(y.get(t, 0)) for t in self.label_names])

        # Train at every step using the current window (even if not full yet)
        if len(self._x_window) > 0:
            X_t = self._to_window_tensor(self._x_window)
            # Target tensor: one target per batch item (timestep in window)
            # Shape: [batch=len(window), n_labels]
            y_vec = torch.tensor(
                list(self._y_window),
                dtype=torch.float32,
                device=self.device,
            )

            # Train for configured number of epochs per online sample/window
            for _ in range(self.epochs):
                self.module.train()
                self.optimizer.zero_grad()
                logits = self.module(X_t)

                # Aplicar loss_fn personalizada si existe
                if self.loss_fn is not None:
                    loss = self.loss_fn(logits, y_vec)
                # Aplicar pos_weight si está configurado
                elif self.pos_weight is not None:
                    # Crear tensor de pos_weight para todas las etiquetas
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
        # Initialize if needed (for observed_features and module)
        if not self.module_initialized:
            self._update_observed_features(x)
            self.initialize_module(x=x, **self.module_kwargs)

        # Check if new features appeared and reinitialize if needed
        prev_num_features = len(self.observed_features)
        self._adapt_input_dim(x)
        if len(self.observed_features) > prev_num_features:
            # New features detected - need to reinitialize module with new size
            print(f"⚠ Warning: New features detected in predict ({prev_num_features} -> {len(self.observed_features)}). Reinitializing model...")
            self.initialize_module(x=x, **self.module_kwargs)
            # Clear the window since the feature dimensions have changed
            self._x_window.clear()
            self._y_window.clear()

        # Prepare window copy including current x
        x_win = self._x_window.copy()
        x_win.append([x.get(feature, 0) for feature in self.observed_features])
        if self.append_predict:
            self._x_window = x_win

        if len(x_win) < self.window_size:
            # Not enough context; return zeros
            return {t: 0.0 for t in self.label_names}

        X_t = self._to_window_tensor(x_win)
        self.module.eval()
        with torch.inference_mode():
            logits = self.module(X_t)  # Shape: [batch=len(window), n_labels]
            # We only care about the prediction for the LAST timestep (current x)
            # Take the last batch item [-1]
            probs = torch.sigmoid(logits[-1]).detach().cpu().tolist()

        return {t: float(probs[i]) for i, t in enumerate(self.label_names)}

    def predict_one(self, x: Dict) -> Dict[str, int]:
        probas = self.predict_proba_one(x)
        it_is = {t: int(probas[t] >= self.thresholds.get(t, 0.5)) for t in self.label_names}
        return it_is


