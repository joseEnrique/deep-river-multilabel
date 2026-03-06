from typing import Callable, Dict, List, Type, Union
import pickle
import torch
import torch.nn.functional as F
from sortedcontainers import SortedSet

from deep_river.base import RollingDeepEstimator
from deep_river.utils.tensor_conversion import deque2rolling_tensor
from river import base as river_base

#NO UPDATED

class PretrainedRollingMultiLabelClassifier(RollingDeepEstimator, river_base.MultiLabelClassifier):
    """
    Multi-label rolling window classifier with pretrained model loading capability.

    This class extends RollingMultiLabelClassifier to support loading pretrained models
    saved from batch training, allowing you to bootstrap online learning with a
    pre-trained model.

    What it does
    ------------
    - Loads a pretrained PyTorch model from a checkpoint file
    - Maintains a fixed-size sliding window of the most recent examples
    - Supports online learning (fine-tuning) on streaming data
    - Produces per-label probabilities via sigmoid and thresholds them into 0/1 decisions

    Key Features
    ------------
    - **Pretrained Loading**: Initialize with weights from batch-trained models
    - **Online Fine-tuning**: Continue training on streaming data
    - **Multi-label Support**: Handle multiple binary classification tasks simultaneously
    - **Flexible Thresholds**: Customize decision thresholds per label

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
    - Set `freeze_pretrained=True` to prevent updating pretrained weights during online learning.

    Parameters
    ----------
    module : Type[torch.nn.Module]
        Torch Module class. Must accept `input_dim` and `output_dim` in the constructor.
    label_names : List[str]
        Ordered list of label keys. The module's `output_dim` should equal len(label_names).
    checkpoint_path : str
        Path to the pretrained model checkpoint (.pt file).
    scaler_path : str | None
        Optional path to the scaler pickle file for feature normalization.
    optimizer_fn : Union[str, Type[torch.optim.Optimizer]]
        Optimizer class or alias (e.g., "adam").
    lr : float
        Learning rate for online learning.
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
    freeze_pretrained : bool
        If True, freezes pretrained weights and only trains new/adapted layers.
    load_optimizer_state : bool
        If True, loads the optimizer state from checkpoint (useful for resuming training).
    epochs : int
        Number of training epochs per online sample/window update.
    **kwargs
        Extra kwargs forwarded to the module constructor (e.g., hidden sizes, dropout).

    Example
    -------
    >>> from deep_river.classification import PretrainedRollingMultiLabelClassifier
    >>> from testbatch.load_and_use_model import LSTM_MultiLabel
    >>>
    >>> labels = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    >>> clf = PretrainedRollingMultiLabelClassifier(
    ...     module=LSTM_MultiLabel,
    ...     label_names=labels,
    ...     checkpoint_path="lstm_multilabel_ai4i_complete.pt",
    ...     scaler_path="scaler_ai4i.pkl",
    ...     window_size=10,
    ...     device="cuda:0",
    ...     lr=1e-4,  # Lower learning rate for fine-tuning
    ...     freeze_pretrained=False,  # Allow fine-tuning
    ... )
    >>>
    >>> # Use for prediction (with pretrained weights)
    >>> x = {"Air temperature [K]": 300.0, "Torque [Nm]": 35.0, ...}
    >>> probs = clf.predict_proba_one(x)
    >>>
    >>> # Continue learning on new data
    >>> y = {"TWF": 0, "HDF": 1, "PWF": 0, "OSF": 0, "RNF": 0}
    >>> clf.learn_one(x, y)
    """

    def __init__(
        self,
        module: Type[torch.nn.Module],
        label_names: List[str],
        checkpoint_path: str,
        scaler_path: str | None = None,
        optimizer_fn: Union[str, Type[torch.optim.Optimizer]] = "adam",
        lr: float = 1e-4,  # Lower default LR for fine-tuning
        device: str = "cpu",
        seed: int = 42,
        window_size: int = 100,
        append_predict: bool = False,
        thresholds: Dict[str, float] | None = None,
        freeze_pretrained: bool = False,
        load_optimizer_state: bool = False,
        epochs: int = 1,
        **kwargs,
    ):
        # Store module class and params for deferred initialization
        self.module_cls = module
        self.module_kwargs = kwargs

        # Pretrained-specific parameters
        self.checkpoint_path = checkpoint_path
        self.scaler_path = scaler_path
        self.freeze_pretrained = freeze_pretrained
        self.load_optimizer_state = load_optimizer_state

        # Multi-label specific attributes
        self.label_names = label_names
        self.thresholds = thresholds or {t: 0.5 for t in label_names}
        self.epochs = max(1, int(epochs))
        self.scaler = None

        # Load scaler if provided
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

        # Instantiate module with temporary input_dim=1
        # Will be properly initialized with checkpoint in initialize_module()
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

        # Mark that we need to initialize with checkpoint
        self.module_initialized = False

    def _adapt_input_dim(self, x: Dict):
        """Check if new features appear and update observed_features if needed."""
        self._update_observed_features(x)

    def initialize_module(self, x: Dict, **kwargs):
        """Initialize module and load pretrained weights."""
        torch.manual_seed(self.seed)

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extract model parameters from checkpoint
        if 'model_params' in checkpoint:
            model_params = checkpoint['model_params']
            self.seq_len = checkpoint.get('seq_len', self.window_size)
            feature_cols = checkpoint.get('feature_cols', [])

            # IMPORTANTE: Guardar el orden de features del checkpoint
            if feature_cols:
                self.feature_cols_order = feature_cols
                # Update observed features from checkpoint
                # Crear un dict con las features del checkpoint
                feature_dict = {feature: 0.0 for feature in feature_cols}
                self._update_observed_features(feature_dict)
        else:
            # Fallback if checkpoint doesn't have model_params
            model_params = {}
            n_features = len(x)
            model_params['input_dim'] = n_features
            model_params['output_dim'] = len(self.label_names)

        # Instantiate module with checkpoint parameters
        if not isinstance(self.module_cls, torch.nn.Module):
            # Merge checkpoint params with kwargs, prioritizing checkpoint
            filtered_kwargs = {
                k: v
                for k, v in self.module_kwargs.items()
                if k not in {"n_features", "input_dim", "output_dim"}
            }

            # Override with checkpoint params where available
            for key in ['hidden_dim', 'num_layers', 'dropout', 'bidirectional']:
                if key in model_params:
                    filtered_kwargs[key] = model_params[key]

            self.module = self.module_cls(
                input_dim=model_params['input_dim'],
                output_dim=model_params['output_dim'],
                **filtered_kwargs,
            )

        # Load pretrained weights
        self.module.load_state_dict(checkpoint['model_state_dict'])
        self.module.to(self.device)

        # Freeze layers if requested
        if self.freeze_pretrained:
            for param in self.module.parameters():
                param.requires_grad = False

        # Build optimizer
        from deep_river.utils import get_optim_fn
        optimizer_func = get_optim_fn(self.optimizer_fn)
        self.optimizer = optimizer_func(
            filter(lambda p: p.requires_grad, self.module.parameters()),
            lr=self.lr
        )

        # Optionally load optimizer state
        if self.load_optimizer_state and 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")

        self.module_initialized = True

        # Print loading summary
        trainable_params = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.module.parameters())
        print(f"✓ Loaded pretrained model from: {self.checkpoint_path}")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Frozen parameters: {total_params - trainable_params:,}")
        if 'epoch' in checkpoint:
            print(f"  - Pretrained epochs: {checkpoint['epoch']}")
        if 'final_loss' in checkpoint:
            print(f"  - Pretrained final loss: {checkpoint['final_loss']:.4f}")

    def _to_window_tensor(self, x_window) -> torch.Tensor:
        """Convert deque window to tensor in batch_first format."""
        # deque2rolling_tensor -> [seq_len, batch=1, n_features]
        x_t = deque2rolling_tensor(x_window, device=self.device)
        # Convert to batch_first [batch=1, seq_len, n_features] for batch_first LSTM
        return x_t.permute(1, 0, 2).contiguous()

    def _preprocess_features(self, x: Dict) -> Dict:
        """Apply feature transformations and scaler."""
        import numpy as np

        # Paso 1: Aplicar one-hot encoding a 'Type' si existe
        x_processed = x.copy()

        # Si tenemos 'Type' en x y esperamos Type_H, Type_L, Type_M
        if 'Type' in x_processed and hasattr(self, 'feature_cols_order'):
            if any(f.startswith('Type_') for f in self.feature_cols_order):
                # Extraer el valor de Type
                type_value = x_processed.pop('Type')

                # Crear las columnas one-hot
                for type_variant in ['H', 'L', 'M']:
                    feature_name = f'Type_{type_variant}'
                    if feature_name in self.feature_cols_order:
                        x_processed[feature_name] = 1.0 if type_value == type_variant else 0.0

        # Paso 2: Filtrar solo las features que el modelo espera
        if hasattr(self, 'feature_cols_order'):
            # Mantener solo las features del checkpoint
            x_filtered = {
                feature: x_processed.get(feature, 0.0)
                for feature in self.feature_cols_order
            }
        else:
            x_filtered = x_processed

        # Paso 3: Aplicar scaler si está disponible
        if self.scaler is None:
            return x_filtered

        # Si tenemos feature_cols del checkpoint, usar ese orden
        if hasattr(self, 'feature_cols_order'):
            feature_vector = [x_filtered.get(feature, 0) for feature in self.feature_cols_order]
            scaled = self.scaler.transform(np.array([feature_vector]))[0]
            # Retornar en el mismo orden
            return {feature: float(scaled[i]) for i, feature in enumerate(self.feature_cols_order)}
        else:
            # Fallback: usar observed_features
            feature_vector = [x_filtered.get(feature, 0) for feature in self.observed_features]
            scaled = self.scaler.transform(np.array([feature_vector]))[0]
            return {feature: float(scaled[i]) for i, feature in enumerate(self.observed_features)}

    def learn_one(self, x: Dict, y: Dict, **kwargs) -> None:
        """Learn from one example using the rolling window."""
        # Initialize module if needed
        if not self.module_initialized:
            # NO llamar a _update_observed_features aquí
            # El initialize_module lo hará usando las features del checkpoint
            self.initialize_module(x=x, **self.module_kwargs)

        # Preprocess features PRIMERO (esto hace one-hot encoding y filtra features)
        x_processed = self._preprocess_features(x)

        # Ahora x_processed tiene solo las features que el modelo espera
        # Append to window usando feature_cols_order (no observed_features)
        if hasattr(self, 'feature_cols_order'):
            self._x_window.append([x_processed.get(feature, 0) for feature in self.feature_cols_order])
        else:
            self._x_window.append([x_processed.get(feature, 0) for feature in self.observed_features])

        # Train at every step using the current window (even if not full yet)
        if len(self._x_window) > 0:
            X_t = self._to_window_tensor(self._x_window)
            # Target tensor [1, L]
            y_vec = torch.tensor(
                [[float(y.get(t, 0)) for t in self.label_names]],
                dtype=torch.float32,
                device=self.device,
            )

            # Train for configured number of epochs per online sample/window
            for _ in range(self.epochs):
                self.module.train()
                self.optimizer.zero_grad()
                logits = self.module(X_t)
                loss = F.binary_cross_entropy_with_logits(logits, y_vec)
                loss.backward()
                self.optimizer.step()

    def predict_proba_one(self, x: Dict) -> Dict[str, float]:
        """Predict probabilities for one example."""
        # Initialize if needed (for observed_features and module)
        if not self.module_initialized:
            # NO llamar a _update_observed_features aquí
            # El initialize_module lo hará usando las features del checkpoint
            self.initialize_module(x=x, **self.module_kwargs)

        # Preprocess features PRIMERO (esto hace one-hot encoding y filtra features)
        x_processed = self._preprocess_features(x)

        # Prepare window copy including current x
        x_win = self._x_window.copy()
        if hasattr(self, 'feature_cols_order'):
            x_win.append([x_processed.get(feature, 0) for feature in self.feature_cols_order])
        else:
            x_win.append([x_processed.get(feature, 0) for feature in self.observed_features])

        if self.append_predict:
            self._x_window = x_win

        if len(x_win) < self.window_size:
            # Not enough context; return zeros
            return {t: 0.0 for t in self.label_names}

        X_t = self._to_window_tensor(x_win)
        self.module.eval()
        with torch.inference_mode():
            logits = self.module(X_t)
            probs = torch.sigmoid(logits).squeeze(0).detach().cpu().tolist()

        return {t: float(probs[i]) for i, t in enumerate(self.label_names)}

    def predict_one(self, x: Dict) -> Dict[str, int]:
        """Predict binary labels for one example."""
        probas = self.predict_proba_one(x)
        it_is = {t: int(probas[t] >= self.thresholds.get(t, 0.5)) for t in self.label_names}
        return it_is

    def save_checkpoint(self, path: str):
        """Save current model state to a checkpoint file."""
        checkpoint = {
            'model_state_dict': self.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'observed_features': list(self.observed_features),
            'label_names': self.label_names,
            'window_size': self.window_size,
            'thresholds': self.thresholds,
        }
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved to: {path}")
