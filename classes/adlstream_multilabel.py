"""
ADLStream Multi-Label Classifier with Temporal Context

Implementation of Asynchronous Deep Learning for Data Streams (ADLStream)
adapted for multi-label classification with deep-river framework.

Key Features:
1. Dual Asynchronous Pipeline (P1: Prediction, P2: Training)
2. Incremental Learning with batching
3. Biased Reservoir Sampling with decay (λ=0.98)
4. Temporal Context Window (past_size) for LSTM sequences
5. Robust to Concept Drift

Reference: ADLStream framework for high-speed data streams
"""

from typing import Dict, List, Type, Union, Callable
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from sortedcontainers import SortedSet
from deep_river.base import DeepEstimator
from river import base as river_base


class ADLStreamMultiLabelClassifier(DeepEstimator, river_base.MultiLabelClassifier):
    """
    ADLStream-based Multi-Label Classifier for streaming data.

    Architecture:
    - P1 (Prediction Layer): Fast inference with current model
    - P2 (Training Layer): Batch training with biased reservoir sampling

    Parameters
    ----------
    module : Type[torch.nn.Module]
        PyTorch module class (e.g., LSTM_MultiLabel)
    label_names : List[str]
        List of label names for multi-label classification
    optimizer_fn : Union[str, Type[torch.optim.Optimizer]]
        Optimizer (e.g., "adam")
    lr : float
        Learning rate
    device : str
        Device ("cpu" or "cuda")
    seed : int
        Random seed
    reservoir_size : int
        Size of biased reservoir (m = s × b)
    batch_size : int
        Batch size for training (s)
    num_batches : int
        Number of batches before model update (b)
    decay_lambda : float
        Decay factor for biased sampling (default: 0.98)
    loss_fn : torch.nn.Module, optional
        Custom loss function
    **module_kwargs
        Additional kwargs for the module (hidden_dim, num_layers, etc.)
    """

    def __init__(
        self,
        module: Type[torch.nn.Module],
        label_names: List[str],
        optimizer_fn: Union[str, Callable] = "adam",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        reservoir_size: int = 200,
        batch_size: int = 32,
        num_batches: int = 1,
        decay_lambda: float = 0.98,
        loss_fn: torch.nn.Module = None,
        use_dual_gpu: bool = False,
        **module_kwargs
    ):
        # Store module class and params for deferred initialization
        self.module_cls = module
        self.module_kwargs = module_kwargs
        self.label_names = label_names
        self.reservoir_size = reservoir_size
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.decay_lambda = decay_lambda
        self.use_dual_gpu = use_dual_gpu

        # Create temporary module with input_dim=1 for DeepEstimator initialization
        torch.manual_seed(seed)
        filtered_kwargs = {
            k: v for k, v in module_kwargs.items()
            if k not in {"n_features", "input_dim", "output_dim"}
        }
        temp_module = module(
            input_dim=1,
            output_dim=len(label_names),
            **filtered_kwargs
        )

        # Initialize DeepEstimator with temp module
        super().__init__(
            module=temp_module,
            loss_fn=loss_fn if loss_fn else nn.BCEWithLogitsLoss(),
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
        )

        # Dual GPU setup
        if self.use_dual_gpu and torch.cuda.device_count() >= 2:
            self.device_p1 = "cuda:0"  # P1 (Prediction) on GPU 0
            self.device_p2 = "cuda:1"  # P2 (Training) on GPU 1
            print(f"🚀 Dual GPU Mode: P1 on {self.device_p1}, P2 on {self.device_p2}")
        else:
            self.device_p1 = device
            self.device_p2 = device
            if self.use_dual_gpu:
                print(f"⚠️  Dual GPU requested but only {torch.cuda.device_count()} GPU(s) available. Using single GPU.")

        # P1: Prediction model (always ready for inference)
        self.prediction_model = None

        # P2: Training components (biased reservoir sampling)
        self.reservoir = []  # List of (x, y, weight) tuples
        self.total_weight = 0.0
        self.instances_seen = 0

        # Track observed features
        self.observed_features = SortedSet()
        self.module_initialized = False

    def _update_observed_features(self, x: Dict):
        """Update set of observed features."""
        for feature in x.keys():
            self.observed_features.add(feature)

    def _initialize_module(self, x: Dict):
        """Initialize the PyTorch module with correct input dimension."""
        import gc

        # Update observed features
        self._update_observed_features(x)
        n_features = len(self.observed_features)

        # Clean up previous module if exists
        if hasattr(self, 'module') and self.module is not None and self.module_initialized:
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                del self.optimizer
            self.module.cpu()
            del self.module
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Initialize new module
        torch.manual_seed(self.seed)
        filtered_kwargs = {
            k: v for k, v in self.module_kwargs.items()
            if k not in {"n_features", "input_dim"}
        }

        self.module = self.module_cls(
            input_dim=n_features,
            output_dim=len(self.label_names),
            **filtered_kwargs
        )
        # P2 (Training) goes to device_p2
        self.module.to(self.device_p2)

        # Initialize optimizer
        from deep_river.utils import get_optim_fn
        optimizer_func = get_optim_fn(self.optimizer_fn)
        self.optimizer = optimizer_func(self.module.parameters(), lr=self.lr)

        # Move loss_fn to P2 device if it has parameters (like pos_weight)
        if hasattr(self.loss_fn, 'to'):
            self.loss_fn.to(self.device_p2)

        # Copy to prediction model (P1)
        self.prediction_model = self._copy_model_weights()

        self.module_initialized = True

    def _copy_model_weights(self):
        """Copy current model weights to prediction model (P1 <- P2)."""
        if not self.module_initialized:
            return None

        # Create a deep copy of the model for P1
        import copy
        prediction_model = copy.deepcopy(self.module)
        # P1 (Prediction) goes to device_p1
        prediction_model.to(self.device_p1)
        prediction_model.eval()
        return prediction_model

    def _biased_reservoir_sample(self, x: Dict, y: Dict):
        """
        Add instance to biased reservoir with decay weighting.

        Implements Algorithm A-Chao with temporal decay:
        - Recent instances have higher weight (more likely to be sampled)
        - Older instances decay with factor λ
        """
        # Decay all existing weights
        self.total_weight *= self.decay_lambda
        for i in range(len(self.reservoir)):
            x_i, y_i, w_i = self.reservoir[i]
            self.reservoir[i] = (x_i, y_i, w_i * self.decay_lambda)

        # New instance has weight 1.0
        new_weight = 1.0
        self.total_weight += new_weight

        if len(self.reservoir) < self.reservoir_size:
            # Reservoir not full, add directly
            self.reservoir.append((x.copy(), y.copy(), new_weight))
        else:
            # Reservoir full, probabilistic replacement
            # Probability of keeping new instance: new_weight / total_weight
            if random.random() < (new_weight / self.total_weight):
                # Replace random instance (weighted by inverse probability)
                # Simpler: replace random instance uniformly
                replace_idx = random.randint(0, self.reservoir_size - 1)
                self.reservoir[replace_idx] = (x.copy(), y.copy(), new_weight)

    def _sample_training_batch(self):
        """
        Sample a batch from reservoir using biased weights.

        Returns batch of (X, Y) where X is [batch_size, seq_len, n_features], Y is [batch_size, n_labels]
        """
        if len(self.reservoir) < self.batch_size:
            # Not enough samples, use all available
            batch = self.reservoir.copy()
        else:
            # Sample with weights
            weights = np.array([w for _, _, w in self.reservoir])
            weights = weights / weights.sum()  # Normalize

            indices = np.random.choice(
                len(self.reservoir),
                size=self.batch_size,
                replace=False,
                p=weights
            )
            batch = [self.reservoir[i] for i in indices]

        # Convert batch to tensors
        X_batch = []
        Y_batch = []

        for x, y, _ in batch:
            # Convert x dict to feature vector
            x_vec = [x.get(feature, 0.0) for feature in self.observed_features]
            # Add sequence dimension: each sample is a sequence of length 1
            X_batch.append([x_vec])

            # Convert y dict to label vector
            y_vec = [float(y.get(label, 0)) for label in self.label_names]
            Y_batch.append(y_vec)

        # P2 uses device_p2 for training
        # X shape: [batch_size, seq_len=1, n_features]
        X_tensor = torch.tensor(X_batch, dtype=torch.float32, device=self.device_p2)
        Y_tensor = torch.tensor(Y_batch, dtype=torch.float32, device=self.device_p2)

        return X_tensor, Y_tensor

    def _train_on_reservoir(self):
        """
        P2: Training phase

        Train model on reservoir using mini-batches.
        After training, sync weights to P1 (prediction model).
        """
        if len(self.reservoir) == 0:
            return

        self.module.train()

        # Train for num_batches iterations
        for _ in range(self.num_batches):
            X_batch, Y_batch = self._sample_training_batch()

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.module(X_batch)
            loss = self.loss_fn(logits, Y_batch)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Sync P2 → P1
        self.prediction_model = self._copy_model_weights()

    def predict_proba_one(self, x: Dict) -> Dict[str, float]:
        """
        P1: Fast prediction using current prediction model.

        Returns probability for each label.
        """
        # Initialize if needed
        if not self.module_initialized:
            self._update_observed_features(x)
            self._initialize_module(x)
            # Return default probabilities (0.5 for all labels)
            return {label: 0.5 for label in self.label_names}

        # Use P1 (prediction model) for fast inference on device_p1
        if self.prediction_model is None:
            # Fallback to main model if P1 not initialized
            model = self.module
            device = self.device_p2
        else:
            model = self.prediction_model
            device = self.device_p1

        model.eval()

        # Convert x to tensor on correct device
        # Shape: [batch_size, seq_len, features] = [1, 1, n_features] for LSTM
        x_vec = [x.get(feature, 0.0) for feature in self.observed_features]
        x_tensor = torch.tensor([[x_vec]], dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        return {label: float(prob) for label, prob in zip(self.label_names, probs)}

    def predict_one(self, x: Dict) -> Dict[str, int]:
        """
        P1: Fast prediction with threshold (default 0.5).
        """
        probs = self.predict_proba_one(x)
        return {label: int(prob > 0.5) for label, prob in probs.items()}

    def learn_one(self, x: Dict, y: Dict, **kwargs):
        """
        P2: Asynchronous learning via biased reservoir sampling.

        Steps:
        1. Add instance to biased reservoir
        2. If reservoir is full (m instances), trigger training
        3. Train on sampled batches
        4. Sync weights to P1
        """
        # Initialize if needed
        if not self.module_initialized:
            self._update_observed_features(x)
            self._initialize_module(x)

        # Check for new features
        prev_num_features = len(self.observed_features)
        self._update_observed_features(x)
        if len(self.observed_features) > prev_num_features:
            # New features detected - reinitialize
            self._initialize_module(x)
            self.reservoir.clear()
            self.total_weight = 0.0

        # Add to biased reservoir
        self._biased_reservoir_sample(x, y)
        self.instances_seen += 1

        # Trigger training when reservoir is full
        if len(self.reservoir) >= self.reservoir_size:
            self._train_on_reservoir()

        return self
