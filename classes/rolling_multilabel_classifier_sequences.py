"""
Direct Multi-Label Classifier for sequences with temporal windows.

Esta clase extiende DeepEstimator para trabajar con DataFrames como entrada
y mantener una ventana de secuencias para entrenamiento multi-label.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import deque
from typing import Union, Callable, Type, Dict, List
from deep_river.base import DeepEstimator
from deep_river.utils.tensor_conversion import deque2rolling_tensor
from river import base as river_base


class RollingMultiLabelClassifierSequences(DeepEstimator, river_base.MultiLabelClassifier):
    """
    A simplified DeepEstimator for multi-label classification with sequence tracking.

    This class extends DeepEstimator to work with DataFrame inputs and maintains
    a window of instances for training multi-label models.

    Parameters
    ----------
    window_size : int
        The number of instances to keep in the window.
    label_names : List[str]
        Names of the labels for multi-label classification.
    module : torch.nn.Module
        A PyTorch module that defines the neural network architecture.
    loss_fn : Union[str, Callable]
        The loss function to use for training.
    optimizer_fn : Union[str, Type[torch.optim.Optimizer]]
        The optimizer to use for training.
    lr : float, default=0.001
        Learning rate for the optimizer.
    output_is_logit : bool, default=True
        Whether the output of the model is a logit.
    device : str, default="cuda" if torch.cuda.is_available() else "cpu"
        The device to use for training and prediction.
    seed : int, default=42
        Random seed for reproducibility.
    epochs : int, default=1
        Number of epochs to train on each batch.
    **kwargs
        Additional keyword arguments to pass to the module constructor.

    Example
    -------
    >>> from testclassifier.model import LSTM_MultiLabel
    >>> from classes.rolling_multilabel_classifier_sequences import RollingMultiLabelClassifierSequences
    >>>
    >>> forecaster = RollingMultiLabelClassifierSequences(
    ...     window_size=100,
    ...     label_names=['TWF', 'HDF', 'PWF', 'OSF', 'RNF'],
    ...     module=LSTM_MultiLabel,
    ...     loss_fn='binary_cross_entropy',
    ...     optimizer_fn='adam',
    ...     lr=1e-4,
    ...     hidden_dim=128,
    ...     num_layers=2,
    ...     output_dim=5
    ... )
    """

    def __init__(
        self,
        window_size: int,
        label_names: List[str],
        module: torch.nn.Module = None,
        loss_fn: Union[str, Callable] = None,
        optimizer_fn: Union[str, Type[torch.optim.Optimizer]] = None,
        lr: float = 0.001,
        output_is_logit: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        epochs: int = 1,
        threshold: float = 0.5,
        past_history: int = 1,
        **kwargs,
    ):
        # Store parameters
        self.window_size = window_size
        self.past_history = past_history
        self.label_names = label_names
        self.n_labels = len(label_names)
        self.epochs = epochs
        self.output_is_logit = output_is_logit
        self.threshold = threshold

        # Store module class and kwargs for later initialization
        self.module_cls = module
        self.module_kwargs = kwargs
        self.module_initialized = False

        # Instantiate module if it's a class
        if isinstance(module, type) and issubclass(module, nn.Module):
            # Module is a class, instantiate it with temporary input_dim
            # Will be re-initialized when first data arrives
            filtered_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in {"n_features", "input_dim"}
            }
            torch.manual_seed(seed)
            # Create temporary module with input_dim=1
            module_instance = module(input_dim=1, **filtered_kwargs)
        else:
            # Module is already an instance
            module_instance = module
            self.module_initialized = True

        # Initialize parent with module instance
        super().__init__(
            module=module_instance,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            output_is_logit=output_is_logit,
            device=device,
            seed=seed,
        )

        # Buffer for the past_history tracking of elements (current sequence)
        self.history_buffer = deque(maxlen=self.past_history)
        
        # Buffer for batches of sequences and targets
        # The window stores window_size elements, each being a sequence of length past_history
        self.sequence_window = deque(maxlen=self.window_size)
        self.target_window = deque(maxlen=self.window_size)
        # Plain feature-vector window used for prediction (mirrors RollingMultiLabelClassifier._x_window)
        # This guarantees the same tensor memory layout for identical GPU float arithmetic
        self._x_window = deque(maxlen=self.window_size)
        # Track observed features for consistent ordering
        from sortedcontainers import SortedSet
        self.observed_features = SortedSet()

    def _update_observed_features(self, x: Dict):
        """Update the set of observed features."""
        if isinstance(x, dict):
            for feature in x.keys():
                self.observed_features.add(feature)

    def _initialize_module(self, x: Dict):
        """Initialize the module with the correct input_dim based on observed features."""
        import gc

        # Update observed features
        self._update_observed_features(x)
        n_features = len(self.observed_features)

        # Clean up previous module
        if hasattr(self, 'module') and self.module is not None:
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                del self.optimizer
            self.module.cpu()
            del self.module
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Instantiate the real module with correct input_dim
        torch.manual_seed(self.seed)
        if not isinstance(self.module_cls, torch.nn.Module):
            filtered_kwargs = {
                k: v
                for k, v in self.module_kwargs.items()
                if k not in {"n_features", "input_dim"}
            }
            self.module = self.module_cls(
                input_dim=n_features,
                **filtered_kwargs,
            )
        else:
            self.module = self.module_cls

        self.module.to(self.device)

        # Rebuild optimizer
        from deep_river.utils import get_optim_fn
        optimizer_func = get_optim_fn(self.optimizer_fn)
        self.optimizer = optimizer_func(self.module.parameters(), lr=self.lr)

        self.module_initialized = True

    def learn_one(self, x: Union[Dict, 'pd.DataFrame'], y: Dict[str, bool] = None):
        """
        Train the model with a new instance or sequence.

        Parameters
        ----------
        x : dict or pd.DataFrame
            If dict: single instance (from pipeline) - builds internal window
            If DataFrame: complete sequence with multiple timesteps
        y : Dict[str, bool]
            Dictionary with multi-label targets {label_name: True/False}.

        Returns
        -------
        self
        """
        if x is None or y is None:
            return self

        # Check if x is a DataFrame (sequence)
        import pandas as pd
        if isinstance(x, pd.DataFrame):
            return self._learn_sequence(x, y)
        else:
            return self._learn_single(x, y)

    def _learn_sequence(self, x_df: 'pd.DataFrame', y: Union[Dict[str, bool], List[Dict[str, bool]]]):
        """
        Train with a batch of instances (DataFrame with multiple rows).

        NEW BEHAVIOR (INDEPENDENT BATCH PROCESSING):
        - Treats the entire DataFrame as a SINGLE temporal sequence
        - Each batch is independent (no mixing with previous batches in buffer)
        - Trains ONCE per batch using the complete sequence
        - Example with past_size=5:
          * Batch 1: [x₂,x₃,x₄,x₅,x₆] → y₆  (train with this 5-timestep sequence)
          * Batch 2: [x₃,x₄,x₅,x₆,x₇] → y₇  (train with this 5-timestep sequence)
          * NO mixing between batches!

        The buffer is updated with ONLY the last instance for future predictions.
        """
        # Initialize module with correct input_dim on first call
        if not self.module_initialized:
            first_row = x_df.iloc[0].to_dict()
            # Filter out target columns to get only features
            first_row_features = {k: v for k, v in first_row.items() if k not in self.label_names}
            self._update_observed_features(first_row_features)
            self._initialize_module(first_row_features)

        # Check for new features from DataFrame (excluding target columns)
        prev_num_features = len(self.observed_features)
        for col in x_df.columns:
            if col not in self.label_names and col not in self.observed_features:
                self.observed_features.add(col)

        if len(self.observed_features) > prev_num_features:
            print(f"⚠ Warning: New features detected ({prev_num_features} -> {len(self.observed_features)}). Reinitializing model...")
            first_row = x_df.iloc[0].to_dict()
            first_row_features = {k: v for k, v in first_row.items() if k not in self.label_names}
            self.history_buffer.clear()
            self.y_window.clear()
            # Clear sequence buffers as well (feature dimension changed)
            if hasattr(self, 'sequence_buffer'):
                self.sequence_buffer.clear()
                self.target_buffer.clear()

        # Check if DataFrame contains target columns
        has_targets = all(label in x_df.columns for label in self.label_names)

        # Extract features from DataFrame rows to create the sequence
        x_sequence = []

        for idx, row in x_df.iterrows():
            # Extract features (excluding targets)
            if has_targets:
                row_features = {k: v for k, v in row.items() if k not in self.label_names}
            else:
                row_features = row.to_dict()

            # CRITICAL FIX: Replace NaN with 0 (e.g., from OneHotEncoder for unseen categories)
            import math
            x_vec = [
                row_features.get(feature, 0) if not (isinstance(row_features.get(feature, 0), float) and math.isnan(row_features.get(feature, 0))) else 0 
                for feature in self.observed_features
            ]
            x_sequence.append(x_vec)

        # Handle target: should be a single dict for the whole sequence
        if isinstance(y, list):
            # If y is a list, take the last one (for the whole sequence)
            y_target = y[-1] if len(y) > 0 else y[0]
        else:
            y_target = y

        # Convert target to vector
        y_vec = [float(y_target.get(label, 0)) for label in self.label_names]

        # Buffer stores INSTANCES (each instance = one sequence of past_size timesteps)
        # Initialize buffers if they don't exist
        if not hasattr(self, 'sequence_buffer'):
            self.sequence_buffer = []  # List of sequences
            self.target_buffer = []    # List of targets

        # ADD current sequence as new instance to buffer
        self.sequence_buffer.append(x_sequence)
        self.target_buffer.append(y_vec)

        # Limit buffer to window_size instances
        if len(self.sequence_buffer) > self.window_size:
            self.sequence_buffer.pop(0)
            self.target_buffer.pop(0)

        # Train with ALL buffered instances in parallel
        # Shape: [batch=len(buffer), seq_len=past_size, features]
        x_t = torch.tensor(
            self.sequence_buffer,
            dtype=torch.float32,
            device=self.device
        )

        # Shape: [batch=len(buffer), n_labels]
        y_t = torch.tensor(
            self.target_buffer,
            dtype=torch.float32,
            device=self.device
        )

        # Train with all buffered instances
        self.module.train()
        for _ in range(self.epochs):
            self._learn(x_t, y_t)

        # Keep history_buffer for predict_one compatibility
        self.history_buffer.clear()
        for x_vec in x_sequence:
            self.history_buffer.append(x_vec)
        self.y_window.clear()
        self.y_window.append(y_vec)
        return self

    def _learn_single(self, x: Dict, y: Dict[str, bool]):
        """
        Train with a single instance (dict).
        Builds internal window of instances.
        """
        # Initialize module with correct input_dim on first call
        if not self.module_initialized:
            self._update_observed_features(x)
            self._initialize_module(x)

        # Check for new features and reinitialize if needed
        prev_num_features = len(self.observed_features)
        self._update_observed_features(x)
        if len(self.observed_features) > prev_num_features:
            print(f"⚠ Warning: New features detected ({prev_num_features} -> {len(self.observed_features)}). Reinitializing model...")
            self._initialize_module(x)
            self.history_buffer.clear()
            self.sequence_window.clear()
            self.target_window.clear()
            self._x_window.clear()
            # Clear sequence buffers as well (feature dimension changed)
            if hasattr(self, 'sequence_buffer'):
                self.sequence_buffer.clear()
                self.target_buffer.clear()

        # Convert x dict to feature vector using observed feature order
        if len(self.sequence_window) > 0 and len(self.sequence_window[-1][0]) != len(self.observed_features):
            self.sequence_window.clear()
            self.target_window.clear()
            self.history_buffer.clear()
            self._x_window.clear()
            
        x_vec = [x.get(feature, 0) for feature in self.observed_features]

        # Append to history buffer (for past_history sequences)
        self.history_buffer.append(x_vec)
        # Append raw vec to flat window (mirrors RollingMultiLabelClassifier for identical prediction)
        self._x_window.append(x_vec)

        # Build current sequence (pad to past_history if needed)
        current_seq = list(self.history_buffer)
        if len(current_seq) < self.past_history:
            pad = [[0.0] * len(self.observed_features)] * (self.past_history - len(current_seq))
            current_seq = pad + current_seq
            
        self.sequence_window.append(current_seq)
        
        y_vec = [float(y.get(label, 0)) for label in self.label_names]
        self.target_window.append(y_vec)

        # Train at every step using the current window (even if not full yet)
        if len(self.sequence_window) > 0:
            x_t = torch.tensor(list(self.sequence_window), dtype=torch.float32, device=self.device)
            y_t = torch.tensor(list(self.target_window), dtype=torch.float32, device=self.device)

            # Train for multiple epochs - match RollingMultiLabelClassifier's exact loop structure
            for _ in range(self.epochs):
                self.module.train()
                self.optimizer.zero_grad()
                logits = self.module(x_t)
                loss = self.loss_fn(logits, y_t)
                loss.backward()
                self.optimizer.step()

        return self

    def _dataframes_to_tensor(self, dataframes):
        """
        Convert a list of DataFrames or dicts to a tensor.

        Parameters
        ----------
        dataframes : List[pd.DataFrame] or List[dict]
            List of DataFrames or dicts to convert.

        Returns
        -------
        torch.Tensor
            A tensor with shape [batch=1, seq_len=len(dataframes), features]
        """
        import pandas as pd

        arrays = []
        for df in dataframes:
            if isinstance(df, pd.DataFrame):
                # DataFrame: convertir directamente a array
                arrays.append(df.values.astype(np.float32))
            elif isinstance(df, dict):
                # Dict: convertir a array de valores
                # Asegurar orden consistente de features
                arr = np.array(list(df.values()), dtype=np.float32)
                arrays.append(arr)
            else:
                raise TypeError(f"Expected DataFrame or dict, got {type(df)}")

        # Apilar arrays en un tensor 3D: [batch=1, seq_len, features]
        tensor = torch.tensor(np.stack(arrays), dtype=torch.float32, device=self.device)

        # Añadir dimensión de batch si no existe
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        return tensor

    def predict_one(self, x: Union[Dict, 'pd.DataFrame']) -> Dict[str, bool]:
        """
        Predict multi-label output for a single instance or sequence.

        Parameters
        ----------
        x : dict or pd.DataFrame
            If dict: single instance (from pipeline)
            If DataFrame: complete sequence with multiple timesteps

        Returns
        -------
        Dict[str, bool]
            Dictionary with multi-label predictions {label_name: True/False}.
        """
        import pandas as pd

        # Check if x is a DataFrame (sequence)
        if isinstance(x, pd.DataFrame):
            return self._predict_sequence(x)
        else:
            return self._predict_single(x)

    def _predict_sequence(self, x_df: 'pd.DataFrame') -> Dict[str, bool]:
        """
        Predict with a complete sequence (DataFrame with multiple rows).

        NEW BEHAVIOR (INDEPENDENT BATCH PROCESSING):
        - Uses the DataFrame directly as the complete sequence
        - Does NOT combine with buffer (each batch is independent)
        - Consistent with new _learn_sequence() behavior

        CRITICAL: Excludes target columns to prevent data leakage.
        """
        # Initialize if needed
        if not self.module_initialized:
            first_row = x_df.iloc[0].to_dict()
            # Filter out targets to avoid leakage
            first_row_features = {k: v for k, v in first_row.items() if k not in self.label_names}
            self._update_observed_features(first_row_features)
            self._initialize_module(first_row_features)

        # Convert DataFrame to vectors - EXCLUDE target columns to prevent leakage
        x_sequence = []
        for _, row in x_df.iterrows():
            # Extract only features, NEVER targets
            row_features = {k: v for k, v in row.items() if k not in self.label_names}
            # CRITICAL FIX: Replace NaN with 0 (e.g., from OneHotEncoder for unseen categories)
            import math
            row_vec = [
                row_features.get(feature, 0) if not (isinstance(row_features.get(feature, 0), float) and math.isnan(row_features.get(feature, 0))) else 0
                for feature in self.observed_features
            ]
            x_sequence.append(row_vec)

        # Check if we have valid data to predict
        if len(x_sequence) == 0:
            return {label: False for label in self.label_names}

        # Use the DataFrame sequence directly (no buffer combination)
        # This matches the training behavior where each batch is independent
        x_t = torch.tensor(
            [x_sequence],
            dtype=torch.float32,
            device=self.device
        )

        # Set module to eval mode
        self.module.eval()

        with torch.no_grad():
            logits = self.module(x_t)

            # Convert to probabilities if output is logit
            if self.output_is_logit:
                probs = torch.sigmoid(logits)
            else:
                probs = logits

            # Convert to binary predictions using threshold
            preds = (probs >= self.threshold).cpu().numpy()[0]

            # Return as dict
            return {label: int(pred) for label, pred in zip(self.label_names, preds)}

    def _predict_single(self, x: Dict) -> Dict[str, bool]:
        """
        Predict with a single instance (dict).
        Uses internal window.
        """
        # Initialize if needed
        if not self.module_initialized:
            self._update_observed_features(x)
            self._initialize_module(x)

        # Check for new features and reinitialize if needed
        prev_num_features = len(self.observed_features)
        self._update_observed_features(x)
        if len(self.observed_features) > prev_num_features:
            print(f"⚠ Warning: New features detected in predict ({prev_num_features} -> {len(self.observed_features)}). Reinitializing model...")
            self._initialize_module(x)

        # Check if we have enough context (RollingMultiLabelClassifier equivalence)
        if len(self.sequence_window) + 1 < self.window_size:
            # Not enough context; return zeros
            return {label: 0 for label in self.label_names}

        # Prepare history copy including current x
        x_vec = [x.get(feature, 0) for feature in self.observed_features]
        current_history = list(self.history_buffer) + [x_vec]
        current_history = current_history[-self.past_history:]

        # If we don't have enough history, pad to past_history
        if len(current_history) < self.past_history:
            pad = [[0.0] * len(self.observed_features)] * (self.past_history - len(current_history))
            current_history = pad + current_history

        x_t = torch.tensor([current_history], dtype=torch.float32, device=self.device)

        # Predict
        self.module.eval()
        with torch.inference_mode():
            y_pred = self.module(x_t)

            # Apply sigmoid if output is logit
            if self.output_is_logit:
                y_pred = torch.sigmoid(y_pred)

            # Convert to binary predictions (threshold=self.threshold)
            y_pred_binary = (y_pred >= self.threshold).squeeze(0).cpu().numpy()

        # Convert to dictionary
        return {label: int(pred) for label, pred in zip(self.label_names, y_pred_binary)}

    def predict_proba_one(self, x: Union[Dict, 'pd.DataFrame']) -> Dict[str, float]:
        """
        Predict multi-label probabilities for a single instance or sequence.

        Parameters
        ----------
        x : dict or pd.DataFrame
            If dict: single instance (from pipeline)
            If DataFrame: complete sequence with multiple timesteps

        Returns
        -------
        Dict[str, float]
            Dictionary with multi-label probabilities {label_name: probability}.
        """
        import pandas as pd
        
        # Check if x is a DataFrame (sequence)
        if isinstance(x, pd.DataFrame):
            return self._predict_proba_sequence(x)

        # Ensure model is initialized
        if not self.module_initialized:
            self._update_observed_features(x)
            self._initialize_module(x)

        if len(self.observed_features) == 0:
            return {label: 0.0 for label in self.label_names}

        prev_num_features = len(self.observed_features)
        self._update_observed_features(x)

        if len(self.observed_features) > prev_num_features:
            print(f"⚠ Warning: New features detected in predict_proba ({prev_num_features} -> {len(self.observed_features)}). Reinitializing model...")
            self._initialize_module(x)
            self.history_buffer.clear()

        # Check if we have enough context (RollingMultiLabelClassifier equivalence)
        if len(self.sequence_window) + 1 < self.window_size:
            # Not enough context; return zeros
            return {label: 0.0 for label in self.label_names}

        # To exactly replicate RollingMultiLabelClassifier's prediction tensor:
        # Use the flat _x_window (raw feature vectors) + current x as the batch.
        # This ensures the same GPU memory layout -> identical float arithmetic -> identical metrics.
        x_vec = [x.get(feature, 0) for feature in self.observed_features]
        x_win = list(self._x_window) + [x_vec]

        # Wrap each feature vector into [seq_len=1, features] to match [batch, 1, features] shape
        x_win_batched = [[v] for v in x_win]
        x_t = torch.tensor(x_win_batched, dtype=torch.float32, device=self.device)

        # Predict
        self.module.eval()
        with torch.inference_mode():
            logits = self.module(x_t)
            # Take last element [-1] and apply sigmoid (same as RollingMultiLabelClassifier)
            probs = torch.sigmoid(logits[-1]).detach().cpu().tolist()

        # Convert to dictionary
        return {t: float(probs[i]) for i, t in enumerate(self.label_names)}

    def _predict_proba_sequence(self, x_df: 'pd.DataFrame') -> Dict[str, float]:
        """
        Predict probabilities with a complete sequence (DataFrame).

        NEW BEHAVIOR (INDEPENDENT BATCH PROCESSING):
        - Uses the DataFrame directly as the complete sequence
        - Does NOT combine with buffer (each batch is independent)
        - Consistent with new _learn_sequence() and _predict_sequence() behavior

        CRITICAL: Excludes target columns to prevent data leakage.
        """
        # Initialize if needed
        if not self.module_initialized:
            first_row = x_df.iloc[0].to_dict()
            # Filter out targets to avoid leakage
            first_row_features = {k: v for k, v in first_row.items() if k not in self.label_names}
            self._update_observed_features(first_row_features)
            self._initialize_module(first_row_features)

        # Convert DataFrame to vectors - EXCLUDE target columns to prevent leakage
        x_sequence = []
        for _, row in x_df.iterrows():
            # Extract only features, NEVER targets
            row_features = {k: v for k, v in row.items() if k not in self.label_names}
            # CRITICAL FIX: Replace NaN with 0 (e.g., from OneHotEncoder for unseen categories)
            import math
            row_vec = [
                row_features.get(feature, 0) if not (isinstance(row_features.get(feature, 0), float) and math.isnan(row_features.get(feature, 0))) else 0
                for feature in self.observed_features
            ]
            x_sequence.append(row_vec)

        # Check if we have valid data
        if len(x_sequence) == 0:
            return {label: 0.0 for label in self.label_names}

        # Use the DataFrame sequence directly (no buffer combination)
        x_t = torch.tensor(
            [x_sequence],
            dtype=torch.float32,
            device=self.device
        )

        # Predict
        self.module.eval()
        with torch.inference_mode():
            y_pred = self.module(x_t)

            # Apply sigmoid if output is logit
            if self.output_is_logit:
                y_pred = torch.sigmoid(y_pred)

            # Convert to probabilities
            y_proba = y_pred.squeeze(0).cpu().numpy()

        # Convert to dictionary
        return {label: float(proba) for label, proba in zip(self.label_names, y_proba)}



    def _learn(self, x: torch.Tensor, y: torch.Tensor):
        """
        Perform a single training step.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch, seq_len, features]
        y : torch.Tensor
            Target tensor [batch, n_labels]
        """
        # Move to device
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Zero gradients (matches Rolling)
        self.optimizer.zero_grad()
        

            
        # # DEBUG STATS
        # if np.random.random() < 0.05:
        #     print(f"X stats: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")

        # Check params
        # for name, param in self.module.named_parameters():
        #     if torch.isnan(param).any():
        #         print(f"NAN IN PARAM BEFORE FORWARD: {name}")
        #         break

        # Forward pass
        logits = self.module(x)
        
        # if torch.isnan(logits).any():
        #     print("NAN IN LOGITS")
        
        # Compute loss
        loss = self.loss_fn(logits, y)
        
        # Backward pass
        loss.backward()
        
        # DEBUG PRINT
        # if np.random.random() < 0.05: # Increased probability for visibility
        #     print(f"DEBUG _learn: x.shape={x.shape}, y.shape={y.shape}, loss={loss.item():.6f}")
        
        # Clip gradients - ENABLED to prevent exploding gradients
        # torch.nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        return self
