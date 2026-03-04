from typing import Dict, List, Type, Union
import torch
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier
from classes.online_scheduler import OnlineLRScheduler

class AdaptiveRollingMultiLabelClassifier(RollingMultiLabelClassifier):
    """
    Extension of RollingMultiLabelClassifier that integrates an adaptive Learning Rate scheduler.
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
        loss_fn=None,
        gradient_scale: float = 1.0,
        scheduler_patience: int = 1000,
        scheduler_factor: float = 0.5,
        scheduler_min_lr: float = 1e-6,
        scheduler_smoothing: float = 0.98,
        scheduler_max_lr: float = 1.0, # [NEW]
        scheduler_boost_factor: float = 1.0, # [NEW]
        scheduler_drift_tolerance: float = 0.1, # [NEW]
        scheduler_sigma_threshold: float = 3.0, # [NEW]
        **kwargs,
    ):
        super().__init__(
            module=module,
            label_names=label_names,
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
            window_size=window_size,
            append_predict=append_predict,
            thresholds=thresholds,
            epochs=epochs,
            pos_weight=pos_weight,
            loss_fn=loss_fn,
            gradient_scale=gradient_scale,
            **kwargs
        )
        
        # Scheduler config
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_min_lr = scheduler_min_lr
        self.scheduler_smoothing = scheduler_smoothing
        self.scheduler_max_lr = scheduler_max_lr
        self.scheduler_boost_factor = scheduler_boost_factor
        self.scheduler_drift_tolerance = scheduler_drift_tolerance
        self.scheduler_sigma_threshold = scheduler_sigma_threshold
        self.scheduler = None

    def initialize_module(self, x: Dict, **kwargs):
        """Initialize module and then the scheduler."""
        super().initialize_module(x, **kwargs)
        
        # After super() creates self.optimizer, we can create the scheduler
        if self.optimizer is not None:
            self.scheduler = OnlineLRScheduler(
                optimizer=self.optimizer,
                patience=self.scheduler_patience,
                factor=self.scheduler_factor,
                min_lr=self.scheduler_min_lr,
                smoothing=self.scheduler_smoothing,
                max_lr=self.scheduler_max_lr,
                boost_factor=self.scheduler_boost_factor,
                drift_tolerance=self.scheduler_drift_tolerance,
                sigma_threshold=self.scheduler_sigma_threshold,
            )

    def learn_one(self, x: Dict, y: Dict, **kwargs) -> None:
        # Capture loss calculation logic from parent is tricky without copying code 
        # because the parent doesn't return loss.
        # So we have to copy the training loop part or hook into it. 
        # To avoid massive duplication, we'll assume we can copy the relevant logic 
        # or we accept that valid design might require slight duplication for the training step.
        # Given the constraint to not modify the original, we must override learn_one completely.
        
        # Initialize module if needed
        if not self.module_initialized:
            self._update_observed_features(x)
            self.initialize_module(x=x, **self.module_kwargs)

        # Check if new features appeared
        prev_num_features = len(self.observed_features)
        self._adapt_input_dim(x)
        if len(self.observed_features) > prev_num_features:
            print(f"⚠ Warning: New features detected ({prev_num_features} -> {len(self.observed_features)}). Reinitializing model...")
            self.initialize_module(x=x, **self.module_kwargs)
            self._x_window.clear()
            self._y_window.clear()

        # Append to window
        self._x_window.append([x.get(feature, 0) for feature in self.observed_features])
        self._y_window.append([float(y.get(t, 0)) for t in self.label_names])

        # Train
        if len(self._x_window) > 0:
            X_t = self._to_window_tensor(self._x_window)
            y_vec = torch.tensor(
                list(self._y_window),
                dtype=torch.float32,
                device=self.device,
            )

            total_loss = 0.0
            
            for _ in range(self.epochs):
                self.module.train()
                self.optimizer.zero_grad()
                logits = self.module(X_t)

                if self.loss_fn is not None:
                    loss = self.loss_fn(logits, y_vec)
                elif self.pos_weight is not None:
                    pos_weight_tensor = torch.full(
                        (len(self.label_names),),
                        self.pos_weight,
                        device=self.device
                    )
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, y_vec, pos_weight=pos_weight_tensor
                    )
                else:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_vec)

                loss.backward()

                if self.gradient_scale != 1.0:
                    for p in self.module.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(self.gradient_scale)

                self.optimizer.step()
                total_loss += loss.item()
            
            # Update scheduler with the average loss of this step (across epochs)
            avg_step_loss = total_loss / self.epochs
            if self.scheduler:
                self.scheduler.step(avg_step_loss)
