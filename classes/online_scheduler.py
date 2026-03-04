import torch
from collections import deque

class OnlineLRScheduler:
    """
    Adaptive Learning Rate Scheduler for Online Learning.
    
    It maintains an Exponential Moving Average (EMA) of the loss.
    If the EMA loss does not improve for `patience` steps, the learning rate is decayed.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        patience: int = 1000,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        max_lr: float = 1.0, 
        smoothing: float = 0.98,
        threshold: float = 1e-4,
        boost_factor: float = 1.0,
        drift_tolerance: float = 0.1,
        sigma_threshold: float = 3.0 # [NEW] Number of std devs to trigger boost
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.smoothing = smoothing
        self.threshold = threshold
        self.boost_factor = boost_factor
        self.drift_tolerance = drift_tolerance
        self.sigma_threshold = sigma_threshold
        
        self.ema_loss = None
        self.ema_loss_sq = None # E[x^2] for variance calculation
        self.best_ema_loss = float('inf')
        self.steps_since_improvement = 0
        self.current_lr = self._get_lr()
        
    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
            
    def step(self, loss: float) -> float:
        """
        Update the scheduler with a new loss value.
        Returns the current learning rate.
        """
        # Update EMA loss and EMA loss^2
        if self.ema_loss is None:
            self.ema_loss = loss
            self.ema_loss_sq = loss ** 2
            std_dev = 0.0
        else:
            self.ema_loss = self.smoothing * self.ema_loss + (1 - self.smoothing) * loss
            self.ema_loss_sq = self.smoothing * self.ema_loss_sq + (1 - self.smoothing) * (loss ** 2)
            # Var = E[x^2] - (E[x])^2
            variance = max(0, self.ema_loss_sq - self.ema_loss ** 2)
            std_dev = variance ** 0.5
            
        # 1. Check for Drift (Boost) - Fast Sigma Detection
        # If current sample is an outlier (loss spike), we boost immediately.
        # OR if the trend is bad (previous drift tolerance check)
        
        is_spike = std_dev > 0 and (loss > self.ema_loss + self.sigma_threshold * std_dev)
        is_trend_drift = (self.ema_loss > self.best_ema_loss * (1 + self.drift_tolerance))
        
        if self.boost_factor > 1.0 and (is_spike or is_trend_drift):
             trigger = "Spike" if is_spike else "Trend"
             self._boost_lr(trigger, loss, std_dev)
             # Reset best_ema to current so we don't keep boosting infinitely if it stays high
             self.best_ema_loss = self.ema_loss 
             self.steps_since_improvement = 0
             return self.current_lr

        # 2. Check for Improvement (Decay Logic)
        if self.ema_loss < self.best_ema_loss - self.threshold:
            self.best_ema_loss = self.ema_loss
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1
            
        # Decay if patience exceeded
        if self.steps_since_improvement >= self.patience:
            self._decay_lr()
            self.steps_since_improvement = 0
            self.best_ema_loss = self.ema_loss # Reset baseline to current level
            
        return self.current_lr
        
    def _decay_lr(self):
        current_lr = self._get_lr()
        new_lr = max(current_lr * self.factor, self.min_lr)
        
        if new_lr < current_lr:
            print(f"📉 Decaying learning rate: {current_lr:.6f} -> {new_lr:.6f} (EMA Loss: {self.ema_loss:.4f})")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.current_lr = new_lr

    def _boost_lr(self, trigger, current_loss, std_dev):
        current_lr = self._get_lr()
        new_lr = min(current_lr * self.boost_factor, self.max_lr)
        
        if new_lr > current_lr:
            info = f"Loss={current_loss:.4f}, EMA={self.ema_loss:.4f}, Sigma={std_dev:.4f}"
            print(f"🚀 Boosting learning rate ({trigger}): {current_lr:.6f} -> {new_lr:.6f} [{info}]")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.current_lr = new_lr
