from typing import Dict, List, Type, Union, Optional
import torch
import copy
import numpy as np
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier

class EvolutionaryRollingClassifier(RollingMultiLabelClassifier):
    """
    Rolling Multi-Label Classifier that adapts Learning Rate using an Evolutionary Strategy (Greedy Population of 3).
    
    Mechanism:
    1. Maintains 3 candidate models:
       - LowLR Candidate (LR / factor)
       - CurrLR Candidate (LR)
       - HighLR Candidate (LR * factor)
    2. On receiving (x, y):
       - Evaluates prediction loss of all 3 candidates on (x, y) BEFORE training.
       - Selects the 'Winner' (lowest loss).
       - Updates the internal Base LR to the Winner's LR.
    3. Regeneration:
       - Overwrites all 3 candidates with the Winner's state.
       - Trains the 3 candidates on (x, y) using their respective scaled LRs (Base/factor, Base, Base*factor).
    
    This essentially performs a 1-step lookahead to see which LR produced the best state for the *current* sample.
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
        # Evolutionary Params
        mutation_factor: float = 2.0,  # Factor to scale LR up/down
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
        self.mutation_factor = mutation_factor
        
        # We need to manage 3 independent model+optimizer pairs
        # We will initialize them when initialize_module is called
        self.candidates = [] # List of dicts: {'module': m, 'optim': o, 'lr_mult': float}
        self.current_base_lr = lr
        self.winner_idx = 1 # Default to center (CurrLR)

    def initialize_module(self, x: Dict, **kwargs):
        """Initialize the main module and then clone it for candidates."""
        super().initialize_module(x, **kwargs)
        
        if self.module_initialized:
            # Create the population
            self.candidates = []
            multipliers = [1.0/self.mutation_factor, 1.0, self.mutation_factor]
            
            for mult in multipliers:
                # Clone module
                cand_mod = copy.deepcopy(self.module)
                cand_mod.to(self.device)
                
                # Create NEW optimizer for this candidate
                from deep_river.utils import get_optim_fn
                optimizer_func = get_optim_fn(self.optimizer_fn)
                cand_optim = optimizer_func(cand_mod.parameters(), lr=self.current_base_lr * mult)
                
                # Copy optimizer state if the main one has state (usually empty at init, but good practice)
                cand_optim.load_state_dict(self.optimizer.state_dict())
                # Fix LR in optimizer (load_state_dict might overwrite it)
                for pg in cand_optim.param_groups:
                    pg['lr'] = self.current_base_lr * mult
                
                self.candidates.append({
                    'module': cand_mod,
                    'optim': cand_optim,
                    'multiplier': mult
                })

    def _get_loss(self, module, X_t, y_vec):
        """Helper to calculate loss/error for a candidate."""
        module.eval()
        with torch.inference_mode():
            logits = module(X_t)
            
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
        return loss.item()

    def learn_one(self, x: Dict, y: Dict, **kwargs) -> None:
        # 1. Init / Reinit logic
        if not self.module_initialized:
            self._update_observed_features(x)
            self.initialize_module(x=x, **self.module_kwargs)

        prev_num_features = len(self.observed_features)
        self._adapt_input_dim(x)
        if len(self.observed_features) > prev_num_features:
            print(f"⚠ Warning: New features detected. Reinitializing population...")
            self.initialize_module(x=x, **self.module_kwargs)
            self._x_window.clear()
            self._y_window.clear()

        # 2. Append to window
        self._x_window.append([x.get(feature, 0) for feature in self.observed_features])
        self._y_window.append([float(y.get(t, 0)) for t in self.label_names])
        
        if len(self._x_window) == 0:
            return

        # 3. Preparation
        X_t = self._to_window_tensor(self._x_window)
        y_vec = torch.tensor(list(self._y_window), dtype=torch.float32, device=self.device)

        # 4. EVALUATION TOURNAMENT (Who predicts this new sample best?)
        # Candidates state is from Previous Step (trained with different LRs)
        losses = []
        for cand in self.candidates:
            loss = self._get_loss(cand['module'], X_t, y_vec)
            losses.append(loss)
            
        # Select Winner
        best_idx = int(np.argmin(losses))
        self.winner_idx = best_idx
        winner_cand = self.candidates[best_idx]
        
        # Update Base LR based on winner's multiplier
        # e.g. if winner was HighLR (x2.0), new base is old base * 2.0
        old_base = self.current_base_lr
        self.current_base_lr = self.current_base_lr * winner_cand['multiplier']
        
        # Cap LR? (Optional, let's keep it simple or adhere to max_lr logic later)
        if self.current_base_lr > 10.0: self.current_base_lr = 10.0
        if self.current_base_lr < 1e-6: self.current_base_lr = 1e-6
        
        if best_idx != 1:
            print(f"🧬 Evolving LR: {old_base:.5f} -> {self.current_base_lr:.5f} (Winner Loss: {losses[best_idx]:.4f} vs Curr: {losses[1]:.4f})")
        
        # 5. REGENERATION (Cloning)
        # All candidates become clones of the winner, but set up to train with divergent LRs next
        
        # Extract Winner State
        winner_state = copy.deepcopy(winner_cand['module'].state_dict())
        # Note: We technically should clone optimizer state too to preserve momentum correctly.
        winner_optim_state = copy.deepcopy(winner_cand['optim'].state_dict())
        
        multipliers = [1.0/self.mutation_factor, 1.0, self.mutation_factor]
        
        for i, mult in enumerate(multipliers):
            target_lr = self.current_base_lr * mult
            
            # Restore state
            self.candidates[i]['module'].load_state_dict(winner_state)
            self.candidates[i]['optim'].load_state_dict(winner_optim_state)
            
            # Update LR in optimizer
            for pg in self.candidates[i]['optim'].param_groups:
                pg['lr'] = target_lr
            
            # 6. TRAIN (Mutation)
            # Train this candidate on the current sample(s) with its specific LR
            self.candidates[i]['module'].train()
            self.candidates[i]['optim'].zero_grad()
            logits = self.candidates[i]['module'](X_t)
            
            # Reuse logic for loss
            if self.loss_fn is not None:
                loss_val = self.loss_fn(logits, y_vec)
            elif self.pos_weight is not None:
                pw_t = torch.full((len(self.label_names),), self.pos_weight, device=self.device)
                loss_val = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_vec, pos_weight=pw_t)
            else:
                loss_val = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_vec)
            
            # Multi-epoch loop supported? Simplification: 1 epoch for now or loop
            # Original wrapper uses self.epochs. Let's do 1 step per learn_one for speed in this complex setup
            loss_val.backward()
            
            # Gradient Scale
            if self.gradient_scale != 1.0:
                for p in self.candidates[i]['module'].parameters():
                    if p.grad is not None:
                       p.grad.data.mul_(self.gradient_scale)
                       
            self.candidates[i]['optim'].step()

        # Update the main 'self.module' to match the winner (for predict_proba_one and external access)
        # We assume the "Center" candidate (index 1) after training is the most representative "Current" state
        # Or we could set it to one of the trained ones. Center is safest.
        self.module = self.candidates[1]['module']


    def predict_proba_one(self, x: Dict) -> Dict[str, float]:
        # Use the "Center" candidate (Current Base LR) for predictions
        # Or better: Use the module we synced at end of learn_one
        if not self.candidates:
            return super().predict_proba_one(x)
            
        # Ensure we use the best synced module
        # But wait, predict_one happens BEFORE learn_one in standard Evaluate-then-Train loops (prequential).
        # In that case, self.candidates[1] holds the state from (t-1).
        
        # Standard flow logic reuse
        return super().predict_proba_one(x)
