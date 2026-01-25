import random

from typing import Dict, List, Type, Union
import torch
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier

class RollingMemoryClassifier(RollingMultiLabelClassifier):
    """
    Rolling Classifier with Prioritized Experience Replay (Memory).
    
    It maintains a 'Rare Memory' buffer. Training steps include data from 
    the current sliding window PLUS a random batch from the rare memory.
    This mimics the 'memory' advantage of Tree/KNN methods for standard Deep Learning.
    """
    def __init__(
        self,
        module: Type[torch.nn.Module],
        label_names: List[str],
        memory_size: int = 2000,
        replay_batch_size: int = 32,
        rare_threshold: float = 0.05, # Labels appearing less than 5% are rare
        n_replay_steps: int = 1,
        **kwargs
    ):
        super().__init__(module, label_names, **kwargs)
        
        self.memory_size = memory_size
        self.replay_batch_size = replay_batch_size
        self.rare_threshold = rare_threshold
        self.n_replay_steps = n_replay_steps
        
        # Memory Buffer: List of (x, y) tuples (Changed from deque for smart eviction)
        self._rare_memory = [] # Manually managed list
        
        # Stats for rarity detection
        self._label_counts = {l: 0 for l in label_names}
        self._total_samples = 0

    def _update_stats(self, y: Dict):
        self._total_samples += 1
        for label, val in y.items():
            if val == 1:
                self._label_counts[label] += 1

    def _get_rareness_score(self, y: Dict) -> float:
        """
        Returns a score indicating how 'rare' this sample is.
        Lower score = More Rare.
        Score = Min(Frequency of active labels).
        """
        active_labels = [l for l, v in y.items() if v == 1]
        if not active_labels:
            return 1.0 # Common
            
        freqs = []
        for label in active_labels:
            f = self._label_counts[label] / max(1, self._total_samples)
            freqs.append(f)
            
        return min(freqs)

    def _is_relevant_for_memory(self, y: Dict) -> float:
        """
        Decides if a sample should be stored in memory.
        Now allows EVERYTHING. We rely on Smart Eviction to filter.
        """
        return True

    def _add_to_memory(self, x: Dict, y: Dict):
        # 1. Standard FIFO - Reverted from Smart Eviction
        if len(self._rare_memory) >= self.memory_size:
            self._rare_memory.pop(0)
        self._rare_memory.append((x.copy(), y.copy()))

    def learn_one(self, x: Dict, y: Dict, **kwargs) -> None:
        # 1. Update Stats
        self._update_stats(y)
        
        # 2. Check Memory Storage
        # We always add to memory now, trusting FIFO to keep recent history
        self._add_to_memory(x, y)
            
        # 3. Standard Window Update & Training (Super call)
        # This trains ONE step on the current window
        super().learn_one(x, y, **kwargs)
        
        # 4. Memory Replay Training (The "Flashback")
        # Only replay if we have enough memory and module is ready
        if self.module_initialized and len(self._rare_memory) >= self.replay_batch_size:
            for _ in range(self.n_replay_steps):
                self._replay_step()
            
    def _replay_step(self):
        # Balanced Sampling Strategy: 50% Rare, 50% Common
        
        # 1. Classify samples in memory
        rare_indices = []
        common_indices = []
        
        for i, (_, y_mem) in enumerate(self._rare_memory):
            if self._get_rareness_score(y_mem) < self.rare_threshold:
                rare_indices.append(i)
            else:
                common_indices.append(i)
                
        # 2. Determine sample counts
        n_rare_needed = self.replay_batch_size // 2
        n_common_needed = self.replay_batch_size - n_rare_needed
        
        # 3. Sample
        batch_indices = []
        
        # Rare Samples
        if rare_indices:
            if len(rare_indices) >= n_rare_needed:
                batch_indices.extend(random.sample(rare_indices, n_rare_needed))
            else:
                # Oversample if not enough rare
                batch_indices.extend(random.choices(rare_indices, k=n_rare_needed))
        else:
            # If absolutely no rare samples, fill with common
            n_common_needed += n_rare_needed

        # Common Samples
        if common_indices:
            if len(common_indices) >= n_common_needed:
                batch_indices.extend(random.sample(common_indices, n_common_needed))
            else:
                batch_indices.extend(random.choices(common_indices, k=n_common_needed))
        elif not rare_indices:
             # Buffer empty? Should not happen due to length check
             return

        # 4. Retrieve batch
        batch = [self._rare_memory[i] for i in batch_indices]
        
        # Convert to tensors
        # Note: We need to respect the input format expected by the module
        # The base class uses '_to_window_tensor' which expects a list of features
        # We need to adapt the sampled batch to this format
        
        x_batch_list = []
        y_batch_list = []
        
        for x_mem, y_mem in batch:
            # Re-check features (just in case)
            x_vec = [x_mem.get(feature, 0) for feature in self.observed_features]
            y_vec = [float(y_mem.get(t, 0)) for t in self.label_names]
            x_batch_list.append(x_vec)
            y_batch_list.append(y_vec)
            
        # Create Tensors
        # Shape: [batch, 1, n_features] -> [batch, seq_len=1, n_features]
        # Same format as _to_window_tensor but for a batch
        x_tensor = self._to_window_tensor(x_batch_list)
        y_tensor = torch.tensor(y_batch_list, dtype=torch.float32, device=self.device)
        
        # Train Step (One Gradient Descent step on Memory)
        self.module.train()
        self.optimizer.zero_grad()
        logits = self.module(x_tensor)
        
        # Reuse the loss logic from base class (handling WFL/PosWeight/LossFn)
        import torch.nn.functional as F
        
        if self.loss_fn is not None:
            loss = self.loss_fn(logits, y_tensor)
        elif self.pos_weight is not None:
            pos_weight_tensor = torch.full(
                (len(self.label_names),),
                self.pos_weight,
                device=self.device
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, y_tensor, pos_weight=pos_weight_tensor
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y_tensor)
            
        loss.backward()
        self.optimizer.step()
