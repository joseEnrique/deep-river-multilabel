import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
from typing import Dict, List, Type, Tuple
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier

class KNNBuffer:
    """
    Smart KNN Buffer.
    Prioritizes Rare Events to ensure they represent a significant portion of memory.
    Stores data on GPU for maximum speed (Full GPU Mode).
    """
    def __init__(self, maxsize=2000, k=5, rare_threshold=0.10):
        self.maxsize = maxsize
        self.k = k
        self.rare_threshold = rare_threshold
        # Stores tensors directly on GPU (no .cpu())
        self.X: List[torch.Tensor] = []
        self.Y: List[torch.Tensor] = []
        
        # Stats
        self._label_counts = {}
        self._total_samples = 0
        
    def _is_rare(self, y: torch.Tensor) -> bool:
        if self._total_samples < 50: return True
        
        is_rare_sample = False
        for i, val in enumerate(y):
            if val > 0.5:
                # We can keep stats calculation on CPU if needed, but here simple indexing is fine
                # y is tensor on GPU
                freq = self._label_counts.get(i, 0) / self._total_samples
                if freq < self.rare_threshold:
                    is_rare_sample = True
        return is_rare_sample

    def _get_rareness_score(self, y: torch.Tensor) -> float:
        """
        Returns a score indicating how 'rare' this sample is.
        Lower score = More Rare (contains labels with low frequency).
        Score = Min(Frequency of active labels).
        If no labels are active (all zeros), returns 1.0 (treated as common).
        """
        active_indices = torch.nonzero(y > 0.5, as_tuple=False).squeeze(1)
        if len(active_indices) == 0:
            return 1.0 # No labels, treat as common/boring
        
        freqs = []
        for idx in active_indices:
            idx_val = idx.item()
            # Calculate frequency: count / total
            f = self._label_counts.get(idx_val, 0) / max(1, self._total_samples)
            freqs.append(f)
            
        return min(freqs)

    def add(self, x: torch.Tensor, y: torch.Tensor):
        # 1. Update Global Stats
        self._total_samples += 1
        y_cpu = y.detach().cpu().numpy()
        active_indices = [i for i, val in enumerate(y_cpu) if val > 0.5]
        for i in active_indices:
            self._label_counts[i] = self._label_counts.get(i, 0) + 1
            
        # 2. Determine if we shoud store this sample
        current_score = self._get_rareness_score(y)
        is_rare = (current_score < self.rare_threshold)
        
        should_store = False
        
        if len(self.X) < self.maxsize:
            should_store = True
            idx_to_remove = -1
        else:
            # Buffer is Full. Logic to decide entry and eviction.
            if is_rare:
                # Always try to store rare samples
                should_store = True
            else:
                # Random admission for common samples to maintain valid distribution
                if random.random() < 0.10: # Increased slightly to 10%
                    should_store = True
            
            if should_store:
                # SMART EVICTION: Find a "Common" victim
                # Sample k candidates from the buffer and pick the one with HIGHEST rareness score (Least Rare)
                # This protects already stored rare samples.
                
                # Heuristic: Check 50 random candidates
                candidates = random.sample(range(len(self.X)), min(len(self.X), 50))
                
                best_victim_idx = -1
                highest_score = -1.0
                
                for idx in candidates:
                    # We can use the cached Y in memory
                    # y_cand is on GPU, _get_rareness_score handles it
                    score = self._get_rareness_score(self.Y[idx])
                    if score > highest_score:
                        highest_score = score
                        best_victim_idx = idx
                
                idx_to_remove = best_victim_idx
        
        # 3. Execute
        if should_store:
            # If we need to remove someone to make space
            if idx_to_remove != -1:
                # Efficient swap-remove (swap with last, pop last) to avoid O(N) shift
                # But we must be careful if order matters? KNN doesn't care about order.
                last_idx = len(self.X) - 1
                if idx_to_remove != last_idx:
                    self.X[idx_to_remove] = self.X[last_idx]
                    self.Y[idx_to_remove] = self.Y[last_idx]
                
                self.X.pop()
                self.Y.pop()

            self.X.append(x.detach()) 
            self.Y.append(y.detach())
        
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        if not self.X:
             return None
            
        # Stack memory (already on GPU)
        # Note: stacking might create a large tensor copy. 
        # For 4000x128 float32, it's ~2MB. Trivial for VRAM.
        X_mem = torch.stack(self.X) # (N, dim)
        Y_mem = torch.stack(self.Y) # (N, labels)
        
        # Ensure z is on same device (it should be)
        if z.device != X_mem.device:
            z = z.to(X_mem.device)
            
        # Cosine Similarity
        z_norm = F.normalize(z.unsqueeze(0), p=2, dim=1)
        X_norm = F.normalize(X_mem, p=2, dim=1)
        sims = torch.mm(z_norm, X_norm.t()).squeeze(0) # (N,)
        
        # Top K
        k_actual = min(len(self.X), self.k)
        topk_vals, topk_idxs = torch.topk(sims, k_actual)
        
        # IDW with Softmax
        weights = F.softmax(topk_vals * 5.0, dim=0).unsqueeze(1) 
        
        neighbors_y = Y_mem[topk_idxs]
        pred = (neighbors_y * weights).sum(dim=0)
        
        return pred

class RollingDeepKNN(RollingMultiLabelClassifier):
    def __init__(
        self,
        module: Type[torch.nn.Module],
        label_names: List[str],
        knn_size: int = 4000,
        knn_k: int = 5,
        rare_threshold: float = 0.10,
        **kwargs
    ):
        super().__init__(module, label_names, **kwargs)
        self.knn = KNNBuffer(maxsize=knn_size, k=knn_k, rare_threshold=rare_threshold)
        self.knn_k = knn_k

    def _get_embedding(self, x_tensor):
        with torch.no_grad():
            self.module.eval()
            x = x_tensor
            if x.dim() == 3 and x.size(1) == 1:
                x = x.squeeze(1)
            x = x.long()
            embedded = self.module.embedding(x)
            _, (hn, _) = self.module.lstm(embedded)
            if self.module.lstm.bidirectional:
                last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
            else:
                last_hidden = hn[-1]
            return last_hidden.squeeze(0)

    def learn_one(self, x: Dict, y: Dict, **kwargs) -> None:
        super().learn_one(x, y, **kwargs)
        
        x_vec_list = [[x.get(feature, 0) for feature in self.observed_features]]
        x_tensor = self._to_window_tensor(x_vec_list)
        y_vec = torch.tensor([float(y.get(t, 0)) for t in self.label_names], dtype=torch.float32, device=self.device)
        
        z = self._get_embedding(x_tensor)
        self.knn.add(z, y_vec)

    def predict_proba_one(self, x: Dict) -> Dict[str, float]:
        if not self.module_initialized:
            return {l: 0.0 for l in self.label_names}
            
        x_vec_list = [[x.get(feature, 0) for feature in self.observed_features]]
        x_tensor = self._to_window_tensor(x_vec_list)
        
        z = self._get_embedding(x_tensor)
        
        pred_probs = self.knn.predict(z)
        
        if pred_probs is None:
            self.module.eval()
            with torch.no_grad():
                logits = self.module(x_tensor)
                pred_probs = torch.sigmoid(logits).squeeze(0)
            
        result = {label: float(pred_probs[i]) for i, label in enumerate(self.label_names)}
        return result
