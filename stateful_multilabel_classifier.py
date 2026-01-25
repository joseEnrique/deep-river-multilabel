"""
Minimal Online LSTM - Solo los cambios esenciales para online learning.

Diferencias vs PyTorch estándar:
1. reduction='sum' en la loss
2. gradient_scale para compensar dilución de gradientes
3. SGD con learning_rate alto (0.5-1.0)

Shape de entrada: (batch, seq_len) donde cada posición es un alarm_id
El embedding convierte cada alarm_id a un vector de embedding_dim dimensiones.
"""

import torch
import torch.nn as nn
from river import base
from typing import Dict
import evaluate


class OnlineLSTM(nn.Module):
    """LSTM minimalista con embeddings para online learning.
    
    Input shape: (batch, seq_len) - IDs de alarma
    After embedding: (batch, seq_len, embedding_dim)
    Output: (batch, n_outputs)
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_size: int, n_outputs: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, n_outputs)
    
    def forward(self, x):
        # x: (batch, seq_len) - alarm IDs
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        _, (h, _) = self.lstm(embedded)
        return self.head(h[-1])


class OnlineMultiLabelLSTM(base.MultiLabelClassifier):
    """Wrapper minimalista para online multi-label learning.
    
    Cambios clave vs PyTorch batch training:
    - reduction='sum' (no 'mean')
    - gradient_scale=10.0
    - SGD con lr=1.0
    
    Input: {0: alarm_id_0, 1: alarm_id_1, ..., 9: alarm_id_9}
    - seq_len se infiere automáticamente de la primera muestra
    - labels dinámicos (se añaden según aparecen en y)
    - num_embeddings fijo a 155 (154 alarmas + padding)
    """
    
    NUM_EMBEDDINGS = 155  # 154 alarmas distintas + 1 para padding (0)
    
    def __init__(self, embedding_dim: int = 8, hidden_size: int = 10,
                 learning_rate: float = 1.0, gradient_scale: float = 10.0):
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gradient_scale = gradient_scale
        
        self._labels = []
        self._label_set = set()
        self._seq_len = None
        self._model =  OnlineLSTM(self.NUM_EMBEDDINGS, self.embedding_dim, 
                                  self.hidden_size, n_labels)
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=self.learning_rate)
    
    def _init_model(self, n_labels: int):
        self._model = OnlineLSTM(self.NUM_EMBEDDINGS, self.embedding_dim, 
                                  self.hidden_size, n_labels)
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=self.learning_rate)
    
    def learn_one(self, x: dict, y: dict):
        # Inferir seq_len de la primera muestra
        if self._seq_len is None:
            self._seq_len = len(x)

        for label in y:
            if label not in self._label_set:
                self._label_set.add(label)
                self._labels.append(label)
                #self._init_model(len(self._labels))
        
        if self._model is None:
            return self
        
        # Extraer alarm IDs (clamp a num_embeddings - 1)
        alarm_ids = [min(int(x.get(i, 0)), self.NUM_EMBEDDINGS - 1) for i in range(self._seq_len)]
        
        # Preparar tensores
        x_t = torch.tensor([alarm_ids], dtype=torch.long)  # (1, seq_len)
        y_t = torch.tensor([[1.0 if y.get(l, False) else 0.0 for l in self._labels]])
        
        # Forward
        logits = self._model(x_t)
        
        # === CAMBIO 1: reduction='sum' ===
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_t, reduction='sum')
        
        # Backward
        self._optimizer.zero_grad()
        loss.backward()
        
        # === CAMBIO 2: gradient scaling ===
        for p in self._model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(self.gradient_scale)
        
        # === CAMBIO 3: SGD con LR alto (definido en __init__) ===
        self._optimizer.step()
        
        return self
    
    def predict_one(self, x: dict) -> Dict[str, bool]:
        if self._model is None:
            return {}
        with torch.no_grad():
            alarm_ids = [min(int(x.get(i, 0)), self.NUM_EMBEDDINGS - 1) for i in range(self._seq_len)]
            x_t = torch.tensor([alarm_ids], dtype=torch.long)
            probs = torch.sigmoid(self._model(x_t)).squeeze()
        return {l: probs[i].item() > 0.5 for i, l in enumerate(self._labels)}


# Test rápido
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    

    
    from datasets.multioutput.newalpi import NewAlpi
    from river.metrics import F1
    from custommetrics.multioutput import MacroAverage, MicroAverage
    from river.metrics.multioutput import ExactMatch
    from river.metrics.base import Metrics
    
    print("=== Test OnlineMultiLabelLSTM con Embeddings ===\n")
    print("Input: {0: 139, 1: 98, ...} -> alarm IDs")
    print("Shape: (1, 10) -> embedding -> (1, 10, 8) -> LSTM -> (1, n_labels)\n")
    
    stream = NewAlpi(machine=4, input_win=1720, output_win=480, delta=0, sigma=120, min_count=0)
    model = OnlineMultiLabelLSTM(
        embedding_dim=8, 
        hidden_size=10, 
        learning_rate=1.0, 
        gradient_scale=10.0
    )
    
    metrics_result = evaluate.progressive_val_score(
        dataset=stream,
        model=model,
        metric=Metrics([ExactMatch(), MacroAverage(F1()), MicroAverage(F1())]),
        show_memory=False,
        print_every=500
        )
    


    print(f"Exact Match: {metrics_result[0].get()*100:.1f}%")
    print(f"MacroF1: {metrics_result[1].get()*100:.1f}%")
    print(f"MicroF1: {metrics_result[2].get()*100:.1f}%")
