"""
Módulo LSTM mejorado para clasificación binaria con RollingClassifier.

Inspirado en MultiLabelLSTM pero adaptado para One-vs-Rest:
- Atención temporal para capturar timesteps importantes
- LayerNorm para estabilizar entrenamiento
- Arquitectura más profunda
- Compatible con deep_river.classification.RollingClassifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


#%%
# ==========================
# 1. Focal Loss para Desbalance Extremo
# ==========================
class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación multi-label con desbalance extremo.

    Focal Loss = -α(1-p)^γ * log(p)

    Donde:
    - α (alpha): Factor de balance de clases [0,1]. Default 0.25
    - γ (gamma): Factor de enfoque [0,5]. Valores altos dan más peso a casos difíciles

    Para desbalance extremo (0.19% positivos):
    - gamma=2: Reduce peso de casos fáciles en ~100x
    - gamma=3: Reduce peso de casos fáciles en ~1000x
    - gamma=4: Reduce peso de casos fáciles en ~10000x

    Referencias:
    - Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    - https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Factor de balance para clase positiva [0,1]. Default 0.25
            gamma: Factor de enfoque para casos difíciles [0,5]. Default 2.0
            reduction: 'mean', 'sum' o 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes) - salida del modelo sin sigmoid
            targets: (batch_size, num_classes) - labels binarios [0,1]

        Returns:
            loss: escalar si reduction='mean', tensor si reduction='none'
        """
        # Aplicar sigmoid para obtener probabilidades
        probs = torch.sigmoid(logits)

        # Para cada ejemplo:
        # - Si target=1: p_t = probs (queremos que sea alto)
        # - Si target=0: p_t = 1-probs (queremos que sea alto)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma

        # BCE loss: -log(p_t)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Alpha balancing
        # Si alpha es scalar/tensor:
        # alpha_t = alpha si target=1, 1-alpha si target=0
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_term * bce_loss
        else:
            focal_loss = focal_term * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==========================
# 2. Weighted Focal Loss (WFL)
# ==========================
class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss (WFL) implementada según la petición del usuario.
    
    Fórmula: WFL(pt, beta_t, gamma) = -beta_t * (1 - pt)^gamma * log(pt)
    
    Componentes:
    - Focal Loss (FL): (1 - pt)^gamma. Reduce la contribución de ejemplos fáciles.
    - Weighted Cross Entropy (WCE): beta_t. Compensa la diferencia de frecuencia entre clases.
    
    Args:
        beta (float): Peso fijo para la clase positiva (equivalente a alpha en otras implementaciones).
                      Si beta > 0.5, se penalizan más los falsos negativos (favorece Recall).
        gamma (float): Factor de enfoque (focusing parameter). Default 2.0.
    """
    def __init__(self, beta=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size, num_classes)
        """
        probs = torch.sigmoid(logits)
        
        # p_t: probabilidad de la clase verdadera
        # Si target=1, p_t = probs
        # Si target=0, p_t = 1 - probs
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # beta_t: peso para la clase verdadera
        # Si target=1, beta_t = beta
        # Si target=0, beta_t = 1 - beta
        beta_t = self.beta * targets + (1 - self.beta) * (1 - targets)
        
        # Focal Term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # BCE Loss part: -log(p_t)
        # Usamos binary_cross_entropy_with_logits para estabilidad numérica, 
        # pero necesitamos aplicar los factores nosotros mismos o usar reduction='none'
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # WFL = beta_t * focal_term * bce_loss
        loss = beta_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ==========================
# 2b. Adaptive Weighted BCE (Online Balancing) - Renamed to avoid collision if needed, but keeping index
# ==========================
class AdaptiveWeightedBCE(nn.Module):
    """
    BCE Loss con pesos adaptativos para streaming.
    
    Mantiene un conteo online de positivos y negativos para ajustar
    pos_weight dinámicamente.
    
    pos_weight = n_neg / n_pos
    """
    def __init__(self, num_classes=5, decay=0.999, epsilon=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.decay = decay
        self.epsilon = epsilon
        
        # Registramos buffers para que sean parte del estado del modelo pero no parámetros entrenables
        self.register_buffer('running_pos', torch.ones(num_classes))
        self.register_buffer('running_neg', torch.ones(num_classes))
        self.register_buffer('pos_weight', torch.ones(num_classes))
        
    def forward(self, logits, targets):
        if self.training:
            with torch.no_grad():
                # Calcular positivos y negativos en el batch actual
                batch_pos = targets.sum(dim=0)
                batch_neg = (1 - targets).sum(dim=0)
                
                # Actualizar estadísticas con media móvil exponencial
                self.running_pos = self.decay * self.running_pos + (1 - self.decay) * batch_pos
                self.running_neg = self.decay * self.running_neg + (1 - self.decay) * batch_neg
                
                # Calcular nuevo peso: n_neg / n_pos
                # Añadimos epsilon para evitar división por cero
                self.pos_weight = (self.running_neg + self.epsilon) / (self.running_pos + self.epsilon)
                
                # Clampear pesos para evitar inestabilidad extrema
                self.pos_weight = torch.clamp(self.pos_weight, min=1.0, max=100.0)

        # Calcular BCE con los pesos actuales
        # F.binary_cross_entropy_with_logits no acepta pos_weight por batch, 
        # pero aquí pos_weight es por clase (broadcastable)
        return F.binary_cross_entropy_with_logits(
            logits, 
            targets, 
            pos_weight=self.pos_weight
        )

#%%
# ==========================
# 3. Modelo LSTM
# ==========================
class LSTM_MultiLabel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        if self.bidirectional:
            last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            last_hidden = hn[-1]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits


#%%
# ==========================
# 4. Modelo MLP Multi-Label
# ==========================
class MLP_MultiLabel(nn.Module):
    """
    MLP (Multi-Layer Perceptron) para clasificación multi-label.

    Arquitectura similar a LSTM_MultiLabel pero para datos tabulares:
    - Capas fully connected con activación ReLU
    - LayerNorm para estabilizar entrenamiento (compatible con batch_size=1)
    - Dropout para regularización
    - Salida con logits para multi-label

    Diseñado para dataset ai4i2020.csv:
    - Input: 7 features (Air temp, Process temp, Rotational speed, Torque, Tool wear, Type_encoded)
    - Output: 5 labels (TWF, HDF, PWF, OSF, RNF)
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=5, dropout=0.3):
        """
        Args:
            input_dim: Dimensión de entrada (número de features)
            hidden_dims: Lista con dimensiones de capas ocultas
            output_dim: Dimensión de salida (número de labels)
            dropout: Tasa de dropout para regularización
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Construir capas ocultas
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # No usar normalización con batch_size=1 en streaming
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Capa de salida
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass del MLP.

        Args:
            x: Tensor de entrada con shape (batch_size, input_dim) o (batch_size, seq_len, input_dim)

        Returns:
            logits: Tensor de salida con shape (batch_size, output_dim)
        """
        # Si viene con dimensión de secuencia (batch, seq, features), tomar el último timestep
        if x.dim() == 3:
            x = x[:, -1, :]  # Tomar último timestep

        logits = self.network(x)
        return logits


#%%
# ==========================
# 5. Modelo CNN Multi-Label
# ==========================
class AlpiEmbeddingLSTM(nn.Module):
    """
    LSTM diseñado específicamente para el dataset ALPI:
    - Usa Embedding para tratar los IDs de alarmas como categorías.
    - Maneja la secuencia temporal de alarmas.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_alarms=256, embedding_dim=32, 
                 num_layers=2, dropout=0.3, bidirectional=True, **kwargs):
        """
        Args:
            input_dim: Ignorado (se usa num_alarms para el embedding)
            hidden_dim: Dimensión oculta del LSTM
            output_dim: Número de etiquetas de salida
            num_alarms: Rango máximo de IDs de alarmas (default 256)
            embedding_dim: Tamaño del vector de embedding
        """
        super().__init__()
        
        # padding_idx=0 permite que la red ignore los ceros de relleno
        self.embedding = nn.Embedding(num_alarms, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, output_dim)
        
    def forward(self, x):
        """
            x: (batch, 1, n_features) o (batch, n_features)
        """
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        
        x = x.long()
        embedded = self.embedding(x)
        
        _, (hn, _) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
            last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            last_hidden = hn[-1]
            
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits

class StatefulAlpiEmbeddingLSTM(AlpiEmbeddingLSTM):
    """
    Version Stateful de AlpiEmbeddingLSTM.
    Acepta hidden state (hx) en forward y retorna (logits, (hn, cn)).
    """
    def forward(self, x, hx=None):
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        
        x = x.long()
        embedded = self.embedding(x)
        
        output, (hn, cn) = self.lstm(embedded, hx)
        
        if self.lstm.bidirectional:
            last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            last_hidden = hn[-1]
            
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        
        return logits, (hn, cn)

class AlpiOneHotLSTM(nn.Module):
    """
    LSTM que utiliza One-Hot encoding en lugar de Embeddings.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_alarms=256,
                 num_layers=2, dropout=0.3, bidirectional=True, **kwargs):
        super().__init__()
        self.num_alarms = num_alarms
        
        # En One-Hot, el input_dim del LSTM es el número de posibles alarmas
        self.lstm = nn.LSTM(
            num_alarms, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, output_dim)

    def forward(self, x):
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        
        x = x.long()
        # Convertimos a One-Hot: (batch, seq_len, num_alarms)
        encoded = F.one_hot(x, num_classes=self.num_alarms).float()
        
        _, (hn, _) = self.lstm(encoded)
        
        if self.lstm.bidirectional:
            last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            last_hidden = hn[-1]
            
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits

class AdaptiveWeightedFocalLoss(nn.Module):
    """
    Adaptive Weighted Focal Loss with "Focus Boost" Strategy (Recall-Based).
    
    Dynamically adjusts the positive class weight (beta) based on the 
    Recall accuracy of each class.
    
    Strategy: Focus Boost
    If a class has low Recall (lots of False Negatives), we boost its weight
    to force the model to focus on it.
    
    beta_t = base_beta * (1 + (1 - Recall_t)^k)
    
    Where:
    - base_beta: Initial balanced weight (e.g., 0.25)
    - Recall_t: EMA of Recall for class t
    - k: boosting power (e.g., 1 or 2)
    
    If Recall is 1.0 -> beta = base_beta
    If Recall is 0.0 -> beta = base_beta * 2
    """
    def __init__(self, gamma=2.0, decay=0.99, base_beta=0.25, epsilon=1e-6, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.decay = decay
        self.base_beta = base_beta
        self.epsilon = epsilon
        self.reduction = reduction
        
        # Register buffers for running stats (TP and FN for Recall)
        self.register_buffer('running_tp', torch.ones(1)) 
        self.register_buffer('running_fn', torch.ones(1))
        self.num_classes = None

    def _update_stats(self, logits, targets):
        # logits: (batch, num_classes)
        # targets: (batch, num_classes)
        
        if self.num_classes is None:
            self.num_classes = targets.shape[1]
            self.running_tp = torch.ones(self.num_classes, device=targets.device)
            self.running_fn = torch.ones(self.num_classes, device=targets.device)
            
        if self.running_tp.device != targets.device:
            self.running_tp = self.running_tp.to(targets.device)
            self.running_fn = self.running_fn.to(targets.device)
            
        # Predictions (Threshold 0.5)
        preds = (torch.sigmoid(logits) > 0.5).float()
        
        # Calculate Batch TP and FN
        # TP: Pred=1, Target=1
        batch_tp = (preds * targets).sum(dim=0)
        # FN: Pred=0, Target=1
        batch_fn = ((1 - preds) * targets).sum(dim=0)
        
        # Update EMAs
        self.running_tp = self.decay * self.running_tp + (1 - self.decay) * batch_tp
        self.running_fn = self.decay * self.running_fn + (1 - self.decay) * batch_fn

    def get_logs(self):
        with torch.no_grad():
            recall = self.running_tp / (self.running_tp + self.running_fn + self.epsilon)
            boost = 1.0 + (1.0 - recall)
            beta = self.base_beta * boost
            beta = torch.clamp(beta, max=0.9)
            return {
                "mean_alpha": beta.mean().item(),
                "min_alpha": beta.min().item(),
                "max_alpha": beta.max().item()
            }

    def forward(self, logits, targets):
        if self.training:
            with torch.no_grad():
                self._update_stats(logits, targets)
                
        # Calculate Recall per class
        # Recall = TP / (TP + FN)
        recall = self.running_tp / (self.running_tp + self.running_fn + self.epsilon)
        
        # Calculate Boost
        # If Recall is low, Boost is high (up to 2.0x)
        boost = 1.0 + (1.0 - recall)
        
        # Dynamic Beta
        beta = self.base_beta * boost
        
        # Clamp just in case, though 0.25 * 2 = 0.5 which is safe
        beta = torch.clamp(beta, max=0.9)
        
        probs = torch.sigmoid(logits)
        
        # p_t: probability of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # beta_t: weight for the true class
        # If target=1, beta_t = beta
        # If target=0, beta_t = 1 - beta
        beta_t = beta * targets + (1 - beta) * (1 - targets)
        
        # Focal Term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # WFL
        loss = beta_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
class FullAdaptiveFocalLoss(nn.Module):
    """
    Full Adaptive Focal Loss: Adapts both Alpha and Gamma.
    
    1. Alpha Adaptation (Focus Boost):
       Increases weight for classes with Low Recall (Missed Positives).
       alpha_t = base_alpha * (1 + (1 - Recall_t))
       
    2. Gamma Adaptation (Hardness Focus):
       Increases focusing parameter for classes with Low Accuracy (Hard Classes).
       gamma_t = base_gamma + (1 - Accuracy_t) * 2.0
       
       If a class is easy (Acc=1.0), gamma stays at base_gamma.
       If a class is hard (Acc=0.0), gamma increases to (base + 2.0).
    """
    def __init__(self, base_gamma=2.0, decay=0.999, base_alpha=0.25, alpha_gain=1.0, gamma_gain=2.0, epsilon=1e-6, reduction='mean'):
        super().__init__()
        self.base_gamma = base_gamma
        self.decay = decay
        self.base_alpha = base_alpha
        self.alpha_gain = alpha_gain
        self.gamma_gain = gamma_gain
        self.epsilon = epsilon
        self.reduction = reduction
        
        # Stats for Recall (TP, FN) and Accuracy (TP+TN, Total)
        self.register_buffer('running_tp', torch.ones(1)) 
        self.register_buffer('running_fn', torch.ones(1))
        self.register_buffer('running_correct', torch.ones(1))
        self.register_buffer('running_count', torch.ones(1))
        
        self.num_classes = None

    def _update_stats(self, logits, targets):
        if self.num_classes is None:
            self.num_classes = targets.shape[1]
            self.running_tp = torch.ones(self.num_classes, device=targets.device)
            self.running_fn = torch.ones(self.num_classes, device=targets.device)
            self.running_correct = torch.ones(self.num_classes, device=targets.device)
            self.running_count = torch.ones(self.num_classes, device=targets.device)
            
        if self.running_tp.device != targets.device:
            self.running_tp = self.running_tp.to(targets.device)
            self.running_fn = self.running_fn.to(targets.device)
            self.running_correct = self.running_correct.to(targets.device)
            self.running_count = self.running_count.to(targets.device)
            
        # Predictions
        preds = (torch.sigmoid(logits) > 0.5).float()
        
        # Recall Stats
        batch_tp = (preds * targets).sum(dim=0)
        batch_fn = ((1 - preds) * targets).sum(dim=0)
        
        # Accuracy Stats
        # Correct = (Pred == Target)
        batch_correct = (preds == targets).float().sum(dim=0)
        batch_count = targets.shape[0] # Broad cast later or just scalar addition if same for all? 
        # Actually batch_count is scalar, but we track per class so adding scalar is fine.
        
        # Update EMAs
        self.running_tp = self.decay * self.running_tp + (1 - self.decay) * batch_tp
        self.running_fn = self.decay * self.running_fn + (1 - self.decay) * batch_fn
        self.running_correct = self.decay * self.running_correct + (1 - self.decay) * batch_correct
        self.running_count = self.decay * self.running_count + (1 - self.decay) * batch_count

    def get_logs(self):
        with torch.no_grad():
            # Alpha
            recall = self.running_tp / (self.running_tp + self.running_fn + self.epsilon)
            alpha_boost = 1.0 + (1.0 - recall) * self.alpha_gain
            alpha = self.base_alpha * alpha_boost
            alpha = torch.clamp(alpha, max=0.9)
            
            # Gamma
            accuracy = self.running_correct / (self.running_count + self.epsilon)
            gamma_boost = (1.0 - accuracy) * self.gamma_gain
            gamma = self.base_gamma + gamma_boost
            
            return {
                "mean_alpha": alpha.mean().item(),
                "mean_gamma": gamma.mean().item(),
                "max_gamma": gamma.max().item()
            }

    def forward(self, logits, targets):
        if self.training:
            with torch.no_grad():
                self._update_stats(logits, targets)
        
        # --- 1. Alpha Adaptation (Recall) ---
        recall = self.running_tp / (self.running_tp + self.running_fn + self.epsilon)
        alpha_boost = 1.0 + (1.0 - recall) * self.alpha_gain
        alpha = self.base_alpha * alpha_boost
        # Clamp alpha
        alpha = torch.clamp(alpha, max=0.9)
        
        # --- 2. Gamma Adaptation (Accuracy) ---
        accuracy = self.running_correct / (self.running_count + self.epsilon)
        # Low Accuracy -> Increase Gamma
        # Boost up to base + gamma_gain
        gamma_boost = (1.0 - accuracy) * self.gamma_gain
        gamma = self.base_gamma + gamma_boost
        
        # --- Loss Calculation ---
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Alpha Term
        # If target=1, alpha_t = alpha
        # If target=0, alpha_t = 1 - alpha
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        
        # Focal Term: (1 - p_t)^gamma
        # Note: gamma is a tensor of shape (num_classes,)
        # p_t is (batch, num_classes)
        # We need to broadcast gamma
        gamma_broadcast = gamma.unsqueeze(0) # (1, num_classes)
        focal_term = (1 - p_t) ** gamma_broadcast
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ==========================
# 10. AlpiEmbeddingCNN (1D CNN for Alarm Sequences)
# ==========================
class AlpiEmbeddingMLP(nn.Module):
    """
    MLP que usa Global Average Pooling sobre la secuencia de embeddings.
    Es un 'Deep Averaging Network' (DAN), robusto a la longitud de la secuencia.
    """
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_alarms=256, embedding_dim=32,
                 maxlen=100, dropout=0.3, **kwargs):
        super().__init__()
        
        self.embedding = nn.Embedding(num_alarms, embedding_dim, padding_idx=0)
        
        # Usamos Global Average Pooling -> Entrada al MLP es embedding_dim
        # Esto hace al modelo invariante a la longitud de secuencia (maxlen se ignora)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
            
        x = x.long()
        # (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # Permutar para Pooling: (batch, embed_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)
        
        # Global Avg Pool: (batch, embed_dim, 1)
        pooled = self.pool(embedded)
        
        # Flatten: (batch, embed_dim)
        x = pooled.squeeze(-1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits

# ==========================
# 13. AlpiEmbeddingGRU (REC Variant)
# ==========================
class BidirectionalAdaptiveFocalLoss(nn.Module):
    """
    Bidirectional Adaptive Focal Loss.
    
    Dynamically adjusts Alpha and Gamma based on a Target Metric (Recall/Accuracy).
    
    Mechanism:
    - If Metric < Target: INCREASE parameter (Focus more on this class).
    - If Metric > Target: DECREASE parameter (Relax focus to avoid false positives).
    
    Formulas:
    - alpha_t = base_alpha * (1 + alpha_gain * (target_recall - recall_t))
    - gamma_t = base_gamma + gamma_gain * (target_accuracy - accuracy_t)
    
    Args:
        base_gamma: Initial gamma (def 2.0)
        base_alpha: Initial alpha (def 0.25)
        target_recall: Desired recall (e.g. 0.75). 
                       If recall < 0.75, alpha increases.
                       If recall > 0.75, alpha decreases.
        target_accuracy: Desired accuracy (e.g. 0.95).
        decay: EMA decay for stats (def 0.99)
    """
    def __init__(self, base_gamma=2.0, base_alpha=0.25, 
                 target_recall=0.75, target_accuracy=0.95,
                 alpha_gain=1.0, gamma_gain=2.0, 
                 decay=0.99, epsilon=1e-6, reduction='mean'):
        super().__init__()
        self.base_gamma = base_gamma
        self.base_alpha = base_alpha
        self.target_recall = target_recall
        self.target_accuracy = target_accuracy
        self.alpha_gain = alpha_gain
        self.gamma_gain = gamma_gain
        self.decay = decay
        self.epsilon = epsilon
        self.reduction = reduction
        
        # Stats
        self.register_buffer('running_tp', torch.ones(1)) 
        self.register_buffer('running_fn', torch.ones(1)) # Missed Positives
        self.register_buffer('running_fp', torch.ones(1)) # False Positives (Extra check)
        self.register_buffer('running_correct', torch.ones(1))
        self.register_buffer('running_count', torch.ones(1))
        
        self.num_classes = None

    def _update_stats(self, logits, targets):
        if self.num_classes is None:
            self.num_classes = targets.shape[1]
            device = targets.device
            self.running_tp = torch.ones(self.num_classes, device=device)
            self.running_fn = torch.ones(self.num_classes, device=device)
            self.running_fp = torch.ones(self.num_classes, device=device)
            self.running_correct = torch.ones(self.num_classes, device=device)
            self.running_count = torch.ones(self.num_classes, device=device)
            
        if self.running_tp.device != targets.device:
            self.running_tp = self.running_tp.to(targets.device)
            self.running_fn = self.running_fn.to(targets.device)
            self.running_fp = self.running_fp.to(targets.device)
            self.running_correct = self.running_correct.to(targets.device)
            self.running_count = self.running_count.to(targets.device)
            
        preds = (torch.sigmoid(logits) > 0.5).float()
        
        batch_tp = (preds * targets).sum(dim=0)
        batch_fn = ((1 - preds) * targets).sum(dim=0)
        batch_fp = (preds * (1 - targets)).sum(dim=0)
        batch_correct = (preds == targets).float().sum(dim=0)
        batch_count = targets.shape[0]
        
        self.running_tp = self.decay * self.running_tp + (1 - self.decay) * batch_tp
        self.running_fn = self.decay * self.running_fn + (1 - self.decay) * batch_fn
        self.running_fp = self.decay * self.running_fp + (1 - self.decay) * batch_fp
        self.running_correct = self.decay * self.running_correct + (1 - self.decay) * batch_correct
        self.running_count = self.decay * self.running_count + (1 - self.decay) * batch_count

    def get_logs(self):
        with torch.no_grad():
            # Recall
            recall = self.running_tp / (self.running_tp + self.running_fn + self.epsilon)
            # Alpha Delta: (Target - Actual) -> Positive if Recall is low
            alpha_delta = self.target_recall - recall
            alpha_factor = 1.0 + (alpha_delta * self.alpha_gain)
            alpha = self.base_alpha * alpha_factor
            alpha = torch.clamp(alpha, min=0.01, max=0.99)
            
            # Accuracy
            acc = self.running_correct / (self.running_count + self.epsilon)
            # Gamma Delta: (Target - Actual) -> Positive if Acc is low
            gamma_delta = self.target_accuracy - acc
            gamma = self.base_gamma + (gamma_delta * self.gamma_gain)
            gamma = torch.clamp(gamma, min=0.0, max=10.0)
            
            return {
                "mean_alpha": alpha.mean().item(),
                "mean_gamma": gamma.mean().item(),
                "mean_recall": recall.mean().item(),
                "mean_acc": acc.mean().item()
            }

    def forward(self, logits, targets):
        if self.training:
            with torch.no_grad():
                self._update_stats(logits, targets)
        
        # --- Alpha Adaptation (Recall Based) ---
        recall = self.running_tp / (self.running_tp + self.running_fn + self.epsilon)
        # If Recall < Target -> (Target - Recall) > 0 -> Increase Alpha
        # If Recall > Target -> (Target - Recall) < 0 -> Decrease Alpha
        alpha_delta = self.target_recall - recall
        alpha_factor = 1.0 + (alpha_delta * self.alpha_gain)
        alpha = self.base_alpha * alpha_factor
        alpha = torch.clamp(alpha, min=0.01, max=0.99)
        
        # --- Gamma Adaptation (Accuracy Based) ---
        acc = self.running_correct / (self.running_count + self.epsilon)
        # If Acc < Target -> Increase Gamma
        gamma_delta = self.target_accuracy - acc
        gamma = self.base_gamma + (gamma_delta * self.gamma_gain)
        gamma = torch.clamp(gamma, min=0.0, max=10.0)
        
        # --- Loss Calc ---
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Alpha term
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        
        # Focal Term: (1 - p_t)^gamma (Broadcasting gamma)
        gamma_broadcast = gamma.unsqueeze(0)
        focal_term = (1 - p_t) ** gamma_broadcast
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ==========================
# 15. Learnable Focal Loss (Gradient Descent)
# ==========================
class LearnableFocalLoss(nn.Module):
    """
    Learnable Focal Loss: Alpha and Gamma are parameters trained by the optimizer.
    """
    def __init__(self, init_gamma=2.0, init_alpha=0.25, reduction='mean'):
        super().__init__()
        import math
        # Initialize logits for alpha so sigmoid(logit) = init_alpha
        # sigmoid(x) = p => x = log(p / (1-p))
        init_alpha = float(init_alpha)
        self.param_alpha = nn.Parameter(torch.tensor(math.log(init_alpha / (1.0 - init_alpha))))
        
        # Initialize param for gamma so softplus(param) = init_gamma
        # softplus(x) = log(1 + exp(x)) => x = log(exp(gamma) - 1)
        init_gamma = float(init_gamma)
        self.param_gamma = nn.Parameter(torch.log(torch.exp(torch.tensor(init_gamma)) - 1))
        
        self.reduction = reduction

    def get_logs(self):
        with torch.no_grad():
            alpha = torch.sigmoid(self.param_alpha)
            gamma = F.softplus(self.param_gamma)
            return {
                "mean_alpha": alpha.item(),
                "mean_gamma": gamma.item(),
            }

    def forward(self, logits, targets):
        # Constrain parameters
        alpha = torch.sigmoid(self.param_alpha)
        gamma = F.softplus(self.param_gamma)
        
        # --- Loss Calculation ---
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Alpha balancing
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        
        # Focal Term
        focal_term = (1 - p_t) ** gamma
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ==========================
# 16. Self-Tuning Focal Loss (Parameter Free)
# ==========================
class RobustFocalLoss(nn.Module):
    """
    Robust Focal Loss: Auto-scales gain based on batch volatility and anchors parameters.
    Designed to work with High Learning Rates.
    
    Mechanisms:
    1. Momentum-based Adaptation: Uses momentum instead of simple EMA for smoothness.
    2. Gain Scaling: If batch statistics oscillate, effective gain is reduced.
    3. Anchor: Pulls Alpha/Gamma towards baseline (0.25, 2.0) to prevent collapse.
    """
    def __init__(self, base_gamma=2.0, base_alpha=0.25, 
                 momentum=0.9, 
                 max_gain=2.0, 
                 anchor_weight=0.1, # Weight of the anchor pull
                 reduction='mean'):
        super().__init__()
        self.base_gamma = base_gamma
        self.base_alpha = base_alpha
        self.momentum = momentum
        self.max_gain = max_gain
        self.anchor_weight = anchor_weight
        self.reduction = reduction
        self.epsilon = 1e-6
        
        # Current State (Momentum)
        # We use buffers so they are saved with state_dict but not trained by optimizer
        self.register_buffer('alpha_velocity', torch.tensor(0.0))
        self.register_buffer('gamma_velocity', torch.tensor(0.0))
        
        # Current Values (starts at base)
        self.register_buffer('current_alpha', torch.tensor(base_alpha))
        self.register_buffer('current_gamma', torch.tensor(base_gamma))
        
        # Batch Stats
        self.register_buffer('running_tp', torch.ones(1)) 
        self.register_buffer('running_fn', torch.ones(1))
        self.register_buffer('running_correct', torch.ones(1))
        self.register_buffer('running_count', torch.ones(1))

    def _update_stats(self, logits, targets):
        preds = (torch.sigmoid(logits) > 0.5).float()
        
        batch_tp = (preds * targets).sum() # Global sum for stability
        batch_fn = ((1 - preds) * targets).sum()
        batch_correct = (preds == targets).float().sum()
        batch_count = torch.tensor(targets.numel(), device=targets.device)
        
        # 1. Calculate Instant Recall (Global)
        instant_recall = batch_tp / (batch_tp + batch_fn + self.epsilon)
        instant_acc = batch_correct / (batch_count + self.epsilon)
        
        # 2. Calculate Desired Shifts 
        # Target Recall is implicitly 1.0 (we always want better recall)
        # Delta = (1 - Recall) -> Positive. We want to INCREASE Alpha.
        alpha_delta = (1.0 - instant_recall) * 0.05 # Small step
        
        # Target Accuracy is implicitly 1.0
        # Delta = (1 - Acc) -> Positive. We want to INCREASE Gamma (focus on hard).
        gamma_delta = (1.0 - instant_acc) * 0.1 # Small step
        
        # 3. Update Velocities (Momentum)
        # v = m * v + (1-m) * delta
        self.alpha_velocity = self.momentum * self.alpha_velocity + (1 - self.momentum) * alpha_delta
        self.gamma_velocity = self.momentum * self.gamma_velocity + (1 - self.momentum) * gamma_delta
        
        # 4. Update Parameters
        # Alpha += velocity
        new_alpha = self.current_alpha + self.alpha_velocity
        new_gamma = self.current_gamma + self.gamma_velocity
        
        # 5. Anchor Pull (Regularization)
        # Pull towards base values to prevent drift
        # new_val = val * (1-w) + base * w
        new_alpha = new_alpha * (1 - self.anchor_weight) + self.base_alpha * self.anchor_weight
        new_gamma = new_gamma * (1 - self.anchor_weight) + self.base_gamma * self.anchor_weight
        
        # 6. Store with clamping
        self.current_alpha = torch.clamp(new_alpha, min=0.05, max=0.95)
        self.current_gamma = torch.clamp(new_gamma, min=0.5, max=5.0)

    def get_logs(self):
        return {
            "mean_alpha": self.current_alpha.item(),
            "mean_gamma": self.current_gamma.item(),
            "alpha_velocity": self.alpha_velocity.item()
        }

    def forward(self, logits, targets):
        if self.training:
            with torch.no_grad():
                self._update_stats(logits, targets)
        
        alpha = self.current_alpha
        gamma = self.current_gamma
        
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        
        # Broadcasting logic
        # alpha_t is (batch, num_classes)
        # focal_term should be (batch, num_classes)
        # gamma is scalar here (global gamma)
        
        focal_term = (1 - p_t) ** gamma
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

