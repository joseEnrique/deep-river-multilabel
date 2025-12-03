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
# 2. Adaptive Weighted BCE (Online Balancing)
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