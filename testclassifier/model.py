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
class CNN_MultiLabel(nn.Module):
    """
    CNN (Convolutional Neural Network) para clasificación temporal multi-label.
    
    Compatible con DirectMultiLabelForecaster.
    Input esperado: (batch, seq_len, features)
    Procesa la dimensión temporal usando Conv1d.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 kernel_size=3, dropout=0.3, **kwargs):
        """
        Args:
            input_dim: Número de features por timestep
            hidden_dim: Número de filtros (canales) en la convolución
            output_dim: Número de etiquetas
            kernel_size: Tamaño de la ventana de convolución
        """
        super().__init__()
        
        # Convolutional Block
        # Conv1d espera (batch, features, seq_len)
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=hidden_dim, 
            kernel_size=kernel_size, 
            padding=kernel_size//2  # Same padding approx
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Pooling to reduce temporal dimension to 1
        # Esto extrae la característica más prominente de toda la secuencia
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Fully Connected Block
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features)
        """
        # Permutar para Conv1d: (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # Conv -> ReLU -> Dropout
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Pooling: (batch, hidden_dim, 1)
        x = self.pool(x)
        
        # Flatten: (batch, hidden_dim)
        x = x.squeeze(-1)
        
        # FC Layer
        logits = self.fc(x)
        
        return logits


# ==========================
# 6. Adaptador para Tratar Características como Secuencia
# ==========================
class FeatureSequenceLSTM(nn.Module):
    """
    Adaptador que trata las características de una sola instancia como una secuencia temporal.
    
    Input esperado: (batch, 1, n_features)
    Output: (batch, n_features, 1) -> LSTM -> (batch, output_dim)
    """
    def __init__(self, input_dim, output_dim, **kwargs):
        """
        Args:
            input_dim: Se ignora (es el número de features)
            output_dim: Número de etiquetas
            **kwargs: Parámetros para LSTM_MultiLabel (hidden_dim, num_layers, etc.)
        """
        super().__init__()
        # Para el LSTM interno, cada 'timestep' tiene dimensión 1
        self.lstm = LSTM_MultiLabel(input_dim=1, output_dim=output_dim, **kwargs)
        
    def forward(self, x):
        """
        Args:
            x: (batch, 1, n_features)
        """
        # Permutar de (batch, 1, n_features) a (batch, n_features, 1)
        x = x.permute(0, 2, 1)
        return self.lstm(x)


# ==========================
# 7. Modelo LSTM con Embedding Generico
# ==========================
class FeatureSequenceEmbeddingLSTM(nn.Module):
    """
    Combina la lógica de FeatureSequece (tratar features como pasos temporales)
    con una capa de Embedding para entradas categóricas.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_embeddings=256, embedding_dim=64, 
                 num_layers=2, dropout=0.3, bidirectional=True, **kwargs):
        super().__init__()
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        
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
        # RollingClassifier envía (batch, 1, n_features)
        # Queremos tratar n_features como la secuencia temporal
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1) # (batch, n_features)
        
        x = x.long()
        # (batch, seq_len=n_features, embedding_dim)
        embedded = self.embedding(x)
        
        _, (hn, _) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
            last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            last_hidden = hn[-1]
            
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits

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

class StatefulAlpiOneHotLSTM(AlpiOneHotLSTM):
    """
    Version Stateful de AlpiOneHotLSTM.
    Acepta hidden state (hx) en forward y retorna (logits, (hn, cn)).
    """
    def forward(self, x, hx=None):
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        
        x = x.long()
        # Convertimos a One-Hot: (batch, seq_len, num_alarms)
        encoded = F.one_hot(x, num_classes=self.num_alarms).float()
        
        output, (hn, cn) = self.lstm(encoded, hx)
        
        if self.lstm.bidirectional:
            last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            last_hidden = hn[-1]
            
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        
        return logits, (hn, cn)

# ==========================
# 8. Componentes Transformer (Inspirado en FORMULA)
# ==========================

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class Transformer_MultiLabel(nn.Module):
    """
    Implementación PyTorch del modelo TRM del paper.
    """
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_alarms=256, maxlen=100, embed_dim=64, 
                 num_heads=2, ff_dim=128, dropout=0.5, **kwargs):
        super().__init__()
        self.token_emb = nn.Embedding(num_alarms, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, embed_dim)
        
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate=dropout)
        
        # MLPs finales
        self.fn_dim = hidden_dim # Usamos hidden_dim como nn_dim del paper
        self.dropout = nn.Dropout(dropout)
        
        self.bn1 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.fn_dim)
        self.relu = nn.ReLU()
        
        self.bn2 = nn.LayerNorm(self.fn_dim)
        self.fc2 = nn.Linear(self.fn_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) o (batch, 1, n_features)
        """
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        
        x = x.long()
        batch_size, seq_len = x.size()
        
        # Positions
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # Embedding
        x = self.token_emb(x) + self.pos_emb(positions)
        
        # Transformer
        x = self.transformer_block(x)
        
        # Global Average Pooling
        x = x.mean(dim=1)
        
        # Final Layers
        x = self.dropout(x)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        x = self.bn2(x)
        logits = self.fc2(x)
        
        return logits

# ==========================
# 9. Static Weighted Focal Loss
# ==========================

class StaticWeightedFocalLoss(nn.Module):
    """
    Focal Loss con pesos (alpha) estáticos por clase.
    """
    def __init__(self, alphas, gamma=2.0, reduction='mean'):
        """
        Args:
            alphas: tensor de forma (num_classes,) con el factor alpha para cada clase
            gamma: factor de enfoque
        """
        super().__init__()
        if not isinstance(alphas, torch.Tensor):
            alphas = torch.tensor(alphas, dtype=torch.float32)
        self.register_buffer('alphas', alphas)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # BCE
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Alpha balancing per class
        # alphas es (num_classes,), targets es (batch, num_classes)
        # alpha_t = alpha si target=1, 1-alpha si target=0
        alpha_t = self.alphas * targets + (1 - self.alphas) * (1 - targets)
        
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
class AlpiEmbeddingCNN(nn.Module):
    """
    CNN 1D sobre la secuencia de embeddings de alarmas.
    Busca patrones locales (motivos) de alarmas.
    """
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_alarms=256, embedding_dim=32,
                 kernel_size=3, dropout=0.3, **kwargs):
        super().__init__()
        
        self.embedding = nn.Embedding(num_alarms, embedding_dim, padding_idx=0)
        
        # Conv1d espera channels en dim 1
        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size//2
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Max Pooling global sobre toda la secuencia
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
            
        x = x.long()
        # (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # Permutar para Conv1d: (batch, embed_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)
        
        x = self.conv1(embedded)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Pooling: (batch, hidden_dim, 1)
        x = self.pool(x)
        x = x.squeeze(-1)
        
        logits = self.fc(x)
        return logits


# ==========================
# 11. AlpiAttentionLSTM (LSTM with Self-Attention)
# ==========================
class AlpiAttentionLSTM(nn.Module):
    """
    LSTM sobre embeddings de alarmas con mecanismo de Attention.
    Permite al modelo ponderar qué alarmas de la secuencia son más relevantes.
    """
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_alarms=256, embedding_dim=32,
                 num_layers=2, dropout=0.3, bidirectional=True, **kwargs):
        super().__init__()
        
        self.embedding = nn.Embedding(num_alarms, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        direction_factor = 2 if bidirectional else 1
        self.hidden_dim_total = hidden_dim * direction_factor
        
        # Attention Layer
        # Proyectamos el estado oculto a un score de atención
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim_total, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_dim_total, output_dim)
        
    def forward(self, x):
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
            
        x = x.long()
        embedded = self.embedding(x)
        
        # output: (batch, seq_len, hidden_dim*dirs)
        lstm_out, _ = self.lstm(embedded)
        
        # Calcular scores de atención para cada timestep
        # (batch, seq_len, 1)
        attn_scores = self.attention(lstm_out)
        
        # Softmax sobre la dimensión temporal (seq_len, dim=1)
        # Necesitamos manejar masking si hay padding, pero aquí asumimos sequences densas (o padding=0 handled by learning)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Suma ponderada: (batch, hidden_dim*dirs)
        # Multiplicamos pesos (broadcast) por output y sumamos en dim 1
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        
        out = self.dropout(context_vector)
        logits = self.fc(out)
        
        return logits

# ==========================
# 12. AlpiEmbeddingMLP (Global Average Pooling MLP)
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
class AlpiEmbeddingGRU(nn.Module):
    """
    Variante de AlpiEmbeddingLSTM usando GRU en lugar de LSTM.
    Replica exactamente el modelo 'REC' de la literatura (Gated Recurrent Unit).
    
    Diferencias con LSTM:
    - GRU tiene menos parámetros (no cell state, solo hidden state).
    - A veces entrena más rápido y con resultados similares.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_alarms=256, embedding_dim=32, 
                 num_layers=2, dropout=0.3, bidirectional=True, **kwargs):
        super().__init__()
        
        self.embedding = nn.Embedding(num_alarms, embedding_dim, padding_idx=0)
        
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, output_dim)
        
    def forward(self, x):
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1) # (batch, n_features)
        
        x = x.long()
        embedded = self.embedding(x)
        
        # GRU devuelve: output, hn
        # (NO devuelve cn como el LSTM)
        _, hn = self.gru(embedded)
        
        if self.gru.bidirectional:
            # Concatenar las dos direcciones de la última capa
            # hn shape: (num_layers * num_directions, batch, hidden_size)
            last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            last_hidden = hn[-1]
            
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits
