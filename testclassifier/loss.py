import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================
# 1. Focal Loss para Desbalance Extremo
# ==========================
class FocalLoss(nn.Module):
    """
    Focal Loss original para clasificación con desbalance extremo de clases.
    
    A diferencia de la entropía cruzada binaria (BCE) normal, Focal Loss añade un factor de
    modulación (1 - pt)^gamma al BCE normal con el objetivo de reducir el peso (penalización)
    que tienen los ejemplos fáciles y que el modelo se centre en aprender los ejemplos más difíciles.
    
    Mecanismo:
    - `alpha`: Factor de balanceo para la clase positiva (ej: 0.25 penaliza menos los Falso Positivos).
    - `gamma`: Parámetro de enfoque. Cuanto más alto (ej: 2.0 o 3.0), más ignora el modelo los
               ejemplos en los que ya está seguro y se concentra en aquellos en los que falla.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
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
    Focal Loss Ponderada (Weighted Focal Loss) para lidiar con el desbalance de clases.
    
    Esta variante combina la penalización de ejemplos asimétricos del Focal Loss original
    con un peso explícito (`beta`) enfocado a priorizar la métrica de Recall sobre Accuracy.
    
    Mecanismo:
    - Utiliza un factor de peso `beta_t` en lugar del 'alpha' tradicional. Si `beta` es alto
      (por ejemplo > 0.5), el modelo penalizará severamente los Falsos Negativos, lo que ayuda
      crucialmente a aumentar el Recall en escenarios de mantenimiento predictivo o alarmas
      críticas donde perderse una anomalía es muy costoso.
    """
    def __init__(self, beta=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        beta_t = self.beta * targets + (1 - self.beta) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = beta_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ==========================
# 2b. Adaptive Weighted BCE (Online Balancing)
# ==========================
class AdaptiveWeightedBCE(nn.Module):
    """
    Entropía Cruzada Binaria (BCE) con Pesos Adaptativos en tiempo real (Online Learning).
    
    En flujos de datos continuos (streaming) altamente desbalanceados, a menudo la proporción real
    de datos positivos/negativos es desconocida o cambia con el tiempo (concept drift).
    
    Mecanismo:
    - Utiliza una Media Móvil Exponencial (EMA) que cuenta los ejemplos positivos y negativos
      que van llegando lote a lote en el tiempo.
    - Calcula dinámicamente un peso para la clase positiva en función de la rareza de dicha clase
      en la ventana pasada, calculándose como: pos_weight = acumulado_negativos / acumulado_positivos.
    - Asegura que el modelo siempre le dé importancia proporcional a la minoría dinámicamente.
    """
    def __init__(self, num_classes=5, decay=0.999, epsilon=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.decay = decay
        self.epsilon = epsilon
        
        self.register_buffer('running_pos', torch.ones(num_classes))
        self.register_buffer('running_neg', torch.ones(num_classes))
        self.register_buffer('pos_weight', torch.ones(num_classes))
        
    def forward(self, logits, targets):
        if self.training:
            with torch.no_grad():
                batch_pos = targets.sum(dim=0)
                batch_neg = (1 - targets).sum(dim=0)
                
                self.running_pos = self.decay * self.running_pos + (1 - self.decay) * batch_pos
                self.running_neg = self.decay * self.running_neg + (1 - self.decay) * batch_neg
                
                self.pos_weight = (self.running_neg + self.epsilon) / (self.running_pos + self.epsilon)
                self.pos_weight = torch.clamp(self.pos_weight, min=1.0, max=100.0)

        return F.binary_cross_entropy_with_logits(
            logits, 
            targets, 
            pos_weight=self.pos_weight
        )


class AdaptiveWeightedFocalLoss(nn.Module):
    """
    Focal Loss Adaptativa Basada en Recall Instantáneo.
    
    Esta función de pérdida adapta su peso positivo (`beta`) dependiendo dinámicamente del
    rendimiento (Recall) que esté teniendo el modelo para cada clase específica en streaming.
    
    Mecanismo de "Boost de Enfoque":
    - Si el recall de una clase baja (se están produciendo muchos falsos negativos), la pérdida aumenta
      su parámetro "beta" para forzar al modelo a centrarse más en esa clase minoritaria.
    - Si el recall se acerca a 1.0 (ya está prediciendo bien esa clase), el parámetro "beta" vuelve a
      su valor base (`base_beta`), evitando así sobre-dimensionar los falsos positivos cuando la clase
      ya la está resolviendo de manera eficiente.
    """
    def __init__(self, gamma=2.0, decay=0.99, base_beta=0.25, epsilon=1e-6, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.decay = decay
        self.base_beta = base_beta
        self.epsilon = epsilon
        self.reduction = reduction
        
        self.register_buffer('running_tp', torch.ones(1)) 
        self.register_buffer('running_fn', torch.ones(1))
        self.num_classes = None

    def _update_stats(self, logits, targets):
        if self.num_classes is None:
            self.num_classes = targets.shape[1]
            self.running_tp = torch.ones(self.num_classes, device=targets.device)
            self.running_fn = torch.ones(self.num_classes, device=targets.device)
            
        if self.running_tp.device != targets.device:
            self.running_tp = self.running_tp.to(targets.device)
            self.running_fn = self.running_fn.to(targets.device)
            
        preds = (torch.sigmoid(logits) > 0.5).float()
        batch_tp = (preds * targets).sum(dim=0)
        batch_fn = ((1 - preds) * targets).sum(dim=0)
        
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
                
        recall = self.running_tp / (self.running_tp + self.running_fn + self.epsilon)
        boost = 1.0 + (1.0 - recall)
        beta = self.base_beta * boost
        beta = torch.clamp(beta, max=0.9)
        
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        beta_t = beta * targets + (1 - beta) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = beta_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ==========================
# 9. AdaptiveFocalLoss
# ==========================
class AdaptiveFocalLoss(nn.Module):
    """
    Focal Loss Adaptativa Optimizada para flujos de datos en streaming desequilibrados (F1-Base).
    
    A diferencia de la focal loss convencional (Alpha y Gamma estáticos) o su versión basada
    solo en recall, esta versión ajusta dinámicamente tanto su Alpha como su Gamma observando
    el F1-score acumulado de cada clase, sirviendo de autoajuste regulatorio.
    
    Mecanismo:
    1. Registra Verdaderos Positivos (TP), Falsos Positivos (FP) y Falsos Negativos (FN) temporalmente (EMA).
    2. Modulación del parámetro de peso (Alpha): Si el F1 de la clase es malo, incrementa el Alpha para
       darle más importancia bruta.
    3. Modulación del parámetro de enfoque (Gamma): Si el F1 es malo, incrementa Gamma, obligando al modelo
       a centrarse en los errores duros y no en los fáciles en esa clase en concreto.
    """
    def __init__(self, base_gamma=2.0, decay=0.999, base_alpha=0.25,
                 alpha_gain=1.0, gamma_gain=2.0, epsilon=1e-6, reduction='mean'):
        super().__init__()
        self.base_gamma = base_gamma
        self.decay = decay
        self.base_alpha = base_alpha
        self.alpha_gain = alpha_gain
        self.gamma_gain = gamma_gain
        self.epsilon = epsilon
        self.reduction = reduction
        self.register_buffer('running_tp', torch.ones(1))
        self.register_buffer('running_fp', torch.ones(1))
        self.register_buffer('running_fn', torch.ones(1))
        self.register_buffer('running_count', torch.ones(1))
        self.num_classes = None

    def _init_buffers(self, n_classes, device):
        self.num_classes = n_classes
        self.running_tp = torch.ones(n_classes, device=device)
        self.running_fp = torch.ones(n_classes, device=device)
        self.running_fn = torch.ones(n_classes, device=device)
        self.running_count = torch.ones(n_classes, device=device)

    def _ensure_device(self, device):
        if self.running_tp.device != device:
            self.running_tp = self.running_tp.to(device)
            self.running_fp = self.running_fp.to(device)
            self.running_fn = self.running_fn.to(device)
            self.running_count = self.running_count.to(device)

    def _update_stats(self, logits, targets):
        if self.num_classes is None:
            self._init_buffers(targets.shape[1], targets.device)
        self._ensure_device(targets.device)
        preds = (torch.sigmoid(logits) > 0.5).float()
        d = self.decay
        self.running_tp = d * self.running_tp + (1 - d) * (preds * targets).sum(dim=0)
        self.running_fp = d * self.running_fp + (1 - d) * (preds * (1 - targets)).sum(dim=0)
        self.running_fn = d * self.running_fn + (1 - d) * ((1 - preds) * targets).sum(dim=0)
        self.running_count = d * self.running_count + (1 - d) * targets.shape[0]

    def _f1(self):
        num = 2 * self.running_tp
        den = num + self.running_fp + self.running_fn + self.epsilon
        return num / den

    def get_logs(self):
        with torch.no_grad():
            f1 = self._f1()
            alpha = torch.clamp(self.base_alpha * (1 + (1 - f1) * self.alpha_gain), max=0.9)
            gamma = self.base_gamma + (1 - f1) * self.gamma_gain
            return {"mean_f1": f1.mean().item(), "mean_alpha": alpha.mean().item(),
                    "mean_gamma": gamma.mean().item(), "max_gamma": gamma.max().item()}

    def forward(self, logits, targets):
        if self.num_classes is None:
            self._init_buffers(targets.shape[1], targets.device)
        self._ensure_device(targets.device)
        if self.training:
            with torch.no_grad():
                self._update_stats(logits, targets)

        f1 = self._f1()
        alpha = torch.clamp(self.base_alpha * (1.0 + (1.0 - f1) * self.alpha_gain), max=0.9)
        gamma = self.base_gamma + (1.0 - f1) * self.gamma_gain

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_term = (1 - p_t) ** gamma.unsqueeze(0)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = alpha_t * focal_term * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BidirectionalAdaptiveFocalLoss(nn.Module):
    """
    Focal Loss Adaptativa Bidireccional basada en Objetivos de Rendimiento.
    
    En lugar de simplemente reaccionar si las métricas caen, este paradigma fija como objetivo un
    Recall específico y una Accuracy (Exactitud) deseados definidos de antemano.
    
    Mecanismo Bidireccional:
    - Adaptación hacia ARRIBA (Aumentar esfuerzo): Si la métrica cae por debajo de la objetivo establecida,
      los hiperparámetros de penalización (Alpha/Gamma) suben para corregir el modelo.
    - Adaptación hacia ABAJO (Relajar esfuerzo): Si la métrica supera el objetivo establecido, Alpha/Gamma
      bajan inmediatamente para evitar que la red cree falsos positivos (sobre-optimice esa clase en
      detrimento de las demás o caiga en colapso).
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
        
        self.register_buffer('running_tp', torch.ones(1)) 
        self.register_buffer('running_fn', torch.ones(1))
        self.register_buffer('running_fp', torch.ones(1))
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
            recall = self.running_tp / (self.running_tp + self.running_fn + self.epsilon)
            alpha_delta = self.target_recall - recall
            alpha_factor = 1.0 + (alpha_delta * self.alpha_gain)
            alpha = self.base_alpha * alpha_factor
            alpha = torch.clamp(alpha, min=0.01, max=0.99)
            
            acc = self.running_correct / (self.running_count + self.epsilon)
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
        
        recall = self.running_tp / (self.running_tp + self.running_fn + self.epsilon)
        alpha_delta = self.target_recall - recall
        alpha_factor = 1.0 + (alpha_delta * self.alpha_gain)
        alpha = self.base_alpha * alpha_factor
        alpha = torch.clamp(alpha, min=0.01, max=0.99)
        
        acc = self.running_correct / (self.running_count + self.epsilon)
        gamma_delta = self.target_accuracy - acc
        gamma = self.base_gamma + (gamma_delta * self.gamma_gain)
        gamma = torch.clamp(gamma, min=0.0, max=10.0)
        
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
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


class LearnableFocalLoss(nn.Module):
    """
    Focal Loss Entrenable (Descenso de Gradiente puro) para Alpha y Gamma.
    
    A diferencia de las adaptaciones heurísticas que dependen de métricas contadas a mano (F1, Recall),
    esta capa inyecta los parámetros originales Alpha y Gamma directo al flujo computacional del modelo,
    conviertiéndolos en parámetros (nn.Parameter).
    
    Mecanismo:
    - El propio optimizador del modelo (por ejemplo, Adam) actualiza el peso de balanceo Alpha y
      el factor de enfoque Gamma internamente en backpropagation, para minimizar la pérdida.
    - Se inicializan como logits para evitar que se desvíen fuera de los rangos válidos numéricos.
    """
    def __init__(self, init_gamma=2.0, init_alpha=0.25, reduction='mean'):
        super().__init__()
        import math
        init_alpha = float(init_alpha)
        self.param_alpha = nn.Parameter(torch.tensor(math.log(init_alpha / (1.0 - init_alpha))))
        
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
        alpha = torch.sigmoid(self.param_alpha)
        gamma = F.softplus(self.param_gamma)
        
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_term = (1 - p_t) ** gamma
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class RobustFocalLoss(nn.Module):
    """
    Focal Loss Robusta: Auto-adaptación estabilizada para configuraciones de alto Learning Rate (Streaming).
    
    Adaptar métricas fuertemente en escenarios online (ej: aprendizaje incremental profundo) frecuentemente
    causa problemas de oscilación intensa y colapso de clase. Esta capa soluciona esto introduciendo física.
    
    Mecanismo de "Regulación con Inercia":
    - En lugar de promedios simples, utiliza Momentum (Velocidad) para que las oscilaciones abruptas de un
      batch extremadamente raro de datos no destruyan el modelo de golpe.
    - Fuerza Tensora (Anchor): Aplica siempre un tirón regularizador continuo (`anchor_weight`) que tira
      constantemente de las variables Alpha y Gamma de vuelta hacia sus valores por defecto racionales
      (evitando "parámetros a la deriva" tras horas de streaming).
    """
    def __init__(self, base_gamma=2.0, base_alpha=0.25, 
                 momentum=0.9, 
                 max_gain=2.0, 
                 anchor_weight=0.1,
                 reduction='mean'):
        super().__init__()
        self.base_gamma = base_gamma
        self.base_alpha = base_alpha
        self.momentum = momentum
        self.max_gain = max_gain
        self.anchor_weight = anchor_weight
        self.reduction = reduction
        self.epsilon = 1e-6
        
        self.register_buffer('alpha_velocity', torch.tensor(0.0))
        self.register_buffer('gamma_velocity', torch.tensor(0.0))
        self.register_buffer('current_alpha', torch.tensor(base_alpha))
        self.register_buffer('current_gamma', torch.tensor(base_gamma))
        
        self.register_buffer('running_tp', torch.ones(1)) 
        self.register_buffer('running_fn', torch.ones(1))
        self.register_buffer('running_correct', torch.ones(1))
        self.register_buffer('running_count', torch.ones(1))

    def _update_stats(self, logits, targets):
        preds = (torch.sigmoid(logits) > 0.5).float()
        
        batch_tp = (preds * targets).sum()
        batch_fn = ((1 - preds) * targets).sum()
        batch_correct = (preds == targets).float().sum()
        batch_count = torch.tensor(targets.numel(), device=targets.device)
        
        instant_recall = batch_tp / (batch_tp + batch_fn + self.epsilon)
        instant_acc = batch_correct / (batch_count + self.epsilon)
        
        alpha_delta = (1.0 - instant_recall) * 0.05
        gamma_delta = (1.0 - instant_acc) * 0.1
        
        self.alpha_velocity = self.momentum * self.alpha_velocity + (1 - self.momentum) * alpha_delta
        self.gamma_velocity = self.momentum * self.gamma_velocity + (1 - self.momentum) * gamma_delta
        
        new_alpha = self.current_alpha + self.alpha_velocity
        new_gamma = self.current_gamma + self.gamma_velocity
        
        new_alpha = new_alpha * (1 - self.anchor_weight) + self.base_alpha * self.anchor_weight
        new_gamma = new_gamma * (1 - self.anchor_weight) + self.base_gamma * self.anchor_weight
        
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
        
        focal_term = (1 - p_t) ** gamma
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
