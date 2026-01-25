import torch
import torch.nn as nn

class ExactOnlineLSTM(nn.Module):
    """
    Exact copy of OnlineLSTM from stateful_multilabel_classifier.py
    Adapted to accept input_dim/output_dim for Rolling compatibility.
    Includes Explainability features (Saliency Map).
    """
    def __init__(self, input_dim, output_dim, embedding_dim=8, hidden_size=10, num_embeddings=155):
        super().__init__()
        
        # HACK: strict reproduction of manual script behavior
        # Manual script re-initializes model for every new label found in first sample
        # It creates models with output_dim = 1, 2, ..., N-1 before the final N.
        # We must burn the RNG for these intermediate initializations to match weights.
        with torch.no_grad():
            for i in range(1, output_dim):
                nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
                nn.LSTM(embedding_dim, hidden_size, batch_first=True)
                nn.Linear(hidden_size, i)

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_dim)
        
        self.num_embeddings = num_embeddings

    def forward(self, x):
        # x: (batch, seq_len) - alarm IDs
        # Ensure x is LongTensor
        if x.dtype != torch.long:
            x = x.long()
        
        # In RollingClassifier, x comes as (batch, 1, n_features) if window_size=1
        # We need to squeeze the middle dim if present
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        # EXACT MATCH: Manual script clips values to NUM_EMBEDDINGS - 1
        x = torch.clamp(x, max=self.num_embeddings - 1)
            
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        _, (h, _) = self.lstm(embedded)
        return self.head(h[-1])

    def get_saliency_map(self, x, target_class_idx):
        """
        Compute Saliency Map (Input Gradients) for a specific target class.
        Returns the importance score for each time step in the input sequence.
        """
        # Ensure x is prepared
        if x.dtype != torch.long:
            x = x.long()
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        # 1. Get Embeddings and Retain Gradients
        x_indices = torch.clamp(x, max=self.num_embeddings - 1)
        
        # We need to manually do the embedding step to attach the hook/grad
        embeddings = self.embedding(x_indices)
        embeddings.retain_grad() # Crucial for Saliency
        
        # 2. Forward Pass
        # Only run the rest of the model
        _, (h, _) = self.lstm(embeddings)
        logits = self.head(h[-1])
        
        # 3. Backward Pass for Target Class
        # We want to maximize the score of the target class
        # logits shape: (batch_size, num_classes)
        # We assume batch_size=1 for explainability usually, or we sum.
        
        self.zero_grad()
        target_score = logits[0, target_class_idx]
        target_score.backward()
        
        # 4. Get Gradients from Embeddings
        # grads shape: (batch, seq_len, embedding_dim)
        grads = embeddings.grad
        
        # 5. Compute Saliency (L2 norm across embedding dimension)
        # result shape: (batch, seq_len)
        saliency = torch.norm(grads, dim=2)
        
        # Normalize to [0, 1] for better visualization
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency

# Helper to load model easily
def load_model(path, input_dim, output_dim, embedding_dim=8, hidden_size=10, num_embeddings=155):
    model = ExactOnlineLSTM(input_dim, output_dim, embedding_dim, hidden_size, num_embeddings)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

class ShapWrapper(nn.Module):
    """
    Wrapper model for SHAP.
    Takes embeddings as input (instead of IDs) to allow gradient computation.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, embeddings):
        # embeddings: (batch, seq_len, embedding_dim)
        # We skip the embedding layer of the original model
        _, (h, _) = self.model.lstm(embeddings)
        return self.model.head(h[-1])
