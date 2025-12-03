#%%
# ==========================
# 1. Imports
# ==========================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Configurar seeds para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # Para multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Detectar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%%
class AI4IDataset(Dataset):
    def __init__(self, features, labels, seq_len=10):
        self.seq_len = seq_len
        self.X_seq, self.y_seq = self.create_sequences(features, labels, seq_len)

    def create_sequences(self, features, labels, seq_len):
        X_seq, y_seq = [], []
        for i in range(len(features) - seq_len + 1):
            X_seq.append(features[i:i+seq_len])
            y_seq.append(labels[i+seq_len-1])  # etiqueta del último paso
        return np.array(X_seq), np.array(y_seq)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return torch.tensor(self.X_seq[idx], dtype=torch.float32), \
               torch.tensor(self.y_seq[idx], dtype=torch.float32)
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
# 4. Cargar y preparar datos
# ==========================
df = pd.read_csv("/home/quique/tesis/OEMLHAT4PdM/datasets/ai4i2020formatted.csv")

feature_cols = ["Air temperature [K]","Process temperature [K]",
                "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]",
                "Type_H","Type_L","Type_M"]
label_cols = ["TWF","HDF","PWF","OSF","RNF"]

X = df[feature_cols].values
y = df[label_cols].values

# Normalizar features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split train/test manteniendo orden temporal
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Crear datasets con secuencias
seq_len = 10
train_dataset = AI4IDataset(X_train, y_train, seq_len)
test_dataset  = AI4IDataset(X_test, y_test, seq_len)

# Generator para reproducibilidad en DataLoader
g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=g)
test_loader  = DataLoader(test_dataset, batch_size=32)

#%%
# ==========================
# 5. Entrenar modelo
# ==========================
input_dim = X.shape[1]
hidden_dim = 64
output_dim = y.shape[1]

model = LSTM_MultiLabel(input_dim, hidden_dim, output_dim, dropout=0.2).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
#%%
# ==========================
# 6. Evaluación en TODO el dataset (train + test)
# ==========================
model.eval()
all_probs, all_preds, all_labels = [], [], []

# Crear un DataLoader con TODO el dataset
full_dataset = AI4IDataset(X, y, seq_len)  # Usar X, y originales (todo el dataset)
full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

print("Evaluando en TODO el dataset...")
with torch.no_grad():
    for X_batch, y_batch in full_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(y_batch.cpu())

all_probs = torch.cat(all_probs).numpy()
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

print(f"Total de muestras evaluadas: {len(all_labels)}")
print("\nEjemplo de probabilidades:\n", all_probs[:5])
print("\nEjemplo de predicciones:\n", all_preds[:5])
print("\nEjemplo de etiquetas reales:\n", all_labels[:5])

# Reporte de métricas
print("\n=== Reporte de clasificación (multi-label) - TODO EL DATASET ===")
print(classification_report(all_labels, all_preds, target_names=label_cols, zero_division=0))
#%%
from sklearn.metrics import hamming_loss, f1_score, accuracy_score
import numpy as np

# Aseguramos que son numpy arrays enteros
y_true = all_labels.astype(int)
y_pred = all_preds.astype(int)



#%%
# ==========================
# 7. Evaluación adicional - Prints como en testaidataset.py
# ==========================
import numpy as np

# Convertir a arrays numpy para análisis
all_probs_np = all_probs
all_preds_np = all_preds
all_labels_np = all_labels

print("\n" + "="*60)
print("COMPARACIÓN CON ENFOQUE STREAMING (testaidataset.py):")
print("="*60)

#%%
# ==========================
# 8. Resumen final (formato testaidataset.py)
# ==========================
from sklearn.metrics import precision_score, recall_score

# Calcular agregados para el resumen final
# Accuracy general (micro, por etiqueta)
accuracy_val = accuracy_score(all_labels.reshape(-1), all_preds.reshape(-1))

# Subset accuracy (exact match ratio)
subset_acc = accuracy_score(all_labels, all_preds)

# Hamming loss
hamming_loss_val = hamming_loss(all_labels, all_preds)

# MicroF1 y MacroF1
micro_f1_val = f1_score(all_labels, all_preds, average="micro", zero_division=0)
macro_f1_val = f1_score(all_labels, all_preds, average="macro", zero_division=0)

# Micro/Macro Precision y Recall
micro_prec = precision_score(all_labels, all_preds, average="micro", zero_division=0)
micro_rec = recall_score(all_labels, all_preds, average="micro", zero_division=0)
macro_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
macro_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)

# Example-based custommetrics (samples average)
example_f1 = f1_score(all_labels, all_preds, average="samples", zero_division=0)
example_prec = precision_score(all_labels, all_preds, average="samples", zero_division=0)
example_rec = recall_score(all_labels, all_preds, average="samples", zero_division=0)

# Casos con fallos y predicciones perfectas
total_positive_cases = int((all_labels.sum(axis=1) > 0).sum())
perfect_predictions = int(((all_labels == all_preds).all(axis=1) & (all_labels.sum(axis=1) > 0)).sum())
perfect_overall = int((all_labels == all_preds).all(axis=1).sum())

print(f"\n{'='*60}")
print(f"RESULTADOS FINALES:")
print(f"{'='*60}")
print(f"Total de muestras procesadas: {len(all_labels)}")
print(f"\nMétricas Generales:")
print(f"  Accuracy:  {accuracy_val:.4f} ({accuracy_val*100:.2f}%)")
print(f"  MicroF1:   {micro_f1_val:.4f}")
print(f"  MacroF1:   {macro_f1_val:.4f}")
print(f"  SubsetAcc: {subset_acc:.4f}")
print(f"  Hamm loss: {hamming_loss_val:.4f}")
print(f"  Examp F1:  {example_f1:.4f}")
print(f"  Examp prec:{example_prec:.4f}")
print(f"  Examp rec: {example_rec:.4f}")
print(f"  Micro prec:{micro_prec:.4f}")
print(f"  Micro rec: {micro_rec:.4f}")
print(f"  Macro prec:{macro_prec:.4f}")
print(f"  Macro rec: {macro_rec:.4f}")
print(f"\nPredicciones Perfectas (casos con fallos):")
print(f"  Total casos con fallos: {total_positive_cases}")
print(f"  Predicciones perfectas: {perfect_predictions}")
if total_positive_cases > 0:
    perfect_rate = (perfect_predictions / total_positive_cases) * 100
    print(f"  Tasa de acierto en fallos: {perfect_rate:.2f}%")
else:
    print(f"  Tasa de acierto en fallos: N/A (no hubo casos con fallos)")
print(f"\nPredicciones perfectas totales (todas las muestras): {perfect_overall}")
print(f"{'='*60}")
