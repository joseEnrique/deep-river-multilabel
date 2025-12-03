"""
Ejemplo de cómo cargar y usar el modelo guardado
"""
import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np

# ==========================
# 1. Definir la arquitectura del modelo (igual que en el entrenamiento)
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

# ==========================
# 2. Cargar el modelo y scaler
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar checkpoint completo
checkpoint = torch.load("lstm_multilabel_ai4i_complete.pt", map_location=device)

# Extraer parámetros
model_params = checkpoint['model_params']
seq_len = checkpoint['seq_len']
feature_cols = checkpoint['feature_cols']
label_cols = checkpoint['label_cols']

print("Parámetros del modelo:")
print(f"  Input dim: {model_params['input_dim']}")
print(f"  Hidden dim: {model_params['hidden_dim']}")
print(f"  Output dim: {model_params['output_dim']}")
print(f"  Seq length: {seq_len}")
print(f"  Épocas entrenadas: {checkpoint['epoch']}")
print(f"  Loss final: {checkpoint['final_loss']:.4f}")

# Crear modelo con los parámetros guardados
model = LSTM_MultiLabel(
    input_dim=model_params['input_dim'],
    hidden_dim=model_params['hidden_dim'],
    output_dim=model_params['output_dim'],
    num_layers=model_params['num_layers'],
    dropout=model_params['dropout'],
    bidirectional=model_params['bidirectional']
).to(device)

# Cargar los pesos
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Modo evaluación
print("\n✓ Modelo cargado exitosamente!")

# Cargar el scaler
with open("scaler_ai4i.pkl", 'rb') as f:
    scaler = pickle.load(f)
print("✓ Scaler cargado exitosamente!")

# ==========================
# 3. Ejemplo de uso: Hacer predicción con nuevos datos
# ==========================
print("\n" + "="*60)
print("EJEMPLO DE PREDICCIÓN CON DATOS NUEVOS")
print("="*60)

# Simular datos nuevos (en un caso real, cargarías datos reales)
# Aquí usamos el dataset original como ejemplo
df = pd.read_csv("/home/quique/tesis/OEMLHAT4PdM/datasets/ai4i2020formatted.csv")
X_new = df[feature_cols].values[:100]  # Primeras 100 muestras como ejemplo

# Normalizar los datos usando el scaler guardado
X_new_scaled = scaler.transform(X_new)

# Crear secuencias
def create_sequences(features, seq_len):
    X_seq = []
    for i in range(len(features) - seq_len + 1):
        X_seq.append(features[i:i+seq_len])
    return np.array(X_seq)

X_new_seq = create_sequences(X_new_scaled, seq_len)

# Convertir a tensor
X_tensor = torch.tensor(X_new_seq, dtype=torch.float32).to(device)

# Hacer predicción
with torch.no_grad():
    logits = model(X_tensor)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int()

# Mostrar resultados
print(f"\nPredicciones realizadas: {len(preds)}")
print(f"\nEtiquetas: {label_cols}")
print(f"\nPrimeras 5 predicciones:")
for i in range(min(5, len(preds))):
    print(f"  Muestra {i+1}:")
    print(f"    Probabilidades: {probs[i].cpu().numpy()}")
    print(f"    Predicción: {preds[i].cpu().numpy()}")
    failures = [label_cols[j] for j in range(len(label_cols)) if preds[i][j] == 1]
    if failures:
        print(f"    Fallos predichos: {', '.join(failures)}")
    else:
        print(f"    Sin fallos predichos")

print("\n" + "="*60)
print("Predicción completada exitosamente!")
print("="*60)
