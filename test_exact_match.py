import torch
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier
from classes.rolling_multilabel_classifier_sequences import RollingMultiLabelClassifierSequences
from testclassifier.model import LSTM_MultiLabel

target_names = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

def test_models():
    # Setup
    torch.manual_seed(42)
    device = "cpu"
    
    # 1. Init Rolling
    rolling = RollingMultiLabelClassifier(
        module=LSTM_MultiLabel,
        label_names=target_names,
        optimizer_fn="adam",
        lr=1e-3,
        device=device,
        window_size=3,
        append_predict=False,
        hidden_dim=64,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
        output_dim=len(target_names),
        seed=42,
        epochs=1,
        loss_fn=torch.nn.BCEWithLogitsLoss()
    )
    
    # 2. Init Direct
    direct = RollingMultiLabelClassifierSequences(
        window_size=3,
        past_history=1,
        label_names=target_names,
        module=LSTM_MultiLabel,
        optimizer_fn="adam",
        lr=1e-3,
        device=device,
        hidden_dim=64,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
        output_dim=len(target_names),
        seed=42,
        epochs=1,
        loss_fn=torch.nn.BCEWithLogitsLoss(),
    )
    
    # Generate some fake inputs
    inputs = [
        {"f1": 0.1, "f2": 0.2},
        {"f1": 0.2, "f2": 0.4},
        {"f1": 0.3, "f2": 0.6},
        {"f1": 0.4, "f2": 0.8},
    ]
    targets = [
        {"TWF": 0, "HDF": 1, "PWF": 0, "OSF": 0, "RNF": 0},
        {"TWF": 0, "HDF": 0, "PWF": 1, "OSF": 0, "RNF": 0},
        {"TWF": 1, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0},
        {"TWF": 0, "HDF": 0, "PWF": 0, "OSF": 1, "RNF": 0},
    ]
    
    # Learn 4 steps
    for i in range(4):
        x = inputs[i]
        y = targets[i]
        
        # Predict
        p1 = rolling.predict_proba_one(x)
        p2 = direct.predict_proba_one(x)
        print(f"\\nStep {i+1} Probas:")
        print(f"Rolling: {[round(p1[k], 4) for k in target_names]}")
        print(f"Direct:  {[round(p2[k], 4) for k in target_names]}")
        
        # Learn
        rolling.learn_one(x, y)
        direct.learn_one(x, y)
        
        # Weights check
        print(f"Rolling weights sum: {rolling.module.rnn.weight_hh_l0.sum().item():.6f}")
        print(f"Direct weights sum:  {direct.module.rnn.weight_hh_l0.sum().item():.6f}")

if __name__ == "__main__":
    test_models()
