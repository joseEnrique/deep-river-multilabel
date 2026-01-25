
import torch
import torch.nn as nn
from deep_river.base import RollingDeepEstimator
from river import metrics


from classes.stateful_multilabel_classifier import StatefulMultiLabelClassifier
from testclassifier.model import AlpiOneHotLSTM

def test_stateful_execution():
    print("=== Testing StatefulMultiLabelClassifier ===")
    
    # 1. Setup small model
    # We use AlpiOneHotLSTM which we just modified to accept hx
    # Config: 5 features, 3 labels, hidden=10
    n_features = 5
    n_labels = 3
    hidden_dim = 10
    
    module_cls = AlpiOneHotLSTM
    
    clf = StatefulMultiLabelClassifier(
        module=module_cls,
        label_names=[f"L{i}" for i in range(n_labels)],
        optimizer_fn="adam",
        lr=0.01,
        hidden_dim=hidden_dim, # passed to AlpiOneHotLSTM
        num_alarms=20, # arbitrary for one-hot
        num_layers=1,
        bidirectional=False,
        seed=42
    )
    
    # 2. Generate Synthetic Stream
    # Features are integers for one-hot (0..19)
    stream = []
    for i in range(20):
        x = {f"F{j}": (i + j) % 20 for j in range(n_features)}
        # Dummy target: L0 is active if i is even
        y = {"L0": 1 if i % 2 == 0 else 0, "L1": 0, "L2": 1}
        stream.append((x, y))
        
    print(f"Stream length: {len(stream)}")
    
    # 3. Running loop
    metric = metrics.Accuracy()
    
    print("\nStarting Training Loop...")
    for i, (x, y) in enumerate(stream):
        # Predict
        p = clf.predict_one(x)
        
        # Learn
        clf.learn_one(x, y)
        
        # Check State
        current_state = clf.hidden_state
        
        if i == 0:
            if current_state is None:
                print(f"Step {i}: State is None! (Expected if model inits after first learn)")
            else:
                print(f"Step {i}: State Initialized. h shape: {current_state[0].shape}")
        
        if i == 1:
             if current_state is not None:
                 print(f"Step {i}: State persisted! h[0][0][0] = {current_state[0][0][0][0]:.4f}")
        
        if i == 5:
             if current_state is not None:
                print(f"Step {i}: State updated!   h[0][0][0] = {current_state[0][0][0][0]:.4f}")

    print("\n✅ Execution finished without crashing.")
    if clf.hidden_state is not None:
        print("✅ Final state exists (Stateful logic active).")
    else:
        print("❌ Final state is None (Something failed).")

if __name__ == "__main__":
    test_stateful_execution()
