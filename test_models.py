import torch
from testclassifier.model import CNN_MultiLabel, Transformer_MultiLabel

batch_size = 32
seq_len = 5
input_dim = 10
output_dim = 5

x = torch.randn(batch_size, seq_len, input_dim)

print("Testing CNN...")
cnn = CNN_MultiLabel(input_dim, hidden_dim=64, output_dim=output_dim, num_layers=2)
out_cnn = cnn(x)
print(f"CNN output shape: {out_cnn.shape}")
assert out_cnn.shape == (batch_size, output_dim)

print("Testing Transformer...")
trans = Transformer_MultiLabel(input_dim, hidden_dim=64, output_dim=output_dim, num_layers=2)
out_trans = trans(x)
print(f"Transformer output shape: {out_trans.shape}")
assert out_trans.shape == (batch_size, output_dim)

# Test sequence length 1 (x.dim() == 2)
x1 = torch.randn(batch_size, input_dim)
out_cnn_1 = cnn(x1)
out_trans_1 = trans(x1)
print(f"CNN output seq=1 shape: {out_cnn_1.shape}")
print(f"Transformer output seq=1 shape: {out_trans_1.shape}")

print("All tests passed.")
