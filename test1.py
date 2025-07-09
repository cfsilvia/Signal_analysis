import torch
import torch.nn as nn

# 1 batch, 2 channels (L & R), 8 time-steps
x = torch.tensor([[
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],   # Left channel
    [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]    # Right channel
]])  # shape = (1, 2, 8)

# Define conv: in=2 channels → out=32 filters, window=5, pad=2
conv = nn.Conv1d(2, 32, kernel_size=5, padding=2)

# Apply it
y = conv(x)
print(y.shape)  # → torch.Size([1, 32, 8])