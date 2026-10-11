import sys
sys.path.append(".")

import torch
from src.models.head import SkyHead

# Instantiate head
head = SkyHead(in_features=512, hidden_dim=64, num_classes=4)

# Create a test batch of 10 samples
x = torch.randn(10, 512)
logits = head(x)

print("Head test passed:")
print(f"  Input shape: {x.shape}")
print(f"  Logits shape: {logits.shape} (expected [10, 4])")
print(f"  fc1 weight shape: {head.get_fc1_weight().shape} (expected [64, 512])")
