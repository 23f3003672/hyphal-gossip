import sys
sys.path.append(".")

import torch
from src.models.head import SkyHead
from src.dataset import generate_synthetic_station_data
from src.optim.muon import train_station_local

# 1. Get synthetic data for Station 0
data = generate_synthetic_station_data(num_stations=1, samples_per_station=30, d=512)
s0_feats = data[0]["features"]
s0_labels = data[0]["labels"]

# 2. Train locally on Station 0
head = SkyHead(in_features=512, hidden_dim=64, num_classes=4)
delta_W = train_station_local(head, s0_feats, s0_labels, epochs=5, lr=0.02)

print("Local training passed:")
print(f"  Delta W shape: {delta_W.shape} (expected [64, 512])")
print(f"  Delta W Frobenius norm: {delta_W.norm().item():.4f}")
print(f"  Max singular value: {torch.linalg.svdvals(delta_W)[0].item():.4f}")
