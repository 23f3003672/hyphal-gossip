import sys
sys.path.append(".")

import torch
from src.models.head import SkyHead
from src.gossip.protocols import gossip_round, compute_conflict_metric

# 1. Instantiate 5 station heads and fake deltas
heads = {i: SkyHead(in_features=512, hidden_dim=64, num_classes=4) for i in range(5)}
deltas = {i: torch.randn(64, 512) * 0.1 for i in range(5)}

# 2. Test conflict metric
c_self = compute_conflict_metric(deltas[0], deltas[0])
c_opp = compute_conflict_metric(deltas[0], -deltas[0])
print(f"Conflict Metric Test:")
print(f"  Self-similarity: {c_self:.4f} (expected 1.0)")
print(f"  Opposite: {c_opp:.4f} (expected -1.0)")

# 3. Test one gossip round for each method (B, C, D, E) at rank 2
for method in ['B', 'C', 'D', 'E']:
    bytes_sent = gossip_round(heads, deltas, method=method, rank=2, d=512)
    print(f"Method {method} executed successfully: {bytes_sent} bytes sent across network")
