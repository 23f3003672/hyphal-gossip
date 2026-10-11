import torch
import torch.nn as nn
import torch.nn.functional as F 

class SkyHead(nn.Module):
    """
    Lightweight 2-layer classifier head:
    Embedding (d) -> Linear(d,64) -> ReLU -> Linear(64,4)
    """

    def __init__(self, in_features: int = 512, hidden_dim: int=64, num_classes: int=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features,hidden_dim,bias=False) #64Xd Muon target
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=True) #4X64 Tiny head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits 

    def get_fc1_weight(self) -> torch.Tensor:
        return self.fc1.weight.data.clone()

    def set_fc1_weight(self, new_weight: torch.Tensor):
        self.fc1.weight.data.copy_(new_weight)

        