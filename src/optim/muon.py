import torch

def zeropower_via_newtonshulz5(G: torch.Tensor, steps: int=5, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes the polar decomposition factor UV^T of 2D matrix G using a
    5 step quintic Newton Schulz polynomila iteration
    """
    assert G.ndim == 2, f"Muon only supports 2D matrices, got {G.ndim}D"
    a,b,c = 3.4445, -4.7750, 2.0315

    # work in float32 for numerical stability
    X = G.float()

    # NOrmalize spectral norm to sit within the convergence basin (0,sqrt(3))
    norm = X.norm() + eps 
    X = X/norm 

    for _ in range(steps):
        A = X @ X.T 
        B = b*A + c*(A @ A)
        X = a*X + B@X 

    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """
    Muon optimizer for 2D weight matrices:
    Accumulates momentum, orthogonalizes via Newton-Schulz, and applies spectral scaling.
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.01):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                # Initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                # Orthogonalize momentum via Newton-Schulz
                ortho_update = zeropower_via_newtonshulz5(buf)

                # Scaling factor sqrt(max(1, m/n)) from Jordan et al.
                m, n = p.shape
                scale = (max(1, m / n)) ** 0.5

                # Weight decay + Update
                p.mul_(1.0 - lr * wd)
                p.add_(ortho_update, alpha=-lr * scale)


def train_station_local(head: torch.nn.Module, features: torch.Tensor, labels: torch.Tensor, 
                        epochs: int = 5, lr: float = 0.02, batch_size: int = 16):
    """
    Trains a station head locally on its private photos.
    Uses Muon for fc1 and AdamW for fc2.
    Returns the update delta_W1 = W_after - W_before.
    """
    criterion = torch.nn.CrossEntropyLoss()
    
    # Muon on 2D matrix (fc1), AdamW on final head (fc2)
    opt_muon = Muon([head.fc1.weight], lr=lr, momentum=0.9, weight_decay=0.01)
    opt_adam = torch.optim.AdamW(head.fc2.parameters(), lr=lr * 0.5, weight_decay=0.01)
    
    W_initial = head.get_fc1_weight()
    
    dataset_size = features.size(0)
    for _ in range(epochs):
        perm = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            idx = perm[i:i + batch_size]
            bx, by = features[idx], labels[idx]
            
            opt_muon.zero_grad()
            opt_adam.zero_grad()
            
            logits = head(bx)
            loss = criterion(logits, by)
            loss.backward()
            
            opt_muon.step()
            opt_adam.step()
            
    delta_W = head.get_fc1_weight() - W_initial
    return delta_W
