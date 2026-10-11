import torch
from src.optim.muon import zeropower_via_newtonshulz5

def get_ring_neighbors(node_id: int, num_nodes: int = 5):
    """Returns the two immediate neighbors in a ring graph."""
    return [(node_id - 1) % num_nodes, (node_id + 1) % num_nodes]

def compress_rank_r(A: torch.Tensor, rank: int):
    """Truncates A to rank r using SVD (Eckart-Young optimal approximation)."""
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    r = min(rank, len(S))
    Ur = U[:, :r]
    Sr = S[:r]
    Vhr = Vh[:r, :]
    # Returns factored form (Ur * Sr, Vhr)
    return Ur @ torch.diag(Sr), Vhr

def count_bytes(m: int, n: int, rank: int, dtype_bytes: int = 4) -> int:
    """Computes exact bytes transmitted for a rank-r factored message."""
    # Transmitting (m x r) and (r x n) floats
    return rank * (m + n) * dtype_bytes

def compute_conflict_metric(delta1: torch.Tensor, delta2: torch.Tensor) -> float:
    """
    Measures conflict/alignment between two station updates:
    Cosine similarity in Frobenius space: 1.0 = identical, -1.0 = opposite.
    """
    dot = (delta1 * delta2).sum().item()
    norm1 = delta1.norm().item()
    norm2 = delta2.norm().item()
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def gossip_round(station_heads: dict, station_deltas: dict, method: str, rank: int, d: int = 512):
    """
    Executes one communication round across the 5 stations according to the chosen method.
    Methods:
      'B': Plain Average (rank-r truncated)
      'C': Orthogonalize-then-average
      'D': Complement Sharing + Orthogonalize on receipt (Yours)
      'E': Random Projection with same budget (Ablation)
    """
    num_nodes = len(station_heads)
    new_weights = {}
    bytes_sent_total = 0
    
    for i in range(num_nodes):
        neighbors = get_ring_neighbors(i, num_nodes)
        incoming_updates = []
        
        for j in neighbors:
            delta_j = station_deltas[j]  # update from neighbor j
            
            if method == 'B':  # Plain rank-r average
                L, R = compress_rank_r(delta_j, rank)
                msg = L @ R
                bytes_sent_total += count_bytes(64, d, rank)
                incoming_updates.append(msg)
                
            elif method == 'C':  # Orthogonalize then average
                ortho_j = zeropower_via_newtonshulz5(delta_j)
                L, R = compress_rank_r(ortho_j, rank)
                msg = L @ R
                bytes_sent_total += count_bytes(64, d, rank)
                incoming_updates.append(msg)
                
            elif method == 'D':  # Complement sharing + Ortho on receipt (Yours)
                # Station i shares top-k sketch of its current knowledge (V_k)
                _, _, Vh_i = torch.linalg.svd(station_deltas[i], full_matrices=False)
                k = min(rank, Vh_i.size(0))
                Vk = Vh_i[:k, :].T  # d x k
                
                # Station j removes i's known directions: delta_j @ (I - Vk Vk^T)
                novel_delta = delta_j - (delta_j @ Vk) @ Vk.T
                
                # Compress the novel complement to rank r
                L, R = compress_rank_r(novel_delta, rank)
                msg = L @ R
                bytes_sent_total += count_bytes(64, d, rank)
                incoming_updates.append(msg)
                
            elif method == 'E':  # Random projection ablation
                torch.manual_seed(100 * j + i)
                rand_proj = torch.randn(d, rank)
                Q, _ = torch.linalg.qr(rand_proj)  # d x rank orthogonal
                projected = delta_j @ Q  # 64 x rank
                msg = projected @ Q.T    # 64 x d
                bytes_sent_total += count_bytes(64, d, rank)
                incoming_updates.append(msg)
                
        # Receiver mixing
        self_weight = station_heads[i].get_fc1_weight()
        
        if method in ['B', 'C', 'E']:
            # Average self update with received updates (1/3 weight each)
            avg_update = (station_deltas[i] + sum(incoming_updates)) / 3.0
            new_weights[i] = self_weight + avg_update
        elif method == 'D':
            # Sum novel updates and orthogonalize on receipt via Muon
            sum_received = sum(incoming_updates)
            ortho_mixed = zeropower_via_newtonshulz5(station_deltas[i] + sum_received)
            # Apply update scaled to average peer step magnitude
            scale = station_deltas[i].norm() / (ortho_mixed.norm() + 1e-7)
            new_weights[i] = self_weight + (ortho_mixed * scale)
            
    # Update model parameters
    for i in range(num_nodes):
        station_heads[i].set_fc1_weight(new_weights[i])
        
    return bytes_sent_total
