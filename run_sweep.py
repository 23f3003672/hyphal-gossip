import os
import json
import argparse
import torch
from src.dataset import generate_synthetic_station_data
from src.models.head import SkyHead
from src.optim.muon import train_station_local
from src.gossip.protocols import gossip_round

METHODS = ['B', 'C', 'D', 'E']
RANKS = [2, 8]
SEEDS = [0, 1, 2, 3, 4]

def index_to_cell(idx: int):
    """M12: Deterministic collision-free bijection from index 0..39 to (method, rank, seed)."""
    seed = SEEDS[idx % 5]
    rank = RANKS[(idx // 5) % 2]
    method = METHODS[idx // 10]
    return method, rank, seed

def evaluate_heads(heads: dict, test_feats: torch.Tensor, test_labels: torch.Tensor):
    """Evaluates each station head on test data and returns per-station and average accuracy."""
    accuracies = []
    for i, head in heads.items():
        with torch.no_grad():
            logits = head(test_feats)
            preds = logits.argmax(dim=-1)
            acc = (preds == test_labels).float().mean().item()
            accuracies.append(acc)
    return accuracies, sum(accuracies) / len(accuracies)

def compute_disagreement(heads: dict):
    """Measures variance/disagreement across station fc1 weights."""
    weights = [h.get_fc1_weight() for h in heads.values()]
    mean_w = torch.stack(weights).mean(dim=0)
    disagreement = torch.stack([(w - mean_w).norm() for w in weights]).mean().item()
    return disagreement

def run_cell(method: str, rank: int, seed: int, rounds: int = 5, local_epochs: int = 3, d: int = 512):
    """Runs one training experiment cell."""
    # 1. Load real extracted station data if available, otherwise fall back to synthetic
    from src.dataset import load_real_station_data
    train_data = load_real_station_data()
    
    if train_data is not None and os.path.exists("data/embeddings/test.pt"):
        test_data = torch.load("data/embeddings/test.pt")
        test_x, test_y = test_data["features"], test_data["labels"]
    else:
        train_data = generate_synthetic_station_data(num_stations=5, samples_per_station=30, d=d, seed=seed)
        test_data = generate_synthetic_station_data(num_stations=1, samples_per_station=100, d=d, seed=seed + 999)[0]
        test_x, test_y = test_data["features"], test_data["labels"]

    
    # 2. Instantiate heads
    torch.manual_seed(seed)
    heads = {i: SkyHead(in_features=d, hidden_dim=64, num_classes=4) for i in range(5)}
    
    total_bytes = 0
    disagreements = []
    
    # 3. Federated Gossip rounds
    for r in range(rounds):
        deltas = {}
        # Local training step at each station
        for i in range(5):
            s_feats = train_data[i]["features"]
            s_labels = train_data[i]["labels"]
            deltas[i] = train_station_local(heads[i], s_feats, s_labels, epochs=local_epochs, lr=0.02)
            
        # Gossip exchange step
        bytes_sent = gossip_round(heads, deltas, method=method, rank=rank, d=d)
        total_bytes += bytes_sent
        disagreements.append(compute_disagreement(heads))
        
    per_station_acc, avg_acc = evaluate_heads(heads, test_x, test_y)
    
    return {
        "method": method,
        "rank": rank,
        "seed": seed,
        "test_acc_mean": avg_acc,
        "per_station_acc": per_station_acc,
        "final_disagreement": disagreements[-1],
        "total_bytes": total_bytes
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None, help="Cell index 0..39")
    parser.add_argument("--all", action="store_true", help="Run full 40-cell grid")
    args = parser.parse_args()

    os.makedirs("results/sweep", exist_ok=True)

    if args.index is not None:
        method, rank, seed = index_to_cell(args.index)
        print(f"Running Cell {args.index}: Method={method}, Rank={rank}, Seed={seed}")
        res = run_cell(method, rank, seed)
        out_path = f"results/sweep/cell_{args.index:02d}.json"
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Cell {args.index} complete -> Acc: {res['test_acc_mean']:.4f}, Disagreement: {res['final_disagreement']:.4f}")

    elif args.all:
        print("Starting full 40-cell sweep...")
        for idx in range(40):
            method, rank, seed = index_to_cell(idx)
            print(f"[{idx+1}/40] Method={method}, Rank={rank}, Seed={seed} ...", end=" ", flush=True)
            res = run_cell(method, rank, seed)
            out_path = f"results/sweep/cell_{idx:02d}.json"
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"Done (Acc: {res['test_acc_mean']:.3f})")
        print("\nAll 40 cells completed successfully!")
