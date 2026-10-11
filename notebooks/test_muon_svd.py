
import sys 
sys.path.append(".")
import torch
import matplotlib.pyplot as plt 
from src.optim.muon import zeropower_via_newtonshulz5

torch.manual_seed(42)
print("Running Muon Newton Schulz sanity check...")

# Test across 100 random rectangular matrices(64*384)
errors = []
orig_singular_vals = []
final_singular_vals = []

for _ in range(100):
    G = torch.randn(64,384)

    # exact svd: G = U*S*Vh --> polar factor = U @ Vh
    U, S, Vh = torch.linalg.svd(G, full_matrices = False)
    exact_UV = U @ Vh
    orig_singular_vals.extend(S.tolist())

    # Muon approximation 5 steps
    approx_UV = zeropower_via_newtonshulz5(G, steps=5)

    # Relative frobenius error
    rel_err = torch.norm(approx_UV - exact_UV).item() / torch.norm(exact_UV).item()
    errors.append(rel_err) 


    # check singular values of the output matrix
    _, approx_S,_ = torch.linalg.svd(approx_UV, full_matrices=False)
    final_singular_vals.extend(approx_S.tolist())

mean_err = sum(errors) / len(errors)
print(f"--> Average relative error vs exact SVd across 100 matrices: {mean_err:.5f}")
print(f"--> Final singular vales: min = {min(final_singular_vals): .4f}, max = {max(final_singular_vals): .4f}")

# Plot singular values before vs after
plt.figure(figsize=(8,4))
plt.hist(orig_singular_vals[:300], bins = 30, alpha=0.6, label='Original Singular Values(G)')
plt.hist(final_singular_vals[:300], bins=30, alpha=0.8, color ="crimson", label='Muon Singula values(UV^T)')
plt.title(f"Muon Newton-schulz 5-step singular value collapse (Avg Rel Error: {mean_err:.4f})")
plt.xlabel("Singular Value Magnitude")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("results/singular_values_checkpoint.png")
print("Saved checkpoint plot to results/singular_values_checkpoint.png")
