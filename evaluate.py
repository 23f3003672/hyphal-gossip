import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_all_results(sweep_dir="results/sweep"):
    files = glob.glob(os.path.join(sweep_dir, "cell_*.json"))
    results = []
    for f in sorted(files):
        with open(f, "r") as fp:
            results.append(json.load(fp))
    return results

def aggregate_and_report():
    results = load_all_results()
    if not results:
        print("No sweep results found! Run `python run_sweep.py --all` first.")
        return

    print(f"Loaded {len(results)} cell results.\n")
    
    # Organize data: (method, rank) -> list of runs
    grid = {}
    for r in results:
        key = (r["method"], r["rank"])
        if key not in grid:
            grid[key] = []
        grid[key].append(r)
        
    print("| Method | Rank Budget | Mean Test Acc | Acc Std Err | Disagreement | Total Bytes Sent |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    method_names = {
        'B': 'B: Plain Average',
        'C': 'C: Orthogonalize-then-Average',
        'D': 'D: Complement + Ortho (Yours)',
        'E': 'E: Random Directions (Ablation)'
    }
    
    summary = {}
    for (m, rank), runs in sorted(grid.items()):
        accs = [run["test_acc_mean"] for run in runs]
        disagreements = [run["final_disagreement"] for run in runs]
        bytes_sent = runs[0]["total_bytes"]
        
        mean_acc = np.mean(accs)
        se_acc = stats.sem(accs) if len(accs) > 1 else 0.0
        mean_dis = np.mean(disagreements)
        
        summary[(m, rank)] = (mean_acc, se_acc, mean_dis, bytes_sent)
        name = method_names.get(m, m)
        print(f"| {name} | r = {rank} | {mean_acc*100:.2f}% | ±{se_acc*100:.2f}% | {mean_dis:.4f} | {bytes_sent:,} B |")
        
    # --- M11: Paired comparison between Method D and Method B ---
    print("\n### M11: Statistical Significance (Paired t-test: Method D vs Method B)")
    for rank in [2, 8]:
        d_accs = [r["test_acc_mean"] for r in grid.get(('D', rank), [])]
        b_accs = [r["test_acc_mean"] for r in grid.get(('B', rank), [])]
        if len(d_accs) == len(b_accs) and len(d_accs) > 1:
            diff = np.array(d_accs) - np.array(b_accs)
            t_stat, p_val = stats.ttest_rel(d_accs, b_accs)
            print(f"Rank r={rank}: Mean Gain = +{np.mean(diff)*100:.2f}%, p-value = {p_val:.4f} (t={t_stat:.2f})")
            
    # --- Generate Chart ---
    methods = ['B', 'C', 'D', 'E']
    ranks = [2, 8]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(methods))
    width = 0.35
    
    for i, rank in enumerate(ranks):
        means = [summary.get((m, rank), (0, 0))[0] * 100 for m in methods]
        errors = [summary.get((m, rank), (0, 0))[1] * 100 for m in methods]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=errors, capsize=5, label=f"Rank Budget r = {rank}")
        
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Hyphal Gossip: Test Accuracy across Gossip Protocols under Rank Budgets")
    ax.set_xticks(x)
    ax.set_xticklabels([method_names[m] for m in methods], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/sweep_comparison.png", dpi=200)
    print("\nSaved publication comparison chart to `results/sweep_comparison.png`")

if __name__ == "__main__":
    aggregate_and_report()
