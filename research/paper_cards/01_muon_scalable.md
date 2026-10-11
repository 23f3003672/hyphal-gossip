# Paper Card 1: Muon Optimizer & Scalability for Deep Learning
**Reference**: Jordan et al. (2024), *Muon is Scalable for LLM Training* / *kellerjordan/Muon*

### 1. The Problem
Standard optimizers like AdamW update weights based on coordinate-wise gradient moments ($G / \sqrt{v}$). This treats every weight independently as a scalar, ignoring the fact that weights in neural networks are 2D linear operators (matrices). This leads to spectral imbalance: some matrix directions undergo huge parameter shifts while others barely update.

### 2. The Method (in your own words)
Muon replaces coordinate-wise normalization with **spectral-norm normalization** on 2D weight matrices:
1. It maintains standard SGD momentum on the gradient matrix: $M_t = \beta M_{t-1} + G_t$.
2. It orthogonalizes the momentum matrix using a fast, GPU-friendly 5-step Newton-Schulz polynomial iteration instead of an exact SVD.
3. It scales the resulting orthogonal update by $\sqrt{\max(1, m/n)}$ and learning rate $\eta$, applying standard weight decay.
4. Non-matrix parameters (biases, embeddings, 1D norms) fall back to standard AdamW.

### 3. One Key Equation Re-derived
**The Newton-Schulz Quintic Recurrence**:
$$X_{k+1} = a X_k + b X_k X_k^T X_k + c X_k (X_k^T X_k)^2$$
Where coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$ map every singular value $\sigma \in (0, \sqrt{3})$ to $1.0$ within 5 matrix multiplications.

### 4. Assumptions
- **Centralized training**: Run on large GPU clusters with tight communication (e.g., Megatron / Distributed Data Parallel).
- **2D matrices only**: The orthogonalization logic only applies to 2D linear/attention projection layers.

### 5. What It Did Not Test
- Decentralized, peer-to-peer (gossip) environments.
- Low-bandwidth edge devices.
- Non-IID or heterogeneous local data splits.

### 6. How Hyphal Gossip Differs
- Muon was designed as a local single-node/cluster optimizer for LLMs.
- **Hyphal Gossip** uses Muon not just locally, but as a **receive-side aggregation rule** in a peer-to-peer gossip network: stations orthogonalize received neighbor updates so no loud or biased neighbor dominates the local consensus.
