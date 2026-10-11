# Paper Card 4: Decentralized Gossip Learning & Ring Mixing
**Reference**: Lian et al. (NeurIPS 2017), *Can Decentralized Algorithms Outperform Centralized Algorithms?* / Gossip Survey

### 1. The Problem
Centralized federated learning relies on a single point of failure (a central server) and creates communication bottlenecks at the hub. In serverless peer-to-peer topologies (like dispersed environmental sensors), nodes must reach consensus purely by communicating with physical neighbors.

### 2. The Method (in your own words)
Decentralized SGD (D-SGD) operates without a central coordinator:
1. Every node maintains its own local model.
2. In each round, nodes perform local gradient updates.
3. Nodes send their model states/updates only to direct graph neighbors.
4. Each node computes a weighted average of received messages using a doubly stochastic mixing matrix $W$.

### 3. One Key Equation Re-derived
**Consensus Spectral Gap on a 5-Node Ring**:
$$W_{ij} = \frac{1}{3} \quad \text{for } j \in \{i-1, i, i+1\} \pmod 5$$
$$\lambda_k(W) = \frac{1 + 2\cos(2\pi k / 5)}{3}$$
$$\lambda_2(W) \approx 0.54$$
Disagreement between nodes contracts geometrically at rate $\mathcal{O}((0.54)^t)$ per gossip exchange round.

### 4. Assumptions
- **Symmetric Graph**: Undirected communication links where weights form a doubly stochastic matrix ($\sum_j W_{ij} = 1, \sum_i W_{ij} = 1$).
- **Full Model Sharing**: Standard D-SGD assumes sharing all $D$ model parameters without rank constraints.

### 5. What It Did Not Test
- Non-Euclidean spectral update geometry (like Muon's polar decomposition).
- Bandwidth-sparing complement projection between peers.
- Outdoor non-IID visual classification under tight message ranks ($r=2, r=8$).

### 6. How Hyphal Gossip Differs
- Standard D-SGD averages raw weights: $W_i \leftarrow \sum_j W_{ij} W_j$.
- **Hyphal Gossip** replaces raw weight averaging with:
  1. **Complement-projected rank-$r$ transmission**: Node $i$ transmits only $U_r \Sigma_r V_r^T$ orthogonal to $j$'s sketch.
  2. **Receive-side Muon orthogonalization**: The receiver normalizes the mixed update matrix to maintain unit spectral strength across non-IID weather splits.
