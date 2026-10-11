# Paper Card 3: FedSubMuon (Subspace Federated Muon)
**Reference**: EMNLP (2026), *FedSubMuon: Subspace Communication for Federated Muon*

### 1. The Problem
While FedMuon solves client drift, it is communication-prohibitive: clients must transmit the full $m \times n$ dense gradient matrix every round. For resource-constrained edge devices or low-bandwidth uplinks, this communication cost creates a severe bottleneck.

### 2. The Method (in your own words)
FedSubMuon reduces communication overhead by constraining matrix updates to a shared low-rank subspace:
1. It factorizes the update into low-rank components using truncated SVD or random projection matrices.
2. Clients transmit only rank-$r$ factors (matrices of size $m \times r$ and $r \times n$).
3. The central server aggregates these low-rank factors and performs the Muon spectral update within the reduced subspace.

### 3. One Key Equation Re-derived
**Rank-Budget Communication Ratio (Eckart–Young Truncation)**:
$$\Delta W \approx U_r \Sigma_r V_r^T$$
Where transmission cost drops from $m \cdot n$ parameters to:
$$\text{Cost}_{\text{low-rank}} = r(m + n)$$
For a layer of size $64 \times d$, rank-$r$ transmission saves bandwidth as long as:
$$r(64 + d) < 64d \implies r < \frac{64d}{64 + d} \approx 64$$

### 4. Assumptions
- **Central Coordinator**: A central server aggregates the subspace representations across all clients.
- **Symmetric Knowledge**: Assumes all clients align to a globally shared coordinate system maintained by the server.

### 5. What It Did Not Test
- Serverless peer-to-peer gossip topologies where no global coordinate system exists.
- Asymmetric knowledge between asymmetric peers (where peer $A$ knows things peer $B$ does not).
- Outdoor sensor networks with localized ambient noise.

### 6. How Hyphal Gossip Differs
- FedSubMuon projects updates onto a central subspace.
- **Hyphal Gossip** introduces **Complement Sharing**: peer $j$ advertises a tiny sketch of its known subspace $P = V_k V_k^T$, and peer $i$ transmits **only the orthogonal complement**:
  $$\Delta W_{\text{sent}} = \Delta W_i (I - V_k V_k^T)$$
  Peer $i$ never sends what peer $j$ already knows, achieving ultra-low byte transmission over a serverless ring topology.
