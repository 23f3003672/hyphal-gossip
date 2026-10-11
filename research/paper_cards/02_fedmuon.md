# Paper Card 2: FedMuon (Federated Muon)
**Reference**: arXiv:2510.27403 (2025), *FedMuon: Client-Drift Mitigation via Momentum-Aligned Muon*

### 1. The Problem
When federated clients have non-IID (statistically skewed) local data, standard FedAvg + Adam suffers from severe client drift. If one naively uses Muon locally on each client, the drift worsens: because Muon flattens all singular values to 1, rare or skewed local gradient directions are scaled up to full unit strength, pulling the global model into extreme divergence.

### 2. The Method (in your own words)
FedMuon corrects this using two server-driven mechanisms:
1. **Global Momentum Aggregation**: Instead of each client maintaining isolated momentum, the central server aggregates client velocity vectors to keep the global momentum direction stable.
2. **Local-Global Alignment**: Before clients apply their local Muon step, their updates are projected and aligned with the server's global direction, preventing local updates from drifting into disjoint orthogonal subspaces.

### 3. One Key Equation Re-derived
**Server-Side Momentum Alignment**:
$$M_{t}^{\text{global}} = \beta M_{t-1}^{\text{global}} + \sum_{k=1}^K w_k \Delta W_{t}^{(k)}$$
$$\Delta W_t^{\text{server}} = \text{Muon}\big( M_{t}^{\text{global}} \big) = U_{\text{global}} V_{\text{global}}^T$$
The orthogonalization happens *centrally* on the aggregated global momentum rather than letting clients diverge in isolation.

### 4. Assumptions
- **Centralized Parameter Server**: Requires a star topology where every client has direct connectivity to a central coordinator.
- **Synchronous Communication Rounds**: All participating clients report back to the server in lockstep.
- **Full-Matrix Transmission**: Clients transmit full $m \times n$ weight matrices back and forth each round.

### 5. What It Did Not Test
- Fully decentralized peer-to-peer (serverless) gossip topologies where no central authority exists.
- Strict per-message rank/byte communication constraints.
- Real-world outdoor sensor distribution (edge sky camera stations).

### 6. How Hyphal Gossip Differs
- FedMuon requires a **central server** to orchestrate momentum alignment. In Hyphal Gossip, the network is **serverless** (a peer-to-peer gossip graph).
- Hyphal Gossip mitigates client drift without a server by using **Complement Sharing** (stations only send what their neighbor doesn't know) and **Receive-Side Orthogonalization** (each receiver flattens incoming updates on arrival so no single skewed peer dominates).
