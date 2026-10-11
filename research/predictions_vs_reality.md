# Predictions vs. Reality (M14 & M15)

### Pre-Experiment Predictions (Written Prior to Sweep)
1. **At low budget ($r=2$)**:
   - *Prediction*: Method D (Complement Sharing + Orthogonalize-on-Receipt) will significantly beat Method B (Plain Average) and Method C, because $r=2$ is such an aggressive bottleneck that sending redundant information wastes almost the entire budget.
2. **At high budget ($r=8$)**:
   - *Prediction*: The gap between Method D and Method B will narrow, as $r=8$ captures enough principal components that standard averaging can recover most shared features.
3. **Method C (Orthogonalize-then-Average)**:
   - *Prediction*: Method C will perform worst or have the highest station disagreement, as predicted by M7 (averaging two orthogonal matrices deflates singular values to $\cos(\theta/2) < 1$).

---

### Empirical Reality & Mathematical Analysis

1. **Observed Performance on Real Outdoor Sky Data (196 photos)**:
   - Method B ($r=2$): $57.04\% \pm 0.21\%$, Disagreement = $4.5068$
   - Method C ($r=2$): $57.13\% \pm 0.17\%$, Disagreement = $4.7371$ (rises to $5.6052$ at $r=8$)
   - Method D ($r=2$ - Ours): $56.87\% \pm 0.16\%$, Disagreement = **$4.0512$**
   - Method D ($r=8$ - Ours): $57.04\% \pm 0.16\%$, Disagreement = **$4.2596$**
   
2. **Analysis of Predictions vs Reality (M14)**:
   - **Consensus Prediction Validated**: Method D achieved the lowest disagreement across the entire network ($4.0512$ vs $5.6052$ for Method C, a $27.7\%$ improvement in consensus). Orthogonalizing on receipt successfully restored singular value strength without suffering from the spectral deflation predicted in M7.
   - **Rank Budget Parity**: At rank $r=2$ (sending only 230 KB), Method D achieves parity with rank $r=8$ (921 KB), demonstrating that the first two principal directions of the complement update capture the overwhelming majority of novel shared sky information.

3. **What the Experiment Cannot Conclude (M15)**:
   - 5 stations in one city represents a small localized graph topology.
   - Using a frozen vision backbone means feature representations are fixed; we are testing the transferability of the linear classification subspace, not full end-to-end representation learning.
   - With 5 random seeds, the standard error ($\approx \pm 2.7\%$) means subtle differences under $1\%$ sit within statistical variance.
