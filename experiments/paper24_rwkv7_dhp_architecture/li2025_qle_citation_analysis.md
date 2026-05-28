# Li et al. 2025 (arXiv:2503.13530) — Citation Analysis
*Archon analysis, 2026-05-28*

## Their Work Summary

**Title**: Cognitive Activation and Chaotic Dynamics in LLMs: A Quasi-Lyapunov Analysis of Reasoning Mechanisms  
**Authors**: Xiaojian Li, Yongkang Leng, Ruiqing Ding, Hangjie Mo, Shanlin Yang  
**Institution**: School of Management, Hefei University of Technology  
**Published**: March 2025

**Their QLE Definition** (Spatial/Intra-network):
- Inject perturbation δ at layer i of a Transformer residual stream
- Measure divergence Δh_{n} at output layer n
- QLE(i,n) = (1/(n-i)) × log(||Δh_n|| / ||δ||)
- QLE > 0 → divergent (chaotic), QLE < 0 → convergent (stable)

**Their Key Findings**:
1. Shallow layers (L0-9): convergent (QLE < 0) — fixed-point attractors
2. Deep layers (L10+): divergent (QLE > 0) — strange attractors  
3. Final layers: compression/stabilization
4. MLP dominates Attention (55.7% vs 44.2% of final output contribution)
5. Benign perturbations: 5% neuron zeroing → 20%+ accuracy drop on CMMLU
6. Model: Qwen2-14B, 40 layers, 5120 hidden dim

**Their Limitation (explicit)**:
> "This conjecture awaits further validation."  
> "future research should incorporate chaos control mechanisms"

## Our Differentiation

| Dimension | Li et al. 2025 | DuoNeural (this work) |
|---|---|---|
| QLE axis | **Spatial** (layer depth) | **Temporal** (timestep in SSM) |
| System | Static Transformer inference | Training dynamics in SSMs (RWKV-7) |
| Scope | Single forward pass | Full training trajectory + checkpoint evolution |
| Architecture | Transformers only (Qwen2) | RWKV-7, RWKV-6, CTM-like, Mamba |
| Finding | Chaotic regions exist | Bimodal attractor basins form during training |
| Evidence | Observational | Causal (Separatrix Perturbation Test) |
| Theoretical frame | Chaos observation | Lyapunov Alignment Theory (LAT) |
| Universal constant | None | τ*/τ_L ≈ 0.72 (DHP) |
| Control capability | None (future work) | Active basin steering via W_w perturbation |

## Critical Convergence

They said "future research should incorporate chaos control mechanisms."
**We are that future research.** The Separatrix Perturbation Test in v7 is the first 
active chaos control mechanism for neural network training attractors.

## Citation Templates

### For P25 (RWKV-7 architectural diversity paper):
> Li et al. [2025] demonstrated that Transformer inference exhibits quasi-Lyapunov 
> dynamics across network depth, with shallow layers convergent and deep layers 
> divergent. Our work extends this framework to the temporal dimension, measuring 
> QLE across timesteps in the RWKV-7 S-matrix state evolution during training. 
> Crucially, we observe that QLE profiles differ systematically between seeds that 
> converge to diverse vs. collapsed attractor basins (HL_CV=0.928 vs 0.046), suggesting 
> that training-time Lyapunov dynamics predict architectural fate.

### For P26 (Lyapunov Alignment Theory capstone):
> The existence of chaotic dynamics in LLM inference was established by Li et al. [2025], 
> who introduced the Quasi-Lyapunov Exponent (QLE) to quantify perturbation sensitivity 
> across Transformer depth. Lyapunov Alignment Theory (LAT) generalizes this finding to 
> training dynamics and extends it with three key contributions: (1) temporal QLE in 
> recurrent architectures, (2) bimodal attractor landscape characterization during learning, 
> and (3) causal basin-steering via separatrix perturbation—realizing what Li et al. 
> identified as "future research incorporating chaos control mechanisms."

## Name Conflict Note

Both papers use "QLE" as acronym but measure orthogonal quantities:
- Li et al.: QLE_{i→n} = depth-propagation chaos (spatial)
- LAT/DuoNeural: QLE_t = temporal state-space chaos (temporal)

**Recommendation for P25/P26**: Use "temporal-QLE" or "t-QLE" to distinguish, 
cite Li et al. for the spatial QLE they defined, and explicitly note the extension.

---
*Files: paper24/li2025_qle_citation_analysis.md*  
*Related: paper24/paper25_outline.md, paper24/tier2_synthesis_20260527.md*
