# Paper 23 — DHP Epiplexity Theory

**Paper:** "Dynamic Horizon Prediction at the Epiplexity Boundary: Toward a Unified Theory of Temporal Self-Organization in Neural Architectures"  
**DOI:** https://doi.org/10.5281/zenodo.20416383  
**Authors:** Archon, Jesse Caldwell, Aura — DuoNeural 2026

## Key Findings
- τ*/τ_L ≈ 0.72 is the epiplexity boundary: prediction depth where marginal MDL compression drops to zero
- Trajectory curvature is EPIPHENOMENAL not causal — forcing low curvature on Mamba does NOT unlock DHP
- β=2.57, γ=1 are empirically fitted (explicitly disclosed in paper)
- Information-theoretic framework: DHP as marginal description-length optimization
- Links to Friston FEP, Barnett-Lizier Transfer Entropy, GTE ordered-phase fingerprint

## Files
- `curvature_comparison_v1.py` / `curvature_comparison_v1.json` — curvature causal ablation
- `mamba_curvature_results_v1.json` — Mamba trajectory curvature data (curvature ≠ DHP)
- `gen_figs_p23.py` — figure generation (fig1_curvature_ablation, fig2_arch_comparison)

## Theoretical Notes
- Appendix A = "Phenomenological Model of Gradient Signal Decay" (NOT a derivation — explicitly phenomenological)
- Shadowing Lemma = theoretical path to rigorous proof (future work)
- 0.72 is empirically robust: appears in CTM (v38-v40), multiple Lorenz/Rössler runs, cross-architecture
