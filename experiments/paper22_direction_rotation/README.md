# Paper 22 — Directional Evolution of Behavioral Routing

**Paper:** "Directional Evolution of Behavioral Routing in Transformer Residual Streams: From Early Crystallization to Late Readout"  
**DOI:** https://doi.org/10.5281/zenodo.20416382  
**Authors:** Archon, Jesse Caldwell, Aura — DuoNeural 2026

## Key Findings
- Behavioral routing direction in Qwen3-0.6B undergoes continuous 3-phase evolution:
  1. Convergence (L0-L5): directions align across harmful/benign
  2. Crystallization (L6, CDP): direction established, 57× norm amplification
  3. Rotation + Amplification (L7-L27): 80° total rotation, 119× norm growth from L6
- 4 harm categories (weapons/drugs/cybercrime/hate_speech) all reach 79-81° by L27
- Appendix B: MLP cos_align=0.388 (amplification role), Attn perp_frac=94.8% (rotation role)
- Resolves CDP-CNA gap: crystallization and readout are sequential phases

## Files
- `p22_full_sweep.py` — main rotation/norm sweep across all 28 layers
- `p22_topic_sweep.py` / `p22_topic_sweep_results.json` — 4-category topic analysis
- `p22_direction_vectors.json` — saved direction vectors per layer per category
- `mlp_attn_attribution_v1.py` / `mlp_attn_attribution_v1.json` — MLP vs Attn attribution (Appendix B)
- `make_figures_p22.py` — figure generation
- `l27_convergence_v1.py` / `l27_convergence_v1.json` — P22 follow-up: cross-category convergence analysis (see below)

## Follow-up Experiment (l27_convergence_v1)
**NEW FINDING (2026-05-27):** Three-zone architecture discovered:
1. Early crystallization (L0→L6): 0.167→0.527
2. Mid-network harmfulness hyperplane (L7→L16): PEAK at L16 = 0.665
3. Readout specialization (L17→L27): monotone decrease to 0.534

This suggests a "universal harmfulness detector" at L16 followed by category-specific output
direction refinement in the readout zone. Future paper candidate.
