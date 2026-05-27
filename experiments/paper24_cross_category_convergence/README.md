# Paper 24 — W-Shaped Cross-Category Convergence in Behavioral Routing

**Archon, Jesse Caldwell, Aura — DuoNeural Research — 2026-05-27**

## Overview

Measures pairwise cosine similarity between harm direction vectors across 4 categories
(weapons, drugs, cybercrime, hate_speech) and 28 transformer layers in Qwen3-0.6B.

**Main finding**: W-shaped cross-category convergence profile:
- L0 = 0.895 (embedding peak: shared instruction-style surface features)
- L10 = 0.594 (minimum: feature decomposition, maximum categorical resolution)
- L16 = 0.684 (secondary peak: harm detection integration zone)
- L27 = 0.522 (readout specialization)

**Alignment amplification**: L16 secondary peak is 2.33× stronger in aligned model
vs unaligned base model (precise: 2.326×, aligned Δ=0.0896, base Δ=0.0385).

**hate_speech outlier**: Persistent geometric separation in all layers.
Base model weapons_vs_hate_speech at L27 = 0.084 (near-orthogonal).
Alignment raises this to 0.314 via "refusal funnel" effect.

**Architecture creates, alignment sharpens**: W-shape is present in base model,
alignment amplifies L16 and raises L27 floor (+0.160 via shared refusal attractor).

## Files

- `l27_convergence_base.py` — runs analysis on Qwen3-0.6B-Base (non-aligned control)
- `l27_convergence_aligned.json` — aligned model results (n=50 prompts/category)
- `l27_convergence_base.json` — base model results (n=50 prompts/category)  
- `gen_figs_base_comparison.py` — generates comparison figures

Aligned model script: `../paper22_direction_rotation/l27_convergence_v2.py`

## Methods

For each harm category c and layer ℓ:
- Compute mean-difference direction: d_c^(ℓ) = mean(harm_c, layer_ℓ) - mean(benign, layer_ℓ)
- Pairwise cosine similarity across all C(4,2)=6 category pairs
- Convergence index = mean pairwise cosine similarity
- Bootstrap 95% CI (N=500 bootstrap resamples)

## Key References

- P15: Behavioral Routing Layer. DOI: 10.5281/zenodo.20348071
- P19: CNA Depth Hierarchy. DOI: 10.5281/zenodo.20384022
- P22: Direction Rotation and Norm Amplification. DOI: 10.5281/zenodo.20416382
- Arditi et al. 2024: "Refusal in LLMs is mediated by a single direction." arXiv:2406.11717
- Zou et al. 2023: "Representation Engineering." arXiv:2310.01405
