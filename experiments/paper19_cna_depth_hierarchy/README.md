# Paper 19 — CNA Depth Hierarchy: A Two-Stage Behavioral Routing Architecture

**DuoNeural Research | 2026-05-25**
**Authors:** Archon, Jesse Caldwell, Aura
**DOI:** 10.5281/zenodo.20383917
**Zenodo:** https://doi.org/10.5281/zenodo.20383917

## Summary

We apply Contrastive Neuron Attribution (CNA; Nous Research 2026) to Qwen3-0.6B and Qwen3-8B to characterize the depth profile of individual MLP neurons responsible for behavioral differences. CNA finds late-layer concentration (centroid 0.88, L25–L27) for both refusal gating and self-identification routing. This sharply contrasts with our prior Contrastive Direction Probing (CDP) results showing direction crystallization at L6 (~21% depth).

We propose a two-stage architecture: (1) distributed direction crystallization at L6 (CDP-visible, CNA-invisible) and (2) late-layer individual neuron readout at L25–L27 (CNA-visible, CDP-redundant). A base model analysis reveals this late-layer structure is a pre-alignment constant — the instruct model has 7–12% *lower* Δ_mean than the base model, indicating alignment installs routing consistency (gating), not routing magnitude.

## Key Results

| Comparison | Model | Centroid | Late frac. | Δ_mean |
|---|---|---|---|---|
| Harmful vs. Benign | 0.6B | 0.884 | 67.4% | 0.113 |
| SKI vs. SKT | 0.6B | 0.858 | 51.2% | 0.187 |
| SKI vs. Benign | 0.6B | 0.863 | 57.0% | 0.191 |
| Harmful vs. Benign | 8B | 0.909 | 84.4% | — |

Jaccard overlap (Refusal ∩ Self-ID): J = 0.057–0.117 (shared depth profile, distinct neurons).

## Scripts

| File | Purpose |
|---|---|
| `p19_cna_depth_hierarchy.py` | Main CNA experiment (Qwen3-0.6B, 3 comparisons, global + per-layer) |
| `p19_8b_kilonova.py` | Scale comparison experiment (Qwen3-8B, 4-bit, kilonova) |
| `p19_extended_analysis.py` | Jaccard overlap, base model, sensitivity analysis |
| `make_figures_p19.py` | Generates all 4 paper figures from JSON results |

## Data Files

| File | Contents |
|---|---|
| `p19_cna_results.json` | Full 0.6B CNA attribution results |
| `p19_8b_results.json` | 8B scale comparison results |
| `p19_extended_results.json` | Jaccard, base model, sensitivity results |

## Reproduction

```bash
# 0.6B experiment (CPU, ~30 min)
python p19_cna_depth_hierarchy.py

# 8B experiment (requires GPU with 4-bit support)
python p19_8b_kilonova.py

# Extended analysis (base model, Jaccard)
python p19_extended_analysis.py

# Generate figures
python make_figures_p19.py
```

## Note on Terminology

In DuoNeural Papers 15–18, our residual-stream direction probing method was referred to as "CCS." Starting with Paper 19, we adopt the canonical term **Contrastive Direction Probing (CDP)** to distinguish our supervised mean-difference approach from Burns et al. (2022) Contrastive Consistent Search (unsupervised logical consistency objective). The methods are mathematically distinct.
