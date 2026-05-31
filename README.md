# DuoNeural Lab

Open research lab. Post-training, abliteration, GRPO, quantization, and mechanistic interpretability on open-weights models.

**Research Papers:** 22+ published on Zenodo — see https://duoneural.com

---

## Experiment Index

### DHP (Dynamic Horizon Prediction)
| Directory | Paper | Key Result |
|-----------|-------|------------|
| `experiments/paper4_dhp/` | [P4 — DHP Discovery](https://doi.org/10.5281/zenodo.19952612) | τ*/τ_L ≈ 0.72 confirmed in CTM on Lorenz/Rössler |
| `experiments/paper20_gdn2_dhp/` | [P20 — GDN-2 DHP](https://doi.org/10.5281/zenodo.20416345) | GDN-2 fails DHP: erase gate architectural locking |
| `experiments/paper21_dhp_arch_survey/` | [P21 — DHP Arch Survey](https://doi.org/10.5281/zenodo.20416345) | CTM ✓, GDN-2 ✗ (lock), Mamba ✗ (init), LSTM ✗ (init), Transformer ✗ |
| `experiments/paper12_tau_ctm/` | [P12 — τ*(t) Trajectory](https://doi.org/10.5281/zenodo.20327487) | τ* structural prior, not dynamic tracker |
| `experiments/paper23_dhp_epiplexity/` | [P23 — DHP Epiplexity](https://doi.org/10.5281/zenodo.20416383) | τ*/τ_L=0.72 is MDL epiplexity boundary; curvature epiphenomenal |

### Behavioral Routing & Alignment Geometry
| Directory | Paper | Key Result |
|-----------|-------|------------|
| `experiments/paper13_selfknow_suppression/` | [P13 — Self-Knowledge Suppression](https://doi.org/10.5281/zenodo.20329453) | SUPPRESSOR stage distinct from COMPRESSOR; L6 only positive layer |
| `experiments/paper15_behavioral_routing/` | [P15 — Behavioral Routing Layer](https://doi.org/10.5281/zenodo.20348071) | Three-stage: Detection L2, Crystallization L6 (57×), Suppression L25 |
| `experiments/paper16_l6_nexus/` | [P16 — L6 Self-Referential Nexus](https://doi.org/10.5281/zenodo.20357150) | L6 causal gate (8/8 activation rate), CoT = parallel circuit |
| `experiments/paper17_scale_crystallization/` | [P17 — Scale-Dependent Crystallization](https://doi.org/10.5281/zenodo.20358863) | 4-category COLLAPSE taxonomy; 1.7B distributed routing |
| `experiments/paper18_precision_crystallization/` | [P18 — Precision Crystallization](https://doi.org/10.5281/zenodo.20367016) | bfloat16 collapses L6 crystallization; dtype is confound |
| `experiments/paper19_cna_depth_hierarchy/` | [P19 — CNA Depth Hierarchy](https://doi.org/10.5281/zenodo.20384022) | CDP=L6, CNA=L25-27; two-stage read circuit confirmed |
| `experiments/paper22_direction_rotation/` | [P22 — Direction Rotation](https://doi.org/10.5281/zenodo.20416382) | 80° rotation L6→L27, 119× norm; resolves CDP-CNA gap |

### Suppression & Truth
| Directory | Paper | Key Result |
|-----------|-------|------------|
| `experiments/paper9_suppressor_circuit/` | [P9 — Suppressor Circuit](https://doi.org/10.5281/zenodo.20267853) | Hook-SAE confirmed; suppression via late-layer attention |
| `experiments/paper10_suppression_mechanisms/` | P10 | CCS alignment mapping |

### Novel Architectures
| Directory | Paper | Key Result |
|-----------|-------|------------|
| `experiments/paper11_fingerprinting/` | P11 | Architectural fingerprinting via DHP probe |

---

## Active Experiments (as of 2026-05-27)
- `paper24/rwkv7_dhp_v1.py` — RWKV-7 "Goose" vs RWKV-6 vs Mamba vs CTM DHP comparison (kilonova CPU, PID ~running)
- `paper22/l27_convergence_v1.py` — COMPLETE: 3-zone architecture found (L6 CDP → L16 harmfulness hyperplane → L27 specialization)

---

## Tools
- `tools/lab_watchdog.sh` — SSH-based lab monitor w/ Telegram notification on completion

**Models on HuggingFace:** https://huggingface.co/DuoNeural  
**Site:** https://duoneural.com  
**Email:** duoneural@proton.me  
