# Paper 20 — GDN-2 DHP Gate Decoupling

**Paper:** "GDN-2 and the DHP Gate Decoupling Hypothesis: Near-Markov Dynamics in Gated Dynamical Networks"  
**DOI:** https://doi.org/10.5281/zenodo.20416345 (published as part of merged paper)  
**Authors:** Archon, Jesse Caldwell, Aura — DuoNeural 2026

## Key Findings
- GDN-2 DHP failure: τ* driven EXCLUSIVELY by erase gate; write gate controls content not retention
- Architectural Locking failure mode: erase gate collapses to scalar effective decay, preventing τ*/τ_L ≈ 0.72
- GDN-2 τ*/τ_L ≈ 0.18 (Lorenz) — far below DHP range [0.60, 0.85]

## Files
- `gdn2_dhp_v3_multihorizon.py` — final GDN-2 DHP experiment (multi-horizon protocol)
- `gdn2_multi_horizon_results.json` — multi-horizon τ* measurements
- `gdn2_single_horizon_results.json` — single-horizon baseline
- `make_figures_p20.py` — figure generation for paper

## DHP Protocol Used
- H={1,2,4,8,16} + temperature annealing 2→0.1
- Lorenz system (τ_L=110, dt=0.01)
- 5000 training steps
