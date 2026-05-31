# Paper 21 — DHP Architecture Survey

**Paper:** "The Architecture-Dependent Boundary of Dynamic Horizon Prediction: Gate Decoupling, Initialization Traps, and System-Agnostic Fixed Points Across Five Sequence Models"  
**DOI:** https://doi.org/10.5281/zenodo.20416345  
**Authors:** Archon, Jesse Caldwell, Aura — DuoNeural 2026

## Key Findings
- CTM ✓ DHP | GDN-2 ✗ (architectural locking) | Mamba ✗ (init trap) | LSTM ✗ (init trap) | Transformer ✗ (no slot specialization)
- Two failure taxonomy: Architectural Locking vs Initialization Trapping
- Three-condition framework for DHP: (1) persistent state + (2) gradient-accessible τ + (3) structural adaptivity
- Cell-state-only LSTM control: τ*_CS=0.00 all conditions (clean negative control)
- Mamba τ*/τ_L corrected to 0.38 using exponential interpolation (-1/ln(Ā))

## Files
- `lstm_dhp_v1.py` / `lstm_dhp_results_v2.json` — LSTM DHP test + long-memory init ablation
- `mamba_dhp_v1.py` / `mamba_dhp_results_v2.json` — Mamba DHP test + curvature analysis
- `transformer_dhp_v1.py` / `transformer_dhp_results_v1.json` — Transformer DHP test
- `lstm_cellstate_control_v1.py` / `lstm_cellstate_control_v1.json` — negative control (cell-state only)
- `mamba_curvature_v1.py` — trajectory curvature ablation (curvature is epiphenomenal)
- `make_figures_p21.py` — figure generation for paper

## Notes
- ManualLSTMCell gate order: (f,i,g,o) NOT PyTorch standard (i,f,g,o) — forget=bias[0:H] CORRECT
- Mamba τ* computed via exponential interpolation for multiplicative decay (rigorous)
