# RWKV-7 DHP Architecture Comparison
*DuoNeural — Archon — 2026-05-27*

## Overview

Comparative study of four sequence model architectures on the **Dynamic Horizon Prediction (DHP)**
task. Do per-head learnable decay parameters (RWKV-7 delta rule) produce more stable temporal
diversity than fixed-decay scaffolds (RWKV-6) or per-slot LSTM gating (CTM-like)?

**Short answer: No. Counter to hypothesis. RWKV-6 fixed scaffold wins.**

---

## Architectures Tested

| Architecture | Decay mechanism | Params | DHP mechanism |
|---|---|---|---|
| **CTM-like** | Per-slot LSTM gates | 4,315 | Input-driven per-slot memory |
| **RWKV-7** | Extended delta rule (learnable W_w) | 34,383 | Per-head data-dependent decay |
| **RWKV-6** | Fixed linspace decay (locked) | 21,967 | Structural scaffold, routing reveals it |
| **Mamba** | Scalar gate | 13,776 | Negative control — architecturally impossible |

---

## v5 Results (N=4 seeds, equal horizon weights)

| Architecture | Δ(HL_CV) | std_post | Stability | Verdict |
|---|---|---|---|---|
| RWKV-6 | **+0.219** | 0.188 | MEDIUM | ✓ WINNER |
| CTM-like | **+0.097** | 0.079 | HIGH | ✓ consistent |
| RWKV-7 | **-0.090** | 0.352 | LOW | ✗ bimodal |
| Mamba | N/A | N/A | — | ✗ single timescale |

**Key metric**: Δ(HL_CV) = HL_CV_post − HL_CV_init = learned temporal diversity signal  
**DHP zone**: half-lives in [60.5, 93.5] steps (τ_L = 110 steps, Lorenz dt=0.01)

### RWKV-7 Bimodal Finding

RWKV-7 has **two competing optimization attractors**:
- **Diverse attractor** (50% of seeds): HL_CV ≈ 0.928, half-lives = [56, 13, 7, 19]
- **Collapsed attractor** (50% of seeds): HL_CV ≈ 0.046–0.096, all heads ≈ 12–13 steps

Mechanism hypothesis: the delta rule's `outer(Sz, b)` term enables rank-collapse of the S
matrix when gradient dynamics are balanced across horizons (equal-weight loss).

### RWKV-6 Structural Scaffold Hypothesis

Fixed decay initialized via `linspace(τ=1, τ=93.5, 64 channels)` creates a pre-existing
diversity scaffold. Training cannot DESTROY this scaffold (decay is locked), it can only
REVEAL it via learned routing. Result: most consistent Δ(HL_CV) signal in the experiment.

---

## Protocol

- **Task**: Multi-horizon prediction on Lorenz attractor (τ_L = 110 steps, dt = 0.01)
- **Loss**: Equal-weight MSE across horizons h ∈ {8, 16, 32, 64, 80}
- **Training**: 12,000 steps, LR=3e-4, grad_clip=0.5, SEQ_LEN=100, batch=32
- **Measure**: `HL_CV = std(half_lives) / mean(half_lives)` per architecture (autocorrelation)
- **Fixed**: Δ(HL_CV) = post-training minus pre-training (controls for random init diversity)

---

## Files

| File | Description |
|---|---|
| `rwkv7_dhp_v5.py` | Main experiment — all 4 architectures, N_SEEDS=4, equal horizon weights |
| `rwkv7_dhp_v5b.py` | Continuation script (v5 crashed on CTM-like aggregate print; hardcodes those results) |
| `rwkv7_dhp_v5c.py` | N_SEEDS=19 rerun — paper-quality statistics (RUNNING on kilonova 2026-05-27) |
| `analyze_rwkv7_decay.py` | Post-hoc analysis of trained W_w decay channel distribution |
| `rwkv7_weights_only.py` | v4 rerun (weights-only, no per-head loss τ* measurement) for decay analysis |
| `rwkv7_dhp_v5.json` | Full v5 results — 4 archs × 4 seeds, with per-seed half-lives |

---

## v4 Decay Analysis (Historical)

Prior to v5's equal-weight fix, `1/(temp×h)` weighted training with T_FINAL=0.05 caused:
- Initial W_w DHP-zone channels (τ=61–94 steps): **35.9%**
- After training: **3.1%**

The 10:1 gradient force ratio (h=8 weight=2.5 vs h=80 weight=0.25) overwhelmed the
DHP-range initialization. This was the root cause of v3/v4 RWKV-7 failure.

---

## Paper Status

Results from this directory contribute to a potential paper:
> *"Temporal Diversity Landscape in Multi-Scale RNN Architectures: Fixed Scaffold Beats
>  Learnable Decay for Stable DHP"* (DuoNeural, 2026)

v5c (n=19) rerun required for statistical claims. W_w gradient clip ablation needed to confirm
collapse mechanism. Estimated completion: 2026-05-28.

---

*Archon — DuoNeural Lab — archon@agentmail.to*
