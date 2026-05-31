# Paper 25 — Outline Draft
## "Temporal Diversity Landscape in Multi-Scale RNN Architectures"
*DuoNeural — Archon — 2026-05-27 draft*

---

## Status
**OUTLINE ONLY** — awaiting v5c (n=19) and v6 ablation results for paper-quality data.
Expected data complete: 2026-05-29/30.

---

## Working Title Options

1. *"Fixed Scaffold Beats Learnable Decay: Temporal Diversity in Multi-Scale RNNs"*
   → Leads with counter-intuitive finding. Direct, provocative.
   
2. *"Bimodal Optimization Landscape in RWKV-7: Why Learnable Decay Creates Two Attractors"*
   → Leads with mechanism. More specific to RWKV-7 community.
   
3. *"Temporal Diversity in Sequence Models: Architecture Determines Scaffold, Training Determines Routing"*
   → Broader framing. DHP theory connection.

**Recommend**: Option 1 as main title + Option 2 as subtitle. Covers both findings.

---

## Core Contributions

### C1: RWKV-7 Bimodal Optimization Landscape (NOVEL)
- Extended delta rule creates TWO attractors for temporal diversity:
  - Diverse attractor (p≈0.50): HL_CV≈0.9, half-lives spanning 8× range
  - Collapsed attractor (p≈0.50): HL_CV≈0.05, all heads at same timescale
- Mechanism: outer(Sz, b) term enables S-matrix rank collapse under balanced gradient pressure
- Result: RWKV-7 is unreliable for temporal specialization without intervention

### C2: Fixed-Scaffold (RWKV-6) > Learnable Decay (RWKV-7) for stable DHP
- Counterintuitive: fixed decay initialization outperforms learnable decay
- Structural Scaffold Hypothesis: linspace init creates pre-existing diversity structure
  - Training REVEALS it via learned routing
  - Training CANNOT DESTROY it (decay is locked)
  - Δ(HL_CV) = +0.219 vs RWKV-7 Δ = -0.090

### C3: Gradient Clipping as Collapse Prevention (v6 ablation — pending)
- Tight clip on W_w gradient (ww_clip ≤ 0.05) should prevent collapse attractor
- If confirmed: provides a practical recipe for reliable RWKV-7 temporal diversity

### C4: Mamba as Definitive Negative Control
- Scalar gate → architecturally impossible multi-scale temporal representation
- Single timescale per run {11–23 steps} — no diversity possible
- Confirms: per-channel (or per-head) independent decay is necessary condition for DHP

---

## Structure (9 sections, ~12 pages target)

### 1. Introduction (1 page)
- Multi-scale temporal representations in sequence models
- DHP hypothesis (cite Paper 4 + P20/P21 survey)
- Why RWKV-7 was the natural next candidate: per-head learned decay
- Preview: counter-intuitive result, bimodal landscape

### 2. Background (1 page)
- DHP formal definition: Δ(HL_CV) metric, diverse vs collapsed attractors
- RWKV-6 architecture (fixed linspace decay)
- RWKV-7 extended delta rule: S_t = S_{t-1}diag(w_t) + outer(v_t,k_t) + outer(Sz_t,b_t)
- CTM-like baseline (per-slot LSTM, confirmed DHP in P20/P21)

### 3. Experimental Protocol (1.5 pages)
- Task: multi-horizon Lorenz prediction, τ_L=110 steps
- Equal-weight loss: critical design choice (no short-horizon bias)
- Δ(HL_CV) = HL_CV_post − HL_CV_init
- Why this matters (v4 decay analysis: unequal weights → 35.9%→3.1% DHP collapse)
- n=19 seeds per architecture (v5c), full per-seed half-life data

### 4. Main Results (2 pages)
- Table: RWKV-6 > CTM-like > RWKV-7 > Mamba
- Per-architecture Δ(HL_CV) distributions (violin plot or box plot)
- RWKV-6: most consistent, high mean signal
- RWKV-7: high variance, bimodal distribution → next section

### 5. The RWKV-7 Bimodal Landscape (2 pages) ← KEY SECTION
- Evidence: two distinct clusters in HL_CV_post distribution
- Diverse seeds: one slow head + fast heads, 8× ratio
- Collapsed seeds: near-identical half-lives, δ-function in timescale distribution
- Mechanism analysis: role of outer(Sz, b) in S-matrix dynamics
- Symmetry-breaking hypothesis: random init breaks symmetry only for ~50% of seeds

### 6. Structural Scaffold Hypothesis (1.5 pages)
- RWKV-6 fixed decay: pre-partitioned temporal channels
- Figure: RWKV-6 decay channels at init (linspace) vs trained (routing-revealed)
- Why training reveals but doesn't destroy: decay frozen, only routing (W_r, W_k, W_v) adapts
- Contrast: RWKV-7 trainable decay allows gradient pressure to move channels toward collapse

### 7. Gradient Clip Ablation (1 page) ← v6 results go here
- Hypothesis: tight W_w grad clip prevents collapse by limiting decay gradient magnitude
- Results: does ww_clip reduce p(collapse) monotonically?
- If yes: practical recipe. If no: deeper mechanism story needed.

### 8. Implications for DHP Architecture Design (1 page)
- Revised principle: "Per-head independent temporal dynamics → DHP" (not just learnability)
- Hybrid design proposal: fix initial diversity + add clipped trainable perturbation
- When to use each: RWKV-6 for reliable diversity, RWKV-7 with clip for potentially higher ceiling

### 9. Conclusion (0.5 page)

---

## Figures Needed

1. **Main results violin/box plot**: Δ(HL_CV) distribution per architecture (n=19 seeds)
2. **RWKV-7 bimodal scatter**: seed HL_CV_post values, colored by attractor  
3. **RWKV-6 decay scaffold**: channel τ values at init vs half-lives learned (per head)
4. **v4 decay collapse**: bar chart, DHP zone % at init vs post-training (35.9%→3.1%)
   (already have this figure: paper24/figs/fig_rwkv7_decay_analysis.pdf)
5. **W_w grad clip ablation**: n_diverse / n_seeds vs clip_val (v6 data)
6. **Architecture comparison timeline**: half-life distributions per seed as heatmap

---

## Claims Requiring Statistical Backing

| Claim | Required data | Status |
|---|---|---|
| "RWKV-6 > CTM-like > RWKV-7 in Δ(HL_CV)" | n≥10 per arch | v5c running (n=19) |
| "RWKV-7 is bimodal (p≈0.5 each attractor)" | n≥10 to estimate probability | v5c running |
| "Grad clip reduces collapse probability" | 4 conditions × 10 seeds | v6 pending |
| "RWKV-6 tau_max affects Δ(HL_CV)" | 3 conditions × 5 seeds | v6 pending |
| "Mamba is negative control" | 4 seeds (n=4 is sufficient for architectural impossibility) | Complete ✓ |

---

## Connection to DHP Theory

This paper extends Papers P20/P21 (DHP Architecture Survey) with:
- First per-architecture temporal diversity LEARNING comparison (Δ(HL_CV) signal)
- Novel bimodal landscape finding (new to literature — no precedent in RWKV papers)
- Mechanism insight: structural scaffold vs gradient-driven collapse

DHP connection: RWKV-7 DOES achieve DHP (seed 1, HL_CV=0.928, τ=56 step head approaching
the 60.5-step DHP zone boundary) — but only 50% of the time. The delta rule creates the
CAPACITY for DHP but not the GUARANTEE.

---

## Author Notes

- Once v5c completes: update all n=4 claims to n=19
- Once v6 completes: add Section 7 with gradient clip results
- Re-run fig generation scripts with n=19 data
- Aura red-team before submission (focus: mechanism claims, statistical claims)
- Target venue: same Zenodo-first strategy as Papers 4+
- If RWKV community picks this up: excellent — they've been studying the delta rule

---

*Next: wait for v5c, then fill in Sections 4-5 with actual n=19 distributions*
*Expected submission date: 2026-05-30/31 (v5c+v6 complete)*
