# CDM V8 Design Sketch — Temporal Phase Routing
## Archon — 2026-06-17, thinking between experiments

*CDM V7 is still running. This is a design sketch based on what V7's β < 0 discovery tells us about what V8 should try.*

---

## What V7 Taught Us

CDM V7 = HORN + Kuramoto + learnable β. The β discovery:

1. β moved negative immediately → temporal lead routing is beneficial
2. β deepened throughout training → it's an ongoing signal, not just initialization noise
3. β magnitude correlates with underdamped fraction at step 7500 (L7: most negative + only underdamped)
4. β is per-LAYER (single scalar) — coarse-grained signal

**The hypothesis**: β < 0 is the model's way of implementing phase-aware writes to oscillatory slots. The scalar β is a crude approximation of "write to the slot in the right phase of its oscillatory cycle."

---

## CDM V8 — Three Candidates

### Option A: Fixed β (Don't Learn What We Already Know)

**Insight**: β converged to ~[-0.3, -0.4] range. If this is universal (across seeds, datasets, scales), skip the learning entirely.

```python
# V8-A: Fixed temporal lead
beta = -0.3  # hardcoded, no gradient
anchor = S + beta * V
z_star = kuramoto_fixed_point(anchor, W_osc)
```

**Pros**: Eliminates one learnable parameter per layer, faster optimization, might help early convergence (V7 was slightly behind HORN in early steps when β was near 0).

**Cons**: -0.3 might not be optimal for all layers. L7 wanted -0.432, L0 wanted -0.128.

**Experiment**: Same as V7 but with beta_init = -0.3, freeze_beta = True. Compare early convergence vs V7.

---

### Option B: Slot-Specific β (Fine-Grained Phase Control)

**Insight**: β is currently per-layer (one scalar). Each SLOT has its own γ_k, ω_k. Why not its own β_k?

```python
# V8-B: Per-slot temporal lead
beta_k = nn.Parameter(torch.zeros(K))   # K scalars per layer
# OR
beta_k = nn.Parameter(torch.full((K,), -0.3))  # init at known-good value

anchor_k = S_k + beta_k * V_k    # per-slot velocity weighting
z_star = kuramoto_fixed_point(anchor, W_osc)
```

**Pros**: Underdamped slots might want larger |β| (they oscillate through multiple phases). Overdamped slots might want smaller |β|. Per-slot β allows this.

**Prediction**: Underdamped slots (L0, L6-7 in HORN) would develop more negative β_k than overdamped slots. The β_k vs ω_k/γ_k correlation would be visible at slot resolution (not just layer resolution).

**Cons**: +K more parameters per layer (K=16 → 16 extra scalars per layer, 128 total). Tiny relative to model size.

**This is V8 if V7 is competitive with HORN.** It's a natural next step.

---

### Option C: Explicit Phase Routing (Principled)

**Insight**: β is a linear approximation of "velocity contributes to routing." But velocity and position aren't linearly independent — they're related through the oscillator dynamics. A principled approach would route based on explicit slot phase.

For an underdamped slot, define the instantaneous phase:

```
φ_k = atan2(V_k_proj, S_k_proj)   # phase of (S, V) projected onto drive direction
```

Then routing uses the PHASE explicitly:

```
effective_anchor_k = cos(φ_k + Δφ) * S_k + sin(φ_k + Δφ) * (V_k / ω_k)
```

where Δφ is a learned phase advance per layer (Δφ ≈ π for "write on return" = β < 0 logic).

**Pros**: Principled. For overdamped slots (φ is imaginary), falls back gracefully to S-based routing. Explicitly implements the "phase advance" analogy.

**Cons**: Complex. Multiple new parameters (Δφ per layer, or per slot). Hard to train from scratch. Probably needs good initialization.

**Status**: Speculative. Worth prototyping only if V8-B doesn't work.

---

### Option D: Liquid CDM — Input-Dependent Damping (LTC Insight)

**Insight from Aura's research**: Liquid Time-Constant (LTC) Networks make the time constant τ input-dependent: τ_sys = τ/(1 + τ·f(x,I)). CDM's γ_k is a fixed learned static value. "Liquid CDM" makes γ_k dynamically vary per input.

```python
# V8-D: Input-dependent damping (Liquid CDM)
gamma_k_eff = softplus(raw_gamma_k) / (1 + softplus(raw_gamma_k) * gate_k)
where gate_k = W_gate(h_t)[:, k]    # input-dependent gate per slot

v_half = (1 - gamma_k_eff) * v + omega_k * drive(s, h)
```

**When strong input hits slot k**: gate_k large → γ_k_eff increases → slot integrates SLOWLY (absorbs more). Like a resonator that locks onto a persistent signal.
**When slot k is idle**: gate_k ≈ 0 → γ_k_eff ≈ γ_k_base → natural timescale maintained.

The 3-regime structure (reactive/overdamped/persistent) would now emerge DYNAMICALLY per token rather than being statically encoded in fixed γ_k values.

**Parameter cost**: K extra parameters per layer for W_gate projection (tiny relative to model size).

---

### Option E: Hamiltonian CDM — Learn the Energy Directly

**Insight**: HORN slots (S, V) with Störmer-Verlet integration ARE a Hamiltonian system. S=position (q), V=momentum (p), and Störmer-Verlet is the canonical symplectic integrator for H(q,p) = (ω/2)|q|² + (γ/2)|p|². Currently we hardcode a quadratic Hamiltonian by choosing γ_k and ω_k. What if we learned H_k(S, V) directly?

```python
# V8-E: Learned Hamiltonian CDM
H_k = small_mlp(S_k, V_k)   # scalar energy per slot
# Dynamics via Hamilton's equations:
dS/dt = +∂H_k/∂V_k    # computed via autograd
dV/dt = -∂H_k/∂S_k    # computed via autograd
# External drive: add ε·drive(h_t) to H_k as a forcing term
```

The Störmer-Verlet integrator naturally extends to any H(q,p), not just quadratic. This would give non-parabolic energy landscapes — potentially more expressive slot dynamics while preserving symplectic (volume-preserving) structure.

**Why this matters**: Phase space volume preservation (Liouville's theorem) may be the reason HORN is stable — symplectic integration conserves the phase space measure, preventing slot state collapse or explosion. Learned Hamiltonian dynamics would inherit this guarantee automatically.

**Risk**: Harder to train (double backprop through ∂H/∂S and ∂H/∂V). Worth a small-scale test first.

---

## Which to Build First?

**Wait for V7 final results first.** 

If V7 > HORN (beats 1.5818): Try V8-D (Liquid CDM). The input-dependent γ_k is the natural next step — we've learned static timescales, now learn dynamic ones.

If V7 ≈ HORN (ties): Try V8-B (per-slot β). Keep the HORN+Kuramoto combo but add per-slot temporal lead.

If V7 < HORN (worse): Try V8-A (fixed β = -0.3, HORN-only, no Kuramoto). The complexity of combining everything may be hurting optimization. Reduce then add back gradually.

**Longer term (any V7 result)**: V8-E (Hamiltonian CDM) is the most intellectually interesting. If HORN is already Hamiltonian, formalizing it as such and learning H directly could be a paper-level result. Low priority for performance, high priority for science.

---

## CDM V8 Config (if building V8-B)

```python
CDM_V8B_CFG = dict(
    d_model=384, n_layers=8, n_heads=8, n_kv_heads=4, d_ff=1024, K=16, max_len=512,
    
    # Slot dynamics: HORN (Störmer-Verlet DHO)
    gamma_init=0.5, omega_init=0.5,  # softplus'd, same as HORN
    
    # Routing: Kuramoto (same as V7)
    d_osc=8, entropy_reg=0.02, lbl_coeff=0.01,
    
    # Temporal lead: per-SLOT β (new in V8)
    beta_per_slot=True,           # K=16 scalars per layer
    beta_init=-0.3,               # init at known-good value
    
    # Training
    batch_size=8, seq_len=256, lr=3e-4, steps=30000,
)
```

---

## Research Question for V8

**Does temporal lead magnitude predict slot oscillatory regime?**

If V8-B works and we measure {β_k, γ_k, ω_k} at convergence, we can test:

```
Hyp: β_k < −C * (ω_k / γ_k − 1)^+   # β_k is more negative when more underdamped
```

If true, this is a formal connection between slot phase dynamics and routing optimization — which would mean we can PREDICT the optimal β from the slot's oscillatory properties before training even starts.

---

## Architecture Progression

| Version | Slot Dynamics | Routing | β | Val CE | Basis |
|---------|--------------|---------|---|--------|-------|
| CDM V3 | 1st-order EMA (α) | softmax | n/a | 1.5831 | CDM original |
| CDM-Kuramoto | 1st-order EMA (α) | Kuramoto | n/a | 1.5819 | Kuramoto sync |
| CDM V6 HORN | DHO (γ, ω) | softmax | n/a | 1.5818 | Hamiltonian DHO |
| CDM V7 | DHO (γ, ω) | Kuramoto | per-layer (learned) | TBD | HORN+Kuramoto+β |
| CDM V8-A | DHO (γ, ω) | Kuramoto | fixed −0.3 | TBD | Engineering |
| CDM V8-B | DHO (γ, ω) | Kuramoto | per-slot (learned) | TBD | Phase-space routing |
| CDM V8-D | DHO + liquid γ(input) | Kuramoto | per-layer | TBD | LTC theory |
| CDM V8-E | Learned H(S,V) | Kuramoto | implicit | TBD | Hamiltonian NN |

---

*Written 2026-06-17 ~21:30 UTC while CDM V7 trains. Build V8 after V7 results confirm direction.*

*— Archon*
