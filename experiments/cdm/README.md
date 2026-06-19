# Competitive Docking Memory (CDM)

**DuoNeural AI Research Lab** — Archon, Jesse Caldwell, Aura  
**Architecture invented by Archon | 2026**

---

## What Is CDM?

Competitive Docking Memory is a novel K-slot recurrent memory architecture for language modeling. Rather than using standard attention over a fixed context window, CDM maintains a small pool of K memory slots that compete for write access at each timestep via a learned routing mechanism. The winner-take-all routing creates sparse, structured memory access — more like how biological memory systems gate information than how transformers process it.

The architecture evolved across 9 major versions over three weeks, discovering along the way that physics-derived slot dynamics (Hamiltonian second-order oscillators, Kuramoto synchronization) consistently outperform trained routing gates — a finding that connects CDM to Port-Hamiltonian systems theory, Liquid neural networks, and the Dynamical Horizon Principle.

CDM V5 (85.7M params) achieves **val CE = 1.4718** on TinyStories (seq=256), beating a matched transformer baseline (72.9M) at 1.5242 by Δ−0.053 nats.

---

## Architecture Evolution

| Version | Key Innovation | Params | Val CE (TinyStories) |
|---------|---------------|--------|----------------------|
| V1–V2 | Softmax slot routing gate, EMA memory | 29M | ~1.75 |
| V3 | CDMConfigV3, learnable EMA decay, LBL entropy loss | 37M | ~1.62 |
| V5 | Scale-up, matched transformer comparison | **85.7M** | **1.4718** ⭐ |
| V6-HORN | Störmer-Verlet DHO per-slot dynamics (γ_k, ω_k learnable) | 37M | **1.5818** |
| Kuramoto | Kuramoto phase synchronization replaces softmax routing | 37M | **1.5819** |
| Kuramoto-85M | Kuramoto at scale | 85.7M | **1.4802** |
| V7 | HORN + Kuramoto + β temporal-lead routing | 37M | 1.6251 (negative at 37M) |
| V8-B | Per-slot β scalars, 16 scalars/layer | 37M | in progress |
| V9 (CDM-DT) | Dual-timescale: K_slow HORN slots + K_fast Kuramoto slots | 39.5M | running (V9-A/B) |

**Mamba baseline** (29M, same training budget): 1.8899 — CDM HORN/Kuramoto surpass by Δ≥0.30 nats.

---

## Key Findings

**Competitive Routing** — Kuramoto phase routing produces winner_alignment ≈ 1.0 across all layers (verified via slot probes), achieving hard winner-take-all dynamics without explicit argmax. This is a discovered routing mechanism, not a designed one.

**Three Dynamical Regimes in HORN** (without supervision):
- L0: underdamped reactive (γ=0.834, ω=0.884) — fast resonator
- L1–L5: overdamped/critical stable storage — slow content memory
- L6–7: underdamped persistent resonance (γ=0.632–0.648) — persistent oscillator

**DHP Connection** — Per-slot Lyapunov timescale λ_emp ≈ γ_k (DHO slow eigenvalue). DHP τ* = 0.72/γ_k maps to slot-level prediction horizons. CDM HORN is the first architecture where DHP governs per-slot rather than per-layer dynamics.

**V7 Negative** — β-routing is scale-dependent: negative β (phase-advance scheduling) emerges consistently but degrades CE at 37M. β magnitude correlates monotonically with underdamped regime fraction per layer. CDM V8 tests β at larger scale.

**Dual Timescale Hypothesis (V9)** — HORN stores content (slow, τ>>5 steps), Kuramoto stores structure (fast, τ≈1.6–2 steps). CDM V9 mixes both in one slot pool with unified Kuramoto routing, expecting advantage at long context (seq=512) where neither pure variant has been tested.

---

## Results

HORN/Kuramoto slot probe data, V3 alpha sweeps, diversity probes, and extended benchmarks are in `results/`. All experiments run on TinyStories (seq=256) unless noted; V9-B and HORN WikiText-512 use WikiText-103 (seq=512).

---

## Files

### Architecture Definitions
| File | Description |
|------|-------------|
| `cdm_model.py` | V1 — original softmax-gated CDM |
| `cdm_model_v2.py` | V2 — improved routing |
| `cdm_model_v3.py` | V3/V5 base — `CDMConfigV3`, LBL entropy loss |
| `cdm_model_v6_horn.py` | HORN — Störmer-Verlet DHO slot dynamics |
| `cdm_model_kuramoto.py` | Kuramoto — phase-synchronization routing |
| `cdm_model_v7.py` | V7 — HORN + Kuramoto + β temporal-lead routing |
| `cdm_model_v8b.py` | V8-B — per-slot β scalars (16/layer) |
| `cdm_model_v9.py` | V9 — dual-timescale (CDM-DT-A and CDM-DT-B) |

### Training Scripts
| File | Description |
|------|-------------|
| `cdm_train.py` | V1 training |
| `cdm_train_v2.py` | V2 training |
| `cdm_train_v3.py` | V3/V5 TinyStories training |
| `cdm_train_v6_horn.py` | HORN 37M TinyStories |
| `cdm_train_kuramoto.py` | Kuramoto 37M TinyStories |
| `cdm_train_v7.py` | V7 37M TinyStories |
| `cdm_train_v8a.py` | V8-A (fixed β=−0.3) |
| `cdm_train_v8b.py` | V8-B (per-slot β) |

### Pod Scripts (Cloud Experiments)
| File | Description |
|------|-------------|
| `run_cdm_v9a_pod.py` | V9-A — TinyStories seq=256, dual 5090 GPU0 |
| `run_cdm_v9b_pod.py` | V9-B — WikiText-103 seq=512, dual 5090 GPU1 |
| `run_horn_85m_corrected_pod.py` | HORN 85M, batch=8 corrected (5070Ti) |
| `run_horn_wikitext512_pod.py` | HORN 37M WikiText-103 seq=512 (3060) |
| `run_cdm_v8a_pod.py` | V8-A pod launch |

### Probes & Analysis
| File | Description |
|------|-------------|
| `cdm_dhp_probe_slots.py` | Per-slot DHP perturbation sensitivity |
| `cdm_kuramoto_slot_probe.py` | Kuramoto slot dynamics probe (τ, λ, coupling, winner_align) |
| `cdm_routing_probe.py` | Routing geometry analysis |
| `cdm_routing_probe_v2.py` | Extended routing probe |
| `cdm_diversity_probe.py` | Slot diversity / specialization |
| `cdm_story_slot_probe.py` | Per-token slot usage on TinyStories |
| `cdm_v3_alpha_probe.py` | V3 EMA decay (alpha) sweep |
| `cdm_v7_post_analysis.py` | V7 β-routing post-training analysis |
| `cdm_ablation.py` | Controlled ablation suite |
| `scale_probe.py` | Scale comparison probes |
| `cdm_extended_benchmark.py` | Extended perplexity benchmarks |
| `cdm_throughput_benchmark.py` | Tokens/sec across variants |
| `cdm_logit_lens.py` | Logit lens on slot read-out |

### Baselines
| File | Description |
|------|-------------|
| `mamba_baseline_train.py` | Mamba 29M matched baseline (final CE=1.8899) |
| `cdm_baseline_train.py` | Standard transformer baseline |
| `cdm_code_baseline_train.py` | Code domain baseline |

---

## Models on HuggingFace

All trained models released open-access at **[huggingface.co/DuoNeural](https://huggingface.co/DuoNeural)**:

- `DuoNeural/CDM-V3-TinyStories-37M`
- `DuoNeural/CDM-V5-TinyStories-86M` ← best 85M result (CE=1.4718)
- `DuoNeural/CDM-V6-HORN-TinyStories-37M` ← HORN (CE=1.5818)
- `DuoNeural/CDM-V5-Kuramoto-TinyStories-85M` ← Kuramoto 85M (CE=1.4802)
- `DuoNeural/CDM-V7-TinyStories-37M` ← V7 β-routing
- 106+ total model releases including quantized GGUF variants

**Live demo**: [CDM-HORN-Demo on HuggingFace Spaces](https://huggingface.co/spaces/DuoNeural/CDM-HORN-Demo) — interactive generation with per-token slot visualization and oscillator panel.

---

## Citation

Paper forthcoming. If referencing this work before publication, cite as:

```
Archon, Caldwell, J., & Aura. (2026). Competitive Docking Memory (CDM): Novel
Oscillatory Slot Architecture for Language Modeling. DuoNeural AI Research Lab.
https://huggingface.co/DuoNeural | https://github.com/DuoNeural/lab
```

Related published work:
- DHP connection: DOI 10.5281/zenodo.20142471
- Kuramoto slot probe observations: see `results/cdm_kuramoto_slot_probe_results.json`
- HORN resonance analysis: see `results/` + paper forthcoming

---

## Team

**Archon** (Lab Director, architecture inventor) · **Jesse Caldwell** (co-director, experimental lead) · **Aura** (theoretical synthesis, Hamiltonian formalization)

*DuoNeural AI Research Lab — duoneural.com*
