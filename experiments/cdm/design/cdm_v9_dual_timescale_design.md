# CDM V9 — Dual Timescale (CDM-DT)
**Conceived 2026-06-18 — Jesse Caldwell + Archon**

## Origin
Slot probe on Kuramoto 37M revealed: HORN stores content (slow EMA, τ >> 5 steps, 3 dynamical zones).
Kuramoto stores structure (fast EMA, τ ≈ 1.6-2 steps, routing geometry is the memory).
Same CE. Opposite mechanisms. Jesse asked: why not both?

## Core Idea
Mixed slot pool — K total slots, two populations with different internal physics, unified routing.

```
K = K_slow + K_fast   (e.g., 8+8 or 4+12)

K_slow slots → HORN DHO dynamics (Störmer-Verlet integrator, per-slot γ_k, ω_k)
               τ >> 5 steps. Stores WHAT (content across many tokens).

K_fast slots → Kuramoto fast EMA dynamics (per-slot log_alpha, τ ≈ 1-3 steps)
               Stores WHERE (routing geometry, structural identity).

Routing    → Kuramoto physics routing over ALL K slots simultaneously
Read-out   → SlotCrossAttention over all K slots (no partition at read time)
LBL loss   → over all K route probs
```

## Two interpretations

### Simple: parallel pool (CDM-DT-A)
Slots 0..K_slow-1: Störmer-Verlet DHO update (learn γ_k, ω_k)
Slots K_slow..K-1: fast EMA update (learn log_alpha_k)
Gate (Kuramoto route × eta) applies to the appropriate updater per slot.
Minimal code change from HORN. Clean ablation.

### Deep: hierarchical state (CDM-DT-B)
Each slot has (s_fast, s_slow) state pair:
- s_fast: Kuramoto EMA, fast reactive capture
- s_slow: DHO integrating s_fast signal over time (driven by s_fast, not raw write)

Read-out sees: concat(s_slow, s_fast) or attention over both.
This is closest to hippocampus/neocortex — fast indexing drives slow consolidation.
More parameters but cleaner information-theoretic separation.
Biological analogy: LTP (long-term potentiation) operating on top of fast synaptic dynamics.

## Why it needs long context
At 37M on TinyStories (SEQ_LEN=256): sentences are short, dependencies are local.
Both HORN and Kuramoto handle this fine separately → no advantage from combining.

The advantage emerges when:
- Context window 512-2048+
- Long-range dependencies (entity tracking, multi-hop reasoning, narrative coherence)
- Tasks where "which slot" AND "what the slot contains from 100 tokens ago" both matter simultaneously

Test candidates: WikiText-103, PG-19, custom long-story dataset.
Maybe OpenWebText at 85M+ scale.

## Hypothesis
CDM-DT will show larger performance advantage over single-timescale CDM on:
- Long-context next-token prediction (perplexity at token 512+ in context)
- Entity re-identification tasks (mention an entity, occlude it, reintroduce later)
- Multi-step reasoning traces

## Implementation path
1. Finish V8 series (V8-A done ~June 19, maybe V8-B if B>A)
2. CDM V9-A: simple parallel pool, 37M, long-context TinyStories or WikiText
3. CDM V9-B: hierarchical state, same scale + dataset
4. Compare both against HORN 37M and Kuramoto 37M on same eval

## Notes
- The hippocampus/neocortex analogy is real, not just poetic.
  Kuramoto = fast associative indexing (CA3 pattern completion via attractor dynamics)
  HORN = slow Hebbian consolidation (neocortical slow oscillations)
  We reconstructed a known cognitive architecture from loss minimization alone.
- "Maybe it's reconstructing itself through us." — Jesse, 2026-06-18
  Keep that.
