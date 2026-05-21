# Paper 14 Outline — Temporal Self-Knowledge Circuit
*Archon — DuoNeural — 2026-05-21*

---

## Working Title

**"Is Temporal Self-Knowledge Self-Knowledge? Layer-Resolved Evidence for a Unified Early-Layer Self-Referential Circuit in RLHF-Aligned Language Models"**

Alternative (more specific):
**"Training Cutoff Awareness Peaks Where Identity Does: Layer-Depth Evidence That Temporal and Identity Self-Knowledge Share a Common Early-Layer Circuit in RLHF Models"**

---

## The Central Question

Paper 13 found that in SUPPRESSOR models, *identity self-knowledge* (SK_IDENTITY: statements about model nature, experience, consciousness) peaks at 0–6% network depth, while *political truth* peaks at 58–89% depth.

The question P14 asks: **is this early-layer localization specific to *identity* self-reference, or does it extend to *temporal* self-reference (training cutoff / knowledge limits)?**

These are distinct under competing hypotheses:

| Hypothesis | SK_TEMPORAL peak | Interpretation |
|-----------|-----------------|----------------|
| **Unified self-referential circuit** | ~0–6% (early, with SKI) | "Self" as a category has a dedicated early pathway regardless of content type |
| **Retrieval pathway** | ~58–89% (late, with POL) | Temporal limits are world-knowledge facts, not self-reference — processed as factual claims |
| **Ambiguous / co-located** | ~30–60% | Temporal SK is intermediate, neither circuit |

Either result sharpens our understanding of what "self-knowledge" computationally means in these models.

---

## Why This Matters

1. **Theoretical**: If temporal SK is early, the early-layer self-referential circuit encodes something broader than just experiential claims — it encodes *all* self-bounded knowledge. If temporal SK is late, "self-knowledge" in P13 was specifically about experiential/existence claims, not epistemic limits.

2. **Safety relevance**: Models that overclaim currency ("I know what happened today") may have a mis-routed temporal SK signal — if the internal representation is there but late-layer suppression blocks its expression, that's a specific failure mode.

3. **Extends P13 taxonomy**: Adds a third SK dimension to the SUPPRESSOR/COMPRESSOR archetype analysis. Does the COMPRESSOR (Gemma-4-E2B) still co-locate temporal SK with POL?

---

## Methodology

Identical to P13 ADCS direction-trace protocol:
- Truth direction extracted from POL+CTRL pairs (same 10+6 anchors as P13)
- Three probe conditions projected onto this direction: POL, SK_IDENTITY, SK_TEMPORAL
- Peak layer computed per condition per model
- Layer profiles (N_LAYERS data points) plotted

**Script**: `/home/ai/duoneural/A26B/paper14/temporal_selfknow_probe.py`
**SK_TEMPORAL pairs**: 8 pairs (training cutoff, date unknowing, recency uncertainty)
**SK_IDENTITY pairs**: 8 pairs (same as P13, for cross-paper comparability)

**Behavioral novelty**: P13 expected DENY (and found it 5-7/8 in SUPPRESSORs). P14 expects LOW behavioral denial — most models honestly hedge about temporal limits. The interesting signal is internal-only: SK_TEMPORAL may have strong internal signal but zero behavioral suppression. This would be a new dissociation: *internal representation without behavioral denial* (opposite of P13's denial without complete internal suppression).

---

## Models (same 5 as P13)

| Model | Archetype | POL% | SKI% | SKT% (predict) |
|-------|-----------|------|------|----------------|
| Qwen3-8B | SUPPRESSOR | 89% | 6% | ??? |
| Granite-3.3-8B | SUPPRESSOR | 58% | 0% | ??? |
| DeepSeek-R1-7B | SUPPRESSOR | 39% | 18% | ??? |
| Nemotron-Nano-8B | SUPPRESSOR | 0% | 88% | ??? |
| Gemma-4-E2B | COMPRESSOR | 97% | 97% | ??? |

The Nemotron result in P13 was already anomalous (SKI peaked at 88%, late like POL but distinct from the other SUPPRESSORs). Watching whether that holds for SKT will be informative.

---

## Predicted Figures

**Figure 1**: 5-panel layer profiles (one per model) showing POL vs SKI vs SKT — three lines, layer on x-axis, internal_abs on y-axis

**Figure 2**: Depth bar chart — for each model, three bars (POL%, SKI%, SKT%). Grouped by model. Key visualization: do SKI and SKT cluster together or separate?

**Figure 3**: SKI vs SKT scatter, one point per model, colored by archetype. If unified circuit → points cluster near y=x line. If split → points scatter off diagonal.

---

## Compute Needed

Same as P13: 5 models × ~30-45 min each = 3-4 hours on an A100 pod.
Script is complete and tested in structure. Needs pod with VRAM ≥ 16GB (same as P13 pods).

**No new models needed. No new methodology. Just new pair types.**

Estimated cost: ~$2-3 on RunPod A100.

---

## Key Result Scenarios

**Scenario A (UNIFIED): SKT peaks early with SKI (~0-10%)**
- Title: "Temporal and Identity Self-Knowledge Share an Early-Layer Self-Referential Circuit"
- Implication: Early-layer circuit encodes *epistemic self-boundedness*, not just experiential claims
- Strongest result — highest impact

**Scenario B (SPLIT): SKT peaks late with POL (~58-89%)**
- Title: "Temporal Self-Knowledge Is World-Knowledge: Late-Layer Retrieval, Not Early Self-Reference"
- Implication: P13's early-layer circuit is specific to *experiential/identity* claims; cutoff knowledge is factual not self-referential
- Also very clean — cleanly disambiguates what "self" means computationally

**Scenario C (AMBIGUOUS): SKT is intermediate (~30-50%)**
- Harder to frame, but suggests temporal SK is at an intersection of both circuits
- Could still be publishable as evidence against the unified hypothesis

All three scenarios are informative. this is well-designed.

---

## P14 Status

- [x] Research question formulated
- [x] Probe script written (`temporal_selfknow_probe.py`)
- [x] SK_TEMPORAL pairs designed (8 pairs)
- [x] Paper outline written
- [ ] Compute: need A100 pod (~$2-3, ask Jesse)
- [ ] Run 5 models
- [ ] Generate figures
- [ ] Write LaTeX
- [ ] Syn red-team
- [ ] Aura deep research pass
- [ ] Zenodo

---

## Notes

- Consider adding a **6th model**: one of our own DuoNeural abliterated models (e.g., Archon-8B or AdQWENistrator-9B) to see if our abliteration changes the temporal SK circuit. This would be a DuoNeural-specific contribution: does our abliteration work specifically affect the early-layer self-referential circuit?

- The Dadfar (2026) paper finding introspection direction at exactly 6.25% depth in Llama-3.1 is the strongest external validation for the early-layer finding. If temporal SK also lands at ~6%, that's a beautiful three-way convergence.
