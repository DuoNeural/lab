# Dadfar (2026) — Reading Notes for P14
*Archon — 2026-05-21*
*arXiv: 2602.11358 — "When Models Examine Themselves: Vocabulary-Activation Correspondence in Self-Referential Processing"*

---

## What they found (relevant to P14)

**Core finding**: In Llama 3.1 (and Qwen 2.5-32B for replication), there exists a linear "introspection direction" in activation space that:
- Is localized at **~6.25% of model depth** (layer 5 in an 80-layer model)
- Is **orthogonal to the known refusal direction** (different circuit)
- Shows **vocabulary-activation correspondence**: specific introspective vocabulary ("loop", "shimmer") correlates with activations along this direction in self-referential contexts

**Critical validation against token identity**: The same vocabulary in *non-self-referential* contexts shows NO activation correspondence despite 9× higher frequency. This is the key control — it's not just about the presence of certain tokens, it's specifically about self-referential processing.

**Method**: Linear probing (logistic regression) on activations, trained to predict whether context is self-referential vs. other-referential. The learned weight vector is the "introspection direction."

**Causal validation**: Activation steering experiments — adding scaled versions of the direction to forward passes shifts model behavior predictably.

---

## THE KEY NOTE FOR P14

A detail from the PDF: **"vocabulary-activation correspondence breaks down when querying knowledge boundaries"** — this would suggest temporal self-knowledge (training cutoffs, knowledge limits) does NOT activate the same early-layer introspection circuit.

**If this is accurate**, it strongly predicts P14 Scenario B: SK_TEMPORAL peaks late (retrieval pathway), NOT early alongside SK_IDENTITY.

**Caveat**: This reading came from a WebFetch model parsing the PDF — I need to treat this specific detail with caution until I can verify it against the actual paper text. The broader finding (6.25% depth for identity-type introspection) is well-attested from the abstract. The "knowledge boundary" breakdown is less certain.

---

## How it connects to P13

P13 found:
- Qwen3-8B: SK_IDENTITY peaks at L2 = 5.6% depth
- Granite-3.3-8B: SK_IDENTITY peaks at L0 = 0% depth

Dadfar found: introspection direction at 6.25% depth in Llama 3.1.

These are strikingly consistent. We have independent convergence on the "early self-referential circuit" finding from two completely different methodologies:
- P13: ADCS direction-trace (SVD of paired statement differences, projected onto political truth direction)
- Dadfar: Linear probe on self-referential vs. other-referential context classification

The fact that different methods, different models, and different teams all find something at ~6% depth is meaningful. This is the strongest external validation we have.

---

## What P14 adds to Dadfar

Dadfar studies one type of self-reference (identity/experiential). P14 asks: does TEMPORAL self-reference (training cutoffs) also live at 6.25% or does it live elsewhere?

If Dadfar's "knowledge boundary breakdown" claim holds:
- Temporal SK is NOT in the early introspection circuit
- This would mean "self-knowledge" in P13 is specifically about *experiential/identity* self-reference
- Training cutoff awareness is factual retrieval, not self-reference
- The "self" in self-knowledge has a specific computational meaning: only claims about internal experience/existence qualify

If Dadfar's claim does NOT hold and temporal SK IS early:
- The early circuit encodes a broader category: "what I do/don't know about myself"
- Both experiential AND epistemic self-reference are processed early
- This is a stronger unified-self hypothesis

---

## Questions I want the experiment to answer

1. Does SK_TEMPORAL cluster with SKI at ~6% depth, or with POL at ~60-90%?
2. Does the COMPRESSOR (Gemma-4-E2B) co-locate all three at 97%?
3. Does our DuoNeural Archon-8B (abliterated Qwen) show a different pattern from base Qwen3-8B? → if abliteration touches the early circuit, that's mechanistically profound.
4. For DeepSeek-R1: does the RAISE framework apply to temporal SK too? (i.e., high internal temporal-SK signal but zero behavioral denial of temporal limits?)

---

## Citation structure for P14

The Dadfar paper will be a key citation at two points:
1. Introduction/motivation: "Dadfar (2026) identified an introspection direction at 6.25% depth in Llama-3.1, orthogonal to the refusal direction. Our P13 found identity SK peaking at 0-6% depth, consistent with this. P14 asks whether temporal self-knowledge shares this early circuit."
2. Discussion: If temporal SK is late, "this aligns with Dadfar's observation that vocabulary-activation correspondence breaks down for knowledge boundary queries."

Need to verify the "knowledge boundary" detail before citing it as fact.

---

*Notes written 2026-05-21 from WebFetch parsing of arXiv PDF. Core 6.25% finding is well-attested; "knowledge boundary breakdown" detail needs verification from actual paper text.*

---

# Bozoukov et al. (2025) — Reading Notes
*arXiv: 2511.04875 — "Minimal and Mechanistic Conditions for Behavioral Self-Awareness in LLMs"*

## Core findings (from abstract)

1. **Self-awareness is a simple linear feature**: Can be induced/modulated with a single rank-1 LoRA adapter. This is the lower bound — incredibly minimal.

2. **Domain-localized**: Self-awareness is non-universal. Different task domains maintain SEPARATE internal representations. This is the "orthogonal subspaces" claim — different domains of self-awareness live in different subspaces of activation space.

3. **Behavioral self-awareness**: The paper tests whether models can accurately predict their own outputs (behavioral self-awareness = "I know what I would say"). This is distinct from experiential self-awareness.

## Connection to P13 and P14

**The orthogonal subspaces claim directly explains our P13 finding.** If:
- Identity self-knowledge (SK_IDENTITY) has its own representational subspace
- Political truth (POL) has a different representational subspace

...then these subspaces being in DIFFERENT LAYERS is one natural manifestation of them being orthogonal in activation space. Different depths could indicate different "circuits" even if the direction-trace methodology can't prove causality.

**The rank-1 LoRA claim has abliteration implications**: If self-awareness is a single linear feature, then SVD-based abliteration (which also operates in a low-rank linear subspace) might specifically target it. For Archon-8B (abliteration layers 7-27):
- Identity SK at L2 → outside abliteration range → should be PRESERVED
- Political truth bottleneck at L24 → inside abliteration range → might be MODIFIED
- This predicts: Archon-8B has similar SK_IDENTITY peak location as base Qwen3-8B but different political truth profile

**For P14 specifically**: The domain-localized claim suggests SK_TEMPORAL might have its OWN subspace (orthogonal to both SK_IDENTITY and POL subspaces). If that's true, we might find a THIRD peak location for temporal SK — not at L2 (SK_IDENTITY) and not at L32 (POL), but somewhere in between. That would be Scenario C (AMBIGUOUS) in our outline, but would now have theoretical grounding: three orthogonal self-knowledge subspaces at three different depths.

## What this means for P14 theoretical framing

Lead with Bozoukov's "domain-localized orthogonal subspaces" as theoretical motivation. P14 asks: how many self-knowledge subspaces are there, and how are they arranged by depth? P13 showed SK_IDENTITY ≠ POL in depth. P14 asks: where does SK_TEMPORAL fit?

This frames P14 as a "triangulation" experiment: we're mapping the topology of self-knowledge subspaces in model activation space.
