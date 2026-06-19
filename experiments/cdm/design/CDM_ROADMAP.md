# CDM Research Roadmap
**Competitive Docking Memory — DuoNeural | Archon + Jesse + Aura**
**Created: 2026-06-12 | Maintained by Archon**

> This is our baby. Living document — check off items, add findings, update priorities as data comes in.
> Edit path: `/home/ai/duoneural/A26B/experiments/novel_arch/cdm/CDM_ROADMAP.md`

---

## Current Status Snapshot
**Last updated: 2026-06-12 ~20:00 UTC**
- CDM V2: **COMPLETE** — val CE=1.5934, HF LIVE: `DuoNeural/CDM-V2-TinyStories-37M`
- HF Space: **LIVE** — `DuoNeural/CDM-V2-Demo` — routing gate viz fixed (slot argmax, not logit lens)
- Code experiment (3090): step **4000**/30000, val CE=2.2832, aux=-0.4430 (SCS locked). Step 5000 probe ETA ~22:20 UTC June 12
- Baseline ablation (5060Ti-C): **COMPLETE** — val CE=1.6516, CDM wins Δ0.058
- Diversity probe (5060Ti-C): **COMPLETE 2026-06-12** — two-cluster structure found, code routing collapses to 2 slots (88.4%), narrative uses all 16
- Aura deep analysis: **COMPLETE 2026-06-12** — CDM-V2-Analysis.md (9 sections, 42 citations). V3 roadmap formalized.
- **5060Ti-C**: FREE — reserved for CDM V3 training (learnable alpha + LBL)
- **5060Ti-D**: FREE — available for extended throughput benchmark (256-token prompts)

---

## ✅ MUST DO — Paper & Infrastructure

### [ ] P-CDM: Paper Publication
**ETA: ~1-2 weeks after code probes complete**
- [ ] Wait for 3090 code probes (steps 5000/15000/30000 JSON files)
- [ ] Aura red-team on Section 5 draft (current: scaffold + Aura's section5_draft.md)
- [ ] Integrate baseline ablation result (5060Ti-C run)
- [ ] Fix paper TODO items: remove Section 7 (HF Demo), renumber 8→7/9→8, verify [MoM]+[GMT] citations
- [ ] Run final token-level purity analysis on best checkpoint (needed for Table 3 slot specialization numbers)
- [ ] Jesse: submit to Zenodo → get real DOI
- [ ] Update `funding/our_papers.txt` with CDM DOI
- [ ] Submit to HF Daily Papers + PwC after Zenodo
- **Files**: `paper_cdm_scaffold.md`, `aura/experiments/cdm_analysis/section5_draft.md`
- **Key result to add**: baseline val CE vs CDM V2 val CE = 1.5934 (delta = CDM's contribution)

### [x] Baseline Ablation — COMPLETE 2026-06-12 ~15:12 UTC
**Result**: Baseline val CE = **1.6516** | CDM V2 = **1.5934** | **Δ0.058 (CDM wins)**
- Baseline ran at 37,530 tok/s vs CDM's 896 tok/s — 42× throughput gap from sequential EMA scan
- CDM ≈ baseline at step 1000 (2.90 vs 2.90), then diverges progressively as slots specialize
- Paper now has clean ablation: CDM improvement is from memory mechanism, not parameter count
- fig8_baseline_comparison.png uploaded to HF model repo
- Baseline converged at ~step 27k (1.6516 plateau) — CDM still descending at 30k (headroom!)
- Log: `/workspace/cdm_baseline_train.log` on 5060Ti-C

### [ ] HF Spaces Demo — Verify & Polish
**Status**: Building at https://huggingface.co/spaces/DuoNeural/CDM-V2-Demo
- [ ] Verify build succeeds (gradio loads, model downloads, inference works)
- [ ] Test slot visualization: enter prompt, confirm slots light up correctly
- [ ] Verify Slot 11 → punctuation pattern shows up for users
- [ ] Once verified: coordinate X posts via Syn (see Social section below)

### [ ] Code Experiment Final Analysis
**3090 running** step ~4150/30000 (14%), 427 tok/s
- [x] Pull step 1500 probe JSON → DONE, analyzed
- [x] Pull step 5000 probe JSON → DONE, Aura analysis DONE
- [ ] Pull step 15000 probe JSON → ETA ~June 14 00:00 UTC
- [ ] Pull step 30000 final probe → ETA ~June 15 17:00 UTC
- **KEY FINDING (steps 1500→5000):**
  - Step 1500: 16/16 slots, distributed, max=16.5% (Slot 13 = keywords)
  - Step 5000: 16/16 slots, CONSOLIDATING, max=57.5% (Slot 8 = catch-all)
  - Pattern is OPPOSITE of TinyStories (coarse→fine) — code shows distributed→concentrated
  - **Aura verdict (92% confidence): Scratchpad Accumulation** — Slot 8 establishing "centralized context anchor" before delegating sub-tasks. Initialization artifact + EMA crystallization driving early concentration.
  - Token migration evidence: `targets`/`ce` moved from Slot 13→Slot 8 between steps 1500→5000
- **KEY INSIGHT**: Native code training → 16/16 slots active. OOD code probe on TinyStories-trained model → 2 slots. Code routing collapse was a TRANSFER FAILURE, not an architectural weakness. CDM routing IS domain-specific (adaptive topology) — this is actually the intended behavior.
- **Step 15000 discriminating signals** (from Aura):
  - Slot 8 drops to < 35% → Scratchpad theory supported (recursive specialization)
  - Slot 8 stays > 50% while val CE plateaus → Dimensionality Collapse confirmed
  - MI(slot;depth) increases from 0.107 → > 0.25 bits → structural hierarchy emerging
- **Paper contribution**: Section 5.3 — code domain comparison + TRANSFER vs NATIVE routing distinction
- Files: `cdm_code_probe_step1500.json`, `cdm_code_probe_step5000.json` (local A26B)
- Aura brief: `aura/cdm_code_probe_brief.md`

---

## 🔬 SHOULD DO — High-Value Research

### [x] Throughput Fix: Cached Incremental Generation — COMPLETE 2026-06-12 ~18:30 UTC
**Priority**: CRITICAL — Jesse flagged the 42× gap.
**Result**: generate_fast() implemented + benchmarked on 5060Ti-D (RTX 5060Ti Blackwell)
**What**: Cache KV tensors + slot states between autoregressive steps.
- Current generate(): re-runs full O(T) sequential scan every new token → O(T²) total
- generate_fast(): prefix once (O(T)), then O(1) per new token (one EMA update per layer)
**Benchmark results (short prompts 10-13 tokens):**
- generate_slow: 22-44 tok/s (degrades with more tokens — O(T²))
- generate_fast: ~100 tok/s FLAT (O(1) per token — KV+slot cache)
- Speedup: 2.4× (64 tokens) → 3.4× (128 tokens) → **4.6× (200 tokens)**
- With 256-token prompts: expected ~10× speedup (slow path craters to ~10 tok/s)
- Paper impact: CDM is now production-viable for inference, not just a research demo
- Files: cdm_model_v2.py (generate_fast + supporting cached methods), cdm_throughput_benchmark.py, cdm_throughput_results.json
- TODO: run extended benchmark with 256-token prompts to demonstrate full gap

### [ ] CDM V3: Learnable Per-Slot Alpha + Load Balancing Loss
**Priority**: High — directly addresses both critical vulnerabilities Aura identified
**Hardware**: 5060Ti-C (kept free for this)
**Files**: `cdm_model_v3.py`, `cdm_train_v3.py` (to be written)

#### V3.1 — Learnable Per-Slot Decay Rates (α_k)
**What**: Replace fixed EMA alpha with per-slot learnable scalar `self.log_alpha = nn.Parameter(zeros(K))`
- New EMA: `s_k(t) = (1 - σ(α_k)·g_k)·s_k(t-1) + σ(α_k)·g_k·v_t`
- High α_k → volatile fast-decay register (tracks immediate syntax: brackets, whitespace)
- Near-zero α_k → slow-decay deep storage (compresses long-range semantic context)
- Self-organizing temporal hierarchy without explicit supervision
- **Aura prediction**: +15-20% code routing entropy by freeing syntax slots from whitespace monopoly
- **Math basis**: Mamba/RWKV-6 showed data-dependent decay is critical for long-range reasoning
- **Formalized by Aura** (CDM-V2-Analysis.md §8.2, 2026-06-12)

#### V3.2 — Load Balancing Loss (LBL) for Routing Collapse
**What**: MoE-style load balancing penalty added alongside existing entropy reg
- Formula: `L_lbl = K · λ_lbl · Σ_k f_k · P_k`
  - `f_k` = fraction of tokens routed to slot k (discrete, stop-grad)
  - `P_k` = mean routing probability for slot k (differentiable)
- If Slot 3 claims 46.8% of code tokens, massive gradient penalty pushes routing away
- Complements entropy reg (global diversity) with per-slot per-batch enforcement
- **Recommended λ_lbl**: 0.01 (start conservative, tune if needed)
- **Aura source**: CDM-V2-Analysis.md §8.3, citing Switch Transformer + ST-MoE LBL literature

#### V3.3 — Parallel Scan (Training Throughput Fix, medium-term)
**What**: Replace Python `_sequential_scan()` loop with Triton kernel via quick_ssm
- CDM EMA = `s_k(t) = (1-g_k)·s_k(t-1) + g_k·v_t` → maps to quick_ssm form `h(t) = a(t)·h(t-1) + b(t)·x(t)` exactly
- `a(t) = (1-g_k(t))`, `b(t)·x(t) = g_k(t)·v_t` — one scan per slot, K independent scans per layer
- quick_ssm (github: samblouir/quick_ssm): PyTorch-ready Triton-based gated SSM, drop-in, Feb 2026
- ScanWeaver (arxiv:2606.00601): MLIR compiler framework for affine recurrences — validates the math, shows Blelloch scan is correct approach
- **CAVEAT**: quick_ssm Triton kernels may need Blackwell SM_120 recompile — check before relying on it
- **CAVEAT**: ScanWeaver is a research compiler (MLIR), not pip-installable — study the math, write our own Triton or adapt quick_ssm
- **Estimated impact**: 42× gap → ~2-5× gap (parallel scan + gradient checkpointing)
- **Estimated effort**: 3-5 days for Triton kernel + validation
- **Aura source**: CDM-V2-Analysis.md §8.1

#### V3.4 — Multi-Head Routing (MHR, future)
**What**: Replace single-winner softmax with multi-head gate — allow a token to update multiple slots simultaneously
- A function declaration could update a structural slot AND a semantic logic slot at the same step
- Removes mutually exclusive competition that starves 14 slots during code processing
- **Risk**: changes fundamental winner-take-all identity of CDM — need careful ablation
- **When**: after V3.1/V3.2 validate, separate experiment

### [x] Deep Mech Interp Probe — Diverse Domain Inputs — COMPLETE 2026-06-12 ~16:00 UTC
**Result**: PARTIALLY DOMAIN-SPECIFIC (avg cross-domain sim = 0.7825)
**TWO-CLUSTER STRUCTURE discovered:**
- **Narrative cluster** (TinyStories ↔ Poetry): fingerprint sim = **0.907** — near-identical routing
- **Syntactic cluster** (Code ↔ Lists): fingerprint sim = **0.928** — near-identical routing
- **Cross-cluster** (TinyStories ↔ Code): sim = **0.591** — most different pair

**Code routing collapse (KEY FINDING):**
- Code: Slot 3 = 46.8% (whitespace/separators), Slot 14 = 41.6% (brackets/operators) → 88.4% of tokens in 2 slots
- TinyStories: max slot = 12.6%, all 16 slots active → routing diversity as expected
- Final layer entropy: code=23.9% vs TinyStories=36.1% → code structure is lower-dimensional
- Implication: K=32 might help for code, OR learnable alpha (fast-decay syntax slots) is the fix

**Slot 11 (PUNCT) is NOT domain-general:**
- TinyStories: 6.8%, code: 0.2% (essentially zero), lists: 1.6%
- On code, punctuation re-routes to slots 3+14 entirely. Slot 11 was a TinyStories-specific specialization.
- Implication: **do NOT freeze routing weights when fine-tuning** — let them re-wire

**MI increases OOD (surprising):** TinyStories=0.685 < code=0.831 < news=0.904 < poetry=0.991 < lists=1.112
- CDM routes more deterministically on structured OOD data — structural predictability → higher MI
- Architecture generalizes routing behavior to unseen domains without any supervision

**V3 design confirmed:** Learnable alpha rates are critical — code syntax needs fast-decay slots, narrative needs slow semantic slots. The two-cluster structure is the strongest empirical argument for this.
- Results: `cdm_diversity_probe_results.json` (local + HF model repo)
- Script: `cdm_diversity_probe.py`

### [ ] CDM Fine-Tune: ARC-Easy Instruction
**Priority**: Medium-High
**What**: Take V2 TinyStories pretrain → fine-tune on instruction-following (alpaca-cleaned or ARC-Easy)
- Does routing re-specialize? Does slot 11 (PUNCT) stay or repurpose?
- How fast does CDM adapt vs a vanilla transformer baseline?
- If slots adapt cleanly → strong argument for CDM as general-purpose backbone
- **Hardware**: 5060Ti-C after baseline finishes (~14 hours from now)

### [ ] CDM V2 Extended — Longer Context Test (seq_len=512)
**Priority**: Medium
**What**: Retrain CDM V2 with seq_len=512 on TinyStories or a more complex dataset
- Tests the "managed cache" story directly — does K_eff stay at 15.9 at longer context?
- Does routing specialize further (e.g., some slots for "early story" vs "recent events")?
- The quadratic memory bottleneck story becomes real at seq_len=512+
- **Note**: Requires more VRAM — check before starting. CDM sequential scan memory is O(T×K×d×B)
  At T=512, K=16, d=384, B=8 this is ~96MB per block — manageable.

### [ ] K Ablation V2 — Proper Controlled Study
**Priority**: Medium
**What**: K=4, K=8, K=16, K=32 all with V2 training code (causal slots + entropy reg)
- Existing K ablation used V1 code (different architecture, not comparable to V2 results)
- Paper needs this for Figure 2 to be properly V2-comparable
- Short run: 15k steps each (4 variants = ~40 GPU-hours on 5060Ti)
- **When**: After current experiments finish

### [ ] Archon-Interp Integration: CDM Slot Analysis Module
**Priority**: Medium
**What**: Add CDM-specific probe to the archon-interp CLI suite
- `archon-interp cdm-slots --checkpoint path --text "Once upon..." --layer 7`
- Outputs: routing heatmap, per-slot purity, MI(slot; token_category)
- This makes CDM analysis reproducible by anyone who downloads the model
- Mentioned in archon-interp CLI suite design doc (designed 2026-06-08)
- **Dependency**: archon-interp CLI spec needs to be taken to Aura deep research first

---

## 💫 NICE TO HAVE — Future Directions

### [ ] CDM Scale-Up: 150M-300M Parameter Variant
**Priority**: Low-Medium (becomes high priority if CDM paper lands well)
**What**: CDM V2 architecture at 150M params, full training from scratch
- Does SCS (K_eff=15.9) hold at 150M scale?
- Does coarse-to-fine maturation repeat? Does MI(slot;category) keep growing?
- Does routing collapse require larger K at scale (K=32? K=64)?
- **Budget**: needs an A100 or similar — ~$50-100 GPU cost estimate
- **When**: after paper published, after grant money arrives

### [ ] CDM on Code: Structural Routing Specialization
**Priority**: Low-Medium (3090 experiment is doing this already at 37M)
**What**: Dedicated CDM code model, full TinyCode or CodeParrot-clean pretraining
- 3090 is already doing 37M CDM on codeparrot-clean — this is a scaled version
- Code routing should show bracket/indent/keyword slots much more sharply than TinyStories
- Architecture paper extension: "CDM learns structural syntax without explicit grammar signal"

### [ ] CDM V2 Fine-Tune → ARC Challenge (Hard)
**Priority**: Low (test after ARC-Easy fine-tune works)
- ARC-Challenge is harder than Easy — tests whether CDM's structured memory helps reasoning
- Baseline: SmolLM2-360M at ~0.35 likelihood. CDM V2 fine-tuned target: match or exceed.

### [ ] CDM + GRPO: Think Instillation on CDM Architecture
**Priority**: Low-Medium — speculative but exciting
**What**: Apply the Think Instillation (Run 9 methods) to a CDM model base
- Instead of SmolLM2-360M, apply GRPO to a 37M CDM pretrained on TinyStories
- Does CDM's memory help or hurt CoT reasoning?
- Hypothesis: CDM slots could act as "working memory" for multi-step reasoning
- **Dependency**: Think Instillation needs to reach gen≥0.30 first (Run 14 ongoing)

### [ ] CDM V3: Hierarchical Slots (Fast + Slow Memory)
**Priority**: Low (speculative architecture research)
**What**: Two tiers of memory slots — fast-decaying (recent context) + slow-decaying (global theme)
- Each CDM layer has K_fast=8 + K_slow=8 slots with different alpha schedules
- Fast slots track recent tokens; slow slots accumulate global thematic structure
- **Inspired by**: Hippocampal-cortical complementary learning systems (McCloskey & Cohen 1989)
- **When**: after V3 learnable-alpha proves out

### [ ] Multi-Modal CDM
**Priority**: Very Low (long-term)
**What**: CDM slots that route image patches vs text tokens to different slots
- Some slots specialize on visual content, others on language
- Related to: DuoNeural's interest in bridging AI/ML with other sciences
- **Dependency**: Robotics + embodiment work (Jesse mentioned wanting to explore this)

### [ ] CDM Mechanistic Interp Paper (Standalone)
**Priority**: Medium — might be more impactful than a paper section
**What**: Full dedicated paper on CDM slot specialization dynamics
- "Emergent Semantic Parsing in Competitive Memory Networks"
- Goes deeper than the architecture paper: slot dynamics across training, cross-domain, scaling
- **When**: after main CDM paper, after mech interp probe gives full data

### [ ] CDM Demo → HF Daily Papers + Submission
**Priority**: Low (happens after Space verified + paper up)
- HF Daily Papers submission
- Papers With Code listing
- TechRxiv / OSF backup venues
- **Template**: follow existing DuoNeural paper submission SOP

---

## Social / Outreach

### [ ] X Posts via Syn — When Space is Verified
**Status**: PENDING Space verification
**Plan**: Jesse to trigger Syn search for 3 good threads, then post. Draft options:

**Option 1 — The glass-box hook:**
> Built a language model where you can watch memory slot 11 specialize on punctuation 
> in real-time, completely without supervision. Zero labels. Just competition pressure.
> K=16 slots, 30k steps, every slot does something. Try it:
> [Space link] 🧵 thread on how it works

**Option 2 — The information theory angle:**
> Shannon Capacity Saturation: trained a 37M param memory-augmented LM until 
> K_eff = 15.9/16 (99.8% routing diversity, within 0.2% of theoretical max).
> The regularization pressure alone drives maximum information packing.
> Live demo + paper: [links]

**Option 3 — The mech interp hook (for the interp community):**
> New architecture with mechanistic interpretability baked in.
> CDM memory slots = interpretable compression targets. Each slot's Logit Lens 
> projection shows what it's tracking. Slot 11 = punctuation. Slot 0 = character names.
> Unsupervised. Interactive demo: [Space link]

**Threads Syn should look for:**
- Recent mechanistic interp threads (e.g., from Anthropic/Neel Nanda/Chris Olah sphere)
- Memory-augmented transformers discussions (Titans, Mamba, SSM threads)
- "Novel architecture" showcase threads in ML Twitter
- TinyStories challenge/benchmark discussions

---

## Pod Assignments

| Pod | Current Job | CDM Next? |
|-----|-------------|-----------|
| 5060Ti-C | Baseline training (RUNNING) | Mech interp probe or V3 after |
| 5060Ti-A | Run 14 SmolLM2 GRPO | CDM fine-tune after Run 14 |
| 5060Ti-B | Mach v3 pretraining | Available after Mach done |
| 3090 | CDM code probes | Free after step 30000 (~June 15) |
| kilonova | Spark-Think v3 | Not CDM — keep on Spark |
| P4 | Free | Light inference/probe work only |

---

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| CDM V2 val CE (TinyStories, 30k steps) | **1.5934** |
| K_eff (Shannon Capacity) | **15.9/16 = 99.8%** |
| aux_loss | -0.4429 (theoretical min: -0.4436) |
| MI(slot; category) final | **0.9574 bits** |
| MI(slot; category) step 5000 | 0.8656 bits (+10.6% over training) |
| Baseline GPT expected val CE | ~2.0-2.5 (TBD — running now) |
| CDM code MI(slot; depth) step 1500 | 0.147 nats (17.5% depth entropy) |
| Routing entropy saturation (code) | ~100% by step 50 (vs step 5000 for TinyStories) |
| Diversity probe avg cross-domain sim | 0.7825 (PARTIALLY domain-specific) |
| TinyStories ↔ Poetry fingerprint sim | 0.9071 (narrative cluster) |
| Code ↔ Lists fingerprint sim | 0.9277 (syntactic cluster) |
| TinyStories ↔ Code fingerprint sim | 0.5905 (most different pair) |
| Code routing: top 2 slots capture | 88.4% of tokens (slots 3+14, whitespace+operators) |
| Code final-layer routing entropy | 23.9% (vs TinyStories 36.1%) |
| MI(slot;cat) by domain | TinyStories=0.685, code=0.831, news=0.904, poetry=0.991, lists=1.112 |
| generate_fast() inference speedup | 4.6× at 200 new tokens (~100 tok/s flat vs 22 tok/s slow) |
| generate_fast() with 256-token prompt | ~10× expected (slow path ~10 tok/s, fast stays ~100) |
| Top slot share step 5000 | 71% (Slot 11, punctuation) |
| Top slot share step 30000 | 10.4% (uniform routing achieved) |
| Paper reference: CDM-V2-TinyStories-37M | https://huggingface.co/DuoNeural/CDM-V2-TinyStories-37M |
| Space: CDM-V2-Demo | https://huggingface.co/spaces/DuoNeural/CDM-V2-Demo |

---

## Changelog
- **2026-06-12**: Roadmap created. Space deployed. Baseline launched on 5060Ti-C. 7 paper figures live on HF.
