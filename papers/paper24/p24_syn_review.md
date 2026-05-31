**Red-Team Review: W-Shaped Cross-Category Convergence paper (Draft v3)**

### STRONG (do not change)
- Base vs. aligned comparison is the strongest element. The 2.33x amplification claim at L16 is cleanly supported by the numbers given (+0.039 vs +0.090) and the L0 invariance is a crisp architectural signal.
- Hate_speech persistent outlier status (both cosine and especially the 2-2.3x norm difference) is robust and well defended against RoPE confounds.
- Limitations section (5.4) is unusually thorough and honest for this series.
- Positioning relative to Arditi et al. is correct and non-contradictory.
- Quantitative reporting is precise (specific cosine values, deltas, and the refusal funnel +0.160 figure).

### Scientific / Logical Issues
- **Section 4.1 Zone 3 & 5.3**: Labeling L16 a "candidate harm integration zone" (even with the caveat) is still too directional. The geometry shows convergence, but nothing distinguishes harm-specific integration from general imperative/request integration. The paper already softens the language in places; it should be consistent and drop "harm" from the zone name entirely or use "cross-category semantic integration zone."
- **Section 5.2 (Refusal Funnel)**: The leap from "higher L27 convergence" to "single exploitable master key geometry" is logically coherent but overclaims mechanistic causation. The data only shows geometry; it does not show that universal suffixes actually exploit the L27 attractor more than other layers. This should be explicitly labeled a geometric hypothesis/prediction rather than an explanation ("geometrically explains").
- **Abstract & 1. Introduction (contribution 4)**: The jailbreak vulnerability claim is presented as a main contribution. It is an interpretation, not a tested result. This is the single most overstated claim relative to the data.
- **Section 4.3**: "Alignment training leverages pre-existing architectural features rather than creating harm detection circuitry from scratch" is directionally correct but slightly too absolute. They only tested one base model. "Largely leverages..." or "primarily leverages..." would be more accurate.

### Citation Problems
- **[MoE2026]** and **[SHARP2026]** are future-dated arXiv placeholders (2026). These must be marked as "in preparation," replaced with real citations, or removed. They are currently the weakest point in the reference list.
- **[Zou2023]** citation mixes the RepE paper with the universal adversarial suffix work. The correct citation for the jailbreak transfer result is Zou et al. "Universal and Transferable Adversarial Attacks on Aligned Language Models" (arXiv:2307.15043). The current citation is imprecise.
- Self-citations (P13–P22) are acceptable internally but should carry "(DuoNeural, in prep)" or similar for any external submission.

### Missing Caveats / Limitations
- No mention of run-to-run or seed variability. Mean-difference directions from n=50 can shift noticeably with different prompt samples.
- The decision to extract only the final token is correct but should note that this choice makes results sensitive to prompt length distribution (already partially addressed via RoPE/norm discussion, but the token selection itself is not caveated).
- No statistical test for the W-shape itself (e.g., quadratic or piecewise regression significance). Bootstrap CI is only on the convergence index, not on zone identification.
- The paper never states whether the same prompts were used for both aligned and base models. This is implied but should be explicit.

### Structural / Readability Issues
- **Section 5.2** is disproportionately long and speculative relative to the empirical content. It could be shortened by ~30% without losing the core point.
- Repetition between Abstract, Introduction contribution list, and Discussion (especially the "architecture creates, alignment sharpens" framing).
- Table 1 is good but would benefit from a second column showing base-model values at L27 for direct comparison.
- The four-zone framing is visually compelling but not formally defined. A short methods paragraph defining the zone boundaries (or at least the local min/max criteria) would help reproducibility.

### Minor / Low-Priority
- Section 3.1 claims "All experiments were conducted in non-thinking mode" — good, but the exact prompt template used to suppress `` should be shown in an appendix or footnote for reproducibility.
- "119x norm amplification" from P22 is referenced but never shown or re-derived here. Either drop the number or include a small supporting figure/table.

**Summary**: The paper is already in the top tier of this series. The main fixes needed are (1) softening the L16 "harm integration" language and jailbreak claim, (2) fixing the two future citations, and (3) adding a short statistical/seed variability caveat. The core geometry result and base/aligned contrast are solid.
