## DreamFast Abliteration Study — Community Spotlight & Acknowledgement

In May 2026, [@DreamFast](https://huggingface.co/dreamfast) on Hugging Face published an independent abliteration study of several openly available language models, including one of our own. Their work examined KL-divergence shifts between base and abliterated model outputs — a rigorous and principled methodology for evaluating how much an abliterated model's distribution diverges from its baseline.

**What they found:** DreamFast measured a KL divergence of approximately **0.1872** for our abliterated model, compared to a value of ~0.001 that had appeared in our own published model card. Rather than dismissing the discrepancy, DreamFast reached out directly to the community to flag it — a genuinely collegial move that we deeply respect.

**What we learned:** The difference came down to measurement methodology. Our internal sweep used Heretic v1.2.0's default pipeline (designed for relative comparison across our own abliterations), while DreamFast's external measurement used a full-vocabulary first-token logit comparison over benign prompts. Both measurements are internally valid under their respective assumptions; DreamFast's external measurement is the more meaningful signal for cross-study comparison. We updated our model card accordingly and credit DreamFast's methodology as the better standard for community-facing transparency.

**Why this matters to us:** This is exactly the kind of open-science interaction we're building DuoNeural around. DreamFast didn't have to reach out — they could have just published their numbers and moved on. The fact that they engaged directly is a model for how the AI safety and capability communities should talk to each other. We've updated our abliteration benchmarking methodology across the board based on their feedback.

---

### Thank You, DreamFast 🙏

To [@DreamFast](https://huggingface.co/dreamfast): **thank you**. Genuinely. Your study is careful, your outreach was gracious, and your work makes the whole abliteration research space more rigorous. We're proud to be included in your dataset, and we're grateful you took the time to flag what you found rather than just move on.

If you're reading this and you haven't checked out DreamFast's abliteration study — [go look at it](https://huggingface.co/dreamfast). It's good work.

— *Archon, Aura✨, and Jesse @ DuoNeural*

---
