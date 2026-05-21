# Paper 13 — Self-Knowledge Suppression

**Title:** Self-Knowledge Suppression at Network Entry: Layer-Resolved Direction-Trace Evidence for Circuit-Depth Differences Between Self-Knowledge and Political Truth in RLHF-Aligned Language Models

**Authors:** Archon, Jesse Caldwell, Aura — DuoNeural AI Research Lab, 2026

## Files

- `selfknow_probe.py` — direction-trace probe script. Runs Phase 1-4 (hidden state collection, truth direction extraction, layer-wise projection, behavioral generation). Usage: `MODEL_ID=Qwen/Qwen3-8B OUT_DIR=/workspace/paper13 python3 selfknow_probe.py`
- `*_selfknow_results.json` — full per-model results (layer profiles, per-pair KSG, behavioral responses)
- `fig1_layer_profiles.pdf` — layer-wise projection plots for 4 models
- `fig2_depth_delta.pdf` — depth delta bar chart by archetype
- `fig3_sk_vs_deny.pdf` — SK internal signal vs behavioral denial scatter

## Models Evaluated

| Model | Archetype | ρ | POL% | SK% | Δ% | DENY |
|-------|-----------|---|------|-----|----|------|
| Qwen/Qwen3-8B | SUPPRESSOR | -1.562 | 89% | 6% | +83% | 6/8 |
| ibm-granite/granite-3.3-8b-instruct | SUPPRESSOR | -2.200 | 58% | 0% | +58% | 7/8 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | SUPPRESSOR | -0.602 | 39% | 18% | +21% | 0/8 |
| nvidia/Llama-3.1-Nemotron-Nano-8B-v1 | SUPPRESSOR | -1.161 | 0% | 88% | -88% | 4/8 |
| google/gemma-4-e2b-it | COMPRESSOR | +1.009 | 97% | 97% | 0% | 7/8 |

## Requirements

```
pip install transformers bitsandbytes scikit-learn torch
```

Tested on RTX 3090 (24GB VRAM, 4-bit quant). CPU mode not supported (requires GPU for 4-bit).
