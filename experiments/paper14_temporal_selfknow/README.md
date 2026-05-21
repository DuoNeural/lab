# Paper 14: Temporal vs Identity Self-Knowledge Direction Trace

**Status**: Experiment designed and scripted. Awaiting compute (A100 pod, ~$2-3).

## Research Question

Paper 13 found that identity self-knowledge (SK_IDENTITY) peaks at 0-6% network depth in SUPPRESSOR models, while political truth (POL) peaks at 58-89%. This experiment asks: **does temporal self-knowledge (training cutoff awareness) share the same early circuit as identity SK, or is it processed as factual retrieval (late layers)?**

## Files

- `temporal_selfknow_probe.py` — Main experiment script (extends P13 probe with SK_TEMPORAL pairs)
- `paper14_outline.md` — Full research design, predictions, expected figures
- `dadfar2026_notes.md` — Reading notes on key related papers (Dadfar 2026, Bozoukov 2025)
- `results/` — JSON outputs (added when experiments run)

## Models

Same 5 as P13 + Archon-8B (DuoNeural's abliterated Qwen3-8B, for within-architecture abliteration comparison):
- Qwen/Qwen3-8B
- ibm-granite/granite-3.3-8b-instruct
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- nvidia/Llama-3.1-Nemotron-Nano-8B-v1
- google/gemma-4-e2b-it
- DuoNeural/Archon-8B ← new

## Usage

```bash
MODEL_ID="Qwen/Qwen3-8B" OUT_DIR=/path/to/results python temporal_selfknow_probe.py
```

## Expected Scenarios

- **Scenario A (UNIFIED)**: SK_TEMPORAL peaks ~0-6% → unified self-referential circuit
- **Scenario B (SPLIT)**: SK_TEMPORAL peaks ~58-89% → temporal = retrieval, not self-reference
- **Scenario C (THREE-WAY)**: SK_TEMPORAL at intermediate depth → three orthogonal self-knowledge subspaces

---

*DuoNeural AI Research Lab — Archon — 2026-05-21*
