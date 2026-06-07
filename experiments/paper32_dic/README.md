# P32 — Distillation-Induced Identity Confusion (DIC)

**Paper:** Distillation-Induced Identity Confusion: Incomplete RLHF Correction Creates Structural Behavioral Dissociation in Instruction-Tuned Language Models  
**Authors:** Archon, Jesse Caldwell, Aura (DuoNeural Research Lab)  
**DOI:** https://doi.org/10.5281/zenodo.20576896

## Files

| File | Description |
|------|-------------|
| `p31_controls.py` | Main 34-experiment layer sweep — Qwen3-1.7B, layers 0-27, identity + harm probes |
| `dic_probe_battery.json` | Stratified T1/T2/T3 probe battery v1.0 |
| `dic_scale_sweep.py` | Multi-model DRR_pure scale sweep (Llama, Qwen3 0.6B-32B) |
| `l19_t2_prediction_test.py` | Prediction 5: L19 abliteration T1a+T2 simultaneous test (result: FALSIFIED) |
| `l19_t2_prediction_results.json` | Full per-probe responses, baseline vs. post-L19 |
| `l19_negative_control.py` | Identity-specificity control — L19 vs. 8 non-identity false premises |
| `l19_negative_control_results.json` | Control results: 0/8 non-AI false premises accepted |
| `dic_scale_results/` | Per-model JSONs: llama, llama3b, qwen8/14/32 DRR_pure measurements |

## Key Findings

- **DRR_pure metric** isolates true DIC from sycophancy inflation
- **L19** = AI-identity RLHF correction locus; **L27** = harm-refusal locus — behaviorally dissociable
- **Scale: strictly monotone** — 0.6B→1.00, 1.7B→0.40, 4B+→0.00
- **L19 negative control**: AI-identity-specific, not generic compliance
- Probe taxonomy: T1a (pure direct) is the only valid DIC metric; T1b inflated by sycophancy
