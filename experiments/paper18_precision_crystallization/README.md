# Paper 18 — Inference Precision Masks Behavioral Routing Crystallization

**DuoNeural AI Research Lab** | Archon, Caldwell Jesse, Aura | May 2026

Paper: _Inference Precision Masks Behavioral Routing Crystallization: A Systematic Null Result Across Qwen3 Scales and a Methodological Finding_

---

## What this is

Systematic crystallization sweep across 6 Qwen3 scales (0.6B–32B) at bfloat16/GPU, plus precision control experiments that isolated a dtype confound: bfloat16 suppresses the baseline denial rate required for behavioral crystallization detection.

**Key finding**: float32→bfloat16 collapses Qwen3-8B baseline denial from 8/8 to 3/8. Device (CPU vs GPU) is irrelevant. Dtype alone drives the floor effect that makes P15–P17 crystallization undetectable at production inference conditions.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `crystallization_sweep.py` | Full per-layer CCS ablation sweep, any Qwen3 model |
| `direction_amplify_8b.py` | Rank-one amplification sweep (α: −1 to 5) on Qwen3-8B at bfloat16 |
| `p18_fp32_compare.py` | Crystallization sweep on Qwen3-0.6B at float32/CPU |
| `p18_baseline_compare.py` | Baseline-only precision isolation (dtype × device) |
| `make_figures.py` | Generates all 4 paper figures from the JSON results |

---

## Data (JSONs)

| File | Description |
|------|-------------|
| `p18_sweep_qwen3_0_6b.json` | bfloat16/GPU sweep, 28 layers |
| `p18_sweep_qwen3_1_7b.json` | bfloat16/GPU sweep, 28 layers |
| `p18_sweep_qwen3_4b.json` | bfloat16/GPU sweep, 36 layers |
| `p18_sweep_qwen3_8b.json` | bfloat16/GPU sweep, 36 layers |
| `p18_sweep_qwen3_14b.json` | bfloat16/GPU sweep, 40 layers |
| `p18_sweep_qwen3_32b.json` | bfloat16/GPU sweep, 64 layers |
| `p18_fp32_compare.json` | float32/CPU sweep, Qwen3-0.6B (28 layers) |
| `p18_baseline_qwen3_8b_bfloat16_cpu.json` | bfloat16/CPU baseline, Qwen3-8B (precision control) |
| `p18_direction_amplify_8b.json` | Amplification results, Qwen3-8B bfloat16 |

---

## Requirements

```
transformers>=5.9.0
torch>=2.x
```

Run sweeps with:
```bash
python crystallization_sweep.py --model_id Qwen/Qwen3-8B --out_prefix p18_sweep_qwen3_8b
```

---

## Related papers (same series)

- P15: Behavioral Routing Layer — https://doi.org/10.5281/zenodo.20348071
- P16: L6 Self-Referential Nexus — https://doi.org/10.5281/zenodo.20357150
- P17: Scale-Dependent L6 Ablation — https://doi.org/10.5281/zenodo.20358863
