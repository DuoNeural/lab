#!/usr/bin/env python3
"""
cdm_v3_alpha_probe.py — Probe learnable alpha decay rates + routing in CDM V3

Runs at a saved checkpoint (default: step_005000) to answer the key V3 question:
  "Do σ(α_k) values differentiate from ~0.5 initialization?"

Expected outcomes:
  - No diff (~0.5 all): LBL not enough pressure, or too early
  - Fast/slow hierarchy:  some slots → high α_k (fast, syntactic), some → low α_k (slow, semantic)
  - Layer-dependent:      early layers might stay near 0.5, deeper layers differentiate first

Also runs the routing probe (same method as V2) to see slot specialization.

Archon — DuoNeural 2026-06-12
"""

import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path

# need cdm_model_v3 to be in path
sys.path.insert(0, str(Path(__file__).parent))
from cdm_model_v3 import CDMConfigV3, CDMLanguageModelV3

CKPT_PATH = "/workspace/cdm_v3_full/step_005000/checkpoint.pt"
OUT_PATH  = "/workspace/cdm_v3_alpha_probe_step5000.json"

# ── mini test sentences for routing ──────────────────────────────────────────
TEST_PROMPTS = [
    "Once upon a time there was a little girl named Lily",
    "The dog ran fast and barked at the cat",
    "She said hello and smiled at her friend",
    "The big blue ball rolled down the hill",
    "Tom liked to eat apples and bananas every day",
]


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_d = ckpt["config"]
    cfg = CDMConfigV3(
        vocab_size = cfg_d.get("vocab_size", 50257),
        d_model    = cfg_d.get("d_model", 384),
        n_layers   = cfg_d.get("n_layers", 8),
        n_heads    = cfg_d.get("n_heads", 8),
        n_kv_heads = cfg_d.get("n_kv_heads", 4),
        d_ff       = cfg_d.get("d_ff", 1024),
        K          = cfg_d.get("K", 16),
        max_len    = cfg_d.get("max_len", 512),
        lbl_coeff  = cfg_d.get("lbl_coeff", 0.01),
    )
    model = CDMLanguageModelV3(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt.get("step", "?")


def extract_alphas(model):
    """Extract σ(log_alpha) for every block. Returns dict layer_idx → list[K floats]."""
    alphas = {}
    for i, block in enumerate(model.blocks):
        log_a = block.cdm.log_alpha.detach()           # (K,)
        sigma_a = torch.sigmoid(log_a).tolist()        # (K,) → list
        alphas[i] = sigma_a
    return alphas


def routing_probe(model, prompts, tokenizer_path="/workspace/smollm2_checkpoints_run9/best"):
    """Run routing on test prompts and return per-layer gate statistics."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception:
        # fallback: use gpt2 tokenizer
        try:
            from transformers import GPT2TokenizerFast
            tok = GPT2TokenizerFast.from_pretrained("gpt2")
        except Exception as e:
            print(f"  WARN: no tokenizer available ({e}), skipping routing probe")
            return None

    all_gates = []  # list of (T, K) tensors per layer
    gate_accum = [[] for _ in range(len(model.blocks))]

    def make_hook(layer_idx):
        def hook(module, inp, out):
            # out = (slots_all, gates, route_probs)
            gates = out[1].detach()   # (B, T, K)
            gate_accum[layer_idx].append(gates[0].cpu())  # (T, K)
        return hook

    hooks = []
    for i, block in enumerate(model.blocks):
        h = block.cdm.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        for prompt in prompts:
            ids = tok.encode(prompt, return_tensors="pt")
            _ = model(ids)

    for h in hooks:
        h.remove()

    # Aggregate per layer
    layer_stats = {}
    for i, gate_list in enumerate(gate_accum):
        cat = torch.cat(gate_list, dim=0)              # (total_T, K)
        entropy = -( (cat + 1e-9).log() * cat ).sum(-1).mean().item()
        max_entropy = torch.log(torch.tensor(cat.shape[-1], dtype=torch.float)).item()
        dominant_slot = cat.mean(0).argmax().item()
        dominant_pct  = cat.mean(0).max().item()
        k_eff = torch.exp(
            -( cat.mean(0) * (cat.mean(0) + 1e-9).log() ).sum()
        ).item()
        layer_stats[i] = {
            "entropy_pct": round(entropy / max_entropy * 100, 2),
            "k_eff":       round(k_eff, 2),
            "dominant_slot": dominant_slot,
            "dominant_pct":  round(float(dominant_pct), 4),
            "mean_gate_per_slot": [round(float(v), 4) for v in cat.mean(0).tolist()],
        }

    return layer_stats


def main():
    print(f"\n{'='*60}")
    print(f"CDM V3 Alpha Probe")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"{'='*60}\n")

    if not Path(CKPT_PATH).exists():
        print(f"ERROR: checkpoint not found: {CKPT_PATH}")
        sys.exit(1)

    model, cfg, step = load_model(CKPT_PATH)
    print(f"  Loaded step {step}, K={cfg.K}, layers={cfg.n_layers}")

    # ── Alpha analysis ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"ALPHA DECAY RATES σ(α_k) per layer  (init=0.5, high=fast, low=slow)")
    print(f"{'─'*60}")

    alphas = extract_alphas(model)
    all_vals = []
    for layer_idx, sigma_a in alphas.items():
        sigma_a_t = torch.tensor(sigma_a)
        min_a = sigma_a_t.min().item()
        max_a = sigma_a_t.max().item()
        std_a = sigma_a_t.std().item()
        # format as small table row
        vals_str = " ".join(f"{v:.3f}" for v in sigma_a)
        print(f"  L{layer_idx}: min={min_a:.3f} max={max_a:.3f} std={std_a:.4f}")
        print(f"       {vals_str}")
        all_vals.extend(sigma_a)

    global_std = torch.tensor(all_vals).std().item()
    global_mean = torch.tensor(all_vals).mean().item()
    differentiated = global_std > 0.05   # threshold: >5% std = meaningful spread

    print(f"\n  GLOBAL: mean={global_mean:.3f} std={global_std:.4f}")
    if differentiated:
        print(f"  VERDICT: ✓ ALPHA DIFFERENTIATION CONFIRMED (std>{0.05:.2f})")
    else:
        print(f"  VERDICT: ✗ No significant differentiation yet (std<{0.05:.2f}, still near init)")

    # ── Routing probe ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"ROUTING PROBE (TinyStories-style sentences)")
    print(f"{'─'*60}")
    routing = routing_probe(model, TEST_PROMPTS)

    if routing:
        for layer_idx, stats in routing.items():
            print(f"  L{layer_idx}: H={stats['entropy_pct']:.1f}% K_eff={stats['k_eff']:.1f} "
                  f"top_slot={stats['dominant_slot']} ({stats['dominant_pct']*100:.1f}%)")

    # ── Save results ──────────────────────────────────────────────────────────
    result = {
        "step": step,
        "alphas_per_layer": alphas,
        "global_alpha_mean": round(global_mean, 4),
        "global_alpha_std": round(global_std, 4),
        "differentiated": differentiated,
        "routing_per_layer": routing,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved → {OUT_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
