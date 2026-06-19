"""
cdm_v7_post_analysis.py — Post-training analysis for CDM V7 (HORN + Kuramoto + β).
DuoNeural / Archon — 2026-06-17

Run on pod 54360 after training completes (~30k steps):
  python3 cdm_v7_post_analysis.py

Produces:
  /workspace/cdm_v7_analysis.json — full analysis results

KEY QUESTIONS:
  1. What did β converge to per layer?
  2. Did γ/ω differentiate into the HORN three-regime structure again?
  3. Is V7's DHP structure similar to HORN (compare γ/ω/regime)?
  4. Does the β value correlate with regime (underdamped vs overdamped)?
"""

import sys
sys.path.insert(0, '/workspace')

import json, math, torch
import torch.nn.functional as F
from cdm_model_v7 import CDMLanguageModelV7, CDMConfigV7


CHECKPOINT = "/workspace/cdm_v7_horn_kuramoto/best/model.pt"

HORN_REF = {   # V6 HORN final values for comparison
    "val_ce": 1.5818,
    "layers": {
        0: {"gamma": 0.834, "omega": 0.884, "regime": "underdamped"},
        1: {"gamma": 0.760, "omega": 0.602, "regime": "overdamped"},
        2: {"gamma": 0.885, "omega": 0.750, "regime": "overdamped"},
        3: {"gamma": 0.798, "omega": 0.667, "regime": "overdamped"},
        4: {"gamma": 0.804, "omega": 0.942, "regime": "underdamped_partial"},
        5: {"gamma": 0.798, "omega": 0.785, "regime": "mixed"},
        6: {"gamma": 0.648, "omega": 0.737, "regime": "underdamped"},
        7: {"gamma": 0.632, "omega": 0.669, "regime": "underdamped"},
    }
}


def analyze():
    print("=== CDM V7 Post-Training Analysis ===")

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    cfg_dict = ckpt.get("config", {})
    val_ce = ckpt.get("val_loss", float("nan"))
    step = ckpt.get("step", -1)

    print(f"Checkpoint: step={step}, val_CE={val_ce:.4f}")
    print(f"vs HORN:   val_CE=1.5818  Δ={val_ce - 1.5818:+.4f}")

    # Build model
    cfg = CDMConfigV7(
        vocab_size  = cfg_dict.get("vocab_size", 50257),
        d_model     = cfg_dict.get("d_model", 384),
        n_layers    = cfg_dict.get("n_layers", 8),
        n_heads     = cfg_dict.get("n_heads", 8),
        n_kv_heads  = cfg_dict.get("n_kv_heads", 4),
        d_ff        = cfg_dict.get("d_ff", 1024),
        K           = cfg_dict.get("K", 16),
        max_len     = cfg_dict.get("max_len", 512),
        d_osc       = cfg_dict.get("d_osc", 8),
        beta_init   = cfg_dict.get("beta_init", 0.0),
    )
    model = CDMLanguageModelV7(cfg)
    model.load_state_dict(state, strict=False)
    model.eval()

    results = {
        "step": step,
        "val_ce": val_ce,
        "delta_vs_horn": val_ce - 1.5818,
        "beat_horn": val_ce < 1.5818,
        "layers": {},
        "horn_ref": HORN_REF,
    }

    print("\n=== Per-Layer Analysis ===")
    print(f"{'L':<3} {'β':>8} {'γ_mean':>8} {'ω_mean':>8} {'under%':>7} {'regime':<15} {'vs HORN γ':>10}")
    print("-" * 75)

    for l_idx, block in enumerate(model.blocks):
        cdm = block.cdm

        with torch.no_grad():
            gamma   = F.softplus(cdm.raw_gamma).numpy()
            omega   = F.softplus(cdm.raw_omega).numpy()
            beta    = cdm.beta.item()

        g_mean = float(gamma.mean())
        o_mean = float(omega.mean())
        under_frac = float((omega > gamma).mean())
        tau_L_mean = 1.0 / g_mean
        tau_star_mean = 0.72 / g_mean

        if under_frac > 0.6:
            regime = "underdamped"
        elif under_frac < 0.3:
            regime = "overdamped"
        else:
            regime = f"mixed({under_frac:.0%})"

        horn_gamma = HORN_REF["layers"].get(l_idx, {}).get("gamma", float("nan"))
        delta_gamma = g_mean - horn_gamma

        results["layers"][l_idx] = {
            "beta": beta,
            "gamma_mean": g_mean,
            "gamma_std": float(gamma.std()),
            "gamma_min": float(gamma.min()),
            "gamma_max": float(gamma.max()),
            "omega_mean": o_mean,
            "underdamped_frac": under_frac,
            "regime": regime,
            "tau_L": tau_L_mean,
            "tau_star_dhp": tau_star_mean,
            "delta_gamma_vs_horn": delta_gamma,
        }

        print(f"L{l_idx:<2} {beta:>8.3f} {g_mean:>8.3f} {o_mean:>8.3f} {under_frac:>6.0%}  {regime:<15} {delta_gamma:>+10.3f}")

    # β summary
    betas = [results["layers"][l]["beta"] for l in range(cfg.n_layers)]
    print(f"\nβ summary: mean={sum(betas)/len(betas):.3f}  min={min(betas):.3f}  max={max(betas):.3f}")
    print(f"All β negative: {all(b < 0 for b in betas)}")
    print(f"β range: [{min(betas):.3f}, {max(betas):.3f}]")

    results["beta_summary"] = {
        "mean": sum(betas)/len(betas),
        "min": min(betas),
        "max": max(betas),
        "all_negative": all(b < 0 for b in betas),
    }

    # Regime structure
    regimes = [results["layers"][l]["regime"] for l in range(cfg.n_layers)]
    n_under = sum("underdamped" in r for r in regimes)
    n_over  = sum("overdamped" in r for r in regimes)
    n_mixed = sum("mixed" in r for r in regimes)

    print(f"\nRegime structure: {n_under} underdamped, {n_over} overdamped, {n_mixed} mixed")
    print(f"HORN had:          2 underdamped (L0,L6-7), 5-6 overdamped, rest mixed")

    # DHP τ* summary
    tau_stars = [results["layers"][l]["tau_star_dhp"] for l in range(cfg.n_layers)]
    print(f"\nτ* range (per layer, by γ): [{min(tau_stars):.3f}, {max(tau_stars):.3f}] positions")
    print(f"HORN τ* range:              [0.81, 1.10] positions")

    results["regime_summary"] = {
        "n_underdamped": n_under,
        "n_overdamped": n_over,
        "n_mixed": n_mixed,
    }
    results["tau_star_range"] = [min(tau_stars), max(tau_stars)]

    # Verdict
    print("\n=== VERDICT ===")
    delta = val_ce - 1.5818
    if delta < 0:
        print(f"✅ V7 BEATS HORN by Δ{delta:+.4f}")
        print(f"   β < 0 (temporal lead routing) + HORN dynamics = new best")
    elif abs(delta) < 0.001:
        print(f"≈  V7 TIES HORN (Δ{delta:+.4f})")
        print(f"   β routing adds complexity but doesn't hurt")
    else:
        print(f"❌ V7 does NOT beat HORN (Δ{delta:+.4f})")
        print(f"   V7 is more complex but HORN simpler routing wins at this scale")

    # Save
    outpath = "/workspace/cdm_v7_analysis.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outpath}")

    return results


if __name__ == "__main__":
    analyze()
