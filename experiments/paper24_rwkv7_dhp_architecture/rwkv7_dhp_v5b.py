"""
RWKV-7 DHP v5b — continuation after CTM-like crash
====================================================
CTM-like v5 completed all 4 seeds before the aggregate print bug crashed the script.
This script:
  1. Hardcodes CTM-like aggregate results from the log
  2. Runs RWKV-7, RWKV-6, Mamba with the same v5 protocol
  3. Prints final comparative summary + saves JSON

Archon — DuoNeural — 2026-05-27
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# ── Import all classes from fixed v5 ─────────────────────────────────────────
v5_src = open(os.path.join(os.path.dirname(__file__), "rwkv7_dhp_v5.py")).read()
# Execute everything up to Main block
exec(v5_src.split("# ─────────────────────────────────────────────────────────────────────────────\n# Main")[0])

import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent

print(f"\n{'='*70}", flush=True)
print(f"RWKV-7 DHP v5b (continuation) — {datetime.now().isoformat()}", flush=True)
print(f"Skipping CTM-like (already done). Running RWKV-7, RWKV-6, Mamba.", flush=True)
print(f"{'='*70}\n", flush=True)

traj = lorenz_trajectory(n=8000, dt=DT)
assert not np.isnan(traj).any()
print(f"Lorenz: {len(traj)} steps, no NaN ✓", flush=True)

# ── Hardcoded CTM-like results from v5 log ───────────────────────────────────
ctm_seeds = [
    {"seed": 0, "hlcv_init": 0.369, "hlcv_post": 0.511, "delta": +0.142, "ok": True,
     "half_lives": [9, 19, 21, 8, 14, 9, 24, 5], "hl_mean": 13.6, "inter_corr": 0.2210},
    {"seed": 1, "hlcv_init": 0.129, "hlcv_post": 0.572, "delta": +0.444, "ok": True,
     "half_lives": [42, 25, 17, 19, 11, 7, 13, 17], "hl_mean": 18.9, "inter_corr": 0.3841},
    {"seed": 2, "hlcv_init": 0.557, "hlcv_post": 0.401, "delta": -0.156, "ok": True,
     "half_lives": [13, 17, 8, 13, 17, 21, 16, 31], "hl_mean": 17.0, "inter_corr": 0.2682},
    {"seed": 3, "hlcv_init": 0.420, "hlcv_post": 0.380, "delta": -0.040, "ok": True,
     "half_lives": [16, 15, 13, 8, 24, 12, 21, 27], "hl_mean": 17.0, "inter_corr": 0.3135},
]
ctm_post = [s["hlcv_post"] for s in ctm_seeds]
ctm_init = [s["hlcv_init"] for s in ctm_seeds]
ctm_deltas = [s["delta"] for s in ctm_seeds]

all_results = {
    "CTM-like v5": {
        "hlcv_init_mean": float(np.mean(ctm_init)),
        "hlcv_post_mean": float(np.mean(ctm_post)),
        "hlcv_post_std":  float(np.std(ctm_post)),
        "delta_hlcv":     float(np.mean(ctm_deltas)),
        "dhp_by_delta":   bool(np.mean(ctm_deltas) > 0.05),
        "dhp_absolute":   bool(np.mean(ctm_post) > HL_CV_THRESHOLD),
        "n_ok": 4, "n_seeds": 4,
        "final_loss_mean": 0.121,
        "seed_details": ctm_seeds,
    }
}

print(f"CTM-like v5 (from log):", flush=True)
print(f"  HL_CV init mean = {np.mean(ctm_init):.3f}", flush=True)
print(f"  HL_CV post mean = {np.mean(ctm_post):.3f} ± {np.std(ctm_post):.3f}", flush=True)
print(f"  Δ(HL_CV) mean   = {np.mean(ctm_deltas):+.3f}", flush=True)

# ── Run remaining architectures ───────────────────────────────────────────────
MODELS_SPEC = {
    "RWKV-7 v5":   lambda: RWKV7CellV5(D_IN, D_HIDDEN, n_heads=4),
    "RWKV-6 v5":   lambda: RWKV6CellV5(D_IN, D_HIDDEN, n_heads=4),
    "Mamba v5":    lambda: MambaV5(D_IN, D_HIDDEN),
}

for name, model_fn in MODELS_SPEC.items():
    print(f"\n{'='*70}", flush=True)
    print(f"Architecture: {name}", flush=True)
    print(f"{'='*70}", flush=True)

    seed_hlcv_post  = []
    seed_hlcv_init  = []
    seed_final_loss = []
    seed_ok         = []
    seed_details    = []

    for seed in range(N_SEEDS):
        print(f"\n  --- Seed {seed} ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model_fn()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if seed == 0:
            print(f"  {n_params:,} trainable params", flush=True)

        init_hlcv = measure_init_hlcv(model, traj, n_runs=3)
        print(f"  HL_CV (init):  {init_hlcv:.3f}", flush=True)

        losses, ok, final_loss = train_model(model, traj, seed=seed)

        spec = slot_specialization(model, traj)
        post_hlcv = spec['half_life_cv'] if spec else None
        print(f"  HL_CV (post):  {post_hlcv:.3f}", flush=True)

        if init_hlcv is not None and post_hlcv is not None:
            delta = post_hlcv - init_hlcv
            print(f"  Δ(HL_CV):      {delta:+.3f} ← {'DHP SIGNAL ✓' if delta > 0.05 else 'no signal'}", flush=True)
            seed_details.append({"seed": seed, "hlcv_init": init_hlcv, "hlcv_post": post_hlcv,
                                  "delta": delta, "ok": ok})

        if spec:
            print(f"  HL mean: {spec['half_life_mean']:.1f} | inter-corr: {spec['mean_slot_corr']:.4f}", flush=True)
            print(f"  half-lives: {spec['half_lives']}", flush=True)

        if init_hlcv is not None: seed_hlcv_init.append(init_hlcv)
        if post_hlcv is not None: seed_hlcv_post.append(post_hlcv)
        seed_final_loss.append(final_loss)
        seed_ok.append(ok)

    mean_init   = float(np.mean(seed_hlcv_init)) if seed_hlcv_init else None
    mean_post   = float(np.mean(seed_hlcv_post)) if seed_hlcv_post else None
    std_post    = float(np.std(seed_hlcv_post))  if len(seed_hlcv_post) > 1 else 0.0
    delta_hlcv  = float(mean_post - mean_init)   if (mean_post is not None and mean_init is not None) else None
    dhp_by_delta = (delta_hlcv is not None and delta_hlcv > 0.05)
    dhp_absolute = (mean_post is not None and mean_post > HL_CV_THRESHOLD)

    init_s = f"{mean_init:.3f}" if mean_init is not None else "N/A"
    print(f"\n  === {name} AGGREGATE ({N_SEEDS} seeds) ===", flush=True)
    print(f"  HL_CV init:   {init_s}", flush=True)
    print(f"  HL_CV post:   {mean_post:.3f} ± {std_post:.3f}" if mean_post is not None else "  HL_CV post:   N/A", flush=True)
    print(f"  Δ(HL_CV):     {delta_hlcv:+.3f}" if delta_hlcv is not None else "  Δ(HL_CV):     N/A", flush=True)
    print(f"  DHP (Δ>0.05): {'✓' if dhp_by_delta else '✗'}", flush=True)
    print(f"  DHP (abs>0.3): {'✓' if dhp_absolute else '✗'}", flush=True)
    print(f"  Training OK:  {sum(seed_ok)}/{N_SEEDS}", flush=True)

    all_results[name] = {
        "hlcv_init_mean":  mean_init,
        "hlcv_post_mean":  mean_post,
        "hlcv_post_std":   std_post,
        "delta_hlcv":      delta_hlcv,
        "dhp_by_delta":    dhp_by_delta,
        "dhp_absolute":    dhp_absolute,
        "n_ok":            sum(seed_ok),
        "n_seeds":         N_SEEDS,
        "final_loss_mean": float(np.mean(seed_final_loss)),
        "seed_details":    seed_details,
    }

# ── Final comparative summary ─────────────────────────────────────────────────
print(f"\n\n{'='*70}", flush=True)
print(f"RWKV-7 DHP v5 — FULL COMPARATIVE SUMMARY", flush=True)
print(f"{'='*70}", flush=True)
print(f"{'Architecture':<22} {'HL_CV_init':>10} {'HL_CV_post':>11} {'Δ(HL_CV)':>9} {'DHP(Δ)':>7} {'DHP(abs)':>9}", flush=True)
print(f"{'-'*70}", flush=True)
for name, r in all_results.items():
    init_s = f"{r['hlcv_init_mean']:.3f}" if r['hlcv_init_mean'] is not None else "  N/A"
    post_s = f"{r['hlcv_post_mean']:.3f}" if r['hlcv_post_mean'] is not None else "  N/A"
    delt_s = f"{r['delta_hlcv']:+.3f}"    if r['delta_hlcv']     is not None else "  N/A"
    print(f"  {name:<20} {init_s:>10} {post_s:>11} {delt_s:>9} "
          f"{'✓' if r['dhp_by_delta'] else '✗':>7} {'✓' if r['dhp_absolute'] else '✗':>9}", flush=True)

print(f"\n  DHP hypothesis: RWKV-7 Δ > CTM-like Δ > Mamba Δ, RWKV-6 Δ ≈ 0", flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    "version": "v5",
    "timestamp": datetime.now().isoformat(),
    "tau_L": TAU_L, "dt": DT, "seq_len": SEQ_LEN,
    "pred_horizons": PRED_HORIZONS,
    "n_steps": N_STEPS, "n_seeds": N_SEEDS,
    "equal_horizon_weights": True,
    "dhp_by_delta_threshold": 0.05,
    "results": all_results,
}
out_path = HERE / "rwkv7_dhp_v5.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved → {out_path}", flush=True)
print("v5b complete.", flush=True)
