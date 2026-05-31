"""
RWKV-7 Decay Parameter Analysis — Post-training DHP Diagnostic
================================================================
After training RWKV-7 on multi-horizon Lorenz prediction (v4),
analyze the learned per-channel decay parameters to test DHP.

DHP prediction for RWKV-7:
  The per-channel decay w_t = sigmoid(W_w @ x + bias) determines
  how fast each channel "forgets" past information.
  Channel timescale: τ_channel ≈ -1 / ln(w_eff)
  where w_eff = mean(w_t) over the Lorenz attractor.

  If RWKV-7 has learned DHP:
    The distribution of τ_channel should cluster near τ* ≈ 0.72 × τ_L ≈ 79 steps.
    Specifically: fraction of channels with τ_channel in [60.5, 93.5] should be elevated
    relative to the initial linspace initialization (which was uniform).

  If RWKV-7 has NOT learned DHP:
    Channels should be roughly uniformly distributed across timescales
    OR clustered at short timescales (if model preferred short-horizon prediction).

Archon — DuoNeural — 2026-05-27
"""
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'figure.dpi': 150})
HERE = Path(__file__).parent

# ── Config (must match v4) ────────────────────────────────────────────────────
DT     = 0.01
TAU_L  = 110.0
DHP_LO, DHP_HI = 0.55, 0.85
D_MODEL = 64
N_HEADS = 4
N       = D_MODEL // N_HEADS  # 16
D_IN    = 3

def lorenz_trajectory(n=3000, dt=DT, sigma=10., rho=28., beta=8/3, seed=42):
    rng = np.random.default_rng(seed)
    x = np.array([1., 1., 1.]) + rng.standard_normal(3) * 0.01
    traj = np.zeros((n, 3))
    for t in range(n):
        dx = sigma * (x[1] - x[0])
        dy = x[0] * (rho - x[2]) - x[1]
        dz = x[0] * x[1] - beta * x[2]
        x = x + np.array([dx, dy, dz]) * dt
        traj[t] = x
    traj = (traj - traj.mean(0)) / (traj.std(0) + 1e-8)
    return traj

def compute_effective_decay(weights_path, traj_data):
    """
    Load trained RWKV-7 weights, run Lorenz attractor through,
    compute per-channel effective decay w_eff = mean(sigmoid(W_w @ x + bias)).
    Returns τ_channel for each of the D_MODEL channels.
    """
    state = torch.load(weights_path, map_location='cpu', weights_only=True)

    W_w_weight = state.get('W_w.weight', None)
    W_w_bias   = state.get('W_w.bias',   None)
    proj_in_w  = state.get('proj_in.weight', None)
    proj_in_b  = state.get('proj_in.bias', None)

    if W_w_weight is None or W_w_bias is None:
        print(f"  No W_w parameters found in {weights_path.name}")
        return None, None

    x = torch.tensor(traj_data, dtype=torch.float32)   # (T, 3)
    x_proj = torch.nn.functional.linear(x, proj_in_w, proj_in_b)  # (T, D_MODEL)

    # w_t = sigmoid(W_w @ x_proj + W_w.bias) for each timestep
    w_all = torch.sigmoid(torch.nn.functional.linear(x_proj, W_w_weight, W_w_bias))  # (T, D_MODEL)
    w_eff = w_all.mean(0)   # (D_MODEL,) — effective decay per channel

    # Timescale: τ = -1 / ln(w). Clamp to avoid log(0) or log(1).
    w_eff_clamped = w_eff.clamp(1e-4, 1 - 1e-4)
    tau_channels = -1.0 / torch.log(w_eff_clamped)   # (D_MODEL,)

    return tau_channels.numpy(), w_eff.numpy()


def analyze(weights_path, traj_data, label="RWKV-7 v4"):
    tau_ch, w_eff = compute_effective_decay(weights_path, traj_data)
    if tau_ch is None:
        return None

    tau_ch = np.clip(tau_ch, 0, TAU_L * 2)  # cap at 2*τ_L for display

    dhp_lo_abs = DHP_LO * TAU_L   # 60.5 steps
    dhp_hi_abs = DHP_HI * TAU_L   # 93.5 steps

    frac_dhp    = np.mean((tau_ch >= dhp_lo_abs) & (tau_ch <= dhp_hi_abs))
    frac_short  = np.mean(tau_ch < 10)
    frac_medium = np.mean((tau_ch >= 10) & (tau_ch < dhp_lo_abs))
    median_tau  = np.median(tau_ch)
    mean_tau    = np.mean(tau_ch)

    print(f"\n{'='*60}")
    print(f"{label} — Decay Parameter Analysis")
    print(f"{'='*60}")
    print(f"  τ_L = {TAU_L}, DHP range = [{dhp_lo_abs:.0f}, {dhp_hi_abs:.0f}] steps")
    print(f"  Channels analyzed: {len(tau_ch)}")
    print(f"  Mean τ_channel:    {mean_tau:.1f} steps")
    print(f"  Median τ_channel:  {median_tau:.1f} steps")
    print(f"  Short (< 10):      {frac_short:.1%}")
    print(f"  Medium (10-60):    {frac_medium:.1%}")
    print(f"  DHP zone (60-93):  {frac_dhp:.1%} ← {'✓ elevated' if frac_dhp > 0.15 else '✗ not elevated'}")
    print(f"  Initial uniform:   {1/(TAU_L*2/10):.1%} expected in DHP zone from linspace init")

    result = {
        "label": label,
        "tau_channels": tau_ch.tolist(),
        "w_eff": w_eff.tolist(),
        "mean_tau": float(mean_tau),
        "median_tau": float(median_tau),
        "frac_dhp_zone": float(frac_dhp),
        "frac_short": float(frac_short),
        "dhp_elevated": bool(frac_dhp > 0.15),
    }
    return result


# ── Main ─────────────────────────────────────────────────────────────────────
traj = lorenz_trajectory(n=3000)
print(f"Lorenz: {len(traj)} steps, no NaN ✓")

# Look for all v4 RWKV-7 weight files
rwkv7_weights = sorted(HERE.glob("weights_v4_RWKV*.pt"))
print(f"\nFound {len(rwkv7_weights)} RWKV-7 weight file(s):")
for f in rwkv7_weights:
    print(f"  {f.name}")

all_results = []
for wf in rwkv7_weights:
    r = analyze(wf, traj, label=wf.stem.replace("weights_v4_", "").replace("_", " "))
    if r:
        all_results.append(r)

if not all_results:
    # Try initial weights as reference
    print("\nNo trained weights found yet. Computing INITIAL decay distribution...")
    target_taus = np.linspace(1.0, TAU_L * DHP_HI, D_MODEL)
    w_targets   = np.exp(-1.0 / target_taus)
    tau_init    = -1.0 / np.log(np.clip(w_targets, 1e-4, 1 - 1e-4))
    dhp_lo_abs  = DHP_LO * TAU_L
    dhp_hi_abs  = DHP_HI * TAU_L
    frac = np.mean((tau_init >= dhp_lo_abs) & (tau_init <= dhp_hi_abs))
    print(f"  Initial distribution: uniform from τ=1 to τ=93.5 steps")
    print(f"  Fraction in DHP zone [{dhp_lo_abs:.0f},{dhp_hi_abs:.0f}]: {frac:.1%}")
    print(f"  After training: if DHP → fraction should INCREASE above {frac:.1%}")
    all_results = [{"label": "INITIAL (linspace)", "tau_channels": tau_init.tolist(),
                    "frac_dhp_zone": float(frac), "dhp_elevated": False}]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
colors = ['#1565C0', '#E65100', '#2E7D32', '#6A1B9A']
bins = np.linspace(0, TAU_L * 1.5, 40)

for i, r in enumerate(all_results):
    ax.hist(r['tau_channels'], bins=bins, alpha=0.6, color=colors[i % len(colors)],
            label=f"{r['label']} (DHP zone: {r['frac_dhp_zone']:.0%})", density=True)

ax.axvspan(DHP_LO * TAU_L, DHP_HI * TAU_L, alpha=0.12, color='gold', label=f'DHP zone [{DHP_LO*TAU_L:.0f}–{DHP_HI*TAU_L:.0f} steps]')
ax.axvline(TAU_L, color='red', ls='--', lw=1.5, alpha=0.7, label=f'τ_L = {TAU_L:.0f}')
ax.set_xlabel("Channel timescale τ_channel (steps)")
ax.set_ylabel("Density")
ax.set_title("RWKV-7 Learned Decay Distribution\n(DHP predicts elevated density in gold zone)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
out_fig = HERE / "figs" / "fig_rwkv7_decay_analysis.pdf"
out_fig.parent.mkdir(exist_ok=True)
fig.savefig(out_fig, bbox_inches='tight')
print(f"\nFigure saved: {out_fig}")

# Save JSON
with open(HERE / "rwkv7_decay_analysis.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Data saved: {HERE / 'rwkv7_decay_analysis.json'}")
