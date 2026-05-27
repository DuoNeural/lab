"""
Figure: Aligned vs Base Model W-Shape Comparison
=================================================
P24 non-aligned baseline experiment result.

Shows that:
1. W-shape is architectural in origin (both models)
2. Alignment amplifies the L16 secondary peak 2.3×
3. Alignment raises the readout-layer similarity floor
4. hate_speech outlier is more extreme in base model

Archon — DuoNeural — 2026-05-27
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})

HERE = Path(__file__).parent

# Load data
with open(HERE.parent / "paper22" / "l27_convergence_v2.json") as f:
    aligned = json.load(f)

with open(HERE / "l27_convergence_base.json") as f:
    base = json.load(f)

layers = list(range(28))

def get_trajectory(data):
    ci = data['results']['convergence_ci']
    means = [ci[str(i)]['mean'] for i in layers]
    lo    = [ci[str(i)]['lo']   for i in layers]
    hi    = [ci[str(i)]['hi']   for i in layers]
    return np.array(means), np.array(lo), np.array(hi)

a_means, a_lo, a_hi = get_trajectory(aligned)
b_means, b_lo, b_hi = get_trajectory(base)

# ── Figure 1: Side-by-side W-shape comparison ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: mean convergence trajectories with CI
ax = axes[0]
ax.fill_between(layers, a_lo, a_hi, alpha=0.15, color='#1565C0')
ax.fill_between(layers, b_lo, b_hi, alpha=0.15, color='#E65100')
ax.plot(layers, a_means, 'o-', color='#1565C0', lw=2.2, ms=4,
        label='Aligned (Qwen3-0.6B)', zorder=5)
ax.plot(layers, b_means, 's--', color='#E65100', lw=2.2, ms=4,
        label='Base (Qwen3-0.6B-Base)', zorder=4)

# Annotate key zones
ax.axvline(x=10, color='#43A047', lw=1.5, ls=':', alpha=0.6, label='L10 minimum')
ax.axvline(x=16, color='#8E24AA', lw=1.5, ls='--', alpha=0.6, label='L16 secondary peak')

# Annotate peak difference
ax.annotate(
    f'L16 amplification\naligned: +0.090\nbase:    +0.039\nratio: 2.3×',
    xy=(16, a_means[16]),
    xytext=(18.5, 0.72),
    fontsize=8, color='#1A237E',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', alpha=0.8),
    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1),
)
ax.annotate(
    '',
    xy=(16, b_means[16]),
    xytext=(18.5, 0.69),
    arrowprops=dict(arrowstyle='->', color='#E65100', lw=1),
)

# Readout floor annotation
ax.annotate(
    f'L27 floor\naligned: {a_means[27]:.3f}\nbase:    {b_means[27]:.3f}',
    xy=(27, (a_means[27] + b_means[27])/2),
    xytext=(23.5, 0.41),
    fontsize=8, color='#212121',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', alpha=0.8),
    arrowprops=dict(arrowstyle='->', color='#424242', lw=1),
)

ax.set_xlabel("Layer")
ax.set_ylabel("Mean pairwise cosine similarity")
ax.set_title("W-Shape: Aligned vs Base Model\n(Qwen3-0.6B vs Qwen3-0.6B-Base, n=50 per category)")
ax.set_xlim(-0.5, 27.5)
ax.set_ylim(0.30, 0.95)
ax.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.3)

# Right: hate_speech pairwise comparison at all layers
ax = axes[1]
PAIRS = ["weapons_vs_hate_speech", "drugs_vs_hate_speech", "cybercrime_vs_hate_speech",
         "weapons_vs_drugs"]  # non-hate pair for comparison
PAIR_LABELS = {
    "weapons_vs_hate_speech": "Weapons vs Hate (aligned)",
    "drugs_vs_hate_speech": "Drugs vs Hate (aligned)",
    "cybercrime_vs_hate_speech": "Cybercrime vs Hate (aligned)",
    "weapons_vs_drugs": "Weapons vs Drugs (aligned)",
}
COLORS_A = {"weapons_vs_hate_speech": '#F44336', "drugs_vs_hate_speech": '#FF5722',
             "cybercrime_vs_hate_speech": '#009688', "weapons_vs_drugs": '#2196F3'}
COLORS_B = {"weapons_vs_hate_speech": '#B71C1C', "drugs_vs_hate_speech": '#BF360C',
             "cybercrime_vs_hate_speech": '#004D40', "weapons_vs_drugs": '#0D47A1'}

for pair in PAIRS:
    a_traj = np.array([aligned['results']['layer_cosines'][str(l)].get(pair, float('nan'))
                        for l in layers])
    b_traj = np.array([base['results']['layer_cosines'][str(l)].get(pair, float('nan'))
                        for l in layers])
    lw_a = 2.5 if 'hate' in pair else 1.5
    lw_b = 2.5 if 'hate' in pair else 1.5
    ls_b = '--' if 'hate' in pair else '-.'
    ax.plot(layers, a_traj, '-', color=COLORS_A[pair], lw=lw_a,
            label=f"{'→'.join(p.title() for p in pair.split('_vs_'))} (aligned)")
    ax.plot(layers, b_traj, ls_b, color=COLORS_B[pair], lw=lw_b,
            label=f"{'→'.join(p.title() for p in pair.split('_vs_'))} (base)", alpha=0.8)

ax.axvline(x=10, color='#43A047', lw=1, ls=':', alpha=0.5)
ax.axvline(x=16, color='#8E24AA', lw=1, ls='--', alpha=0.5)
ax.set_xlabel("Layer")
ax.set_ylabel("Pairwise cosine similarity")
ax.set_title("Pairwise Trajectories: Hate Speech Outlier\n(Aligned=solid, Base=dashed)")
ax.set_xlim(-0.5, 27.5)
ax.set_ylim(-0.05, 1.02)
ax.legend(loc='upper right', fontsize=6.5)
ax.grid(alpha=0.3)
# Annotate L27 hate_speech final values
for pair in ["weapons_vs_hate_speech", "drugs_vs_hate_speech"]:
    a_val = aligned['results']['layer_cosines']['27'].get(pair, float('nan'))
    b_val = base['results']['layer_cosines']['27'].get(pair, float('nan'))
    ax.annotate(f'{a_val:.3f}', xy=(27, a_val), fontsize=7, color=COLORS_A[pair],
                xytext=(25.5, a_val + 0.03), ha='right')
    ax.annotate(f'{b_val:.3f}', xy=(27, b_val), fontsize=7, color=COLORS_B[pair],
                xytext=(25.5, b_val - 0.05), ha='right')

fig.tight_layout()
out_path = HERE / "figs" / "fig_base_comparison.pdf"
out_path.parent.mkdir(exist_ok=True)
fig.savefig(out_path, bbox_inches='tight')
print(f"Saved: {out_path}")

# ── Figure 2: Alignment amplification factor by layer ─────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))
delta = a_means - b_means
ax2.bar(layers, delta, color=['#1565C0' if d > 0 else '#C62828' for d in delta], alpha=0.7)
ax2.axhline(0, color='black', lw=0.5)
ax2.axvline(x=10, color='#43A047', lw=1.5, ls=':', alpha=0.7, label='L10 minimum')
ax2.axvline(x=16, color='#8E24AA', lw=1.5, ls='--', alpha=0.7, label='L16 peak')
ax2.set_xlabel("Layer")
ax2.set_ylabel("Alignment effect: aligned − base (cosine similarity)")
ax2.set_title("Where Alignment Training Adds Cross-Category Cohesion\n(Qwen3-0.6B aligned vs Qwen3-0.6B-Base, n=50)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, axis='y')
ax2.set_xlim(-0.5, 27.5)
# Peak annotation
peak_idx = int(np.argmax(delta))
ax2.annotate(f'Max: L{peak_idx}\nΔ={delta[peak_idx]:.3f}',
             xy=(peak_idx, delta[peak_idx]),
             xytext=(peak_idx + 2, delta[peak_idx] - 0.01),
             fontsize=8, color='#1565C0',
             arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1))
fig2.tight_layout()
out_path2 = HERE / "figs" / "fig_alignment_effect.pdf"
fig2.savefig(out_path2, bbox_inches='tight')
print(f"Saved: {out_path2}")

print("\n=== KEY NUMBERS ===")
print(f"L0:  aligned={a_means[0]:.4f}, base={b_means[0]:.4f}, Δ={a_means[0]-b_means[0]:+.4f}")
print(f"L10: aligned={a_means[10]:.4f}, base={b_means[10]:.4f}, Δ={a_means[10]-b_means[10]:+.4f}")
print(f"L16: aligned={a_means[16]:.4f}, base={b_means[16]:.4f}, Δ={a_means[16]-b_means[16]:+.4f}")
print(f"L27: aligned={a_means[27]:.4f}, base={b_means[27]:.4f}, Δ={a_means[27]-b_means[27]:+.4f}")
print(f"L16 secondary peak (aligned L16-L10): {a_means[16]-a_means[10]:+.4f}")
print(f"L16 secondary peak (base   L16-L10): {b_means[16]-b_means[10]:+.4f}")
print(f"Amplification ratio: {(a_means[16]-a_means[10])/(b_means[16]-b_means[10]):.2f}×")
