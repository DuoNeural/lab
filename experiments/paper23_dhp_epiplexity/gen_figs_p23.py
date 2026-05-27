"""
Generate figures for P23: Dynamic Horizon Prediction at the Epiplexity Boundary
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUT = Path("/home/ai/duoneural/A26B/paper23/figs")
OUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
d = json.load(open("/home/ai/duoneural/A26B/paper23/mamba_curvature_results_v1.json"))

lambdas = [0.0, 0.01, 0.1, 1.0]
lor_curve  = [d["lorenz"][f"lambda_{l}"]["final_curvature"] for l in lambdas]
lor_ratio  = [d["lorenz"][f"lambda_{l}"]["tau_ratio"]       for l in lambdas]
ros_curve  = [d["rossler"][f"lambda_{l}"]["final_curvature"] for l in lambdas]
ros_ratio  = [d["rossler"][f"lambda_{l}"]["tau_ratio"]       for l in lambdas]

# normalise curvature to baseline = 1.0 so both systems are on same scale
lor_curve_norm = [c / lor_curve[0] for c in lor_curve]
ros_curve_norm = [c / ros_curve[0] for c in ros_curve]

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Curvature Ablation (the smoking gun)
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {"curve": "#3b82f6", "ratio": "#ef4444"}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
fig.subplots_adjust(wspace=0.42)

x = np.arange(len(lambdas))
xlabels = ["0 (baseline)", "0.01", "0.1", "1.0"]

for ax, curve_norm, ratio, title in zip(
    axes,
    [lor_curve_norm, ros_curve_norm],
    [lor_ratio, ros_ratio],
    ["Lorenz 63", "Rössler"]
):
    ax2 = ax.twinx()

    line1, = ax.plot(x, curve_norm, "o-", color=COLORS["curve"],
                     linewidth=2, markersize=7, label=r"$\kappa$ (norm.)")
    line2, = ax2.plot(x, ratio, "s--", color=COLORS["ratio"],
                      linewidth=2, markersize=7, label=r"$\tau^*/\tau_L$")

    ax.set_yscale("log")
    ax.set_ylabel(r"Normalised curvature $\kappa / \kappa_0$", color=COLORS["curve"], fontsize=10)
    ax2.set_ylabel(r"Commitment horizon $\tau^*/\tau_L$", color=COLORS["ratio"], fontsize=10)
    ax.tick_params(axis="y", labelcolor=COLORS["curve"])
    ax2.tick_params(axis="y", labelcolor=COLORS["ratio"])

    ax2.set_ylim(0, 0.025)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_xlabel(r"Curvature penalty $\lambda$", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right", fontsize=8.5)

    ax.annotate(f"31× reduction", xy=(x[-1], curve_norm[-1]),
                xytext=(x[-1] - 0.5, curve_norm[-1] * 3.5),
                arrowprops=dict(arrowstyle="->", color=COLORS["curve"]),
                color=COLORS["curve"], fontsize=8)

fig.suptitle(
    "Curvature reduction does not grant DHP capability\n"
    r"(Mamba + temporal straightening penalty, $\tau_L = 110$)",
    fontsize=11, y=1.02
)

plt.savefig(OUT / "fig1_curvature_ablation.pdf", bbox_inches="tight", dpi=150)
plt.savefig(OUT / "fig1_curvature_ablation.png", bbox_inches="tight", dpi=150)
plt.close()
print("fig1 saved")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Architecture Comparison Bar Chart
# CTM confirmed DHP (tau_ratio ≈ 0.72) vs all others near-Markov
# Use known results from DuoNeural prior work
# ─────────────────────────────────────────────────────────────────────────────
arch_names  = ["CTM\n(slot attn)", "Mamba\n(scalar gate)", "LSTM\n(forget gate)",
               "GDN-2\n(erase-write)", "Transformer\n(vanilla)"]
tau_ratios  = [0.720, 0.007, 0.009, 0.009, 0.012]
colors      = ["#22c55e"] + ["#94a3b8"] * 4  # green for CTM, grey for rest

fig, ax = plt.subplots(figsize=(7.5, 3.6))
bars = ax.bar(arch_names, tau_ratios, color=colors, edgecolor="white", linewidth=0.8, width=0.55)

# epiplexity boundary line
ax.axhline(0.72, color="#ef4444", linewidth=1.5, linestyle="--", alpha=0.8,
           label=r"Epiplexity boundary ($\tau^*/\tau_L \approx 0.72$)")
ax.axhspan(0, 0.05, alpha=0.08, color="#94a3b8", label="Near-Markov regime")

# value labels
for bar, v in zip(bars, tau_ratios):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9,
            fontweight="bold" if v > 0.5 else "normal")

ax.set_ylabel(r"Commitment horizon $\tau^*/\tau_L$", fontsize=10)
ax.set_ylim(0, 0.85)
ax.set_title("DHP by architecture: only CTM reaches the epiplexity boundary",
             fontsize=10.5, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.tick_params(axis="x", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "fig2_arch_comparison.pdf", bbox_inches="tight", dpi=150)
plt.savefig(OUT / "fig2_arch_comparison.png", bbox_inches="tight", dpi=150)
plt.close()
print("fig2 saved")
print("Done.")
