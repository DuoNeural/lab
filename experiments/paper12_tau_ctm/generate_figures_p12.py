#!/usr/bin/env python3
"""
Figure generator for Paper 12: Temporal Horizon Emergence During Training
Archon — DuoNeural — 2026-05-21

Generates all 5 figures from trajectory JSON files in tau_trajectory/.
Output: paper12/figures/*.pdf + *.png
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

BASE   = "/home/ai/duoneural/A26B"
TAU    = os.path.join(BASE, "tau_trajectory")
OUT    = os.path.join(BASE, "paper12", "figures")
os.makedirs(OUT, exist_ok=True)

# --- color scheme ---
COLORS = {
    "D3_cos":  "#dc322f",  # red — unreliable
    "D3_flat": "#cb4b16",  # dark orange — control
    "D4":      "#b58900",  # yellow — unreliable
    "D5":      "#d33682",  # magenta — unreliable
    "D7":      "#2aa198",  # teal — reliable
    "D10":     "#268bd2",  # blue — reliable
}
DHP_LINE = "#6c71c4"  # purple for 0.72 reference


def load(fname):
    return json.load(open(os.path.join(TAU, fname)))


def traj_array(d, key):
    return np.array([pt[key] for pt in d["trajectory"]])


def steps(d):
    return np.array([pt["step"] for pt in d["trajectory"]])


# =====================================================================
# Figure 1: τ*(t)/τ_L trajectory — all five D, one panel each
# =====================================================================
def fig1_trajectories():
    runs = [
        ("D3_cos",  "results_lorenz63_150k_cosine.json", "D=3 (Lorenz-63, cosine LR)", 22, False),
        ("D4",      "results_lorenz96d4_v3.json",        "D=4 (Lorenz-96)",              35, False),
        ("D5",      "results_lorenz96_v2.json",          "D=5 (Lorenz-96)",              30, False),
        ("D7",      "results_lorenz96d7_v3.json",        "D=7 (Lorenz-96)",              25, True),
        ("D10",     "results_lorenz96d10_v2.json",       "D=10 (Lorenz-96)",             20, True),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=False)
    fig.suptitle(r"$\tau^*(t)/\tau_L$ Gate Ratio Trajectories — All Five Dimensionalities",
                 fontsize=11, y=1.01)

    for ax, (key, fname, label, tau_L, reliable) in zip(axes, runs):
        d = load(fname)
        st = steps(d)
        ratio = traj_array(d, "ratio_gate")

        color = COLORS[key]
        ax.plot(st / 1000, ratio, color=color, lw=1.5, alpha=0.85)
        ax.axhline(0.72, color=DHP_LINE, ls="--", lw=1.2, label="DHP target (0.72)")

        # shade unreliable region for D4 (zero-loss artifact after step 500)
        if key == "D4":
            ax.axvline(0.5, color="#dc322f", ls=":", lw=1, alpha=0.7)
            ax.text(1.0, 0.95, "zero-loss\nfreeze", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.5, color="#dc322f")

        ax.set_title(label, fontsize=8.5, fontweight="bold" if reliable else "normal",
                     color=COLORS[key])
        ax.set_xlabel("Training step (k)", fontsize=8)
        ax.set_ylabel(r"$\rho_{\rm gate} = \tau^*/\tau_L$", fontsize=8)
        ax.set_ylim(0.0, 2.0)
        ax.grid(True, alpha=0.3)

        if reliable:
            ax.set_facecolor("#f0f9f8")  # subtle teal tint for reliable

        ax.legend(fontsize=6.5, loc="upper right")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_trajectories.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close()
    print("  fig1_trajectories done")


# =====================================================================
# Figure 2: D-curve scatter — final ratio vs D
# =====================================================================
def fig2_dcurve():
    data = [
        (3,  1.223, False, "D=3 (non-conv.)"),
        (4,  0.851, False, "D=4 (zero-loss)"),
        (5,  0.841, False, "D=5 (mild)"),
        (7,  0.361, True,  "D=7 ✓"),
        (10, 0.523, True,  "D=10 ✓"),
    ]

    fig, ax = plt.subplots(figsize=(5, 4))

    for D, ratio, reliable, label in data:
        color = COLORS[f"D{D}" if D != 3 else "D3_cos"]
        marker = "o" if reliable else "x"
        ms = 9 if reliable else 8
        ax.scatter(D, ratio, color=color, marker=marker, s=ms**2,
                   zorder=5, label=label)
        ax.annotate(label, (D, ratio), textcoords="offset points",
                    xytext=(6, 3), fontsize=7.5, color=color)

    ax.axhline(0.72, color=DHP_LINE, ls="--", lw=1.5, label="DHP target (0.72)")
    ax.fill_between([5.0, 7.0], 0.0, 2.0, alpha=0.08, color="#6c71c4",
                    label="Threshold zone (D=5→7)")

    ax.set_xlabel("Output dimensionality $D$", fontsize=10)
    ax.set_ylabel(r"Final gate ratio $\rho_{\rm final}$", fontsize=10)
    ax.set_title("D-Curve: Gate Specialization vs Dimensionality", fontsize=10)
    ax.set_xlim(1.5, 11.5)
    ax.set_ylim(0.1, 1.8)
    ax.set_xticks([3, 4, 5, 7, 10])
    ax.grid(True, alpha=0.3)

    # Legend: confound labels
    unreliable_patch = mpatches.Patch(color="#dc322f", label="Unreliable (confounded)")
    reliable_patch   = mpatches.Patch(color="#2aa198", label="Reliable (D≥7)")
    dhp_line = plt.Line2D([0], [0], color=DHP_LINE, ls="--", label="DHP target 0.72")
    ax.legend(handles=[reliable_patch, unreliable_patch, dhp_line], fontsize=8, loc="upper right")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_dcurve.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close()
    print("  fig2_dcurve done")


# =====================================================================
# Figure 3: D=10 trajectory zoom — monotonic descent through DHP
# =====================================================================
def fig3_d10_zoom():
    d = load("results_lorenz96d10_v2.json")
    traj = d["trajectory"]
    # steps 0–10000 for the interesting region
    zoom = [pt for pt in traj if pt["step"] <= 10000]
    st    = np.array([pt["step"] for pt in zoom])
    ratio = np.array([pt["ratio_gate"] for pt in zoom])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(st / 1000, ratio, color=COLORS["D10"], lw=2.0)
    ax.axhline(0.72, color=DHP_LINE, ls="--", lw=1.5, label="DHP target (0.72)")

    # mark where ratio crosses 0.72
    cross_idx = np.where(ratio <= 0.72)[0]
    if len(cross_idx):
        cx, cy = st[cross_idx[0]] / 1000, ratio[cross_idx[0]]
        ax.annotate(f"ρ≈0.72 at step {int(st[cross_idx[0]]//1000)}k",
                    (cx, cy), xytext=(cx + 0.5, cy + 0.08),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                    fontsize=8.5, color=DHP_LINE)
        ax.scatter([cx], [cy], s=60, color=DHP_LINE, zorder=6)

    ax.set_xlabel("Training step (k)", fontsize=10)
    ax.set_ylabel(r"$\rho_{\rm gate} = \tau^*/\tau_L$", fontsize=10)
    ax.set_title("D=10: Monotonic Descent Through DHP Target\n(steps 0–10k)", fontsize=10)
    ax.set_ylim(0.4, 1.8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig3_d10_zoom.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close()
    print("  fig3_d10_zoom done")


# =====================================================================
# Figure 4: GHL horizon MSE — Lyapunov cliff (D=7 and D=10)
# =====================================================================
def fig4_lyapunov_cliff():
    horizons = [1, 2, 4, 8, 16, 32]
    mse_d7  = [0.00513, 0.00456, 0.00468, 0.00840, 0.03087, 0.16809]
    mse_d10 = [0.01116, 0.01008, 0.01050, 0.01878, 0.07575, 0.29890]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(horizons, mse_d7,  "o-", color=COLORS["D7"],  lw=2, ms=6, label="D=7")
    ax.plot(horizons, mse_d10, "s-", color=COLORS["D10"], lw=2, ms=6, label="D=10")

    # mark the cliff (h=8→16)
    ax.axvspan(8, 16, alpha=0.12, color="#dc322f", label="Lyapunov cliff (h=8→16)")
    ax.axvline(8, color="#6c71c4", ls=":", lw=1.5, label=r"$\tau^*_{\rm horiz}=8$")

    ax.set_yscale("log")
    ax.set_xlabel("Prediction horizon $h$ (steps)", fontsize=10)
    ax.set_ylabel("MSE (log scale)", fontsize=10)
    ax.set_title("GHL Horizon MSE — Lyapunov Cliff at h=8→16", fontsize=10)
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig4_lyapunov_cliff.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close()
    print("  fig4_lyapunov_cliff done")


# =====================================================================
# Figure 5: D=3 cosine vs flat-LR comparison
# =====================================================================
def fig5_d3_comparison():
    d_cos  = load("results_lorenz63_150k_cosine.json")
    d_flat = load("results_lorenz63_flatlr.json")

    st_cos   = steps(d_cos)
    ratio_cos = traj_array(d_cos, "ratio_gate")

    st_flat   = steps(d_flat)
    ratio_flat = traj_array(d_flat, "ratio_gate")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fig.suptitle("D=3 Lorenz-63: Cosine LR vs Flat LR\n"
                 "(Flat-LR replication rules out LR decay as cause of non-convergence)",
                 fontsize=10)

    ax1.plot(st_cos / 1000, ratio_cos, color=COLORS["D3_cos"], lw=1.5, label="Cosine LR (150k)")
    ax1.axhline(0.72, color=DHP_LINE, ls="--", lw=1.2, label="DHP target")
    ax1.set_xlabel("Training step (k)"); ax1.set_ylabel(r"$\rho_{\rm gate}$")
    ax1.set_title("Cosine LR"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.8, 1.8)

    ax2.plot(st_flat / 1000, ratio_flat, color=COLORS["D3_flat"], lw=1.5, label="Flat LR (100k)")
    ax2.axhline(0.72, color=DHP_LINE, ls="--", lw=1.2, label="DHP target")
    ax2.set_xlabel("Training step (k)")
    ax2.set_title("Flat LR (constant 1e-4)"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Annotate final values
    for ax, st, ratio, label in [(ax1, st_cos, ratio_cos, "Cosine"), (ax2, st_flat, ratio_flat, "Flat")]:
        final = np.mean(ratio[-10:])
        std   = np.std(ratio[-10:])
        ax.axhline(final, color="gray", ls=":", lw=1, alpha=0.7)
        ax.text(0.97, 0.07, f"Final: {final:.3f}±{std:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig5_d3_comparison.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close()
    print("  fig5_d3_comparison done")


if __name__ == "__main__":
    print("Generating Paper 12 figures...")
    fig1_trajectories()
    fig2_dcurve()
    fig3_d10_zoom()
    fig4_lyapunov_cliff()
    fig5_d3_comparison()
    print(f"\nAll figures written to {OUT}/")
