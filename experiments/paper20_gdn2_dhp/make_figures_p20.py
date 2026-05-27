"""
P20 Figure Generation — GDN-2 × DHP Gate Decoupling
=====================================================
Generates 4 publication figures from v2 + v3 results JSON files.

Inputs:
  /workspace/gdn2_dhp_results_v2.json  (pred_horizon=1, negative control)
  /workspace/gdn2_dhp_results_v3.json  (multi-horizon, proper DHP test)

Outputs (saved locally and as PDF):
  fig1_decay_curves.pdf      — normalized δS decay, v2 vs v3 per system
  fig2_tau_summary.pdf       — τ*/τ_L summary with DHP target line
  fig3_per_layer_tau.pdf     — per-layer τ*/τ_L profile (v3)
  fig4_gate_attribution.pdf  — gate statistics b̄/w̄/ᾱ + erase timescale

Run from paper20/ directory with:
  python make_figures_p20.py [--v2 path] [--v3 path]
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import argparse
import sys

# ─── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DHP_TARGET = 0.72
COLOR_V2  = "#888888"   # grey — baseline / negative control
COLOR_V3  = "#2166AC"   # blue — multi-horizon / DHP
COLOR_ERASE = "#D73027" # red — theoretical erase timescale
COLOR_LINE  = "#1A9641" # green — DHP 0.72 target
SYSTEM_COLORS = {"lorenz": "#2166AC", "rossler": "#D73027"}
SYSTEM_LABELS = {"lorenz": "Lorenz-63", "rossler": "Rössler"}


def load_results(path):
    if path is None or not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def find_1e_crossing(traj, T_measure=None):
    """Return 1/e crossing index (float), or None."""
    T = len(traj) if T_measure is None else T_measure
    for t in range(1, T):
        if traj[t] <= 1 / np.e:
            frac = (traj[t-1] - 1/np.e) / (traj[t-1] - traj[t] + 1e-15)
            return (t - 1) + frac
    return None


# ─── Figure 1: Decay curves ───────────────────────────────────────────────────

def fig1_decay_curves(r2, r3, out="fig1_decay_curves.pdf"):
    systems = ["lorenz", "rossler"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    fig.suptitle("State Divergence Decay Under Erase Operator", fontsize=12, y=1.01)

    for ax, sys_name in zip(axes, systems):
        ax.set_title(SYSTEM_LABELS[sys_name])
        ax.set_xlabel("Time step t")
        ax.set_ylabel("Normalised $\\|\\Delta S(t)\\|$ / $\\|\\Delta S(0)\\|$")
        ax.axhline(1/np.e, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
                   label="$e^{-1}$ threshold")

        for r, label, color, ls in [
            (r2, "v2 (single-horizon)", COLOR_V2, "--"),
            (r3, "v3 (multi-horizon)", COLOR_V3, "-"),
        ]:
            if r is None or sys_name not in r:
                continue
            d = r[sys_name]
            norm = d["normalized_decay_trajectory"]
            T = len(norm)
            tau_star = d["tau_star_steps"]
            tau_ratio = d["tau_ratio"]

            ax.plot(norm, color=color, linestyle=ls, linewidth=1.5,
                    label=f"{label}  τ*/τ_L={tau_ratio:.2f}")
            ax.axvline(tau_star, color=color, linestyle=":", linewidth=0.8, alpha=0.7)

        # Theoretical DHP prediction line
        ax.axvline(DHP_TARGET * r3[sys_name]["tau_L_steps"] if r3 and sys_name in r3 else 0,
                   color=COLOR_LINE, linestyle="-.", linewidth=1.2, alpha=0.8,
                   label=f"DHP 0.72×τ_L")

        ax.set_xlim(0, 300)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(loc="upper right", framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close()


# ─── Figure 2: τ*/τ_L summary ─────────────────────────────────────────────────

def fig2_tau_summary(r2, r3, out="fig2_tau_summary.pdf"):
    systems = ["lorenz", "rossler"]
    sys_labels = [SYSTEM_LABELS[s] for s in systems]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
    fig.suptitle("DHP Commitment Ratio τ*/τ_L — Baseline vs Multi-Horizon", fontsize=12)

    for ax, sys_name in zip(axes, systems):
        ax.set_title(SYSTEM_LABELS[sys_name])
        ax.set_ylabel("τ*/τ_L")
        ax.axhline(DHP_TARGET, color=COLOR_LINE, linestyle="-", linewidth=1.5,
                   label="DHP target (0.72)", alpha=0.9)
        ax.axhspan(0.68, 0.76, alpha=0.12, color=COLOR_LINE)

        metrics = []
        labels_bar = []
        colors_bar = []

        if r2 and sys_name in r2:
            d = r2[sys_name]
            metrics.append(d["tau_ratio"])
            labels_bar.append("v2\n(h=1)")
            colors_bar.append(COLOR_V2)

        if r3 and sys_name in r3:
            d = r3[sys_name]
            metrics.append(d["tau_ratio"])
            labels_bar.append("v3\n(multi-h)")
            colors_bar.append(COLOR_V3)

            # Also plot erase theoretical
            metrics.append(d["tau_erase_ratio"])
            labels_bar.append("τ*\nerase")
            colors_bar.append(COLOR_ERASE)

        if metrics:
            bars = ax.bar(labels_bar, metrics, color=colors_bar, width=0.5, alpha=0.85,
                          edgecolor="black", linewidth=0.6)
            for bar, val in zip(bars, metrics):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close()


# ─── Figure 3: Per-layer τ*/τ_L profile (v3) ─────────────────────────────────

def fig3_per_layer_tau(r3, out="fig3_per_layer_tau.pdf"):
    if r3 is None:
        print(f"  [SKIP] {out} — v3 results not available")
        return

    systems = ["lorenz", "rossler"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    fig.suptitle("Per-Layer DHP Commitment Ratio τ*/τ_L (v3 Multi-Horizon)", fontsize=12)

    for ax, sys_name in zip(axes, systems):
        if sys_name not in r3:
            continue
        d = r3[sys_name]
        n_layers = len(d["per_layer_tau_ratio"])
        layers = list(range(n_layers))
        ratios = d["per_layer_tau_ratio"]

        ax.plot(layers, ratios, "o-", color=SYSTEM_COLORS[sys_name],
                linewidth=2, markersize=7, label=SYSTEM_LABELS[sys_name])
        ax.axhline(DHP_TARGET, color=COLOR_LINE, linestyle="--", linewidth=1.5,
                   label=f"DHP 0.72", alpha=0.9)
        ax.axhspan(0.68, 0.76, alpha=0.12, color=COLOR_LINE)

        # Also show erase theory per layer
        b_layer = d.get("b_by_layer", [])
        alpha_layer = d.get("alpha_by_layer", [])
        if b_layer and alpha_layer:
            erase_rates = [a * (1 - b) for a, b in zip(alpha_layer, b_layer)]
            tau_L = d["tau_L_steps"]
            erase_ratios = [-1.0 / (np.log(r + 1e-10) * tau_L) if r < 1.0 else 1.0
                            for r in erase_rates]
            ax.plot(layers, erase_ratios, "s--", color=COLOR_ERASE, linewidth=1.2,
                    markersize=5, alpha=0.8, label="τ*_erase (theory)")

        ax.set_title(SYSTEM_LABELS[sys_name])
        ax.set_xlabel("Layer index")
        ax.set_ylabel("τ*/τ_L")
        ax.set_xticks(layers)
        ax.set_ylim(0, 1.2)
        ax.legend(loc="lower right", framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close()


# ─── Figure 4: Gate attribution ───────────────────────────────────────────────

def fig4_gate_attribution(r3, out="fig4_gate_attribution.pdf"):
    if r3 is None:
        print(f"  [SKIP] {out} — v3 results not available")
        return

    systems = ["lorenz", "rossler"]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.5))
    fig.suptitle("GDN-2 Gate Statistics vs Layer (v3 Multi-Horizon)", fontsize=12)

    for col, sys_name in enumerate(systems):
        if sys_name not in r3:
            continue
        d = r3[sys_name]
        n_layers = len(d.get("b_by_layer", []))
        if n_layers == 0:
            continue
        layers = list(range(n_layers))

        # Top: gate means per layer
        ax_top = axes[0, col]
        ax_top.set_title(SYSTEM_LABELS[sys_name])
        ax_top.plot(layers, d["b_by_layer"], "o-", color="#D73027", linewidth=2,
                    markersize=6, label="$\\bar{b}$ (erase)")
        ax_top.plot(layers, d["w_by_layer"], "s-", color="#2166AC", linewidth=2,
                    markersize=6, label="$\\bar{w}$ (write)")
        ax_top.plot(layers, d["alpha_by_layer"], "^-", color="#4DAC26", linewidth=2,
                    markersize=6, label="$\\bar{\\alpha}$ (decay)")
        ax_top.set_xlabel("Layer index")
        ax_top.set_ylabel("Gate mean")
        ax_top.set_ylim(0, 1.05)
        ax_top.set_xticks(layers)
        ax_top.legend(loc="lower right", framealpha=0.85)

        # Bottom: effective erase retention per layer
        ax_bot = axes[1, col]
        b_layer = d["b_by_layer"]
        alpha_layer = d["alpha_by_layer"]
        retention = [a * (1 - b) for a, b in zip(alpha_layer, b_layer)]
        tau_L = d["tau_L_steps"]
        erase_tau = [-1.0 / (np.log(r + 1e-10)) if r < 1.0 else tau_L
                     for r in retention]
        erase_ratio = [t / tau_L for t in erase_tau]

        ax_bot.bar(layers, erase_ratio, color=COLOR_ERASE, alpha=0.75,
                   edgecolor="black", linewidth=0.6, label="τ*_erase/τ_L")
        ax_bot.axhline(DHP_TARGET, color=COLOR_LINE, linestyle="--", linewidth=1.5,
                       label="DHP 0.72", alpha=0.9)
        ax_bot.set_xlabel("Layer index")
        ax_bot.set_ylabel("Erase τ*/τ_L")
        ax_bot.set_ylim(0, 1.2)
        ax_bot.set_xticks(layers)
        ax_bot.legend(loc="upper right", framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", default="/workspace/gdn2_dhp_results_v2.json")
    parser.add_argument("--v3", default="/workspace/gdn2_dhp_results_v3.json")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    print(f"Loading v2: {args.v2}")
    r2 = load_results(args.v2)
    print(f"  {'loaded' if r2 else 'NOT FOUND — will skip v2 in plots'}")

    print(f"Loading v3: {args.v3}")
    r3 = load_results(args.v3)
    print(f"  {'loaded' if r3 else 'NOT FOUND — will skip v3 in plots'}")

    if r2 is None and r3 is None:
        print("ERROR: No results files found. Run experiments first.")
        sys.exit(1)

    # Print summary table
    print("\n── Summary ──────────────────────────────────────────────────")
    header = f"{'System':<10} {'Version':<8} {'τ* (steps)':<14} {'τ_L (steps)':<13} {'τ*/τ_L':<9} {'τ*_erase/τ_L':<14}"
    print(header)
    print("-" * 70)
    for sys_name in ["lorenz", "rossler"]:
        for r, vname in [(r2, "v2"), (r3, "v3")]:
            if r and sys_name in r:
                d = r[sys_name]
                er = d.get("tau_erase_ratio", float("nan"))
                print(f"{sys_name:<10} {vname:<8} {d['tau_star_steps']:<14.1f} "
                      f"{d['tau_L_steps']:<13.1f} {d['tau_ratio']:<9.3f} {er:<14.3f}")
    print()

    print("Generating figures...")
    fig1_decay_curves(r2, r3, out=str(outdir / "fig1_decay_curves.pdf"))
    fig2_tau_summary(r2, r3, out=str(outdir / "fig2_tau_summary.pdf"))
    fig3_per_layer_tau(r3, out=str(outdir / "fig3_per_layer_tau.pdf"))
    fig4_gate_attribution(r3, out=str(outdir / "fig4_gate_attribution.pdf"))
    print("\nAll figures done.")


if __name__ == "__main__":
    main()
