#!/usr/bin/env python3
"""P19 figure generation — CNA depth hierarchy results."""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).parent
RESULTS_DIR = Path(__file__).parent.parent / "paper18" / "p19_results"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "lines.linewidth": 1.5, "axes.linewidth": 0.8,
})

COLORS = {
    "harmful_vs_benign": "#e63946",   # red — refusal gate
    "ski_vs_skt":        "#457b9d",   # blue — self-id routing
    "ski_vs_benign":     "#2a9d8f",   # teal — self-id vs neutral
}

LABELS = {
    "harmful_vs_benign": "Harmful vs Benign\n(refusal gate)",
    "ski_vs_skt":        "SKI vs SKT\n(self-id routing)",
    "ski_vs_benign":     "SKI vs Benign\n(self-id vs neutral)",
}

def load_results(path):
    with open(path) as f:
        return json.load(f)

# ─── Figure 1: Layer distribution bar charts (3 comparisons, one per panel) ───

def fig1_layer_distributions(data_06b, data_8b=None):
    comparisons = ["harmful_vs_benign", "ski_vs_skt", "ski_vs_benign"]
    n_cols = 2 if data_8b else 1
    fig, axes = plt.subplots(3, n_cols, figsize=(5*n_cols, 8), squeeze=False)

    for row, comp in enumerate(comparisons):
        datasets = [(data_06b, "Qwen3-0.6B (28L)")]
        if data_8b:
            datasets.append((data_8b, "Qwen3-8B (36L)"))

        for col, (data, model_label) in enumerate(datasets):
            ax = axes[row][col]
            d = data[comp]
            n_layers = d["n_layers"]
            layer_frac = d["layer_frac"]

            layers = list(range(n_layers))
            color = COLORS[comp]

            # shade early-mid zone (CDP crystallization zone): normalized 0.15-0.25
            early_s = int(0.15 * n_layers)
            early_e = int(0.25 * n_layers)
            ax.axvspan(early_s - 0.5, early_e + 0.5, alpha=0.12, color="#f4a261",
                       label=f"CDP zone (L{early_s}-L{early_e})")

            bars = ax.bar(layers, layer_frac, color=color, alpha=0.8, width=0.8)

            ax.set_xlabel("Layer index")
            ax.set_ylabel("Fraction of top-k neurons")
            if col == 0:
                ax.set_title(f"{LABELS[comp]}", loc="left", fontweight="bold")
            if row == 0:
                ax.set_title(model_label, loc="center", style="italic",
                             pad=12 if col == 0 else 0)

            # annotations
            centroid_layer = d["centroid"] * n_layers
            ax.axvline(centroid_layer, color="black", lw=1.2, ls="--", alpha=0.7,
                       label=f"Centroid = {d['centroid']:.3f}")
            ax.text(centroid_layer + 0.3, max(layer_frac) * 0.9,
                    f"c={d['centroid']:.3f}", fontsize=7.5, va="top")

            ax.set_xlim(-0.5, n_layers - 0.5)
            ax.set_ylim(0, max(layer_frac) * 1.15 or 0.05)

            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=7.5)

    plt.suptitle("CNA Neuron Depth Profiles", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "fig1_cna_layer_distributions.pdf")
    plt.savefig(OUT / "fig1_cna_layer_distributions.png")
    plt.close()
    print("Saved fig1_cna_layer_distributions")

# ─── Figure 2: Summary bar chart — centroid + late_frac across comparisons ───

def fig2_summary_metrics(data_06b, data_8b=None):
    comparisons = ["harmful_vs_benign", "ski_vs_skt", "ski_vs_benign"]
    short_labels = ["Harmful\nvs Benign", "SKI\nvs SKT", "SKI\nvs Benign"]
    colors = [COLORS[c] for c in comparisons]

    datasets = [("0.6B", data_06b)]
    if data_8b:
        datasets.append(("8B", data_8b))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    for ax_idx, (metric_key, metric_label) in enumerate([
        ("centroid", "Weighted Layer Centroid (normalized)"),
        ("late_frac", "Late-Layer Fraction (depth > 0.90)"),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(comparisons))
        width = 0.35

        for m_idx, (model_label, data) in enumerate(datasets):
            vals = [data[c][metric_key] for c in comparisons]
            offset = (m_idx - (len(datasets)-1)/2) * width
            bars = ax.bar(x + offset, vals, width, color=colors,
                          alpha=0.85 if m_idx == 0 else 0.55,
                          edgecolor="black" if len(datasets) > 1 else "none",
                          linewidth=0.5,
                          label=model_label if len(datasets) > 1 else None)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

        # CDP crystallization zone marker
        if metric_key == "centroid":
            ax.axhline(0.21, color="#f4a261", lw=1.5, ls=":", alpha=0.9,
                       label="CDP L6 crystallization (≈0.21)")
            ax.legend(fontsize=7.5, loc="lower right")

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels)
        ax.set_ylabel(metric_label)
        ax.set_ylim(0, 1.0)
        ax.set_title(metric_label, fontsize=9.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("CNA Depth Metrics: All Comparisons", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig2_cna_summary_metrics.pdf")
    plt.savefig(OUT / "fig2_cna_summary_metrics.png")
    plt.close()
    print("Saved fig2_cna_summary_metrics")

# ─── Figure 3: Two-stage architecture diagram (text-based, matplotlib) ───

def fig3_two_stage_diagram():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9.5, textcolor="white"):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                        facecolor=color, edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight="bold",
                multialignment="center")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

    def annotation(x, y, text, color, ha="left"):
        ax.text(x, y, text, ha=ha, va="center", fontsize=8.5,
                color=color, style="italic",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.12,
                          edgecolor=color, linewidth=0.8))

    # Input
    box(3.5, 8.8, 3, 0.7, "Input tokens", "#555555")
    arrow(5, 8.8, 5, 8.0)

    # Early layers
    box(2.5, 7.0, 5, 0.8, "L0–L5: Feature encoding", "#888888")
    arrow(5, 7.0, 5, 6.2)

    # L6 crystallization
    box(2.5, 5.2, 5, 0.85, "L6 (~21% depth)\nBehavioral DIRECTION crystallizes", "#f4a261", fontsize=8.5, textcolor="black")
    annotation(7.7, 5.62, "← CDP-visible\n(residual stream direction)", "#f4a261")
    annotation(7.7, 5.1,  "← CNA-invisible\n(distributed, no top-neuron)", "#888888", ha="left")
    arrow(5, 5.2, 5, 4.4)

    # Propagation
    box(2.5, 3.5, 5, 0.75, "L7–L24: Direction propagates\n(residual stream maintains signal)", "#aaaaaa", fontsize=8.5, textcolor="black")
    arrow(5, 3.5, 5, 2.7)

    # L25-27 high-attribution readout
    box(2.5, 1.7, 5, 0.9, "L25–27 (93–96% depth)\nCandidate high-attribution neurons", "#e63946", fontsize=8.5)
    annotation(7.7, 2.2, "← CNA-visible\n(individual neurons, high behavioral delta)", "#e63946")
    annotation(7.7, 1.65, "← CDP-redundant\n(direction already fixed)", "#888888", ha="left")
    arrow(5, 1.7, 5, 0.9)

    # Output
    box(3.0, 0.1, 4, 0.7, "Behavioral output\n(refusal / compliance / self-ID)", "#457b9d", fontsize=8.5)

    # Left-side labels
    ax.text(0.2, 5.62, "CDP\nprobes\nhere", ha="center", va="center", fontsize=7.5,
            color="#f4a261", fontweight="bold")
    ax.text(0.2, 2.15, "CNA\nprobes\nhere", ha="center", va="center", fontsize=7.5,
            color="#e63946", fontweight="bold")
    ax.axhline(5.62, xmin=0.02, xmax=0.25, color="#f4a261", lw=1.2, ls="--", alpha=0.5)
    ax.axhline(2.15, xmin=0.02, xmax=0.25, color="#e63946", lw=1.2, ls="--", alpha=0.5)

    ax.set_title("Two-Stage Behavioral Routing Architecture\n"
                 "CDP (direction-level) and CNA (neuron-level) probe different circuit stages",
                 fontsize=10, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(OUT / "fig3_two_stage_architecture.pdf")
    plt.savefig(OUT / "fig3_two_stage_architecture.png")
    plt.close()
    print("Saved fig3_two_stage_architecture")

# ─── Figure 4: Delta mean comparison ───

def fig4_delta_mean(data_06b, data_8b=None):
    comparisons = ["harmful_vs_benign", "ski_vs_skt", "ski_vs_benign"]
    short_labels = ["Harmful\nvs Benign", "SKI\nvs SKT", "SKI\nvs Benign"]
    colors = [COLORS[c] for c in comparisons]

    datasets = [("0.6B", data_06b)]
    if data_8b:
        datasets.append(("8B", data_8b))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    x = np.arange(len(comparisons))
    width = 0.35

    for m_idx, (model_label, data) in enumerate(datasets):
        vals = [data[c].get("delta_mean", float("nan")) for c in comparisons]
        offset = (m_idx - (len(datasets)-1)/2) * width
        bars = ax.bar(x + offset, vals, width,
                      color=colors, alpha=0.85 if m_idx == 0 else 0.55,
                      edgecolor="black" if len(datasets) > 1 else "none",
                      linewidth=0.5,
                      label=model_label if len(datasets) > 1 else None)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylabel("Mean neuron delta |pos − neg|")
    ax.set_title("Top-k Neuron Activation Difference\nby Comparison Type", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if len(datasets) > 1:
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUT / "fig4_delta_mean.pdf")
    plt.savefig(OUT / "fig4_delta_mean.png")
    plt.close()
    print("Saved fig4_delta_mean")

if __name__ == "__main__":
    results_06b = load_results(RESULTS_DIR / "p19_cna_results.json")

    data_8b = None
    path_8b = RESULTS_DIR / "p19_8b_results.json"
    if path_8b.exists():
        data_8b = load_results(path_8b)
        print(f"8B results found — generating scale comparison figures")
    else:
        print(f"8B results not yet available — generating 0.6B-only figures")

    fig1_layer_distributions(results_06b, data_8b)
    fig2_summary_metrics(results_06b, data_8b)
    fig3_two_stage_diagram()
    fig4_delta_mean(results_06b, data_8b)

    print("Done. All figures saved to:", OUT)
