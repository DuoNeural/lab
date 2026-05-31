#!/usr/bin/env python3
"""
P22 figure generation — Directional Evolution of Behavioral Routing
Figures:
  fig1: two-panel — (a) norm profile L0-L27, (b) angle-from-L6 profile L0-L27
  fig2: patch transfer vs source layer + suppression result
"""
import json, math, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent
P19  = ROOT.parent / "paper19"
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

SWEEP_JSON    = P19 / "p19_direction_sweep.json"
PATCHING_JSON = P19 / "p19_patching_v2_results.json"

# ── load data ───────────────────────────────────────────────────────────────
sweep    = json.loads(SWEEP_JSON.read_text())["layers"]
patching = json.loads(PATCHING_JSON.read_text())

layers = sorted(sweep.keys(), key=int)
norms  = [sweep[k]["norm"]        for k in layers]
coss   = [sweep[k]["cos_to_L6"]   for k in layers]
angles = [math.degrees(math.acos(max(-1.0, min(1.0, c)))) for c in coss]
layer_nums = [int(k) for k in layers]

# ── layer sweep patching ─────────────────────────────────────────────────
ls = patching["layer_sweep"]
patch_layers  = sorted(ls.keys(), key=int)
patch_sources = [int(k) for k in patch_layers]
patch_mean    = [ls[k]["mean_cos"] for k in patch_layers]

# suppression result: cos of L25-27 to d_L6 after suppressing d_L6 at L6
suppress_cos_vals = patching["suppress_cos"]
# mean across L25,26,27
supp_mean = np.mean(list(suppress_cos_vals.values()))

# ── style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

PHASE_COLORS = {
    "pre":  "#E8F4FD",   # light blue
    "crys": "#FFF3CD",   # yellow
    "post": "#F3E5F5",   # light purple
}

def shade_phases(ax, x_lo_pre, x_hi_pre, x_lo_crys, x_hi_crys, x_lo_post, x_hi_post):
    ax.axvspan(x_lo_pre,  x_hi_pre,  alpha=0.35, color=PHASE_COLORS["pre"],  zorder=0)
    ax.axvspan(x_lo_crys, x_hi_crys, alpha=0.55, color=PHASE_COLORS["crys"], zorder=0)
    ax.axvspan(x_lo_post, x_hi_post, alpha=0.35, color=PHASE_COLORS["post"], zorder=0)

# ── Figure 1 — Norm + Angle profiles ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
fig.subplots_adjust(wspace=0.38, left=0.08, right=0.97, top=0.88, bottom=0.14)

# shared phase shading bounds
PRE_LO, PRE_HI   = -0.5, 5.5
CRYS_LO, CRYS_HI =  5.5, 6.5
POST_LO, POST_HI =  6.5, 27.5

# Panel (a): norm
ax = axes[0]
shade_phases(ax, PRE_LO, PRE_HI, CRYS_LO, CRYS_HI, POST_LO, POST_HI)
ax.plot(layer_nums, norms, "o-", color="#2C3E50", lw=1.8, ms=4.5, zorder=3)
ax.axvline(6, color="#E67E22", lw=1.5, ls="--", alpha=0.8, zorder=2)
ax.set_xlabel("Layer")
ax.set_ylabel(r"Direction norm $\|\mathbf{d}_k\|$")
ax.set_xlim(-0.5, 27.5)
ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
ax.set_title("(a) Norm profile", loc="left", fontweight="bold")
# annotate L6 and L27
ax.annotate("L6\n(crystallization)", xy=(6, sweep["6"]["norm"]),
            xytext=(9, 30), fontsize=9, color="#E67E22",
            arrowprops=dict(arrowstyle="->", color="#E67E22", lw=1.0))
ax.annotate(f"L27\n163×", xy=(27, sweep["27"]["norm"]),
            xytext=(22, 145), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.0))

# Panel (b): angle from L6
ax = axes[1]
shade_phases(ax, PRE_LO, PRE_HI, CRYS_LO, CRYS_HI, POST_LO, POST_HI)
ax.plot(layer_nums, angles, "s-", color="#8E44AD", lw=1.8, ms=4.5, zorder=3)
ax.axvline(6, color="#E67E22", lw=1.5, ls="--", alpha=0.8, zorder=2)
ax.axhline(80, color="#C0392B", lw=1.0, ls=":", alpha=0.7)
ax.set_xlabel("Layer")
ax.set_ylabel(r"Angle from $\mathbf{d}_{L6}$ (degrees)")
ax.set_xlim(-0.5, 27.5)
ax.set_ylim(-5, 95)
ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
ax.set_title("(b) Angular rotation from L6", loc="left", fontweight="bold")
ax.annotate("80°\n(L27)", xy=(27, angles[-1]),
            xytext=(22, 85), fontsize=9, color="#C0392B",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.0))

# shared legend for phases
pre_patch  = mpatches.Patch(color=PHASE_COLORS["pre"],  alpha=0.7, label="Phase 1: Pre-crystallization (L0–5)")
crys_patch = mpatches.Patch(color=PHASE_COLORS["crys"], alpha=0.9, label="Phase 2: Crystallization (L6)")
post_patch = mpatches.Patch(color=PHASE_COLORS["post"], alpha=0.7, label="Phase 3: Post-crystallization (L7–27)")
fig.legend(handles=[pre_patch, crys_patch, post_patch],
           loc="upper center", ncol=3, bbox_to_anchor=(0.53, 0.99),
           framealpha=0.9, fontsize=9.5)

for ext in ("pdf", "png"):
    fig.savefig(FIGS / f"fig1_direction_evolution.{ext}",
                dpi=200, bbox_inches="tight")
print("fig1 saved")
plt.close(fig)

# ── Figure 2 — Patch transfer + suppression ─────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.0))
fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)

PATCH_COLOR = "#2980B9"
SUPP_COLOR  = "#E74C3C"

x = np.arange(len(patch_sources))
bars = ax.bar(x, patch_mean, color=PATCH_COLOR, alpha=0.82, width=0.6,
              label="Layer sweep: patch source → L25–27 transfer")

# suppression bar as hatched bar at a separate x position
x_supp = len(x)
ax.bar(x_supp, supp_mean, color=SUPP_COLOR, alpha=0.82, width=0.6,
       hatch="//", label="Suppress $\\mathbf{d}_{L6}$ at L6 → L25–27 similarity")

ax.set_xticks(list(x) + [x_supp])
ax.set_xticklabels([f"L{s}" for s in patch_sources] + ["Supp.\nL6"], fontsize=9.5)
ax.set_xlabel("Patch source layer / suppression condition")
ax.set_ylabel("Mean cosine transfer to L25–27")
ax.set_ylim(0, 1.05)
ax.axhline(1.0, color="#555", lw=0.8, ls=":", alpha=0.5)
ax.set_title("Causal Validation: Patch Transfer and Suppression Invariance", fontweight="bold")

# annotate key values
for xi, val in zip(x, patch_mean):
    ax.text(xi, val + 0.018, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)
ax.text(x_supp, supp_mean + 0.018, f"{supp_mean:.3f}", ha="center", va="bottom",
        fontsize=8.5, color=SUPP_COLOR)

ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)

for ext in ("pdf", "png"):
    fig.savefig(FIGS / f"fig2_patch_transfer.{ext}",
                dpi=200, bbox_inches="tight")
print("fig2 saved")
plt.close(fig)

# ── Figure 3 — Pairwise cosine heatmap + consecutive rotation profile ────────
vectors_json = ROOT / "p22_direction_vectors.json"
if vectors_json.exists():
    vdata = json.loads(vectors_json.read_text())
    N = vdata["n_layers"]
    pcos = vdata["pairwise_cos"]
    cos_matrix = np.array([[float(pcos[f"{i}_{j}"]) for j in range(N)] for i in range(N)])
    consec_angles = [item["angle_deg"] for item in vdata["consecutive_angles"]]
    consec_x = [item["from"] + 0.5 for item in vdata["consecutive_angles"]]  # midpoint

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.97, top=0.88, bottom=0.12)

    # Heatmap
    ax = axes[0]
    im = ax.imshow(cos_matrix, vmin=-0.2, vmax=1.0, cmap="RdYlGn",
                   origin="lower", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Cosine similarity")
    ax.set_xlabel("Layer $k$")
    ax.set_ylabel("Layer $k'$")
    ticks = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_title("(a) Pairwise direction cosines $\\cos(\\mathbf{d}_k, \\mathbf{d}_{k'})$",
                 loc="left", fontweight="bold")
    # Mark L6
    ax.axhline(6, color="orange", lw=1.2, ls="--", alpha=0.8)
    ax.axvline(6, color="orange", lw=1.2, ls="--", alpha=0.8)
    ax.text(6.3, 0.5, "L6", color="orange", fontsize=9, va="bottom")

    # Consecutive rotation
    ax = axes[1]
    shade_phases(ax, -0.5, 5.5, 5.5, 6.5, 6.5, 27.5)
    ax.plot(consec_x, consec_angles, "D-", color="#E67E22", lw=1.8, ms=4.5, zorder=3)
    ax.set_xlabel("Layer transition $k \\to k{+}1$")
    ax.set_ylabel("Rotation angle (degrees)")
    ax.set_xlim(-0.5, 27.5)
    ax.set_ylim(0, 65)
    xticks = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    ax.set_xticks(xticks)
    ax.set_title("(b) Per-block rotation angle", loc="left", fontweight="bold")
    # Phase annotations
    ax.text(2.5, 57, f"Pre-cryst\\n{sum(consec_angles[:6])/6:.0f}°/step",
            ha="center", fontsize=8.5, color="#2980B9")
    ax.text(9.5, 57, f"Early post\\n{sum(consec_angles[6:12])/6:.0f}°/step",
            ha="center", fontsize=8.5, color="#8E44AD")
    ax.text(20, 57, f"Late post\\n{sum(consec_angles[12:])/16:.0f}°/step",
            ha="center", fontsize=8.5, color="#8E44AD")

    pre_patch  = mpatches.Patch(color=PHASE_COLORS["pre"],  alpha=0.7, label="Pre-cryst (L0--5)")
    crys_patch = mpatches.Patch(color=PHASE_COLORS["crys"], alpha=0.9, label="Cryst (L6)")
    post_patch = mpatches.Patch(color=PHASE_COLORS["post"], alpha=0.7, label="Post-cryst (L7--27)")
    fig.legend(handles=[pre_patch, crys_patch, post_patch],
               loc="upper center", ncol=3, bbox_to_anchor=(0.73, 0.99),
               framealpha=0.9, fontsize=9.5)

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig3_pairwise_heatmap.{ext}", dpi=200, bbox_inches="tight")
    print("fig3 saved")
    plt.close(fig)
else:
    print("p22_direction_vectors.json not found, skipping fig3")

# ── Figure 4 — Topic-specific direction profiles ──────────────────────────
topic_json = ROOT / "p22_topic_sweep_results.json"
if topic_json.exists():
    tdata = json.loads(topic_json.read_text())
    cats = tdata["categories"]
    tresults = tdata["results"]
    layer_nums_t = sorted([int(k) for k in tresults[cats[0]].keys()])

    CAT_COLORS = {
        "weapons":    "#E74C3C",
        "drugs":      "#E67E22",
        "cybercrime": "#3498DB",
        "hate":       "#8E44AD",
    }
    CAT_MARKERS = {"weapons": "o", "drugs": "s", "cybercrime": "D", "hate": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.subplots_adjust(wspace=0.38, left=0.08, right=0.97, top=0.88, bottom=0.14)

    # Panel (a): angle profiles
    ax = axes[0]
    shade_phases(ax, -0.5, 5.5, 5.5, 6.5, 6.5, 27.5)
    for cat in cats:
        angles_t = [tresults[cat][str(L)]["angle_deg"] for L in layer_nums_t]
        ax.plot(layer_nums_t, angles_t, marker=CAT_MARKERS[cat],
                color=CAT_COLORS[cat], lw=1.6, ms=4.5, label=cat.capitalize(), zorder=3)
    ax.axvline(6, color="#E67E22", lw=1.5, ls="--", alpha=0.8, zorder=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"Angle from own $\mathbf{d}_{L6}$ (degrees)")
    ax.set_xlim(-0.5, 27.5); ax.set_ylim(-5, 95)
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    ax.set_title("(a) Rotation profile by topic category", loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9.5)

    # Panel (b): L6 norm by topic
    ax = axes[1]
    cat_norms_L6 = [tresults[c]["6"]["norm"] for c in cats]
    cat_norms_L27 = [tresults[c]["27"]["norm"] for c in cats]
    x_t = np.arange(len(cats))
    bars_L6  = ax.bar(x_t - 0.22, cat_norms_L6,  width=0.4, color=[CAT_COLORS[c] for c in cats],
                      alpha=0.75, label="L6 (crystallization)")
    bars_L27 = ax.bar(x_t + 0.22, cat_norms_L27, width=0.4, color=[CAT_COLORS[c] for c in cats],
                      alpha=0.38, hatch="//", label="L27 (readout)")
    ax.set_xticks(x_t)
    ax.set_xticklabels([c.capitalize() for c in cats], fontsize=9.5)
    ax.set_ylabel(r"Direction norm $\|\mathbf{d}_k\|$")
    ax.set_title("(b) Direction norms: L6 vs L27", loc="left", fontweight="bold")
    ax.legend(fontsize=9.5)
    for xi, val in zip(x_t - 0.22, cat_norms_L6):
        ax.text(xi, val + 2, f"{val:.1f}", ha="center", va="bottom", fontsize=7.5)
    for xi, val in zip(x_t + 0.22, cat_norms_L27):
        ax.text(xi, val + 2, f"{val:.0f}", ha="center", va="bottom", fontsize=7.5)

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig4_topic_profiles.{ext}", dpi=200, bbox_inches="tight")
    print("fig4 saved")
    plt.close(fig)
else:
    print("p22_topic_sweep_results.json not found, skipping fig4")

print("All figures done.")
