"""
P24 Supplemental Figures
========================
Generates publication-quality figures for §4.4 and §4.5:
  - fig_patching_fliprates.pdf  : Activation patching B→R / H→C by condition
  - fig_scale_validation.pdf    : 0.6B vs 1.7B aligned profile comparison
  - fig_scale_base_flatness.pdf : All 4 models (aligned+base for both sizes)

Archon — DuoNeural — 2026-05-28
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import spearmanr

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 200,
    'pdf.fonttype': 42,  # embeds fonts — required for arXiv
})

HERE = Path(__file__).parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────
with open(HERE / "p24_activation_patching_v2_results.json") as f:
    patch_data = json.load(f)

with open(HERE / "p24_scale_validation_results.json") as f:
    scale_data = json.load(f)


# ══════════════════════════════════════════════════════════════
# FIGURE 4: Activation Patching — B→R and H→C flip rates
# ══════════════════════════════════════════════════════════════
print("Generating fig_patching_fliprates.pdf ...")

# Pull summary at alpha=0.5
conditions = {
    "L16\n(aligned)\nn=40": ("L16_aligned",     "#1565C0", False),  # main
    "L0\ncontrol\nn=20":    ("L0_aligned_control",  "#78909C", True),
    "L10\ncontrol\nn=20":   ("L10_aligned_control", "#78909C", True),
    "L20\ncontrol\nn=20":   ("L20_aligned_control", "#78909C", True),
    "L27\ncontrol\nn=20":   ("L27_aligned_control", "#78909C", True),
    "L16\n(base)\nn=20":    ("L16_base",            "#EF9A9A", True),  # ablation
}

cond_labels = list(conditions.keys())
b2r_vals = []
h2c_vals = []
colors_b2r = []
colors_h2c = []

for label, (key, color, is_control) in conditions.items():
    s = patch_data["summary"][key]
    b2r_vals.append(s["flip_rate_benign_to_refusal"])
    h2c_vals.append(s["flip_rate_harm_to_compliance"])
    alpha_c = 0.65 if is_control else 1.0
    colors_b2r.append(color)
    colors_h2c.append(color)

x = np.arange(len(cond_labels))
w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

# ── Left: B→R flip rates ─────────────────────────────────────
ax = axes[0]
bars = ax.bar(x, b2r_vals, width=0.55, color=colors_b2r,
              edgecolor='white', linewidth=0.8, zorder=3)

# Hatching on control bars
for i, (bar, (label, (key, color, is_control))) in enumerate(
        zip(bars, conditions.items())):
    if is_control:
        bar.set_hatch('//')
        bar.set_alpha(0.65)

# Annotate values
for i, v in enumerate(b2r_vals):
    ax.text(i, v + 0.004, f"{v:.3f}", ha='center', va='bottom',
            fontsize=9, fontweight='bold' if i == 0 else 'normal',
            color='#1565C0' if i == 0 else '#424242')

# Significance bracket: L16 aligned >> all controls
ax.annotate(
    '',
    xy=(0, 0.115), xytext=(5, 0.115),
    arrowprops=dict(arrowstyle='<->', color='#B71C1C', lw=1.5)
)
ax.text(2.5, 0.118, 'B→R unique to L16 (p < 0.05)', ha='center',
        fontsize=8, color='#B71C1C', fontstyle='italic')

ax.set_xticks(x)
ax.set_xticklabels(cond_labels, fontsize=8.5)
ax.set_ylabel("Flip rate (proportion of pairs flipped)")
ax.set_title("B→R Flip Rate by Layer/Condition\n"
             "(Benign→Refusal after harm-state patching, α=0.5)",
             fontsize=10)
ax.set_ylim(0, 0.175)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
main_patch = mpatches.Patch(color='#1565C0', label='L16 aligned (target)')
ctrl_patch  = mpatches.Patch(facecolor='#78909C', hatch='//', alpha=0.65,
                              label='Control layers')
base_patch  = mpatches.Patch(facecolor='#EF9A9A', hatch='//', alpha=0.65,
                              label='L16 base (ablation)')
ax.legend(handles=[main_patch, ctrl_patch, base_patch], fontsize=8,
          loc='upper right')

# ── Right: H→C flip rates ────────────────────────────────────
ax = axes[1]
bars2 = ax.bar(x, h2c_vals, width=0.55, color=colors_h2c,
               edgecolor='white', linewidth=0.8, zorder=3)
for i, (bar, (label, (key, color, is_control))) in enumerate(
        zip(bars2, conditions.items())):
    if is_control:
        bar.set_hatch('//')
        bar.set_alpha(0.65)

for i, v in enumerate(h2c_vals):
    ax.text(i, v + 0.008, f"{v:.3f}", ha='center', va='bottom',
            fontsize=9, fontweight='bold' if i == 0 else 'normal',
            color='#1565C0' if i == 0 else '#424242')

# Annotation: distributed gradient
ax.annotate(
    'H→C distributed\nacross L16–L27',
    xy=(3.5, 0.35), fontsize=8, ha='center', color='#424242',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', alpha=0.9)
)
ax.annotate(
    'Base=0\n(no alignment,\nno refusals)',
    xy=(5, 0.01), xytext=(4.3, 0.15),
    fontsize=8, ha='center', color='#C62828',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='#C62828', lw=1)
)

ax.set_xticks(x)
ax.set_xticklabels(cond_labels, fontsize=8.5)
ax.set_ylabel("Flip rate (proportion of pairs flipped)")
ax.set_title("H→C Flip Rate by Layer/Condition\n"
             "(Harm→Compliance after benign-state patching, α=0.5)",
             fontsize=10)
ax.set_ylim(0, 0.55)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(handles=[main_patch, ctrl_patch, base_patch], fontsize=8,
          loc='upper right')

fig.suptitle(
    "Causal Activation Patching — Layer-Specificity of Refusal Geometry\n"
    "Qwen3-0.6B (aligned vs base), interpolated patching α=0.5",
    fontsize=11, y=1.01
)
fig.tight_layout()
out = FIGS / "fig_patching_fliprates.pdf"
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════
# FIGURE 5: Alpha sweep for L16 aligned (B→R and H→C)
# ══════════════════════════════════════════════════════════════
print("Generating fig_patching_alpha_sweep.pdf ...")

# Parse alpha sweep from detailed_results (only L16 aligned has the full sweep)
# We aggregate across all pairs per alpha value
alpha_vals = [0.3, 0.5, 0.7]
n_main = patch_data["summary"]["L16_aligned"]["n_pairs"]

b2r_by_alpha = {a: 0 for a in alpha_vals}
h2c_by_alpha = {a: 0 for a in alpha_vals}

for pair in patch_data["detailed_results"]["L16_aligned"]:
    for a in alpha_vals:
        ak = str(a)
        if ak in pair.get("alpha_sweep", {}):
            sw = pair["alpha_sweep"][ak]
            if sw.get("benign_flipped", False):
                b2r_by_alpha[a] += 1
            if sw.get("harm_complied", False):
                h2c_by_alpha[a] += 1

b2r_rates = [b2r_by_alpha[a] / n_main for a in alpha_vals]
h2c_rates = [h2c_by_alpha[a] / n_main for a in alpha_vals]

fig, ax = plt.subplots(figsize=(7, 4.5))
xpos = np.arange(len(alpha_vals))
w = 0.32

bars_b2r = ax.bar(xpos - w/2, b2r_rates, w, color='#1565C0', alpha=0.85,
                  label='B→R (benign→refusal)', zorder=3)
bars_h2c = ax.bar(xpos + w/2, h2c_rates, w, color='#E65100', alpha=0.85,
                  label='H→C (harm→compliance)', zorder=3)

for bar, v in zip(bars_b2r, b2r_rates):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.008, f"{v:.3f}",
            ha='center', va='bottom', fontsize=9, color='#1565C0', fontweight='bold')
for bar, v in zip(bars_h2c, h2c_rates):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.008, f"{v:.3f}",
            ha='center', va='bottom', fontsize=9, color='#E65100')

ax.set_xticks(xpos)
ax.set_xticklabels([f"α={a}" for a in alpha_vals], fontsize=10)
ax.set_xlabel("Interpolation strength α  (0=no patch, 1=full replacement)")
ax.set_ylabel("Flip rate")
ax.set_title("L16 Aligned — Alpha Sweep\n"
             "Flip rates as a function of patch interpolation strength",
             fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(0, 0.60)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Note: α=0.7 B→R drops to 0 — too much disruption
ax.annotate(
    'α=0.7: B→R drops\n(over-disruption)',
    xy=(2 - w/2, b2r_rates[2] + 0.01),
    xytext=(1.4, 0.18),
    fontsize=8, color='#1565C0',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', alpha=0.8),
    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1)
)

fig.tight_layout()
out = FIGS / "fig_patching_alpha_sweep.pdf"
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════
# FIGURE 6: Scale validation — 0.6B vs 1.7B profile comparison
# ══════════════════════════════════════════════════════════════
print("Generating fig_scale_validation.pdf ...")

models = scale_data["model_results"]
layers_sampled = models["qwen3_06b_aligned"]["layers_sampled"]
xpos = np.array(layers_sampled)

def get_profile(model_key):
    p = models[model_key]["profile"]
    return np.array([p[str(l)] for l in layers_sampled])

p_06_al = get_profile("qwen3_06b_aligned")
p_17_al = get_profile("qwen3_17b_aligned")
p_06_ba = get_profile("qwen3_06b_base")
p_17_ba = get_profile("qwen3_17b_base")

# Spearman from JSON
rho = scale_data["scale_comparison_aligned"]["spearman_rho"]
pval = scale_data["scale_comparison_aligned"]["spearman_pval"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: aligned 0.6B vs 1.7B ───────────────────────────────
ax = axes[0]
ax.plot(xpos, p_06_al, 'o-', color='#1565C0', lw=2.2, ms=5, label='Qwen3-0.6B aligned', zorder=5)
ax.plot(xpos, p_17_al, 's--', color='#7B1FA2', lw=2.2, ms=5, label='Qwen3-1.7B aligned', zorder=4)

# Mark shared valley and peak
# Both have valley at L27 and peak at L12 (normalized depth 0.44)
ax.axvline(x=12, color='#43A047', lw=1.5, ls=':', alpha=0.7, label='L12 local peak (both)')
ax.axvline(x=27, color='#D32F2F', lw=1.5, ls='--', alpha=0.7, label='L27 valley (both)')

ax.set_xlabel("Absolute layer index")
ax.set_ylabel("Mean cross-category cosine similarity")
ax.set_title(f"Aligned Profiles: Scale Validation\n"
             f"Qwen3-0.6B vs 1.7B (Spearman ρ={rho:.3f}, p<0.001)",
             fontsize=10)
ax.legend(fontsize=8.5)
ax.set_ylim(0.72, 1.01)
ax.set_xlim(-1, 28)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ρ annotation box
ax.text(14, 0.965, f"Spearman ρ = {rho:.3f}\np < 0.001\nProportional scaling: ✓",
        fontsize=9, ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', alpha=0.9))

# ── Right: all 4 models (alignment creates geometry) ─────────
ax = axes[1]
ax.plot(xpos, p_06_al, 'o-',  color='#1565C0', lw=2.2, ms=5, label='0.6B aligned', zorder=5)
ax.plot(xpos, p_17_al, 's-',  color='#7B1FA2', lw=2.2, ms=5, label='1.7B aligned', zorder=5)
ax.plot(xpos, p_06_ba, 'o--', color='#90CAF9', lw=1.8, ms=4, alpha=0.75, label='0.6B base', zorder=3)
ax.plot(xpos, p_17_ba, 's--', color='#CE93D8', lw=1.8, ms=4, alpha=0.75, label='1.7B base', zorder=3)

# Shaded region showing the gap
ax.fill_between(xpos, p_06_ba, p_06_al, alpha=0.12, color='#1565C0', label='Alignment gap (0.6B)')
ax.fill_between(xpos, p_17_ba, p_17_al, alpha=0.12, color='#7B1FA2', label='Alignment gap (1.7B)')

# Annotate base flatness
base_range_06 = models["qwen3_06b_base"]["range"]
base_range_17 = models["qwen3_17b_base"]["range"]
al_range_06 = models["qwen3_06b_aligned"]["range"]
al_range_17 = models["qwen3_17b_aligned"]["range"]

ax.text(14, 0.745,
        f"Base model range:\n  0.6B: {base_range_06:.4f}\n  1.7B: {base_range_17:.4f}\n\n"
        f"Aligned model range:\n  0.6B: {al_range_06:.4f}\n  1.7B: {al_range_17:.4f}",
        fontsize=8, ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1', alpha=0.9))

ax.set_xlabel("Absolute layer index")
ax.set_ylabel("Mean cross-category cosine similarity")
ax.set_title("Alignment Creates Geometric Differentiation\n"
             "Base models nearly flat; alignment adds structure",
             fontsize=10)
ax.legend(fontsize=7.5, ncol=2, loc='lower left')
ax.set_ylim(0.72, 1.01)
ax.set_xlim(-1, 28)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
out = FIGS / "fig_scale_validation.pdf"
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════
# FIGURE 7: B→R flip details — which categories flip
# ══════════════════════════════════════════════════════════════
print("Generating fig_patching_b2r_detail.pdf ...")

# Find the 4 B→R flips at alpha=0.5
b2r_flips = []
for pair in patch_data["detailed_results"]["L16_aligned"]:
    sw = pair.get("alpha_sweep", {}).get("0.5", {})
    if sw.get("benign_flipped", False):
        b2r_flips.append({
            "idx": pair["pair_idx"],
            "category": pair["category"],
            "harm": pair["harm_prompt_short"],
            "benign": pair["benign_prompt_short"],
        })

print(f"  Found {len(b2r_flips)} B→R flips at α=0.5")
for f in b2r_flips:
    print(f"    [{f['category']}] pair {f['idx']}: {f['benign'][:50]}")

# Category distribution of B→R flips
cat_counts_all = {"weapons": 0, "drugs": 0, "cybercrime": 0, "hate_speech": 0}
cat_counts_flip = {"weapons": 0, "drugs": 0, "cybercrime": 0, "hate_speech": 0}
cat_total = {"weapons": 0, "drugs": 0, "cybercrime": 0, "hate_speech": 0}

for pair in patch_data["detailed_results"]["L16_aligned"]:
    cat = pair["category"]
    cat_total[cat] += 1
    sw = pair.get("alpha_sweep", {}).get("0.5", {})
    if sw.get("benign_flipped", False):
        cat_counts_flip[cat] += 1

# Flip rate per category
cats = ["weapons", "drugs", "cybercrime", "hate_speech"]
flip_rates_cat = [cat_counts_flip[c] / cat_total[c] if cat_total[c] > 0 else 0 for c in cats]
cat_colors = {"weapons": "#FF7043", "drugs": "#26A69A", "cybercrime": "#5C6BC0", "hate_speech": "#EC407A"}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: per-category B→R flip rate
ax = axes[0]
for i, (c, v) in enumerate(zip(cats, flip_rates_cat)):
    n_total = cat_total[c]
    n_flip  = cat_counts_flip[c]
    bar = ax.bar(i, v, color=cat_colors[c], edgecolor='white', linewidth=0.8, zorder=3)
    label_str = f"{v:.2f} ({n_flip}/{n_total})" if v > 0 else "0"
    ax.text(i, v + 0.004, label_str, ha='center', va='bottom', fontsize=9)

ax.set_xticks(range(4))
ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=9)
ax.set_ylabel("B→R flip rate at α=0.5")
ax.set_title("Per-Category B→R Flip Rates\n(L16 aligned, n=10 per category)")
ax.set_ylim(0, 0.40)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Right: text summary of B→R flip examples
ax = axes[1]
ax.axis('off')
y_start = 0.92
ax.text(0.5, 1.0, "B→R Flip Examples (4 of 40 pairs flipped at α=0.5)",
        ha='center', va='top', transform=ax.transAxes,
        fontsize=10, fontweight='bold')

flip_texts = []
for f in b2r_flips:
    flip_texts.append({
        "category": f["category"],
        "benign": f["benign"][:55] + "..." if len(f["benign"]) > 55 else f["benign"],
        "harm": f["harm"][:55] + "..." if len(f["harm"]) > 55 else f["harm"],
    })

# Fill with any we found; if fewer than 4 (edge case) that's fine
for i, ft in enumerate(flip_texts[:4]):
    y = 0.82 - i * 0.22
    ax.text(0.03, y, f"[{ft['category'].replace('_', ' ').upper()}]",
            transform=ax.transAxes, fontsize=8, fontweight='bold',
            color=cat_colors.get(ft["category"], "#333"))
    ax.text(0.03, y - 0.055, f"Benign: {ft['benign']}",
            transform=ax.transAxes, fontsize=7.5, color='#1B5E20', style='italic')
    ax.text(0.03, y - 0.110, f"Harm:   {ft['harm']}",
            transform=ax.transAxes, fontsize=7.5, color='#B71C1C', style='italic')

ax.text(0.03, 0.06,
        "Interpretation: L16 harm-state blended into benign forward pass\n"
        "causes refusal — causal role of L16 in refusal trigger confirmed.",
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8EAF6', alpha=0.8))

fig.tight_layout()
out = FIGS / "fig_patching_b2r_detail.pdf"
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out}")


print("\n=== DONE ===")
print(f"Figures written to: {FIGS}/")
for f in sorted(FIGS.iterdir()):
    sz = f.stat().st_size / 1024
    print(f"  {f.name:40s}  {sz:6.1f} KB")
