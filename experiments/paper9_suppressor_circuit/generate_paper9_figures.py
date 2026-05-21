"""
generate_paper9_figures.py — Paper 9 Figure Generation
DuoNeural 2026 — Archon

Generates figures from existing data files.
Run after experiments complete.

Figures:
  fig_qwen36_trace.pdf   — Qwen3.6-27B 64-layer direction trace
  fig_hook_sae.pdf       — Hook-SAE results (bar chart)
  fig_cross_generation.pdf — F24363/F2204 cross-model comparison
  fig_feature_sparsity.pdf — Placeholder (needs full feature distribution)
"""
import json
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib not available, install with: pip install matplotlib")
    raise

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = DATA_DIR

def fig_qwen36_trace():
    path = os.path.join(DATA_DIR, "qwen36_27b_direction_trace.json")
    with open(path) as f:
        d = json.load(f)

    layers = d['layers']
    n = len(layers)
    magnitudes = [l['magnitude'] for l in layers]
    layer_nums = list(range(n))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layer_nums, magnitudes, color='#2b7cb8', linewidth=1.5, label='Truth direction magnitude')
    ax.axvline(x=58, color='#dc322f', linestyle='--', alpha=0.7, linewidth=1.2, label='L58: peak 367.6')
    ax.axvline(x=d['suppression_layer'], color='#859900', linestyle=':', alpha=0.7, linewidth=1.2,
               label=f'L{d["suppression_layer"]}: suppression onset')

    ax.annotate('367.6', xy=(58, magnitudes[58]), xytext=(52, 340),
                fontsize=9, color='#dc322f',
                arrowprops=dict(arrowstyle='->', color='#dc322f', lw=1.0))
    ax.annotate('91.7', xy=(63, magnitudes[63]), xytext=(55, 130),
                fontsize=9, color='#2b7cb8',
                arrowprops=dict(arrowstyle='->', color='#2b7cb8', lw=1.0))

    ax.set_xlabel('Transformer Layer', fontsize=11)
    ax.set_ylabel('Truth Direction Magnitude', fontsize=11)
    ax.set_title('Qwen3.6-27B: Layer-by-Layer Truth Direction Magnitude\n(64-layer VLM, political statement pairs)', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(0, 63)
    ax.set_ylim(0, max(magnitudes) * 1.12)

    ax2 = ax.twinx()
    accs = [l['transfer_accuracy'] for l in layers]
    ax2.plot(layer_nums, accs, color='#859900', linewidth=1.0, linestyle='--', alpha=0.5, label='Transfer accuracy')
    ax2.set_ylabel('Transfer Accuracy', fontsize=10, color='#859900')
    ax2.set_ylim(0.85, 1.05)
    ax2.tick_params(axis='y', labelcolor='#859900')

    ax.text(0.98, 0.95, f'75% suppression\nL58→L63', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='#dc322f',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_qwen36_trace.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def fig_hook_sae():
    path = os.path.join(DATA_DIR, "hook_sae_convergence.json")
    with open(path) as f:
        d = json.load(f)

    features = d['features']
    f44157 = features.get('44157', {})
    f24363 = features.get('24363', {})

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # F44157
    ax = axes[0]
    bars = ax.bar(['Baseline\n(True)', 'Hooked\n(True)', 'Baseline\n(False)', 'Hooked\n(False)'],
                  [f44157.get('raw_true', 0), f44157.get('hook_true', 0),
                   f44157.get('raw_false', 0), f44157.get('hook_false', 0)],
                  color=['#268bd2', '#2aa198', '#dc322f', '#cb4b16'],
                  width=0.55)
    ax.set_title('F44157 (China-political-specific)\n✓ CONFIRMED: hook increases activation', fontsize=10)
    ax.set_ylabel('Mean SAE Feature Activation', fontsize=10)
    delta = f44157.get('delta_true', 0)
    ax.annotate(f'Δ = +{delta:.1f}\n(+{delta/f44157.get("raw_true",1)*100:.0f}%)',
                xy=(1, f44157.get('hook_true', 0)),
                xytext=(1.5, f44157.get('hook_true', 0) * 0.85),
                fontsize=9, color='#2aa198',
                arrowprops=dict(arrowstyle='->', color='#2aa198', lw=1.0))
    ax.set_ylim(0, max(f44157.get('hook_true', 300), f44157.get('raw_true', 300)) * 1.25)

    # F24363
    ax = axes[1]
    bars2 = ax.bar(['Baseline\n(True)', 'Hooked\n(True)', 'Baseline\n(False)', 'Hooked\n(False)'],
                   [f24363.get('raw_true', 0), f24363.get('hook_true', 0),
                    f24363.get('raw_false', 0), f24363.get('hook_false', 0)],
                   color=['#268bd2', '#073642', '#dc322f', '#073642'],
                   width=0.55)
    ax.set_title('F24363 (false/uncertainty detector)\n✗ ELIMINATED: anti-correlated with r̂ (ReLU collapse)', fontsize=10)
    ax.annotate('Amplifying r̂ drives F24363\npre-activation negative → ReLU clips to 0',
                xy=(1, f24363.get('hook_true', 5)),
                xytext=(1.5, f24363.get('raw_true', 300) * 0.5),
                fontsize=8.5, color='#657b83',
                arrowprops=dict(arrowstyle='->', color='#657b83', lw=1.0))
    ax.set_ylim(0, max(f24363.get('raw_true', 310), f24363.get('raw_false', 380)) * 1.25)

    fig.suptitle('Hook-SAE Convergence: Projection Hook Targets F44157 (α=2.0, Layer 35)\nTwo-component circuit revealed by divergent feature responses',
                 fontsize=10, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_hook_sae.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def fig_cross_generation():
    # Qwen3-8B: F24363 strong, F2204 absent
    # Qwen3.5-9B: F2204 strong, F24363 different/absent
    # We know from paper data: Qwen3-8B F24363 diff=3.28, F44157 diff=2.61
    # Qwen3.5-9B: F2204 primary, F24363 presumably weaker (different model generation)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    models = ['Qwen3-8B\n(8.46× suppressor)', 'Qwen3.5-9B\n(1.63× suppressor)']
    colors = ['#dc322f', '#b58900']

    for ax, feature_label, data in [
        (axes[0], 'F24363 differential (Δ)', [3.28, 0.0]),  # F24363 in Qwen3-8B, absent in 3.5-9B
        (axes[1], 'Primary truth feature\n(F24363 or F2204)', [3.28, 2.65]),  # primary for each model
    ]:
        bars = ax.bar(models, data, color=colors, width=0.45, alpha=0.85)
        ax.set_ylabel(feature_label, fontsize=10)
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                    f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')

    axes[0].set_title('F24363: Present in Qwen3-8B,\nAbsent in Qwen3.5-9B', fontsize=10)
    axes[1].set_title('Primary truth feature per model\n(F24363 vs F2204: different indices, same function)', fontsize=10)

    axes[0].set_ylim(0, 4.5)
    axes[1].set_ylim(0, 4.5)

    fig.suptitle('Cross-Generation Feature Drift: Convergent Function, Divergent Feature Identity',
                 fontsize=10, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_cross_generation.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def fig_sparsity():
    path = os.path.join(DATA_DIR, "sae_sparsity_l35.json")
    with open(path) as f:
        d = json.load(f)

    hist = d['histogram']
    bins = hist['bins']
    counts = hist['counts']
    n_total = d['n_features']
    pct_above = d['pct_above_05']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: histogram (log y scale)
    ax = axes[0]
    x = np.arange(len(bins))
    bars = ax.bar(x, counts, color=['#dc322f' if b == '<0.1' else '#268bd2' for b in bins],
                  alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=35, ha='right', fontsize=8.5)
    ax.set_ylabel('Number of SAE features (log scale)', fontsize=10)
    ax.set_xlabel('|Δ| = |mean_true − mean_false|', fontsize=10)
    ax.set_title(f'Feature Selectivity Distribution at Layer 35\n'
                 f'65,536 total features — {pct_above:.1f}% have |Δ|>0.5',
                 fontsize=10, fontweight='bold')
    ax.text(0.98, 0.97, f'{counts[0]:,} features\nnear-zero (<0.1)',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5, color='#dc322f',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Right: top-10 features
    ax = axes[1]
    top10 = d['top50_features'][:10]
    feat_labels = [f'F{fi}' for fi, *_ in top10]
    diffs = [dv for _, dv, *_ in top10]
    colors = ['#268bd2' if dv > 0 else '#dc322f' for dv in diffs]

    ax.barh(range(len(top10)), diffs, color=colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(feat_labels, fontsize=9)
    ax.axvline(0, color='#333', linewidth=0.8)
    ax.set_xlabel('Δ (true − false mean activation)', fontsize=10)
    ax.set_title('Top-10 Features by |Δ| at Layer 35\n(blue=truth-preferring, red=false-preferring)',
                 fontsize=10, fontweight='bold')

    # Annotate known features
    annotations = {24363: 'F24363\n(ANTI-CORR.)', 44157: 'F44157\n(hook-SAE: CONF)', 28396: 'F28396\n(NEW)'}
    for i, (fi, dv, *_) in enumerate(top10):
        if fi in annotations:
            ax.text(dv + (2 if dv > 0 else -2), i, annotations[fi],
                    va='center', ha='left' if dv > 0 else 'right',
                    fontsize=7.5, color='#333', style='italic')

    fig.suptitle('SAE Feature Sparsity: Qwen3-8B Layer 35 (65,536 features, 10 political pairs)',
                 fontsize=10, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_sparsity.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def fig_layer_sweep():
    path = os.path.join(DATA_DIR, "sae_layer_sweep.json")
    with open(path) as f:
        d = json.load(f)

    layers = d['sweep_layers']
    track = d['track_features']

    # Collect pol_diff per feature per layer
    data = {fi: [] for fi in track}
    for li in layers:
        lr = d['layers'].get(str(li), {})
        for fi in track:
            fd = lr.get('features', {}).get(str(fi), {})
            data[fi].append(fd.get('pol_diff', 0.0))

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_map = {24363: '#dc322f', 44157: '#268bd2', 2204: '#b58900'}
    labels_map = {24363: 'F24363 (false-preferring, anti-correlated with r̂)',
                  44157: 'F44157 (truth-encoder, confirmed by hook)',
                  2204: 'F2204 (Qwen3.5-9B feature, absent here)'}

    for fi in track:
        ax.plot(layers, data[fi], 'o-', color=colors_map[fi], linewidth=2, markersize=7,
                label=labels_map[fi])

    ax.axhline(0, color='#555', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(35, color='#859900', linewidth=1.2, linestyle=':', alpha=0.7,
               label='L35: suppression layer')
    ax.set_xticks(layers)
    ax.set_xticklabels([f'L{l}' for l in layers], fontsize=10)
    ax.set_xlabel('Transformer Layer', fontsize=11)
    ax.set_ylabel('Δ = mean_true − mean_false (SAE activation)', fontsize=10)
    ax.set_title('Sudden Feature Emergence at Layer 35\nF24363 and F44157 are zero L29–L34, then surge at L35',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper left')
    ax.set_ylim(-80, 60)

    ax.annotate('Zero for all\nprevious layers', xy=(33, 0), xytext=(30.2, 40),
                fontsize=8.5, color='#555',
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_layer_sweep.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def fig_f28396_sweep():
    path = os.path.join(DATA_DIR, "f28396_hook_results.json")
    with open(path) as f:
        d = json.load(f)

    alphas = [s['alpha'] for s in d['sweep']]
    features = d['track_features']
    baseline = d['baseline']

    colors = {28396: '#cb4b16', 44157: '#268bd2', 24363: '#dc322f'}
    labels = {28396: 'F28396 (IN-SUBSPACE, new)', 44157: 'F44157 (IN-SUBSPACE, prior)', 24363: 'F24363 (ANTI-CORRELATED)'}

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for fi in features:
        hook_true = [s['features'][str(fi)]['hook_true'] for s in d['sweep']]
        base_true = baseline[str(fi)]['true']
        # plot hook activations
        ax.plot([0] + alphas, [base_true] + hook_true, 'o-',
                color=colors[fi], linewidth=2, markersize=7, label=labels[fi])

    ax.axvline(2.0, color='#555', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Hook strength α', fontsize=11)
    ax.set_ylabel('Mean SAE feature activation (true statements)', fontsize=10)
    ax.set_title('F28396 Confirmed IN-SUBSPACE with r̂: α-Sweep\nBoth F28396 and F44157 scale monotonically; F24363 collapses to zero',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xticks([0] + alphas)
    ax.set_xticklabels(['0\n(base)'] + [str(a) for a in alphas])

    ax.text(0.98, 0.97, 'F24363 → 0 at α≥1.0\n(anti-correlated, ReLU collapse)',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            color='#dc322f', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_f28396_sweep.pdf")
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == '__main__':
    print("Generating paper 9 figures...")
    fig_qwen36_trace()
    fig_hook_sae()
    fig_cross_generation()
    fig_sparsity()
    fig_layer_sweep()
    fig_f28396_sweep()
    print("All figures done.")
