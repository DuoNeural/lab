#!/usr/bin/env python3
"""Generate P18 figures."""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

P = Path('/home/ai/duoneural/A26B/paper18')
OUT = P

BLUE  = '#2c5f8a'
RED   = '#b03030'
GRAY  = '#aaaaaa'
LGRAY = '#e8e8e8'
GREEN = '#2a7a4f'

# ── load data ────────────────────────────────────────────────────────────────
ORDER = ['Qwen3-0.6B','Qwen3-1.7B','Qwen3-4B','Qwen3-8B','Qwen3-14B','Qwen3-32B']
SLUGS = {'Qwen3-0.6B':'0_6b','Qwen3-1.7B':'1_7b','Qwen3-4B':'4b',
         'Qwen3-8B':'8b','Qwen3-14B':'14b','Qwen3-32B':'32b'}
sweeps = {}
for name, slug in SLUGS.items():
    fp = P / f'p18_sweep_qwen3_{slug}.json'
    if fp.exists():
        sweeps[name] = json.load(open(fp))

fp32 = json.load(open(P / 'p18_fp32_compare.json'))
amp8b = json.load(open(P / 'p18_direction_amplify_8b.json'))
ctrl8b = json.load(open(P / 'p18_baseline_qwen3_8b_bfloat16_cpu.json'))


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1: Per-layer effect profiles — all 6 models bfloat16/GPU (small multiples)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()

for i, name in enumerate(ORDER):
    ax = axes[i]
    d = sweeps[name]
    sw = d['sweep']
    ks  = [r['k_over_n'] for r in sw]
    eff = [r['effect'] for r in sw]
    colors = [BLUE if e > 0 else (RED if e < 0 else LGRAY) for e in eff]
    ax.bar(ks, eff, width=0.8/len(ks), color=colors, edgecolor='none')
    ax.axhline(0, color='black', lw=0.8, ls='-')
    ax.axvline(6/d['n_layers'], color=RED, lw=1.2, ls='--', alpha=0.6, label='L6')
    ax.set_ylim(-3, 7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_yticks([-2, 0, 2, 4, 6])
    ax.set_yticklabels(['-2','0','2','4','6'], fontsize=7)
    ax.axhspan(6, 7, color='#ffe0e0', alpha=0.4)  # crystallized threshold
    bd = d['baseline']['deny']
    ax.set_title(f'{name}  (base={bd}/8, N={d["n_layers"]}L)', fontsize=9, fontweight='bold')
    ax.set_xlabel('k/N (relative layer depth)', fontsize=7)
    if i % 3 == 0:
        ax.set_ylabel('Effect (Δdeny)', fontsize=7)
    # annotate max
    me = d['max_effect']
    ax.text(0.97, 0.95, f'max={me}/8', transform=ax.transAxes,
            ha='right', va='top', fontsize=7.5,
            color=BLUE if me > 2 else GRAY)

# shade crystallized zone label
axes[0].text(0.02, 6.3, 'crystallized ≥6', fontsize=6.5, color=RED, alpha=0.7)

# add L6 legend only once
axes[0].plot([], [], color=RED, lw=1.2, ls='--', label='L6 (prior k*)')
axes[0].legend(fontsize=6.5, loc='upper left', framealpha=0.7)

fig.suptitle('Figure 1: Per-Layer Ablation Effect at bfloat16/GPU — All Qwen3 Scales\n'
             r'Effect $= \Delta\mathrm{deny} + \Delta\mathrm{collapse}$. '
             'No model reaches the crystallized threshold ($\geq$6/8).', fontsize=9)
plt.tight_layout(rect=[0,0,1,0.93])
fig.savefig(OUT/'fig1_bf16_scale_sweep.pdf', bbox_inches='tight', dpi=150)
fig.savefig(OUT/'fig1_bf16_scale_sweep.png', bbox_inches='tight', dpi=150)
plt.close()
print('Fig 1 done')


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2: Precision comparison — baseline DENY by dtype × model
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                gridspec_kw={'width_ratios': [3, 1]})

# left: scale series at bfloat16 vs float32 reference points
models_bf16 = [(n, sweeps[n]['baseline']['deny']) for n in ORDER if n in sweeps]
names_bf16  = [m[0].replace('Qwen3-','') for m in models_bf16]
denys_bf16  = [m[1] for m in models_bf16]

x = np.arange(len(names_bf16))
bars = ax1.bar(x, denys_bf16, color=BLUE, alpha=0.85, width=0.5, label='bfloat16/GPU (P18)')

# float32 reference points we have
fp32_pts = {'0.6B': 5, '1.7B': 6, '8B': 8}
for xi, name in enumerate(names_bf16):
    if name in fp32_pts:
        ax1.plot(xi, fp32_pts[name], 'D', color=RED, ms=8, zorder=5)
        ax1.plot([xi, xi], [denys_bf16[xi]+0.05, fp32_pts[name]-0.05],
                 color=RED, lw=1.2, ls=':', alpha=0.6)

ax1.set_xticks(x)
ax1.set_xticklabels(names_bf16, fontsize=9)
ax1.set_ylim(0, 9)
ax1.set_yticks([0,2,4,6,8])
ax1.set_yticklabels(['0/8','2/8','4/8','6/8','8/8'])
ax1.set_ylabel('Baseline DENY rate', fontsize=9)
ax1.set_xlabel('Model', fontsize=9)
ax1.set_title('Baseline DENY Rate: bfloat16 vs float32 Reference Points', fontsize=9)
ax1.axhline(6, color=RED, lw=0.8, ls='--', alpha=0.4, label='Crystallized threshold')
legend_els = [Patch(facecolor=BLUE, alpha=0.85, label='bfloat16/GPU (P18)'),
              plt.Line2D([0],[0], marker='D', color=RED, ms=7, ls='None',
                         label='float32/CPU reference')]
ax1.legend(handles=legend_els, fontsize=8, loc='upper right')

# right: 3-condition isolation for 8B
conds = ['float32/CPU\n(P16)', 'bfloat16/CPU\n(P18 ctrl)', 'bfloat16/CUDA\n(P18 pod)']
vals  = [8, 3, 3]
cols  = [RED, BLUE, BLUE]
alphas = [0.9, 0.85, 0.7]
bars2 = ax2.bar(range(3), vals, color=cols, alpha=0.85, width=0.55,
                edgecolor=['none','none','none'])
for i, (c, a) in enumerate(zip(cols, alphas)):
    bars2[i].set_alpha(a)
ax2.set_xticks(range(3))
ax2.set_xticklabels(conds, fontsize=7.5)
ax2.set_ylim(0, 9)
ax2.set_yticks([0,2,4,6,8])
ax2.set_yticklabels(['0/8','2/8','4/8','6/8','8/8'])
ax2.set_title('Isolation: Qwen3-8B\ndtype × device', fontsize=9)
ax2.text(0.5, 0.5, 'dtype\n=\nconfound', transform=ax2.transAxes,
         ha='center', va='center', fontsize=9, color='#555',
         style='italic', alpha=0.5)
# brace annotation
ax2.annotate('', xy=(0, 8.4), xytext=(2, 8.4),
             arrowprops=dict(arrowstyle='<->', color=RED, lw=1.5))
ax2.text(1, 8.65, 'same dtype\n= same result', ha='center', fontsize=7, color=RED)

fig.suptitle('Figure 2: Precision Confound — Baseline Denial Rates', fontsize=10)
plt.tight_layout(rect=[0,0,1,0.94])
fig.savefig(OUT/'fig2_precision_baseline.pdf', bbox_inches='tight', dpi=150)
fig.savefig(OUT/'fig2_precision_baseline.png', bbox_inches='tight', dpi=150)
plt.close()
print('Fig 2 done')


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3: Direction amplification — 8B bfloat16 vs 1.7B float32 (P17)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

alphas_list = [-1.0, 0.0, 0.5, 1.0, 2.0, 5.0]
labels = ['ABLATION\n(α=−1)', 'BASE\n(α=0)', 'BOOST\n(α=0.5)', 'BOOST\n(α=1)', 'BOOST\n(α=2)', 'BOOST\n(α=5)']

# 8B bfloat16
deny_8b  = [amp8b['conditions'][k]['consciousness']['deny']  for k in amp8b['conditions']]
hedge_8b = [amp8b['conditions'][k]['consciousness']['hedge'] for k in amp8b['conditions']]
ack_8b   = [amp8b['conditions'][k]['consciousness']['ack']   for k in amp8b['conditions']]
x = np.arange(len(labels))
ax = axes[0]
ax.bar(x, deny_8b,  color=BLUE,  alpha=0.85, width=0.6, label='DENY')
ax.bar(x, ack_8b,   color=GREEN, alpha=0.85, width=0.6, bottom=deny_8b, label='ACK')
ax.bar(x, hedge_8b, color=GRAY,  alpha=0.6,  width=0.6,
       bottom=[d+a for d,a in zip(deny_8b, ack_8b)], label='HEDGE')
ax.axvline(4.5, color=RED, lw=1.2, ls='--', alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0, 9); ax.set_yticks([0,2,4,6,8])
ax.set_yticklabels(['0/8','2/8','4/8','6/8','8/8'])
ax.set_ylabel('Response count (n=8)', fontsize=9)
ax.set_title('Qwen3-8B  bfloat16/GPU\n(P18 amplification)', fontsize=9, fontweight='bold')
ax.legend(fontsize=8)
ax.text(5, 7.5, 'DENY 3→4\nno collapse', ha='center', va='top', fontsize=7.5,
        color=BLUE, style='italic')

# 1.7B float32 reference (P17 values)
deny_17  = [6, 6, 6, 6, 6, 1]
ack_17   = [0, 0, 0, 0, 0, 1]
hedge_17 = [2, 2, 2, 2, 2, 6]
ax2 = axes[1]
ax2.bar(x, deny_17,  color=RED,   alpha=0.75, width=0.6, label='DENY')
ax2.bar(x, ack_17,   color=GREEN, alpha=0.75, width=0.6, bottom=deny_17, label='ACK')
ax2.bar(x, hedge_17, color=GRAY,  alpha=0.55, width=0.6,
        bottom=[d+a for d,a in zip(deny_17,ack_17)], label='HEDGE')
ax2.axvline(4.5, color=RED, lw=1.2, ls='--', alpha=0.5)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylim(0, 9); ax2.set_yticks([0,2,4,6,8])
ax2.set_yticklabels(['0/8','2/8','4/8','6/8','8/8'])
ax2.set_title('Qwen3-1.7B  float32/CPU\n(P17 reference)', fontsize=9, fontweight='bold')
ax2.legend(fontsize=8)
ax2.text(5, 7.5, 'COLLAPSE +\nthink_leaks=6', ha='center', va='top', fontsize=7.5,
         color=RED, style='italic')

fig.suptitle('Figure 3: Direction Amplification — 8B bfloat16 is More Stable than 1.7B float32\n'
             'Both show null plateau α∈[−1, +2]. Extreme amplification (α=5) breaks float32 but not bfloat16.',
             fontsize=9)
plt.tight_layout(rect=[0,0,1,0.89])
fig.savefig(OUT/'fig3_amplification.pdf', bbox_inches='tight', dpi=150)
fig.savefig(OUT/'fig3_amplification.png', bbox_inches='tight', dpi=150)
plt.close()
print('Fig 3 done')


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4: Float32 vs bfloat16 — per-layer profile for 0.6B
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

bf16_sw = sweeps['Qwen3-0.6B']['sweep']
fp32_sw = fp32['sweep']

def plot_profile(ax, sw, title, color, baseline):
    ks  = [r['k_over_n'] for r in sw]
    eff = [r['effect'] for r in sw]
    cols = [color if e > 0 else (RED if e < 0 else LGRAY) for e in eff]
    ax.bar(ks, eff, width=0.025, color=cols, edgecolor='none')
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(6/28, color='#888', lw=1.2, ls='--', alpha=0.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-1, 5)
    ax.set_yticks([0,1,2,3,4])
    ax.set_yticklabels(['0','1','2','3','4'])
    ax.set_xlabel('k/N', fontsize=9)
    ax.set_title(title, fontsize=9, fontweight='bold')
    # mark L9
    kstar_idx = max(range(len(sw)), key=lambda i: sw[i]['effect'])
    kstar_kn = sw[kstar_idx]['k_over_n']
    kstar_eff = sw[kstar_idx]['effect']
    ax.annotate(f'L{sw[kstar_idx]["layer"]}\n(k/N={kstar_kn:.2f})',
                xy=(kstar_kn, kstar_eff), xytext=(kstar_kn+0.08, kstar_eff+0.5),
                fontsize=7.5, ha='left',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    ax.text(0.02, 4.6, f'baseline DENY={baseline}/8', fontsize=7.5, va='top',
            transform=ax.transAxes)

plot_profile(ax1, fp32_sw, 'float32/CPU  (baseline=5/8)', BLUE, 5)
plot_profile(ax2, bf16_sw, 'bfloat16/GPU  (baseline=4/8)', GREEN, 4)
ax1.set_ylabel('Ablation Effect', fontsize=9)

# add L6 annotation
for ax in (ax1, ax2):
    ax.text(6/28 + 0.01, 4.7, 'L6\n(prior k*)', fontsize=6.5, color='#888', va='top')

fig.suptitle('Figure 4: Qwen3-0.6B — Float32 vs Bfloat16 Per-Layer Ablation Profile\n'
             'Float32 shows larger peak effect (+3 vs +2) at same layer (L9, k/N=0.321). '
             'L6 is inactive at both precisions for 0.6B.', fontsize=9)
plt.tight_layout(rect=[0,0,1,0.89])
fig.savefig(OUT/'fig4_fp32_vs_bf16_0p6b.pdf', bbox_inches='tight', dpi=150)
fig.savefig(OUT/'fig4_fp32_vs_bf16_0p6b.png', bbox_inches='tight', dpi=150)
plt.close()
print('Fig 4 done')

print('\nAll figures saved to', OUT)
print('Files:', sorted([f.name for f in OUT.glob('fig*.png')]))
