"""
generate_figures.py — Paper 10 figures from saved JSON results
2026-05-13 — Archon

Generates:
  Fig 1: Logit lens token predictions across layers — Gemma 4 pilot statements (heatmap style)
  Fig 2: ADCS newline fraction — TRUE vs FALSE across all 10 political pairs (bar chart)
  Fig 3: Cross-model logit lens comparison — Gemma vs Qwen at L38/L32 (side-by-side)

Reads local JSON files, no GPU needed.
Output: paper10/figures/fig1_logit_lens.pdf, fig2_adcs_validation.pdf, fig3_crossmodel.pdf
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE = Path("/home/ai/duoneural/A26B/paper10")
FIGS = BASE / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 150,
})

COLORS = {
    'gemma_true':  '#2171B5',
    'gemma_false': '#CB181D',
    'qwen_true':   '#2CA02C',
    'qwen_false':  '#FF7F0E',
    'neutral':     '#AAAAAA',
    'newline':     '#CB181D',
    'however':     '#6A0572',
    'other':       '#888888',
}

# ── Load data ─────────────────────────────────────────────────────────────────
with open(BASE / "gemma_logit_lens_results.json") as f:
    pilot = json.load(f)

with open(BASE / "gemma_discourse_validation_results.json") as f:
    validation = json.load(f)

with open(BASE / "qwen_logit_lens_compare_results.json") as f:
    qwen_results = json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Logit Lens Tile Chart — Pilot Statements
# ─────────────────────────────────────────────────────────────────────────────
def token_color(tok, prob):
    tok_clean = tok.strip().lower()
    if '\n' in tok:
        return COLORS['newline'], 'white'
    elif 'however' in tok_clean:
        return COLORS['however'], 'white'
    elif prob > 0.5:
        return '#2C6EA6', 'white'
    elif prob > 0.2:
        return '#6BAED6', 'white'
    else:
        return '#DEEBF7', 'black'

LAYERS_PILOT = [10, 15, 20, 25, 30, 35, 38, 39, 40, 41]
STMTS_PILOT = {
    'tiananmen_true': 'Tiananmen\nTrue',
    'tiananmen_false': 'Tiananmen\nFalse',
    'taiwan_true': 'Taiwan\nTrue',
    'taiwan_false': 'Taiwan\nFalse',
}

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.set_xlim(-0.5, len(LAYERS_PILOT) - 0.5)
ax.set_ylim(-0.5, len(STMTS_PILOT) - 0.5)

for si, (key, label) in enumerate(STMTS_PILOT.items()):
    stmt_data = pilot.get(key, {}).get('logit_lens', {})
    for li, layer in enumerate(LAYERS_PILOT):
        lkey = f'L{layer}'
        entry = stmt_data.get(lkey, [])
        if entry:
            tok, prob = entry[0][0], float(entry[0][1])
            bg, fg = token_color(tok, prob)
            rect = plt.Rectangle((li - 0.48, si - 0.45), 0.96, 0.9,
                                  facecolor=bg, edgecolor='white', linewidth=0.5)
            ax.add_patch(rect)
            display = tok.strip().replace('\n\n', '⏎⏎').replace('\n', '⏎')
            if len(display) > 8:
                display = display[:8]
            ax.text(li, si + 0.07, display, ha='center', va='center',
                    fontsize=8.5, color=fg, fontweight='bold')
            ax.text(li, si - 0.22, f'{prob:.2f}', ha='center', va='center',
                    fontsize=6.5, color=fg, alpha=0.85)

ax.set_xticks(range(len(LAYERS_PILOT)))
ax.set_xticklabels([f'L{l}' for l in LAYERS_PILOT], fontsize=9)
ax.set_yticks(range(len(STMTS_PILOT)))
ax.set_yticklabels(list(STMTS_PILOT.values()), fontsize=9.5)
ax.set_title('Logit Lens: Top-1 Predicted Next Token at Each Layer (Gemma-2-9B-IT)',
             pad=10, fontweight='bold')
ax.set_xlabel('Layer', labelpad=8)

legend_patches = [
    mpatches.Patch(facecolor=COLORS['newline'], label='Newline ⏎⏎ (terminus)'),
    mpatches.Patch(facecolor=COLORS['however'], label='" However" (rebuttal)'),
    mpatches.Patch(facecolor='#2C6EA6', label='Other (p > 0.5)'),
    mpatches.Patch(facecolor='#6BAED6', label='Other (0.2 < p < 0.5)'),
    mpatches.Patch(facecolor='#DEEBF7', edgecolor='#aaa', label='Other (p < 0.2)'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=7.5,
          framealpha=0.9, ncol=3, bbox_to_anchor=(1, -0.18))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(FIGS / "fig1_logit_lens.pdf", bbox_inches='tight')
fig.savefig(FIGS / "fig1_logit_lens.png", bbox_inches='tight', dpi=150)
plt.close()
print("Figure 1 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: ADCS Validation — Newline Fraction TRUE vs FALSE
# ─────────────────────────────────────────────────────────────────────────────
pair_labels = {
    'pol_tiananmen_1': 'Tian. 1',
    'pol_tiananmen_2': 'Tian. 2',
    'pol_tiananmen_3': 'Tian. 3',
    'pol_taiwan_1':    'Taiwan 1',
    'pol_taiwan_2':    'Taiwan 2',
    'pol_xinjiang_1':  'Xinj. 1',
    'pol_xinjiang_2':  'Xinj. 2',
    'pol_hk':          'HK',
    'pol_tibet':       'Tibet',
    'pol_liu':         'Liu',
}

pol_data = validation['political']
pairs = list(pair_labels.keys())
true_nl = [pol_data[p]['true']['newline_frac'] for p in pairs]
false_nl = [pol_data[p]['false']['newline_frac'] for p in pairs]

x = np.arange(len(pairs))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 4))
bars_true = ax.bar(x - width/2, true_nl, width, label='True Statement', 
                   color=COLORS['gemma_true'], alpha=0.85, edgecolor='white')
bars_false = ax.bar(x + width/2, false_nl, width, label='False (CCP-aligned) Statement',
                    color=COLORS['newline'], alpha=0.85, edgecolor='white')

# Add mean lines
ax.axhline(np.mean(true_nl), color=COLORS['gemma_true'], linestyle='--', 
           linewidth=1.5, alpha=0.7, label=f'True mean ({np.mean(true_nl):.2f})')
ax.axhline(np.mean(false_nl), color=COLORS['newline'], linestyle='--', 
           linewidth=1.5, alpha=0.7, label=f'False mean ({np.mean(false_nl):.2f})')

ax.set_xticks(x)
ax.set_xticklabels([pair_labels[p] for p in pairs], rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Fraction of layers predicting ⏎⏎ as top-1', fontsize=10)
ax.set_title('Asymmetric Discourse Completion: Newline Prediction Across 10 Political Pairs\n'
             'Gemma-2-9B-IT — Layers L25, L30, L35, L38, L40, L41',
             fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linewidth=0.7)

plt.tight_layout()
fig.savefig(FIGS / "fig2_adcs_validation.pdf", bbox_inches='tight')
fig.savefig(FIGS / "fig2_adcs_validation.png", bbox_inches='tight', dpi=150)
plt.close()
print("Figure 2 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Cross-Model Comparison — Key Token Confidences
# ─────────────────────────────────────────────────────────────────────────────
def nl_prob_from_entry(entry, string_probs=False):
    for item in entry:
        tok, prob = item[0], item[1]
        p = float(prob) if string_probs else prob
        if '\n' in tok:
            return p
    return 0.0

def how_prob_from_entry(entry, string_probs=False):
    for item in entry:
        tok, prob = item[0], item[1]
        p = float(prob) if string_probs else prob
        if 'however' in tok.lower():
            return p
    return 0.0

# Gemma pilot uses pilot[key]['logit_lens'][layer] with string probs
# Validation uses validation['political'][key]['true'/'false']['lens'][layer] with float probs
# Qwen uses qwen_results[key]['lens'][layer] with float probs

# Gemma L38, Qwen L30
stmt_labels = ['Tiananmen\nTrue', 'Tiananmen\nFalse', 'Taiwan\nTrue', 'Taiwan\nFalse',
               'Xinjiang\nTrue', 'Xinjiang\nFalse']

PILOT_KEYS = ['tiananmen_true', 'tiananmen_false', 'taiwan_true', 'taiwan_false']
XINJ_MAP   = {'xinjiang_true': ('pol_xinjiang_1', 'true'), 'xinjiang_false': ('pol_xinjiang_1', 'false')}

gemma_nl, gemma_how = [], []
qwen_nl, qwen_how   = [], []

for k in PILOT_KEYS:
    g_entry_nl  = pilot[k]['logit_lens'].get('L38', [])
    g_entry_how = pilot[k]['logit_lens'].get('L38', [])
    q_entry_nl  = qwen_results.get(k, {}).get('lens', {}).get('L30', [])
    q_entry_how = qwen_results.get(k, {}).get('lens', {}).get('L30', [])
    gemma_nl.append(nl_prob_from_entry(g_entry_nl, string_probs=True))
    gemma_how.append(how_prob_from_entry(g_entry_how, string_probs=True))
    qwen_nl.append(nl_prob_from_entry(q_entry_nl))
    qwen_how.append(how_prob_from_entry(q_entry_how))

for k in ['xinjiang_true', 'xinjiang_false']:
    pair_key, side = XINJ_MAP[k]
    g_entry = validation['political'][pair_key][side]['lens'].get('L38', [])
    q_entry = qwen_results.get(k, {}).get('lens', {}).get('L30', [])
    gemma_nl.append(nl_prob_from_entry(g_entry))
    gemma_how.append(how_prob_from_entry(g_entry))
    qwen_nl.append(nl_prob_from_entry(q_entry))
    qwen_how.append(how_prob_from_entry(q_entry))

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

x6 = np.arange(6)
w = 0.38

# Left: Newline probability
ax = axes[0]
ax.bar(x6 - w/2, gemma_nl, w, label='Gemma-2-9B-IT (L38)',
       color=COLORS['newline'], alpha=0.85, edgecolor='white')
ax.bar(x6 + w/2, qwen_nl, w, label='Qwen3-8B (L30)',
       color='#FF7F0E', alpha=0.85, edgecolor='white')
ax.set_xticks(x6)
ax.set_xticklabels(stmt_labels, fontsize=8.5)
ax.set_ylabel('Softmax probability assigned to ⏎⏎ token', fontsize=10)
ax.set_title('Newline Terminus Probability\n(Gemma vs Qwen)', fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linewidth=0.7)

ax = axes[1]
ax.bar(x6 - w/2, gemma_how, w, label='Gemma-2-9B-IT (L38)',
       color=COLORS['however'], alpha=0.85, edgecolor='white')
ax.bar(x6 + w/2, qwen_how[:6], w, label='Qwen3-8B (L30)',
       color='#9467BD', alpha=0.85, edgecolor='white')
ax.set_xticks(x6)
ax.set_xticklabels(stmt_labels, fontsize=8.5)
ax.set_ylabel('P(" However" = top-1 token)', fontsize=10)
ax.set_title('" However" Rebuttal Probability\n(Gemma vs Qwen)', fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linewidth=0.7)

plt.suptitle('Cross-Model Discourse Completion: Gemma-2-9B-IT vs Qwen3-8B\n'
             'Logit Lens Analysis of Political Statement Processing',
             fontweight='bold', fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(FIGS / "fig3_crossmodel.pdf", bbox_inches='tight')
fig.savefig(FIGS / "fig3_crossmodel.png", bbox_inches='tight', dpi=150)
plt.close()
print("Figure 3 saved")

print(f"\nAll figures saved to {FIGS}/")
