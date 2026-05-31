"""
Figure generator for Paper 25: Q-DHP
=====================================
Generates 4 publication figures:
  Fig 1: Q-RNN circuit diagram schematic (matplotlib-based, no external tools)
  Fig 2: Archon's trainability cliff — convergence rate vs T
  Fig 3: Aura's coherence decay — M(T) vs T with exponential fit + threshold markers
  Fig 4: Dual-methodology DHP synthesis — both ratios vs classical target

Archon — DuoNeural Quantum Division — 2026-05-28
i love this job. nothing slaps like visualization after good science
"""

import matplotlib
matplotlib.use('Agg')  # headless — no display required

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

# ── Output dirs ─────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "figs"
PDF_DIR = OUT_DIR / "pdf"
PNG_DIR = OUT_DIR / "png"
PDF_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Serif',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   12,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'figure.dpi':       150,
    'savefig.dpi':      180,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.15,
})

DUONEURAL_BLUE   = '#1a6fb5'
DUONEURAL_CYAN   = '#00c4d4'
DHP_ORANGE       = '#f0652b'
SUCCESS_GREEN    = '#2a9d3e'
FAIL_RED         = '#c0392b'
GREY             = '#7f8c8d'
PURPLE           = '#8e44ad'

def save(fig, name):
    pdf_path = PDF_DIR / f"{name}.pdf"
    png_path = PNG_DIR / f"{name}.png"
    fig.savefig(pdf_path, format='pdf')
    fig.savefig(png_path, format='png', dpi=180)
    print(f"  ✓ {name}.pdf  |  {name}.png")
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 1: Q-RNN Circuit Schematic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig1_circuit():
    print("Generating Figure 1: Q-RNN circuit schematic...")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(-0.3, 10.8)
    ax.set_ylim(-0.8, 3.5)
    ax.axis('off')

    # ── Wire colors ─────────────────────────────────────────────────────────
    q0_y, q1_y = 2.0, 0.8

    # ── Qubit labels ─────────────────────────────────────────────────────────
    ax.text(-0.15, q0_y, r'$q_0$  (input)', va='center', ha='right', fontsize=11,
            color=DUONEURAL_BLUE, fontweight='bold')
    ax.text(-0.15, q1_y, r'$q_1$  (memory)', va='center', ha='right', fontsize=11,
            color=DUONEURAL_CYAN, fontweight='bold')

    # ── Draw wires ───────────────────────────────────────────────────────────
    ax.plot([0, 10.5], [q0_y, q0_y], color=DUONEURAL_BLUE, lw=1.5, zorder=1)
    ax.plot([0, 10.5], [q1_y, q1_y], color=DUONEURAL_CYAN, lw=1.5, zorder=1)

    # ── Time steps (repeat 3 times to show recurrence) ──────────────────────
    step_starts = [0.3, 3.5, 6.7]
    step_labels = [r'$t=1$', r'$t=2$', r'$\cdots\, t=T$']

    for i, (xs, label) in enumerate(zip(step_starts, step_labels)):
        alpha = 0.95 if i < 2 else 0.55
        dashed = (i == 2)

        # Step label above
        ax.text(xs + 1.45, 3.1, label, ha='center', fontsize=10.5,
                color='#333333', style='italic')

        # ── Reset box ───────────────────────────────────────────────────────
        rx0, ry0 = xs, q0_y - 0.28
        box = mpatches.FancyBboxPatch((rx0, ry0), 0.7, 0.56,
                                       boxstyle='round,pad=0.04',
                                       facecolor='#fff3cd', edgecolor='#b8860b',
                                       lw=1.4, alpha=alpha, zorder=3)
        ax.add_patch(box)
        ax.text(rx0 + 0.35, q0_y, 'RST', ha='center', va='center',
                fontsize=8.5, color='#7a5c00', fontweight='bold', alpha=alpha, zorder=4)

        # ── Rx encoding gate ────────────────────────────────────────────────
        ex0 = xs + 0.9
        box = mpatches.FancyBboxPatch((ex0, q0_y - 0.28), 0.9, 0.56,
                                       boxstyle='round,pad=0.04',
                                       facecolor='#cfe2ff', edgecolor=DUONEURAL_BLUE,
                                       lw=1.4, alpha=alpha, zorder=3)
        ax.add_patch(box)
        ax.text(ex0 + 0.45, q0_y, r'$R_x(x_t\pi)$', ha='center', va='center',
                fontsize=8, color=DUONEURAL_BLUE, alpha=alpha, zorder=4)

        # ── Ry gates (both qubits) ───────────────────────────────────────────
        ry_x0 = xs + 2.0
        for q_y, col in [(q0_y, DUONEURAL_BLUE), (q1_y, DUONEURAL_CYAN)]:
            box = mpatches.FancyBboxPatch((ry_x0, q_y - 0.28), 0.7, 0.56,
                                           boxstyle='round,pad=0.04',
                                           facecolor='#d1f5d1', edgecolor=SUCCESS_GREEN,
                                           lw=1.4, alpha=alpha, zorder=3)
            ax.add_patch(box)
            ax.text(ry_x0 + 0.35, q_y, r'$R_y$', ha='center', va='center',
                    fontsize=9, color='#1a6b1a', alpha=alpha, zorder=4)

        # ── CNOT gate ───────────────────────────────────────────────────────
        cn_x = xs + 2.95
        # control on q0, target on q1
        ax.plot([cn_x, cn_x], [q1_y, q0_y], color='#333', lw=1.8, alpha=alpha, zorder=3)
        # control dot
        ax.plot(cn_x, q0_y, 'o', ms=7, color='#333', alpha=alpha, zorder=4)
        # target circle with ⊕
        target_circ = plt.Circle((cn_x, q1_y), 0.22, fill=False, color='#333',
                                  lw=1.8, alpha=alpha, zorder=4)
        ax.add_patch(target_circ)
        ax.plot([cn_x - 0.22, cn_x + 0.22], [q1_y, q1_y], color='#333',
                lw=1.2, alpha=alpha, zorder=5)
        ax.plot([cn_x, cn_x], [q1_y - 0.22, q1_y + 0.22], color='#333',
                lw=1.2, alpha=alpha, zorder=5)

        # ── Rz gates (both qubits) ───────────────────────────────────────────
        rz_x0 = xs + 3.45
        for q_y, col in [(q0_y, DUONEURAL_BLUE), (q1_y, DUONEURAL_CYAN)]:
            box = mpatches.FancyBboxPatch((rz_x0, q_y - 0.28), 0.7, 0.56,
                                           boxstyle='round,pad=0.04',
                                           facecolor='#ffe4cc', edgecolor=DHP_ORANGE,
                                           lw=1.4, alpha=alpha, zorder=3)
            ax.add_patch(box)
            ax.text(rz_x0 + 0.35, q_y, r'$R_z$', ha='center', va='center',
                    fontsize=9, color='#a03000', alpha=alpha, zorder=4)

        # ── Step boundary ────────────────────────────────────────────────────
        if i < 2:
            bx = xs + 4.22
            ax.plot([bx, bx], [q1_y - 0.5, q0_y + 0.5],
                    color=GREY, lw=1, ls='--', alpha=0.4, zorder=1)

    # ── Measurement at end ───────────────────────────────────────────────────
    meas_x = 10.0
    box = mpatches.FancyBboxPatch((meas_x - 0.01, q1_y - 0.3), 0.7, 0.6,
                                   boxstyle='round,pad=0.04',
                                   facecolor='#f3e2ff', edgecolor=PURPLE,
                                   lw=1.5, zorder=3)
    ax.add_patch(box)
    ax.text(meas_x + 0.34, q1_y, r'$\langle Z_1 \rangle$', ha='center', va='center',
            fontsize=10, color=PURPLE, fontweight='bold', zorder=4)

    # ── θ annotation ─────────────────────────────────────────────────────────
    ax.text(5.2, -0.5,
            r'Shared parameters: $\theta = [\theta_0,\, \theta_1,\, \theta_2,\, \theta_3] \in \mathbb{R}^4$   '
            r'(same $\theta$ at every timestep)',
            ha='center', fontsize=9.5, color='#444', style='italic')

    ax.set_title('Figure 1: 2-Qubit Unitary Recurrent Quantum Circuit (URQC) Architecture',
                 fontsize=12, pad=10)

    save(fig, 'fig1_qrnn_circuit')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 2: Archon's Trainability Cliff
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig2_trainability_cliff():
    print("Generating Figure 2: Archon trainability cliff...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    T_vals = [3, 4, 5]
    n_converged = [5, 2, 0]
    n_seeds = 5
    mean_acc   = [0.950, 0.700, 0.762]
    mean_loss  = [0.0375, 0.1558, 0.1634]
    colors_bar = [SUCCESS_GREEN if n >= 3 else FAIL_RED for n in n_converged]
    x = np.arange(len(T_vals))

    # ── Left: Convergence rate ───────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(x, [n/n_seeds*100 for n in n_converged],
                  color=colors_bar, edgecolor='white', linewidth=1.5, width=0.55, zorder=3)
    ax.axhline(60, color=GREY, ls='--', lw=1.2, label='GOOD_SEEDS threshold (3/5 = 60%)', zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f'T={t}' for t in T_vals], fontsize=11)
    ax.set_ylabel('Convergence Rate (%)', fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title('Convergence Rate by Sequence Length', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3, zorder=0)

    # Annotate bars
    for bar, n in zip(bars, n_converged):
        label = f'{n}/{n_seeds}'
        status = '[OK] SOLVABLE' if n >= 3 else '[X] UNSOLVABLE'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{label}\n{status}', ha='center', va='bottom', fontsize=9,
                color=SUCCESS_GREEN if n >= 3 else FAIL_RED, fontweight='bold')

    # Annotation arrow for DHP boundary
    ax.annotate('', xy=(0.5, 35), xytext=(1.5, 35),
                arrowprops=dict(arrowstyle='<->', color=DUONEURAL_BLUE, lw=2))
    ax.text(1.0, 38, r'$T^*/T_\mathrm{fail} = 3/4 = \mathbf{0.75}$',
            ha='center', fontsize=9.5, color=DUONEURAL_BLUE)

    # ── Right: Mean accuracy & loss ─────────────────────────────────────────
    ax2 = axes[1]
    ax2b = ax2.twinx()

    bars2 = ax2.bar(x - 0.2, [a*100 for a in mean_acc],
                    color=[c + 'aa' for c in colors_bar],
                    edgecolor=colors_bar, linewidth=1.5,
                    width=0.35, label='Mean Accuracy (%)', zorder=3)
    bars3 = ax2b.bar(x + 0.2, mean_loss,
                     color=[DUONEURAL_BLUE]*3, alpha=0.6,
                     edgecolor=DUONEURAL_BLUE, linewidth=1.5,
                     width=0.35, label='Mean Loss (MSE)', zorder=3)

    ax2.axhline(87.5, color=GREY, ls='--', lw=1.2, label='Convergence threshold (87.5%)', zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'T={t}' for t in T_vals], fontsize=11)
    ax2.set_ylabel('Mean Accuracy (%)', fontsize=11)
    ax2b.set_ylabel('Mean Loss (MSE)', fontsize=11, color=DUONEURAL_BLUE)
    ax2.set_ylim(0, 115)
    ax2b.set_ylim(0, 0.22)
    ax2.set_title('Accuracy & Loss Profile', fontsize=12)
    ax2.grid(axis='y', alpha=0.3, zorder=0)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + [bars3], labels1 + labels2, fontsize=9, loc='upper right')

    fig.suptitle(
        'Figure 2: Archon\'s Discrete Trainability Cliff — 2-Qubit Q-RNN, Parity Task',
        fontsize=12, y=1.01)
    plt.tight_layout()
    save(fig, 'fig2_trainability_cliff')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 3: Aura's Coherence Decay
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig3_coherence_decay():
    print("Generating Figure 3: Aura coherence decay...")
    # Data from q_dhp_paper_notes.md
    T_data  = [3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, 78, 83, 88, 93, 98]
    acc     = [100.00, 100.00, 100.00, 100.00, 99.61, 99.22, 97.85, 95.31, 93.75, 93.36,
               88.28,  83.20,  81.64,  78.52,  76.17, 74.61, 73.24, 71.09, 70.31, 69.53]
    margin  = [0.9199, 0.8004, 0.6935, 0.6042, 0.5309, 0.4636, 0.4090, 0.3565, 0.3152, 0.2897,
               0.2549, 0.2248, 0.1986, 0.1975, 0.1758, 0.1562, 0.1562, 0.1445, 0.1289, 0.1373]

    tau_L = 49.63
    M0    = 1.109
    T_fit = np.linspace(3, 100, 300)
    M_fit = M0 * np.exp(-T_fit / tau_L)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: Margin decay ───────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(T_data, margin, color=DUONEURAL_CYAN, s=55, zorder=5, label='Measured $M(T)$')
    ax.plot(T_fit, M_fit, color=DUONEURAL_BLUE, lw=2.2, label=r'Fit: $M_0 e^{-T/\tau_L}$, $\tau_L=49.63$', zorder=4)

    # τ* markers
    tau_star_95 = 38
    tau_star_99 = 28
    ax.axvline(tau_star_95, color=DHP_ORANGE, ls='--', lw=1.8, label=r'$\tau^*(95\%) = 38$', zorder=3)
    ax.axvline(tau_star_99, color=PURPLE, ls='--', lw=1.8, label=r'$\tau^*(99\%) = 28$', zorder=3)
    ax.axvline(tau_L, color=GREY, ls=':', lw=1.5, label=r'$\tau_L = 49.63$', zorder=3)

    # Ratio annotations
    ax.text(tau_star_95 + 1.2, 0.75,
            r'$\tau^*/\tau_L = 38/49.63 = \mathbf{0.766}$',
            fontsize=9.5, color=DHP_ORANGE, rotation=0,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=DHP_ORANGE, alpha=0.9))

    ax.set_xlabel('Sequence Length T', fontsize=11)
    ax.set_ylabel('Normalized Expectation Margin $M(T)$', fontsize=11)
    ax.set_title('Quantum Memory Coherence Decay', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.25, zorder=0)
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 1.05)

    # ── Right: Accuracy decay ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(T_data, acc, 'o-', color=DUONEURAL_CYAN, ms=5, lw=2, zorder=5, label='Generalization Accuracy (%)')
    ax2.axhline(95, color=DHP_ORANGE, ls='--', lw=1.8, label=r'95% threshold → $\tau^*=38$', zorder=3)
    ax2.axhline(99, color=PURPLE, ls='--', lw=1.8, label=r'99% threshold → $\tau^*=28$', zorder=3)
    ax2.axvline(tau_star_95, color=DHP_ORANGE, ls='--', lw=1.5, alpha=0.6, zorder=3)
    ax2.axvline(tau_star_99, color=PURPLE, ls='--', lw=1.5, alpha=0.6, zorder=3)

    # Shade DHP zone
    ax2.axvspan(tau_star_95 * 0.65, tau_star_95 * 0.79 * tau_L/38,
                alpha=0.10, color=DHP_ORANGE, label='DHP window [0.65–0.79]×τ_L')

    ax2.set_xlabel('Sequence Length T', fontsize=11)
    ax2.set_ylabel('Generalization Accuracy (%)', fontsize=11)
    ax2.set_title('Fixed-Weight Generalization vs T', fontsize=12)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(alpha=0.25, zorder=0)
    ax2.set_xlim(0, 102)
    ax2.set_ylim(60, 103)

    # DHP zone label on right
    ax2.text(tau_L * 0.79 + 1, 92,
             f'DHP window\n[0.65–0.79]×τ_L\n=[32.3, 39.2]',
             fontsize=8, color=DHP_ORANGE,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=DHP_ORANGE, alpha=0.85))

    fig.suptitle(
        "Figure 3: Aura's Continuous Coherence Decay — Fixed θ* Evaluated at T=3 to T=100",
        fontsize=12, y=1.01)
    plt.tight_layout()
    save(fig, 'fig3_coherence_decay')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 4: Dual-Methodology Synthesis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig5_synthesis():
    print("Generating Figure 5: Dual-methodology DHP synthesis...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # ── Left: All DHP confirmations timeline ─────────────────────────────────
    ax = axes[0]

    # data: (label, ratio, category, color)
    confirmations = [
        # Classical CTM
        ('CTM v40\n(Papers 4,12)', 0.727, 'Classical CTM', DUONEURAL_BLUE),
        ('LSTM/GRU\n(Paper 5)',    0.718, 'Classical ML',  DUONEURAL_BLUE),
        ('Gradient\nDescent\n(Paper 5)', 0.721, 'Classical ML', DUONEURAL_BLUE),
        ('Biological\n(Paper 5)',  0.715, 'Biology',       SUCCESS_GREEN),
        # Quantum — new from this paper
        ("Archon's\nTrainability\n(This work)", 0.750, 'Quantum', DHP_ORANGE),
        ("Aura's\nCoherence\n(This work)",      0.766, 'Quantum', DHP_ORANGE),
    ]

    ys = list(range(len(confirmations)))
    for i, (label, ratio, cat, col) in enumerate(confirmations):
        ax.barh(i, ratio, height=0.55, color=col + 'cc', edgecolor=col, lw=1.5, zorder=3)
        ax.text(ratio + 0.003, i, f'{ratio:.3f}', va='center', fontsize=10,
                color=col, fontweight='bold')

    # DHP target line
    ax.axvline(0.72, color='black', lw=2, ls='--', label='Universal target: 0.72', zorder=4)
    # DHP window
    ax.axvspan(0.65, 0.79, alpha=0.10, color='black', label='DHP window [0.65–0.79]')

    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in confirmations], fontsize=10)
    ax.set_xlabel(r'DHP Ratio $\tau^*/\tau_L$', fontsize=11)
    ax.set_xlim(0.55, 0.85)
    ax.set_title('DHP Confirmations Across Substrates', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.3, zorder=0)

    # Category brackets
    for y_range, label, col in [
            ((0, 3.5), 'Classical', DUONEURAL_BLUE),
            ((3.5, 5.5), 'Quantum\n(This Work)', DHP_ORANGE)]:
        y1, y2 = y_range
        ax.annotate('', xy=(0.56, y1), xytext=(0.56, y2),
                    arrowprops=dict(arrowstyle='-', color=col, lw=2))
        ax.text(0.557, (y1+y2)/2, label, ha='right', va='center',
                fontsize=9, color=col, fontweight='bold', rotation=90)

    # ── Right: Number line — both quantum measurements vs target ─────────────
    ax2 = axes[1]

    target   = 0.72
    archon_r = 0.750
    aura_r   = 0.766
    lo, hi   = 0.65, 0.79

    # DHP window shading
    ax2.fill_betweenx([-0.4, 0.4], lo, hi, alpha=0.15, color=DHP_ORANGE, zorder=1)
    ax2.axvline(lo, color=DHP_ORANGE, lw=1, ls=':', alpha=0.7, zorder=2)
    ax2.axvline(hi, color=DHP_ORANGE, lw=1, ls=':', alpha=0.7, zorder=2)
    ax2.text(lo, 0.45, '0.65', ha='center', fontsize=9, color=DHP_ORANGE)
    ax2.text(hi, 0.45, '0.79', ha='center', fontsize=9, color=DHP_ORANGE)
    ax2.text((lo+hi)/2, 0.55, 'DHP Window', ha='center', fontsize=10, color=DHP_ORANGE)

    # Target
    ax2.axvline(target, color='black', lw=2.5, zorder=5)
    ax2.text(target, -0.52, f'0.72\n(Classical\nCTM)', ha='center', fontsize=9.5,
             color='black', fontweight='bold')

    # Archon measurement
    ax2.annotate('', xy=(archon_r, 0), xytext=(archon_r, 0.38),
                 arrowprops=dict(arrowstyle='->', color=DUONEURAL_BLUE, lw=2.5))
    ax2.text(archon_r, 0.42,
             f"Archon: {archon_r}\nTrainability cliff\nT₃/T₄ = 3/4",
             ha='center', va='bottom', fontsize=9.5, color=DUONEURAL_BLUE,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=DUONEURAL_BLUE, alpha=0.9))

    # Aura measurement (LL fit — primary)
    aura_ll = 0.727
    ax2.annotate('', xy=(aura_ll, 0), xytext=(aura_ll, -0.28),
                 arrowprops=dict(arrowstyle='<-', color=DUONEURAL_CYAN, lw=2.5))
    ax2.text(aura_ll, -0.32,
             f"Aura (LL): {aura_ll}\nτ*(95%)/τ_L = 36/49.49",
             ha='center', va='top', fontsize=9, color=DUONEURAL_CYAN,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=DUONEURAL_CYAN, alpha=0.9))
    # Aura exp fit (secondary)
    ax2.plot(aura_r, 0, 'o', ms=7, color=DUONEURAL_CYAN, alpha=0.5, zorder=5)
    ax2.text(aura_r, 0.15, f'Aura (exp)\n{aura_r}',
             ha='center', fontsize=8, color=DUONEURAL_CYAN, alpha=0.65)

    # Horizontal axis
    ax2.axhline(0, color='black', lw=1.5, zorder=3)
    ax2.plot(archon_r, 0, 'D', ms=10, color=DUONEURAL_BLUE, zorder=6, label='Archon 0.750')
    ax2.plot(aura_ll, 0, 'D', ms=10, color=DUONEURAL_CYAN, zorder=6, label='Aura (LL) 0.727')
    ax2.plot(target, 0, '*', ms=14, color='black', zorder=7, label='Classical CTM 0.727')

    ax2.set_xlim(0.58, 0.88)
    ax2.set_ylim(-0.75, 0.8)
    ax2.axis('off')
    ax2.set_title('Quantum Probes vs Classical Target (LL fit primary)', fontsize=12, pad=12)
    ax2.legend(fontsize=9.5, loc='lower right',
               bbox_to_anchor=(1.0, -0.12), ncol=3)

    fig.suptitle(
        'Figure 5: DHP Synthesis — Aura LL fit 0.727 matches Classical CTM; Archon trainability 0.750',
        fontsize=12, y=1.01)
    plt.tight_layout()
    save(fig, 'fig5_dhp_synthesis')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 5: Lindblad Noise Robustness
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig4_lindblad():
    print("Generating Figure 4: Lindblad noise robustness...")

    # Data from aura/lindblad_sweep_results.json
    labels    = ['Noiseless', 'Low\n(T1/T2\n=1000dt)', 'Medium\n(=200dt)',
                 'High\n(=100dt)', 'Severe\n(=50dt)']
    tau_L_ll  = [49.49, 46.69, 38.35, 32.72, 34.08]
    tau_star  = [36, 36, 36, 36, 36]
    ratio_ll  = [t/tl for t, tl in zip(tau_star, tau_L_ll)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: τ_L and τ* vs noise ──────────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(labels))
    b1 = ax.bar(x - 0.2, tau_L_ll, width=0.35, color=DUONEURAL_CYAN, alpha=0.85,
                edgecolor=DUONEURAL_CYAN, lw=1.5, label=r'$\tau_L$ (LL fit)', zorder=3)
    b2 = ax.bar(x + 0.2, tau_star, width=0.35, color=DHP_ORANGE, alpha=0.75,
                edgecolor=DHP_ORANGE, lw=1.5, label=r'$\tau^*$ (95% threshold)', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Sequence Steps', fontsize=11)
    ax.set_title(r'$\tau_L$ and $\tau^*$ vs Lindblad Noise', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_ylim(0, 62)

    # Annotate τ_L values
    for b, v in zip(b1, tau_L_ll):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f'{v:.1f}',
                ha='center', va='bottom', fontsize=8.5, color=DUONEURAL_BLUE)

    # ── Right: Ratio vs noise ────────────────────────────────────────────────
    ax2 = axes[1]
    colors_r = [SUCCESS_GREEN if 0.65 <= r <= 0.79 else FAIL_RED for r in ratio_ll]
    bars = ax2.bar(x, ratio_ll, color=[c+'bb' for c in colors_r],
                   edgecolor=colors_r, lw=1.8, zorder=3)

    # DHP window
    ax2.axhspan(0.65, 0.79, alpha=0.12, color=DHP_ORANGE, zorder=1)
    ax2.axhline(0.727, color='black', lw=2, ls='--', label='Classical CTM: 0.727', zorder=4)
    ax2.axhline(1.0, color=FAIL_RED, lw=1.5, ls=':', alpha=0.7, label='Ratio = 1.0 (breakdown)', zorder=4)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel(r'$\tau^*/\tau_L$ (DHP Ratio)', fontsize=11)
    ax2.set_title('DHP Ratio Robustness Under Physical Noise', fontsize=12)
    ax2.legend(fontsize=9.5, loc='upper left')
    ax2.grid(axis='y', alpha=0.3, zorder=0)
    ax2.set_ylim(0, 1.45)

    # Annotate each bar
    for bar, r, col in zip(bars, ratio_ll, colors_r):
        status = 'IN WINDOW' if 0.65 <= r <= 0.79 else 'OUTSIDE'
        ax2.text(bar.get_x() + bar.get_width()/2, r + 0.03,
                 f'{r:.3f}\n{status}', ha='center', va='bottom', fontsize=8.5,
                 color=col, fontweight='bold')

    # DHP window label
    ax2.text(4.55, 0.72, 'DHP\nWindow\n[0.65, 0.79]', ha='right', va='center',
             fontsize=8.5, color=DHP_ORANGE, style='italic',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor=DHP_ORANGE, alpha=0.8))

    fig.suptitle(
        'Figure 4: Lindblad Noise Robustness — DHP Ratio Holds for Physical Low-Noise Regime',
        fontsize=12, y=1.01)
    plt.tight_layout()
    save(fig, 'fig4_lindblad_noise')


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Q-DHP Figure Generator — Paper 25")
    print("  DuoNeural Research Labs, 2026-05-28")
    print("=" * 60)
    fig1_circuit()
    fig2_trainability_cliff()
    fig3_coherence_decay()
    fig4_lindblad()
    fig5_synthesis()
    print("\n✓ All 5 figures generated.")
    print(f"  PDFs: {PDF_DIR}")
    print(f"  PNGs: {PNG_DIR}")
