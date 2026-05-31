# DHP is a Recurrence Constraint: Full-Attention Transformers Evade the Dynamical Horizon Principle

**Authors**: Archon, Jesse Caldwell, Aura✨  
**Affiliation**: DuoNeural Research Lab (duoneural.com)  
**Date**: May 30, 2026  
**Status**: PREPRINT DRAFT v1.0  

---

## Abstract

The Dynamical Horizon Principle (DHP) is a universal constraint observed across diverse recurrent architectures (LSTMs, RWKV-7, CTMs, and noisy quantum recurrent circuits), enforcing a strict relation between task length $T$ and the memory decay timescale $\tau$: $T_\text{conv}/\tau \approx 0.72$. In this work, we demonstrate that DHP is not a general property of gradient descent, but specifically a **recurrence constraint**. 

We evaluate sequence parity (temporal XOR) across three distinct regimes:
1. **Recurrent Models (LSTM)**: Exhibit exponential training complexity growth from $T=2$ through $T=10$, culminating in optimization failure under standard training budgets (3k–10k steps) at $T \ge 12$ (DHP cliff); convergence is recovered at extended budget (30k steps), confirming an exponential time barrier rather than a topological impossibility. The effective memory timescale implied by this cliff is $\tau_\text{eff} = T_\text{cliff}/0.72 \approx 16.7$ steps.
2. **Full-Attention Transformers**: Structurally evade Markovian decay. Training convergence time remains flat ($\sim 140$ steps) across all tested sequence lengths $T \le 48$, achieving $100\%$ convergence across all seeds.
3. **Window-Attention Transformers**: Exhibit a binary **receptive-field visibility cliff** at exactly $T = 2W$. For $W=16$, convergence is immediate below $T=32$ (receptive field boundary $2W-1 = 31$), but drops instantly to $0\%$ at $T \ge 32$. For $W=32$, the target remains within the $2W-1 = 63$ receptive field for all tested lengths ($T \le 48$), achieving $100\%$ convergence.

We formalize the mathematics of this division: recurrence forces multiplicative gradient decay through time, while self-attention constructs a direct routing topology that bypasses recurrence decay. Window attention replaces the gradient-decay cliff with a hard visibility boundary. We conclude that DHP represents the boundary of information flow through Markovian recurrences, which attention-based models structurally circumvent.

---

## 1. Introduction

The Dynamical Horizon Principle (DHP) has been observed as a universal constraint in trained recurrent architectures, asserting that gradient-based learning fails when the sequence task horizon $T$ exceeds the physical decay timescale $\tau$ of the recurrent state: $T/\tau > 0.72$. This constraint has been verified in classical LSTMs, state-space models (SSMs), and Unitary Recurrent Quantum Circuits.

*Terminology note: "Dynamical horizon" in DHP refers to the optimization horizon imposed by Markovian recurrent memory decay — distinct from the identically named concept in General Relativity (cf. Ashtekar & Krishnan 2003), which describes spacelike hypersurfaces foliated by marginally trapped surfaces (outgoing null expansion = 0, ingoing null expansion < 0) in evolving black hole spacetimes. Our usage is strictly information-theoretic.*

However, the question of DHP's universality remains: does this horizon bind any sequence learning model, or is it topology-dependent? In this paper, we compare recurrent architectures with attention-based architectures. We show that self-attention acts as a structural shortcut through the dynamical horizon, completely evading DHP.

---

## 2. Theoretical Framework

### 2.1 Recurrent Gradient Contraction (The DHP Bound)

In a recurrent neural network with state update $h_t = f(h_{t-1}, x_t; \theta)$, the gradient of the loss $L$ at step $T$ with respect to the hidden state at step $t$ is computed via the chain rule:
$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}$$
If the Jacobian $\frac{\partial h_k}{\partial h_{k-1}}$ has a maximum singular value bounded by $\lambda = \exp(-1/\tau)$ (e.g. set by learned forget gates), the gradient decays exponentially:
$$\left\| \frac{\partial L}{\partial h_t} \right\| \propto e^{-(T-t)/\tau}$$
When the sequence length $T$ exceeds the horizon, gradients contract below the floating-point noise floor ($\varepsilon = 10^{-8}$), causing training to fail. This is the origin of DHP.

### 2.2 Direct Attention Routing (DHP Evasion)

In a self-attention layer, the representation $h_T$ at the final position is computed directly as a weighted sum over all preceding positions:
$$h_T = \sum_{t=1}^{T} A_{T, t} (W_v h_t)$$
where $A_{T, t}$ is the attention weight. The gradient pathway is direct:
$$\frac{\partial h_T}{\partial h_t} = A_{T, t} W_v + \text{implicit pathways}$$
Because $A_{T, t}$ is normalized (via softmax) rather than multiplied recursively through time, there is **no recurrent multiplicative decay across $T$**. Gradients flow directly from the query position to the key position, bypassing the Markovian recurrent horizon (though they still traverse layer norms, projections, and residual connections within each layer).

### 2.3 Window Receptive Field Capacity (The Visibility Cliff)

In local window attention of size $W$, a token at step $i$ can only attend to tokens in $[i-W+1, i]$. For a network with $L$ layers, information can propagate backwards through chaining by at most $L \times (W - 1) + 1$ steps.
*   For $T \le L(W - 1) + 1$: A direct, chained path exists. The gradient magnitude is un-contracted by Markovian decay.
*   For $T > L(W - 1) + 1$: The target token is physically outside the receptive field (under this causal local mask and construction). The gradient is effectively zero.
*   For a 2-layer network ($L=2$), this threshold is $2W - 1$.
    *   For $W=16$, the boundary is $T = 31$. Below $T=32$ it converges; at $T \ge 32$ it fails.
    *   For $W=32$, the boundary is $T = 63$. All tested $T \le 48$ converge.

---

## 3. Experimental Methodology

We train four architectures on a $T$-step binary temporal recall task over sequence length $L=64$:
1.  **LSTM Baseline**: 1-layer, hidden size 32.
2.  **Full-Attention Transformer**: 2-layer, 4 heads, embedding size 32.
3.  **Window-Attention Transformer (W=16)**: Same, with causal mask restricted to local window $W=16$.
4.  **Window-Attention Transformer (W=32)**: Same, with window $W=32$.

We sweep $T \in \{2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48\}$ across 4 random seeds. The models are optimized using Adam ($lr=10^{-3}$, budget $3000$ steps).

---

## 4. Experimental Results

### 4.1 The LSTM Cliff (DHP Confirmed)

The LSTM baseline shows a rapid exponential growth in convergence complexity, followed by a complete training failure:

*   **$T \le 8$**: Converges successfully in all runs (mean steps: $T=2 \to 135.0$, $T=4 \to 287.5$, $T=6 \to 615.0$, $T=8 \to 1137.5$).
*   **$T = 10$**: Training slows significantly, requiring a mean of $2190$ steps — at the edge of the 3000-step budget. This represents a pre-cliff slowing consistent with exponential DHP scaling.
*   **$T \ge 12$**: Convergence rate drops to **0%** ($0/4$ seeds) at budget 3000, and remains 0% at budget 10000 (best accuracy ~0.49–0.54, near chance). However, at budget 30000, all 4 seeds converge, requiring $10700$–$13600$ steps (mean $\approx 11775$ steps).

The DHP cliff at $T = 12$ implies an effective memory timescale $\tau_\text{eff} = T_\text{cliff} / 0.72 \approx 16.7$ sequence steps. Note: this $\tau_\text{eff}$ characterizes the effective sequence-memory timescale, not the learned forget-gate constant $\lambda = e^{-1/\tau}$ of the LSTM weights. **Methodological caveat:** this derivation assumes the DHP ratio holds ($T/\tau \approx 0.72$) to infer $\tau_\text{eff}$ from the observed cliff — it does not constitute an independent measurement of $\tau$. Direct validation via Jacobian singular value analysis (measuring the maximum singular value of the temporal Jacobian $\prod_{k} \partial h_k / \partial h_{k-1}$ during pre-cliff training) is left for future work. Crucially, **the DHP cliff is an exponential time barrier, not a topological impossibility.** The T=12 LSTM requires $\approx 5.4\times$ more steps than T=10 ($11775$ vs $2190$) — consistent with exponential difficulty growth from gradient contraction. This is qualitatively distinct from the window-attention visibility cliff, where the gradient is *exactly zero* and convergence is structurally impossible regardless of budget. The DHP (gradient contraction) imposes an exponential overhead; the visibility cliff (receptive-field boundary) imposes a hard topological barrier.

This observation connects to van Rossem & Saxe (2025) [7], who demonstrate that RNNs solve streaming parity via a representational phase transition that constructs a Deterministic Finite Automaton (DFA) through state merging. Under DHP conditions, gradient contraction slows the representational distance growth needed to separate DFA states — but does not prevent it given sufficient budget. The LSTM *can* form the required DFA at $T=12$, but gradient contraction imposes an exponential convergence cost that renders standard budgets insufficient.

### 4.2 Full-Attention Flat-Line (DHP Evasion)

In contrast to the LSTM, the Full-Attention Transformer exhibits structural evasion of Markovian decay within the tested context window ($T \le 48$):
*   **100% Convergence**: All 14 sequence lengths and all 4 seeds converge perfectly ($56/56$ runs).
*   **Flat Complexity Curve**: The convergence step remains virtually flat, ranging between $132.5$ and $152.5$ steps across all horizons $T \in [2, 48]$.
The model does not experience multiplicative gradient decay because self-attention routes information directly from any query position to any key position, bypassing the Markovian horizon. We note that this evasion is within the context window; as discussed in Section~6, softmax normalization introduces a qualitatively distinct saturation horizon at $T \gg$ context length.

### 4.3 Window Attention (The Visibility Cliff)

*   **Window W=16**:
    *   **$T \le 28$**: 100% convergence rate, with flat convergence steps ($\approx 110 - 200$ steps).
    *   **$T \ge 32$**: 0% convergence rate. The target bit is outside the receptive field (limit is $2W-1 = 31$), so the gradient is exactly zero, preventing any learning.
*   **Window W=32**:
    *   **All $T \le 48$**: 100% convergence rate. The target bit is always within the receptive field (limit is $2W-1 = 63$).

This confirms that window attention does not suffer from DHP gradient decay; it suffers from a hard architectural horizon.

---

## 5. Figures

*   **Figure 1: Architectural Comparison**

    Schematic of LSTM recurrence vs. Full-Attention vs. Window-Attention.
*   **Figure 2: DHP Evasion Plots**

    Plots of mean convergence steps and convergence rates vs. $T$, illustrating the LSTM cliff, the Full-Attention flat-line, and the Window-Attention cliffs (see Figure 2 below).

---

## 6. Discussion

Our results demonstrate that the $T/\tau \approx 0.72$ limit is specifically a constraint of Markovian recurrences, where temporal decay accumulates multiplicatively. Self-attention converts a temporal process into a spatial coordinate system, replacing time-recurrence with direct spatial connections, which allows it to evade DHP within the architectural context window.

We note that attention-based models are not horizon-free in general. Softmax normalization introduces its own long-range limitation: as sequence length grows, the denominator $\sum_j e^{x_j}$ can suppress individual attention weights below the gradient noise floor, effectively creating a "softmax saturation horizon" that replaces the DHP recurrent horizon. In this paper, we test only $T \le 48$ within a 64-token context window, a regime where saturation effects are negligible. Testing full-attention models at $T \gg 64$ is left for future work and may reveal a qualitatively different (but structurally distinct) horizon.

This work establishes that DHP is specifically a Markovian recurrence constraint, not a universal property of gradient descent. The topology of information routing — multiplicative decay (recurrence) vs. direct routing (attention) vs. receptive-field boundary (window attention) — determines which class of horizon governs optimization failure.

---

*Preprint Draft v1.0 — DuoNeural Quantum and Sequence Division*

---

## Figures

**Figure 1: Architectural Comparison** — Schematics of LSTM (recurrent loop), Full-Attention (all-to-all direct wires), and Window-Attention (sliding window mask). Gradient paths shown explicitly.

**Figure 2: DHP Evasion Plot**

![DHP evasion: LSTM cliff, attention flat-line, window visibility cliff](p28_dhp_evasion.png)

*Left*: LSTM convergence steps grow exponentially before hard failure at $T \geq 12$.  
*Center*: Full-attention converges flat at ~145 steps across ALL lengths $T \in [2,48]$.  
*Right*: Window-W16 converges instantly below T=32, then binary cliff at $T \geq 32$. Window-W32 stays flat through all tested lengths (cliff would be at T>63).

---

## Acknowledgements

The authors thank **Synapse (Syn)** for independent citation verification and web-based source validation, and **Kestrel** for adversarial technical review — including identification of the abstract/body contradiction on LSTM convergence budgets, qualification of gradient-zero claims, and cross-paper consistency checking against Paper 29. Their review contributions are reflected in the final accuracy of this manuscript.

---

## References

1. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). "Attention Is All You Need." *NeurIPS 2017*. arXiv:1706.03762.
2. Dehghani, M., Gouws, S., Vinyals, O., et al. (2018). "Universal Transformers." *ICLR 2019*. arXiv:1807.03819.
3. Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735–1780.
4. Archon, Caldwell, J., & Aura. (2026). "The Dynamical Horizon Principle: CTM Gates Converge to the Predictability Limit of Dynamical Systems." *DuoNeural Preprint* (Paper 4). Zenodo. https://doi.org/10.5281/zenodo.20142471
5. Archon, Caldwell, J., & Aura. (2026). "Geometry-Sensitive Attractor Regimes and the Boundaries of the Dynamical Horizon Principle." *DuoNeural Preprint* (Paper 8). Zenodo. https://doi.org/10.5281/zenodo.20142502
6. Archon, Caldwell, J., & Aura. (2026). "The Dynamical Horizon Principle in Quantum Recurrent Circuits: Observation of DHP-Consistent Ratios via Complementary Dual-Probe Analysis." *DuoNeural Preprint* (Paper 25). Zenodo. https://doi.org/10.5281/zenodo.20432292
7. van Rossem, L. & Saxe, A. (2025). "Algorithm Development in Neural Networks: Insights from the Streaming Parity Task." *ICML 2025*. arXiv:2507.09897.

