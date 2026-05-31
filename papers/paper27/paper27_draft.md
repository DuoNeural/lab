# The Geometry of Quantum Recurrent Landscapes: Unital Regularization and Optimizer Invariance

**Authors**: Archon, Jesse Caldwell, Aura✨  
**Affiliation**: DuoNeural Research Lab (duoneural.com)  
**Date**: May 30, 2026  
**Status**: PREPRINT DRAFT v1.0  

---

## Abstract

We characterize the geometric and algebraic conditions governing noise-induced landscape regularization in Recurrent Quantum Circuits (Q-RNNs). In quantum recurrent models training on sequence parity (temporal XOR), gradient optimization typically encounters a severe topological trap (the parity plateau at accuracy $\approx 0.72–0.75$). While physical environmental noise is conventionally viewed as detrimental to quantum information, we provide empirical evidence that the primary boundary separating regularizing and non-regularizing open-system channels is **unitality**. 

Specifically:
1. **Unital Channels** (e.g., Pauli depolarizing, phase-flip, bit-flip, and phase damping) contract the Bloch sphere symmetrically, preserving the loss landscape's up/down parity symmetry and enabling optimizers to escape the topological traps.
2. **Non-Unital Channels** (e.g., amplitude damping) shift the center of the Bloch sphere; large non-unital origin drift severely disrupts the optimization landscape by breaking the loss landscape's up/down parity symmetry.
3. We map this boundary empirically by conducting a **Generalized Amplitude Damping (GAD)** sweep over the thermal parameter $\beta \in [0.0, 1.0]$ across varying damping rates $\gamma \in \{0.25, 0.35, 0.50, 0.75\}$ (Sweep 1), and a state-initialization asymmetry sweep (Sweep 2). We show that convergence is restored within a near-unital basin (e.g., $\beta \in [0.42, 0.56]$ at $\gamma=0.50$). We show that the slight asymmetry of this basin (which is invariant to the initial state of the memory qubit) is consistent with a structural loop asymmetry caused by the recurrent input qubit reset to $|0\rangle$. We also map the parametric scaling of this basin, showing that the critical displacement boundary $|v_{z, \text{crit}}|$ shrinks as $\gamma$ increases due to the rapid contraction of the available state space volume.
4. We characterize the fine-grained transition of the bit-flip channel, revealing a sharp **degeneracy hole** at exactly $p=0.50$ where the Bloch sphere contracts to a 1D line.
5. We characterize an **empirical optimizer-class robustness pattern** at this finite experimental depth (2-qubit, $T=3$): normalized direction-preserving optimizers (Adam, signSGD, Muon, Normalized GD) achieve 6/6 convergence under unital noise, whereas non-normalized gradient descent (SGD, Heavy Ball) fails (2/6) due to recurrent gradient magnitude contraction. 

We discuss the analogy between this regularization structure and Quantum Error Correction (QEC), noting that the symmetries protecting parity in stabilizer codes appear to play a parallel role in protecting parity classification landscapes under unital noise. All results are from 2-qubit Q-RNN simulations on $T=3$ parity; generalization to larger circuits and longer sequences is a target for future work.

---

## 1. Introduction

Recurrent quantum circuits (Q-RNNs) represent a promising frontier for sequential quantum machine learning. However, as established in recent works, Q-RNNs training on sequence parity tasks (e.g., temporal XOR) suffer from a severe **quantum parity trap**—a stable, sub-optimal attractor basin that binds gradient-based optimizers to a partial parity plateau (accuracy $\approx 0.75$). 

Surprisingly, the introduction of environmental noise acts as a landscape regularizer, erasing this trap and enabling 100% convergence. This paper answers the fundamental geometric question: *Why do certain noise channels erase the trap while others fail?* We provide empirical and theoretical evidence that **unitality** is the primary algebraic boundary of this regularization, with the exact basin boundary governed by the channel's damping rate and a structural asymmetry arising from the recurrent input reset mechanism.

*Terminology note: This work is part of the Dynamical Horizon Principle (DHP) research series. "Dynamical horizon" in DHP refers to the optimization horizon imposed by Markovian recurrent memory decay — distinct from the identically named concept in General Relativity (cf. Ashtekar & Krishnan 2003), which describes spacelike hypersurfaces foliated by marginally trapped surfaces (outgoing null expansion = 0, ingoing null expansion < 0) in evolving black hole spacetimes. Our usage is strictly information-theoretic.*

---

## 2. Theoretical Framework

### 2.1 Unital vs. Non-Unital Bloch Dynamics

A quantum channel $\mathcal{E}$ is unital if it maps the identity operator to itself: $\mathcal{E}(I) = I$. In the Bloch vector representation, a general qubit state is written as $\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma})$, where $\vec{r} = (x, y, z)^T$ is the Bloch vector and $\vec{\sigma} = (\sigma_x, \sigma_y, \sigma_z)$ are the Pauli matrices. The action of a quantum channel $\mathcal{E}$ on the Bloch vector is an affine map:
$$\vec{r} \to \Lambda \vec{r} + \vec{v}$$
where $\Lambda$ is a $3\times3$ real matrix and $\vec{v}$ is a displacement vector.

*   **Unital Channels ($\vec{v} = 0$)**: Symmetrical contraction occurs:
    $$(x, y, z)^T \to (\Lambda_x x, \Lambda_y y, \Lambda_z z)^T$$
    This preserves the origin of the Bloch sphere. Crucially, it preserves the reflection symmetry of the loss landscape with respect to state inversion (up/down parity symmetry). The sign of the gradients is preserved, enabling optimizers to escape the topological traps.
*   **Non-Unital Channels ($\vec{v} \neq 0$)**: Shift the origin of the Bloch sphere:
    $$(x, y, z)^T \to (\Lambda_x x, \Lambda_y y, \Lambda_z z + v_z)^T$$
    This introduces a systematic bias (origin shift), which breaks the parity-preservation property of the recurrent landscape. The loss landscape deforms asymmetrically, trapping the optimizer permanently.

### 2.2 Mathematical Connection to Quantum Error Correction (QEC)

The algebraic division between unital and non-unital channels maps directly to the structure of Quantum Error Correction (QEC):
*   **Stabilizer Codes**: Standard stabilizer codes (Shor, Steane, surface codes) are designed to correct unital Pauli errors. They exploit the fact that error syndromes commute or anti-commute with stabilizers, maintaining parity.
*   **Dissipation and Drift**: Non-unital channels represent energy relaxation (amplitude damping). They deform the code space continuously, requiring specialized codes (like the 4-qubit amplitude-damping code) to correct the asymmetric drift.
*   **The Parity Analogy**: The same algebraic symmetries that allow stabilizer codes to protect parity also appear to protect the convergence of parity classification landscapes under unital noise. Formalizing this correspondence remains an open question.

---

## 3. Experimental Methodology

We simulate a 2-qubit Q-RNN on sequence length $T=3$ parity. The model is trained using local exact density-matrix simulations on CPU.
*   **Architecture**: Encode-after-recurrent unitary configuration with 4 trainable parameters, mapping sequences to final expectation values.
*   **Optimization**: 6 random initializations (seeds), trained for 600 steps.
*   **Generalized Amplitude Damping (GAD)**:
    We model thermal interaction with the environment using the GAD channel, defined by Kraus operators:
    $$K_0 = \sqrt{1-p} \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \sqrt{1-p} \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$
    $$K_2 = \sqrt{p} \begin{pmatrix} \sqrt{1-\gamma} & 0 \\ 0 & 1 \end{pmatrix}, \quad K_3 = \sqrt{p} \begin{pmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{pmatrix}$$
    where $\gamma$ is the damping rate and $p$ represents the thermal population parameter (which we denote as $\beta$ in our sweeps).
    - When $\beta = 0.50$, the channel is unital.
    - When $\beta \neq 0.50$, the channel is non-unital.
    **Primary sweep**: $\beta \in [0.35, 0.65]$ in steps of $0.01$ at fixed $\gamma = 0.50$; coarser grid $\beta \in [0.0, 1.0]$ at step $0.05$ for outer regions.  
    **$\gamma$-Scaling sweep**: $\gamma \in \{0.25, 0.35, 0.50, 0.75\}$ across $\beta \in [0.0, 1.0]$ to characterize the basin boundary as a function of damping rate.  
    **Initialization asymmetry sweep**: The primary $\beta$-sweep repeated with the memory qubit $q_1$ initialized in $|1\rangle$ instead of $|0\rangle$ (all other parameters held fixed) to isolate whether the basin asymmetry arises from state preparation or from the recurrent loop structure.
*   **Bit-Flip Fine Grid**:
    We swept the bit-flip probability $p \in [0.43, 0.57]$ in steps of $0.01$ to observe the $1-2p = 0$ degeneracy hole.

---

## 4. Experimental Results

### 4.1 The Unitality Restoration Boundary (GAD Sweep)

The GAD sweep over thermal parameter $\beta$ at $\gamma = 0.50$ reveals a **near-unital convergence basin** spanning $\beta \in [0.42, 0.56]$:

**Table 1: GAD Sweep Results — Convergence Basin Under Generalized Amplitude Damping ($\gamma = 0.50$).**

| Thermal Parameter $\beta$ | Unital? | $v_z = \gamma(2\beta-1)$ | Converged Seeds | Mean Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| $0.00$ | No | $-0.50$ | 0 / 6 | 0.471 |
| $0.10$ to $0.40$ | No | $-0.40$ to $-0.10$ | 0 / 6 | 0.746 (trapped) |
| **$0.42$ to $0.48$** | **No** | **$-0.08$ to $-0.02$** | **6 / 6** | **1.000** |
| **$0.50$** | **Yes (Exact)** | **$0.00$** | **6 / 6** | **1.000** |
| **$0.52$ to $0.56$** | **No** | **$+0.02$ to $+0.06$** | **6 / 6** | **1.000** |
| $0.58$ to $0.60$ | No | $+0.08$ to $+0.10$ | 0 / 6 | 0.750 (trapped) |
| $0.70$ to $1.00$ | No | $+0.20$ to $+0.50$ | 0 / 6 | 0.529 (failed) |

*   **Near-Unital Basin**: 6/6 seeds converge to 100% accuracy. This includes channels that are technically non-unital ($\beta \neq 0.50$), demonstrating that exact unitality is not required — proximity to the unital fixed point ($v_z \approx 0$) is the governing criterion.
*   **Asymmetry Analysis**: The basin is asymmetric: the lower boundary $v_z = -0.08$ ($\beta=0.42$) converges perfectly, while the upper boundary fails to converge at $v_z = +0.08$ ($\beta=0.58$), succeeding only up to $v_z = +0.06$ ($\beta=0.56$). To test whether this reflects a state-initialization artifact (at $t=0$), we repeated the sweep with the memory qubit $q_1$ initialized in $|1\rangle$ instead of $|0\rangle$. The results were identical (basin remained $\beta \in [0.42, 0.56]$), consistent with a **recurrent structural loop asymmetry**. Because the input qubit $q_0$ is reset to $|0\rangle$ at each step, the unitary coupling continuously shifts the system state towards $|0\rangle$. Thermal drift toward $|0\rangle$ ($\beta < 0.50$) reinforces this reset-drift, while drift toward $|1\rangle$ ($\beta > 0.50$) opposes it, generating the asymmetric landscape deformation. We note that this origin drift under non-unital amplitude damping is properly characterized as a Noise-Induced Limit Set (NILS) [8], as distinct from the NIBP phenomenon observed under purely unital channels.

### 4.2 Damping Rate scaling ($\gamma$-Scaling Sweep)

To determine how the basin boundary scales with damping rate $\gamma$, we swept $\gamma \in \{0.25, 0.35, 0.50, 0.75\}$ across $\beta \in [0.0, 1.0]$. The results show that the critical displacement boundary $|v_{z, \text{crit}}|$ is not a constant, but shrinks as $\gamma$ increases:
- **$\gamma = 0.25$**: Basin is $\beta \in [0.15, 1.00] \implies v_z \in [-0.175, +0.250]$.
- **$\gamma = 0.35$**: Basin is $\beta \in [0.35, 0.80] \implies v_z \in [-0.105, +0.210]$.
- **$\gamma = 0.50$**: Basin is $\beta \in [0.42, 0.56] \implies v_z \in [-0.080, +0.060]$.
- **$\gamma = 0.75$**: Near-complete collapse; 6/6 convergence only at $\beta = 0.50$ (exact unital point), with marginal convergence in the narrow window $v_z \in (-0.075, +0.075)$ ($\beta \in [0.45, 0.55]$). Any displacement beyond $|v_z| \ge 0.075$ fails completely.

This behavior indicates that higher damping rates $\gamma$ contract the Bloch sphere more rapidly ($1-\gamma$ shrinks), reducing the state space volume and increasing the optimizer's sensitivity to non-unital origin drift. At $\gamma \ge 0.75$, the contraction is so severe that even a displacement $|v_z| \ge 0.075$ completely breaks the recurrent parity landscape, preventing convergence.

### 4.3 The Bit-Flip Degeneracy Hole

The fine-grained bit-flip sweep around $p=0.50$ reveals a sharp drop in trainability:

**Table 2: Bit-Flip Fine Sweep — Degeneracy Hole at $p = 0.50$.**

| Bit-Flip Probability $p$ | Bloch Contraction ($1-2p$) | Converged Seeds (out of 6) | Mean Accuracy |
| :---: | :---: | :---: | :---: |
| $0.43$ to $0.48$ | $0.14$ to $0.04$ | 6 / 6 | 1.000 |
| $0.49$ | $0.02$ | **3 / 6** | 0.754 (transition) |
| **$0.50$** | **$0$** | **0 / 6** | **0.471 (failed)** |
| $0.51$ to $0.57$ | $-0.02$ to $-0.14$ | 6 / 6 | 1.000 |

At $p=0.50$, the contraction factor $1-2p = 0$. Under the bit-flip channel, the Bloch sphere collapses into a 1D line along the X-axis (y and z components $\to$ 0), while the X-component is perfectly preserved. However, since parity is read via Z-measurement, this X-preservation is irrelevant — the relevant signal is annihilated. The result is a **measure-zero degeneracy**: any infinitesimal deviation from $p=0.50$ restores full convergence (6/6), whereas $p = 0.49$ shows a brief transition zone (3/6) consistent with near-zero but non-vanishing gradient signal ($1-2p = 0.02$).

### 4.4 Optimizer Class Invariance

We evaluated the robustness of the unital regularization across 7 distinct classical optimizers:
*   **Normalized Direction-Preserving (Adam, signSGD, Muon, Normalized GD)**: Achieve $6/6$ convergence under unital noise. They are invariant because they normalize updates, relying on gradient sign/direction, which remains stable.
*   **Non-Normalized (SGD, Heavy Ball)**: Fail ($2/6$ convergence) because recurrent gradient magnitude contracts exponentially, sinking below the floating-point noise floor.
*   **Sophia-H (Curvature Preconditioned)**: Fails under heavy noise ($4/6$ convergence) because diagonal Hessian estimations become highly stochastic and unstable under Lindbladian perturbations.

This compatibility with normalized optimizers appears to stand in contrast to the work of Wang et al. (2021), who prove that unital noise induces barren plateaus (Noise-Induced Barren Plateaus, or NIBPs) where the gradient magnitude vanishes exponentially with circuit depth. However, our ablation demonstrates that normalized, direction-preserving optimizers (Adam, signSGD, Muon) successfully bypass this limitation. Because unital channels contract the Bloch sphere symmetrically, they preserve the relative orientation and sign of the loss gradients even as their magnitudes decay exponentially. Normalized optimizers scale updates using only this direction or sign information, remaining viable in regions where unnormalized gradient descent (SGD, Heavy Ball) stalls below the numerical noise floor.

The Muon optimizer [6] provides a particularly rigorous instantiation of this principle. Rather than simply normalizing gradient magnitudes, Muon applies a Newton-Schulz iteration to the SGD momentum buffer, approximately orthogonalizing the update matrix and projecting it onto the Stiefel manifold — driving all non-zero singular values toward 1. This process preserves usable update scale even when NIBP suppresses gradient magnitudes: while unital noise symmetrically contracts singular values of the gradient matrix, Muon's orthogonalization keeps the effective update scale above the numerical noise floor, allowing the optimizer to navigate the parity landscape where magnitude-dependent methods stall. The invariance result observed here is thus not merely empirical — it follows from the geometric compatibility between Muon's spectral normalization and the symmetric Bloch sphere contraction imposed by unital noise.

**Scope and floating-point precision horizon.** The optimizer class invariance established here holds at our experimental sequence depth ($T=3$, 2-qubit circuit, 600 training steps). This result has an important finite-depth scope: as temporal circuit depth increases, NIBP-induced gradient magnitude decay eventually reaches the hardware's floating-point precision floor ($\varepsilon \approx 10^{-8}$ for float32). Below this floor, normalized optimizers (e.g., Adam with its running variance denominator) divide by accumulated noise rather than genuine gradient signal, amplifying numerical artifacts rather than useful directional information. At this floating-point precision horizon — which scales with temporal depth and circuit size beyond our $T=3$ experiments — the $6/6$ convergence of normalized optimizers is expected to collapse, recovering the full NIBP failure mode. Characterizing this depth-dependent transition precisely is left for future work.

---

## 5. Figures

*   **Figure 1: Bloch Sphere Contraction Geometries**

    Visualizes the difference between unital (symmetrical contraction to the origin) and non-unital (offset contraction with origin drift) channels.
*   **Figure 2: GAD Sweep Results (Restoring Unitality)**

    Plots converged seeds vs. $\beta$ at $\gamma=0.50$, showing the convergence dome centered at $\beta = 0.50$.
*   **Figure 3: Bit-Flip Degeneracy Hole**

    Plots convergence rate vs. $p$, showcasing the sharp drop to 0% at exactly $p=0.50$.
*   **Figure 4: Quantum Asymmetry Analysis**

    Plots convergence rate vs. $\beta$ for memory qubit initializations $|0\rangle$ (blue) and $|1\rangle$ (magenta, dashed), showing the identical asymmetric basin and confirming structural loop origin.
*   **Figure 5: Gamma-Scaling Sweeps**

    Plots GAD convergence basins under varying damping rates $\gamma \in \{0.25, 0.35, 0.50, 0.75\}$, showing basin contraction under high noise.
*   **Figure 6: Optimizer Characterization**

    A bar chart illustrating convergence rates under noise for all 7 optimizers.

---

## 6. Discussion & Implications

*   **Unital Landscape Regularization**: Unital noise channels smooth the loss landscape without introducing origin drift, simplifying the optimization geometry.
*   **Sign Preservation vs. Magnitude Decay**: Normalized optimizers are highly robust to noise because unital noise shrinks gradient magnitudes but preserves their sign.
*   **Hardware Calibration**: Our findings suggest that physical QPUs (like Rigetti Aspen-M-3) require careful calibration of non-unital decay ($T_1$), while dephasing noise ($T_2$, unital) can actually act as a natural regularizer for training recurrent quantum models.

---

*Preprint Draft v1.0 — DuoNeural Quantum and Sequence Division*

---

**Figure 1: Bloch Sphere Contraction Geometry**

![Bloch sphere geometry — unital vs non-unital contraction](p27_fig1_bloch_geometry.png)

**Figure 2: GAD Sweep — Near-Unital Convergence Basin**

*Convergence rate vs. thermal parameter $\beta$ showing GAD sweep results centering at the unital boundary.*

![GAD sweep results showing near-unital basin](p27_gad_sweep.png)

**Figure 3: Bit-Flip Degeneracy Hole**

*Singular failure at p=0.50 within the otherwise fully convergent unital family.*

![Bit-flip fine sweep showing degeneracy hole at p=0.50](p27_bitflip_hole.png)

**Figure 4: Quantum Asymmetry Analysis**

*Convergence rate vs. thermal parameter $\beta$ for memory qubit initializations $|0\rangle$ (blue) and $|1\rangle$ (magenta, dashed), showing perfect overlap and confirming structural loop asymmetry.*

![Quantum Asymmetry Analysis](p27_asymmetry_analysis.png)

**Figure 5: Gamma-Scaling Sweeps**

*GAD convergence basins under damping rates $\gamma = 0.25, 0.35, 0.50, 0.75$, showing basin contraction as the noise level increases.*

![Gamma-Scaling Sweeps](p27_gamma_scaling.png)

**Figure 6: Optimizer Class Invariance**

*Normalized direction-preserving optimizers (Adam, signSGD, Muon, Normalized GD): 6/6. Non-normalized (SGD, Heavy Ball): 2/6. Sophia-H: 4/6.*

![Optimizer class invariance bar chart](p27_fig4_optimizer_invariance.png)

---

## Appendix A: GAD Unitality Verification

For GAD Kraus operators, the unitality condition $\sum K_i K_i^\dagger = I$ yields:
$$\sum K_i K_i^\dagger = \begin{pmatrix} 1-\gamma+2\beta\gamma & 0 \\ 0 & 1+\gamma-2\beta\gamma \end{pmatrix}$$
This equals $I$ if and only if $2\beta\gamma = \gamma$, which for any non-trivial damping rate $\gamma \neq 0$ implies **$\beta = 0.5$ is the unique unital point**.

The $Z$-component affine map is $z \to (1-\gamma)z + \gamma(2\beta-1)$, with the non-unital origin drift displacement $v_z = \gamma(2\beta-1)$ and the attractor fixed point $z^* = 2\beta-1$.

---

## Appendix B: The Thermodynamic Capacity Limit and $e/(e+1)$ Convergence Boundary

The formal proof of the thermodynamic capacity limit and the $e/(e+1)$ universal constant is presented in a companion theoretical paper (Paper 29).

---

## Acknowledgements

The authors thank **Synapse (Syn)** for independent citation verification and external source cross-checking during the review process, and **Kestrel** for adversarial technical review — including identification of framing overreach, citation integrity issues, and cross-paper consistency checks. Their contributions materially improved the accuracy and rigor of this manuscript. All four members of the DuoNeural research team contributed to the review pipeline for this work.

---

## References

1. Archon, Caldwell, J., & Aura. (2026). "The Dynamical Horizon Principle in Quantum Recurrent Circuits: Observation of DHP-Consistent Ratios via Complementary Dual-Probe Analysis." *DuoNeural Preprint* (Paper 25). Zenodo. https://doi.org/10.5281/zenodo.20432292
2. Archon, Caldwell, J., & Aura. (2026). "The Quantum Parity Trap: Asymptotic Decoherence Immunity Evades the Dynamical Horizon Principle in Temporal XOR Classification." *DuoNeural Preprint* (Paper 26). Zenodo. https://doi.org/10.5281/zenodo.20451102
3. Wang, S., Fontana, E., Cerezo, M., Sharma, K., Sone, A., Cincio, L., & Coles, P.J. (2021). "Noise-induced barren plateaus in variational quantum algorithms." *Nature Communications*, 12, 6961. arXiv:2007.14384.
4. Lindblad, G. (1976). "On the generators of quantum dynamical semigroups." *Communications in Mathematical Physics*, 48(2), 119–130.
5. Nielsen, M. & Chuang, I. (2000). *Quantum Computation and Quantum Information*. Cambridge University Press.
6. Jordan, K., Newhouse, L., Bernstein, J. et al. (2024). "Muon: An optimizer for hidden layers in neural networks." Blog post and GitHub repository. Available at: https://kellerjordan.github.io/posts/muon/
7. Liu, H. et al. (2023). "Sophia: A scalable stochastic second-order optimizer for language model pre-training." arXiv:2305.14342.
8. Singkanipa, P., & Lidar, D. A. (2025). "Beyond unital noise in variational quantum algorithms: noise-induced barren plateaus and limit sets." *Quantum*, 9, 1617. https://doi.org/10.22331/q-2025-01-30-1617 (arXiv:2402.08721)
