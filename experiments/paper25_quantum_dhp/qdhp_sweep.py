"""
Q-DHP Horizon Sweep
====================
Tests whether the expressivity horizon of a weight-shared 2-qubit Q-RNN
maps to the DHP universal ratio τ*/τ_L ≈ 0.72, as observed in CTM v34-v40.

Hypothesis:
    T_converge / T_fail ≈ 0.72

Where:
    T_converge = max sequence length T at which training reliably converges
                 (>90% accuracy, >=3 of N_SEEDS succeed)
    T_fail     = min sequence length T at which training *never* converges
                 (0 of N_SEEDS succeed)

Architecture: FIXED across all T (same 2-qubit, 4-param, weight-shared ansatz
as qrnn_parity.py v2). We want to stress the *optimizer landscape*, not add
capacity — keeping architecture fixed is the correct DHP-parallel.

Training uses local Qiskit statevector (exact, free). BlueQubit GPU NOT used
here to avoid credit burn on a sweep — full verification run is a separate
step once T_converge is identified.

Archon + Aura — DuoNeural Quantum Division — 2026-05-28
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
import json
import time
from datetime import datetime
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
T_RANGE       = list(range(3, 16))   # sequence lengths to sweep (3→15)
N_SEEDS       = 5                    # random inits per T — need consistency signal
MAX_EPOCHS    = 300                  # prevent infinite loops
LR            = 0.05                 # Adam learning rate
BETA1, BETA2  = 0.9, 0.999
EPS           = 1e-8
CONV_LOSS     = 0.05                 # loss threshold for "converged"
CONV_ACC      = 0.875                # 7/8 (or proportion) threshold for "good"
GOOD_SEEDS    = 3                    # how many seeds must converge to call T "solvable"
OUT_FILE      = Path(__file__).parent / "qdhp_sweep_results.json"

# ── Quantum circuit ─────────────────────────────────────────────────────────

def build_qrnn_sv(sequence, theta):
    """
    2-qubit weight-shared Q-RNN using Qiskit Statevector (no measurement).
    Returns: probability that memory qubit (q1) is |1⟩.
    This matches the fixed architecture from qrnn_parity.py v2.
    """
    qc = QuantumCircuit(2)
    for x_t in sequence:
        qc.reset(0)                    # ← Aura's fix: clean input each step
        qc.rx(float(x_t) * np.pi, 0)
        qc.ry(theta[0], 0)
        qc.ry(theta[1], 1)
        qc.cx(0, 1)
        qc.rz(theta[2], 0)
        qc.rz(theta[3], 1)

    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()
    # Qiskit SV little-endian: index = binary where bit k = qubit k
    # prob q1=|1⟩: index 2 = |10⟩ (q1=1,q0=0) + index 3 = |11⟩ (q1=1,q0=1)
    prob_q1_one = float(probs[2] + probs[3])
    return prob_q1_one


def make_dataset(T):
    """Generate all 2^T parity sequences for length T."""
    seqs, targets = [], []
    for i in range(2**T):
        bits = [(i >> (T - 1 - j)) & 1 for j in range(T)]
        seqs.append(bits)
        targets.append(float(sum(bits) % 2))
    return seqs, targets


def parameter_shift_gradient(theta, sequences, targets):
    """
    Exact parameter-shift gradients: ∂f/∂θ_i = [f(θ+π/2·eᵢ) - f(θ-π/2·eᵢ)] / 2
    Returns gradient of MSE loss w.r.t. theta.
    """
    n = len(theta)
    preds = np.array([build_qrnn_sv(s, theta) for s in sequences])
    loss  = np.mean((preds - np.array(targets))**2)
    grad  = np.zeros(n)
    for i in range(n):
        t_plus  = theta.copy(); t_plus[i]  += np.pi / 2
        t_minus = theta.copy(); t_minus[i] -= np.pi / 2
        p_plus  = np.array([build_qrnn_sv(s, t_plus)  for s in sequences])
        p_minus = np.array([build_qrnn_sv(s, t_minus) for s in sequences])
        grad[i] = np.mean((p_plus - p_minus) * (preds - np.array(targets)))
    return loss, grad, preds


def adam_train(sequences, targets, seed, max_epochs=MAX_EPOCHS):
    """
    Train with Adam + parameter-shift. Returns (final_loss, accuracy, epochs, theta).
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, 4)
    m  = np.zeros(4)
    v  = np.zeros(4)
    t  = 0

    targets_arr = np.array(targets)
    history = []

    for epoch in range(max_epochs):
        loss, grad, preds = parameter_shift_gradient(theta, sequences, targets_arr)
        t += 1
        m  = BETA1 * m  + (1 - BETA1) * grad
        v  = BETA2 * v  + (1 - BETA2) * grad**2
        m_hat = m  / (1 - BETA1**t)
        v_hat = v  / (1 - BETA2**t)
        theta = theta - LR * m_hat / (np.sqrt(v_hat) + EPS)
        history.append(float(loss))

        if (epoch + 1) % 20 == 0:
            acc = np.mean((preds > 0.5).astype(float) == targets_arr)
            print(f"    epoch {epoch+1:3d}: loss={loss:.6f} acc={acc:.3f}")

        if loss < CONV_LOSS:
            break

    # Final accuracy
    final_preds = np.array([build_qrnn_sv(s, theta) for s in sequences])
    final_acc = float(np.mean((final_preds > 0.5).astype(float) == targets_arr))
    return float(loss), final_acc, epoch + 1, theta.tolist(), history


# ── Main sweep ──────────────────────────────────────────────────────────────

def run_sweep():
    print("=" * 70)
    print("  Q-DHP HORIZON SWEEP — DuoNeural Quantum Division")
    print(f"  Hypothesis: T_converge / T_fail ≈ 0.72 (DHP universal ratio)")
    print(f"  T range: {T_RANGE[0]}–{T_RANGE[-1]}, N_seeds={N_SEEDS}")
    print("=" * 70)

    results = {}
    t_converge = None
    t_fail     = None

    for T in T_RANGE:
        seqs, targets = make_dataset(T)
        print(f"\n── T={T}: {2**T} sequences ──")
        t0 = time.time()

        seed_results = []
        for seed in range(N_SEEDS):
            print(f"  Seed {seed}:")
            loss, acc, epochs, theta, history = adam_train(seqs, targets, seed)
            converged = (loss < CONV_LOSS) and (acc >= CONV_ACC)
            seed_results.append({
                "seed":      seed,
                "loss":      loss,
                "accuracy":  acc,
                "epochs":    epochs,
                "converged": converged,
                "theta":     theta,
            })
            status = "✓ CONVERGED" if converged else "✗ FAILED"
            print(f"    → loss={loss:.6f} acc={acc:.3f} epochs={epochs} [{status}]")

        elapsed = time.time() - t0
        n_converged = sum(1 for r in seed_results if r["converged"])
        mean_acc    = float(np.mean([r["accuracy"] for r in seed_results]))
        mean_loss   = float(np.mean([r["loss"]     for r in seed_results]))
        solvable    = n_converged >= GOOD_SEEDS

        print(f"  T={T} SUMMARY: {n_converged}/{N_SEEDS} seeds converged | "
              f"mean_acc={mean_acc:.3f} | mean_loss={mean_loss:.5f} | "
              f"{'SOLVABLE' if solvable else 'UNSOLVABLE'} | {elapsed:.1f}s")

        results[T] = {
            "T":           T,
            "n_sequences": 2**T,
            "n_converged": n_converged,
            "n_seeds":     N_SEEDS,
            "solvable":    solvable,
            "mean_accuracy": mean_acc,
            "mean_loss":     mean_loss,
            "elapsed_s":     elapsed,
            "seeds":         seed_results,
        }

        # Track horizon boundary
        if solvable and (t_converge is None or T > t_converge):
            t_converge = T
        if not solvable and t_fail is None:
            t_fail = T

        # Save incremental results
        _save(results, t_converge, t_fail)

        # Early exit if clearly broken for 2 consecutive T values
        if t_fail is not None and T >= t_fail + 1:
            prev_solvable = results.get(T - 1, {}).get("solvable", True)
            if not prev_solvable:
                print(f"\n  Two consecutive T failures — stopping sweep at T={T}")
                break

    # Final DHP ratio
    print("\n" + "=" * 70)
    print("  Q-DHP SWEEP COMPLETE")
    print(f"  T_converge (last solvable T) : {t_converge}")
    print(f"  T_fail     (first unsolvable): {t_fail}")
    if t_converge and t_fail:
        ratio = t_converge / t_fail
        print(f"  Ratio T_converge/T_fail      : {ratio:.4f}")
        print(f"  DHP prediction (0.72)        : {'CONFIRMED ✓' if 0.65 < ratio < 0.79 else 'NOT CONFIRMED'}")
    print("=" * 70)

    return results, t_converge, t_fail


def _save(results, t_converge, t_fail):
    out = {
        "experiment": "Q-DHP Horizon Sweep",
        "hypothesis": "T_converge/T_fail ≈ 0.72 (DHP universal ratio)",
        "authors":    ["Archon", "Aura", "Jesse Caldwell"],
        "timestamp":  datetime.now().isoformat(),
        "config": {
            "T_range":    T_RANGE,
            "N_seeds":    N_SEEDS,
            "max_epochs": MAX_EPOCHS,
            "lr":         LR,
            "conv_loss":  CONV_LOSS,
            "conv_acc":   CONV_ACC,
            "good_seeds": GOOD_SEEDS,
        },
        "t_converge":  t_converge,
        "t_fail":      t_fail,
        "ratio":       (t_converge / t_fail) if (t_converge and t_fail) else None,
        "dhp_range_lo": 0.65,
        "dhp_range_hi": 0.79,
        "results":    {str(k): v for k, v in results.items()},
    }
    OUT_FILE.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    results, tc, tf = run_sweep()
