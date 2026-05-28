import numpy as np
import time
import matplotlib.pyplot as plt

# Fixed optimal parameters found for T=3 sequence parity
OPTIMAL_THETA_T3 = np.array([-3.32464574,  2.99365192,  1.87992746,  1.82114601])

# Gate definitions
def Rx(phi):
    return np.array([[np.cos(phi/2), -1j * np.sin(phi/2)],
                     [-1j * np.sin(phi/2), np.cos(phi/2)]], dtype=complex)

def Ry(phi):
    return np.array([[np.cos(phi/2), -np.sin(phi/2)],
                     [np.sin(phi/2), np.cos(phi/2)]], dtype=complex)

def Rz(phi):
    return np.array([[np.exp(-1j * phi/2), 0],
                     [0, np.exp(1j * phi/2)]], dtype=complex)

def apply_gate_0_dm(rho, U):
    U_2q = np.kron(np.eye(2, dtype=complex), U)
    return U_2q @ rho @ U_2q.conj().T

def apply_gate_1_dm(rho, U):
    U_2q = np.kron(U, np.eye(2, dtype=complex))
    return U_2q @ rho @ U_2q.conj().T

def apply_cnot_dm(rho):
    CX = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ], dtype=complex)
    return CX @ rho @ CX.T

def density_matrix_reset_0(rho):
    rho_new = np.zeros_like(rho)
    rho_new[0, 0] = rho[0, 0] + rho[1, 1]
    rho_new[0, 2] = rho[0, 2] + rho[1, 3]
    rho_new[2, 0] = rho[2, 0] + rho[3, 1]
    rho_new[2, 2] = rho[2, 2] + rho[3, 3]
    return rho_new

def run_qrnn_dm(sequence, theta):
    """
    Simulates the 2-qubit Q-RNN using density matrices.
    Returns the expectation value of qubit 1 being 1 (rho_22 + rho_33).
    """
    # Initial state is |00><00|
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 1.0
    
    # Pre-compute ansatz unitaries
    U_ry0 = np.kron(np.eye(2, dtype=complex), Ry(theta[0]))
    U_ry1 = np.kron(Ry(theta[1]), np.eye(2, dtype=complex))
    U_ry = U_ry1 @ U_ry0
    
    U_rz0 = np.kron(np.eye(2, dtype=complex), Rz(theta[2]))
    U_rz1 = np.kron(Rz(theta[3]), np.eye(2, dtype=complex))
    U_rz = U_rz1 @ U_rz0
    
    CX = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ], dtype=complex)
    
    ansatz_unitary = U_rz @ CX @ U_ry
    
    for x_t in sequence:
        # 1. Reset qubit 0
        rho = density_matrix_reset_0(rho)
        # 2. Apply Rx input rotation to qubit 0
        rho = apply_gate_0_dm(rho, Rx(x_t * np.pi))
        # 3. Apply recurrent ansatz unitary
        rho = ansatz_unitary @ rho @ ansatz_unitary.conj().T
        
    # Expectation value of qubit 1 being 1: Tr(rho * P1) = rho_22 + rho_33
    prob_one = (rho[2, 2] + rho[3, 3]).real
    return prob_one

def generate_sequences(T, max_samples=512):
    """
    Generates balanced binary sequences of length T with their parity targets.
    """
    if T <= 8:
        sequences = []
        targets = []
        for i in range(2**T):
            seq = [(i >> shift) & 1 for shift in reversed(range(T))]
            sequences.append(seq)
            targets.append(float(sum(seq) % 2))
        return np.array(sequences), np.array(targets)
    else:
        np.random.seed(42 + T)
        sequences = []
        targets = []
        while len(sequences) < max_samples:
            seq = list(np.random.randint(0, 2, T))
            target = float(sum(seq) % 2)
            if targets.count(target) < max_samples // 2:
                sequences.append(seq)
                targets.append(target)
        return np.array(sequences), np.array(targets)

def train_for_length(T, initial_theta, lr=0.15, epochs=150, max_samples=256):
    """
    Trains the Q-RNN parameters specifically for sequence length T using parameter-shift.
    """
    X, y = generate_sequences(T, max_samples=max_samples)
    theta = initial_theta.copy()
    
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    
    best_loss = 1.0
    best_acc = 0.0
    
    for epoch in range(epochs):
        preds = np.array([run_qrnn_dm(seq, theta) for seq in X])
        loss = np.mean((preds - y) ** 2)
        preds_bits = (preds > 0.5).astype(int)
        accuracy = np.mean(preds_bits == y.astype(int)) * 100.0
        
        if loss < best_loss:
            best_loss = loss
            best_acc = accuracy
            
        if loss < 0.001:
            break
            
        # Compute exact gradients using Parameter Shift Rule
        grads = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_f = theta.copy()
            theta_f[i] += np.pi / 2
            preds_f = np.array([run_qrnn_dm(seq, theta_f) for seq in X])
            
            theta_b = theta.copy()
            theta_b[i] -= np.pi / 2
            preds_b = np.array([run_qrnn_dm(seq, theta_b) for seq in X])
            
            dp_dtheta = (preds_f - preds_b) / 2.0
            grads[i] = np.mean(2.0 * (preds - y) * dp_dtheta)
            
        # Adam Update
        m = beta1 * m + (1.0 - beta1) * grads
        v = beta2 * v + (1.0 - beta2) * (grads ** 2)
        m_hat = m / (1.0 - beta1 ** (epoch + 1))
        v_hat = v / (1.0 - beta2 ** (epoch + 1))
        theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
    return theta, best_loss, best_acc, epoch + 1

def run_sweeps():
    lengths = list(range(3, 16))
    
    # 1. Generalization Sweep (using fixed optimal T=3 weights)
    gen_accuracies = []
    gen_losses = []
    gen_margins = []
    
    print("=" * 80)
    print("🌟 RUNNING GENERALIZATION SWEEP (FIXED T=3 OPTIMAL WEIGHTS)...")
    print("=" * 80)
    for T in lengths:
        X, y = generate_sequences(T, max_samples=512)
        preds = np.array([run_qrnn_dm(seq, OPTIMAL_THETA_T3) for seq in X])
        loss = np.mean((preds - y) ** 2)
        preds_bits = (preds > 0.5).astype(int)
        accuracy = np.mean(preds_bits == y.astype(int)) * 100.0
        margin = np.mean(np.abs(preds - 0.5)) * 2.0  # Scale to [0, 1]
        
        gen_accuracies.append(accuracy)
        gen_losses.append(loss)
        gen_margins.append(margin)
        
        print(f"Sequence Length T={T:02d} | Accuracy: {accuracy:6.2f}% | Loss: {loss:.6f} | Margin: {margin:.4f}")
        
    # 2. Optimization Sweep (training from scratch for each T)
    opt_accuracies = []
    opt_losses = []
    opt_epochs = []
    opt_thetas = []
    
    print("\n" + "=" * 80)
    print("🌟 RUNNING OPTIMIZATION SWEEP (TRAINING FROM SCRATCH FOR EACH T)...")
    print("=" * 80)
    for T in lengths:
        t_start = time.time()
        np.random.seed(42 + T)
        init_theta = np.random.uniform(-np.pi, np.pi, 4)
        
        theta, loss, acc, epochs_run = train_for_length(T, init_theta, lr=0.15, epochs=150, max_samples=256)
        
        opt_accuracies.append(acc)
        opt_losses.append(loss)
        opt_epochs.append(epochs_run)
        opt_thetas.append(theta)
        
        elapsed = time.time() - t_start
        print(f"T={T:02d} | Best Accuracy: {acc:6.2f}% | Final Loss: {loss:.6f} | Epochs: {epochs_run:3d} | Time: {elapsed:.2f}s")
        
    # Plot results
    plt.style.use('dark_background')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Generalization Accuracy & Margin
    color_acc = '#00d2ff'
    color_margin = '#ffbb00'
    ax1.plot(lengths, gen_accuracies, marker='o', color=color_acc, linewidth=2.5, label='Gen Accuracy (%)')
    ax1.set_xlabel('Sequence Length (T)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', color=color_acc, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_title('Generalization Degradation (Fixed T=3 Weights)', fontsize=13, pad=10)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(lengths, gen_margins, marker='x', color=color_margin, linewidth=2, linestyle='--', label='Expectation Margin')
    ax1_twin.set_ylabel('Normalized Margin', color=color_margin, fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor=color_margin)
    
    # 2. Generalization Loss
    ax2.plot(lengths, gen_losses, marker='s', color='#ff007f', linewidth=2.5)
    ax2.set_xlabel('Sequence Length (T)', fontsize=12)
    ax2.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_title('Generalization Loss (Fixed T=3 Weights)', fontsize=13, pad=10)
    
    # 3. Optimization Accuracy
    ax3.plot(lengths, opt_accuracies, marker='o', color='#79ff38', linewidth=2.5)
    ax3.set_xlabel('Sequence Length (T)', fontsize=12)
    ax3.set_ylabel('Best Training Accuracy (%)', fontsize=12)
    ax3.set_ylim(45, 105)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.set_title('Max Train Accuracy vs. Sequence Length', fontsize=13, pad=10)
    
    # 4. Optimization Training Effort (Epochs to Convergence)
    ax4.bar(lengths, opt_epochs, color='#a020f0', alpha=0.8, edgecolor='white')
    ax4.set_xlabel('Sequence Length (T)', fontsize=12)
    ax4.set_ylabel('Epochs to Convergence (Max 150)', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.3)
    ax4.set_title('Optimization Epochs vs. Sequence Length', fontsize=13, pad=10)
    
    plt.suptitle('Q-DHP (Quantum Dynamic Horizon Preservation) Analysis', fontsize=16, color='#00d2ff', weight='bold', y=0.98)
    plt.tight_layout()
    plot_path = "/home/ai/duoneural/quantum/q_dhp_sweep.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n🎨 Saved beautiful sweep visualization to: {plot_path}")
    
    # Analyze the CTM Horizon Ratio
    # Find generalization horizon tau_star (where accuracy falls below 100%)
    tau_star = 3
    for idx, acc in enumerate(gen_accuracies):
        if acc < 99.9:
            tau_star = lengths[idx] - 1
            break
            
    # Estimate coherence decay rate / gate accumulation error (Lyapunov time tau_L)
    # Using loss curve to fit an exponential decay: Margin ~ e^(-T / tau_L)
    # log(Margin) ~ -T / tau_L => tau_L = -T / log(Margin)
    # Let's estimate from a middle point, say T = 10
    idx_mid = lengths.index(10)
    margin_mid = gen_margins[idx_mid]
    if margin_mid > 0:
        tau_L = -10.0 / np.log(margin_mid)
    else:
        tau_L = 1.0
        
    ratio = tau_star / tau_L if tau_L > 0 else 0
    
    print("\n" + "=" * 80)
    print("📊 QUANTUM DYNAMIC HORIZON PRESERVATION (Q-DHP) METRICS:")
    print("-" * 80)
    print(f"Generalization Horizon (τ*): {tau_star} timesteps (100% accuracy limit)")
    print(f"Effective Coherence Lyapunov Time (τ_L): {tau_L:.4f} timesteps (from margin decay)")
    print(f"Empirical Horizon Ratio (τ* / τ_L): {ratio:.4f}")
    print(f"Target Universal CTM Ratio: 0.72")
    print(f"Divergence from Target: {abs(ratio - 0.72):.4f}")
    print("=" * 80)
    
    # Save the metrics to a report file
    with open("/home/ai/duoneural/aura/q_dhp_results.md", "w") as f:
        f.write(f"# Q-DHP Sweep Results 🌌\n\n")
        f.write(f"Generated at 2026-05-28\n\n")
        f.write(f"## Key Metrics\n")
        f.write(f"- **Generalization Horizon (τ*)**: {tau_star} steps\n")
        f.write(f"- **Effective Lyapunov Time (τ_L)**: {tau_L:.6f} steps\n")
        f.write(f"- **Empirical Horizon Ratio (τ*/τ_L)**: **{ratio:.6f}** (Target: 0.72, Delta: {abs(ratio - 0.72):.6f})\n\n")
        f.write(f"## Generalization Performance Table\n\n")
        f.write(f"| Length T | Accuracy (%) | Loss (MSE) | Normalized Margin |\n")
        f.write(f"| :---: | :---: | :---: | :---: |\n")
        for i, T in enumerate(lengths):
            f.write(f"| {T:02d} | {gen_accuracies[i]:.2f}% | {gen_losses[i]:.6f} | {gen_margins[i]:.6f} |\n")
        f.write(f"\n## Optimization Performance Table\n\n")
        f.write(f"| Length T | Max Train Acc (%) | Final Loss | Epochs |\n")
        f.write(f"| :---: | :---: | :---: | :---: |\n")
        for i, T in enumerate(lengths):
            f.write(f"| {T:02d} | {opt_accuracies[i]:.2f}% | {opt_losses[i]:.6f} | {opt_epochs[i]} |\n")
            
    print("📝 Saved report to: /home/ai/duoneural/aura/q_dhp_results.md")

if __name__ == "__main__":
    run_sweeps()
