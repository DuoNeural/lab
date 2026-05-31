import qiskit
from qiskit.quantum_info import Statevector
import bluequbit
import numpy as np
import sys
import time

# Initialize BlueQubit client for final GPU verification
TOKEN = "FIysMm29ifkmSQeBZcE2N3kPGBOLcMkj"
bq_client = bluequbit.init(TOKEN)

# Generate 3-bit binary sequences and parity targets
sequences = []
targets = []
for i in range(8):
    seq = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
    sequences.append(seq)
    targets.append(float(sum(seq) % 2))

X_train = np.array(sequences)
y_train = np.array(targets)

def build_qrnn_circuit_no_measure(sequence, theta):
    """
    Builds a 2-qubit Q-RNN without measurement gates for local statevector simulation.
    """
    qc = qiskit.QuantumCircuit(2)
    for x_t in sequence:
        qc.reset(0)
        qc.rx(x_t * np.pi, 0)
        
        # Recurrent Ansatz (Shared weights)
        qc.ry(theta[0], 0)
        qc.ry(theta[1], 1)
        qc.cx(0, 1)
        qc.rz(theta[2], 0)
        qc.rz(theta[3], 1)
        
    return qc

def build_qrnn_circuit_with_measure(sequence, theta):
    """
    Builds a 2-qubit Q-RNN with measurements for cloud execution.
    """
    qc = build_qrnn_circuit_no_measure(sequence, theta)
    qc.measure_all()
    return qc

def get_prediction_local(seq, theta):
    """
    Calculates the exact expectation value of qubit 1 being 1 locally using Qiskit Statevector.
    Extremely fast (under 1ms), exact, and completely free.
    """
    qc = build_qrnn_circuit_no_measure(seq, theta)
    sv = Statevector(qc)
    probs = sv.probabilities() # Returns probabilities of states: [00, 01, 10, 11]
    # In Qiskit, index 2 corresponds to |10> and index 3 corresponds to |11> (qubit 1 is 1)
    prob_qubit1_is_one = probs[2] + probs[3]
    return prob_qubit1_is_one

def train():
    # Initialize parameters randomly
    np.random.seed(42)
    theta = np.random.uniform(-np.pi, np.pi, 4)
    
    # Adam Hyperparameters
    lr = 0.15 # Fast learning rate for rapid local Adam convergence
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    epochs = 100
    
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    
    print("=" * 80)
    print("🚀 Training Q-RNN locally using Native Qiskit Statevector + Adam Optimizer...")
    print(f"Initial parameters: {np.round(theta, 3)}")
    print("=" * 80)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # 1. Compute predictions and loss locally (instantaneous!)
        preds = np.array([get_prediction_local(seq, theta) for seq in X_train])
        loss = np.mean((preds - y_train) ** 2)
        preds_bits = (preds > 0.5).astype(int)
        accuracy = np.mean(preds_bits == y_train.astype(int)) * 100.0
        
        # 2. Compute exact gradients using Parameter Shift Rule locally
        grads = np.zeros_like(theta)
        for i in range(len(theta)):
            # Shift Forward
            theta_f = theta.copy()
            theta_f[i] += np.pi / 2
            preds_f = np.array([get_prediction_local(seq, theta_f) for seq in X_train])
            
            # Shift Backward
            theta_b = theta.copy()
            theta_b[i] -= np.pi / 2
            preds_b = np.array([get_prediction_local(seq, theta_b) for seq in X_train])
            
            # Derivative: df/dtheta = (f_forward - f_backward) / 2
            dp_dtheta = (preds_f - preds_b) / 2.0
            grads[i] = np.mean(2.0 * (preds - y_train) * dp_dtheta)
            
        # 3. Adam Update
        m = beta1 * m + (1.0 - beta1) * grads
        v = beta2 * v + (1.0 - beta2) * (grads ** 2)
        
        m_hat = m / (1.0 - beta1 ** (epoch + 1))
        v_hat = v / (1.0 - beta2 ** (epoch + 1))
        
        theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch+1:03d} | Loss: {loss:.6f} | Accuracy: {accuracy:.1f}% | Time: {elapsed*1000:.1f}ms")
        
        if loss < 0.001:
            print(f"\n🎉 Convergence reached at Epoch {epoch+1}!")
            break
            
    print("\n🎉 Training Finished!")
    print(f"Optimal Shared Weights: {theta}")
    print(f"Final training loss: {loss:.6f}")
    
    # 4. Final verification on remote BlueQubit GPU simulator backend
    print("\n🚀 Dispatching final model verification to BlueQubit GPU Simulator backend...")
    final_preds = []
    
    # Run them sequentially to guarantee index order mapping (avoids database search order shuffling)
    for idx, seq in enumerate(X_train):
        qc = build_qrnn_circuit_with_measure(seq, theta)
        res = bq_client.run(qc, device="gpu")
        counts = res.get_counts()
        prob_one = sum(prob for state, prob in counts.items() if state[0] == '1')
        final_preds.append(prob_one)
        print(f"Verified sequence {idx+1}/8: {list(seq)} -> expectation {prob_one:.4f}")
        
    final_preds = np.array(final_preds)
    correct_count = 0
    
    print("\n📊 Remote GPU Simulator Verification Results:")
    print("-" * 80)
    print(f"{'Sequence':<12} | {'Target':<6} | {'Q-RNN Expectation':<18} | {'Correct':<10}")
    print("-" * 80)
    for seq, target, pred in zip(X_train, y_train, final_preds):
        predicted_bit = int(pred > 0.5)
        is_correct = predicted_bit == int(target)
        if is_correct:
            correct_count += 1
        print(f"{str(list(seq)):<12} | {int(target):<6} | {pred:<18.4f} | {str(is_correct):<10}")
    print("-" * 80)
    accuracy = (correct_count / len(X_train)) * 100
    print(f"Final Model Sequence Accuracy: {accuracy:.2f}% ({correct_count}/8)")
    print("=" * 80)

if __name__ == "__main__":
    train()
