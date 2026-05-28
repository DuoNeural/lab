import qiskit
import bluequbit
import numpy as np
from scipy.optimize import minimize
import sys

# Initialize BlueQubit client
TOKEN = "FIysMm29ifkmSQeBZcE2N3kPGBOLcMkj"
bq_client = bluequbit.init(TOKEN)

# Generate all 8 binary sequences of length 3 and their XOR parities
sequences = []
targets = []
for i in range(8):
    # Convert index to 3-bit binary list
    seq = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
    sequences.append(seq)
    # Target parity is the sum modulo 2
    targets.append(float(sum(seq) % 2))

X_train = np.array(sequences)
y_train = np.array(targets)

def build_qrnn_circuit(sequence, theta):
    """
    Builds a 2-qubit Unitary Recurrent Quantum Circuit.
    sequence: list of length 3 containing 0s and 1s.
    theta: recurrent weight parameters (4 parameters).
    """
    qc = qiskit.QuantumCircuit(2)
    
    # Process sequence step by step (Time steps t = 0, 1, 2)
    for x_t in sequence:
        # 1. Encoding Layer: Encode input bit into qubit 0 as an RX rotation (0 or pi)
        qc.rx(x_t * np.pi, 0)
        
        # 2. Recurrent Weight Ansatz (Shared weights across all steps)
        qc.ry(theta[0], 0)
        qc.ry(theta[1], 1)
        qc.cx(0, 1) # Entangle qubit 0 (input) and qubit 1 (memory)
        qc.rz(theta[2], 0)
        qc.rz(theta[3], 1)
        
    # Measure qubits to read final state
    qc.measure_all()
    return qc

def get_prediction(seq, theta, device="cpu"):
    """
    Runs the QRNN circuit on BlueQubit and returns probability of memory qubit (qubit 1) being in state 1.
    """
    qc = build_qrnn_circuit(seq, theta)
    result = bq_client.run(qc, device=device)
    counts = result.get_counts()
    
    # Probability of qubit 1 being 1 (states '10' and '11')
    prob_one = sum(prob for state, prob in counts.items() if state[0] == '1')
    return prob_one

def cost_function(theta):
    """
    MSE loss function over the dataset.
    """
    predictions = []
    for seq in X_train:
        # Use CPU simulation for training feedback loop
        pred = get_prediction(seq, theta, device="cpu")
        predictions.append(pred)
        
    predictions = np.array(predictions)
    loss = np.mean((predictions - y_train) ** 2)
    print(f"Weights: {np.round(theta, 3)} | Loss: {loss:.5f}")
    return loss

def main():
    print("=" * 80)
    print("🌌  D U O N E U R A L   Q U A N T U M   R E C U R R E N T   N E T W O R K  🌌")
    print("================================================================================")
    print(f"Goal: Train a 2-qubit Q-RNN to solve 3-step Temporal Parity (XOR over time)")
    print(f"Training Dataset: {len(X_train)} sequences of length 3.")
    print("=" * 80)
    
    # Initial weights (shared recurrent parameters)
    initial_theta = np.array([0.5, -0.5, 0.2, 0.1])
    
    print("\n🔄 Running Nelder-Mead Optimization on BlueQubit CPU simulator...")
    res = minimize(
        cost_function,
        initial_theta,
        method='Nelder-Mead',
        options={'maxiter': 40, 'disp': True}
    )
    
    print("\n🎉 Training Loop Complete!")
    print(f"Optimal Shared Weights: {res.x}")
    print(f"Final training loss: {res.fun:.6f}")
    
    # Verify results using high-performance GPU simulator
    print("\n🚀 Verifying trained Q-RNN predictions on BlueQubit GPU Simulator backend...")
    final_preds = []
    for seq in X_train:
        pred = get_prediction(seq, res.x, device="gpu")
        final_preds.append(pred)
        
    print("\n📊 Sequence Classification Performance:")
    print("-" * 80)
    print(f"{'Sequence':<12} | {'Target Parity':<15} | {'Q-RNN Prediction Prob':<22} | {'Correct':<10}")
    print("-" * 80)
    
    correct_count = 0
    for seq, target, pred in zip(X_train, y_train, final_preds):
        predicted_bit = int(pred > 0.5)
        is_correct = predicted_bit == int(target)
        if is_correct:
            correct_count += 1
        print(f"{str(list(seq)):<12} | {int(target):<15} | {pred:<22.4f} | {str(is_correct):<10}")
    
    print("-" * 80)
    accuracy = (correct_count / len(X_train)) * 100
    print(f"Final Model Sequence Accuracy: {accuracy:.2f}% ({correct_count}/8)")
    print("=" * 80)

if __name__ == "__main__":
    main()
