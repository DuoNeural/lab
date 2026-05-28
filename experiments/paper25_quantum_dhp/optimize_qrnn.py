import qiskit
import bluequbit
import numpy as np
from scipy.optimize import minimize

# Initialize BlueQubit client
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

def build_qrnn_circuit(sequence, theta):
    qc = qiskit.QuantumCircuit(2)
    for x_t in sequence:
        qc.rx(x_t * np.pi, 0)
        qc.ry(theta[0], 0)
        qc.ry(theta[1], 1)
        qc.cx(0, 1)
        qc.rz(theta[2], 0)
        qc.rz(theta[3], 1)
    qc.measure_all()
    return qc

def get_prediction(seq, theta):
    qc = build_qrnn_circuit(seq, theta)
    result = bq_client.run(qc, device="cpu")
    counts = result.get_counts()
    prob_one = sum(prob for state, prob in counts.items() if state[0] == '1')
    return prob_one

def cost_function(theta):
    predictions = []
    for seq in X_train:
        pred = get_prediction(seq, theta)
        predictions.append(pred)
    predictions = np.array(predictions)
    loss = np.mean((predictions - y_train) ** 2)
    # Print progress inline
    sys.stdout.write(f"\rLoss: {loss:.5f} | Theta: {np.round(theta, 3)}")
    sys.stdout.flush()
    return loss

if __name__ == "__main__":
    import sys
    print("=" * 80)
    print("🔄 Optimizing Q-RNN Parity Classifier to 100% Accuracy...")
    print("Method: COBYLA (Constrained Optimization BY Linear Approximation)")
    print("Max Iterations: 150")
    print("=" * 80)
    
    # Start with our best weights from the previous run
    initial_theta = np.array([0.859, 0.389, 1.064, -0.378])
    
    res = minimize(
        cost_function,
        initial_theta,
        method='COBYLA',
        options={'maxiter': 150, 'disp': True}
    )
    
    print("\n\n🎉 COBYLA Optimization Complete!")
    print(f"Optimal Shared Weights: {res.x}")
    print(f"Final training loss: {res.fun:.6f}")
    
    # Run final verification
    final_preds = []
    correct_count = 0
    for seq, target in zip(X_train, y_train):
        pred = get_prediction(seq, res.x)
        predicted_bit = int(pred > 0.5)
        is_correct = predicted_bit == int(target)
        if is_correct:
            correct_count += 1
        final_preds.append((seq, target, pred, is_correct))
        
    print("\n📊 Sequence Classification Performance:")
    print("-" * 80)
    print(f"{'Sequence':<12} | {'Target':<6} | {'Prediction':<12} | {'Correct':<10}")
    print("-" * 80)
    for seq, target, pred, corr in final_preds:
        print(f"{str(list(seq)):<12} | {int(target):<6} | {pred:<12.4f} | {str(corr):<10}")
    print("-" * 80)
    print(f"Accuracy: {(correct_count/8)*100:.2f}% ({correct_count}/8)")
    print("=" * 80)
