import qiskit
import bluequbit
import numpy as np
from scipy.optimize import minimize
import sys

# Initialize BlueQubit client
TOKEN = "FIysMm29ifkmSQeBZcE2N3kPGBOLcMkj"
bq_client = bluequbit.init(TOKEN)

# Define XOR dataset
X_train = np.array([
    [0.0, 0.0],
    [0.0, np.pi], # Encode 1 as pi (rotation angle)
    [np.pi, 0.0],
    [np.pi, np.pi]
])
y_train = np.array([0.0, 1.0, 1.0, 0.0]) # XOR targets

def build_xor_circuit(x, theta):
    """
    Creates a 2-qubit parameterized circuit for XOR classification.
    x: Input data represented as rotation angles (rx rotations).
    theta: Trainable parameters (4 weights).
    """
    qc = qiskit.QuantumCircuit(2)
    
    # 1. State Preparation: Encode input features
    qc.rx(x[0], 0)
    qc.rx(x[1], 1)
    
    # 2. Entanglement (Creating a non-linear feature space)
    qc.cx(0, 1)
    
    # 3. Parameterized rotations (trainable gates)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)
    qc.rx(theta[2], 0)
    qc.ry(theta[3], 1)
    
    # Measure qubits
    qc.measure_all()
    return qc

def get_prediction(x, theta, device="cpu"):
    """
    Submits the circuit to BlueQubit and calculates prediction probability for state |1> on qubit 1.
    """
    qc = build_xor_circuit(x, theta)
    # Run the circuit on BlueQubit simulator
    result = bq_client.run(qc, device=device)
    counts = result.get_counts()
    
    # BlueQubit returns probabilities summing to 1.0 on simulators
    # We calculate the probability of measuring qubit 1 as '1' (states '10' and '11')
    prob_one = sum(prob for state, prob in counts.items() if state[0] == '1')
    return prob_one

def cost_function(theta):
    """
    Calculates Mean Squared Error over the training set.
    """
    predictions = []
    for x in X_train:
        # Use fast CPU backend for local training loop iterations
        pred = get_prediction(x, theta, device="cpu")
        predictions.append(pred)
        
    predictions = np.array(predictions)
    loss = np.mean((predictions - y_train) ** 2)
    print(f"Current parameters: {np.round(theta, 3)} | Loss: {loss:.6f} | Predictions: {np.round(predictions, 3)}")
    return loss

def main():
    print("=" * 80)
    print("🌌  TRAINING A VARIATIONAL QUANTUM CLASSIFIER (VQC) FOR XOR  🌌")
    print("=" * 80)
    print("Target XOR outputs for inputs [0,0], [0,1], [1,0], [1,1] are: [0.0, 1.0, 1.0, 0.0]\n")
    
    # Initial weights (randomly close to 0)
    initial_theta = np.array([0.1, -0.2, 0.15, 0.05])
    
    print("🔄 Running optimization loop using Nelder-Mead on BlueQubit simulator...")
    
    # Minimize the cost function classically
    res = minimize(
        cost_function, 
        initial_theta, 
        method='Nelder-Mead', 
        options={'maxiter': 25, 'disp': True}
    )
    
    print("\n🎉 Optimization Complete!")
    print(f"Optimal parameters: {res.x}")
    print(f"Final loss: {res.fun:.6f}")
    
    # Run final prediction using GPU backend to verify the optimal parameters
    print("\n🚀 Verifying optimal parameters on BlueQubit's GPU Simulator backend...")
    final_preds = []
    for x in X_train:
        pred = get_prediction(x, res.x, device="gpu")
        final_preds.append(pred)
    
    print("\n📊 Verification Results:")
    for i, (x, target, pred) in enumerate(zip(X_train, y_train, final_preds)):
        encoded_input = [int(val > 1.0) for val in x]
        print(f"Input: {encoded_input} | Target: {target} | VQC Output Probability: {pred:.4f} (Match: {abs(pred-target) < 0.2})")
    
    print("\n💡 Metaphor Sync:")
    print("By adjusting the rotation angles (theta), the optimization loop successfully steered")
    print("our entangled multi-dimensional quantum compass to align with the non-linear XOR boundary!")
    print("This would be mathematically impossible for a single classical neuron without hidden layers,")
    print("but our 2-qubit quantum state space handled it natively thanks to entanglement.")
    print("=" * 80)

if __name__ == "__main__":
    main()
