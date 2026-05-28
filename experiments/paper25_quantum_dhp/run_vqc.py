import qiskit
import bluequbit
import pennylane as qml
import numpy as np

def run_vqc_pennylane():
    print("🌌 Initializing PennyLane + BlueQubit Integration...")
    # Initialize BlueQubit client with our active subscription
    token = "FIysMm29ifkmSQeBZcE2N3kPGBOLcMkj"
    
    # Let's set up a PennyLane device that runs via BlueQubit's GPU simulator.
    # PennyLane supports different device interfaces, but we can also use BlueQubit's 
    # client directly to evaluate quantum expectations and steer classical vectors.
    bq_client = bluequbit.init(token)
    
    print("\n💡 Metaphor: The Quantum Compass (Variational Quantum Circuit)")
    print("Think of a 2-qubit state like a compass pointing somewhere in a multi-dimensional space.")
    print("We want to feed this compass a coordinate (classical data), rotate it using gates,")
    print("and then measure where it points. By tweaking the 'rotation angles' (weights),")
    print("the compass learns to align itself with a target mapping—just like a standard neural network layer!")
    
    # 1. Prepare dummy classical data (e.g. coordinates to encode)
    classical_data = np.array([0.5, -0.2]) # e.g. features we want to project
    # 2. Initialize random quantum weights (angles of rotation)
    weights = np.array([0.1, 0.8, -0.4]) 
    
    # Let's build a parameterized circuit that encodes this data and applies trainable rotations
    def build_parameterized_circuit(x, theta):
        qc = qiskit.QuantumCircuit(2)
        
        # --- Feature Map (Encoding classical data into quantum amplitudes) ---
        # Rotate Qubit 0 around X axis by classical data x[0]
        qc.rx(x[0], 0)
        # Rotate Qubit 1 around Y axis by classical data x[1]
        qc.ry(x[1], 1)
        
        # --- Entanglement Barrier ---
        qc.cx(0, 1)
        
        # --- Trainable Ansatz (Our weights change these angles) ---
        qc.rx(theta[0], 0)
        qc.ry(theta[1], 1)
        qc.rz(theta[2], 0)
        
        # Measure expectation of state
        qc.measure_all()
        return qc

    print("\n⚡ Constructing parameterized circuit...")
    qc = build_parameterized_circuit(classical_data, weights)
    
    print("🚀 Dispatching Variational circuit simulation to BlueQubit GPU backend...")
    try:
        # Run on high-performance GPU simulator
        job_result = bq_client.run(qc, device="gpu")
        counts = job_result.get_counts()
        print(f"✅ Simulation Complete! Expectation counts: {counts}")
        
        # Let's calculate the expectation value of Qubit 1 being in state |1> 
        # (This is like reading out the Y-axis of our quantum compass)
        # Counts are formatted as 'q1q0' -> e.g. '10' means Qubit 1 is 1, Qubit 0 is 0.
        total_shots = sum(counts.values())
        prob_q1_is_one = sum(val for key, val in counts.items() if key[0] == '1') / total_shots
        print(f"🎯 Output Measurement (Activation Value): {prob_q1_is_one:.4f}")
        
    except Exception as e:
        print(f"❌ BlueQubit execution failed: {e}")

if __name__ == "__main__":
    run_vqc_pennylane()
