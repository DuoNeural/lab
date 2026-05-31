import qiskit
import bluequbit
import os

def main():
    print("🌌 Initializing BlueQubit Client...")
    # Use the token provided by Jesse
    token = "FIysMm29ifkmSQeBZcE2N3kPGBOLcMkj"
    bq_client = bluequbit.init(token)
    
    print("⚡ Building a 2-qubit Entangled (Bell) State circuit in Qiskit...")
    # Create a Quantum Circuit with 2 qubits
    qc = qiskit.QuantumCircuit(2)
    # Put qubit 0 into a superposition of |0> and |1>
    qc.h(0)
    # Entangle qubit 1 with qubit 0
    qc.cx(0, 1)
    
    # Measure both qubits to see the entanglement in action
    qc.measure_all()
    
    print("🚀 Dispatching circuit simulation job to BlueQubit (CPU backend)...")
    try:
        # Run on BlueQubit's simulator
        job_result = bq_client.run(qc, device="cpu")
        
        print("\n🎉 Job Complete! Retrieving results:")
        counts = job_result.get_counts()
        print(f"Expectation Counts: {counts}")
        
        print("\n💡 Explanation of the results:")
        print("In a perfect Bell state, measuring one qubit immediately tells you the state of the other.")
        print("Therefore, we should only see '00' and '11' as outcomes (within statistical noise).")
        print("If we get '01' or '10', the entanglement would be broken or not configured as a Bell state.")
        
    except Exception as e:
        print(f"❌ Failed to run job on BlueQubit: {e}")

if __name__ == "__main__":
    main()
