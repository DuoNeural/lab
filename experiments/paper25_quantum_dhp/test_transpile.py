import qiskit
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap
import numpy as np

def build_multi_wire_circuit(sequence, theta):
    qc = qiskit.QuantumCircuit(4)
    memory_qubit = 3
    for t, x_t in enumerate(sequence):
        input_qubit = t
        qc.rx(x_t * np.pi, input_qubit)
        qc.ry(theta[0], input_qubit)
        qc.ry(theta[1], memory_qubit)
        qc.cx(input_qubit, memory_qubit)
        qc.rz(theta[2], input_qubit)
        qc.rz(theta[3], memory_qubit)
    qc.measure_all()
    return qc

def main():
    print("=" * 80)
    print("🔬  T R A N S P I L A T I O N   A N A L Y S I S  🔬")
    print("=" * 80)
    
    theta = np.array([-3.3246, 2.9937, 1.8799, 1.8211])
    seq = [0, 1, 1]
    qc = build_multi_wire_circuit(seq, theta)
    
    print("Original Circuit:")
    print(f"Depth: {qc.depth()}")
    print(f"Gates: {qc.count_ops()}")
    
    # Let's define a typical linear coupling map for physical QPUs (e.g., 0-1-2-3 line connectivity)
    # Qubit 3 (memory) is only connected to Qubit 2! It cannot directly CNOT with Qubit 0 or 1.
    linear_map = CouplingMap([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]])
    print("\n1. Transpiling for Linear Coupling Map (0 - 1 - 2 - 3)...")
    transpiled_linear = transpile(qc, coupling_map=linear_map, optimization_level=3, seed_transpiler=42)
    print(f"Linear Transpiled Depth: {transpiled_linear.depth()}")
    print(f"Linear Transpiled Gates: {transpiled_linear.count_ops()}")
    
    # Let's define a T-shaped or star coupling map where Qubit 3 is connected to all (0-3, 1-3, 2-3)
    # This represents all-to-all star connectivity (no SWAPs needed for the memory qubit).
    star_map = CouplingMap([[0, 3], [3, 0], [1, 3], [3, 1], [2, 3], [3, 2]])
    print("\n2. Transpiling for Star Coupling Map (All inputs connected directly to Memory)...")
    transpiled_star = transpile(qc, coupling_map=star_map, optimization_level=3, seed_transpiler=42)
    print(f"Star Transpiled Depth: {transpiled_star.depth()}")
    print(f"Star Transpiled Gates: {transpiled_star.count_ops()}")
    
    print("\n💡 Insights:")
    print("If the QPU coupling map does not support direct connections between all inputs and the memory qubit,")
    print("the transpiler must insert SWAP gates to route qubits. Each SWAP gate contains 3 physical CNOT gates.")
    print("For a linear QPU, the gate count and depth increase significantly, multiplying decoherence and gate errors.")

if __name__ == "__main__":
    main()
