import qiskit
from qiskit.quantum_info import Statevector
import numpy as np

def build_multi_wire_circuit(sequence, theta):
    """
    Builds a Q-RNN using the multi-wire input architecture (no resets).
    For a sequence of length T, we use T+1 qubits:
    Qubits 0 to T-1 are the input wires for steps 0 to T-1.
    Qubit T is the memory register.
    """
    T = len(sequence)
    # T input qubits + 1 memory qubit
    qc = qiskit.QuantumCircuit(T + 1)
    
    memory_qubit = T # The last qubit is the memory register
    
    for t, x_t in enumerate(sequence):
        input_qubit = t
        
        # 1. Encode input bit into the dedicated input wire
        qc.rx(x_t * np.pi, input_qubit)
        
        # 2. Recurrent Ansatz on (input_qubit, memory_qubit)
        qc.ry(theta[0], input_qubit)
        qc.ry(theta[1], memory_qubit)
        qc.cx(input_qubit, memory_qubit)
        qc.rz(theta[2], input_qubit)
        qc.rz(theta[3], memory_qubit)
        
    return qc

def main():
    print("=" * 80)
    print("🌌  D U O N E U R A L   M U L T I - W I R E   Q - R N N   V E R I F I C A T I O N  🌌")
    print("================================================================================")
    
    # Optimal weights from Paper 25/26
    theta = np.array([-3.3246, 2.9937, 1.8799, 1.8211])
    print(f"Optimal Weights: {theta}")
    
    # Generate 3-bit binary sequences and targets
    sequences = []
    targets = []
    for i in range(8):
        seq = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
        sequences.append(seq)
        targets.append(int(sum(seq) % 2))
        
    correct_count = 0
    print("\n📊 Local Statevector Verification:")
    print("-" * 80)
    print(f"{'Sequence':<12} | {'Target':<6} | {'Expectation (Memory)':<22} | {'Correct':<10}")
    print("-" * 80)
    
    for seq, target in zip(sequences, targets):
        qc = build_multi_wire_circuit(seq, theta)
        sv = Statevector(qc)
        probs = sv.probabilities() # Returns probabilities of 2^(T+1) states
        
        # We want to measure the probability of the memory qubit (qubit T) being in state 1.
        # In Qiskit's binary representation, state strings are ordered as q_T q_{T-1} ... q_0.
        # Qubit T is the first character in the binary state string.
        # So we sum probabilities for all states where the binary string starts with '1'.
        prob_one = 0.0
        num_qubits = T = len(seq)
        for state_idx in range(len(probs)):
            binary_str = format(state_idx, f"0{num_qubits+1}b")
            if binary_str[0] == '1': # The memory qubit is index T (MSB in string)
                prob_one += probs[state_idx]
                
        predicted_bit = int(prob_one > 0.5)
        is_correct = predicted_bit == target
        if is_correct:
            correct_count += 1
            
        print(f"{str(seq):<12} | {target:<6} | {prob_one:<22.4f} | {str(is_correct):<10}")
        
    print("-" * 80)
    print(f"Accuracy: {(correct_count / len(sequences)) * 100:.2f}% ({correct_count}/8)")
    print("=" * 80)

if __name__ == "__main__":
    main()
