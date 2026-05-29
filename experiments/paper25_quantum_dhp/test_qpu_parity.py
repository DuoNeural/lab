import qiskit
import bluequbit
import numpy as np
import time
import os
import json

def build_multi_wire_circuit_with_measure(sequence, theta):
    """
    Builds a Q-RNN using the multi-wire input architecture (no resets) with measurement gates.
    For a sequence of length T, we use T+1 qubits.
    Qubits 0 to T-1 are the input wires for steps 0 to T-1.
    Qubit T is the memory register.
    """
    T = len(sequence)
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
        
    qc.measure_all()
    return qc

def main():
    print("=" * 80)
    print("🌌  D U O N E U R A L   P H Y S I C A L   Q P U   Q - R N N   V E R I F I C A T I O N  🌌")
    print("================================================================================")
    
    # Initialize BlueQubit client
    token = "FIysMm29ifkmSQeBZcE2N3kPGBOLcMkj"
    bq_client = bluequbit.init(token)
    
    # Optimal weights from Paper 25/26
    theta = np.array([-3.3246, 2.9937, 1.8799, 1.8211])
    print(f"Using Optimal Shared Weights: {theta}")
    
    # Generate 3-bit binary sequences and target XOR parity
    sequences = []
    targets = []
    for i in range(8):
        seq = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
        sequences.append(seq)
        targets.append(int(sum(seq) % 2))
        
    X_train = np.array(sequences)
    y_train = np.array(targets)
    
    print("\n🚀 Dispatching 8 multi-wire circuits in parallel to BlueQubit QPU backend...")
    jobs = []
    shots_count = 100 # 100 shots to balance credit usage and statistics
    
    for seq in X_train:
        qc = build_multi_wire_circuit_with_measure(seq, theta)
        job = bq_client.run(qc, device="quantum", shots=shots_count, asynchronous=True)
        jobs.append((seq, job))
        print(f"Submitted job for sequence {list(seq)} | Job ID: {job.job_id}")
        
    print("\n⏳ Polling QPU jobs status (checking every 10 seconds)...")
    completed = [False] * len(jobs)
    job_results = [None] * len(jobs)
    
    start_time = time.time()
    while not all(completed):
        # We will wait 10 seconds between polls to conserve API requests
        time.sleep(10)
        
        for idx, (seq, job) in enumerate(jobs):
            if not completed[idx]:
                try:
                    updated_job = bq_client.get(job.job_id)
                    status = updated_job.run_status
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:.1f}s] Checking Job {job.job_id} | Status: {status}")
                    
                    if status == "COMPLETED":
                        completed[idx] = True
                        job_results[idx] = updated_job
                    elif status in ["FAILED", "CANCELLED"]:
                        completed[idx] = True
                        print(f"❌ Job {job.job_id} failed or was cancelled!")
                except Exception as e:
                    print(f"⚠️ Error polling job {job.job_id}: {e}")
                    
    print("\n🎉 All jobs finished! Analyzing results...")
    
    final_preds = []
    correct_count = 0
    results_dict = {}
    
    print("\n📊 Physical QPU Verification Results:")
    print("-" * 80)
    print(f"{'Sequence':<12} | {'Target':<6} | {'QPU Counts':<25} | {'QPU Expectation':<18} | {'Correct':<10}")
    print("-" * 80)
    
    for seq, target, res in zip(X_train, y_train, job_results):
        if res is None:
            final_preds.append(0.5)
            print(f"{str(list(seq)):<12} | {target:<6} | {'ERROR':<25} | {0.5:<18.4f} | {'False':<10}")
            continue
            
        counts = res.get_counts()
        # Endianness/Bit-Ordering Discrepancy:
        # In Qiskit Statevector, binary state strings are ordered as q_T q_{T-1} ... q_0 (big-endian).
        # However, BlueQubit's physical QPU readout returns strings ordered as q_0 q_1 ... q_T (little-endian).
        # Therefore, the memory qubit (qubit T) is the rightmost character (index -1) in the counts dict.
        count_ones = sum(count for state, count in counts.items() if state[-1] == '1')
        total_shots = sum(counts.values())
        prob_one = count_ones / total_shots if total_shots > 0 else 0.5
        final_preds.append(prob_one)
        
        predicted_bit = int(prob_one > 0.5)
        is_correct = predicted_bit == target
        if is_correct:
            correct_count += 1
            
        print(f"{str(list(seq)):<12} | {target:<6} | {str(counts):<25} | {prob_one:<18.4f} | {str(is_correct):<10}")
        
        # Save results for plotting/logging
        seq_list = [int(x) for x in seq]
        results_dict[str(seq_list)] = {
            "target": int(target),
            "counts": {state: int(cnt) for state, cnt in counts.items()},
            "prob_one": float(prob_one),
            "correct": bool(is_correct)
        }
        
    print("-" * 80)
    accuracy = (correct_count / len(X_train)) * 100
    print(f"Final Physical QPU Sequence Accuracy: {accuracy:.2f}% ({correct_count}/8)")
    print("=" * 80)
    
    # Save results to a JSON file
    output_path = "qpu_verification_results.json"
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
