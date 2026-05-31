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

def build_cal_zero_circuit():
    """
    Builds a 4-qubit circuit where all qubits start in |0> and are measured.
    Calibrates eta_0: false positive probability P(1|0).
    """
    qc = qiskit.QuantumCircuit(4)
    qc.measure_all()
    return qc

def build_cal_one_circuit():
    """
    Builds a 4-qubit circuit where memory qubit (q3) is set to |1> via X gate and all are measured.
    Calibrates eta_1: false negative probability P(0|1).
    """
    qc = qiskit.QuantumCircuit(4)
    qc.x(3) # Qubit 3 is the memory qubit
    qc.measure_all()
    return qc

def main():
    print("=" * 80)
    print("🌌  D U O N E U R A L   H I G H - S H O T   P H Y S I C A L   Q P U   Q - R N N  🌌")
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
    
    shots_count = 4096  # High-shot count requested for statistical resolution
    print(f"Shots Count: {shots_count}")
    
    # Build list of all circuits to run (8 sequences + 2 calibrations)
    circuits_to_run = []
    for seq in X_train:
        qc = build_multi_wire_circuit_with_measure(seq, theta)
        circuits_to_run.append(('sequence', list(seq), qc))
        
    circuits_to_run.append(('cal_zero', 'cal_zero', build_cal_zero_circuit()))
    circuits_to_run.append(('cal_one', 'cal_one', build_cal_one_circuit()))
    
    print("\n🚀 Dispatching 10 circuits (8 sequences + 2 calibrations) in parallel to BlueQubit QPU...")
    jobs = []
    for item_type, key, qc in circuits_to_run:
        job = bq_client.run(qc, device="quantum", shots=shots_count, asynchronous=True)
        jobs.append((item_type, key, job))
        print(f"Submitted {item_type} job ({key}) | Job ID: {job.job_id}")
        
    print("\n⏳ Polling QPU jobs status (checking every 10 seconds)...")
    completed = [False] * len(jobs)
    job_results = [None] * len(jobs)
    
    start_time = time.time()
    while not all(completed):
        time.sleep(10)
        
        for idx, (item_type, key, job) in enumerate(jobs):
            if not completed[idx]:
                try:
                    updated_job = bq_client.get(job.job_id)
                    status = updated_job.run_status
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:.1f}s] Checking {item_type} ({key}) | Status: {status}")
                    
                    if status == "COMPLETED":
                        completed[idx] = True
                        job_results[idx] = updated_job
                    elif status in ["FAILED", "CANCELLED"]:
                        completed[idx] = True
                        print(f"❌ {item_type} job ({key}) failed or was cancelled!")
                except Exception as e:
                    print(f"⚠️ Error polling job {job.job_id}: {e}")
                    
    print("\n🎉 All jobs finished! Analyzing results...")
    
    # Separate calibration results
    cal_zero_counts = {}
    cal_one_counts = {}
    
    for (item_type, key, job), res in zip(jobs, job_results):
        if res is not None:
            if item_type == 'cal_zero':
                cal_zero_counts = res.get_counts()
            elif item_type == 'cal_one':
                cal_one_counts = res.get_counts()
                
    # Calculate readout flip probabilities
    # Remember: little-endian bit-ordering (q3 is at index -1)
    def get_prob_one(counts):
        total = sum(counts.values())
        if total == 0:
            return 0.5
        ones = sum(cnt for state, cnt in counts.items() if state[-1] == '1')
        return ones / total

    eta_0 = get_prob_one(cal_zero_counts) # False positive rate P(1|0)
    eta_1 = 1.0 - get_prob_one(cal_one_counts) # False negative rate P(0|1)
    
    print("\n📊 Readout Calibration Metrics:")
    print("-" * 50)
    print(f"Cal 0 counts (Prepared |0>): {cal_zero_counts}")
    print(f"Cal 1 counts (Prepared |1>): {cal_one_counts}")
    print(f"False Positive Rate (eta_0): {eta_0:.4f}")
    print(f"False Negative Rate (eta_1): {eta_1:.4f}")
    print(f"Calibration Denominator (1 - eta_0 - eta_1): {1.0 - eta_0 - eta_1:.4f}")
    print("-" * 50)
    
    correct_raw = 0
    correct_mit = 0
    results_dict = {
        "calibration": {
            "eta_0": float(eta_0),
            "eta_1": float(eta_1),
            "cal_zero_counts": cal_zero_counts,
            "cal_one_counts": cal_one_counts
        },
        "results": {},
        "shots": shots_count
    }
    
    print("\n📊 Sequence Parity Results:")
    print("-" * 120)
    print(f"{'Sequence':<10} | {'Target':<6} | {'Raw P(1)':<10} | {'Raw ⟨Z⟩':<8} | {'Raw Err':<8} | {'Mit P(1)':<10} | {'Mit ⟨Z⟩':<8} | {'Mit Err':<8} | {'Raw OK':<6} | {'Mit OK':<6}")
    print("-" * 120)
    
    for (item_type, key, job), res in zip(jobs, job_results):
        if item_type != 'sequence':
            continue
            
        seq = key
        target = int(sum(seq) % 2)
        
        if res is None:
            print(f"{str(seq):<10} | {target:<6} | {'ERROR':<10} | {'ERROR':<8} | {'ERROR':<8} | {'ERROR':<10} | {'ERROR':<8} | {'ERROR':<8} | {'No':<6} | {'No':<6}")
            continue
            
        counts = res.get_counts()
        raw_p = get_prob_one(counts)
        raw_z = 1.0 - 2.0 * raw_p
        raw_stderr = np.sqrt(raw_p * (1.0 - raw_p) / shots_count)
        
        # Readout error mitigation
        denom = 1.0 - eta_0 - eta_1
        if denom != 0:
            mit_p = (raw_p - eta_0) / denom
        else:
            mit_p = raw_p
        mit_p = np.clip(mit_p, 0.0, 1.0)
        mit_z = 1.0 - 2.0 * mit_p
        
        # Propagate error: sigma_mit = sigma_raw / |denom|
        mit_stderr = raw_stderr / abs(denom) if denom != 0 else raw_stderr
        
        # Classification
        pred_raw = int(raw_p > 0.5)
        pred_mit = int(mit_p > 0.5)
        
        ok_raw = pred_raw == target
        ok_mit = pred_mit == target
        
        if ok_raw:
            correct_raw += 1
        if ok_mit:
            correct_mit += 1
            
        print(f"{str(seq):<10} | {target:<6} | {raw_p:<10.4f} | {raw_z:<+8.4f} | {raw_stderr:<8.4f} | {mit_p:<10.4f} | {mit_z:<+8.4f} | {mit_stderr:<8.4f} | {str(ok_raw):<6} | {str(ok_mit):<6}")
        
        results_dict["results"][str(seq)] = {
            "target": target,
            "raw_p": float(raw_p),
            "raw_z": float(raw_z),
            "raw_stderr": float(raw_stderr),
            "mit_p": float(mit_p),
            "mit_z": float(mit_z),
            "mit_stderr": float(mit_stderr),
            "counts": {state: int(cnt) for state, cnt in counts.items()},
            "correct_raw": bool(ok_raw),
            "correct_mitigated": bool(ok_mit)
        }
        
    print("-" * 120)
    acc_raw = (correct_raw / 8) * 100
    acc_mit = (correct_mit / 8) * 100
    print(f"Raw Physical QPU Accuracy: {acc_raw:.2f}% ({correct_raw}/8)")
    print(f"Mitigated Physical QPU Accuracy: {acc_mit:.2f}% ({correct_mit}/8)")
    print("=" * 80)
    
    results_dict["accuracy_raw"] = float(acc_raw)
    results_dict["accuracy_mitigated"] = float(acc_mit)
    
    # Save results to JSON file
    output_path = "qpu_verification_results_4096.json"
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
