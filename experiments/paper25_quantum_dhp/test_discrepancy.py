import qiskit
import bluequbit
import numpy as np

TOKEN = "FIysMm29ifkmSQeBZcE2N3kPGBOLcMkj"
bq_client = bluequbit.init(TOKEN)

# Best weights from the Adam run
theta = np.array([-2.88306388,  3.23250803,  1.63857646,  0.63482619])

# The problematic sequence [0, 0, 1]
seq = [0, 0, 1]

def build_circuit(seq, theta):
    qc = qiskit.QuantumCircuit(2)
    for x_t in seq:
        qc.reset(0)
        qc.rx(x_t * np.pi, 0)
        qc.ry(theta[0], 0)
        qc.ry(theta[1], 1)
        qc.cx(0, 1)
        qc.rz(theta[2], 0)
        qc.rz(theta[3], 1)
    qc.measure_all()
    return qc

qc = build_circuit(seq, theta)

print("⚡ Running [0, 0, 1] on BlueQubit CPU simulator...")
res_cpu = bq_client.run(qc, device="cpu")
counts_cpu = res_cpu.get_counts()
prob_cpu = sum(prob for state, prob in counts_cpu.items() if state[0] == '1')
print(f"CPU Counts: {counts_cpu} | Prob(q1=1): {prob_cpu:.4f}")

print("\n⚡ Running [0, 0, 1] on BlueQubit GPU simulator...")
res_gpu = bq_client.run(qc, device="gpu")
counts_gpu = res_gpu.get_counts()
prob_gpu = sum(prob for state, prob in counts_gpu.items() if state[0] == '1')
print(f"GPU Counts: {counts_gpu} | Prob(q1=1): {prob_gpu:.4f}")
