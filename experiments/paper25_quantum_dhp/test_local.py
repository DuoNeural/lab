import qiskit
import bluequbit
import time

TOKEN = "FIysMm29ifkmSQeBZcE2N3kPGBOLcMkj"
# Initialize in local execution mode
bq_client = bluequbit.init(TOKEN, execution_mode="local")

circuits = []
for _ in range(72):
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    circuits.append(qc)

print("⚡ Running 72 circuits locally in batches of 5...")
start = time.time()
results = []
for i in range(0, len(circuits), 5):
    res_batch = bq_client.run(circuits[i:i+5], device="cpu")
    results.extend(res_batch)
elapsed = time.time() - start

print(f"🎉 Completed in {elapsed:.4f} seconds!")
print(f"Result type: {type(results)}")
print(f"First result counts: {results[0].get_counts()}")
