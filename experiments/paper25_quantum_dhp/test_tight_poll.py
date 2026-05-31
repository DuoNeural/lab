import qiskit
import bluequbit
import time

TOKEN = "FIysMm29ifkmSQeBZcE2N3kPGBOLcMkj"
bq_client = bluequbit.init(TOKEN)

qc1 = qiskit.QuantumCircuit(1)
qc1.x(0)
qc1.measure_all()

qc2 = qiskit.QuantumCircuit(1)
qc2.measure_all()

batch = [qc1, qc2]

print("⚡ Submitting batch asynchronously...")
jobs = bq_client.run(batch, device="cpu", asynchronous=True)

print("⏳ Tight polling for completion (50ms intervals)...")
start = time.time()
while True:
    # Query server for all jobs in batch
    updated_jobs = bq_client._get(jobs, need_qc_unprocessed=False)
    # Check if all completed
    if all(job.run_status in ['COMPLETED', 'FAILED'] for job in updated_jobs):
        jobs = updated_jobs
        break
    time.sleep(0.05)
    
elapsed = time.time() - start
print(f"🎉 All jobs completed in {elapsed:.4f} seconds!")
for idx, job in enumerate(jobs):
    print(f"Job {idx} counts: {job.get_counts()}")
