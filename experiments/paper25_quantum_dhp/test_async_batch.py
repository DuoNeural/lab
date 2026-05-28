import qiskit
import bluequbit

TOKEN = "FIysMm29ifkmSQeBZcE2N3kPGBOLcMkj"
bq_client = bluequbit.init(TOKEN)

qc1 = qiskit.QuantumCircuit(1)
qc1.x(0)
qc1.measure_all()

qc2 = qiskit.QuantumCircuit(1)
qc2.measure_all()

print("⚡ Submitting batch job asynchronously...")
job_list = bq_client.run([qc1, qc2], device="cpu", asynchronous=True)
print(f"Returned job_list type: {type(job_list)}")
for idx, job in enumerate(job_list):
    print(f"Job {idx} type: {type(job)}")
    print(f"Job {idx} status: {job.run_status}")
    print(f"Job {idx} ID: {job.job_id}")

print("⏳ Waiting for first job...")
completed_job_1 = bq_client.wait(job_list[0].job_id)
print(f"Job 1 counts: {completed_job_1.get_counts()}")

print("⏳ Waiting for second job...")
completed_job_2 = bq_client.wait(job_list[1].job_id)
print(f"Job 2 counts: {completed_job_2.get_counts()}")
