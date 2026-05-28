import qiskit
import bluequbit

TOKEN = "FIysMm29ifkmSQeBZcE2N3kPGBOLcMkj"
bq_client = bluequbit.init(TOKEN)

qc = qiskit.QuantumCircuit(1)
qc.x(0)
qc.measure_all()

print("⚡ Submitting job asynchronously...")
job = bq_client.run(qc, device="cpu", asynchronous=True)
print(f"Initial run status: {job.run_status}")

print("⏳ Waiting for job via bq_client.wait...")
completed_job = bq_client.wait(job.job_id)
print(f"Completed job type: {type(completed_job)}")
if completed_job is not None:
    print(f"Completed job attributes: {dir(completed_job)}")
    try:
        print(f"Completed job status: {completed_job.run_status}")
        print(f"Completed job counts: {completed_job.get_counts()}")
    except Exception as e:
        print(f"Error on completed_job: {e}")

try:
    print(f"Original job status: {job.run_status}")
    print(f"Original job counts: {job.get_counts()}")
except Exception as e:
    print(f"Error on original job: {e}")
