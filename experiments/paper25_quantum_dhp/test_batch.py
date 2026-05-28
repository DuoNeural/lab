import qiskit
import bluequbit

TOKEN = "FIysMm29ifkmSQeBZcE2N3kPGBOLcMkj"
bq_client = bluequbit.init(TOKEN)

qc1 = qiskit.QuantumCircuit(1)
qc1.x(0)
qc1.measure_all()

qc2 = qiskit.QuantumCircuit(1)
qc2.measure_all()

print("⚡ Running batch of 2 circuits...")
res = bq_client.run([qc1, qc2], device="cpu")

print(f"Res type: {type(res)}")
try:
    print(f"Res counts: {res.get_counts()}")
except Exception as e:
    print(f"Error calling get_counts() directly on res: {e}")
    try:
        # Check if it is a list of results
        for idx, item in enumerate(res):
            print(f"Item {idx} type: {type(item)}")
            print(f"Item {idx} counts: {item.get_counts()}")
    except Exception as e2:
        print(f"Error iterating: {e2}")
