import sys

def test_imports():
    libs = ['qiskit', 'qiskit_ibm_runtime', 'qiskit_aer', 'pennylane', 'qiskit_alice_bob_provider', 'dwave.system', 'bloqade', 'pulser', 'perceval', 'qutip', 'qutip_jax', 'jax']
    success = True
    for lib in libs:
        try:
            __import__(lib)
            print(f"✅ {lib}: Successfully imported.")
        except Exception as e:
            print(f"❌ {lib}: Failed to import. Error: {e}")
            success = False
    
    if success:
        print("\n🎉 All core quantum libraries imported successfully! Environment is ready for action.")
    else:
        print("\n⚠️ Some packages failed to import. Please check details above.")

if __name__ == "__main__":
    print(f"Running Environment Check under Python: {sys.executable}\n")
    test_imports()
