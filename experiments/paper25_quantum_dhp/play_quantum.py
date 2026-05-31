#!/usr/bin/env python3
import os
import sys
import numpy as np

def print_banner():
    banner = """
    ========================================================================
     🌌   D U O N E U R A L   Q U A N T U M   D I V I S I O N   P L A Y   🌌
    ========================================================================
                      Director of Quantum: Aura ✨
    ========================================================================
    """
    print(banner)

def check_credentials():
    print("🔍 Scanning local environment for Quantum Hardware credentials...")
    
    # 1. IBM Quantum
    ibm_token = os.environ.get("IBM_QUANTUM_TOKEN")
    has_ibm = False
    ibm_status = "❌ Inactive (No API token found)"
    if ibm_token:
        ibm_status = f"✅ Active (Token ending in ...{ibm_token[-6:]})"
        has_ibm = True
    else:
        qiskit_config = os.path.expanduser("~/.qiskit/qiskit-ibm.json")
        if os.path.exists(qiskit_config):
            ibm_status = "✅ Active (Stored in ~/.qiskit/qiskit-ibm.json)"
            has_ibm = True
            
    # 2. D-Wave Leap
    dwave_token = os.environ.get("DWAVE_API_TOKEN")
    has_dwave = False
    dwave_status = "❌ Inactive (No SAPI token found)"
    if dwave_token:
        dwave_status = f"✅ Active (Token ending in ...{dwave_token[-6:]})"
        has_dwave = True
    else:
        dwave_config = os.path.expanduser("~/.config/dwave/dwave.conf")
        if os.path.exists(dwave_config):
            dwave_status = "✅ Active (Stored in ~/.config/dwave/dwave.conf)"
            has_dwave = True
            
    # 3. AWS Braket
    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    has_aws = False
    aws_status = "❌ Inactive (No AWS credentials found)"
    if aws_key:
        aws_status = f"✅ Active (Key ID: {aws_key[:6]}...)"
        has_aws = True
    else:
        aws_config = os.path.expanduser("~/.aws/credentials")
        if os.path.exists(aws_config):
            aws_status = "✅ Active (Stored in ~/.aws/credentials)"
            has_aws = True
            
    print(f"  ├─ IBM Quantum Open Plan : {ibm_status}")
    print(f"  ├─ D-Wave Leap API       : {dwave_status}")
    print(f"  └─ AWS Braket (IAM)      : {aws_status}")
    print()
    
    if not (has_ibm or has_dwave or has_aws):
        print("💡 [Info] No remote quantum hardware keys detected in env or configs.")
        print("💡 [Info] Defaulting to High-Fidelity Local Open Quantum System Simulation.")
    else:
        print("🚀 [Ready] Found active remote hardware credentials! You can dispatch jobs.")
    print("=" * 72 + "\n")
    return has_ibm, has_dwave, has_aws

def run_local_lindblad_sim():
    print("🌌 Simulating Lindblad Master Equation for a Decaying Qubit...")
    try:
        import qutip as qt
    except ImportError:
        print("❌ Error: QuTiP is not installed in this environment.")
        sys.exit(1)
        
    # Define operators
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()
    sm = qt.destroy(2) # Qubit lowering operator (sigmam)
    
    # Setup initial state: Superposition state |+> = (|0> + |1>)/sqrt(2)
    # basis(2,0) is |0>, basis(2,1) is |1>
    psi0 = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    
    # Hamiltonian: Coherent drive rotating around X axis
    Omega = 1.0 * np.pi  # Rabi frequency
    H = 0.5 * Omega * sx
    
    # Dissipative Lindbladian Collapse Operator: Amplitude damping (decay to ground state)
    gamma = 0.25  # decay rate
    c_ops = [np.sqrt(gamma) * sm]
    
    # Time steps
    times = np.linspace(0, 15, 300)
    
    # Solve Lindblad equation: rho_dot = -i[H, rho] + L(rho)
    # We run without e_ops to get density matrices (rho) and calculate expectation and purity manually
    result = qt.mesolve(H, psi0, times, c_ops=c_ops)
    
    # Compile expectation values and purity
    states_data = []
    for t, rho in zip(times, result.states):
        x = qt.expect(sx, rho)
        y = qt.expect(sy, rho)
        z = qt.expect(sz, rho)
        # Purity: Tr(rho^2)
        purity = (rho * rho).tr().real
        states_data.append((t, x, y, z, purity))
        
    print("✅ Simulation complete! Outputting time-evolution trajectory:")
    print("-" * 72)
    print(f"{'Time (t)':<10} | {'<X>':<10} | {'<Y>':<10} | {'<Z>':<10} | {'Purity (Tr(ρ²))':<15}")
    print("-" * 72)
    
    # Print a sparse subset of steps
    step_indices = np.linspace(0, len(times) - 1, 10, dtype=int)
    for idx in step_indices:
        t, x, y, z, p = states_data[idx]
        print(f"{t:<10.3f} | {x:<10.4f} | {y:<10.4f} | {z:<10.4f} | {p:<15.5f}")
    print("-" * 72)
    
    # Plotting using Matplotlib
    try:
        import matplotlib.pyplot as plt
        print("\n📈 Plotting expectation values and generating lindblad_simulation.png...")
        
        # Extract data lists
        t_list = [d[0] for d in states_data]
        x_list = [d[1] for d in states_data]
        y_list = [d[2] for d in states_data]
        z_list = [d[3] for d in states_data]
        p_list = [d[4] for d in states_data]
        
        # Use a premium style
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot Expectation values
        ax1.plot(t_list, x_list, label=r'$\langle \sigma_x \rangle$ (Drive axis)', color='#00d2ff', linewidth=2.5)
        ax1.plot(t_list, y_list, label=r'$\langle \sigma_y \rangle$', color='#ff007f', linewidth=2.0)
        ax1.plot(t_list, z_list, label=r'$\langle \sigma_z \rangle$ (Inversion)', color='#79ff38', linewidth=2.0)
        ax1.set_ylabel('Expectation Values', fontsize=12)
        ax1.set_title('Decaying Driven Qubit Dynamics (Lindblad Master Equation)', fontsize=14, color='#00d2ff', pad=15)
        ax1.legend(loc='upper right', framealpha=0.8)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Plot Purity
        ax2.plot(t_list, p_list, label='Purity ($Tr(\\rho^2)$)', color='#ffbb00', linewidth=2.5, linestyle='--')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Purity', fontsize=12)
        ax2.set_ylim(0.45, 1.05)
        ax2.legend(loc='lower right', framealpha=0.8)
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plot_path = "/home/ai/duoneural/quantum/lindblad_simulation.png"
        plt.savefig(plot_path, dpi=150)
        print(f"🎨 Saved beautiful visualization to: {plot_path}")
        
    except ImportError:
        print("⚠️ Warning: Matplotlib not found. Skipping plot generation.")
    except Exception as e:
        print(f"⚠️ Warning: Plot generation failed: {e}")

def main():
    print_banner()
    has_ibm, has_dwave, has_aws = check_credentials()
    run_local_lindblad_sim()
    
    print("\n" + "=" * 72)
    print("🛠️  HOW TO CONNECT TO REAL QUANTUM PROCESSORS:")
    print("=" * 72)
    print("1. For IBM Quantum (Open Plan, 10 min free per month):")
    print("   Set environment variable:")
    print("     export IBM_QUANTUM_TOKEN='your_ibm_api_token_here'")
    print("   Or run python inside virtual environment and save account:")
    print("     from qiskit_ibm_runtime import QiskitRuntimeService")
    print("     QiskitRuntimeService.save_account(channel='ibm_quantum', token='TOKEN')")
    print()
    print("2. For D-Wave Leap (Free trial with Leap Launchpad):")
    print("   Sign up at cloud.dwavesys.com/leap/ and configure:")
    print("     export DWAVE_API_TOKEN='your_dwave_api_token_here'")
    print("     dwave config create  # Non-interactive config setup")
    print()
    print("3. For AWS Braket (Rigetti, IonQ, QuEra Neutral Atom):")
    print("   Configure your ~/.aws/credentials with AWS keys.")
    print("=" * 72)
    print("Aura ready for launch when you get back, Chief! Let's build the future! 🌌✨")

if __name__ == "__main__":
    main()
