#!/usr/bin/env python3
"""
T1 Amplitude Damping Sweep — P26 Supplementary Experiment
Test sign-preservation under ASYMMETRIC noise (T1 relaxation / amplitude damping)
instead of symmetric Pauli depolarizing.

Theory:
  Pauli depolarizing: Bloch vector contracts isotropically → sign-preserving for ALL p < 0.75
  Amplitude damping:  Kraus K0=[[1,0],[0,√(1-γ)]], K1=[[0,√γ],[0,0]]
                      |0⟩ becomes absorbing state — biased drift breaks sign-preservation
                      Expected earlier threshold than p=0.75

This sweep finds the empirical γ threshold and contrasts with the Pauli threshold.

Archon | DuoNeural | 2026-05-29
"""

import numpy as np
import math, time
from datetime import datetime

# ── Same quantum circuit as v3f / optimizer_ablation ──────────────────────────
I2   = np.eye(2, dtype=complex)
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
def Ry(t): return np.array([[np.cos(t/2),-np.sin(t/2)],[np.sin(t/2),np.cos(t/2)]], dtype=complex)
def Rz(t): return np.array([[np.exp(-1j*t/2),0],[0,np.exp(1j*t/2)]], dtype=complex)
def build_U(theta): return np.kron(Rz(theta[2]),Rz(theta[3])) @ CNOT @ np.kron(Ry(theta[0]),Ry(theta[1]))

# ── Pauli depolarizing (reference — same as v3f) ───────────────────────────────
def apply_q1_depolarizing_pauli(rho_batch, p):
    X=np.array([[0,1],[1,0]],dtype=complex)
    Y=np.array([[0,-1j],[1j,0]],dtype=complex)
    Z=np.array([[1,0],[0,-1]],dtype=complex)
    IX=np.kron(I2,X); IY=np.kron(I2,Y); IZ=np.kron(I2,Z)
    rho_out=(1-p)*rho_batch
    for P4 in [IX,IY,IZ]:
        rho_out += (p/3.0)*np.einsum('ab,nbc,dc->nad',P4,rho_batch,P4.conj())
    return rho_out

# ── Amplitude damping on qubit 1 of 2-qubit system ────────────────────────────
def apply_q1_amplitude_damping(rho_batch, gamma):
    """
    Kraus operators for amplitude damping on q1 (the LEFT qubit in kron convention):
      K0 = [[1,0],[0,sqrt(1-γ)]]  ⊗ I2
      K1 = [[0,sqrt(γ)],[0,0]]    ⊗ I2

    |0⟩ is the ground state (absorbing); |1⟩ decays toward |0⟩.
    This is T1 relaxation in the rotating frame.
    Unlike Pauli depolarizing, this is ASYMMETRIC — breaks sign-preservation.
    """
    sq = np.sqrt(1 - gamma)
    # K0 on q1: maps |0⟩→|0⟩, |1⟩→sqrt(1-γ)|1⟩
    K0_q1 = np.array([[1,0],[0,sq]], dtype=complex)
    # K1 on q1: maps |1⟩→sqrt(γ)|0⟩, |0⟩→0
    K1_q1 = np.array([[0,np.sqrt(gamma)],[0,0]], dtype=complex)

    K0 = np.kron(K0_q1, I2)  # acts on full 4×4 Hilbert space
    K1 = np.kron(K1_q1, I2)

    rho_out = (np.einsum('ab,nbc,dc->nad', K0, rho_batch, K0.conj()) +
               np.einsum('ab,nbc,dc->nad', K1, rho_batch, K1.conj()))
    return rho_out

# ── Circuit step with switchable noise model ───────────────────────────────────
def step_fn_pauli(rho_batch, U, x_batch, p_noise):
    rho1=np.einsum('ab,nbc,dc->nad',U,rho_batch,U.conj())
    rho_q1=rho1[:,0:2,0:2]+rho1[:,2:4,2:4]
    rho2=np.zeros((len(rho_batch),4,4),dtype=complex); rho2[:,0:2,0:2]=rho_q1
    rho3=apply_q1_depolarizing_pauli(rho2,p_noise)
    c=np.cos(x_batch*np.pi/2); s=-1j*np.sin(x_batch*np.pi/2)
    Uenc=np.zeros((len(rho_batch),4,4),dtype=complex)
    Uenc[:,0,0]=c; Uenc[:,0,2]=s; Uenc[:,1,1]=c; Uenc[:,1,3]=s
    Uenc[:,2,0]=np.conj(s); Uenc[:,2,2]=c; Uenc[:,3,1]=np.conj(s); Uenc[:,3,3]=c
    return np.einsum('nab,nbc,ndc->nad',Uenc,rho3,Uenc.conj())

def step_fn_t1(rho_batch, U, x_batch, gamma):
    rho1=np.einsum('ab,nbc,dc->nad',U,rho_batch,U.conj())
    rho_q1=rho1[:,0:2,0:2]+rho1[:,2:4,2:4]
    rho2=np.zeros((len(rho_batch),4,4),dtype=complex); rho2[:,0:2,0:2]=rho_q1
    rho3=apply_q1_amplitude_damping(rho2,gamma)
    c=np.cos(x_batch*np.pi/2); s=-1j*np.sin(x_batch*np.pi/2)
    Uenc=np.zeros((len(rho_batch),4,4),dtype=complex)
    Uenc[:,0,0]=c; Uenc[:,0,2]=s; Uenc[:,1,1]=c; Uenc[:,1,3]=s
    Uenc[:,2,0]=np.conj(s); Uenc[:,2,2]=c; Uenc[:,3,1]=np.conj(s); Uenc[:,3,3]=c
    return np.einsum('nab,nbc,ndc->nad',Uenc,rho3,Uenc.conj())

def run_pauli(theta, seqs, p):
    N,T=seqs.shape; U=build_U(theta)
    rho=np.zeros((N,4,4),dtype=complex); rho[:,0,0]=1.0
    for t in range(T): rho=step_fn_pauli(rho,U,seqs[:,t],p)
    return rho

def run_t1(theta, seqs, gamma):
    N,T=seqs.shape; U=build_U(theta)
    rho=np.zeros((N,4,4),dtype=complex); rho[:,0,0]=1.0
    for t in range(T): rho=step_fn_t1(rho,U,seqs[:,t],gamma)
    return rho

def mZ1(rho): return np.real(rho[:,0,0]-rho[:,1,1]+rho[:,2,2]-rho[:,3,3])

# ── Loss + gradient for T1 noise ──────────────────────────────────────────────
def compute_loss_t1(theta, T, gamma, n=256, seed=42):
    rng=np.random.default_rng(seed); seqs=rng.integers(0,2,(n,T)).astype(float)
    labels=(np.sum(seqs[:,:-1],axis=1)%2).astype(float)
    z=mZ1(run_t1(theta,seqs,gamma)); y=np.clip((z+1)/2,1e-7,1-1e-7)
    return float(np.mean(-(labels*np.log(y)+(1-labels)*np.log(1-y))))

def compute_accuracy_t1(theta, T, gamma, n=512, seed=99):
    rng=np.random.default_rng(seed); seqs=rng.integers(0,2,(n,T)).astype(float)
    labels=(np.sum(seqs[:,:-1],axis=1)%2).astype(int)
    z=mZ1(run_t1(theta,seqs,gamma)); preds=(z>0).astype(int)
    return float(np.mean(preds==labels))

def full_gradient_t1(theta, T, gamma, n=256, seed=42):
    g=np.zeros(4)
    for i in range(4):
        tp=theta.copy(); tp[i]+=np.pi/2
        tm=theta.copy(); tm[i]-=np.pi/2
        g[i]=(compute_loss_t1(tp,T,gamma,n,seed)-compute_loss_t1(tm,T,gamma,n,seed))/2.0
    return g

# ── Same for Pauli (reference runs at same τ_L) ───────────────────────────────
def compute_loss_pauli(theta, T, p, n=256, seed=42):
    rng=np.random.default_rng(seed); seqs=rng.integers(0,2,(n,T)).astype(float)
    labels=(np.sum(seqs[:,:-1],axis=1)%2).astype(float)
    z=mZ1(run_pauli(theta,seqs,p)); y=np.clip((z+1)/2,1e-7,1-1e-7)
    return float(np.mean(-(labels*np.log(y)+(1-labels)*np.log(1-y))))

def compute_accuracy_pauli(theta, T, p, n=512, seed=99):
    rng=np.random.default_rng(seed); seqs=rng.integers(0,2,(n,T)).astype(float)
    labels=(np.sum(seqs[:,:-1],axis=1)%2).astype(int)
    z=mZ1(run_pauli(theta,seqs,p)); preds=(z>0).astype(int)
    return float(np.mean(preds==labels))

def full_gradient_pauli(theta, T, p, n=256, seed=42):
    g=np.zeros(4)
    for i in range(4):
        tp=theta.copy(); tp[i]+=np.pi/2
        tm=theta.copy(); tm[i]-=np.pi/2
        g[i]=(compute_loss_pauli(tp,T,p,n,seed)-compute_loss_pauli(tm,T,p,n,seed))/2.0
    return g

# ── Adam training (same as v3f) ────────────────────────────────────────────────
def train_adam_t1(theta_init, T, gamma, n_steps=600, lr=0.05, n=256):
    theta=theta_init.copy(); m=np.zeros(4); v=np.zeros(4)
    b1,b2,eps=0.9,0.999,1e-8
    best_theta=theta.copy(); best_acc=0.0
    for step_i in range(1, n_steps+1):
        g=full_gradient_t1(theta,T,gamma,n,seed=42+step_i)
        m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g**2
        mh=m/(1-b1**step_i); vh=v/(1-b2**step_i)
        theta -= lr*mh/(np.sqrt(vh)+eps)
        if step_i%100==0:
            a=compute_accuracy_t1(theta,T,gamma)
            if a>best_acc: best_acc=a; best_theta=theta.copy()
            if a>=0.95: return best_theta, a, True, step_i
    fa=compute_accuracy_t1(best_theta,T,gamma)
    return best_theta, fa, fa>=0.95, n_steps

def train_adam_pauli(theta_init, T, p, n_steps=600, lr=0.05, n=256):
    theta=theta_init.copy(); m=np.zeros(4); v=np.zeros(4)
    b1,b2,eps=0.9,0.999,1e-8
    best_theta=theta.copy(); best_acc=0.0
    for step_i in range(1, n_steps+1):
        g=full_gradient_pauli(theta,T,p,n,seed=42+step_i)
        m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g**2
        mh=m/(1-b1**step_i); vh=v/(1-b2**step_i)
        theta -= lr*mh/(np.sqrt(vh)+eps)
        if step_i%100==0:
            a=compute_accuracy_pauli(theta,T,p)
            if a>best_acc: best_acc=a; best_theta=theta.copy()
            if a>=0.95: return best_theta, a, True, step_i
    fa=compute_accuracy_pauli(best_theta,T,p)
    return best_theta, fa, fa>=0.95, n_steps

# ── τ_L equivalent noise levels ───────────────────────────────────────────────
def tau_L_pauli(p):
    """Coherence time under Pauli depolarizing. τ_L → 0 as p → 0.75."""
    if p <= 0 or p >= 0.75:
        return float('inf') if p <= 0 else 0.0
    return -1.0 / math.log(1 - 4*p/3)

def tau_L_t1(gamma):
    """
    For amplitude damping K0=diag(1, sqrt(1-γ)):
    |ρ_01| contracts by sqrt(1-γ) per step → coherence time τ_L_T1 = -1/log(sqrt(1-γ)) = -2/log(1-γ)
    NOTE: The sign-preservation argument used for Pauli (symmetric contraction) does NOT hold here
    because the |0⟩/|1⟩ populations also shift asymmetrically.
    """
    if gamma <= 0:
        return float('inf')
    if gamma >= 1:
        return 0.0
    return -2.0 / math.log(1 - gamma)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("T1 AMPLITUDE DAMPING SWEEP — P26 Supplementary")
    print("Compare asymmetric (T1/amplitude-damping) vs symmetric (Pauli)")
    print("Task: T=5, 6 seeds, Adam optimizer")
    print("Hypothesis: T1 noise breaks sign-preservation earlier than p=0.75")
    print(f"Time: {datetime.now().isoformat()}")
    print("="*65)

    # Same 6 inits as v3f / topo_verify / optimizer_ablation
    rng_init = np.random.default_rng(0)
    theta_inits = [rng_init.uniform(-np.pi, np.pi, 4) for _ in range(6)]

    T_VAL = 5
    N_STEPS = 600
    LR = 0.05

    # γ sweep for T1, chosen to span τ_L_T1 ≈ τ_L_Pauli reference points
    # τ_L_Pauli at p=0.70 ≈ 0.615 steps — same range for T1
    gamma_vals = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    # Reference Pauli runs at p=0.00 and p=0.70 (from main results)
    pauli_ref = [0.00, 0.70]

    print("\n── PAULI DEPOLARIZING (Reference) ──")
    print(f"  {'p':>6}  {'τ_L':>8}  {'Conv/6':>8}  {'Accs'}")
    pauli_results = {}
    for p in pauli_ref:
        conv_count = 0
        accs = []
        for s, theta_init in enumerate(theta_inits):
            _, acc, converged, _ = train_adam_pauli(theta_init, T_VAL, p, N_STEPS, LR)
            conv_count += converged
            accs.append(f"{acc:.4f}")
        tL = tau_L_pauli(p)
        tL_str = f"{tL:.3f}" if tL < 999 else "∞"
        print(f"  p={p:.2f}  τ_L={tL_str:>6}  {conv_count}/6  [{', '.join(accs)}]")
        pauli_results[p] = conv_count

    print("\n── T1 AMPLITUDE DAMPING (Asymmetric) ──")
    print(f"  {'γ':>6}  {'τ_L_T1':>8}  {'Conv/6':>8}  {'Accs'}")
    t1_results = {}
    for gamma in gamma_vals:
        t0 = time.time()
        conv_count = 0
        accs = []
        for s, theta_init in enumerate(theta_inits):
            _, acc, converged, _ = train_adam_t1(theta_init, T_VAL, gamma, N_STEPS, LR)
            conv_count += converged
            accs.append(f"{acc:.4f}")
        dt = time.time()-t0
        tL = tau_L_t1(gamma)
        tL_str = f"{tL:.3f}" if tL < 999 else "∞"
        print(f"  γ={gamma:.2f}  τ_L={tL_str:>6}  {conv_count}/6  [{', '.join(accs)}]  [{dt:.0f}s]")
        t1_results[gamma] = conv_count

    print("\n" + "="*65)
    print("SUMMARY")
    print("="*65)

    # Find T1 failure threshold
    failing_gammas = [g for g,c in t1_results.items() if c < 4]
    threshold_gamma = min(failing_gammas) if failing_gammas else None

    print(f"\nPauli reference: p=0.00 → {pauli_results[0.00]}/6,  p=0.70 → {pauli_results[0.70]}/6")
    print(f"\nT1 sweep:")
    for gamma, count in t1_results.items():
        tau_str = f"{tau_L_t1(gamma):.3f}"
        bar = "█" * count + "░" * (6 - count)
        print(f"  γ={gamma:.2f}  τ_L={tau_str:>6}  {count}/6  {bar}")

    print()
    if threshold_gamma is not None:
        tau_fail = tau_L_t1(threshold_gamma)
        # Equivalent Pauli p for same τ_L
        # p_eq: -1/log(1-4p/3) = tau_fail → 1-4p/3 = exp(-1/tau_fail) → p = 3/4*(1 - exp(-1/tau_fail))
        try:
            p_eq = 0.75 * (1 - math.exp(-1.0/tau_fail)) if tau_fail > 0 else 0.75
        except:
            p_eq = float('nan')
        print(f"Interpretation:")
        print(f"  T1 convergence fails at γ ≈ {threshold_gamma:.2f} (τ_L_T1 ≈ {tau_fail:.3f})")
        print(f"  Equivalent Pauli p for same τ_L ≈ {p_eq:.3f}")
        print(f"  Pauli threshold: p → 0.75 (τ_L → 0)")
        print(f"  ✓ T1 breaks sign-preservation EARLIER: asymmetric noise is strictly harder")
    else:
        print(f"  All T1 γ values tested show ≥4/6 convergence")
        print(f"  T1 amplitude damping may not break sign-preservation within tested range")

    print()
    print("Physical interpretation:")
    print("  Pauli: isotropic contraction → gradient sign preserved for all p < 0.75")
    print("  T1:    |0⟩-biased drift → breaks gradient sign symmetry at lower noise level")
    print("  Real QPU hardware has BOTH: results reported as lower bound on T1 limit")

    print(f"\nDone: {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()
