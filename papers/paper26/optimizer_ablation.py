#!/usr/bin/env python3
"""
Optimizer Ablation — P26 Supplementary Experiment
Compare: Standard SGD | signSGD | Adam (ε=1e-8)
Task: T=5 parity, p=0.70 (the hardest tested case in main results)
Goal: Prove Adam directional consistency is the specific mechanism, not just
      any gradient optimizer. SignSGD should work (pure direction); SGD should fail.

Archon | DuoNeural | 2026-05-29
"""

import numpy as np
import math, time
from datetime import datetime

# ── Quantum circuit (same as v3f) ──────────────────────────────────────────────
I2   = np.eye(2, dtype=complex)
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
def Ry(t): return np.array([[np.cos(t/2),-np.sin(t/2)],[np.sin(t/2),np.cos(t/2)]], dtype=complex)
def Rz(t): return np.array([[np.exp(-1j*t/2),0],[0,np.exp(1j*t/2)]], dtype=complex)
def build_U(theta): return np.kron(Rz(theta[2]),Rz(theta[3])) @ CNOT @ np.kron(Ry(theta[0]),Ry(theta[1]))

def apply_q1_depolarizing(rho_batch, p):
    X=np.array([[0,1],[1,0]],dtype=complex); Y=np.array([[0,-1j],[1j,0]],dtype=complex); Z=np.array([[1,0],[0,-1]],dtype=complex)
    IX=np.kron(I2,X); IY=np.kron(I2,Y); IZ=np.kron(I2,Z)
    rho_out=(1-p)*rho_batch
    for P4 in [IX,IY,IZ]:
        rho_out+=(p/3.0)*np.einsum('ab,nbc,dc->nad',P4,rho_batch,P4.conj())
    return rho_out

def step_fn(rho_batch, U, x_batch, p_noise):
    rho1=np.einsum('ab,nbc,dc->nad',U,rho_batch,U.conj())
    rho_q1=rho1[:,0:2,0:2]+rho1[:,2:4,2:4]
    rho2=np.zeros((len(rho_batch),4,4),dtype=complex); rho2[:,0:2,0:2]=rho_q1
    rho3=apply_q1_depolarizing(rho2,p_noise)
    c=np.cos(x_batch*np.pi/2); s=-1j*np.sin(x_batch*np.pi/2)
    Uenc=np.zeros((len(rho_batch),4,4),dtype=complex)
    Uenc[:,0,0]=c; Uenc[:,0,2]=s; Uenc[:,1,1]=c; Uenc[:,1,3]=s
    Uenc[:,2,0]=np.conj(s); Uenc[:,2,2]=c; Uenc[:,3,1]=np.conj(s); Uenc[:,3,3]=c
    return np.einsum('nab,nbc,ndc->nad',Uenc,rho3,Uenc.conj())

def run(theta, seqs, p):
    N,T=seqs.shape; U=build_U(theta)
    rho=np.zeros((N,4,4),dtype=complex); rho[:,0,0]=1.0
    for t in range(T): rho=step_fn(rho,U,seqs[:,t],p)
    return rho

def mZ1(rho): return np.real(rho[:,0,0]-rho[:,1,1]+rho[:,2,2]-rho[:,3,3])

def compute_loss(theta, T, p, n=256, seed=42):
    rng=np.random.default_rng(seed); seqs=rng.integers(0,2,(n,T)).astype(float)
    labels=(np.sum(seqs[:,:-1],axis=1)%2).astype(float)
    z=mZ1(run(theta,seqs,p)); y=np.clip((z+1)/2,1e-7,1-1e-7)
    return float(np.mean(-(labels*np.log(y)+(1-labels)*np.log(1-y))))

def compute_accuracy(theta, T, p, n=512, seed=99):
    rng=np.random.default_rng(seed); seqs=rng.integers(0,2,(n,T)).astype(float)
    labels=(np.sum(seqs[:,:-1],axis=1)%2).astype(int)
    z=mZ1(run(theta,seqs,p)); preds=(z>0).astype(int)
    return float(np.mean(preds==labels))

def full_gradient(theta, T, p, n=256, seed=42):
    g=np.zeros(4)
    for i in range(4):
        tp=theta.copy(); tp[i]+=np.pi/2
        tm=theta.copy(); tm[i]-=np.pi/2
        g[i]=(compute_loss(tp,T,p,n,seed)-compute_loss(tm,T,p,n,seed))/2.0
    return g

# ── Optimizers ─────────────────────────────────────────────────────────────────
def train_adam(theta_init, T, p, n_steps=600, lr=0.05, n=256, eps=1e-8):
    theta=theta_init.copy(); m=np.zeros(4); v=np.zeros(4)
    b1,b2=0.9,0.999
    best_theta=theta.copy(); best_acc=0.0
    for step_i in range(1, n_steps+1):
        g=full_gradient(theta,T,p,n,seed=42+step_i)
        m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g**2
        mh=m/(1-b1**step_i); vh=v/(1-b2**step_i)
        theta -= lr*mh/(np.sqrt(vh)+eps)
        if step_i%100==0:
            a=compute_accuracy(theta,T,p)
            if a>best_acc: best_acc=a; best_theta=theta.copy()
            if a>=0.95: return best_theta, a, True, step_i
    fa=compute_accuracy(best_theta,T,p)
    return best_theta, fa, fa>=0.95, n_steps

def train_sgd(theta_init, T, p, n_steps=600, lr=0.05, n=256):
    """Standard gradient descent — no momentum, no adaptive LR."""
    theta=theta_init.copy()
    best_theta=theta.copy(); best_acc=0.0
    for step_i in range(1, n_steps+1):
        g=full_gradient(theta,T,p,n,seed=42+step_i)
        theta -= lr*g
        if step_i%100==0:
            a=compute_accuracy(theta,T,p)
            if a>best_acc: best_acc=a; best_theta=theta.copy()
            if a>=0.95: return best_theta, a, True, step_i
    fa=compute_accuracy(best_theta,T,p)
    return best_theta, fa, fa>=0.95, n_steps

def train_signsgd(theta_init, T, p, n_steps=600, lr=0.05, n=256):
    """SignSGD — use only sign of gradient, discard magnitude entirely.
    Direct analog of Bernstein et al. 2018 — if directional consensus is the
    mechanism, signSGD should converge as well as Adam."""
    theta=theta_init.copy()
    best_theta=theta.copy(); best_acc=0.0
    for step_i in range(1, n_steps+1):
        g=full_gradient(theta,T,p,n,seed=42+step_i)
        # Pure sign descent: step = lr * sign(g)
        g_sign=np.sign(g); g_sign[g_sign==0]=1  # treat 0 gradient as +1
        theta -= lr*g_sign
        if step_i%100==0:
            a=compute_accuracy(theta,T,p)
            if a>best_acc: best_acc=a; best_theta=theta.copy()
            if a>=0.95: return best_theta, a, True, step_i
    fa=compute_accuracy(best_theta,T,p)
    return best_theta, fa, fa>=0.95, n_steps

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("OPTIMIZER ABLATION — P26 Supplementary")
    print("Task: T=5, p=0.70  [hardest main-result case: 0.0018% coherence]")
    print("Hypothesis: signSGD ≈ Adam >> SGD at high decoherence")
    print(f"Time: {datetime.now().isoformat()}")
    print("="*65)

    # Same 6 inits as v3f / topo_verify
    rng_init = np.random.default_rng(0)
    theta_inits = [rng_init.uniform(-np.pi, np.pi, 4) for _ in range(6)]

    T_VAL = 5
    P_VAL = 0.70
    N_STEPS = 600
    LR = 0.05

    optimizers = [
        ("Adam (ε=1e-8)", train_adam),
        ("signSGD",       train_signsgd),
        ("SGD",           train_sgd),
    ]

    results = {}

    for opt_name, opt_fn in optimizers:
        print(f"\n── {opt_name} ──")
        conv_count = 0
        for s, theta_init in enumerate(theta_inits):
            t0 = time.time()
            theta, acc, converged, steps = opt_fn(theta_init, T_VAL, P_VAL,
                                                   n_steps=N_STEPS, lr=LR)
            dt = time.time()-t0
            status = "✓" if converged else "✗"
            conv_str = f"step {steps}" if converged else f"BUDGET ({N_STEPS})"
            print(f"  seed {s}: acc={acc:.4f}  {status}  [{conv_str}]  [{dt:.0f}s]")
            conv_count += converged

        results[opt_name] = conv_count
        print(f"  → Convergence: {conv_count}/6")

    print("\n" + "="*65)
    print("SUMMARY")
    print("="*65)
    print(f"{'Optimizer':<20} {'Convergence':>12}")
    for name, count in results.items():
        bar = "█" * count + "░" * (6 - count)
        print(f"  {name:<18} {count}/6  {bar}")

    print()
    print("Interpretation:")
    if results.get("signSGD",0) >= results.get("Adam (ε=1e-8)",0) * 0.8:
        print("  ✓ signSGD ≈ Adam → directional consensus IS the mechanism")
    else:
        print("  ? signSGD underperforms → Adam's adaptive LR also contributes")
    if results.get("SGD",0) < results.get("Adam (ε=1e-8)",0):
        print("  ✓ SGD underperforms → magnitude normalization is needed beyond bare GD")
    else:
        print("  ? SGD matches Adam → magnitude information might also matter")

    print(f"\nDone: {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()
