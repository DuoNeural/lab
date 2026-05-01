#!/usr/bin/env python3
"""
CTM World Model v33b — Universal Noise Scaling Curve (CORRECTED)
Archon / DuoNeural 2026-05-01

BUGFIX from v33: lorenz_step was using dt=0.01, giving τ_L=110 steps discrete.
Paper (and v28) use dt=0.05, giving τ_L=22 steps. T_GATE=32 spans the Lyapunov
horizon only at dt=0.05. v33 results were therefore confounded — the clean
baseline never reached τ_L, so the σ-curve was measuring "window ceiling" not
"Lyapunov horizon contraction."

This run: identical to v33 except dt=0.05 (matching v28 and paper Section 2.2).
Expected clean baseline: τ* ≈ 23.5 (reproducing v28). THEN the σ-curve should
show genuine logarithmic contraction from the actual Lyapunov horizon.

    σ ∈ {0.0, 0.01, 0.05, 0.1, 0.2, 0.5}

Hypothesis: τ*(σ) ≈ τ_L - (1/λ_max) * ln(σ_signal²/σ_noise²)
           = 22  - (1/0.906) * ln(81/σ²)

DT = 0.05  ← THE FIX
"""

import torch, numpy as np, json, os, math, time
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

# ── Config ─────────────────────────────────────────────────────────────────────
N_OBJ        = 8
T_GATE       = 32
TRAIN_STEPS  = 60_000       # match v28 for consistency
BATCH_SIZE   = 128
K_PRED       = 1
HIDDEN_DIM   = 128
N_SLOTS      = 4
OBJ_DIM      = 3

DT           = 0.05         # FIXED: matches paper Section 2.2 and v28
TAU_L        = 22.0         # τ_L = 1/0.906 / 0.05 = 22.0 discrete steps

SIGMA_SWEEP  = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

V28_CLEAN_DELAY = 23.5      # v28 reference — should reproduce with dt=0.05

LOG_FILE = os.path.expanduser("~/duoneural/ctm_world_model_v33b/wm_v33b.log")
OUT_DIR  = os.path.expanduser("~/duoneural/ctm_world_model_v33b")
os.makedirs(OUT_DIR, exist_ok=True)

def ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def log(msg):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + "\n")

# ── Lorenz (dt=0.05 — THE FIX) ─────────────────────────────────────────────────
def lorenz_step(state, dt=DT, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """RK4 integration at dt=0.05. τ_L = 1/(λ_max * dt) = 1/(0.906*0.05) ≈ 22 steps."""
    def deriv(s):
        x, y, z = s[...,0], s[...,1], s[...,2]
        return torch.stack([sigma*(y-x), x*(rho-z)-y, x*y-beta*z], dim=-1)
    k1 = deriv(state)
    k2 = deriv(state + 0.5*dt*k1)
    k3 = deriv(state + 0.5*dt*k2)
    k4 = deriv(state + dt*k3)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def generate_lorenz_batch(batch_size, seq_len, sigma_obs=0.0):
    state = torch.randn(batch_size, N_OBJ, 3) * 5.0
    state[..., 2] += 25.0
    for _ in range(500):
        state = lorenz_step(state)
    traj = []
    for _ in range(seq_len):
        state = lorenz_step(state)
        traj.append(state.clone())
    traj = torch.stack(traj, dim=1)
    norms = torch.tensor([9.0, 9.0, 8.5]).view(1, 1, 1, 3)
    traj = traj / norms
    if sigma_obs > 0.0:
        traj = traj + sigma_obs * torch.randn_like(traj)
    return traj

# ── Architecture (unchanged from v28/v33) ──────────────────────────────────────
class LearnedTemporalGateEncoder(nn.Module):
    def __init__(self, t_gate, obj_dim, hidden_dim):
        super().__init__()
        self.t_gate = t_gate
        self.gate_logits = nn.Parameter(torch.zeros(t_gate))
        self.encoder = nn.Sequential(
            nn.Linear(obj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    def forward(self, history):
        B, T, N, D = history.shape
        gates    = torch.softmax(self.gate_logits, dim=0)
        enc_flat = self.encoder(history.reshape(B*T*N, D))
        enc      = enc_flat.reshape(B, T, N, -1)
        out      = (enc * gates.view(1, T, 1, 1)).sum(dim=1)
        return out, gates

class SlotDynamics(nn.Module):
    def __init__(self, hidden_dim, obj_dim):
        super().__init__()
        self.attn    = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm1   = nn.LayerNorm(hidden_dim)
        self.ff      = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.GELU(), nn.Linear(hidden_dim*2, hidden_dim))
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, obj_dim)
    def forward(self, enc):
        x, _ = self.attn(enc, enc, enc)
        x = self.norm1(enc + x)
        x = self.norm2(x + self.ff(x))
        return self.decoder(x)

class LorenzCTM(nn.Module):
    def __init__(self, t_gate):
        super().__init__()
        self.gate_enc = LearnedTemporalGateEncoder(t_gate, OBJ_DIM, HIDDEN_DIM)
        self.dynamics = SlotDynamics(HIDDEN_DIM, OBJ_DIM)
    def forward(self, history):
        enc, gates = self.gate_enc(history)
        return self.dynamics(enc), gates

def analyze_gates(model):
    with torch.no_grad():
        g        = torch.softmax(model.gate_enc.gate_logits, dim=0).cpu().numpy()
        delays   = np.arange(T_GATE)[::-1].copy()
        eff_delay = float(np.sum(delays * g))
        peak_idx  = int(np.argmax(g))
        peak_delay = T_GATE - 1 - peak_idx
        peak_prob  = float(g[peak_idx])
    return eff_delay, peak_delay, peak_prob, g

def theoretical_tau(sigma_obs, tau_L=TAU_L, lambda_max=0.906, sigma_signal=9.0):
    """τ*(σ) ≈ τ_L - (1/λ_max) * ln(σ_signal² / σ_obs²)"""
    if sigma_obs <= 0:
        return tau_L
    snr = (sigma_signal ** 2) / (sigma_obs ** 2)
    correction = (1.0 / lambda_max) * math.log(snr)
    return max(1.0, tau_L - correction)

# ── Training loop ──────────────────────────────────────────────────────────────
def run_sigma(sigma_obs):
    sigma_tag = f"s{str(sigma_obs).replace('.','p')}"
    log(f"\n{'='*60}")
    log(f"sigma_obs={sigma_obs:.3f}  (tag={sigma_tag})  dt={DT}")
    theoretical = theoretical_tau(sigma_obs)
    log(f"Theory predicts tau* ~= {theoretical:.1f} steps  (tau_L={TAU_L} at dt={DT})")
    log(f"{'='*60}")

    SEQ_LEN   = T_GATE + K_PRED + 20
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_v33b_{sigma_tag}.pt")

    model     = LorenzCTM(T_GATE).to(DEVICE)
    opt       = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TRAIN_STEPS)
    start_step = 0
    best_mse   = float('inf')

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        start_step = ckpt['step']
        best_mse   = ckpt.get('best_mse', float('inf'))
        log(f"  Resumed at step {start_step}")

    for step in range(start_step + 1, TRAIN_STEPS + 1):
        model.train()
        seq     = generate_lorenz_batch(BATCH_SIZE, SEQ_LEN, sigma_obs=sigma_obs).to(DEVICE)
        t0      = torch.randint(0, SEQ_LEN - T_GATE - K_PRED, (1,)).item()
        history = seq[:, t0:t0+T_GATE]
        target  = seq[:, t0+T_GATE+K_PRED-1]
        pred, _ = model(history)
        loss    = ((pred - target)**2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if loss.item() < best_mse:
            best_mse = loss.item()
        if step % 5000 == 0 or step == TRAIN_STEPS:
            eff_delay, peak_delay, peak_prob, _ = analyze_gates(model)
            delta = eff_delay - V28_CLEAN_DELAY
            log(f"  step {step:6d} | loss={loss.item():.6f} | eff_delay={eff_delay:.2f} | "
                f"peak@t-{peak_delay}({peak_prob:.3f}) | Dclean={delta:+.2f} | theory={theoretical:.1f}")
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'sched': sched.state_dict(), 'step': step, 'best_mse': best_mse}, ckpt_path)

    eff_delay, peak_delay, peak_prob, gate_dist = analyze_gates(model)
    theoretical = theoretical_tau(sigma_obs)
    delta_clean  = round(eff_delay - V28_CLEAN_DELAY, 3)
    delta_theory = round(eff_delay - theoretical, 3)

    log(f"\n  -- sigma={sigma_obs:.3f} FINAL --")
    log(f"  eff_delay:   {eff_delay:.2f}  (theory: {theoretical:.1f}, D={delta_theory:+.2f})")
    log(f"  D vs clean:  {delta_clean:+.3f}")
    log(f"  best_loss:   {best_mse:.8f}")

    return {
        "sigma_obs":       sigma_obs,
        "dt":              DT,
        "eff_delay":       round(eff_delay, 3),
        "peak_delay":      peak_delay,
        "peak_prob":       round(peak_prob, 4),
        "best_loss":       round(best_mse, 8),
        "theoretical_tau": round(theoretical, 2),
        "delta_vs_clean":  delta_clean,
        "delta_vs_theory": delta_theory,
        "gate_distribution": [round(float(x), 4) for x in gate_dist],
    }

# ── Main ───────────────────────────────────────────────────────────────────────
log("CTM World Model v33b — Universal Noise Scaling Curve (dt=0.05 CORRECTED)")
log(f"T_GATE={T_GATE} | dt={DT} | tau_L={TAU_L} steps | sigma sweep={SIGMA_SWEEP}")
log(f"TRAIN_STEPS={TRAIN_STEPS} (matches v28)")
log(f"Device: {DEVICE}")

results_path = os.path.join(OUT_DIR, "results_v33b.json")
all_results  = {}
if os.path.exists(results_path):
    with open(results_path) as f:
        all_results = json.load(f)
    log(f"Loaded existing: sigma keys done = {list(all_results.keys())}")

for sigma in SIGMA_SWEEP:
    key = str(sigma)
    if key in all_results and "eff_delay" in all_results[key]:
        log(f"sigma={sigma} already done, skipping")
        continue
    r = run_sigma(sigma)
    all_results[key] = r
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"[checkpoint] results_v33b.json updated (sigma={sigma} done)")

# ── Summary + log-linear fit ───────────────────────────────────────────────────
log(f"\n{'='*60}")
log("V33b COMPLETE — tau*(sigma) CURVE (CORRECTED dt=0.05)")
log(f"{'='*60}")
log(f"{'sigma_obs':>10} | {'tau_learned':>12} | {'tau_theory':>12} | {'D theory':>10} | {'D clean':>8}")
sigma_vals, tau_vals = [], []
for key in sorted(all_results.keys(), key=lambda x: float(x)):
    r = all_results[key]
    if not isinstance(r, dict) or "eff_delay" not in r:
        continue
    log(f"{float(key):>10.3f} | {r['eff_delay']:>12.2f} | {r['theoretical_tau']:>12.1f} | "
        f"{r['delta_vs_theory']:>+10.2f} | {r['delta_vs_clean']:>+8.3f}")
    sigma_vals.append(float(key) if float(key) > 0 else 1e-6)
    tau_vals.append(r['eff_delay'])

if len(tau_vals) >= 4:
    noisy_pairs = [(s, t) for s, t in zip(sigma_vals, tau_vals) if s > 1e-5]
    if len(noisy_pairs) >= 3:
        xs = np.array([math.log(s) for s, _ in noisy_pairs])
        ys = np.array([t for _, t in noisy_pairs])
        b  = np.cov(xs, ys)[0, 1] / np.var(xs)
        a  = np.mean(ys) - b * np.mean(xs)
        residuals = ys - (a + b * xs)
        r2 = 1 - np.var(residuals) / np.var(ys)
        log(f"\n  Log-linear fit: tau* = {a:.2f} + {b:.2f}*ln(sigma),  R2={r2:.4f}")
        theory_b = -1.0 / 0.906  # theoretical slope = -1/lambda_max
        log(f"  Theory predicts slope = {theory_b:.3f}, got {b:.3f}")
        if r2 > 0.90:
            log(f"  *** UNIVERSAL_SCALING_LAW: R2={r2:.4f} -- logarithmic decay confirmed ***")
            all_results["global_verdict"] = "UNIVERSAL_SCALING_LAW"
            all_results["log_fit"] = {"a": round(a,4), "b": round(b,4), "r2": round(r2,4),
                                       "theory_b": round(theory_b,4)}
        elif r2 > 0.70:
            log(f"  *** PARTIAL_LOG: R2={r2:.4f} -- approximately logarithmic ***")
            all_results["global_verdict"] = "PARTIAL_LOG"
        else:
            log(f"  *** NON_LOG: R2={r2:.4f} -- need different model ***")
            all_results["global_verdict"] = "NON_LOG"

with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2)
log("All results saved.")
