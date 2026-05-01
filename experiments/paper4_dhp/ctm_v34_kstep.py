#!/usr/bin/env python3
"""
CTM World Model v34 — Multi-Step Prediction Horizon Sweep
Archon / DuoNeural 2026-05-01

Motivated by Aura's review (rev1) and the Jesse/Archon open question:
"Does τ* scale with prediction horizon k?"

Setup:
  - 3D Lorenz, dt=0.05, T_GATE=32 (same as v28 — fully validated config)
  - Sweep k ∈ {1, 2, 4, 8}  (k-steps-ahead prediction target)
  - Train independent model for each k

Hypothesis (Dynamical Horizon Principle extension):
  For k-step-ahead prediction, the informationally optimal integration window
  should extend to cover k Lyapunov times:
      τ*(k) ≈ k * τ_L = k * 22

  At k=1: τ* ≈ 22  (reproduces v28)
  At k=2: τ* ≈ 44  (but capped at T_GATE=32 → expect τ* → T_GATE)
  At k=4: τ* → T_GATE=32  (wall)
  At k=8: τ* → T_GATE=32  (wall, prediction very hard)

So we also sweep T_GATE ∈ {32, 64} for k=2 to see if the window expands.

This directly tests whether the Dynamical Horizon Principle generalizes beyond k=1.
It also produces data relevant to Paper 5 and the robotics framing.
"""

import torch, numpy as np, json, os, math, time
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

# ── Config ─────────────────────────────────────────────────────────────────────
N_OBJ        = 8
DT           = 0.05          # matches paper + v28 + v33b
TAU_L        = 22.0          # Lyapunov time at dt=0.05
TRAIN_STEPS  = 60_000
BATCH_SIZE   = 128
HIDDEN_DIM   = 128
OBJ_DIM      = 3

# Sweep: (k_pred, t_gate) pairs
# k=1,2,4,8 at T_GATE=32; also k=2 at T_GATE=64 to test if horizon expands
K_TGATE_SWEEP = [
    (1,  32),   # baseline — should reproduce τ*≈23.5
    (2,  32),   # 2-step: τ* should push toward 2*22=44 but cap at 32
    (2,  64),   # 2-step with bigger window: should τ* → ~44?
    (4,  32),   # 4-step: τ* → T_GATE=32 (saturated)
    (8,  32),   # 8-step: τ* → T_GATE=32 (saturated, but test MSE cliff)
]

LOG_DIR  = "/root/ctm_v34"
LOG_FILE = f"{LOG_DIR}/v34.log"
os.makedirs(LOG_DIR, exist_ok=True)

def ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def log(msg):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + "\n")

# ── Lorenz dt=0.05 ─────────────────────────────────────────────────────────────
def lorenz_step(state, dt=DT, sigma=10.0, rho=28.0, beta=8.0/3.0):
    def deriv(s):
        x, y, z = s[...,0], s[...,1], s[...,2]
        return torch.stack([sigma*(y-x), x*(rho-z)-y, x*y-beta*z], dim=-1)
    k1 = deriv(state)
    k2 = deriv(state + 0.5*dt*k1)
    k3 = deriv(state + 0.5*dt*k2)
    k4 = deriv(state + dt*k3)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def generate_lorenz_batch(batch_size, seq_len):
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
    return traj / norms

# ── Architecture ───────────────────────────────────────────────────────────────
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

def analyze_gates(model, t_gate):
    with torch.no_grad():
        g         = torch.softmax(model.gate_enc.gate_logits, dim=0).cpu().numpy()
        delays    = np.arange(t_gate)[::-1].copy()
        eff_delay = float(np.sum(delays * g))
        peak_idx  = int(np.argmax(g))
        peak_delay = t_gate - 1 - peak_idx
        peak_prob  = float(g[peak_idx])
        mean_peak  = float(np.max(g))
    return eff_delay, peak_delay, peak_prob, mean_peak, g

# ── Training loop ──────────────────────────────────────────────────────────────
def run_condition(k_pred, t_gate):
    tag = f"k{k_pred}_tg{t_gate}"
    log(f"\n{'='*60}")
    log(f"k={k_pred}, T_GATE={t_gate}  (tag={tag})")
    theory_window = min(k_pred * TAU_L, float(t_gate))
    log(f"Theory: tau*(k={k_pred}) ~ {k_pred}*{TAU_L} = {k_pred*TAU_L:.1f} steps "
        f"(window={t_gate}, {'SATURATED' if k_pred*TAU_L >= t_gate else 'FITS'})")
    log(f"{'='*60}")

    SEQ_LEN   = t_gate + k_pred + 20
    ckpt_path = os.path.join(LOG_DIR, f"ckpt_v34_{tag}.pt")

    model    = LorenzCTM(t_gate).to(DEVICE)
    opt      = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TRAIN_STEPS)
    start_step = 0
    best_mse = float('inf')

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
        seq     = generate_lorenz_batch(BATCH_SIZE, SEQ_LEN).to(DEVICE)
        max_t0  = SEQ_LEN - t_gate - k_pred
        if max_t0 <= 0:
            continue
        t0      = torch.randint(0, max_t0, (1,)).item()
        history = seq[:, t0:t0+t_gate]
        target  = seq[:, t0+t_gate+k_pred-1]   # k steps ahead
        pred, _ = model(history)
        loss    = ((pred - target)**2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if loss.item() < best_mse:
            best_mse = loss.item()
        if step % 5000 == 0 or step == TRAIN_STEPS:
            eff_delay, peak_delay, peak_prob, mean_pk, _ = analyze_gates(model, t_gate)
            log(f"  step {step:6d} | loss={loss.item():.6f} | eff_delay={eff_delay:.2f} | "
                f"peak@t-{peak_delay}({peak_prob:.3f}) | mean_peak={mean_pk:.3f}")
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'sched': sched.state_dict(), 'step': step, 'best_mse': best_mse}, ckpt_path)

    eff_delay, peak_delay, peak_prob, mean_pk, gate_dist = analyze_gates(model, t_gate)
    theory_w = k_pred * TAU_L

    log(f"\n  -- k={k_pred} T_GATE={t_gate} FINAL --")
    log(f"  eff_delay:   {eff_delay:.2f}  (theory_window: {theory_w:.1f})")
    log(f"  mean_peak:   {mean_pk:.3f}  (1.0=delta, <1=broad)")
    log(f"  best_loss:   {best_mse:.8f}")
    if eff_delay >= theory_w * 0.85:
        verdict = "HORIZON_SCALES_WITH_K"
    elif eff_delay >= t_gate * 0.85:
        verdict = "WINDOW_SATURATED"
    else:
        verdict = "BELOW_THEORY"
    log(f"  verdict:     {verdict}")

    return {
        "k_pred":       k_pred,
        "t_gate":       t_gate,
        "eff_delay":    round(eff_delay, 3),
        "peak_delay":   peak_delay,
        "peak_prob":    round(peak_prob, 4),
        "mean_peak":    round(mean_pk, 4),
        "best_loss":    round(best_mse, 8),
        "theory_window": round(theory_w, 1),
        "verdict":      verdict,
        "gate_distribution": [round(float(x), 5) for x in gate_dist],
    }

# ── Main ───────────────────────────────────────────────────────────────────────
log("CTM World Model v34 — Multi-Step Prediction Horizon Sweep")
log(f"Lorenz dt={DT}, tau_L={TAU_L} | k sweep: {K_TGATE_SWEEP}")
log(f"Hypothesis: tau*(k) ~ k * tau_L = {TAU_L} * k")
log(f"Device: {DEVICE}")

results_path = os.path.join(LOG_DIR, "results_v34.json")
all_results  = {}
if os.path.exists(results_path):
    with open(results_path) as f:
        all_results = json.load(f)
    log(f"Loaded existing: keys done = {list(all_results.keys())}")

for k_pred, t_gate in K_TGATE_SWEEP:
    key = f"k{k_pred}_tg{t_gate}"
    if key in all_results and "eff_delay" in all_results[key]:
        log(f"{key} already done, skipping")
        continue
    r = run_condition(k_pred, t_gate)
    all_results[key] = r
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"[checkpoint] results_v34.json updated ({key} done)")

# ── Summary ────────────────────────────────────────────────────────────────────
log(f"\n{'='*60}")
log("V34 COMPLETE — MULTI-STEP HORIZON SWEEP")
log(f"{'='*60}")
log(f"{'k':>4} | {'T_GATE':>6} | {'tau* learned':>13} | {'theory window':>14} | {'verdict'}")
for key in [f"k{k}_tg{tg}" for k,tg in K_TGATE_SWEEP]:
    r = all_results.get(key, {})
    if "eff_delay" not in r:
        continue
    log(f"{r['k_pred']:>4} | {r['t_gate']:>6} | {r['eff_delay']:>13.2f} | "
        f"{r['theory_window']:>14.1f} | {r['verdict']}")

# Test if eff_delay scales linearly with k (at T_GATE=32)
k32 = [(r['k_pred'], r['eff_delay'])
       for k, r in all_results.items()
       if isinstance(r, dict) and r.get('t_gate') == 32 and 'eff_delay' in r]
k32.sort()
if len(k32) >= 3:
    ks  = np.array([p[0] for p in k32], dtype=float)
    taus = np.array([p[1] for p in k32])
    b   = np.cov(ks, taus)[0,1] / np.var(ks)
    a   = np.mean(taus) - b * np.mean(ks)
    res = taus - (a + b * ks)
    r2  = 1 - np.var(res) / np.var(taus)
    log(f"\n  Linear fit (T_GATE=32): tau* = {a:.2f} + {b:.2f}*k,  R2={r2:.4f}")
    log(f"  Theory slope: {TAU_L:.1f} steps/k,  measured: {b:.2f}")
    if r2 > 0.90 and abs(b - TAU_L) < 5.0:
        log("  *** K_SCALING_CONFIRMED: tau* scales linearly with k at rate ~tau_L ***")
        all_results["k_scaling_verdict"] = "K_SCALING_CONFIRMED"
        all_results["k_scaling_fit"] = {"a": round(a,3), "b": round(b,3), "r2": round(r2,4)}
    elif r2 > 0.70:
        log("  *** K_SCALING_PARTIAL: approximately linear but slope off ***")
        all_results["k_scaling_verdict"] = "K_SCALING_PARTIAL"
    else:
        log("  *** K_SCALING_NONE: no linear trend with k ***")
        all_results["k_scaling_verdict"] = "K_SCALING_NONE"

with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2)
log("All results saved.")
