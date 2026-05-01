#!/usr/bin/env python3
"""
CTM World Model v29 — Periodic Signal
Archon / DuoNeural 2026-04-30

Test: does CTM's temporal window lock onto the task's characteristic period?

Hypothesis: N=8 sine wave oscillators, period T=8 steps, partial obs (position only).
With T_GATE=32 (4 full periods available), CTM should reference t-T=8 as primary gate.

Completes the tripartite picture for Paper 4:
  Markovian (ball):  max_delay ≈ 0       (just use present state)
  Periodic (sine):   max_delay ≈ T       (one period back = phase twin)
  Chaotic (Lorenz):  max_delay ≈ λ⁻¹    (Lyapunov horizon)

Same CTM architecture as v27/v28. T_GATE sweep {8, 16, 32}.
"""

import torch, numpy as np, json, os, math, time
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Config ────────────────────────────────────────────────────────────────────
N_OBJ       = 8
SINE_PERIOD = 8          # T. Hypothesis: max_delay → 8 at T_GATE=32
T_GATE_LIST = [8, 16, 32]
TRAIN_STEPS = 40_000     # quicker than v28 (oscillators are simpler)
BATCH_SIZE  = 128
K_PRED      = 4          # predict k steps ahead
HIDDEN_DIM  = 128
N_SLOTS     = 4
LOG_FILE    = os.path.expanduser("~/duoneural/ctm_world_model_v29/wm_v29.log")
OUT_DIR     = os.path.expanduser("~/duoneural/ctm_world_model_v29")
os.makedirs(OUT_DIR, exist_ok=True)

# v27 ball bounce reference (same architecture)
V27_REF = {4: 0.0, 8: 1.0, 16: 2.0, 32: 2.8}

def ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{ts()}] {msg}\n")

# ── Sine wave data ─────────────────────────────────────────────────────────────
def generate_sine_batch(batch_size, seq_len, n_obj=N_OBJ, period=SINE_PERIOD):
    """
    N independent sine waves with random phases and slight amplitude variation.
    Position-only (partial obs). Shape: (B, seq_len, N_OBJ).
    """
    t = torch.arange(seq_len, dtype=torch.float32)  # (seq_len,)
    phases = torch.rand(batch_size, n_obj) * 2 * math.pi  # (B, N_OBJ)
    amps   = 0.8 + 0.4 * torch.rand(batch_size, n_obj)    # (B, N_OBJ) ∈ [0.8, 1.2]
    # noise
    noise  = 0.02 * torch.randn(batch_size, seq_len, n_obj)
    
    # x(t) = A * sin(2π t / T + φ)
    omega = 2 * math.pi / period
    t_exp    = t.unsqueeze(0).unsqueeze(-1)   # (1, seq_len, 1)
    phases_e = phases.unsqueeze(1)             # (B, 1, N_OBJ)
    amps_e   = amps.unsqueeze(1)               # (B, 1, N_OBJ)
    
    x = amps_e * torch.sin(omega * t_exp + phases_e) + noise  # (B, seq_len, N_OBJ)
    return x

# ── Temporal gate encoder ──────────────────────────────────────────────────────
class LearnedTemporalGateEncoder(nn.Module):
    def __init__(self, t_gate, obj_dim, hidden_dim):
        super().__init__()
        self.t_gate = t_gate
        self.obj_dim = obj_dim
        self.hidden_dim = hidden_dim
        # Gate: softmax over T_GATE timesteps
        self.gate_logits = nn.Parameter(torch.zeros(t_gate))
        # Per-timestep encoder (shared)
        self.encoder = nn.Sequential(
            nn.Linear(obj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, history):
        """
        history: (B, T_GATE, N_OBJ, obj_dim)
        returns: encoded (B, N_OBJ, hidden_dim), gates (T_GATE,)
        """
        B, T, N, D = history.shape
        gates = torch.softmax(self.gate_logits, dim=0)  # (T,)
        
        # Encode each timestep
        h_flat = history.reshape(B * T * N, D)
        enc_flat = self.encoder(h_flat)  # (B*T*N, hidden_dim)
        enc = enc_flat.reshape(B, T, N, self.hidden_dim)
        
        # Weighted sum over time
        gates_e = gates.view(1, T, 1, 1)
        out = (enc * gates_e).sum(dim=1)  # (B, N, hidden_dim)
        return out, gates

class SlotDynamics(nn.Module):
    def __init__(self, hidden_dim, n_slots, obj_dim):
        super().__init__()
        self.n_slots = n_slots
        # Self-attention over objects
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        # Decode to position
        self.decoder = nn.Linear(hidden_dim, obj_dim)

    def forward(self, enc):
        """enc: (B, N_OBJ, hidden_dim)"""
        x, _ = self.attn(enc, enc, enc)
        x = self.norm1(enc + x)
        x = self.norm2(x + self.ff(x))
        pred = self.decoder(x)  # (B, N_OBJ, obj_dim)
        return pred

class SineCTM(nn.Module):
    def __init__(self, t_gate, obj_dim=1, hidden_dim=HIDDEN_DIM, n_slots=N_SLOTS):
        super().__init__()
        self.t_gate = t_gate
        self.gate_enc = LearnedTemporalGateEncoder(t_gate, obj_dim, hidden_dim)
        self.dynamics = SlotDynamics(hidden_dim, n_slots, obj_dim)

    def forward(self, history):
        """history: (B, T_GATE, N_OBJ, 1)"""
        enc, gates = self.gate_enc(history)  # (B, N_OBJ, hidden_dim), (T_GATE,)
        pred = self.dynamics(enc)            # (B, N_OBJ, 1)
        return pred.squeeze(-1), gates       # (B, N_OBJ), (T_GATE,)

# ── Training ───────────────────────────────────────────────────────────────────
def run_experiment(t_gate):
    log(f"\n{'='*60}")
    log(f"T_GATE={t_gate} — Periodic Sine Test")
    log(f"{'='*60}")
    
    SEQ_LEN = t_gate + K_PRED + 10  # enough context
    
    model = SineCTM(t_gate=t_gate).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TRAIN_STEPS)
    
    best_mse = float('inf')
    
    for step in range(1, TRAIN_STEPS + 1):
        model.train()
        seq = generate_sine_batch(BATCH_SIZE, SEQ_LEN).to(DEVICE)  # (B, seq_len, N_OBJ)
        
        # History window
        t_start = torch.randint(0, SEQ_LEN - t_gate - K_PRED, (1,)).item()
        history = seq[:, t_start:t_start+t_gate, :].unsqueeze(-1)  # (B, T_GATE, N_OBJ, 1)
        target  = seq[:, t_start+t_gate+K_PRED-1, :]               # (B, N_OBJ)
        
        pred, gates = model(history)
        loss = ((pred - target)**2).mean()
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        if step % 5000 == 0 or step == TRAIN_STEPS:
            # Gate analysis
            with torch.no_grad():
                g = torch.softmax(model.gate_enc.gate_logits, dim=0).cpu().numpy()
                peak_idx    = int(np.argmax(g))
                peak_prob   = float(g[peak_idx])
                # effective delay = sum(i * g[i]) for i=0..T-1 reversed
                delays      = np.arange(t_gate)[::-1]  # t-0=most recent = delay 0
                eff_delay   = float(np.sum(delays * g))
                
            log(f"  step {step:6d} | loss={loss.item():.6f} | "
                f"peak@t-{t_gate-1-peak_idx}({peak_prob:.3f}) | eff_delay={eff_delay:.2f}")
            
            if loss.item() < best_mse:
                best_mse = loss.item()
    
    # Final gate analysis
    with torch.no_grad():
        g    = torch.softmax(model.gate_enc.gate_logits, dim=0).cpu().numpy()
        peak_idx   = int(np.argmax(g))
        peak_prob  = float(g[peak_idx])
        delays     = np.arange(t_gate)[::-1]
        eff_delay  = float(np.sum(delays * g))
        # Effective delay from most recent = index 0 → delay 0, index T-1 → delay T-1
    
    v27_ref = V27_REF.get(t_gate, "N/A")
    delta   = round(eff_delay - (v27_ref if isinstance(v27_ref, float) else 0), 2)
    period_delta = round(eff_delay - SINE_PERIOD, 2)
    
    log(f"\n  ── T_GATE={t_gate} FINAL ──")
    log(f"  gate dist: {np.round(g, 3).tolist()}")
    log(f"  peak: t-{t_gate-1-peak_idx} (prob={peak_prob:.4f})")
    log(f"  eff_delay:       {eff_delay:.2f}")
    log(f"  vs ball (v27):   {v27_ref} (Δ={delta:+.2f})")
    log(f"  vs T_period={SINE_PERIOD}: Δ={period_delta:+.2f}  ← key test")
    log(f"  best_loss:       {best_mse:.6f}")
    
    result = {
        "t_gate": t_gate,
        "max_delay_used": round(eff_delay, 2),
        "peak_delay": t_gate - 1 - peak_idx,
        "peak_prob": round(peak_prob, 4),
        "gate_spec": round(float(np.sum(g * (delays - eff_delay)**2)**0.5), 4),
        "mean_peak": round(peak_prob, 4),
        "best_loss": round(best_mse, 8),
        "v27_ball_ref": v27_ref,
        "delta_vs_ball": delta,
        "delta_vs_period": period_delta,
        "sine_period": SINE_PERIOD,
        "gate_distribution": [round(float(x), 4) for x in g],
    }
    
    # Interpretation
    if abs(period_delta) < 2.0:
        log(f"  *** PERIOD HYPOTHESIS CONFIRMED: eff_delay ≈ T_period={SINE_PERIOD} ***")
        result["verdict"] = "PERIOD_TRACKING"
    elif eff_delay < 3.0:
        log(f"  *** MARKOVIAN: CTM uses only recent frames (like ball bounce) ***")
        result["verdict"] = "MARKOVIAN"
    else:
        log(f"  *** EXTENDED WINDOW: CTM reaches deeper than Markovian baseline ***")
        result["verdict"] = "EXTENDED"
    
    return result

# ── Main ───────────────────────────────────────────────────────────────────────
log(f"CTM World Model v29 — Periodic Sine Test")
log(f"N_OBJ={N_OBJ}, SINE_PERIOD={SINE_PERIOD}, TRAIN_STEPS={TRAIN_STEPS}")
log(f"T_GATE sweep: {T_GATE_LIST}")
log(f"Hypothesis: max_delay → {SINE_PERIOD} at large T_GATE (period tracking)")
log(f"Device: {DEVICE}")

all_results = {}
for tg in T_GATE_LIST:
    r = run_experiment(tg)
    all_results[str(tg)] = r
    
    with open(f"{OUT_DIR}/results_v29.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"[checkpoint] results_v29.json saved")

log(f"\n{'='*60}")
log(f"V29 COMPLETE — Summary")
log(f"{'='*60}")
log(f"{'T_GATE':>8} | {'max_delay':>10} | {'vs_period':>10} | verdict")
for tg, r in all_results.items():
    log(f"{tg:>8} | {r['max_delay_used']:>10.2f} | {r['delta_vs_period']:>+10.2f} | {r['verdict']}")

log("All results saved to results_v29.json")
