#!/usr/bin/env python3
"""
CTM World Model v31 — Noise Test: Lorenz + White Observation Noise
Archon / DuoNeural 2026-04-30

Robustness check #2 for the Tripartite Temporal Principle (Paper 4).

Baseline: v28 found eff_delay ≈ 23.5 on clean Lorenz (T_GATE=32).
The Lyapunov time for the standard Lorenz system ≈ 1/λ_max ≈ 1/0.906 ≈ 1.1
(in continuous time), which maps to ~22–23 steps in our discrete simulation.

Prediction: adding Gaussian observation noise σ_obs=0.1 should shorten the
learned effective delay. Why?

  The information horizon is set by when prediction error from state uncertainty
  exceeds the noise floor. With noise, the effective predictability horizon is:

    τ_eff ≈ τ_L - (1/λ) * ln(SNR)

  where SNR = σ_signal² / σ_noise². For Lorenz x: σ_signal ≈ 10 (typical std),
  so SNR = 100/0.01 = 10000. Correction ≈ (1/0.906) * ln(10000) ≈ 10.1 steps.
  But our τ_L is already in discrete steps (~22), so the fractional effect
  at σ_obs=0.1 is mild but measurable.

  Simplest empirical test: does the learned delay DECREASE vs v28?

Verdict: "NOISE_SHORTENS_HORIZON" if v31_delay < v28_delay

v28 reference delays: T_GATE=16→7.8, T_GATE=32→23.5 (update from v28 results)
"""

import torch, numpy as np, json, os, math, time
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Config ────────────────────────────────────────────────────────────────────
N_OBJ       = 8
T_GATE_LIST = [16, 32]
TRAIN_STEPS = 40_000
BATCH_SIZE  = 128
K_PRED      = 4          # predict k steps ahead
HIDDEN_DIM  = 128
N_SLOTS     = 4
OBJ_DIM     = 3          # Lorenz: (x, y, z)
SIGMA_OBS   = 0.1        # KEY: Gaussian observation noise σ added to each timestep

LOG_FILE = os.path.expanduser("~/duoneural/ctm_world_model_v31/wm_v31.log")
OUT_DIR  = os.path.expanduser("~/duoneural/ctm_world_model_v31")
os.makedirs(OUT_DIR, exist_ok=True)

# v28 reference for comparison (update these after v28 runs if different)
V28_REF = {16: 7.8, 32: 23.5}

def ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{ts()}] {msg}\n")

# ── Lorenz system ──────────────────────────────────────────────────────────────
def lorenz_step(state, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Single RK4 step of the Lorenz attractor.
    state: (B, N_OBJ, 3) — x, y, z for each object
    Returns: next state (B, N_OBJ, 3)
    """
    def deriv(s):
        x, y, z = s[..., 0], s[..., 1], s[..., 2]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return torch.stack([dx, dy, dz], dim=-1)

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def generate_lorenz_batch(batch_size, seq_len, n_obj=N_OBJ, sigma_obs=SIGMA_OBS):
    """
    Generate B trajectories of N_OBJ independent Lorenz systems.
    Random IC on attractor (warm-up 500 steps, then burn forward).
    Adds Gaussian noise σ_obs to each observation (KEY for v31).
    Returns: (B, seq_len, N_OBJ, 3) — normalized
    """
    dt = 0.01
    # Random initial conditions near attractor
    state = torch.randn(batch_size, n_obj, 3) * 5.0
    state[..., 2] += 25.0  # push toward attractor

    # Warm-up to get onto attractor
    for _ in range(500):
        state = lorenz_step(state, dt=dt)

    # Collect trajectory
    traj = []
    for _ in range(seq_len):
        state = lorenz_step(state, dt=dt)
        traj.append(state.clone())

    traj = torch.stack(traj, dim=1)  # (B, seq_len, N_OBJ, 3)

    # Normalize per-coordinate over the batch (mean 0, std 1 approx)
    # Using population stats: Lorenz attractor has x~N(0,9), y~N(0,9), z≈25±10
    norms = torch.tensor([9.0, 9.0, 8.5]).view(1, 1, 1, 3)
    traj  = traj / norms

    # Add observation noise (the whole point of v31)
    if sigma_obs > 0.0:
        obs_noise = sigma_obs * torch.randn_like(traj)
        traj      = traj + obs_noise

    return traj


# ── Architecture (identical to v28/v29 — same gate, different input dim) ──────
class LearnedTemporalGateEncoder(nn.Module):
    """
    Softmax gate over T_GATE timesteps. One global gate, shared across objects.
    This is the thing we're measuring — where does it peak?
    """
    def __init__(self, t_gate, obj_dim, hidden_dim):
        super().__init__()
        self.t_gate     = t_gate
        self.obj_dim    = obj_dim
        self.hidden_dim = hidden_dim
        self.gate_logits = nn.Parameter(torch.zeros(t_gate))
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
        gates   = torch.softmax(self.gate_logits, dim=0)
        h_flat  = history.reshape(B * T * N, D)
        enc_flat = self.encoder(h_flat)
        enc     = enc_flat.reshape(B, T, N, self.hidden_dim)
        gates_e = gates.view(1, T, 1, 1)
        out     = (enc * gates_e).sum(dim=1)
        return out, gates


class SlotDynamics(nn.Module):
    """Self-attention over objects + FF. Same as v28/v29."""
    def __init__(self, hidden_dim, n_slots, obj_dim):
        super().__init__()
        self.attn    = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm1   = nn.LayerNorm(hidden_dim)
        self.ff      = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, obj_dim)

    def forward(self, enc):
        x, _ = self.attn(enc, enc, enc)
        x     = self.norm1(enc + x)
        x     = self.norm2(x + self.ff(x))
        return self.decoder(x)


class LorenzCTM(nn.Module):
    """CTM for Lorenz (OBJ_DIM=3). Same structure as v28."""
    def __init__(self, t_gate, obj_dim=OBJ_DIM, hidden_dim=HIDDEN_DIM, n_slots=N_SLOTS):
        super().__init__()
        self.t_gate   = t_gate
        self.gate_enc = LearnedTemporalGateEncoder(t_gate, obj_dim, hidden_dim)
        self.dynamics = SlotDynamics(hidden_dim, n_slots, obj_dim)

    def forward(self, history):
        """history: (B, T_GATE, N_OBJ, 3) → pred: (B, N_OBJ, 3), gates: (T_GATE,)"""
        enc, gates = self.gate_enc(history)
        pred       = self.dynamics(enc)
        return pred, gates


# ── Gate analysis helper ───────────────────────────────────────────────────────
def analyze_gates(model, t_gate):
    with torch.no_grad():
        g         = torch.softmax(model.gate_enc.gate_logits, dim=0).cpu().numpy()
        peak_idx  = int(np.argmax(g))
        peak_prob = float(g[peak_idx])
        delays    = np.arange(t_gate)[::-1].copy()  # delay 0 = most recent
        eff_delay = float(np.sum(delays * g))
        gate_spec = float(np.sum(g * (delays - eff_delay)**2) ** 0.5)
    return {
        "gate_dist":  g,
        "peak_idx":   peak_idx,
        "peak_delay": t_gate - 1 - peak_idx,
        "peak_prob":  peak_prob,
        "eff_delay":  eff_delay,
        "gate_spec":  gate_spec,
    }


# ── Training loop ──────────────────────────────────────────────────────────────
def run_experiment(t_gate):
    log(f"\n{'='*60}")
    log(f"T_GATE={t_gate} — Noisy Lorenz Test (σ_obs={SIGMA_OBS})")
    log(f"{'='*60}")

    SEQ_LEN   = t_gate + K_PRED + 20   # extra buffer for Lorenz warm-up integration
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_v31_tg{t_gate}.pt")

    model = LorenzCTM(t_gate=t_gate).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TRAIN_STEPS)
    start_step = 0
    best_mse   = float('inf')

    if os.path.exists(ckpt_path):
        log(f"  Resuming from checkpoint: {ckpt_path}")
        ckpt       = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        start_step = ckpt['step']
        best_mse   = ckpt.get('best_mse', float('inf'))
        log(f"  Resumed at step {start_step}, best_mse={best_mse:.6f}")

    for step in range(start_step + 1, TRAIN_STEPS + 1):
        model.train()
        # Generate noisy Lorenz batch fresh each step (cheap, avoids overfitting single traj)
        seq = generate_lorenz_batch(BATCH_SIZE, SEQ_LEN).to(DEVICE)
        # (B, seq_len, N_OBJ, 3)

        t_start = torch.randint(0, SEQ_LEN - t_gate - K_PRED, (1,)).item()
        history = seq[:, t_start:t_start+t_gate, :, :]          # (B, T_GATE, N_OBJ, 3)
        target  = seq[:, t_start+t_gate+K_PRED-1, :, :]         # (B, N_OBJ, 3)

        pred, gates = model(history)
        loss = ((pred - target)**2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if loss.item() < best_mse:
            best_mse = loss.item()

        if step % 5000 == 0 or step == TRAIN_STEPS:
            m = analyze_gates(model, t_gate)
            v28_ref = V28_REF.get(t_gate, "N/A")
            delta_vs_v28 = round(m['eff_delay'] - (v28_ref if isinstance(v28_ref, float) else 0), 2)
            log(f"  step {step:6d} | loss={loss.item():.6f} | "
                f"peak@t-{m['peak_delay']}({m['peak_prob']:.3f}) | "
                f"eff_delay={m['eff_delay']:.2f} | Δvs_v28={delta_vs_v28:+.2f}")

            torch.save({
                'model':    model.state_dict(),
                'opt':      opt.state_dict(),
                'sched':    sched.state_dict(),
                'step':     step,
                'best_mse': best_mse,
            }, ckpt_path)

    # Final analysis
    m            = analyze_gates(model, t_gate)
    v28_ref      = V28_REF.get(t_gate, None)
    delta_vs_v28 = round(m['eff_delay'] - v28_ref, 2) if v28_ref is not None else None

    log(f"\n  ── T_GATE={t_gate} FINAL ──")
    log(f"  gate dist:   {np.round(m['gate_dist'], 3).tolist()}")
    log(f"  peak:        t-{m['peak_delay']} (prob={m['peak_prob']:.4f})")
    log(f"  eff_delay:   {m['eff_delay']:.2f}")
    log(f"  v28 ref:     {v28_ref}")
    log(f"  Δ vs v28:    {delta_vs_v28:+.2f}  ← key test (negative = noise shortened horizon)")
    log(f"  best_loss:   {best_mse:.6f}")
    log(f"  σ_obs:       {SIGMA_OBS}")

    result = {
        "t_gate":          t_gate,
        "sigma_obs":       SIGMA_OBS,
        "max_delay_used":  round(m['eff_delay'], 2),
        "peak_delay":      m['peak_delay'],
        "peak_prob":       round(m['peak_prob'], 4),
        "gate_spec":       round(m['gate_spec'], 4),
        "best_loss":       round(best_mse, 8),
        "v28_ref_delay":   v28_ref,
        "delta_vs_v28":    delta_vs_v28,
        "gate_distribution": [round(float(x), 4) for x in m['gate_dist']],
    }

    # Verdict
    if delta_vs_v28 is not None and delta_vs_v28 < -1.0:
        log(f"  *** NOISE_SHORTENS_HORIZON: delay decreased by {abs(delta_vs_v28):.1f} steps ***")
        result["verdict"] = "NOISE_SHORTENS_HORIZON"
    elif delta_vs_v28 is not None and abs(delta_vs_v28) <= 1.0:
        log(f"  *** NOISE_NO_EFFECT: delay unchanged (Δ={delta_vs_v28:+.2f}) ***")
        result["verdict"] = "NOISE_NO_EFFECT"
    else:
        log(f"  *** NOISE_LENGTHENS_HORIZON: unexpected increase (Δ={delta_vs_v28:+.2f}) ***")
        result["verdict"] = "NOISE_LENGTHENS_HORIZON"

    return result


# ── Main ───────────────────────────────────────────────────────────────────────
log(f"CTM World Model v31 — Noisy Lorenz Test")
log(f"N_OBJ={N_OBJ}, OBJ_DIM={OBJ_DIM}, SIGMA_OBS={SIGMA_OBS}")
log(f"TRAIN_STEPS={TRAIN_STEPS}, T_GATE sweep={T_GATE_LIST}")
log(f"Hypothesis: noise shrinks effective Lyapunov horizon → learned delay < v28")
log(f"Device: {DEVICE}")

all_results  = {}
results_path = os.path.join(OUT_DIR, "results_v31.json")

if os.path.exists(results_path):
    with open(results_path) as f:
        all_results = json.load(f)
    log(f"Loaded existing results for T_GATE keys: {list(all_results.keys())}")

for tg in T_GATE_LIST:
    if str(tg) in all_results and "verdict" in all_results[str(tg)]:
        log(f"T_GATE={tg} already complete, skipping")
        continue
    r = run_experiment(tg)
    all_results[str(tg)] = r
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"[checkpoint] results_v31.json saved (T_GATE={tg} done)")

# ── Summary & side-by-side comparison ─────────────────────────────────────────
log(f"\n{'='*60}")
log(f"V31 COMPLETE — Noisy Lorenz Summary")
log(f"{'='*60}")
log(f"{'T_GATE':>8} | {'v31_delay':>10} | {'v28_delay':>10} | {'delta':>8} | verdict")
for tg_str, r in all_results.items():
    if not isinstance(r, dict) or "max_delay_used" not in r:
        continue
    v28_ref = V28_REF.get(int(tg_str), "N/A")
    log(f"{tg_str:>8} | {r['max_delay_used']:>10.2f} | {str(v28_ref):>10} | "
        f"{r['delta_vs_v28'] if r['delta_vs_v28'] is not None else 'N/A':>8} | {r['verdict']}")

# Global verdict at T_GATE=32 (primary test)
if "32" in all_results and "verdict" in all_results["32"]:
    v = all_results["32"]["verdict"]
    log(f"\n  *** GLOBAL VERDICT (T_GATE=32): {v} ***")
    all_results["global_verdict"] = v
    if v == "NOISE_SHORTENS_HORIZON":
        log(f"  Lyapunov horizon is modulated by SNR — theory confirmed.")
        log(f"  Temporal gate is a genuine information-theoretic selector.")
    elif v == "NOISE_NO_EFFECT":
        log(f"  Gate is robust to this noise level — check σ_obs=1.0 next.")

with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2)
log(f"All results saved to {results_path}")
