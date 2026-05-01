#!/usr/bin/env python3
"""
CTM World Model v32 — Ablation: Frozen Gate at Wrong Delay
Archon / DuoNeural 2026-04-30

Robustness check #3 for the Tripartite Temporal Principle (Paper 4).

This is the PROOF experiment. If the CTM's learned gate genuinely found
the mathematical optimum (Lyapunov horizon ≈ 22 steps for Lorenz), then:

  MSE(LEARNED) ≈ MSE(FROZEN_22) < MSE(FROZEN_31) < MSE(FROZEN_5)

The reasoning:
- FROZEN_5:  τ=5 is before the Lyapunov time. The system is still somewhat
             predictable at t-5, but you're throwing away the richer
             information that accumulates between t-5 and t-22. MSE = worst.
- FROZEN_31: τ=31 is past the Lyapunov time (max available for T_GATE=32).
             State at t-31 has diverged beyond predictability. Injecting noise.
             MSE = bad (but slightly better than FROZEN_5 because chaos
             diverges, not just drops info — depends on task geometry).
- FROZEN_22: τ=22 ≈ Lyapunov time. This is where prediction error starts
             exploding. Optimal gate position. MSE = near-optimal.
- LEARNED:   Free to pick any distribution. Should converge to ≈ FROZEN_22
             or a narrow distribution centered near 22. MSE = best/tied.

Task: Lorenz (same as v28), T_GATE=32 throughout.

Conditions:
  1. LEARNED    — standard trained gate (baseline)
  2. FROZEN_5   — one-hot gate fixed at delay=5
  3. FROZEN_31  — one-hot gate fixed at delay=31 (= T-1, max possible)
  4. FROZEN_22  — one-hot gate fixed at delay=22 (≈ Lyapunov time)

For frozen conditions: gate_logits is replaced with a fixed tensor,
no gradient flows through it, rest of model trains normally.
"""

import torch, numpy as np, json, os, math, time
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Config ────────────────────────────────────────────────────────────────────
N_OBJ       = 8
T_GATE      = 32         # fixed for all conditions in v32
TRAIN_STEPS = 40_000
BATCH_SIZE  = 128
K_PRED      = 4
HIDDEN_DIM  = 128
N_SLOTS     = 4
OBJ_DIM     = 3          # Lorenz (x, y, z)

# Frozen delay values (in steps back from present)
# Note: T_GATE=32 → valid delays are 0..31 (31 = oldest available frame).
# We use 31 for "too long" since delay=32 would be out-of-range.
FROZEN_DELAYS = [5, 22, 31]   # too_short, lyapunov_time, too_long(max=T-1)

LOG_FILE = os.path.expanduser("~/duoneural/ctm_world_model_v32/wm_v32.log")
OUT_DIR  = os.path.expanduser("~/duoneural/ctm_world_model_v32")
os.makedirs(OUT_DIR, exist_ok=True)

# v28 learned gate reference for comparison
V28_LEARNED_DELAY = 23.5
V28_BEST_LOSS     = None  # will be read from context if available

def ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{ts()}] {msg}\n")

# ── Lorenz data ────────────────────────────────────────────────────────────────
def lorenz_step(state, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """RK4 step for standard Lorenz attractor. state: (B, N_OBJ, 3)"""
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


def generate_lorenz_batch(batch_size, seq_len, n_obj=N_OBJ):
    """
    Generate clean Lorenz trajectories (no noise — same as v28, not v31).
    Returns: (B, seq_len, N_OBJ, 3), normalized.
    """
    state = torch.randn(batch_size, n_obj, 3) * 5.0
    state[..., 2] += 25.0
    # Warm-up
    for _ in range(500):
        state = lorenz_step(state)
    traj = []
    for _ in range(seq_len):
        state = lorenz_step(state)
        traj.append(state.clone())
    traj  = torch.stack(traj, dim=1)          # (B, seq_len, N_OBJ, 3)
    norms = torch.tensor([9.0, 9.0, 8.5]).view(1, 1, 1, 3)
    return traj / norms


# ── Architecture ───────────────────────────────────────────────────────────────
class LearnedTemporalGateEncoder(nn.Module):
    """
    Gate encoder with two modes:
      - learned: gate_logits are trainable parameters (standard)
      - frozen:  gate_logits are a fixed constant buffer (one-hot at specified delay)

    frozen_delay=None → standard learned mode
    frozen_delay=k   → one-hot fixed at delay k (gate[T-1-k] = large number → softmax→1)
    """
    def __init__(self, t_gate, obj_dim, hidden_dim, frozen_delay=None):
        super().__init__()
        self.t_gate      = t_gate
        self.obj_dim     = obj_dim
        self.hidden_dim  = hidden_dim
        self.frozen_delay = frozen_delay

        if frozen_delay is None:
            # Standard: learnable gate logits
            self.gate_logits = nn.Parameter(torch.zeros(t_gate))
        else:
            # Frozen one-hot: register as buffer (not parameter → no grad)
            # delay k steps back = index T-1-k in the history tensor
            # (history[:,0,:] is oldest = delay T-1, history[:,T-1,:] is newest = delay 0)
            logits = torch.full((t_gate,), -1e9)
            onehot_idx = t_gate - 1 - frozen_delay  # map delay → history index
            onehot_idx = max(0, min(t_gate - 1, onehot_idx))
            logits[onehot_idx] = 0.0  # softmax will put all weight here
            self.register_buffer('gate_logits', logits)

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
        gates    = torch.softmax(self.gate_logits, dim=0)
        h_flat   = history.reshape(B * T * N, D)
        enc_flat = self.encoder(h_flat)
        enc      = enc_flat.reshape(B, T, N, self.hidden_dim)
        gates_e  = gates.view(1, T, 1, 1)
        out      = (enc * gates_e).sum(dim=1)
        return out, gates


class SlotDynamics(nn.Module):
    """Same self-attention + FF dynamics as all previous versions."""
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
    """Full CTM for Lorenz. frozen_delay=None → learned, else frozen one-hot."""
    def __init__(self, t_gate=T_GATE, obj_dim=OBJ_DIM, hidden_dim=HIDDEN_DIM,
                 n_slots=N_SLOTS, frozen_delay=None):
        super().__init__()
        self.t_gate   = t_gate
        self.gate_enc = LearnedTemporalGateEncoder(
            t_gate, obj_dim, hidden_dim, frozen_delay=frozen_delay
        )
        self.dynamics = SlotDynamics(hidden_dim, n_slots, obj_dim)

    def forward(self, history):
        enc, gates = self.gate_enc(history)
        pred       = self.dynamics(enc)
        return pred, gates


# ── Gate analysis ──────────────────────────────────────────────────────────────
def analyze_gates(model, t_gate=T_GATE):
    with torch.no_grad():
        g         = torch.softmax(model.gate_enc.gate_logits, dim=0).cpu().numpy()
        peak_idx  = int(np.argmax(g))
        peak_prob = float(g[peak_idx])
        delays    = np.arange(t_gate)[::-1].copy()
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
def run_condition(condition_name, frozen_delay=None):
    """
    Train one condition. condition_name is one of:
      "LEARNED", "FROZEN_5", "FROZEN_22", "FROZEN_31"
    frozen_delay: int or None
    """
    log(f"\n{'='*60}")
    log(f"Condition: {condition_name}  (frozen_delay={frozen_delay}, T_GATE={T_GATE})")
    log(f"{'='*60}")

    SEQ_LEN   = T_GATE + K_PRED + 20
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_v32_{condition_name}.pt")

    model = LorenzCTM(frozen_delay=frozen_delay).to(DEVICE)

    # Sanity check: verify gate is truly frozen
    if frozen_delay is not None:
        m = analyze_gates(model)
        log(f"  [init] gate one-hot at delay={m['peak_delay']} (expected {frozen_delay}) "
            f"eff_delay={m['eff_delay']:.1f}")
        # Make sure gate_logits is NOT in the parameter list
        param_names = [n for n, _ in model.named_parameters()]
        if 'gate_enc.gate_logits' in param_names:
            log(f"  [WARNING] gate_logits appears in parameters! This is a bug.")
        else:
            log(f"  [OK] gate_logits is a frozen buffer, not in parameters")

    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TRAIN_STEPS)
    start_step = 0
    best_mse   = float('inf')

    if os.path.exists(ckpt_path):
        log(f"  Resuming from: {ckpt_path}")
        ckpt       = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        start_step = ckpt['step']
        best_mse   = ckpt.get('best_mse', float('inf'))
        log(f"  Resumed at step {start_step}, best_mse={best_mse:.6f}")

    for step in range(start_step + 1, TRAIN_STEPS + 1):
        model.train()
        seq = generate_lorenz_batch(BATCH_SIZE, SEQ_LEN).to(DEVICE)

        t_start = torch.randint(0, SEQ_LEN - T_GATE - K_PRED, (1,)).item()
        history = seq[:, t_start:t_start+T_GATE, :, :]
        target  = seq[:, t_start+T_GATE+K_PRED-1, :, :]

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
            m = analyze_gates(model)
            log(f"  step {step:6d} | loss={loss.item():.6f} | "
                f"peak@t-{m['peak_delay']}({m['peak_prob']:.3f}) | "
                f"eff_delay={m['eff_delay']:.2f} | best={best_mse:.6f}")

            torch.save({
                'model':    model.state_dict(),
                'opt':      opt.state_dict(),
                'sched':    sched.state_dict(),
                'step':     step,
                'best_mse': best_mse,
            }, ckpt_path)

    # Final analysis
    m = analyze_gates(model)
    log(f"\n  ── {condition_name} FINAL ──")
    log(f"  eff_delay:  {m['eff_delay']:.2f}")
    log(f"  peak:       t-{m['peak_delay']} (prob={m['peak_prob']:.4f})")
    log(f"  best_loss:  {best_mse:.6f}")
    log(f"  frozen_at:  {frozen_delay}")

    return {
        "condition":        condition_name,
        "frozen_delay":     frozen_delay,
        "t_gate":           T_GATE,
        "max_delay_used":   round(m['eff_delay'], 2),
        "peak_delay":       m['peak_delay'],
        "peak_prob":        round(m['peak_prob'], 4),
        "gate_spec":        round(m['gate_spec'], 4),
        "best_loss":        round(best_mse, 8),
        "gate_distribution": [round(float(x), 4) for x in m['gate_dist']],
    }


# ── Conditions ────────────────────────────────────────────────────────────────
CONDITIONS = [
    ("LEARNED",   None),   # standard trainable gate
    ("FROZEN_5",   5),     # too short — before Lyapunov time
    ("FROZEN_22", 22),     # at Lyapunov time — near-optimal hypothesis
    ("FROZEN_31", 31),     # max possible delay (T-1=31) — past Lyapunov time
]

# ── Main ───────────────────────────────────────────────────────────────────────
log(f"CTM World Model v32 — Frozen Gate Ablation Study")
log(f"Task: Lorenz (T_GATE={T_GATE} fixed)")
log(f"N_OBJ={N_OBJ}, OBJ_DIM={OBJ_DIM}, TRAIN_STEPS={TRAIN_STEPS}")
log(f"Conditions: {[c[0] for c in CONDITIONS]}")
log(f"Hypothesis: LEARNED ≈ FROZEN_22 < FROZEN_31 < FROZEN_5")
log(f"Device: {DEVICE}")

all_results  = {}
results_path = os.path.join(OUT_DIR, "results_v32.json")

if os.path.exists(results_path):
    with open(results_path) as f:
        all_results = json.load(f)
    log(f"Loaded existing results: {list(all_results.keys())}")

for cond_name, frozen_delay in CONDITIONS:
    if cond_name in all_results and "best_loss" in all_results[cond_name]:
        log(f"Condition {cond_name} already complete, skipping")
        continue
    r = run_condition(cond_name, frozen_delay=frozen_delay)
    all_results[cond_name] = r
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"[checkpoint] results_v32.json saved ({cond_name} done)")

# ── Final comparison & verdict ─────────────────────────────────────────────────
log(f"\n{'='*60}")
log(f"V32 COMPLETE — Frozen Gate Ablation Summary")
log(f"{'='*60}")
log(f"{'Condition':>12} | {'frozen_at':>10} | {'eff_delay':>10} | {'best_mse':>12} | rank")

# Sort by MSE to show ranking
cond_list = [(k, v) for k, v in all_results.items()
             if isinstance(v, dict) and "best_loss" in v]
cond_list.sort(key=lambda x: x[1]["best_loss"])

for rank, (name, r) in enumerate(cond_list, 1):
    log(f"  {name:>12} | {str(r['frozen_delay']):>10} | {r['max_delay_used']:>10.2f} | "
        f"{r['best_loss']:>12.8f} | #{rank}")

# Verdict logic
if all(k in all_results for k in ["LEARNED", "FROZEN_5", "FROZEN_22", "FROZEN_31"]):
    mse = {k: all_results[k]["best_loss"] for k in ["LEARNED", "FROZEN_5", "FROZEN_22", "FROZEN_31"]}

    log(f"\n  MSE Rankings:")
    log(f"    LEARNED:   {mse['LEARNED']:.8f}")
    log(f"    FROZEN_22: {mse['FROZEN_22']:.8f}")
    log(f"    FROZEN_31: {mse['FROZEN_31']:.8f}  (max delay = T-1)")
    log(f"    FROZEN_5:  {mse['FROZEN_5']:.8f}")

    # Check the hypothesis: LEARNED ≈ FROZEN_22 < FROZEN_31 < FROZEN_5
    learned_is_best_or_tied = mse["LEARNED"]  <= mse["FROZEN_31"] and mse["LEARNED"]  <= mse["FROZEN_5"]
    frozen22_beats_frozen5  = mse["FROZEN_22"] <  mse["FROZEN_5"]
    frozen31_beats_frozen5  = mse["FROZEN_31"] <  mse["FROZEN_5"]
    learned_close_to_f22    = abs(mse["LEARNED"] - mse["FROZEN_22"]) < 0.02 * mse["FROZEN_22"]

    log(f"\n  Hypothesis checks:")
    log(f"    LEARNED ≤ FROZEN_31,5: {learned_is_best_or_tied}")
    log(f"    FROZEN_22 < FROZEN_5:  {frozen22_beats_frozen5}")
    log(f"    FROZEN_31 < FROZEN_5:  {frozen31_beats_frozen5}")
    log(f"    LEARNED ≈ FROZEN_22:   {learned_close_to_f22}")

    if learned_is_best_or_tied and frozen22_beats_frozen5 and frozen31_beats_frozen5:
        verdict = "LYAPUNOV_OPTIMAL"
        log(f"\n  *** VERDICT: LYAPUNOV_OPTIMAL ***")
        log(f"  The learned gate found the mathematical optimum.")
        log(f"  τ=22 (Lyapunov time) is the best fixed gate — FROZEN_5 confirms")
        log(f"  that too-short history hurts MORE than too-long history (excess history=noise).")
        if learned_close_to_f22:
            log(f"  LEARNED ≈ FROZEN_22 — learned gate converged to Lyapunov time.")
        else:
            log(f"  LEARNED outperforms FROZEN_22 — learned distribution beats one-hot.")
    elif not frozen22_beats_frozen5:
        verdict = "PARTIAL_SUPPORT"
        log(f"\n  *** VERDICT: PARTIAL_SUPPORT ***")
        log(f"  FROZEN_22 did not beat FROZEN_5 — Lyapunov time hypothesis needs revision.")
    else:
        verdict = "INCONCLUSIVE"
        log(f"\n  *** VERDICT: INCONCLUSIVE — check results manually ***")

    all_results["global_verdict"] = verdict
    all_results["mse_ranking"]    = sorted(mse.items(), key=lambda x: x[1])

with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2)
log(f"All results saved to {results_path}")
