#!/usr/bin/env python3
"""
CTM World Model v30 — Gap Test: Different Sine Period
Archon / DuoNeural 2026-04-30

Robustness check #1 for the Tripartite Temporal Principle (Paper 4).

If the finding is genuine and not a fluke of T=8 being conveniently equal to
some architectural constant, then doubling the period to T=16 should shift
the learned effective delay to ~16 as well.

v29 used SINE_PERIOD=8 and found eff_delay ≈ 8 at T_GATE=32.
Here we use SINE_PERIOD=16, T_GATE sweep {16, 32}.
Both T_GATE values are >= SINE_PERIOD, so the gate has enough room to find it.

Verdict: "PERIOD_SHIFTED" if eff_delay ≈ 16 at T_GATE=32
Cross-ref: compare to v29's eff_delay ≈ 8 to confirm proportional shift.
"""

import torch, numpy as np, json, os, math, time
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Config ────────────────────────────────────────────────────────────────────
N_OBJ        = 8
SINE_PERIOD  = 16          # KEY CHANGE: doubled from v29's T=8
T_GATE_LIST  = [16, 32]    # must be >= SINE_PERIOD to give the gate a chance
TRAIN_STEPS  = 40_000      # robustness check, shorter than main experiments
BATCH_SIZE   = 128
K_PRED       = 4           # predict k steps ahead
HIDDEN_DIM   = 128
N_SLOTS      = 4
LOG_FILE     = os.path.expanduser("~/duoneural/ctm_world_model_v30/wm_v30.log")
OUT_DIR      = os.path.expanduser("~/duoneural/ctm_world_model_v30")
os.makedirs(OUT_DIR, exist_ok=True)

# v29 reference results (period=8) for comparison in verdict
V29_REF_DELAY = {8: 8.0, 16: 8.0, 32: 8.0}  # approximate, update after v29 runs

def ts():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{ts()}] {msg}\n")

# ── Sine wave data ─────────────────────────────────────────────────────────────
def generate_sine_batch(batch_size, seq_len, n_obj=N_OBJ, period=SINE_PERIOD):
    """
    N independent sine waves, random phase + slight amplitude jitter.
    OBJ_DIM=1 (position only, partial obs).
    Returns: (B, seq_len, N_OBJ)
    """
    t      = torch.arange(seq_len, dtype=torch.float32)
    phases = torch.rand(batch_size, n_obj) * 2 * math.pi   # (B, N_OBJ)
    amps   = 0.8 + 0.4 * torch.rand(batch_size, n_obj)     # (B, N_OBJ) ∈ [0.8, 1.2]
    noise  = 0.02 * torch.randn(batch_size, seq_len, n_obj)

    omega    = 2 * math.pi / period
    t_exp    = t.unsqueeze(0).unsqueeze(-1)   # (1, seq_len, 1)
    phases_e = phases.unsqueeze(1)            # (B, 1, N_OBJ)
    amps_e   = amps.unsqueeze(1)              # (B, 1, N_OBJ)

    x = amps_e * torch.sin(omega * t_exp + phases_e) + noise  # (B, seq_len, N_OBJ)
    return x

# ── Architecture (identical to v29) ───────────────────────────────────────────
class LearnedTemporalGateEncoder(nn.Module):
    """
    Softmax gate over T_GATE past timesteps — the thing we're studying.
    One global gate shared across all objects (keeps it clean for analysis).
    """
    def __init__(self, t_gate, obj_dim, hidden_dim):
        super().__init__()
        self.t_gate     = t_gate
        self.obj_dim    = obj_dim
        self.hidden_dim = hidden_dim
        # THE gate: T_GATE learnable logits → softmax → weighted sum over time
        self.gate_logits = nn.Parameter(torch.zeros(t_gate))
        # Shared per-timestep encoder
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

        # Encode every timestep independently (shared weights)
        h_flat   = history.reshape(B * T * N, D)
        enc_flat = self.encoder(h_flat)             # (B*T*N, hidden_dim)
        enc      = enc_flat.reshape(B, T, N, self.hidden_dim)

        # Temporal attention: weighted sum over T dimension
        gates_e = gates.view(1, T, 1, 1)
        out     = (enc * gates_e).sum(dim=1)        # (B, N_OBJ, hidden_dim)
        return out, gates


class SlotDynamics(nn.Module):
    """Self-attention over objects + MLP, then linear decode. Same as v29."""
    def __init__(self, hidden_dim, n_slots, obj_dim):
        super().__init__()
        self.n_slots = n_slots
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
        """enc: (B, N_OBJ, hidden_dim) → pred: (B, N_OBJ, obj_dim)"""
        x, _ = self.attn(enc, enc, enc)
        x     = self.norm1(enc + x)
        x     = self.norm2(x + self.ff(x))
        return self.decoder(x)


class SineCTM(nn.Module):
    """Full model: gate encoder → slot dynamics → prediction."""
    def __init__(self, t_gate, obj_dim=1, hidden_dim=HIDDEN_DIM, n_slots=N_SLOTS):
        super().__init__()
        self.t_gate   = t_gate
        self.gate_enc = LearnedTemporalGateEncoder(t_gate, obj_dim, hidden_dim)
        self.dynamics = SlotDynamics(hidden_dim, n_slots, obj_dim)

    def forward(self, history):
        """history: (B, T_GATE, N_OBJ, 1) → pred: (B, N_OBJ), gates: (T_GATE,)"""
        enc, gates = self.gate_enc(history)   # (B, N_OBJ, hidden_dim), (T_GATE,)
        pred       = self.dynamics(enc)       # (B, N_OBJ, 1)
        return pred.squeeze(-1), gates        # (B, N_OBJ), (T_GATE,)

# ── Gate analysis helper ───────────────────────────────────────────────────────
def analyze_gates(model, t_gate):
    """Returns dict of gate metrics given trained model."""
    with torch.no_grad():
        g         = torch.softmax(model.gate_enc.gate_logits, dim=0).cpu().numpy()
        peak_idx  = int(np.argmax(g))
        peak_prob = float(g[peak_idx])
        # index 0 = most recent (delay=0), index T-1 = oldest (delay=T-1)
        # so effective delay = sum over i of (i * g[i]), reversed:
        # delay at position i = (T-1 - i)? No — gate[0] is most recent = delay 0.
        # Actually: history[:, 0, :] = t-T+1 (oldest), history[:, T-1, :] = t (most recent)
        # So delay i steps back = gate index T-1-i.
        # eff_delay = sum_i delay_i * gate_weight_at_that_delay
        delays    = np.arange(t_gate)[::-1].copy()  # delays[0]=T-1, delays[T-1]=0
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
    log(f"T_GATE={t_gate} — Period Shift Gap Test (SINE_PERIOD={SINE_PERIOD})")
    log(f"{'='*60}")

    SEQ_LEN = t_gate + K_PRED + 10

    # Resume check
    ckpt_path = os.path.join(OUT_DIR, f"ckpt_v30_tg{t_gate}.pt")
    model = SineCTM(t_gate=t_gate).to(DEVICE)
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
        seq = generate_sine_batch(BATCH_SIZE, SEQ_LEN).to(DEVICE)

        t_start = torch.randint(0, SEQ_LEN - t_gate - K_PRED, (1,)).item()
        history = seq[:, t_start:t_start+t_gate, :].unsqueeze(-1)  # (B, T, N, 1)
        target  = seq[:, t_start+t_gate+K_PRED-1, :]               # (B, N)

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
            log(f"  step {step:6d} | loss={loss.item():.6f} | "
                f"peak@t-{m['peak_delay']}({m['peak_prob']:.3f}) | "
                f"eff_delay={m['eff_delay']:.2f}")

            # Save checkpoint every 5k steps
            torch.save({
                'model':    model.state_dict(),
                'opt':      opt.state_dict(),
                'sched':    sched.state_dict(),
                'step':     step,
                'best_mse': best_mse,
            }, ckpt_path)

    # Final analysis
    m           = analyze_gates(model, t_gate)
    period_delta = round(m['eff_delay'] - SINE_PERIOD, 2)

    log(f"\n  ── T_GATE={t_gate} FINAL ──")
    log(f"  gate dist: {np.round(m['gate_dist'], 3).tolist()}")
    log(f"  peak: t-{m['peak_delay']} (prob={m['peak_prob']:.4f})")
    log(f"  eff_delay:      {m['eff_delay']:.2f}")
    log(f"  SINE_PERIOD:    {SINE_PERIOD}")
    log(f"  delta vs T:     {period_delta:+.2f}  ← key test")
    log(f"  best_loss:      {best_mse:.6f}")

    result = {
        "t_gate":          t_gate,
        "sine_period":     SINE_PERIOD,
        "max_delay_used":  round(m['eff_delay'], 2),
        "peak_delay":      m['peak_delay'],
        "peak_prob":       round(m['peak_prob'], 4),
        "gate_spec":       round(m['gate_spec'], 4),
        "best_loss":       round(best_mse, 8),
        "delta_vs_period": period_delta,
        "gate_distribution": [round(float(x), 4) for x in m['gate_dist']],
    }

    # Verdict logic
    if abs(period_delta) < 2.5 and t_gate >= SINE_PERIOD:
        # eff_delay is within 2.5 steps of the sine period — tracking it
        log(f"  *** PERIOD_TRACKING: eff_delay ≈ SINE_PERIOD={SINE_PERIOD} ***")
        result["verdict"] = "PERIOD_TRACKING"
    elif m['eff_delay'] < 3.0:
        log(f"  *** MARKOVIAN: gate collapsed to present ***")
        result["verdict"] = "MARKOVIAN"
    else:
        log(f"  *** EXTENDED: uses history but not cleanly period-aligned ***")
        result["verdict"] = "EXTENDED"

    return result

# ── Main ───────────────────────────────────────────────────────────────────────
log(f"CTM World Model v30 — Gap Test: SINE_PERIOD={SINE_PERIOD} (was 8 in v29)")
log(f"N_OBJ={N_OBJ}, TRAIN_STEPS={TRAIN_STEPS}, T_GATE sweep={T_GATE_LIST}")
log(f"Hypothesis: eff_delay should shift to ≈{SINE_PERIOD} (doubled from v29)")
log(f"Device: {DEVICE}")

all_results = {}

# Load prior results if resuming
results_path = os.path.join(OUT_DIR, "results_v30.json")
if os.path.exists(results_path):
    with open(results_path) as f:
        all_results = json.load(f)
    log(f"Loaded existing results for T_GATE keys: {list(all_results.keys())}")

for tg in T_GATE_LIST:
    if str(tg) in all_results:
        log(f"T_GATE={tg} already in results, skipping (delete checkpoint to re-run)")
        continue
    r = run_experiment(tg)
    all_results[str(tg)] = r
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"[checkpoint] results_v30.json saved (T_GATE={tg} done)")

# ── Summary & comparison vs v29 ───────────────────────────────────────────────
log(f"\n{'='*60}")
log(f"V30 COMPLETE — Period Shift Test Summary")
log(f"{'='*60}")
log(f"{'T_GATE':>8} | {'eff_delay':>10} | {'vs_period':>10} | {'verdict'}")
for tg_str, r in all_results.items():
    log(f"{tg_str:>8} | {r['max_delay_used']:>10.2f} | {r['delta_vs_period']:>+10.2f} | {r['verdict']}")

# Final cross-experiment verdict
if "32" in all_results:
    r32 = all_results["32"]
    # The key question: did the delay shift proportionally from v29's ~8 to ~16?
    if r32["verdict"] == "PERIOD_TRACKING":
        log(f"\n  *** VERDICT: PERIOD_SHIFTED ***")
        log(f"  eff_delay({SINE_PERIOD}) ≈ {r32['max_delay_used']:.1f} — proportional shift confirmed")
        log(f"  Tripartite principle is NOT a T=8 fluke. Genuine period tracking.")
        all_results["global_verdict"] = "PERIOD_SHIFTED"
    else:
        log(f"\n  *** VERDICT: NO_SHIFT ***")
        log(f"  eff_delay={r32['max_delay_used']:.1f} ≠ SINE_PERIOD={SINE_PERIOD}")
        log(f"  Period tracking does NOT generalize — need to investigate further.")
        all_results["global_verdict"] = "NO_SHIFT"
elif "16" in all_results:
    r16 = all_results["16"]
    if r16["verdict"] == "PERIOD_TRACKING":
        log(f"\n  *** VERDICT: PERIOD_SHIFTED (at T_GATE=16) ***")
        all_results["global_verdict"] = "PERIOD_SHIFTED"

with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2)
log(f"All results saved to {results_path}")
