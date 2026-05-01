#!/usr/bin/env python3
"""
CTM World Model v28 — Chaotic Dynamics (Lorenz Attractor)
Archon / DuoNeural, 2026-04-29

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v21-v27 used bouncing balls (near-Markovian physics).
Gates universally collapsed to [0, ~3]. Max delay ≤ 2.8 even with T_GATE=32.
VERDICT: window is task-intrinsic.

BUT — "task-intrinsic" only means as much as the task demands.
Ball bounce is nearly Markovian. Present state ~fully determines future.
Of COURSE the model stops at t-1.

v28 asks the harder question:

  What happens when history GENUINELY HELPS?

The Lorenz attractor (σ=10, ρ=28, β=8/3) is deterministic chaos:
  - Lyapunov exponent λ ≈ 0.9 per unit time
  - Predictability horizon ≈ 1 / λ ≈ 1.1 time units
  - With dt=0.05: ~22 steps before trajectories diverge
  - Knowing t-5 genuinely improves prediction vs t-1 alone
    because it lets you estimate dx/dt, d²x/dt² → better initial condition

If CTM gates still cluster at [0, ~3] on Lorenz:
  → The gating mechanism is architecturally present-biased.
    Even when deeper history IS informative, CTM won't use it.
    This is a fundamental architectural finding, not a task artifact.

If CTM gates reach to t-5, t-10 on Lorenz:
  → v21-v27 result was purely task-driven.
    Ball bounce was just too Markovian to need it.
    The architecture CAN use history — it just doesn't when it doesn't need to.

Same architecture as v27. Same T_GATE sweep {4, 8, 16, 32}.
Only the physics changes.

kilonova (gfx1103, ROCm).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import random
import json
import statistics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
N_OBJ         = 8          # 8 independent Lorenz attractors (structural parallel to v27's 8 balls)
SLOT_DIM      = 256
N_HEADS       = 8
OBJ_DIM       = 3          # (x, y, z) — no velocity needed, it's embedded in x(t), x(t-1)
PRED_HORIZONS = [1, 2, 4, 8, 16]
HORIZON_WEIGHTS = {1: 2.0, 2: 1.5, 4: 1.0, 8: 0.75, 16: 0.5}

# Lorenz parameters — classical butterfly attractor
LORENZ_SIGMA  = 10.0
LORENZ_RHO    = 28.0
LORENZ_BETA   = 8.0 / 3.0
DT            = 0.05       # same as v27 for direct comparison
                            # at dt=0.05, Lyapunov time ≈ 22 steps

# Normalization bounds (Lorenz attractor bounds with margin)
# x ∈ [-25, 25], y ∈ [-35, 35], z ∈ [0, 55]
LORENZ_NORM   = np.array([25.0, 35.0, 55.0], dtype=np.float32)
LORENZ_OFFSET = np.array([0.0, 0.0, 27.5], dtype=np.float32)  # center z

BATCH         = 128
TRAIN_STEPS   = 60_000
LR            = 2e-4
GRAD_CLIP     = 1.0
LOG_EVERY     = 5000
TBPTT_CUTOFF  = 8
TEMP_START    = 2.0
TEMP_END      = 0.1
LAMBDA_GATE   = 0.001

T_GATE_VALUES = [4, 8, 16, 32]   # identical sweep to v27

# spinup steps to get trajectories onto the attractor before recording
LORENZ_SPINUP = 500

OUT_DIR = Path("/home/ai/duoneural/ctm_world_model_v28")
OUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42); np.random.seed(42); random.seed(42)


def log(msg, log_path=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    path = log_path or (OUT_DIR / "wm_v28.log")
    with open(path, "a") as f:
        f.write(line + "\n")


def get_temperature(step):
    frac = step / TRAIN_STEPS
    return TEMP_START + (TEMP_END - TEMP_START) * frac


# ──────────────────────────────────────────────────────
# Lorenz physics
# ──────────────────────────────────────────────────────

def lorenz_deriv(state):
    """RK4 derivative for single (x,y,z) state."""
    x, y, z = state
    dx = LORENZ_SIGMA * (y - x)
    dy = x * (LORENZ_RHO - z) - y
    dz = x * y - LORENZ_BETA * z
    return np.array([dx, dy, dz])


def lorenz_step_rk4(state, dt):
    """Single RK4 step for Lorenz. More accurate than Euler at dt=0.05."""
    k1 = lorenz_deriv(state)
    k2 = lorenz_deriv(state + 0.5 * dt * k1)
    k3 = lorenz_deriv(state + 0.5 * dt * k2)
    k4 = lorenz_deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def normalize_lorenz(state):
    """Normalize to roughly [-1, 1] for each dimension."""
    return (state - LORENZ_OFFSET) / LORENZ_NORM


def generate_lorenz_trajectories(n_traj, t_len, n_obj, dt, spinup=LORENZ_SPINUP):
    """
    Generate n_traj trajectories, each containing n_obj independent Lorenz attractors.
    Shape: (n_traj, t_len, n_obj, 3)
    """
    all_trajs = []

    for _ in range(n_traj):
        # random ICs spread across attractor basin
        # Lorenz attractor lives roughly in x∈[-20,20], y∈[-27,27], z∈[2,48]
        init_states = []
        for _ in range(n_obj):
            x0 = np.random.uniform(-15, 15)
            y0 = np.random.uniform(-20, 20)
            z0 = np.random.uniform(5, 40)
            init_states.append(np.array([x0, y0, z0], dtype=np.float64))

        # spin up to get on the attractor
        states = [s.copy() for s in init_states]
        for _ in range(spinup):
            states = [lorenz_step_rk4(s, dt) for s in states]

        # record trajectory
        traj = []
        for _ in range(t_len):
            # normalize and store
            frame = np.stack([normalize_lorenz(s.astype(np.float32)) for s in states], axis=0)
            traj.append(frame)
            states = [lorenz_step_rk4(s, dt) for s in states]

        all_trajs.append(np.stack(traj, axis=0))  # (t_len, n_obj, 3)

    return np.stack(all_trajs, axis=0).astype(np.float32)  # (n_traj, t_len, n_obj, 3)


# ──────────────────────────────────────────────────────
# CTM Architecture — identical to v27, only OBJ_DIM changes
# ──────────────────────────────────────────────────────

class LearnedTemporalGateEncoder(nn.Module):
    def __init__(self, obj_dim, slot_dim, n_slots, t_gate):
        super().__init__()
        self.n_slots = n_slots
        self.t_gate  = t_gate
        self.scene_enc = nn.Sequential(
            nn.Linear(obj_dim * n_slots, slot_dim), nn.SiLU(),
            nn.Linear(slot_dim, slot_dim),
        )
        self.gate_logits = nn.Parameter(torch.zeros(n_slots, t_gate))
        self.slot_proj   = nn.ModuleList([nn.Linear(slot_dim, slot_dim) for _ in range(n_slots)])

    def get_gates(self, temperature=1.0):
        return F.softmax(self.gate_logits / temperature, dim=-1)

    def gate_entropy(self, temperature=1.0):
        gates = self.get_gates(temperature)
        return -(gates * torch.log(gates + 1e-10)).sum(dim=-1).mean()

    def forward(self, history, temperature=1.0):
        B, T, N, D = history.shape
        t_use = min(self.t_gate, T)
        recent = history[:, -t_use:, :, :].reshape(B, t_use, -1)
        scene_encs = self.scene_enc(recent)
        gates = self.get_gates(temperature)[:, -t_use:]
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        g = gates.unsqueeze(0).unsqueeze(-1)
        s = scene_encs.unsqueeze(1)
        gated = (g * s).sum(dim=2)
        return torch.stack([self.slot_proj[i](gated[:, i, :]) for i in range(self.n_slots)], dim=1)


class SlotGNNDynamics(nn.Module):
    def __init__(self, slot_dim, n_heads):
        super().__init__()
        self.slot_dim = slot_dim
        self.n_heads  = n_heads
        self.head_dim = slot_dim // n_heads
        self.scale    = self.head_dim ** -0.5
        self.q_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.k_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.v_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.o_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.norm1  = nn.LayerNorm(slot_dim)
        self.norm2  = nn.LayerNorm(slot_dim)
        self.mlp    = nn.Sequential(nn.Linear(slot_dim, slot_dim*2), nn.SiLU(), nn.Linear(slot_dim*2, slot_dim))

    def tick(self, slots):
        B, N, D = slots.shape
        H, Dh   = self.n_heads, self.head_dim
        q = self.q_proj(slots).view(B, N, H, Dh).transpose(1, 2)
        k = self.k_proj(slots).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(slots).view(B, N, H, Dh).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        msg  = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        slots = self.norm1(slots + self.o_proj(msg))
        return self.norm2(slots + self.mlp(slots))

    def forward(self, slots, n_ticks):
        for _ in range(n_ticks):
            slots = self.tick(slots)
        return slots


class SlotDecoder(nn.Module):
    def __init__(self, slot_dim, n_slots, obj_dim):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Linear(slot_dim * n_slots, slot_dim), nn.SiLU(),
            nn.Linear(slot_dim, obj_dim * n_slots),
        )
        self.obj_dim = obj_dim
        self.n_slots = n_slots

    def forward(self, slots):
        B = slots.shape[0]
        return self.dec(slots.view(B, -1)).view(B, self.n_slots, self.obj_dim)


class CTMWorldModelV28(nn.Module):
    def __init__(self, t_gate):
        super().__init__()
        self.encoder  = LearnedTemporalGateEncoder(OBJ_DIM, SLOT_DIM, N_OBJ, t_gate)
        self.dynamics = SlotGNNDynamics(SLOT_DIM, N_HEADS)
        self.decoder  = SlotDecoder(SLOT_DIM, N_OBJ, OBJ_DIM)

    def predict_all_horizons(self, history, horizons, temperature=1.0):
        slots  = self.encoder(history, temperature)
        preds  = {}
        prev_k = 0
        for k in sorted(horizons):
            delta  = k - prev_k
            slots  = self.dynamics(slots.detach() if k > TBPTT_CUTOFF else slots, delta)
            preds[k] = self.decoder(slots)
            prev_k = k
        return preds

    def forward(self, history, n_ticks, temperature=1.0):
        return self.decoder(self.dynamics(self.encoder(history, temperature), n_ticks))


# ──────────────────────────────────────────────────────
# Experiment runner — identical structure to v27
# ──────────────────────────────────────────────────────

def run_experiment(t_gate, train_data, test_data):
    log(f"\n{'='*60}")
    log(f"T_GATE={t_gate} | Lorenz chaos | window = {t_gate} steps back")
    log(f"Lyapunov time ≈ 22 steps at dt=0.05 — history genuinely informative past t-1")
    log(f"PREDICTION: uncertain — if gates reach t-5+, history matters. if [0,~3], arch is present-biased.")
    log(f"{'='*60}")

    torch.manual_seed(42); np.random.seed(42)
    model = CTMWorldModelV28(t_gate).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_STEPS)

    best_k1 = float("inf")
    t_ids   = np.arange(len(train_data))
    max_h   = max(PRED_HORIZONS)

    for step in range(TRAIN_STEPS):
        model.train()
        temp = get_temperature(step)
        idx  = np.random.choice(t_ids, BATCH, replace=True)
        batch = torch.from_numpy(train_data[idx]).to(DEVICE)
        B, T, N, D = batch.shape
        t0 = random.randint(t_gate + 1, T - max_h - 1)
        history = batch[:, :t0, :, :]

        preds = model.predict_all_horizons(history, PRED_HORIZONS, temp)
        loss  = sum(HORIZON_WEIGHTS[k] * F.mse_loss(preds[k], batch[:, t0+k, :, :])
                    for k in PRED_HORIZONS)
        gate_loss = model.encoder.gate_entropy(temp)
        (loss + LAMBDA_GATE * gate_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step(); opt.zero_grad(); sch.step()

        if step % LOG_EVERY == 0:
            model.eval()
            with torch.no_grad():
                eb   = torch.from_numpy(test_data[:256]).to(DEVICE)
                et0  = t_gate + 2
                eh   = eb[:, :et0, :, :]
                mse_k1 = F.mse_loss(model(eh, 1, temp), eb[:, et0+1, :, :]).item()
                gates  = model.encoder.get_gates(temp).cpu().numpy()
                mean_peak = float(np.max(gates, axis=1).mean())
            log(f"  T_GATE={t_gate} step {step:>5} | k1={mse_k1:.5f} peak={mean_peak:.3f} temp={temp:.3f}")
            if mse_k1 < best_k1:
                best_k1 = mse_k1
                torch.save(model.state_dict(), OUT_DIR / f"best_tgate{t_gate}.pt")

    # final gate analysis
    model.load_state_dict(torch.load(OUT_DIR / f"best_tgate{t_gate}.pt", map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        gates = model.encoder.get_gates(TEMP_END).cpu().numpy()

    t_indices = np.arange(t_gate)
    results   = []
    log(f"\n[T_GATE={t_gate}] Final gate distributions:")
    for s in range(N_OBJ):
        g = gates[s]
        peak_prob = float(g.max())
        eff_t_idx = float(np.dot(g, t_indices))
        eff_delay = (t_gate - 1) - eff_t_idx
        bar = "".join(("█" if v >= 0.1 else ("▓" if v >= 0.05 else ("░" if v >= 0.02 else "·"))) for v in g)
        log(f"  slot {s}: [{bar}]  peak@t-{t_gate-1-int(g.argmax())} ({peak_prob:.3f})  eff_delay={eff_delay:.1f}")
        results.append({"slot": s, "effective_delay": float(eff_delay), "peak_prob": peak_prob})

    mean_peak    = np.mean([r["peak_prob"] for r in results])
    eff_delays   = [r["effective_delay"] for r in results]
    gate_spec    = statistics.stdev(eff_delays)
    max_delay    = max(eff_delays)

    log(f"\n  T_GATE={t_gate} summary:")
    log(f"  mean_peak={mean_peak:.4f}  gate_spec={gate_spec:.4f}  max_delay_used={max_delay:.1f}")
    log(f"  best_k1_mse={best_k1:.6f}")

    # compare to v27 ball-bounce result for same T_GATE (reference)
    v27_ref = {4: 0.0, 8: 1.0, 16: 2.0, 32: 2.8}
    ref = v27_ref.get(t_gate)
    if ref is not None:
        delta = max_delay - ref
        if delta > 2.0:
            log(f"  → LORENZ UNLOCKS DEEPER HISTORY: max_delay={max_delay:.1f} vs ball={ref:.1f} (+{delta:.1f})")
            log(f"     Chaotic task drives gates further back than Markovian task.")
        elif delta > 0.5:
            log(f"  → SLIGHT HISTORY EXTENSION: max_delay={max_delay:.1f} vs ball={ref:.1f} (+{delta:.1f})")
        else:
            log(f"  → SAME AS BALL BOUNCE: max_delay={max_delay:.1f} vs ball={ref:.1f} (Δ={delta:+.1f})")
            log(f"     Architecture is present-biased regardless of task chaos.")

    return {
        "t_gate":           t_gate,
        "task":             "lorenz_chaos",
        "gate_results":     results,
        "mean_peakedness":  float(mean_peak),
        "gate_spec_score":  gate_spec,
        "max_delay_used":   float(max_delay),
        "best_k1_mse":      best_k1,
        "effective_delays": eff_delays,
        "v27_ball_ref":     ref,
    }


def main():
    log("=" * 60)
    log("CTM World Model v28 — Lorenz Chaotic Dynamics")
    log(f"device={DEVICE}, T_GATE sweep: {T_GATE_VALUES}")
    log(f"Lorenz σ={LORENZ_SIGMA}, ρ={LORENZ_RHO}, β={LORENZ_BETA:.4f}, dt={DT}")
    log(f"Lyapunov time ≈ {1/0.9/DT:.0f} steps — chaos sets in fast")
    log("Question: does CTM use deeper history when the task genuinely demands it?")
    log("=" * 60)

    log("generating Lorenz trajectories (spinup=500 steps to get on attractor)...")
    T_GEN = 120
    train_data = generate_lorenz_trajectories(6000, T_GEN, N_OBJ, DT)
    test_data  = generate_lorenz_trajectories(1000, T_GEN, N_OBJ, DT)
    log(f"train: {train_data.shape}, test: {test_data.shape}")
    log(f"data range: x∈[{train_data[:,:,:,0].min():.2f},{train_data[:,:,:,0].max():.2f}] "
        f"y∈[{train_data[:,:,:,1].min():.2f},{train_data[:,:,:,1].max():.2f}] "
        f"z∈[{train_data[:,:,:,2].min():.2f},{train_data[:,:,:,2].max():.2f}]")

    all_results = []
    for t_gate in T_GATE_VALUES:
        result = run_experiment(t_gate, train_data, test_data)
        all_results.append(result)

    log("\n" + "=" * 60)
    log("T_GATE SCALING SUMMARY — Lorenz vs Ball Bounce (v27)")
    log("=" * 60)
    log(f"\n{'T_GATE':>8} | {'mean_peak':>10} | {'gate_spec':>10} | {'max_delay':>10} | {'k1_mse':>10} | {'ball_ref':>8}")
    log("-" * 70)
    for r in all_results:
        ref_str = f"{r['v27_ball_ref']:.1f}" if r['v27_ball_ref'] is not None else "  N/A"
        log(f"{r['t_gate']:>8} | {r['mean_peakedness']:>10.4f} | {r['gate_spec_score']:>10.4f} | "
            f"{r['max_delay_used']:>10.1f} | {r['best_k1_mse']:>10.6f} | {ref_str:>8}")

    # overall verdict
    all_delays = [r["max_delay_used"] for r in all_results]
    ball_delays = [0.0, 1.0, 2.0, 2.8]
    lorenz_extends = sum(1 for l, b in zip(all_delays, ball_delays) if l - b > 2.0)

    log("\n" + "=" * 60)
    if lorenz_extends >= 2:
        log("  VERDICT: LORENZ EXTENDS TEMPORAL WINDOW")
        log("  Chaotic task unlocks deeper history use vs Markovian ball task.")
        log("  CTM CAN reach back further — it just didn't need to for balls.")
        log("  The [0,~3] window in v21-v27 was task-driven, not arch-limited.")
    else:
        log("  VERDICT: ARCHITECTURE IS PRESENT-BIASED")
        log("  Even Lorenz chaos doesn't push gates past [0,~4].")
        log("  CTM's temporal gating fundamentally favors the present.")
        log("  The delta function / short-window behavior is architectural,")
        log("  not merely a reflection of the task's Markov structure.")

    with open(OUT_DIR / "results_v28.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nresults → {OUT_DIR}/results_v28.json")
    log("CTM World Model v28 COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    main()
