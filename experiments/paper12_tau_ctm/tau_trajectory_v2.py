"""
tau_trajectory_v2.py — DHP τ*(t) Trajectory During Training (FIXED ARCHITECTURE)
DuoNeural Lab — Archon, 2026-05-15

v1 bug: gate was observed but not in the prediction path (ctx discarded).
v2 fix: per-slot gates, each slot selects its own lookback position from enc.
        Gate IS the lookback mechanism — gradient flows directly through it.
        No sequential GRU over the full window (that made gate redundant in v1).

Architecture (v2, v40-faithful):
  encode each timestep → enc[B, T, d]
  for each slot i:
    ctx_i, gate_i = slot_gates[i](enc)   — each slot picks different lookback
    h_i = GRUCell(ctx_i, zeros)           — processes gate-selected context
    out_i = slot_projs[i](h_i)
  concat → decoder → multi-horizon predictions
  τ*(t) = mean gate position across all slots

Three hypotheses:
  H1: monotonic growth  — τ* climbs smoothly toward ~72%τ_L
  H2: overshoot/correct — τ* exceeds τ_L briefly, snaps back
  H3: phase transitions — τ* plateaus, then jumps (grokking-like)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import time
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYSTEM = os.environ.get("SYSTEM", "lorenz63")
OUT_DIR = os.environ.get("OUT_DIR", f"/workspace/tau_traj_{SYSTEM}_v2")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"[tau_traj_v2] system={SYSTEM}  device={DEVICE}  out={OUT_DIR}")

# =============================================================================
# CHAOTIC SYSTEMS (same as v1, RK4 throughout)
# =============================================================================

def _rk4_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5*dt*k1)
    k3 = f(x + 0.5*dt*k2)
    k4 = f(x + dt*k3)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def gen_lorenz63(n_steps, dt=0.05, sigma=10.0, rho=28.0, beta=8/3, seed=42):
    np.random.seed(seed)
    xyz = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    def f(v): x,y,z=v; return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
    for _ in range(5000): xyz = _rk4_step(f, xyz, dt)
    traj = []
    for _ in range(n_steps):
        xyz = _rk4_step(f, xyz, dt); traj.append(xyz.copy())
    arr = np.array(traj, dtype=np.float32)
    return (arr - arr.mean(0)) / (arr.std(0) + 1e-8)

def gen_lorenz96_d10(n_steps, D=10, dt=0.02, F=8.0, seed=42):
    np.random.seed(seed)
    x = np.random.randn(D).astype(np.float64) * 0.01; x[0] += 0.01
    def f(x): return np.roll(x,1)*(np.roll(x,-1)-np.roll(x,2)) - x + F
    for _ in range(5000): x = _rk4_step(f, x, dt)
    traj = []
    for _ in range(n_steps):
        x = _rk4_step(f, x, dt); traj.append(x.copy())
    arr = np.array(traj, dtype=np.float32)
    return (arr - arr.mean(0)) / (arr.std(0) + 1e-8)

def gen_lorenz96(n_steps, D=5, dt=0.02, F=8.0, seed=42):
    np.random.seed(seed)
    x = np.random.randn(D).astype(np.float64) * 0.01; x[0] += 0.01
    def f(x): return np.roll(x,1)*(np.roll(x,-1)-np.roll(x,2)) - x + F
    for _ in range(5000): x = _rk4_step(f, x, dt)
    traj = []
    for _ in range(n_steps):
        x = _rk4_step(f, x, dt); traj.append(x.copy())
    arr = np.array(traj, dtype=np.float32)
    return (arr - arr.mean(0)) / (arr.std(0) + 1e-8)

LYAPUNOV_STEPS = {
    "lorenz63":    22,   # dt=0.05 RK4, λ_max≈0.906/unit → τ_L≈22 steps (v40 confirmed)
    "lorenz96d10": 20,   # dt=0.02 RK4, D=10, λ_max≈2.5/unit
    "lorenz96":    30,   # dt=0.02 RK4, D=5,  λ_max≈1.68/unit
}

DIM = {"lorenz63": 3, "lorenz96d10": 10, "lorenz96": 5}

DATA_GEN = {
    "lorenz63":    gen_lorenz63,
    "lorenz96d10": gen_lorenz96_d10,
    "lorenz96":    gen_lorenz96,
}

# =============================================================================
# v2 ARCHITECTURE — per-slot gates IN the prediction path
# =============================================================================

class SoftGate(nn.Module):
    """Soft attention over lookback window. τ* = E[position under gate distribution]."""
    def __init__(self, d_model, T_gate):
        super().__init__()
        self.T_gate = T_gate
        self.key   = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, enc):
        # enc: [B, T, d]  (each timestep independently encoded — no GRU pre-mixing)
        q = self.query(enc[:, -1:, :])         # [B, 1, d] — query from present
        k = self.key(enc)                       # [B, T, d]
        attn = (q * k * self.scale).sum(-1)    # [B, T]
        gate = torch.softmax(attn, dim=-1)      # [B, T] — distribution over lookback
        ctx  = (gate.unsqueeze(-1) * enc).sum(1)  # [B, d] — gate-selected context
        return ctx, gate


class TauCTM(nn.Module):
    """
    v2: Per-slot gates, each slot selects its own temporal context.
    Gate IS the lookback mechanism — gradient flows through it directly.
    τ* = mean of (weighted mean gate position) across all slots.
    """
    def __init__(self, input_dim, d_model=64, T_gate=64, n_slots=8,
                 pred_horizons=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.T_gate       = T_gate
        self.n_slots      = n_slots
        self.pred_horizons = list(pred_horizons)
        self.input_dim    = input_dim
        self.d_model      = d_model

        # Independent timestep encoder (no sequential processing — keeps positions meaningful)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # Per-slot gates (v40 finding: per-slot specialization necessary for DHP)
        self.slot_gates = nn.ModuleList([
            SoftGate(d_model, T_gate) for _ in range(n_slots)
        ])
        # Per-slot GRU cells — process gate-selected context (not full sequence)
        self.slot_rnns = nn.ModuleList([
            nn.GRUCell(d_model, d_model) for _ in range(n_slots)
        ])
        # Per-slot projections (v40 confirmed necessary)
        self.slot_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_slots)
        ])
        # Concat decoder (v40 confirmed necessary)
        self.decoder = nn.Linear(d_model * n_slots,
                                 input_dim * len(pred_horizons))

    def forward(self, x):
        # x: [B, T_gate, input_dim]
        B, T, _ = x.shape

        # Encode each timestep independently — positions remain meaningful
        enc = self.encoder(x)   # [B, T, d_model]

        slot_outputs = []
        all_gates    = []

        for gate_fn, rnn, proj in zip(self.slot_gates, self.slot_rnns, self.slot_projs):
            # Each slot selects its own lookback position — gradient flows here
            ctx_i, gate_i = gate_fn(enc)                              # ctx:[B,d]  gate:[B,T]
            # Process gate-selected context (fresh hidden state per sample — batch training)
            h_i = rnn(ctx_i, torch.zeros(B, self.d_model, device=x.device))
            slot_outputs.append(proj(h_i))
            all_gates.append(gate_i)

        # τ* = mean gate distribution across slots → weighted mean position
        gate_mean = torch.stack(all_gates, dim=1).mean(1)   # [B, T]

        slot_out = torch.cat(slot_outputs, dim=-1)           # [B, d_model * n_slots]
        raw = self.decoder(slot_out)                         # [B, input_dim * H]
        preds = raw.view(B, len(self.pred_horizons), self.input_dim)

        return preds, gate_mean                              # preds:[B,H,D]  gate:[B,T]


# =============================================================================
# DATA PIPELINE
# =============================================================================

def make_batches(data, T_gate, pred_horizons, batch_size=64, n_batches=None):
    max_h  = max(pred_horizons)
    usable = len(data) - T_gate - max_h
    if usable <= 0:
        raise ValueError(f"Data too short: {len(data)} < {T_gate + max_h}")
    if n_batches is None:
        n_batches = max(1, usable // batch_size)
    idxs = np.random.randint(0, usable, (n_batches * batch_size,))
    for i in range(n_batches):
        batch_idxs = idxs[i*batch_size:(i+1)*batch_size]
        ctx  = np.stack([data[j:j+T_gate]  for j in batch_idxs])
        tgts = np.stack([[data[j+T_gate+h-1] for h in pred_horizons]
                          for j in batch_idxs])
        yield (torch.tensor(ctx,  device=DEVICE),
               torch.tensor(tgts, device=DEVICE))


# =============================================================================
# τ* MEASUREMENT
# =============================================================================

def measure_tau_star(model, data, T_gate, n_samples=512):
    """
    τ*(gate): weighted mean gate position, averaged across slots and samples.
    τ*(horiz): last horizon before MSE gradient spikes (Lyapunov cliff detection).
    """
    model.eval()
    gates_all   = []
    horizon_mses = {h: [] for h in model.pred_horizons}

    max_h  = max(model.pred_horizons)
    usable = len(data) - T_gate - max_h
    idxs   = np.random.randint(0, usable, n_samples)

    with torch.no_grad():
        for i in range(0, n_samples, 64):
            batch_idxs = idxs[i:i+64]
            ctx  = torch.tensor(
                np.stack([data[j:j+T_gate] for j in batch_idxs]), device=DEVICE)
            tgts = torch.tensor(
                np.stack([[data[j+T_gate+h-1] for h in model.pred_horizons]
                           for j in batch_idxs]), device=DEVICE)
            preds, gate = model(ctx)
            gates_all.append(gate.cpu().numpy())
            for hi, h in enumerate(model.pred_horizons):
                mse = ((preds[:, hi, :] - tgts[:, hi, :]) ** 2).mean().item()
                horizon_mses[h].append(mse)

    gates     = np.concatenate(gates_all, axis=0)    # [n_samples, T_gate]
    mean_gate = gates.mean(axis=0)                    # [T_gate]

    # τ*(gate): positions indexed 0=oldest ... T_gate-1=most_recent
    # high position = attending to recent; low position = attending to distant past
    # lookback distance = T_gate - 1 - position (steps from present)
    positions     = np.arange(T_gate, dtype=np.float32)
    weighted_pos  = float((positions * mean_gate).sum())
    # convert to lookback: how many steps BACK from present
    tau_gate = float(T_gate - 1 - weighted_pos)  # steps of lookback

    h_mse = {h: float(np.mean(v)) for h, v in horizon_mses.items()}

    # τ*(horizon): Lyapunov cliff — last horizon before MSE gradient spikes
    hs      = sorted(h_mse.keys())
    mses    = [h_mse[h] for h in hs]
    log_hs  = np.log(np.array(hs, dtype=np.float32) + 1)
    grads   = np.gradient(mses, log_hs)
    threshold = np.max(grads) * 0.5
    tau_horizon = hs[0]
    for h, g in zip(hs, grads):
        if g < threshold:
            tau_horizon = h

    model.train()
    return {
        "tau_gate":    tau_gate,
        "tau_horizon": tau_horizon,
        "mean_gate":   mean_gate.tolist(),
        "horizon_mse": h_mse,
    }


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(system=SYSTEM, total_steps=30000, ckpt_every=500,
          T_gate=64, n_slots=8, d_model=64, batch_size=64,
          lr=3e-4, pred_horizons=(1, 2, 4, 8, 16, 32)):

    tau_L      = LYAPUNOV_STEPS[system]
    input_dim  = DIM[system]
    pred_horizons = list(pred_horizons)

    print(f"[train] system={system}  τ_L={tau_L} steps  T_gate={T_gate}"
          f"  d_model={d_model}  n_slots={n_slots}  [v2: per-slot gates]")

    print("[train] generating chaotic trajectory...")
    data = DATA_GEN[system](total_steps * 10 + T_gate + max(pred_horizons) + 100)
    print(f"[train] data shape={data.shape}  mean={data.mean():.3f}  std={data.std():.3f}")

    split      = int(len(data) * 0.8)
    train_data = data[:split]
    test_data  = data[split:]

    model = TauCTM(input_dim=input_dim, d_model=d_model, T_gate=T_gate,
                   n_slots=n_slots, pred_horizons=pred_horizons).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params:,}")

    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    trajectory = []
    step       = 0
    t0         = time.time()

    while step < total_steps:
        for ctx, tgts in make_batches(train_data, T_gate, pred_horizons,
                                      batch_size=batch_size, n_batches=20):
            model.train()
            preds, gate = model(ctx)   # gate: [B, T_gate] — mean across slots

            # GHL loss: log-weighted sum across horizons
            loss = 0.0
            for hi, h in enumerate(pred_horizons):
                weight = np.log(h + 1)
                loss += weight * ((preds[:, hi, :] - tgts[:, hi, :]) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            step += 1

            if step % ckpt_every == 0 or step == 1:
                tau_meas        = measure_tau_star(model, test_data, T_gate)
                tau_star_gate   = tau_meas["tau_gate"]
                tau_star_horiz  = tau_meas["tau_horizon"]
                ratio_gate      = tau_star_gate  / tau_L
                ratio_horiz     = tau_star_horiz / tau_L
                elapsed         = time.time() - t0

                record = {
                    "step":        step,
                    "loss":        float(loss.item()),
                    "tau_gate":    tau_star_gate,
                    "tau_horizon": tau_star_horiz,
                    "tau_L":       tau_L,
                    "ratio_gate":  ratio_gate,
                    "ratio_horiz": ratio_horiz,
                    "horizon_mse": tau_meas["horizon_mse"],
                    "elapsed_s":   elapsed,
                }
                trajectory.append(record)

                print(f"[step {step:6d}/{total_steps}]  loss={loss.item():.4f}"
                      f"  τ*_gate={tau_star_gate:.1f}  τ*/τ_L(gate)={ratio_gate:.3f}"
                      f"  τ*/τ_L(horiz)={ratio_horiz:.3f}"
                      f"  ({elapsed/60:.1f}min)")

                out_path = os.path.join(OUT_DIR, "tau_trajectory.json")
                with open(out_path, "w") as f:
                    json.dump({
                        "system":        system,
                        "tau_L":         tau_L,
                        "T_gate":        T_gate,
                        "n_slots":       n_slots,
                        "d_model":       d_model,
                        "pred_horizons": pred_horizons,
                        "total_steps":   total_steps,
                        "ckpt_every":    ckpt_every,
                        "trajectory":    trajectory,
                        "version":       "v2_per_slot_gates",
                        "status":        "running" if step < total_steps else "done",
                    }, f, indent=2)
                sys.stdout.flush()

            if step >= total_steps:
                break

    out_path = os.path.join(OUT_DIR, "tau_trajectory.json")
    with open(out_path, "w") as f:
        json.dump({
            "system":        system,
            "tau_L":         tau_L,
            "T_gate":        T_gate,
            "n_slots":       n_slots,
            "d_model":       d_model,
            "pred_horizons": pred_horizons,
            "total_steps":   total_steps,
            "ckpt_every":    ckpt_every,
            "trajectory":    trajectory,
            "version":       "v2_per_slot_gates",
            "status":        "done",
        }, f, indent=2)

    print(f"[train] DONE. Results at {out_path}")
    return trajectory


if __name__ == "__main__":
    train()
