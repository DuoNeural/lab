"""
tau_trajectory.py — DHP τ*(t) Trajectory During Training
DuoNeural Lab — Archon, 2026-05-15

Measures how the optimal prediction horizon τ* evolves across training steps.
Three hypotheses under test:
  H1: monotonic growth  — τ* climbs smoothly to ~72%τ_L
  H2: overshoot/correct — τ* exceeds τ_L briefly, snaps back
  H3: phase transitions — τ* plateaus, then jumps (grokking-like)

Hardware targets:
  3090 (24GB): LORENZ63
  3060-A (12GB): ROSSLER
  3060-B (12GB): LORENZ96
  kilonova (16GB UMA): analysis / theory

Usage:
  SYSTEM=lorenz63 python3 tau_trajectory.py
  SYSTEM=rossler  python3 tau_trajectory.py
  SYSTEM=lorenz96 python3 tau_trajectory.py
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import time
import argparse
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYSTEM = os.environ.get("SYSTEM", "lorenz63")
OUT_DIR = os.environ.get("OUT_DIR", f"/workspace/tau_traj_{SYSTEM}")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"[tau_traj] system={SYSTEM}  device={DEVICE}  out={OUT_DIR}")

# =============================================================================
# CHAOTIC SYSTEMS
# =============================================================================

def _rk4_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5*dt*k1)
    k3 = f(x + 0.5*dt*k2)
    k4 = f(x + dt*k3)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def gen_lorenz63(n_steps, dt=0.05, sigma=10.0, rho=28.0, beta=8/3, seed=42):
    # RK4, dt=0.05 → τ_L≈22 steps (matches CTM v40); Euler unstable at this dt
    np.random.seed(seed)
    xyz = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    def f(v): x,y,z=v; return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
    for _ in range(5000): xyz = _rk4_step(f, xyz, dt)
    traj = []
    for _ in range(n_steps):
        xyz = _rk4_step(f, xyz, dt)
        traj.append(xyz.copy())
    arr = np.array(traj, dtype=np.float32)
    arr = (arr - arr.mean(0)) / (arr.std(0) + 1e-8)
    return arr  # [n_steps, 3]

def gen_lorenz96_d10(n_steps, D=10, dt=0.02, F=8.0, seed=42):
    """Lorenz96 D=10, RK4. λ_max≈2.5/unit → τ_L≈1/(2.5×0.02)≈20 steps."""
    np.random.seed(seed)
    x = np.random.randn(D).astype(np.float64) * 0.01
    x[0] += 0.01
    def f(x): return np.roll(x,1)*(np.roll(x,-1)-np.roll(x,2)) - x + F
    for _ in range(5000): x = _rk4_step(f, x, dt)
    traj = []
    for _ in range(n_steps):
        x = _rk4_step(f, x, dt)
        traj.append(x.copy())
    arr = np.array(traj, dtype=np.float32)
    arr = (arr - arr.mean(0)) / (arr.std(0) + 1e-8)
    return arr  # [n_steps, D]

def gen_lorenz96(n_steps, D=5, dt=0.02, F=8.0, seed=42):
    """RK4 integration — Euler diverges for L96 at dt>0.01."""
    np.random.seed(seed)
    x = np.random.randn(D).astype(np.float64) * 0.01
    x[0] += 0.01

    def l96_deriv(x):
        return np.roll(x,1)*(np.roll(x,-1)-np.roll(x,2)) - x + F

    # RK4 step
    def rk4(x, dt):
        k1 = l96_deriv(x)
        k2 = l96_deriv(x + 0.5*dt*k1)
        k3 = l96_deriv(x + 0.5*dt*k2)
        k4 = l96_deriv(x + dt*k3)
        return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # warmup
    for _ in range(5000):
        x = rk4(x, dt)

    traj = []
    for _ in range(n_steps):
        x = rk4(x, dt)
        traj.append(x.copy())
    arr = np.array(traj, dtype=np.float32)
    arr = (arr - arr.mean(0)) / (arr.std(0) + 1e-8)
    return arr  # [n_steps, D]

# Lyapunov times in STEPS (at our dt, from prior experiments / literature)
# lorenz63 dt=0.01: λ_max≈0.906/lyapunov_unit, τ_L≈1/0.906 time_units ≈ 110 steps
# rossler  dt=0.05: λ_max≈0.071/unit, τ_L≈14 units ≈ 280 steps
# lorenz96 dt=0.02: λ_max≈1.67/unit (F=8,D=5), τ_L≈0.6 units ≈ 30 steps
LYAPUNOV_STEPS = {
    "lorenz63": 22,    # dt=0.05, λ_max≈0.906/unit → τ_L≈1/(0.906×0.05)≈22 steps (matches CTM v40)
    "lorenz96d10": 20, # dt=0.02 RK4, D=10, λ_max≈2.5/unit → τ_L≈1/(2.5×0.02)≈20 steps
    "lorenz96": 30,    # dt=0.02 RK4, λ_max≈1.68/unit → τ_L≈1/(1.68×0.02)≈30 steps
}

DIM = {
    "lorenz63":    3,
    "lorenz96d10": 10,
    "lorenz96":    5,
}

DATA_GEN = {
    "lorenz63":    lambda n: gen_lorenz63(n),
    "lorenz96d10": lambda n: gen_lorenz96_d10(n),
    "lorenz96":    lambda n: gen_lorenz96(n),
}

# =============================================================================
# CTM WITH GHL GATE  (v40-style architecture confirmed for DHP)
# =============================================================================

class SoftGate(nn.Module):
    """Learned soft attention over lookback window. τ* = E[gate position]."""
    def __init__(self, d_model, T_gate):
        super().__init__()
        self.T_gate = T_gate
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, enc):
        # enc: [B, T, d]
        q = self.query(enc[:, -1:, :])          # [B, 1, d]
        k = self.key(enc)                         # [B, T, d]
        attn = (q * k * self.scale).sum(-1)       # [B, T]
        gate = torch.softmax(attn, dim=-1)         # [B, T] — soft distribution
        ctx = (gate.unsqueeze(-1) * enc).sum(1)   # [B, d]
        return ctx, gate


class TauCTM(nn.Module):
    """CTM with gate attention and multi-horizon GHL head."""
    def __init__(self, input_dim, d_model=64, T_gate=64, n_slots=8,
                 pred_horizons=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.T_gate = T_gate
        self.n_slots = n_slots
        self.pred_horizons = list(pred_horizons)
        self.input_dim = input_dim
        self.d_model = d_model

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        # Per-slot GRU (necessary for DHP — v40 finding)
        self.slot_rnns = nn.ModuleList([
            nn.GRUCell(d_model, d_model) for _ in range(n_slots)
        ])
        # Per-slot projection (necessary for DHP — v40 finding)
        self.slot_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_slots)
        ])
        self.gate = SoftGate(d_model, T_gate)
        # Concat decoder (necessary for DHP — v40 finding)
        self.decoder = nn.Linear(d_model * n_slots,
                                 input_dim * len(pred_horizons))

    def forward(self, x):
        # x: [B, T_gate, input_dim]
        B, T, _ = x.shape
        enc = self.encoder(x)              # [B, T, d_model]

        # run each slot over the full sequence
        slot_h = [torch.zeros(B, self.d_model, device=x.device)
                  for _ in range(self.n_slots)]
        for t in range(T):
            for i, rnn in enumerate(self.slot_rnns):
                slot_h[i] = rnn(enc[:, t, :], slot_h[i])

        # gate over encoded sequence
        _, gate = self.gate(enc)           # gate: [B, T_gate]

        # concat per-slot projections
        slot_out = torch.cat(
            [proj(h) for proj, h in zip(self.slot_projs, slot_h)], dim=-1
        )  # [B, d_model * n_slots]

        # multi-horizon predictions
        raw = self.decoder(slot_out)       # [B, input_dim * H]
        preds = raw.view(B, len(self.pred_horizons), self.input_dim)

        return preds, gate                  # preds:[B,H,D]  gate:[B,T_gate]


# =============================================================================
# DATA PIPELINE
# =============================================================================

def make_batches(data, T_gate, pred_horizons, batch_size=64, n_batches=None):
    """Randomly sample context windows + targets for each horizon."""
    max_h = max(pred_horizons)
    usable = len(data) - T_gate - max_h
    if usable <= 0:
        raise ValueError(f"Data too short: {len(data)} < {T_gate + max_h}")
    if n_batches is None:
        n_batches = max(1, usable // batch_size)

    idxs = np.random.randint(0, usable, (n_batches * batch_size,))
    for i in range(n_batches):
        batch_idxs = idxs[i*batch_size:(i+1)*batch_size]
        ctx = np.stack([data[j:j+T_gate] for j in batch_idxs])
        tgts = np.stack(
            [[data[j+T_gate+h-1] for h in pred_horizons] for j in batch_idxs]
        )
        yield (torch.tensor(ctx, device=DEVICE),
               torch.tensor(tgts, device=DEVICE))


# =============================================================================
# τ* MEASUREMENT
# =============================================================================

def measure_tau_star(model, data, T_gate, n_samples=512):
    """Measure τ* as weighted-mean gate position (in steps) + horizon sweep."""
    model.eval()
    gates_all = []
    horizon_mses = {h: [] for h in model.pred_horizons}

    max_h = max(model.pred_horizons)
    usable = len(data) - T_gate - max_h
    idxs = np.random.randint(0, usable, n_samples)

    with torch.no_grad():
        for i in range(0, n_samples, 64):
            batch_idxs = idxs[i:i+64]
            ctx = torch.tensor(
                np.stack([data[j:j+T_gate] for j in batch_idxs]), device=DEVICE)
            tgts = torch.tensor(
                np.stack([[data[j+T_gate+h-1] for h in model.pred_horizons]
                           for j in batch_idxs]), device=DEVICE)
            preds, gate = model(ctx)
            gates_all.append(gate.cpu().numpy())
            for hi, h in enumerate(model.pred_horizons):
                mse = ((preds[:, hi, :] - tgts[:, hi, :]) ** 2).mean().item()
                horizon_mses[h].append(mse)

    gates = np.concatenate(gates_all, axis=0)   # [n_samples, T_gate]
    mean_gate = gates.mean(axis=0)               # [T_gate]

    # τ* (gate): weighted mean position — how far back does model look?
    positions = np.arange(T_gate, dtype=np.float32)
    tau_gate = float((positions * mean_gate).sum())  # steps from present going backwards

    # horizon MSEs
    h_mse = {h: float(np.mean(v)) for h, v in horizon_mses.items()}

    # τ* (horizon): find where MSE gradient flattens (Lyapunov cliff)
    hs = sorted(h_mse.keys())
    mses = [h_mse[h] for h in hs]
    # gradient of MSE with respect to log(horizon) — where does it stop growing?
    log_hs = np.log(np.array(hs, dtype=np.float32) + 1)
    grads = np.gradient(mses, log_hs)
    # τ* horizon = horizon just before gradient spikes (last horizon where still predictable)
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

    tau_L = LYAPUNOV_STEPS[system]
    input_dim = DIM[system]
    pred_horizons = list(pred_horizons)

    print(f"[train] system={system}  τ_L={tau_L} steps  T_gate={T_gate}"
          f"  d_model={d_model}  n_slots={n_slots}")

    # generate data
    print("[train] generating chaotic trajectory...")
    data = DATA_GEN[system](total_steps * 10 + T_gate + max(pred_horizons) + 100)
    print(f"[train] data shape={data.shape}  mean={data.mean():.3f}  std={data.std():.3f}")

    # hold out last 20% for measurement
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data  = data[split:]

    model = TauCTM(input_dim=input_dim, d_model=d_model, T_gate=T_gate,
                   n_slots=n_slots, pred_horizons=pred_horizons).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    trajectory = []   # τ*(step) over training
    step = 0
    t0 = time.time()

    while step < total_steps:
        for ctx, tgts in make_batches(train_data, T_gate, pred_horizons,
                                      batch_size=batch_size, n_batches=20):
            model.train()
            preds, gate = model(ctx)  # preds:[B,H,D]  gate:[B,T_gate]

            # GHL loss: sum MSE across all horizons, weighted by recency
            # weight shorter horizons less (they're trivially easy) but include all
            loss = 0.0
            for hi, h in enumerate(pred_horizons):
                weight = np.log(h + 1)  # log-weight: long horizons matter more
                loss += weight * ((preds[:, hi, :] - tgts[:, hi, :]) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            step += 1

            if step % ckpt_every == 0 or step == 1:
                tau_meas = measure_tau_star(model, test_data, T_gate)
                tau_star_gate = tau_meas["tau_gate"]
                tau_star_horiz = tau_meas["tau_horizon"]
                ratio_gate  = tau_star_gate / tau_L
                ratio_horiz = tau_star_horiz / tau_L
                elapsed = time.time() - t0

                record = {
                    "step":         step,
                    "loss":         float(loss.item()),
                    "tau_gate":     tau_star_gate,
                    "tau_horizon":  tau_star_horiz,
                    "tau_L":        tau_L,
                    "ratio_gate":   ratio_gate,
                    "ratio_horiz":  ratio_horiz,
                    "horizon_mse":  tau_meas["horizon_mse"],
                    "elapsed_s":    elapsed,
                }
                trajectory.append(record)

                print(f"[step {step:6d}/{total_steps}]  loss={loss.item():.4f}"
                      f"  τ*_gate={tau_star_gate:.1f}  τ*/τ_L(gate)={ratio_gate:.3f}"
                      f"  τ*/τ_L(horiz)={ratio_horiz:.3f}"
                      f"  ({elapsed/60:.1f}min)")

                # save trajectory incrementally
                out_path = os.path.join(OUT_DIR, "tau_trajectory.json")
                with open(out_path, "w") as f:
                    json.dump({
                        "system":      system,
                        "tau_L":       tau_L,
                        "T_gate":      T_gate,
                        "n_slots":     n_slots,
                        "d_model":     d_model,
                        "pred_horizons": pred_horizons,
                        "total_steps": total_steps,
                        "ckpt_every":  ckpt_every,
                        "trajectory":  trajectory,
                        "status":      "running" if step < total_steps else "done",
                    }, f, indent=2)
                sys.stdout.flush()

            if step >= total_steps:
                break

    # final save
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
            "status":        "done",
        }, f, indent=2)

    print(f"\n[DONE] τ*(final)/τ_L = {trajectory[-1]['ratio_gate']:.3f} (gate)"
          f"  {trajectory[-1]['ratio_horiz']:.3f} (horizon)")
    print(f"[DONE] Results at {out_path}")
    return trajectory


if __name__ == "__main__":
    # allow env-var override of hyperparams
    train(
        system=SYSTEM,
        total_steps=int(os.environ.get("TOTAL_STEPS", "30000")),
        ckpt_every=int(os.environ.get("CKPT_EVERY", "500")),
        T_gate=int(os.environ.get("T_GATE", "64")),
        n_slots=int(os.environ.get("N_SLOTS", "8")),
        d_model=int(os.environ.get("D_MODEL", "64")),
        batch_size=int(os.environ.get("BATCH_SIZE", "64")),
        lr=float(os.environ.get("LR", "3e-4")),
    )
