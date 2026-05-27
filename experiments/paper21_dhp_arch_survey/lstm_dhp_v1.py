"""
LSTM DHP Experiment v1
======================
Does LSTM's forget gate align its cell state memory to the Lyapunov time
under multi-horizon training? Completing the 4-architecture DHP survey:

  CTM    → DHP ✓ (τ*/τ_L ≈ 0.72, confirmed Papers 1-6)
  GDN-2  → DHP ✗ (τ*/τ_L ≈ 0.015, confirmed Paper 20)
  Mamba  → DHP ? (running in parallel, mamba_dhp_v1.py)
  LSTM   → DHP ? (this script)

LSTM has explicit cell state (long-term memory) + hidden state (short-term).
Forget gate f_t = σ(W_f·[h_{t-1},x_t] + b_f) controls memory decay.
Effective memory timescale: τ* ≈ -1/log(f̄) where f̄ = mean forget gate.

If DHP: multi-horizon loss pressure should shift f̄ toward 1 - 1/τ_L,
        so that cell state naturally decays at the Lyapunov horizon.

Archon / DuoNeural Research / 2026-05-25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
import time
from datetime import datetime
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Time: {datetime.now().isoformat()}")
print("=" * 60)
print("LSTM DHP Experiment v1 — 4-Architecture Survey")
print("=" * 60)

# ── Chaotic Systems (same as P20 / Mamba v1) ──────────────────────────────────

def lorenz_trajectory(n_steps, dt=0.01):
    traj = np.zeros((n_steps, 3))
    x = np.array([1.0, 0.0, 0.0])
    for i in range(n_steps):
        traj[i] = x
        dx = 10.0 * (x[1] - x[0])
        dy = x[0] * (28.0 - x[2]) - x[1]
        dz = x[0] * x[1] - (8/3) * x[2]
        x = x + dt * np.array([dx, dy, dz])
    return traj

def rossler_trajectory(n_steps, dt=0.02):
    traj = np.zeros((n_steps, 3))
    x = np.array([1.0, 0.0, 0.0])
    for i in range(n_steps):
        traj[i] = x
        dx = -x[1] - x[2]
        dy = x[0] + 0.2 * x[1]
        dz = 0.2 + x[2] * (x[0] - 5.7)
        x = x + dt * np.array([dx, dy, dz])
    return traj

TAU_L = {"lorenz": 110.0, "rossler": 700.0}

# ── LSTM Model ─────────────────────────────────────────────────────────────────

class LSTMPredictor(nn.Module):
    """
    Multi-layer LSTM for chaotic system prediction.
    We use vanilla nn.LSTM for clean forget gate access.
    """
    def __init__(self, input_dim=3, hidden_dim=128, n_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward_step(self, x, state):
        """x: (B, input_dim), state: (h, c) each (n_layers, B, hidden_dim)"""
        x_proj = self.input_proj(x).unsqueeze(1)  # (B, 1, hidden_dim)
        out, new_state = self.lstm(x_proj, state)
        pred = self.output_proj(out.squeeze(1))   # (B, input_dim)
        return pred, new_state

    def init_state(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=DEVICE)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=DEVICE)
        return (h, c)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_forget_gate_mean(self, x_seq, n_steps=500):
        """
        Run x_seq through LSTM and collect mean forget gate values.
        LSTM forget gate = σ(W_hh_f·h + W_ih_f·x + b_hf + b_ih_f)
        We reconstruct from weight matrices.
        """
        self.eval()
        forget_vals = []
        x_t = torch.tensor(x_seq[:n_steps], dtype=torch.float32, device=DEVICE)
        x_proj = self.input_proj(x_t)  # (n_steps, hidden_dim)

        state = self.init_state(1)
        h, c = state
        # Access LSTM weight matrices for layer 0
        W_ih = self.lstm.weight_ih_l0  # (4*hidden, input)
        W_hh = self.lstm.weight_hh_l0  # (4*hidden, hidden)
        b_ih = self.lstm.bias_ih_l0
        b_hh = self.lstm.bias_hh_l0
        H = self.hidden_dim

        with torch.no_grad():
            for t in range(n_steps):
                x_in = x_proj[t:t+1]  # (1, hidden_dim)
                h_prev = h[0]          # (1, hidden_dim)

                # LSTM gates: i, f, g, o — split W_ih and W_hh into 4
                gates = x_in @ W_ih.t() + h_prev @ W_hh.t() + b_ih + b_hh
                f_gate = torch.sigmoid(gates[:, H:2*H])  # forget gate
                forget_vals.append(f_gate.mean().item())

                # Full LSTM step to update state
                _, state = self.forward_step(x_t[t:t+1].unsqueeze(0).squeeze(1)
                    if t == 0 else x_t[t].unsqueeze(0), state)
                h, c = state

        return float(np.mean(forget_vals))


# ── Training ───────────────────────────────────────────────────────────────────

def make_batches(traj_norm, seq_len=64, batch_size=32, max_horizon=16):
    traj_t = torch.tensor(traj_norm, dtype=torch.float32, device=DEVICE)
    N = len(traj_t)
    starts = torch.randint(0, N - seq_len - max_horizon, (batch_size,))
    return torch.stack([traj_t[s:s+seq_len] for s in starts])

def multi_horizon_loss(model, x_batch, horizons, temperature):
    B, T, D = x_batch.shape
    state = model.init_state(B)
    warmup = min(32, T // 2)

    for t in range(warmup):
        _, state = model.forward_step(x_batch[:, t], state)

    preds = []
    for t in range(warmup, T):
        pred, state = model.forward_step(x_batch[:, t], state)
        preds.append(pred)
    preds = torch.stack(preds, dim=1)

    loss, total_w = 0.0, 0.0
    max_k = max(horizons)
    for k in horizons:
        w = math.exp((k - max_k) / temperature)  # normalized: max weight = 1
        tlen = T - warmup - k
        if tlen <= 0:
            continue
        loss += w * F.mse_loss(preds[:, :tlen], x_batch[:, warmup+k:warmup+k+tlen])
        total_w += w
    return loss / (total_w + 1e-8)

def train(system, traj_norm, tau_L, horizons, n_steps=5000,
          temp_start=2.0, temp_end=0.1, lr=3e-4, batch_size=32):
    model = LSTMPredictor(input_dim=3, hidden_dim=128, n_layers=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps, eta_min=lr*0.1)

    print(f"\n{'='*50}")
    print(f"System: {system}  |  Horizons: {horizons}  |  τ_L={tau_L}")
    print(f"Params: {model.param_count():,}")

    t0 = time.time()
    for step in range(n_steps):
        frac = step / n_steps
        temp = temp_start * (temp_end / temp_start) ** frac
        x_batch = make_batches(traj_norm, seq_len=96, batch_size=batch_size,
                               max_horizon=max(horizons))
        loss = multi_horizon_loss(model, x_batch, horizons, temp)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"  step {step:5d}/{n_steps}  loss={loss.item():.6f}  "
                  f"t={time.time()-t0:.0f}s  temp={temp:.3f}")

    print(f"  Done. {time.time()-t0:.0f}s")
    return model

# ── State Injection ────────────────────────────────────────────────────────────

def measure_tau_star(model, traj_norm, n_seeds=6, warmup=200,
                     measure_steps=500, eps=1e-4):
    model.eval()
    traj_t = torch.tensor(traj_norm, dtype=torch.float32, device=DEVICE)
    N = len(traj_t)
    all_deltas = []

    with torch.no_grad():
        for seed in range(n_seeds):
            start = seed * (N // (n_seeds + 2)) + 50

            # Warm up
            state = model.init_state(1)
            for t in range(warmup):
                _, state = model.forward_step(traj_t[start+t].unsqueeze(0), state)

            # Fork and perturb BOTH h and c
            h_ref, c_ref = state
            h_pert = h_ref.clone() + eps * torch.randn_like(h_ref)
            c_pert = c_ref.clone() + eps * torch.randn_like(c_ref)
            state_ref = (h_ref, c_ref)
            state_pert = (h_pert, c_pert)

            delta_0 = (h_pert - h_ref).norm().item() + (c_pert - c_ref).norm().item()

            deltas = []
            for t in range(measure_steps):
                idx = start + warmup + t
                if idx >= N - 1:
                    break
                inp = traj_t[idx].unsqueeze(0)
                _, state_ref = model.forward_step(inp, state_ref)
                _, state_pert = model.forward_step(inp, state_pert)

                h_r, c_r = state_ref
                h_p, c_p = state_pert
                delta_t = (h_p - h_r).norm().item() + (c_p - c_r).norm().item()
                deltas.append(delta_t / (delta_0 + 1e-12))

            all_deltas.append(deltas)

    min_len = min(len(d) for d in all_deltas)
    avg = np.array([d[:min_len] for d in all_deltas]).mean(0)

    tau_star = float(min_len)
    for t, v in enumerate(avg):
        if v < 1.0 / math.e:
            if t > 0:
                prev = avg[t-1]
                frac = (prev - 1/math.e) / (prev - v + 1e-12)
                tau_star = (t - 1) + frac
            else:
                tau_star = float(t)
            break

    return tau_star, avg.tolist()

# ── Main ───────────────────────────────────────────────────────────────────────

MULTI = [1, 2, 4, 8, 16]
SINGLE = [1]
N_STEPS = 5000
results = {}

for system in ["lorenz", "rossler"]:
    traj_raw = lorenz_trajectory(50000) if system == "lorenz" else rossler_trajectory(50000)
    mean, std = traj_raw.mean(0), traj_raw.std(0) + 1e-8
    traj_norm = (traj_raw - mean) / std
    tau_L = TAU_L[system]

    for horizons, label in [(MULTI, "multi"), (SINGLE, "single")]:
        key = f"{system}_{label}"
        t_start = 2.0 if label == "multi" else 1.0
        t_end   = 0.1 if label == "multi" else 1.0

        model = train(system, traj_norm, tau_L, horizons, N_STEPS,
                      temp_start=t_start, temp_end=t_end)

        print(f"  [Measuring τ*...]")
        tau_star, deltas = measure_tau_star(model, traj_norm)
        ratio = tau_star / tau_L
        dhp = 0.60 <= ratio <= 0.85
        f_mean = model.get_forget_gate_mean(traj_norm)
        # Theoretical τ* from forget gate: -1/log(f̄)
        tau_theoretical = -1.0 / math.log(f_mean + 1e-12) if f_mean < 1.0 else float("inf")

        print(f"\n  ── Results ({system}, {label}) ──")
        print(f"  τ_L         = {tau_L:.1f}")
        print(f"  τ* (inject) = {tau_star:.2f}")
        print(f"  τ*/τ_L      = {ratio:.4f}")
        print(f"  DHP         = {'✓ YES' if dhp else '✗ NO'}")
        print(f"  f̄ (forget) = {f_mean:.4f}")
        print(f"  τ* (theory) = {tau_theoretical:.2f}  (-1/log f̄)")

        results[key] = {
            "system": system, "label": label,
            "tau_L": tau_L, "tau_star": tau_star,
            "tau_ratio": ratio, "dhp": dhp,
            "f_mean": f_mean,
            "tau_theoretical": tau_theoretical,
            "delta_curve": deltas[:100],
        }

# Summary
print("\n\n── FINAL SUMMARY ────────────────────────────────────────────────")
print(f"{'System':<12}{'Cond':<10}{'τ*':<10}{'τ*/τ_L':<10}{'f̄':<10}{'DHP'}")
print("-" * 55)
for key, r in results.items():
    print(f"{r['system']:<12}{r['label']:<10}{r['tau_star']:<10.2f}"
          f"{r['tau_ratio']:<10.4f}{r['f_mean']:<10.4f}{'✓' if r['dhp'] else '✗'}")

results["metadata"] = {
    "experiment": "lstm_dhp_v1",
    "date": datetime.now().isoformat(),
    "model": "LSTM (nn.LSTM, 2-layer, hidden=128)",
    "multi_horizons": MULTI,
    "n_steps": N_STEPS,
    "dhp_window": [0.60, 0.85],
    "note": "4-architecture DHP survey: CTM✓, GDN-2✗, Mamba(?), LSTM(?)"
}

out = Path("/workspace/lstm_dhp_results_v1.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out}")
