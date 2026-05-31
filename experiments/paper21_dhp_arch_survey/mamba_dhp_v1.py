"""
Mamba DHP Experiment v1
=======================
Does Mamba's selective scan align its effective memory to the Lyapunov time
under multi-horizon training, the same way CTM does?

Architecture: pure-PyTorch Mamba SSM (no mamba_ssm/Triton dependency).
Gives us full state access for the state injection measurement protocol.

Protocol mirrors P20 (GDN-2 DHP):
  - Train on Lorenz-63 and Rössler chaotic systems
  - Multi-horizon: H={1,2,4,8,16} with temperature annealing (DHP condition)
  - Single-horizon: H={1} (negative control)
  - Measure τ* via state injection: fork state, measure δ(t) decay
  - Compare τ*/τ_L — DHP predicts ≈ 0.72

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

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Time: {datetime.utcnow().isoformat()}")
print("=" * 60)
print("Mamba DHP Experiment v1")
print("=" * 60)

# ── Chaotic Systems ────────────────────────────────────────────────────────────

def lorenz_trajectory(n_steps, dt=0.01, rho=28.0, sigma=10.0, beta=8/3,
                       x0=None, noise=0.0):
    if x0 is None:
        x0 = np.array([1.0, 0.0, 0.0])
    traj = np.zeros((n_steps, 3))
    x = x0.copy()
    for i in range(n_steps):
        traj[i] = x
        dx = sigma * (x[1] - x[0])
        dy = x[0] * (rho - x[2]) - x[1]
        dz = x[0] * x[1] - beta * x[2]
        x = x + dt * np.array([dx, dy, dz])
        if noise > 0:
            x += noise * np.random.randn(3)
    return traj

def rossler_trajectory(n_steps, dt=0.02, a=0.2, b=0.2, c=5.7, x0=None):
    if x0 is None:
        x0 = np.array([1.0, 0.0, 0.0])
    traj = np.zeros((n_steps, 3))
    x = x0.copy()
    for i in range(n_steps):
        traj[i] = x
        dx = -x[1] - x[2]
        dy = x[0] + a * x[1]
        dz = b + x[2] * (x[0] - c)
        x = x + dt * np.array([dx, dy, dz])
    return traj

def lyapunov_time(system):
    """Analytical Lyapunov times (steps) matching P20."""
    return {"lorenz": 110.0, "rossler": 700.0}[system]

# ── Pure-PyTorch Mamba SSM ─────────────────────────────────────────────────────

class MambaSSM(nn.Module):
    """
    Simplified Mamba block with selective scan.
    State: h ∈ R^(d_state)
    Input-dependent gates: dt (step size), B (input proj), C (output proj)
    Fixed structured A (diagonal, initialized for stability)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # Input projection (splits into x and z for gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Conv layer for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt_raw
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)  # dt broadcast

        # A: log-parameterized diagonal, shape (d_inner, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D (skip connection scalar per channel)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward_step(self, x, h):
        """
        Single-step forward. x: (batch, d_model), h: (batch, d_inner, d_state)
        Returns: y (batch, d_model), h_new (batch, d_inner, d_state)
        """
        B_batch = x.shape[0]

        # Input gate split
        xz = self.in_proj(x)  # (B, d_inner*2)
        x_in, z = xz.chunk(2, dim=-1)  # each (B, d_inner)

        # Apply activation
        x_in = F.silu(x_in)

        # SSM parameter computation (selective = input-dependent)
        ssm_params = self.x_proj(x_in)  # (B, 2*d_state + 1)
        B_param = ssm_params[:, :self.d_state]           # (B, d_state)
        C_param = ssm_params[:, self.d_state:2*self.d_state]  # (B, d_state)
        dt_raw = ssm_params[:, -1:]                       # (B, 1)

        # dt: softplus ensures positivity, broadcast to d_inner
        dt = F.softplus(self.dt_proj(dt_raw))  # (B, d_inner)

        # A: discretize using ZOH: A_bar = exp(dt * A)
        A = -torch.exp(self.A_log)  # (d_inner, d_state) — negative for stability
        # (B, d_inner, d_state) discretized A
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)

        # B_bar = dt * B (broadcast B across d_inner)
        B_bar = dt.unsqueeze(-1) * B_param.unsqueeze(1)  # (B, d_inner, d_state)

        # SSM step: h_new = A_bar * h + B_bar * x_in
        h_new = A_bar * h + B_bar * x_in.unsqueeze(-1)  # (B, d_inner, d_state)

        # Output: y = C * h_new + D * x_in
        # C: (B, d_state), h_new: (B, d_inner, d_state)
        y_ssm = (h_new * C_param.unsqueeze(1)).sum(-1)  # (B, d_inner)
        y = y_ssm + self.D.unsqueeze(0) * x_in           # (B, d_inner)

        # Gate with z
        y = y * F.silu(z)

        # Output projection
        out = self.out_proj(y)  # (B, d_model)
        return out, h_new

    def init_state(self, batch_size):
        return torch.zeros(batch_size, self.d_inner, self.d_state, device=DEVICE)


class MambaDHP(nn.Module):
    """Stack of Mamba blocks for chaotic system prediction."""
    def __init__(self, input_dim=3, d_model=64, d_state=16, n_layers=3, expand=2):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            MambaSSM(d_model, d_state=d_state, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward_step(self, x, states):
        """x: (B, input_dim), states: list of h tensors per layer"""
        h = self.input_proj(x)
        new_states = []
        for i, layer in enumerate(self.layers):
            h_out, h_new = layer.forward_step(h, states[i])
            h = h + h_out  # residual
            new_states.append(h_new)
        h = self.norm(h)
        pred = self.output_proj(h)
        return pred, new_states

    def init_states(self, batch_size):
        return [layer.init_state(batch_size) for layer in self.layers]

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ── Training ───────────────────────────────────────────────────────────────────

def make_batches(traj, seq_len=64, batch_size=32, max_horizon=16):
    """Sample random windows from trajectory for multi-horizon training."""
    traj_t = torch.tensor(traj, dtype=torch.float32, device=DEVICE)
    N = len(traj_t)
    starts = torch.randint(0, N - seq_len - max_horizon, (batch_size,))
    x_batch = torch.stack([traj_t[s:s+seq_len] for s in starts])
    return x_batch

def multi_horizon_loss(model, x_batch, horizons, temperature):
    """
    Multi-horizon prediction loss with temperature weighting.
    horizons: list of k values (e.g. [1,2,4,8,16])
    temperature: controls weighting (high T = uniform, low T = longer horizons)
    """
    B, T, D = x_batch.shape
    states = model.init_states(B)
    loss = 0.0
    total_weight = 0.0

    # Warm up on first 32 steps
    warmup = min(32, T // 2)
    for t in range(warmup):
        _, states = model.forward_step(x_batch[:, t], states)

    # Predict from warmup onward
    preds = []
    for t in range(warmup, T):
        pred, states = model.forward_step(x_batch[:, t], states)
        preds.append(pred)

    preds = torch.stack(preds, dim=1)  # (B, T-warmup, D)

    # Compute losses at each horizon — normalize by max weight for numerical stability
    max_k = max(horizons)
    for k in horizons:
        weight = math.exp((k - max_k) / temperature)  # max weight = 1 (at k=max_k)
        target_len = T - warmup - k
        if target_len <= 0:
            continue
        pred_k = preds[:, :target_len]
        target_k = x_batch[:, warmup+k:warmup+k+target_len]
        mse_k = F.mse_loss(pred_k, target_k)
        loss += weight * mse_k
        total_weight += weight

    return loss / (total_weight + 1e-8)

def train_model(system_name, traj, tau_L, pred_horizons, n_steps=4000,
                lr=3e-4, seq_len=96, batch_size=32, temp_start=2.0, temp_end=0.1):
    """Train MambaDHP model."""
    model = MambaDHP(input_dim=3, d_model=64, d_state=16, n_layers=3, expand=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr*0.1)

    # Normalize trajectory
    mean = traj.mean(0)
    std = traj.std(0) + 1e-8
    traj_norm = (traj - mean) / std

    print(f"\n{'='*50}")
    print(f"System: {system_name}")
    print(f"  τ_L = {tau_L} steps")
    print(f"  Params: {model.param_count():,}")
    print(f"  Horizons: {pred_horizons}")
    print(f"  Training {n_steps} steps...")

    t0 = time.time()
    for step in range(n_steps):
        frac = step / n_steps
        temperature = temp_start * (temp_end / temp_start) ** frac

        x_batch = make_batches(traj_norm, seq_len, batch_size, max(pred_horizons))
        loss = multi_horizon_loss(model, x_batch, pred_horizons, temperature)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if step % 500 == 0:
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{n_steps}  loss={loss.item():.6f}  t={elapsed:.0f}s  temp={temperature:.3f}")

    print(f"  Training complete. Total time: {time.time()-t0:.0f}s")
    return model, mean, std

# ── State Injection Measurement ────────────────────────────────────────────────

def measure_tau_star(model, traj_norm, n_seeds=6, warmup_steps=200,
                     measure_steps=300, eps=1e-4):
    """
    State injection protocol for Mamba.
    Forks hidden states after warmup, adds ε perturbation to all layer states,
    runs both on same inputs, measures δ(t) = ||h_ref - h_pert||_F decay.
    τ* = 1/e decay time.
    """
    model.eval()
    traj_t = torch.tensor(traj_norm, dtype=torch.float32, device=DEVICE)
    N = len(traj_t)

    all_deltas = []

    with torch.no_grad():
        for seed in range(n_seeds):
            start = seed * (N // (n_seeds + 2)) + 100

            # Warm up reference state
            states = model.init_states(1)
            for t in range(warmup_steps):
                idx = start + t
                _, states = model.forward_step(traj_t[idx:idx+1], states)

            # Fork and perturb
            states_ref = states
            states_pert = [s.clone() + eps * torch.randn_like(s) for s in states]

            # Initial perturbation norm
            delta_0 = sum(
                (states_pert[i] - states_ref[i]).norm().item()
                for i in range(len(states))
            )

            deltas = []
            for t in range(measure_steps):
                idx = start + warmup_steps + t
                if idx >= N - 1:
                    break
                inp = traj_t[idx:idx+1]
                _, states_ref = model.forward_step(inp, states_ref)
                _, states_pert = model.forward_step(inp, states_pert)

                delta_t = sum(
                    (states_pert[i] - states_ref[i]).norm().item()
                    for i in range(len(states))
                )
                deltas.append(delta_t / (delta_0 + 1e-12))

            all_deltas.append(deltas)

    # Average across seeds
    min_len = min(len(d) for d in all_deltas)
    avg_deltas = np.array([d[:min_len] for d in all_deltas]).mean(0)

    # Find τ* = 1/e crossing
    tau_star = None
    for t, delta in enumerate(avg_deltas):
        if delta < 1.0 / math.e:
            # Linear interpolation
            if t > 0:
                prev = avg_deltas[t-1]
                frac = (prev - 1/math.e) / (prev - delta + 1e-12)
                tau_star = (t - 1) + frac
            else:
                tau_star = float(t)
            break

    if tau_star is None:
        tau_star = float(min_len)  # didn't decay — very long memory

    return tau_star, avg_deltas.tolist()

# ── Gate Analysis ──────────────────────────────────────────────────────────────

def measure_gate_stats(model, traj_norm, n_steps=500):
    """
    Measure mean gate values to characterize Mamba's learned dynamics.
    For Mamba: dt (step size), A_bar (effective state decay) are the key gates.
    """
    model.eval()
    traj_t = torch.tensor(traj_norm, dtype=torch.float32, device=DEVICE)

    dt_vals = []
    A_bar_diag = []

    with torch.no_grad():
        states = model.init_states(1)
        for t in range(200, 200 + n_steps):
            inp = traj_t[t:t+1]
            # Run through first layer to get gate stats
            layer = model.layers[0]
            xz = layer.in_proj(model.input_proj(inp))
            x_in, _ = xz.chunk(2, dim=-1)
            x_in = F.silu(x_in)
            ssm_params = layer.x_proj(x_in)
            dt_raw = ssm_params[:, -1:]
            dt = F.softplus(layer.dt_proj(dt_raw))  # (1, d_inner)
            A = -torch.exp(layer.A_log)  # (d_inner, d_state)
            A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # (1, d_inner, d_state)

            dt_vals.append(dt.mean().item())
            A_bar_diag.append(A_bar.mean().item())

            _, states = model.forward_step(inp, states)

    return {
        "dt_mean": float(np.mean(dt_vals)),
        "A_bar_mean": float(np.mean(A_bar_diag)),
        "retention_per_step": float(np.mean(A_bar_diag)),  # analogous to GDN-2 α
    }

# ── Main Experiment ────────────────────────────────────────────────────────────

def run_experiment(system_name, pred_horizons, label, n_steps=4000):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {system_name.upper()} — {label}")
    print(f"{'='*60}")

    # Generate trajectory
    if system_name == "lorenz":
        traj = lorenz_trajectory(50000, dt=0.01)
        tau_L = lyapunov_time("lorenz")
    else:
        traj = rossler_trajectory(50000, dt=0.02)
        tau_L = lyapunov_time("rossler")

    # Normalize
    mean = traj.mean(0)
    std = traj.std(0) + 1e-8
    traj_norm = (traj - mean) / std

    # Train
    model, _, _ = train_model(
        system_name, traj, tau_L,
        pred_horizons=pred_horizons,
        n_steps=n_steps,
        temp_start=2.0 if len(pred_horizons) > 1 else 1.0,
        temp_end=0.1 if len(pred_horizons) > 1 else 1.0,
    )

    # Measure τ*
    print(f"  [Phase 2] State injection measurement...")
    tau_star, deltas = measure_tau_star(model, traj_norm)

    # Gate stats
    gate_stats = measure_gate_stats(model, traj_norm)

    tau_ratio = tau_star / tau_L
    dhp = (0.60 <= tau_ratio <= 0.85)

    print(f"\n  ── Results ({system_name}, {label}) ──")
    print(f"  τ_L                = {tau_L:.1f} steps")
    print(f"  τ*                 = {tau_star:.2f} steps")
    print(f"  τ*/τ_L             = {tau_ratio:.4f}")
    print(f"  DHP                = {'✓ YES' if dhp else '✗ NO'}")
    print(f"  dt_mean            = {gate_stats['dt_mean']:.4f}")
    print(f"  A_bar_mean (decay) = {gate_stats['A_bar_mean']:.4f}")

    return {
        "system": system_name,
        "label": label,
        "tau_L": tau_L,
        "tau_star": tau_star,
        "tau_ratio": tau_ratio,
        "dhp": dhp,
        "gate_stats": gate_stats,
        "delta_curve": deltas[:100],  # first 100 steps of decay curve
    }

# ── Run All Conditions ─────────────────────────────────────────────────────────

MULTI_HORIZONS = [1, 2, 4, 8, 16]
SINGLE_HORIZON = [1]
N_STEPS = 5000  # enough to converge cleanly

results = {}

# Multi-horizon (DHP condition) — both systems
for system in ["lorenz", "rossler"]:
    key = f"{system}_multi"
    results[key] = run_experiment(system, MULTI_HORIZONS, "multi-horizon (DHP condition)", N_STEPS)

# Single-horizon (negative control) — both systems
for system in ["lorenz", "rossler"]:
    key = f"{system}_single"
    results[key] = run_experiment(system, SINGLE_HORIZON, "single-horizon (negative control)", N_STEPS)

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n\n── FINAL SUMMARY ───────────────────────────────────────────────")
print(f"{'System':<12} {'Condition':<20} {'τ* (steps)':<14} {'τ*/τ_L':<10} {'DHP'}")
print("-" * 65)
for key, r in results.items():
    label = "multi" if "multi" in r["label"] else "single"
    print(f"{r['system']:<12} {label:<20} {r['tau_star']:<14.2f} {r['tau_ratio']:<10.4f} {'✓' if r['dhp'] else '✗'}")

# Save results
results["metadata"] = {
    "experiment": "mamba_dhp_v1",
    "date": datetime.utcnow().isoformat(),
    "model": "MambaDHP (pure-PyTorch)",
    "d_model": 64, "d_state": 16, "n_layers": 3,
    "multi_horizons": MULTI_HORIZONS,
    "n_steps": N_STEPS,
    "dhp_window": [0.60, 0.85],
    "note": "DHP in Mamba — does selective scan align to Lyapunov time?"
}

out_path = Path("/workspace/mamba_dhp_results_v1.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print(f"Timestamp: {datetime.utcnow().isoformat()}")
