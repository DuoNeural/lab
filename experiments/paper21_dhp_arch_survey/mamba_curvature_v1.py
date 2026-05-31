"""
Mamba + Temporal Straightening Curvature Regularizer
======================================================
Tests whether geometric trajectory straightening helps Mamba achieve DHP.

Hypothesis (from temporal straightening paper 2603.12231 + DHP connection):
  DHP-capable architectures have geometrically straight hidden-state trajectories.
  CTM's slot attention may produce lower curvature than Mamba/LSTM naturally.
  If explicit curvature regularization helps Mamba reach τ*/τ_L≈0.72,
  then geometry explains DHP. If it still fails, the constraint is structural.

Curvature regularizer (discrete 2nd derivative of hidden state trajectory):
  κ(t) = ||h(t+2) - 2h(t+1) + h(t)||²
  L_curve = mean(κ(t)) over sequence
  L_total = L_multihorizon + λ * L_curve

Lambda values: 0.0 (baseline), 0.01, 0.1, 1.0

Protocol: same as mamba_dhp_v1.py
  - Systems: Lorenz (τ_L=110), Rössler (τ_L=700)
  - H={1,2,4,8,16}, temp anneal 2.0→0.1, 5000 steps
  - State injection measurement, 6 seeds

Output: mamba_curvature_results_v1.json

DuoNeural / Archon / 2026-05-26
Inspired by arXiv:2603.12231 (Temporal Straightening for Latent Planning)
"""

import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32
SEED   = 42

# Mamba architecture (same as v1)
D_MODEL  = 64
D_STATE  = 16
N_LAYERS = 3

# Training
SEQ_LEN    = 96
BATCH      = 64
LR         = 3e-4
N_STEPS    = 5000
HORIZONS   = [1, 2, 4, 8, 16]
TEMP_INIT  = 2.0
TEMP_FINAL = 0.1

# Measurement
N_SEEDS  = 6
EPS      = 1e-4
N_MEAS   = 500
TAU_L    = {"lorenz": 110.0, "rossler": 700.0}
DHP_LO, DHP_HI = 0.60, 0.85

# Curvature regularizer lambda values to sweep
LAMBDA_VALUES = [0.0, 0.01, 0.1, 1.0]

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Data generators ───────────────────────────────────────────────────────────

def lorenz_trajectory(n, dt=0.01, warmup=5000):
    x = np.array([1.0, 1.0, 1.0])
    σ, ρ, β = 10.0, 28.0, 8/3
    for _ in range(warmup):
        dx = σ*(x[1]-x[0]); dy = x[0]*(ρ-x[2])-x[1]; dz = x[0]*x[1]-β*x[2]
        x = x + dt * np.array([dx, dy, dz])
    traj = np.zeros((n, 3))
    for i in range(n):
        dx = σ*(x[1]-x[0]); dy = x[0]*(ρ-x[2])-x[1]; dz = x[0]*x[1]-β*x[2]
        x = x + dt * np.array([dx, dy, dz])
        traj[i] = x
    return traj

def rossler_trajectory(n, dt=0.05, warmup=5000):
    x = np.array([0.1, 0.1, 0.1])
    a, b, c = 0.2, 0.2, 5.7
    for _ in range(warmup):
        dx = -x[1]-x[2]; dy = x[0]+a*x[1]; dz = b+x[2]*(x[0]-c)
        x = x + dt * np.array([dx, dy, dz])
    traj = np.zeros((n, 3))
    for i in range(n):
        dx = -x[1]-x[2]; dy = x[0]+a*x[1]; dz = b+x[2]*(x[0]-c)
        x = x + dt * np.array([dx, dy, dz])
        traj[i] = x
    return traj

TRAJ_GEN = {"lorenz": lorenz_trajectory, "rossler": rossler_trajectory}

def normalize(traj):
    mu = traj.mean(0); std = traj.std(0).clip(1e-6)
    return (traj - mu) / std

def make_batch(traj, seq_len, batch_size, max_horizon, rng):
    T = len(traj) - seq_len - max_horizon
    starts = rng.integers(0, T, batch_size)
    inp = np.stack([traj[s:s+seq_len] for s in starts])
    tgts = {k: np.stack([traj[s+k:s+seq_len+k] for s in starts]) for k in HORIZONS}
    return inp, tgts

# ── Mamba SSM (same as v1, with hidden state output) ─────────────────────────

class MambaLayer(nn.Module):
    def __init__(self, d_model=D_MODEL, d_state=D_STATE, d_conv=4, expand=2):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = d_model * expand

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                  padding=d_conv-1, groups=self.d_inner, bias=True)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state+1, dtype=DTYPE)
                                             .unsqueeze(0).expand(self.d_inner, -1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        nn.init.constant_(self.dt_proj.bias, math.log(math.expm1(1.0)))

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_.transpose(1,2))[:,:,:L].transpose(1,2)
        x_act  = F.silu(x_conv)

        params  = self.x_proj(x_act)
        B_t     = params[..., :self.d_state]
        C_t     = params[..., self.d_state:2*self.d_state]
        dt_raw  = params[..., -1:]
        dt      = F.softplus(self.dt_proj(dt_raw))

        A       = -torch.exp(self.A_log.float())
        A_bar   = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=DTYPE)
        ys = []
        for t in range(L):
            h = A_bar[:,t] * h + dt[:,t].unsqueeze(-1) * B_t[:,t].unsqueeze(1) * x_act[:,t].unsqueeze(-1)
            y = (h * C_t[:,t].unsqueeze(1)).sum(-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)
        output = y * self.D + x_act * self.D
        output = output * F.silu(z)
        return self.out_proj(output)

class MambaModel(nn.Module):
    def __init__(self, in_dim=3, d_model=D_MODEL, n_layers=N_LAYERS, out_dim=3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.layers     = nn.ModuleList([MambaLayer(d_model) for _ in range(n_layers)])
        self.norms      = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.heads      = nn.ModuleDict({str(k): nn.Linear(d_model, out_dim) for k in HORIZONS})

    def forward(self, x, return_hidden=False):
        """Returns predictions dict + optionally the hidden state trajectory."""
        h = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(norm(h))
        # h: (B, T, D) — the hidden state trajectory
        preds = {k: head(h) for k, head in self.heads.items()}
        if return_hidden:
            return preds, h
        return preds

# ── Curvature regularizer ─────────────────────────────────────────────────────

def curvature_loss(h):
    """
    Discrete curvature: L2 norm of second finite difference of hidden trajectory.
    κ(t) = ||h(t+2) - 2h(t+1) + h(t)||²
    L_curve = mean(κ(t)) over t ∈ [0, T-2]
    h: (B, T, D)
    """
    # second finite difference
    d2h = h[:, 2:, :] - 2 * h[:, 1:-1, :] + h[:, :-2, :]
    return (d2h ** 2).mean()

def mean_curvature(h):
    """Same but returns float for logging."""
    with torch.no_grad():
        return curvature_loss(h).item()

# ── Multi-horizon weights ─────────────────────────────────────────────────────

def horizon_weights(temperature):
    logits = torch.tensor([-math.log(k) / temperature for k in HORIZONS])
    return F.softmax(logits, dim=0)

# ── Training ──────────────────────────────────────────────────────────────────

def train(system, lam):
    print(f"\n  λ={lam:.3f} | system={system}")
    rng = np.random.default_rng(SEED)

    traj_raw = TRAJ_GEN[system](50000 + max(HORIZONS))
    traj = normalize(traj_raw)
    traj_t = torch.tensor(traj, dtype=DTYPE, device=DEVICE)

    model = MambaModel().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS, eta_min=LR*0.1)

    curve_log = []   # track mean curvature during training

    for step in range(1, N_STEPS + 1):
        progress = (step - 1) / N_STEPS
        temp = TEMP_INIT * (TEMP_FINAL / TEMP_INIT) ** progress
        weights = horizon_weights(temp)

        inp, tgts = make_batch(traj, SEQ_LEN, BATCH, max(HORIZONS), rng)
        x  = torch.tensor(inp, dtype=DTYPE, device=DEVICE)
        ys = {k: torch.tensor(tgts[k], dtype=DTYPE, device=DEVICE) for k in HORIZONS}

        preds, h = model(x, return_hidden=True)
        pred_loss = sum(weights[i] * F.mse_loss(preds[str(k)], ys[k])
                        for i, k in enumerate(HORIZONS))
        curve_reg = curvature_loss(h) if lam > 0 else torch.tensor(0.0, device=DEVICE)
        loss = pred_loss + lam * curve_reg

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 1000 == 0:
            curve_log.append(mean_curvature(h))
            print(f"    step={step:5d} pred={pred_loss.item():.5f} "
                  f"curve={curve_reg.item():.6f} κ̄={curve_log[-1]:.6f}")

    # measure gate statistics
    model.eval()
    with torch.no_grad():
        x_test = traj_t[:SEQ_LEN].unsqueeze(0)
        _, h_test = model(x_test, return_hidden=True)
        final_curve = mean_curvature(h_test)

    # measure A_bar (state retention per step)
    A_bar_vals = []
    with torch.no_grad():
        for layer in model.layers:
            if hasattr(layer, 'A_log'):
                A = -torch.exp(layer.A_log.float())
                # estimate mean A_bar using typical dt
                dt_typical = F.softplus(layer.dt_proj.bias).mean().item()
                A_bar_layer = torch.exp(dt_typical * A).mean().item()
                A_bar_vals.append(A_bar_layer)
    A_bar_mean = float(np.mean(A_bar_vals)) if A_bar_vals else None

    return model, traj_t, final_curve, A_bar_mean, curve_log

# ── State injection measurement ───────────────────────────────────────────────

@torch.no_grad()
def measure_tau_star(model, system, n_seeds=N_SEEDS):
    traj_raw = TRAJ_GEN[system](N_MEAS + 2000)
    traj = normalize(traj_raw[-N_MEAS:])
    traj_t = torch.tensor(traj, dtype=DTYPE, device=DEVICE)

    all_curves = []
    model.eval()
    inj_pos = 5

    for seed in range(n_seeds):
        h_clean = traj_t.unsqueeze(0)
        torch.manual_seed(seed)
        noise = torch.randn_like(h_clean[:, inj_pos:inj_pos+1, :])
        noise = noise / (noise.norm() + 1e-12)
        h_pert = h_clean.clone()
        h_pert[:, inj_pos] = h_pert[:, inj_pos] + EPS * noise.squeeze(1)

        # get model output states (not SSM hidden states, but output representations)
        with torch.no_grad():
            _, rep_clean = model(h_clean, return_hidden=True)
            _, rep_pert  = model(h_pert,  return_hidden=True)

        delta = (rep_pert - rep_clean).squeeze(0)
        delta0 = delta[inj_pos].norm().item()
        if delta0 < 1e-10:
            continue

        max_dt = N_MEAS - inj_pos - 1
        curve = [delta[inj_pos + dt].norm().item() / delta0 for dt in range(max_dt)]
        all_curves.append(curve)

    if not all_curves:
        return None, []

    min_len = min(len(c) for c in all_curves)
    mean_curve = np.mean([c[:min_len] for c in all_curves], axis=0)

    thresh = 1.0 / math.e
    tau_star = None
    for dt, val in enumerate(mean_curve):
        if val < thresh:
            if dt > 0:
                prev = mean_curve[dt - 1]
                frac = (prev - thresh) / (prev - val + 1e-12)
                tau_star = float((dt - 1) + frac)
            else:
                tau_star = 0.0
            break
    if tau_star is None:
        tau_star = float(min_len)

    return tau_star, mean_curve.tolist()

# ── Main sweep ────────────────────────────────────────────────────────────────

results = {}

for system in ["lorenz", "rossler"]:
    tau_L = TAU_L[system]
    print(f"\n{'#'*60}")
    print(f"# System: {system} (τ_L={tau_L})")
    print(f"# DHP target τ*: {DHP_LO*tau_L:.1f}–{DHP_HI*tau_L:.1f} steps")
    print(f"{'#'*60}")

    sys_results = {}
    for lam in LAMBDA_VALUES:
        t0 = time.time()
        model, traj_t, final_curve, A_bar, curve_log = train(system, lam)
        tau_star, delta_curve = measure_tau_star(model, system)

        tau_ratio = tau_star / tau_L if tau_star is not None else None
        dhp = bool(tau_ratio is not None and DHP_LO <= tau_ratio <= DHP_HI)

        elapsed = time.time() - t0
        print(f"  τ*={tau_star:.2f if tau_star else 'N/A':>8}  "
              f"τ*/τ_L={tau_ratio:.4f if tau_ratio else 'N/A':>8}  "
              f"DHP={'✓' if dhp else '✗'}  Ā={A_bar:.4f if A_bar else '?':>8}  "
              f"κ_final={final_curve:.6f}  {elapsed:.0f}s")

        sys_results[f"lambda_{lam}"] = {
            "lambda": float(lam),
            "tau_L": float(tau_L),
            "tau_star": float(tau_star) if tau_star is not None else None,
            "tau_ratio": float(tau_ratio) if tau_ratio is not None else None,
            "dhp": dhp,
            "A_bar_mean": float(A_bar) if A_bar is not None else None,
            "final_curvature": float(final_curve),
            "curvature_log": [float(c) for c in curve_log],
            "delta_curve": [float(v) for v in delta_curve[:200]],
        }

    results[system] = sys_results

results["metadata"] = {
    "arch": "Mamba + Temporal Straightening",
    "lambda_values": [float(l) for l in LAMBDA_VALUES],
    "d_model": D_MODEL, "d_state": D_STATE, "n_layers": N_LAYERS,
    "seq_len": SEQ_LEN, "n_steps": N_STEPS,
    "horizons": HORIZONS, "temp_init": TEMP_INIT, "temp_final": TEMP_FINAL,
    "note": "Curvature regularizer: L_curve = mean(||h(t+2)-2h(t+1)+h(t)||^2). "
            "Inspired by arXiv:2603.12231.",
    "device": DEVICE,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

outpath = Path(__file__).parent / "mamba_curvature_results_v1.json"
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {outpath}")

# Summary table
print("\n" + "="*72)
print("CURVATURE SWEEP SUMMARY")
print(f"{'λ':>6}  {'sys':>8}  {'τ*':>8}  {'τ*/τ_L':>8}  {'DHP':>4}  {'Ā':>8}  {'κ_final':>12}")
print("-"*72)
for system in ["lorenz", "rossler"]:
    for lam in LAMBDA_VALUES:
        r = results[system][f"lambda_{lam}"]
        print(f"{lam:>6.3f}  {system:>8}  "
              f"{r['tau_star']:>8.3f}  {r['tau_ratio']:>8.4f}  "
              f"{'✓' if r['dhp'] else '✗':>4}  "
              f"{r['A_bar_mean']:>8.4f}  "
              f"{r['final_curvature']:>12.6f}")
