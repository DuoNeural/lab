"""
Transformer DHP Baseline v1
============================
Tests whether a causal Transformer achieves DHP under multi-horizon training.

Protocol mirrors Mamba/LSTM experiments:
  - Systems: Lorenz (τ_L=110), Rössler (τ_L=700)
  - Training: H={1,2,4,8,16}, temp annealing 2.0→0.1, 5000 steps
  - Measurement: input-perturbation state injection, 6 seeds, N_meas steps

Key difference from RNNs: state injection perturbs input at position τ
and tracks how the divergence in residual-stream representations decays
across future positions (t > τ). This is the Transformer analog of
hidden-state injection for RNNs.

Usage: python transformer_dhp_v1.py
Output: transformer_dhp_results_v1.json (same schema as mamba/lstm results)

DuoNeural / Archon / 2026-05-26
"""

import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32
SEED   = 42

# Architecture
D_MODEL   = 128
N_HEADS   = 4
N_LAYERS  = 4
D_FF      = 256
DROPOUT   = 0.0

# Training
SEQ_LEN   = 96          # training context window
BATCH     = 64
LR        = 1e-3
N_STEPS   = 5000
HORIZONS  = [1, 2, 4, 8, 16]
TEMP_INIT = 2.0
TEMP_FINAL = 0.1

# Measurement
N_SEEDS   = 6
EPS       = 1e-4
INJ_POS   = 20          # inject perturbation at this position in measurement sequence
# Measurement lengths — must cover DHP target: τ*_DHP = 0.72 * τ_L
# Lorenz: 0.72*110=79, use 200 (comfortably covers decay)
# Rössler: 0.72*700=504, use 620 (covers ~0.89*τ_L decay window)
MEAS_LEN  = {"lorenz": 200, "rossler": 650}
TAU_L     = {"lorenz": 110.0, "rossler": 700.0}
DHP_LO, DHP_HI = 0.60, 0.85

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
    return (traj - mu) / std, mu, std

def make_batch(traj, seq_len, batch_size, max_horizon, rng):
    """Sample random sub-sequences for training."""
    T = len(traj) - seq_len - max_horizon
    starts = rng.integers(0, T, batch_size)
    inp = np.stack([traj[s:s+seq_len] for s in starts])
    tgts = {k: np.stack([traj[s+k:s+seq_len+k] for s in starts]) for k in HORIZONS}
    return inp, tgts

# ── Model ─────────────────────────────────────────────────────────────────────

class SinePosEnc(nn.Module):
    """Fixed sinusoidal position encoding — no learnable params, extrapolates."""
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CausalTransformer(nn.Module):
    def __init__(self, in_dim=3, d_model=D_MODEL, nhead=N_HEADS,
                 num_layers=N_LAYERS, d_ff=D_FF, out_dim=3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_enc    = SinePosEnc(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=DROPOUT, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        # Per-horizon prediction heads
        self.heads = nn.ModuleDict({str(k): nn.Linear(d_model, out_dim) for k in HORIZONS})

    def _causal_mask(self, seq_len, device):
        """Upper-triangular (True = masked out) causal attention mask."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x):
        """x: (B, T, 3) → dict {k: (B, T, 3)} for each horizon k."""
        h = self.input_proj(x)
        h = self.pos_enc(h)
        mask = self._causal_mask(x.size(1), x.device)
        h = self.transformer(h, mask=mask)
        return {k: head(h) for k, head in self.heads.items()}, h

    def get_representations(self, x):
        """Returns residual-stream repr h: (B, T, D) — used for state injection measurement."""
        h = self.input_proj(x)
        h = self.pos_enc(h)
        mask = self._causal_mask(x.size(1), x.device)
        return self.transformer(h, mask=mask)

# ── Training ──────────────────────────────────────────────────────────────────

def horizon_weights(temperature):
    """Soft weight over horizons using Boltzmann weighting by -log(k)."""
    logits = torch.tensor([-math.log(k) / temperature for k in HORIZONS])
    return F.softmax(logits, dim=0)

def train(system):
    print(f"\n{'='*60}\nTraining Transformer on {system} | device={DEVICE}")
    rng = np.random.default_rng(SEED)

    # Generate long trajectory
    traj_raw = TRAJ_GEN[system](50000 + max(HORIZONS))
    traj, mu, std = normalize(traj_raw)
    traj_t = torch.tensor(traj, dtype=DTYPE, device=DEVICE)

    model = CausalTransformer().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_STEPS, eta_min=LR*0.1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        progress = (step - 1) / N_STEPS
        temp = TEMP_INIT * (TEMP_FINAL / TEMP_INIT) ** progress
        weights = horizon_weights(temp)

        inp, tgts = make_batch(traj, SEQ_LEN, BATCH, max(HORIZONS), rng)
        x  = torch.tensor(inp, dtype=DTYPE, device=DEVICE)
        ys = {k: torch.tensor(tgts[k], dtype=DTYPE, device=DEVICE) for k in HORIZONS}

        preds, _ = model(x)
        loss = sum(weights[i] * F.mse_loss(preds[str(k)], ys[k])
                   for i, k in enumerate(HORIZONS))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 500 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"  step={step:5d}  loss={loss.item():.4f}  temp={temp:.3f}  {elapsed:.0f}s")

    return model, traj_t, mu, std

# ── State injection measurement ───────────────────────────────────────────────

@torch.no_grad()
def measure_tau_star(model, system, traj_t, n_seeds=N_SEEDS):
    """
    Inject perturbation at input position INJ_POS in a measurement sequence.
    Track ||h_t - h'_t||₂ / ||h_{INJ_POS} - h'_{INJ_POS}||₂ for t > INJ_POS.
    τ* = first Δt where δ(Δt) < 1/e.
    """
    meas_len = MEAS_LEN[system]
    traj_raw  = TRAJ_GEN[system](meas_len + 2000)
    traj_meas, mu, std = normalize(traj_raw[-meas_len:])
    traj_m = torch.tensor(traj_meas, dtype=DTYPE, device=DEVICE)

    all_curves = []
    model.eval()

    for seed in range(n_seeds):
        # Perturb input at INJ_POS
        x_clean = traj_m.unsqueeze(0)  # (1, T, 3)
        noise   = torch.randn(1, 1, 3, device=DEVICE, generator=torch.Generator(DEVICE).manual_seed(seed))
        noise   = noise / (noise.norm() + 1e-12)

        x_pert  = x_clean.clone()
        x_pert[:, INJ_POS] = x_pert[:, INJ_POS] + EPS * noise.squeeze(1)

        h_clean = model.get_representations(x_clean)  # (1, T, D)
        h_pert  = model.get_representations(x_pert)   # (1, T, D)

        delta_all = (h_pert - h_clean).squeeze(0)     # (T, D)
        delta0    = delta_all[INJ_POS].norm().item()

        if delta0 < 1e-10:
            print(f"  [!] seed {seed}: delta0 too small, skipping")
            continue

        # δ(Δt) for Δt = 0, 1, 2, ...
        max_dt = meas_len - INJ_POS - 1
        curve  = [(delta_all[INJ_POS + dt].norm().item() / delta0) for dt in range(max_dt)]
        all_curves.append(curve)

    if not all_curves:
        return None, None, []

    min_len = min(len(c) for c in all_curves)
    mean_curve = np.mean([c[:min_len] for c in all_curves], axis=0)

    # τ* = first Δt where mean_curve < 1/e
    thresh = 1.0 / math.e
    tau_star = None
    for dt, val in enumerate(mean_curve):
        if val < thresh:
            # Linear interpolation for sub-step precision
            if dt > 0:
                prev = mean_curve[dt - 1]
                frac = (prev - thresh) / (prev - val)
                tau_star = (dt - 1) + frac
            else:
                tau_star = 0.0
            break

    if tau_star is None:
        tau_star = float(min_len)  # didn't cross — report ceiling

    return tau_star, mean_curve.tolist(), all_curves

# ── Main ──────────────────────────────────────────────────────────────────────

results = {}

for system in ["lorenz", "rossler"]:
    print(f"\n{'#'*60}")
    print(f"# System: {system}")
    print(f"# τ_L = {TAU_L[system]}, DHP target τ*/τ_L ∈ [{DHP_LO}, {DHP_HI}]")
    print(f"# DHP τ* target: {DHP_LO*TAU_L[system]:.1f} – {DHP_HI*TAU_L[system]:.1f} steps")
    print(f"{'#'*60}")

    model, traj_t, mu, std = train(system)

    print(f"\nMeasuring τ* for {system}...")
    tau_star, delta_curve, all_curves = measure_tau_star(model, system, traj_t)

    tau_L     = TAU_L[system]
    tau_ratio = tau_star / tau_L if tau_star is not None else None
    dhp       = (tau_ratio is not None and DHP_LO <= tau_ratio <= DHP_HI)

    print(f"  τ* = {tau_star:.2f} steps" if tau_star else "  τ* = None (no crossing)")
    print(f"  τ*/τ_L = {tau_ratio:.4f}" if tau_ratio else "  τ*/τ_L = N/A")
    print(f"  DHP: {'✓' if dhp else '✗'}")

    results[f"{system}_multi"] = {
        "system":     system,
        "label":      "multi-horizon (DHP condition)",
        "tau_L":      tau_L,
        "tau_star":   tau_star,
        "tau_ratio":  tau_ratio,
        "dhp":        dhp,
        "delta_curve": delta_curve if delta_curve else [],
        "meas_len":   MEAS_LEN[system],
        "inj_pos":    INJ_POS,
        "n_seeds":    N_SEEDS,
    }

results["metadata"] = {
    "arch":       "Transformer",
    "d_model":    D_MODEL,
    "n_heads":    N_HEADS,
    "n_layers":   N_LAYERS,
    "d_ff":       D_FF,
    "seq_len":    SEQ_LEN,
    "n_steps":    N_STEPS,
    "horizons":   HORIZONS,
    "temp_init":  TEMP_INIT,
    "temp_final": TEMP_FINAL,
    "device":     DEVICE,
    "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

def _json_safe(obj):
    """Recursively convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, 'item'):     # numpy scalar (float32, bool_, int64, ...)
        return obj.item()
    if isinstance(obj, (bool,)):
        return bool(obj)
    return obj

outpath = Path(__file__).parent / "transformer_dhp_results_v1.json"
with open(outpath, "w") as f:
    json.dump(_json_safe(results), f, indent=2)
print(f"\nResults saved to {outpath}")

# Summary
print("\n" + "="*60)
print("TRANSFORMER DHP SURVEY SUMMARY")
print("="*60)
for system in ["lorenz", "rossler"]:
    key = f"{system}_multi"
    if key in results:
        r = results[key]
        print(f"  {system:10s}: τ*={r['tau_star']:.2f if r['tau_star'] else 'N/A':>8}  "
              f"τ*/τ_L={r['tau_ratio']:.4f if r['tau_ratio'] else 'N/A':>8}  "
              f"DHP={'✓' if r['dhp'] else '✗'}")
