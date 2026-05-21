"""
tau_continuous.py — P12 Follow-Up: Continuous Horizon Evaluation
2026-05-21 — Archon / DuoNeural

P12 open question: Is the universal h=8 GHL horizon mechanistic or a grid artifact?
The original runs used H={1,2,4,8,16,32} (power-of-two). If the Lyapunov cliff falls
anywhere in (8,16), the grid deterministically selects h=8.

This run uses H={1,2,3,...,16,20,24,28,32} — continuous integer steps 1-16 plus
a sparse tail. If h=8 is mechanistic, the GHL cliff should appear between steps 8-10.
If it's a grid artifact, we'll see the cliff at a different continuous value.

Runs on D=7 (reliable, hyperchaotic) only — the one confirmed DHP system.
kilonova CPU mode is fine; the model is tiny (150K params).

Usage:
  HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' python3 tau_continuous.py
"""
import os, sys
# force CPU mode to avoid ROCm torch.stack segfault on gfx1103
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("ROCR_VISIBLE_DEVICES", "")

# patch the PRED_HORIZONS and D before importing the main training function
import importlib.util, types, pathlib

# -- load tau_trajectory.py as a module without executing __main__ --
src = pathlib.Path(__file__).parent / "tau_trajectory.py"
spec = importlib.util.spec_from_file_location("tau_trajectory", src)
tt   = importlib.util.module_from_spec(spec)

# inject overrides BEFORE exec_module so they're visible at module level
tt.PRED_HORIZONS_OVERRIDE = list(range(1, 17)) + [20, 24, 28, 32]
# will be used in train_and_measure if we patch it

import torch
import torch.nn as nn
import numpy as np
import json, time
from pathlib import Path

# replicate the relevant pieces from tau_trajectory.py inline
# (avoids the __main__ execution issue)
DEVICE = torch.device("cpu")  # kilonova CPU mode — safe, no ROCm segfault

PRED_HORIZONS = list(range(1, 17)) + [20, 24, 28, 32]  # 20 horizons
D = 7      # hyperchaotic Lorenz-96
F = 8.0
T_GATE = 16
N_TRAIN = 50_000
BATCH_SIZE = 64
N_EVAL_BATCHES = 50

print(f"=== Continuous Horizon Test — Lorenz-96 D={D} ===")
print(f"  PRED_HORIZONS = {PRED_HORIZONS}")
print(f"  Device: {DEVICE} | Steps: {N_TRAIN}")

OUT_DIR = os.environ.get("OUT_DIR", str(Path(__file__).parent / "continuous_results"))
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
OUT_FILE = f"{OUT_DIR}/tau_continuous_D{D}.json"
print(f"  Output: {OUT_FILE}")


# ─── Lorenz-96 generator ─────────────────────────────────────────────────────
def _rk4_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5*dt*k1)
    k3 = f(x + 0.5*dt*k2)
    k4 = f(x + dt*k3)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def gen_lorenz96(n_steps, D=7, F=8.0, dt=0.05, seed=42):
    np.random.seed(seed)
    x = np.random.randn(D) * 0.01
    x[0] += 0.01
    def f(v):
        return np.array([(v[(i+1)%D] - v[(i-2)%D]) * v[(i-1)%D] - v[i] + F
                         for i in range(D)], dtype=np.float64)
    for _ in range(5000): x = _rk4_step(f, x, dt)
    traj = []
    for _ in range(n_steps):
        x = _rk4_step(f, x, dt)
        traj.append(x.copy())
    return np.array(traj, dtype=np.float32)


# ─── TauCTM architecture (v40-style) ─────────────────────────────────────────
class GateAttention(nn.Module):
    def __init__(self, d, T):
        super().__init__()
        self.T = T
        self.q = nn.Linear(d, 32)
        self.k = nn.Embedding(T, 32)
        self.pos = torch.arange(T)

    def forward(self, h):
        Q = self.q(h[:, -1, :])          # (B, 32)
        K = self.k(self.pos.to(h.device))  # (T, 32)
        scores = Q @ K.T / 32**0.5         # (B, T)
        return torch.softmax(scores, dim=-1)

class TauCTM(nn.Module):
    def __init__(self, d, hidden=128, T=16, horizons=(1,2,4,8,16,32)):
        super().__init__()
        self.T = T
        self.horizons = list(horizons)
        self.rnn = nn.GRU(d, hidden, batch_first=True)
        self.gate_attn = GateAttention(hidden, T)
        self.pred_head = nn.Linear(hidden, d * len(horizons))

    def forward(self, x):                  # x: (B, T, d)
        h, _ = self.rnn(x)                 # (B, T, hidden)
        weights = self.gate_attn(h)        # (B, T)
        ctx = (weights.unsqueeze(-1) * h).sum(1)  # (B, hidden)
        raw = self.pred_head(ctx)          # (B, d*H)
        preds = raw.view(x.size(0), len(self.horizons), x.size(2))
        return preds, weights


def make_batches(data, T, horizons, bs=64, n=None):
    max_h = max(horizons)
    valid_start = len(data) - T - max_h
    n = n or 200
    idxs = np.random.randint(0, valid_start, n * bs)
    for i in range(n):
        batch_idxs = idxs[i*bs:(i+1)*bs]
        X = np.stack([data[j:j+T] for j in batch_idxs])
        Y = np.stack([[data[j+T+h-1] for h in horizons] for j in batch_idxs])
        yield torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def measure(model, data, T, horizons, n_eval=N_EVAL_BATCHES):
    model.eval()
    h_mse = {h: [] for h in horizons}
    gate_means = []
    with torch.no_grad():
        for X, Y in make_batches(data, T, horizons, bs=64, n=n_eval):
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            preds, weights = model(X)
            for hi, h in enumerate(horizons):
                mse = ((preds[:, hi] - Y[:, hi]) ** 2).mean().item()
                h_mse[h].append(mse)
            pos = torch.arange(T, dtype=torch.float32, device=weights.device)
            gate_mean = (weights * pos).sum(1).mean().item()
            gate_means.append(gate_mean)
    h_mse_mean = {h: float(np.mean(v)) for h, v in h_mse.items()}
    tau_star = float(np.mean(gate_means))

    # find the Lyapunov cliff: biggest MSE jump in consecutive horizons
    hs_sorted = sorted(horizons)
    mses_sorted = [h_mse_mean[h] for h in hs_sorted]
    # compute gradient (log-space since horizons span decades)
    log_h = np.log(hs_sorted)
    grad = np.gradient(mses_sorted, log_h)
    cliff_idx = int(np.argmax(grad))
    tau_horizon = hs_sorted[max(0, cliff_idx - 1)]  # last stable horizon before cliff

    return tau_star, tau_horizon, h_mse_mean


# ─── Main training loop ───────────────────────────────────────────────────────
data = gen_lorenz96(200_000, D=D, F=F)
split = int(0.8 * len(data))
train_data, eval_data = data[:split], data[split:]
input_dim = D

model = TauCTM(input_dim, hidden=128, T=T_GATE, horizons=PRED_HORIZONS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_TRAIN)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Model params: {n_params:,}")

checkpoints = [100, 250, 500, 1000, 2000, 3000, 5000, 7500, 10_000,
               15_000, 20_000, 30_000, 40_000, 50_000]

trajectory = []
t0 = time.time()
step = 0
gen = make_batches(train_data, T_GATE, PRED_HORIZONS, bs=BATCH_SIZE, n=N_TRAIN)

for X, Y in gen:
    step += 1
    model.train()
    X, Y = X.to(DEVICE), Y.to(DEVICE)
    optimizer.zero_grad()
    preds, _ = model(X)
    loss = sum(((preds[:, hi] - Y[:, hi])**2).mean()
               for hi, h in enumerate(PRED_HORIZONS))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if step in checkpoints:
        tau_star, tau_horizon, h_mse = measure(model, eval_data, T_GATE, PRED_HORIZONS)
        elapsed = time.time() - t0
        ratio = tau_star / T_GATE
        print(f"  step={step:6d}  τ*={tau_star:.2f}  τ*_horiz=h{tau_horizon}  "
              f"ratio={ratio:.3f}  loss={loss.item():.4f}  t={elapsed:.0f}s")
        trajectory.append({
            'step': step,
            'tau_star': round(tau_star, 4),
            'tau_horizon': tau_horizon,
            'tau_ratio': round(ratio, 4),
            'loss': round(loss.item(), 6),
            'horizon_mse': {str(h): round(v, 6) for h, v in h_mse.items()},
        })

print(f"\n=== DONE ===")
print(f"  Final: τ*={trajectory[-1]['tau_star']:.2f}  τ*_horiz=h{trajectory[-1]['tau_horizon']}")
print(f"\n  Horizon MSE profile (final checkpoint):")
final_mse = trajectory[-1]['horizon_mse']
for h in PRED_HORIZONS:
    bar = '█' * min(40, int(float(final_mse[str(h)]) / float(final_mse[str(PRED_HORIZONS[-1])]) * 40))
    print(f"    h={h:2d}: {float(final_mse[str(h)]):.4f}  {bar}")

print(f"\n  KEY QUESTION: Does the cliff appear at h=8 (artifact) or elsewhere?")
cliff_h = trajectory[-1]['tau_horizon']
if 7 <= cliff_h <= 9:
    print(f"  → h={cliff_h}: CONSISTENT with h=8 artifact OR mechanistic (inconclusive)")
elif cliff_h < 7:
    print(f"  → h={cliff_h}: CLIFF IS EARLIER — h=8 was grid artifact, true cliff at h={cliff_h}")
else:
    print(f"  → h={cliff_h}: CLIFF IS LATER at h={cliff_h} — h=8 was grid artifact")

output = {
    'system': f'lorenz96_D{D}_F{F}',
    'D': D, 'F': F, 'T_gate': T_GATE,
    'pred_horizons': PRED_HORIZONS,
    'n_train': N_TRAIN,
    'n_params': n_params,
    'trajectory': trajectory,
    'final_tau_star': trajectory[-1]['tau_star'],
    'final_tau_horizon': trajectory[-1]['tau_horizon'],
    'note': 'continuous h=1-16+sparse tail. cliff location resolves P12 open question.',
}
with open(OUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {OUT_FILE}")
