"""
P23 Curvature Comparison — Natural trajectory curvature of DHP-capable vs DHP-failed architectures.
Observational (no regularization): trains CTM-like, Mamba, LSTM on Lorenz, measures curvature.

Metrics:
  κ̄ = E[||h(t+2)-2h(t+1)+h(t)||²]          (second-difference curvature, Archon)
  GSI = displacement / arc_length             (Global Straightness Index, Aura)
  C   = E[1 - cos(Δh_t, Δh_{t+1})]          (Mean Curvature, Aura)

Hypothesis: CTM-like (per-slot) << Mamba ≈ LSTM in {κ̄, C}, >> in {GSI}

Archon + Aura / DuoNeural / 2026-05-26
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""

import torch, json, math, numpy as np
from datetime import datetime
from pathlib import Path

torch.manual_seed(42)
DEVICE = "cpu"
DTYPE  = torch.float32

# ── Lorenz Attractor ──────────────────────────────────────────────────────────
def lorenz_trajectory(n=8000, dt=0.01, sigma=10., rho=28., beta=8/3, seed=0):
    rng = np.random.default_rng(seed)
    x = np.array([1., 1., 1.]) + rng.standard_normal(3) * 0.01
    traj = np.zeros((n, 3))
    for t in range(n):
        dx = sigma * (x[1] - x[0])
        dy = x[0] * (rho - x[2]) - x[1]
        dz = x[0] * x[1] - beta * x[2]
        x = x + np.array([dx, dy, dz]) * dt
        traj[t] = x
    traj = (traj - traj.mean(0)) / (traj.std(0) + 1e-8)
    return traj

TAU_L = 110.0  # Lorenz Lyapunov time (steps, dt=0.01)
DHP_LO, DHP_HI = 0.60, 0.85

D_IN, D_HIDDEN, D_OUT = 3, 64, 3
N_STEPS = 5000
PRED_HORIZONS = [1, 2, 4, 8, 16]
T_INIT, T_FINAL = 2.0, 0.1
SEQ_LEN = 32

# ── Architecture 1: CTM-like (per-slot slot attention) ───────────────────────
class SlotAttentionCTM(torch.nn.Module):
    """Simplified CTM: N_SLOTS independent LSTM cells + learned slot attention."""
    def __init__(self, d_in=3, d_hidden=64, n_slots=8, n_pred=3):
        super().__init__()
        self.n_slots = n_slots
        self.d_slot  = d_hidden // n_slots
        # Independent LSTM for each slot
        self.slot_lstms = torch.nn.ModuleList([
            torch.nn.LSTMCell(d_in, self.d_slot) for _ in range(n_slots)
        ])
        # Slot attention: scores each slot against current input
        self.attn = torch.nn.Linear(d_in + self.d_slot, 1)
        # Output from aggregated slots
        self.out  = torch.nn.Linear(d_hidden, n_pred)

    def forward(self, x, states=None):
        """
        x: (B, T, D_in)
        returns: preds (B, T, n_pred), trajectory (B, T, D_hidden)
        """
        B, T, _ = x.shape
        if states is None:
            states = [(torch.zeros(B, self.d_slot), torch.zeros(B, self.d_slot))
                      for _ in range(self.n_slots)]
        traj = []
        preds = []
        for t in range(T):
            xt = x[:, t, :]
            slot_outs = []
            new_states = []
            for i, lstm in enumerate(self.slot_lstms):
                h, c = states[i]
                # Slot attention score from input + current slot state
                score = self.attn(torch.cat([xt, h], -1))
                h_new, c_new = lstm(xt, (h, c))
                slot_outs.append(h_new * score.sigmoid())
                new_states.append((h_new, c_new))
            states = new_states
            h_agg = torch.cat(slot_outs, -1)  # (B, D_hidden)
            traj.append(h_agg)
            preds.append(self.out(h_agg))
        traj  = torch.stack(traj,  1)  # (B, T, D_hidden)
        preds = torch.stack(preds, 1)
        return preds, traj

# ── Architecture 2: Mamba-like (SSM with scalar gate) ────────────────────────
class MambaModel(torch.nn.Module):
    def __init__(self, d_in=3, d_model=64, d_state=16, n_pred=3):
        super().__init__()
        self.d_model = d_model
        self.in_proj  = torch.nn.Linear(d_in, d_model)
        self.A_log    = torch.nn.Parameter(torch.zeros(d_model, d_state))
        self.B_proj   = torch.nn.Linear(d_in, d_state)
        self.C_proj   = torch.nn.Linear(d_state, d_model)
        self.dt_proj  = torch.nn.Linear(d_model, d_model)
        self.out      = torch.nn.Linear(d_model, n_pred)

    def forward(self, x, h=None):
        B, T, _ = x.shape
        A = -torch.exp(self.A_log.float())  # (D, S), negative
        if h is None:
            h = torch.zeros(B, self.A_log.shape[1], device=x.device, dtype=x.dtype)
        traj = []
        preds = []
        for t in range(T):
            xt   = self.in_proj(x[:, t, :])   # (B, D)
            dt   = torch.nn.functional.softplus(self.dt_proj(xt))  # (B, D)
            B_t  = self.B_proj(x[:, t, :])           # (B, S)
            # Simplified scalar gate: collapse dt and A to per-sample scalar
            A_bar = torch.exp(dt.mean(-1, keepdim=True) * A.mean())  # (B, 1)
            h = A_bar * h + B_t  # (B, 1) * (B, S) → (B, S)
            y = self.C_proj(h)          # (B, D)
            y = y + xt                  # residual
            traj.append(y)
            preds.append(self.out(y))
        traj  = torch.stack(traj,  1)
        preds = torch.stack(preds, 1)
        return preds, traj

# ── Architecture 3: LSTM ─────────────────────────────────────────────────────
class LSTMModel(torch.nn.Module):
    def __init__(self, d_in=3, d_hidden=64, n_pred=3):
        super().__init__()
        self.lstm = torch.nn.LSTMCell(d_in, d_hidden)
        self.out  = torch.nn.Linear(d_hidden, n_pred)

    def forward(self, x, state=None):
        B, T, _ = x.shape
        if state is None:
            h = torch.zeros(B, self.lstm.hidden_size)
            c = torch.zeros(B, self.lstm.hidden_size)
        else:
            h, c = state
        traj = []
        preds = []
        for t in range(T):
            h, c = self.lstm(x[:, t, :], (h, c))
            traj.append(h)
            preds.append(self.out(h))
        traj  = torch.stack(traj,  1)
        preds = torch.stack(preds, 1)
        return preds, traj

# ── Training ─────────────────────────────────────────────────────────────────
def multi_horizon_loss(model, batch, horizons, temp):
    B, T, _ = batch.shape
    max_h = max(horizons)
    if T <= max_h:
        return torch.tensor(0.0), 0.0
    x_in = batch[:, :-max_h, :]
    preds, _ = model(x_in)
    losses = []
    for k in horizons:
        tgt = batch[:, k:T-max_h+k, :]
        losses.append(torch.nn.functional.mse_loss(preds[:, :T-max_h, :], tgt))
    weights = torch.softmax(torch.tensor([l.item() for l in losses]) / temp, 0)
    return sum(w * l for w, l in zip(weights, losses)), sum(l.item() for l in losses)/len(losses)

def train_model(model, traj_data, n_steps=5000, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(traj_data)
    losses = []
    for step in range(1, n_steps + 1):
        temp = T_INIT * (T_FINAL / T_INIT) ** (step / n_steps)
        start = torch.randint(0, n - SEQ_LEN - max(PRED_HORIZONS) - 1, (8,))
        seqs  = torch.stack([torch.tensor(traj_data[s:s+SEQ_LEN+max(PRED_HORIZONS)], dtype=DTYPE)
                             for s in start])
        loss, plain = multi_horizon_loss(model, seqs, PRED_HORIZONS, temp)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(plain)
        if step % 1000 == 0:
            print(f"  step={step:5d} loss={plain:.6f}", flush=True)
    return losses

# ── Curvature Measurement ─────────────────────────────────────────────────────
def measure_trajectory_curvature(model, traj_data, n_samples=200):
    """
    Measures three curvature metrics on the trained model's trajectory:
    - kap: mean ||h(t+2) - 2h(t+1) + h(t)||² (Archon κ̄)
    - C:   mean (1 - cos(Δh_t, Δh_{t+1}))     (Aura Mean Curvature)
    - GSI: mean(displacement / arc_length)      (Aura Global Straightness Index)
    """
    model.eval()
    n = len(traj_data)
    eval_len = 64
    kap_vals, C_vals, gsi_vals = [], [], []

    with torch.no_grad():
        for _ in range(n_samples):
            start = np.random.randint(0, n - eval_len - 1)
            x = torch.tensor(traj_data[start:start+eval_len], dtype=DTYPE).unsqueeze(0)
            _, traj = model(x)  # (1, T, D)
            h = traj.squeeze(0)  # (T, D)

            # κ̄: second difference
            d2h = h[2:] - 2*h[1:-1] + h[:-2]
            kap = (d2h ** 2).sum(-1).mean().item()
            kap_vals.append(kap)

            # Step vectors Δh
            dh = h[1:] - h[:-1]  # (T-1, D)
            norms = dh.norm(dim=-1, keepdim=True) + 1e-8

            # Aura Mean Curvature: 1 - cos(Δh_t, Δh_{t+1})
            dh_unit = dh / norms
            cos_seq = (dh_unit[:-1] * dh_unit[1:]).sum(-1)
            C = (1.0 - cos_seq).mean().item()
            C_vals.append(C)

            # Aura GSI: displacement / arc_length
            displacement = (h[-1] - h[0]).norm().item()
            arc_length   = norms.squeeze().sum().item()
            gsi_vals.append(displacement / (arc_length + 1e-8))

    return {
        "kap_mean": float(np.mean(kap_vals)),
        "kap_std":  float(np.std(kap_vals)),
        "C_mean":   float(np.mean(C_vals)),
        "C_std":    float(np.std(C_vals)),
        "GSI_mean": float(np.mean(gsi_vals)),
        "GSI_std":  float(np.std(gsi_vals)),
    }

# ── State injection τ* measurement ───────────────────────────────────────────
def measure_tau_star(model, traj_data, n_seeds=6, eps=1e-4):
    model.eval()
    n = len(traj_data)
    n_meas = 200
    inj_pos = 5
    all_curves = []
    with torch.no_grad():
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            start = np.random.randint(0, n - n_meas - 1)
            x = torch.tensor(traj_data[start:start+n_meas], dtype=DTYPE).unsqueeze(0)
            noise = torch.randn_like(x[:, inj_pos:inj_pos+1, :])
            noise = noise / (noise.norm() + 1e-12)
            x_pert = x.clone()
            x_pert[:, inj_pos] = x_pert[:, inj_pos] + eps * noise.squeeze(1)
            _, traj_clean = model(x)
            _, traj_pert  = model(x_pert)
            delta = (traj_pert - traj_clean).squeeze(0)
            d0 = delta[inj_pos].norm().item()
            if d0 < 1e-10:
                continue
            curve = [delta[inj_pos + dt].norm().item() / d0
                     for dt in range(n_meas - inj_pos - 1)]
            all_curves.append(curve)

    if not all_curves:
        return None, None
    min_len = min(len(c) for c in all_curves)
    mean_curve = [np.mean([c[t] for c in all_curves]) for t in range(min_len)]
    tau_star = next((t for t, v in enumerate(mean_curve) if v <= 1/math.e), None)
    return tau_star, mean_curve

# ── Main Experiment ───────────────────────────────────────────────────────────
print(f"Time: {datetime.now().isoformat()}", flush=True)
print("Generating Lorenz trajectory...", flush=True)
traj = lorenz_trajectory(n=8000)

MODELS = {
    "CTM-like (per-slot)": SlotAttentionCTM(D_IN, D_HIDDEN, n_slots=8, n_pred=D_OUT),
    "Mamba-like (scalar gate)": MambaModel(D_IN, D_HIDDEN, n_pred=D_OUT),
    "LSTM (forget gate)": LSTMModel(D_IN, D_HIDDEN, n_pred=D_OUT),
}

results = {}
for name, model in MODELS.items():
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}", flush=True)
    print(f"Training: {name} ({n_params:,} params)", flush=True)
    print(f"{'='*60}", flush=True)
    train_model(model, traj, n_steps=N_STEPS)

    print(f"  Measuring τ*...", flush=True)
    tau_star, _ = measure_tau_star(model, traj)
    tau_ratio = tau_star / TAU_L if tau_star is not None else None
    dhp = (tau_ratio is not None and DHP_LO <= tau_ratio <= DHP_HI)

    print(f"  Measuring trajectory curvature...", flush=True)
    curv = measure_trajectory_curvature(model, traj)

    tr_str = f"{tau_ratio:.4f}" if tau_ratio is not None else "N/A"
    print(f"\n  Results for {name}:", flush=True)
    print(f"    τ*       = {tau_star} (τ*/τ_L = {tr_str})", flush=True)
    print(f"    DHP      = {dhp}", flush=True)
    print(f"    κ̄       = {curv['kap_mean']:.6f} ± {curv['kap_std']:.6f}", flush=True)
    print(f"    C (Aura) = {curv['C_mean']:.4f} ± {curv['C_std']:.4f}", flush=True)
    print(f"    GSI      = {curv['GSI_mean']:.4f} ± {curv['GSI_std']:.4f}", flush=True)

    results[name] = {
        "tau_star": tau_star,
        "tau_ratio": tau_ratio,
        "dhp": dhp,
        **curv,
    }

# Summary table
print("\n\n" + "="*72, flush=True)
print("CURVATURE COMPARISON SUMMARY", flush=True)
print("="*72, flush=True)
print(f"{'Architecture':<28} {'τ*/τ_L':>8} {'DHP':>5} {'κ̄':>10} {'C(Aura)':>10} {'GSI':>8}", flush=True)
print("-"*72, flush=True)
for name, r in results.items():
    tr = f"{r['tau_ratio']:.3f}" if r['tau_ratio'] else "N/A"
    print(f"  {name:<26} {tr:>8} {str(r['dhp']):>5} {r['kap_mean']:>10.6f} {r['C_mean']:>10.4f} {r['GSI_mean']:>8.4f}", flush=True)

out = Path("curvature_comparison_v1.json")
with open(out, "w") as f:
    json.dump({"date": datetime.now().isoformat(), "results": results,
               "tau_L": TAU_L, "n_steps": N_STEPS}, f, indent=2)
print(f"\nSaved to {out}", flush=True)
print(f"Time: {datetime.now().isoformat()}", flush=True)
