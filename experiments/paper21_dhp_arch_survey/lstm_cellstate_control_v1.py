"""
LSTM Cell-State-Only Control Experiment v1
===========================================
Purpose: address Aura's methodological concern from P20/P21 red-team.

Standard injection perturbs (h_t, c_t) together. Because f_t, i_t, o_t all
depend on h_{t-1}, the perturbation alters subsequent gate values, mixing:
  (a) pure linear forget-gate decay: Π_{s} f_s^ref * δc_t0
  (b) nonlinear Jacobian contamination via the perturbed h trajectory

This control cleanly isolates (a) by:
  1. Perturbing ONLY c_t (keeping h_t = h_t^ref at injection)
  2. For all subsequent steps: freezing gate values to reference trajectory
     so c_{t+1}^pert = f_{t+1}^ref * c_t^pert + i_{t+1}^ref * g_{t+1}^ref
     and h_{t+1}^pert = o_{t+1}^ref * tanh(c_{t+1}^pert)

Expected outcomes:
  A) τ*_cellstate ≈ τ*_standard:
     Gate contamination doesn't explain DHP failure. LSTM's near-Markov
     behavior is genuine — forget gate simply doesn't adapt to τ_L.
  B) τ*_cellstate >> τ*_standard:
     Standard injection was artificially deflated. True cell-state memory
     is longer, but still potentially fails DHP.

Both outcomes are informative. (A) strengthens P20/P21 claim. (B) adds
nuance requiring discussion of measurement protocol.

Archon / DuoNeural Research / 2026-05-26
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
print("=" * 70)
print("LSTM Cell-State-Only Control — Methodological Purity Experiment")
print("=" * 70)


# ── Chaotic Systems ────────────────────────────────────────────────────────────

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


# ── Manual LSTM with explicit gate access ─────────────────────────────────────

class ManualLSTMCell(nn.Module):
    """
    Single-layer LSTM cell with explicit gate computation.
    Returns (h_t, c_t, gates) where gates = (f_t, i_t, g_t, o_t).
    This lets us freeze gate values during cell-state-only perturbation.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # All gate weight matrices in a single linear for efficiency
        # Internal gate order: (f, i, g, o) — forget first, matching forward().
        # Note: differs from PyTorch nn.LSTM order (i, f, g, o).
        self.W_ih = nn.Linear(input_dim, 4 * hidden_dim, bias=True)
        self.W_hh = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)

    def forward(self, x, h, c, frozen_gates=None):
        """
        x: (B, input_dim)
        h: (B, hidden_dim) — previous hidden state
        c: (B, hidden_dim) — previous cell state
        frozen_gates: if given, (f_t, i_t, g_t, o_t) each (B, hidden_dim)
                      overrides recomputed gates (cell-state-only mode)
        Returns: h_new, c_new, (f, i, g, o)
        """
        gates_pre = self.W_ih(x) + self.W_hh(h)
        H = self.hidden_dim

        if frozen_gates is not None:
            f, ig, g, o = frozen_gates
        else:
            f  = torch.sigmoid(gates_pre[:, 0*H:1*H])   # forget
            ig = torch.sigmoid(gates_pre[:, 1*H:2*H])   # input
            g  = torch.tanh(   gates_pre[:, 2*H:3*H])   # cell gate
            o  = torch.sigmoid(gates_pre[:, 3*H:4*H])   # output

        c_new = f * c + ig * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new, (f, ig, g, o)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class LSTMPredictor(nn.Module):
    """
    Two-layer stacked LSTM using ManualLSTMCell.
    Matches v1 architecture (input_dim=3, hidden=128, 2 layers).
    """
    def __init__(self, input_dim=3, hidden_dim=128, forget_bias=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cell1 = ManualLSTMCell(hidden_dim, hidden_dim)
        self.cell2 = ManualLSTMCell(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        if forget_bias is not None:
            # Long-memory init: bias forget gate toward 1.
            # NOTE: our ManualLSTMCell uses internal gate order (f, i, g, o) -- forget FIRST.
            # This differs from PyTorch nn.LSTM which uses (i, f, g, o).
            # So forget gate = bias[0:H], NOT bias[H:2H].
            H = hidden_dim
            with torch.no_grad():
                self.cell1.W_ih.bias.data[0:H] += forget_bias
                self.cell2.W_ih.bias.data[0:H] += forget_bias

    def forward_step(self, x, h1, c1, h2, c2):
        """
        x: (B, input_dim)
        Returns: pred, h1_new, c1_new, h2_new, c2_new, gates_l1, gates_l2
        """
        x_proj = self.input_proj(x)                          # (B, hidden)
        h1_new, c1_new, gates1 = self.cell1(x_proj, h1, c1)
        h2_new, c2_new, gates2 = self.cell2(h1_new, h2, c2)
        pred = self.output_proj(h2_new)
        return pred, h1_new, c1_new, h2_new, c2_new, gates1, gates2

    def init_state(self, batch_size):
        z = lambda: torch.zeros(batch_size, self.hidden_dim, device=DEVICE)
        return z(), z(), z(), z()  # h1, c1, h2, c2

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_forget_gate_mean(self, traj_norm, n_steps=500):
        self.eval()
        traj_t = torch.tensor(traj_norm[:n_steps], dtype=torch.float32, device=DEVICE)
        h1, c1, h2, c2 = self.init_state(1)
        forget_vals = []
        with torch.no_grad():
            for t in range(n_steps):
                x = traj_t[t].unsqueeze(0)
                _, h1, c1, h2, c2, gates1, _ = self.forward_step(x, h1, c1, h2, c2)
                f1 = gates1[0]  # forget gate, layer 1
                forget_vals.append(f1.mean().item())
        return float(np.mean(forget_vals))


# ── Training ───────────────────────────────────────────────────────────────────

def make_batches(traj_norm, seq_len=64, batch_size=32, max_horizon=16):
    traj_t = torch.tensor(traj_norm, dtype=torch.float32, device=DEVICE)
    N = len(traj_t)
    starts = torch.randint(0, N - seq_len - max_horizon, (batch_size,))
    return torch.stack([traj_t[s:s+seq_len] for s in starts])

def multi_horizon_loss(model, x_batch, horizons, temperature):
    B, T, D = x_batch.shape
    warmup = min(32, T // 2)
    h1, c1, h2, c2 = model.init_state(B)

    for t in range(warmup):
        _, h1, c1, h2, c2, _, _ = model.forward_step(x_batch[:, t], h1, c1, h2, c2)

    preds = []
    for t in range(warmup, T):
        pred, h1, c1, h2, c2, _, _ = model.forward_step(x_batch[:, t], h1, c1, h2, c2)
        preds.append(pred)
    preds = torch.stack(preds, dim=1)   # (B, T-warmup, D)

    loss, total_w = 0.0, 0.0
    max_k = max(horizons)
    for k in horizons:
        w = math.exp((k - max_k) / temperature)
        tlen = T - warmup - k
        if tlen <= 0:
            continue
        target = x_batch[:, warmup + k : warmup + k + tlen]
        loss += w * F.mse_loss(preds[:, :tlen], target)
        total_w += w
    return loss / (total_w + 1e-8)

def train(system, traj_norm, tau_L, horizons, n_steps=5000,
          temp_start=2.0, temp_end=0.1, lr=3e-4, batch_size=32,
          forget_bias=None, label=""):
    model = LSTMPredictor(input_dim=3, hidden_dim=128,
                          forget_bias=forget_bias).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps, eta_min=lr*0.1)

    tag = f"{system}/{label}"
    print(f"\n{'='*60}")
    print(f"Training: {tag}  |  τ_L={tau_L}  |  params={model.param_count():,}")
    if forget_bias is not None:
        print(f"  forget_bias={forget_bias} (long-memory init)")

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
        if step % 1000 == 0:
            print(f"  step {step:5d}/{n_steps}  loss={loss.item():.6f}  "
                  f"temp={temp:.3f}  t={time.time()-t0:.0f}s")

    print(f"  Done. total={time.time()-t0:.0f}s")
    return model


# ── Reference Trajectory Collection ───────────────────────────────────────────

def collect_reference(model, traj_norm, warmup=200, measure_steps=500):
    """
    Run a reference trajectory through the model, storing ALL state at every step.
    Returns a list of dicts: {h1, c1, h2, c2, gates1:(f,i,g,o), gates2:(f,i,g,o)}
    """
    model.eval()
    traj_t = torch.tensor(traj_norm, dtype=torch.float32, device=DEVICE)
    ref_states = []

    with torch.no_grad():
        h1, c1, h2, c2 = model.init_state(1)

        # Warmup (don't record)
        for t in range(warmup):
            x = traj_t[t].unsqueeze(0)
            _, h1, c1, h2, c2, _, _ = model.forward_step(x, h1, c1, h2, c2)

        # Measure window (record everything)
        for t in range(measure_steps + 1):
            idx = warmup + t
            if idx >= len(traj_t) - 1:
                break
            x = traj_t[idx].unsqueeze(0)
            _, h1, c1, h2, c2, gates1, gates2 = model.forward_step(x, h1, c1, h2, c2)
            ref_states.append({
                'h1': h1.clone(), 'c1': c1.clone(),
                'h2': h2.clone(), 'c2': c2.clone(),
                'gates1': tuple(g.clone() for g in gates1),
                'gates2': tuple(g.clone() for g in gates2),
            })

    return ref_states


# ── Standard Injection ─────────────────────────────────────────────────────────

def measure_tau_standard(model, traj_norm, n_seeds=6, warmup=200,
                         measure_steps=500, eps=1e-4):
    """
    Standard injection: perturb BOTH h and c at injection point.
    This is the original v1 protocol — used as comparison baseline.
    """
    model.eval()
    traj_t = torch.tensor(traj_norm, dtype=torch.float32, device=DEVICE)
    N = len(traj_t)
    all_deltas = []

    with torch.no_grad():
        for seed in range(n_seeds):
            start = seed * (N // (n_seeds + 2)) + 50

            # Warmup
            h1, c1, h2, c2 = model.init_state(1)
            for t in range(warmup):
                idx = start + t
                if idx >= N: break
                x = traj_t[idx].unsqueeze(0)
                _, h1, c1, h2, c2, _, _ = model.forward_step(x, h1, c1, h2, c2)

            # Fork: perturb both h2 and c2 (top layer, matches v1)
            h2_r = h2.clone(); c2_r = c2.clone()
            h2_p = h2.clone() + eps * torch.randn_like(h2)
            c2_p = c2.clone() + eps * torch.randn_like(c2)
            h1_r = h1.clone(); c1_r = c1.clone()
            h1_p = h1.clone(); c1_p = c1.clone()  # don't perturb layer 1

            delta_0 = (h2_p - h2_r).norm().item() + (c2_p - c2_r).norm().item()

            deltas = []
            for t in range(measure_steps):
                idx = start + warmup + t
                if idx >= N - 1: break
                x = traj_t[idx].unsqueeze(0)
                _, h1_r, c1_r, h2_r, c2_r, _, _ = model.forward_step(
                    x, h1_r, c1_r, h2_r, c2_r)
                _, h1_p, c1_p, h2_p, c2_p, _, _ = model.forward_step(
                    x, h1_p, c1_p, h2_p, c2_p)
                d = (h2_p - h2_r).norm().item() + (c2_p - c2_r).norm().item()
                deltas.append(d / (delta_0 + 1e-12))
            all_deltas.append(deltas)

    return _compute_tau_star(all_deltas)


# ── Cell-State-Only Injection ──────────────────────────────────────────────────

def measure_tau_cellstate(model, traj_norm, n_seeds=6, warmup=200,
                          measure_steps=500, eps=1e-4):
    """
    Cell-state-only injection with frozen gates.

    Protocol:
      1. Run reference trajectory, collecting ALL gate values at each step
      2. At injection: perturb ONLY c2 by eps; keep h2 = h2_ref EXACTLY
      3. For all subsequent steps t > t0:
           use gates1^ref, gates2^ref (frozen to reference trajectory)
           c_new^pert = f^ref * c^pert + i^ref * g^ref
           h_new^pert = o^ref * tanh(c_new^pert)
      4. Measure ||h2^pert - h2^ref|| decay

    This isolates ONLY the forget-gate product:
      δc_t = Π_{s=t0}^t f_s^ref * δc_t0  (pure linear decay)
    """
    model.eval()
    traj_t = torch.tensor(traj_norm, dtype=torch.float32, device=DEVICE)
    N = len(traj_t)
    all_deltas = []

    with torch.no_grad():
        for seed in range(n_seeds):
            start = seed * (N // (n_seeds + 2)) + 50

            # Warmup to injection point
            h1, c1, h2, c2 = model.init_state(1)
            for t in range(warmup):
                idx = start + t
                if idx >= N: break
                x = traj_t[idx].unsqueeze(0)
                _, h1, c1, h2, c2, _, _ = model.forward_step(x, h1, c1, h2, c2)

            # Collect reference gate trajectory for the measure window
            # We need gates at steps warmup..warmup+measure_steps
            ref_gates1 = []
            ref_gates2 = []
            ref_c1 = [c1.clone()]
            ref_c2 = [c2.clone()]
            h1_tmp, c1_tmp, h2_tmp, c2_tmp = h1.clone(), c1.clone(), h2.clone(), c2.clone()
            for t in range(measure_steps):
                idx = start + warmup + t
                if idx >= N - 1: break
                x = traj_t[idx].unsqueeze(0)
                _, h1_tmp, c1_tmp, h2_tmp, c2_tmp, g1, g2 = model.forward_step(
                    x, h1_tmp, c1_tmp, h2_tmp, c2_tmp)
                ref_gates1.append(tuple(g.clone() for g in g1))
                ref_gates2.append(tuple(g.clone() for g in g2))
                ref_c1.append(c1_tmp.clone())
                ref_c2.append(c2_tmp.clone())

            # Injection: perturb ONLY c2 at t0, keep h2 = h2_ref
            direction = torch.randn_like(c2)
            direction = direction / (direction.norm() + 1e-12)
            c2_pert = c2.clone() + eps * direction
            h2_pert = h2.clone()          # h2 is NOT perturbed — key point
            c1_pert = c1.clone()
            h1_pert = h1.clone()

            delta_0 = eps   # ||c2_pert - c2_ref|| = eps * 1.0 (unit direction)

            deltas = []
            for t in range(len(ref_gates2)):
                # Apply frozen gates from reference to PERTURBED cell state
                f2, i2, g2_gate, o2 = ref_gates2[t]
                f1, i1, g1_gate, o1 = ref_gates1[t]

                # Layer 1: perturbed trajectory uses frozen layer-1 gates
                # (layer 1 wasn't perturbed, so this is identical to reference,
                #  but we freeze anyway for methodological purity)
                c1_pert_new = f1 * c1_pert + i1 * g1_gate
                h1_pert_new = o1 * torch.tanh(c1_pert_new)

                # Layer 2: frozen gates applied to perturbed c2
                c2_pert_new = f2 * c2_pert + i2 * g2_gate
                h2_pert_new = o2 * torch.tanh(c2_pert_new)

                # Reference states from pre-collected (already stepped via ref run above)
                c2_ref_t = ref_c2[t + 1]   # ref_c2[0] is c2 at injection

                # Divergence measured in h space (most natural; same as standard)
                # h2_pert vs reference h (reconstructed with frozen gates)
                # ref h2 = o2 * tanh(c2_ref[t+1])
                h2_ref_t = o2 * torch.tanh(c2_ref_t)
                d_h = (h2_pert_new - h2_ref_t).norm().item()
                deltas.append(d_h / (delta_0 + 1e-12))

                # Also track δc for diagnostic
                c2_pert = c2_pert_new
                h2_pert = h2_pert_new
                c1_pert = c1_pert_new
                h1_pert = h1_pert_new

            all_deltas.append(deltas)

    return _compute_tau_star(all_deltas)


# ── Tau* Computation ───────────────────────────────────────────────────────────

def _compute_tau_star(all_deltas):
    """Compute τ* (1/e crossing) from list of delta curves."""
    if not all_deltas:
        return float('nan'), []
    min_len = min(len(d) for d in all_deltas)
    if min_len == 0:
        return float('nan'), []
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


# ── Main Loop ──────────────────────────────────────────────────────────────────

MULTI = [1, 2, 4, 8, 16]
N_STEPS = 5000
results = {}

configs = [
    # (system, forget_bias, config_label)
    ("lorenz",  None,  "std_init"),
    ("rossler", None,  "std_init"),
    ("lorenz",  3.0,   "long_init"),
    ("rossler", 3.0,   "long_init"),
]

for system, forget_bias, config_label in configs:
    traj_raw = lorenz_trajectory(60000) if system == "lorenz" else rossler_trajectory(60000)
    mean, std = traj_raw.mean(0), traj_raw.std(0) + 1e-8
    traj_norm = (traj_raw - mean) / std
    tau_L = TAU_L[system]

    exp_key = f"{system}_{config_label}"
    label = f"multi/{config_label}"
    print(f"\n{'#'*70}")
    print(f"# Experiment: {exp_key}")
    print(f"{'#'*70}")

    model = train(system, traj_norm, tau_L, MULTI, N_STEPS,
                  temp_start=2.0, temp_end=0.1,
                  forget_bias=forget_bias, label=label)

    f_mean = model.get_forget_gate_mean(traj_norm)
    tau_theoretical = -1.0 / math.log(f_mean + 1e-12) if f_mean < 1.0 else float('inf')

    print(f"\n  [Standard injection...]")
    tau_std, curve_std = measure_tau_standard(model, traj_norm)
    ratio_std = tau_std / tau_L

    print(f"  [Cell-state-only injection...]")
    tau_cs, curve_cs = measure_tau_cellstate(model, traj_norm)
    ratio_cs = tau_cs / tau_L

    dhp_std = 0.60 <= ratio_std <= 0.85
    dhp_cs  = 0.60 <= ratio_cs  <= 0.85

    print(f"\n  ── Results: {exp_key} ──")
    print(f"  τ_L                     = {tau_L:.1f}")
    print(f"  f̄ (forget gate)        = {f_mean:.4f}")
    print(f"  τ* theoretical          = {tau_theoretical:.2f}  (-1/log f̄)")
    print(f"  τ* standard injection   = {tau_std:.4f}  (ratio={ratio_std:.4f})  DHP={'✓' if dhp_std else '✗'}")
    print(f"  τ* cell-state-only      = {tau_cs:.4f}  (ratio={ratio_cs:.4f})  DHP={'✓' if dhp_cs else '✗'}")
    print(f"  Δτ* (CS - Std)          = {tau_cs - tau_std:+.4f}  (gate contamination effect)")

    if abs(tau_cs - tau_std) < 0.5:
        verdict = "GATE_CONTAMINATION_NEGLIGIBLE"
        interp = "Standard injection faithfully measures forget-gate decay; DHP failure is genuine."
    elif tau_cs > tau_std * 1.5:
        verdict = "GATE_CONTAMINATION_SIGNIFICANT"
        interp = "Standard injection artificially deflated τ*; cell-state memory is longer."
    else:
        verdict = "GATE_CONTAMINATION_MINOR"
        interp = "Modest gate contamination effect; DHP failure claim still holds."

    print(f"  Verdict: {verdict}")
    print(f"  Interpretation: {interp}")

    results[exp_key] = {
        "system": system,
        "config": config_label,
        "forget_bias": forget_bias,
        "tau_L": tau_L,
        "f_mean": round(f_mean, 4),
        "tau_theoretical": round(tau_theoretical, 2),
        "standard_injection": {
            "tau_star": tau_std,
            "tau_ratio": ratio_std,
            "dhp": dhp_std,
            "delta_curve": curve_std[:100],
        },
        "cellstate_only": {
            "tau_star": tau_cs,
            "tau_ratio": ratio_cs,
            "dhp": dhp_cs,
            "delta_curve": curve_cs[:100],
        },
        "delta_tau": tau_cs - tau_std,
        "verdict": verdict,
        "interpretation": interp,
    }


# ── Summary ────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 80)
print("FINAL SUMMARY — LSTM Gate Contamination Control")
print("=" * 80)
print(f"{'Experiment':<22}{'τ_L':>7}{'τ*_std':>10}{'r_std':>8}{'τ*_cs':>10}{'r_cs':>8}{'Δτ*':>8}  Verdict")
print("-" * 80)
for key, r in results.items():
    s = r['standard_injection']
    c = r['cellstate_only']
    print(f"{key:<22}{r['tau_L']:>7.0f}{s['tau_star']:>10.3f}{s['tau_ratio']:>8.4f}"
          f"{c['tau_star']:>10.3f}{c['tau_ratio']:>8.4f}{r['delta_tau']:>+8.3f}"
          f"  {r['verdict']}")

results["metadata"] = {
    "experiment": "lstm_cellstate_control_v1",
    "date": datetime.now().isoformat(),
    "model": "ManualLSTMCell 2-layer hidden=128",
    "protocol": "cell-state-only injection with frozen reference gates",
    "multi_horizons": MULTI,
    "n_training_steps": N_STEPS,
    "n_injection_seeds": 6,
    "eps": 1e-4,
    "dhp_window": [0.60, 0.85],
    "purpose": "Aura red-team methodological control — gate contamination in standard LSTM injection",
}

out_path = Path(__file__).parent / "lstm_cellstate_control_v1.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {out_path}")
print(f"Done: {datetime.now().isoformat()}")
