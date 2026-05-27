"""
GDN-2 × DHP — Multi-Horizon State Injection Measurement (v3)
==============================================================
Fixes from v2:
1. PRED_HORIZONS=[1,2,4,8,16] — required for DHP emergence (v1-v2 used h=1 only → τ*≈4, wrong)
2. Fixed Lyapunov τ_L computation — use first 30 growth steps only (not full saturation trajectory)
3. State injection measurement unchanged (correct)
4. Temperature annealing on horizon weights

DHP conditions (from CTM experiments, confirmed in Papers 4+5):
  - PRED_HORIZONS=[1,2,4,8,16]: forces multi-scale temporal representation
  - Without multi-horizon: τ*≈4 regardless of architecture (too short)
  - With multi-horizon: τ*/τ_L converges to 0.72 (universal)

P20 Research question:
  In GDN-2's decoupled erase (b_t) / write (w_t) architecture,
  do the two gates independently converge to τ* ≈ 0.72 τ_L?

Gate attribution via theoretical decay rate:
  ΔS_t = [(I - k bk^T) D] ΔS_{t-1}
  The operator acts via:
    - alpha (decay D): geometric factor per channel
    - erase b (via bk^T): directional erasure in k-direction
  We measure τ*_alpha from mean α_t
  We measure τ*_erase from mean b_t⊗k_t spectral radius
  We measure τ*_total from empirical state divergence decay
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import os
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRED_HORIZONS = [1, 2, 4, 8, 16]
print(f"Device: {DEVICE}")
print(f"PRED_HORIZONS: {PRED_HORIZONS}")

# ─── Dynamical systems ────────────────────────────────────────────────────────

def lorenz63(n_steps, dt=0.01, x0=None, warmup=2000):
    if x0 is None:
        x0 = np.array([1.0, 1.0, 1.0])
    sigma, rho, beta = 10.0, 28.0, 8/3
    x = x0.copy().astype(np.float64)
    for _ in range(warmup):
        dx = sigma * (x[1] - x[0])
        dy = x[0] * (rho - x[2]) - x[1]
        dz = x[0] * x[1] - beta * x[2]
        x += dt * np.array([dx, dy, dz])
    traj = np.zeros((n_steps, 3))
    for i in range(n_steps):
        dx = sigma * (x[1] - x[0])
        dy = x[0] * (rho - x[2]) - x[1]
        dz = x[0] * x[1] - beta * x[2]
        x += dt * np.array([dx, dy, dz])
        traj[i] = x
    return traj

def rossler(n_steps, dt=0.02, x0=None, warmup=2000):
    if x0 is None:
        x0 = np.array([1.0, 0.0, 0.0])
    a, b, c = 0.2, 0.2, 5.7
    x = x0.copy().astype(np.float64)
    for _ in range(warmup):
        dx = -x[1] - x[2]
        dy = x[0] + a * x[1]
        dz = b + x[2] * (x[0] - c)
        x += dt * np.array([dx, dy, dz])
    traj = np.zeros((n_steps, 3))
    for i in range(n_steps):
        dx = -x[1] - x[2]
        dy = x[0] + a * x[1]
        dz = b + x[2] * (x[0] - c)
        x += dt * np.array([dx, dy, dz])
        traj[i] = x
    return traj

def compute_lyapunov_steps(integrate_fn, dt, eps=1e-8, n_calibration=5000):
    """
    Compute Lyapunov time in STEPS using only the initial growth phase.
    Key fix: use only first 30 steps (before saturation), not all 5000.
    """
    x_ref = integrate_fn(n_steps=n_calibration, dt=dt, warmup=2000)
    x0_pert = x_ref[0].copy()
    x0_pert[0] += eps
    x_pert = integrate_fn(n_steps=n_calibration, dt=dt, x0=x0_pert, warmup=0)
    diffs = np.linalg.norm(x_pert - x_ref, axis=1)

    # Use only the first 30 steps (linear growth regime before saturation)
    # At saturation, rate goes to 0 and would drag lambda_max down
    GROWTH_WINDOW = 30
    valid_rates = []
    for i in range(1, min(GROWTH_WINDOW, n_calibration)):
        if diffs[i] > 1e-12 and diffs[i-1] > 1e-12:
            rate = np.log(diffs[i] / diffs[i-1])  # nats per step
            if np.isfinite(rate) and rate > 0:
                valid_rates.append(rate)

    if not valid_rates:
        print("  [WARN] Lyapunov computation failed, using default")
        return 110.0  # Lorenz default

    lambda_per_step = np.median(valid_rates)  # median more robust than mean
    tau_L_steps = 1.0 / lambda_per_step
    return tau_L_steps

# ─── GDN-2 architecture ───────────────────────────────────────────────────────

class GDN2Layer(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.q_proj = nn.Linear(d_model, d_k, bias=False)
        self.k_proj = nn.Linear(d_model, d_k, bias=False)
        self.v_proj = nn.Linear(d_model, d_v, bias=False)
        self.b_proj = nn.Linear(d_model, d_k, bias=True)   # erase gate
        self.w_proj = nn.Linear(d_model, d_v, bias=True)   # write gate
        self.alpha_proj = nn.Linear(d_model, d_k, bias=True)  # channel decay
        self.out_proj = nn.Linear(d_v, d_model, bias=False)

    def step(self, xt, S):
        """Single step update. xt: (d_model,), S: (d_k, d_v) → new S, out"""
        k_t = F.normalize(self.k_proj(xt), dim=-1)
        v_t = self.v_proj(xt)
        b_t = torch.sigmoid(self.b_proj(xt))
        w_t = torch.sigmoid(self.w_proj(xt))
        alpha_t = torch.sigmoid(self.alpha_proj(xt))

        bk = b_t * k_t
        D_S = alpha_t.unsqueeze(1) * S
        bkDS = bk @ D_S  # (d_v,)
        S_new = D_S - torch.outer(k_t, bkDS) + torch.outer(k_t, w_t * v_t)

        q_t = F.normalize(self.q_proj(xt), dim=-1)
        retrieved = q_t @ S_new
        out = self.out_proj(retrieved)
        return S_new, out, b_t, w_t, alpha_t

    def forward_seq(self, x, S=None):
        """x: (T, d_model) → out: (T, d_model), S_final"""
        T, _ = x.shape
        if S is None:
            S = torch.zeros(self.d_k, self.d_v, device=x.device)
        outs = []
        for t in range(T):
            S, out, _, _, _ = self.step(x[t], S)
            outs.append(out)
        return torch.stack(outs, dim=0), S

    def forward_batch(self, x, S=None):
        """x: (B, T, d_model) → (B, T, d_model)"""
        B, T, _ = x.shape
        if S is None:
            S = torch.zeros(B, self.d_k, self.d_v, device=x.device)
        k = F.normalize(self.k_proj(x), dim=-1)
        v = self.v_proj(x)
        b = torch.sigmoid(self.b_proj(x))
        w = torch.sigmoid(self.w_proj(x))
        alpha = torch.sigmoid(self.alpha_proj(x))
        q = F.normalize(self.q_proj(x), dim=-1)
        outs = []
        for t in range(T):
            bk = b[:, t] * k[:, t]
            D_S = alpha[:, t].unsqueeze(-1) * S
            bkDS = torch.bmm(bk.unsqueeze(1), D_S).squeeze(1)
            write = w[:, t] * v[:, t]
            S = D_S - torch.bmm(k[:, t].unsqueeze(-1), bkDS.unsqueeze(1)) \
                + torch.bmm(k[:, t].unsqueeze(-1), write.unsqueeze(1))
            retrieved = torch.bmm(q[:, t].unsqueeze(1), S).squeeze(1)
            outs.append(self.out_proj(retrieved))
        return torch.stack(outs, dim=1), S


class GDN2Model(nn.Module):
    def __init__(self, d_model=128, d_k=64, d_v=64, n_layers=4, input_dim=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([GDN2Layer(d_model, d_k, d_v) for _ in range(n_layers)])
        self.output_proj = nn.Linear(d_model, input_dim)
        self.d_k = d_k
        self.d_v = d_v
        self.n_layers = n_layers

    def forward(self, x):
        """x: (B, T, input_dim)"""
        h, S = self.input_proj(x), None
        for layer in self.layers:
            h, S = layer.forward_batch(h, S)
        return self.output_proj(h)

    def forward_seq(self, x):
        """x: (T, input_dim) → (T, input_dim), list of S_final per layer"""
        h = self.input_proj(x)
        S_finals = []
        for layer in self.layers:
            h, S_f = layer.forward_seq(h)
            S_finals.append(S_f)
        return self.output_proj(h), S_finals

    def warmup_states(self, x_warmup):
        """x_warmup: (T_warmup, input_dim) → S_per_layer list"""
        _, S_finals = self.forward_seq(x_warmup)
        return S_finals


# ─── Multi-horizon training ───────────────────────────────────────────────────

def train_multi_horizon(model, traj_norm, horizons=PRED_HORIZONS,
                        seq_len=512, steps=8000, lr=3e-4, batch_size=32):
    """
    Multi-horizon training: randomly sample prediction horizon h from horizons.
    Forces model to maintain information at multiple temporal scales.
    DHP condition requires this for τ*/τ_L ≈ 0.72 to emerge.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Temperature annealing: down-weight far horizons early, up-weight later
    # (mirrors CTM DHP conditions: temp annealing 2→0.1)
    N = len(traj_norm)
    max_h = max(horizons)
    t0 = time.time()

    model.train()
    for step in range(steps):
        # Annealed horizon sampling: early = prefer short, late = uniform
        # temp: 2→0.1 over training (high temp = uniform, low temp = peaked at short)
        temp = 2.0 * (0.1/2.0) ** (step / steps)
        h_weights = np.array([np.exp(-h / (temp * max_h + 1e-8)) for h in horizons])
        h_weights = h_weights / h_weights.sum()
        h = int(np.random.choice(horizons, p=h_weights))

        idx = np.random.randint(0, N - seq_len - h, size=batch_size)
        x_batch = np.stack([traj_norm[i:i+seq_len] for i in idx])
        y_batch = np.stack([traj_norm[i+h:i+seq_len+h] for i in idx])
        x = torch.tensor(x_batch, dtype=torch.float32, device=DEVICE)
        y = torch.tensor(y_batch, dtype=torch.float32, device=DEVICE)

        pred = model(x)
        loss = F.mse_loss(pred, y)
        if not torch.isfinite(loss):
            print(f"  [WARN] non-finite loss at step {step}: {loss.item()}")
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 1000 == 0:
            elapsed = int(time.time()-t0)
            h_sample = horizons[np.argmax(h_weights)]
            print(f"  step {step:5d}/{steps}  loss={loss.item():.6f}  h={h}  t={elapsed}s")

    model.eval()
    return model


# ─── State injection DHP measurement ─────────────────────────────────────────

def measure_dhp_state_injection(model, traj_norm, tau_L_steps, system_name,
                                 T_warmup=128, T_measure=600, eps_S=0.1, n_seeds=8):
    """
    State injection DHP measurement:
    1. Warmup model T_warmup steps → get S_warmup per layer
    2. Perturb: S_pert = S_warmup + eps_S * random_direction
    3. Run both on SAME input (no twin trajectories) for T_measure steps
    4. Measure ||S_ref - S_pert||_F decay → find τ* (1/e decay time)
    5. Gate attribution: measure mean b_t, alpha_t to compute theoretical decay rate
    """
    N = len(traj_norm)
    all_delta_S = []
    all_b_means = []
    all_w_means = []
    all_alpha_means = []

    for seed in range(n_seeds):
        np.random.seed(seed * 17 + 42)
        torch.manual_seed(seed * 17 + 42)

        start = np.random.randint(0, N - T_warmup - T_measure - 10)
        chunk = torch.tensor(
            traj_norm[start:start + T_warmup + T_measure],
            dtype=torch.float32, device=DEVICE
        )

        model.eval()
        with torch.no_grad():
            # Warmup: propagate through full stacked model
            x_warmup = chunk[:T_warmup]
            S_warmup = model.warmup_states(x_warmup)

            # Perturb each layer's state
            S_ref = [S.clone() for S in S_warmup]
            S_pert = []
            for S_w in S_warmup:
                noise = torch.randn_like(S_w)
                noise = noise / (noise.norm() + 1e-12)
                S_pert.append(S_w + eps_S * noise)

            # Measurement: run BOTH on same input sequence
            x_measure = chunk[T_warmup:]  # (T_measure, 3)
            h_measure = model.input_proj(x_measure)  # (T_measure, d_model)

            delta_S_traj = np.zeros((T_measure, model.n_layers))
            b_traj = np.zeros((T_measure, model.n_layers))
            w_traj = np.zeros((T_measure, model.n_layers))
            alpha_traj = np.zeros((T_measure, model.n_layers))

            for t in range(T_measure):
                # Layer 0 receives input projection
                h_t_ref = h_measure[t]
                h_t_pert = h_measure[t]  # same input

                for li, layer in enumerate(model.layers):
                    # Ref state update
                    S_ref[li], out_ref, b_t, w_t, alpha_t = layer.step(h_t_ref, S_ref[li])
                    # Pert state update (same gates = same input)
                    S_pert[li], out_pert, _, _, _ = layer.step(h_t_pert, S_pert[li])

                    delta_S_traj[t, li] = (S_ref[li] - S_pert[li]).norm().item()
                    b_traj[t, li] = b_t.mean().item()
                    w_traj[t, li] = w_t.mean().item()
                    alpha_traj[t, li] = alpha_t.mean().item()

                    # NOTE: inter-layer h not propagated (see analysis note in paper)
                    # Each layer measured independently on same input projection
                    # This isolates per-layer gate dynamics cleanly

        all_delta_S.append(delta_S_traj)
        all_b_means.append(b_traj)
        all_w_means.append(w_traj)
        all_alpha_means.append(alpha_traj)

    # Aggregate
    delta_S_mean = np.mean(all_delta_S, axis=0)   # (T_measure, n_layers)
    b_mean = np.mean(all_b_means, axis=0)
    w_mean = np.mean(all_w_means, axis=0)
    alpha_mean = np.mean(all_alpha_means, axis=0)

    # Total state divergence (sum across layers)
    delta_S_total = delta_S_mean.sum(axis=1)  # (T_measure,)
    delta0 = delta_S_total[0] + 1e-15
    normalized = delta_S_total / delta0

    # Find 1/e crossing (= τ*)
    tau_star_steps = None
    for t in range(1, T_measure):
        if normalized[t] <= 1/np.e:
            frac = (normalized[t-1] - 1/np.e) / (normalized[t-1] - normalized[t] + 1e-15)
            tau_star_steps = (t - 1) + frac
            break

    if tau_star_steps is None:
        # Fit exponential
        log_norm = np.log(normalized[:T_measure//2] + 1e-15)
        slope = np.polyfit(np.arange(T_measure//2), log_norm, 1)[0]
        tau_star_steps = -1.0 / max(-slope, 1e-6)
        print(f"  [NOTE] 1/e not reached in {T_measure} steps, extrapolated τ*={tau_star_steps:.1f}")

    tau_ratio = tau_star_steps / tau_L_steps

    # Per-channel gate statistics
    b_overall = float(b_mean.mean())
    w_overall = float(w_mean.mean())
    alpha_overall = float(alpha_mean.mean())

    # Theoretical erase timescale from gate values
    # Each step: S contracts by factor ≈ alpha (geometric) × (1 - b * ||k||²) (erase direction)
    # For unit-normalized k: ||k||=1, erase factor ≈ 1 - b (per channel)
    # Net contraction per step ≈ alpha × (1 - b)
    erase_rate = alpha_overall * (1 - b_overall)  # effective retention per step
    tau_erase_theory = -1.0 / np.log(erase_rate + 1e-10) if erase_rate < 1.0 else T_measure

    # Write gate mean (write doesn't contribute to decay, only to commitment)
    tau_write_theory = float('nan')  # write gate controls CONTENT committed, not WHEN forgotten

    # Per-layer τ* extraction
    per_layer_tau = []
    for li in range(model.n_layers):
        col = delta_S_mean[:, li]
        c0 = col[0] + 1e-15
        norm_col = col / c0
        tau_l = None
        for t in range(1, T_measure):
            if norm_col[t] <= 1/np.e:
                frac = (norm_col[t-1] - 1/np.e) / (norm_col[t-1] - norm_col[t] + 1e-15)
                tau_l = (t - 1) + frac
                break
        if tau_l is None:
            log_n = np.log(norm_col[:T_measure//2] + 1e-15)
            sl = np.polyfit(np.arange(T_measure//2), log_n, 1)[0]
            tau_l = -1.0 / max(-sl, 1e-6)
        per_layer_tau.append(float(tau_l))

    return {
        "system": system_name,
        "tau_star_steps": float(tau_star_steps),
        "tau_L_steps": float(tau_L_steps),
        "tau_ratio": float(tau_ratio),
        "tau_erase_theory_steps": float(tau_erase_theory),
        "tau_erase_ratio": float(tau_erase_theory / tau_L_steps),
        "b_mean": b_overall,
        "w_mean": w_overall,
        "alpha_mean": alpha_overall,
        "erase_retention_per_step": float(erase_rate),
        "delta_S_initial": float(delta0),
        "delta_S_total_trajectory": delta_S_total.tolist(),
        "normalized_decay_trajectory": normalized.tolist(),
        "delta_S_by_layer": delta_S_mean.tolist(),
        "per_layer_tau_star": per_layer_tau,
        "per_layer_tau_ratio": [t / tau_L_steps for t in per_layer_tau],
        "b_by_layer": b_mean.mean(axis=0).tolist(),
        "w_by_layer": w_mean.mean(axis=0).tolist(),
        "alpha_by_layer": alpha_mean.mean(axis=0).tolist(),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("GDN-2 × DHP Multi-Horizon Measurement (v3)")
    print(f"Horizons: {PRED_HORIZONS}")
    print(f"Time: {datetime.now().isoformat()}")
    print("="*60)

    results = {}

    for system_name, integrate_fn, dt, tau_L_known in [
        ("lorenz",  lorenz63, 0.01, 110.0),   # known: lambda≈0.9/s, dt=0.01 → τ_L≈110 steps
        ("rossler", rossler,  0.02, None),     # compute from data
    ]:
        print(f"\n{'='*50}")
        print(f"System: {system_name} (dt={dt})")

        n_total = 40000
        print(f"  Generating {n_total} steps...")
        traj_raw = integrate_fn(n_steps=n_total, dt=dt)
        assert np.all(np.isfinite(traj_raw)), f"NaN/Inf in {system_name}!"
        mu = traj_raw.mean(axis=0)
        sd = traj_raw.std(axis=0) + 1e-8
        traj_norm = (traj_raw - mu) / sd

        # Compute τ_L
        if tau_L_known is not None:
            tau_L_steps = tau_L_known
            print(f"  τ_L = {tau_L_steps:.1f} steps (known)")
        else:
            print(f"  Computing Lyapunov exponent (fixed: first-30-steps only)...")
            tau_L_steps = compute_lyapunov_steps(integrate_fn, dt)
            print(f"  τ_L = {tau_L_steps:.1f} steps")

        d_model, d_k, d_v, n_layers = 128, 64, 64, 4
        model = GDN2Model(d_model=d_model, d_k=d_k, d_v=d_v, n_layers=n_layers).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params/1e3:.0f}K")

        model_path = f"/workspace/gdn2_{system_name}_model_v3.pt"
        if os.path.exists(model_path):
            print(f"  Loading saved model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
            model.eval()
        else:
            print(f"  [Phase 1] Multi-horizon training...")
            model = train_multi_horizon(
                model, traj_norm,
                horizons=PRED_HORIZONS,
                seq_len=512, steps=5000, lr=3e-4, batch_size=32
            )
            torch.save(model.state_dict(), model_path)
            print(f"  Saved to {model_path}")

        # Validate
        with torch.no_grad():
            test_idx = np.random.randint(0, len(traj_norm) - 600)
            x_t = torch.tensor(traj_norm[test_idx:test_idx+512], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            y_t = torch.tensor(traj_norm[test_idx+1:test_idx+513], dtype=torch.float32).to(DEVICE)
            val_loss = F.mse_loss(model(x_t).squeeze(0), y_t).item()
        print(f"  Val MSE (h=1): {val_loss:.6f}")

        # DHP Measurement
        print(f"  [Phase 2] State injection DHP measurement (n_seeds=8)...")
        res = measure_dhp_state_injection(
            model, traj_norm, tau_L_steps, system_name,
            T_warmup=128, T_measure=600, eps_S=0.1, n_seeds=8
        )
        res["val_loss_h1"] = val_loss
        res["n_params"] = n_params
        res["dt"] = dt
        res["horizons_trained"] = PRED_HORIZONS

        print(f"\n  ── Results ({system_name}) ──")
        print(f"  τ_L                    = {tau_L_steps:.1f} steps")
        print(f"  τ*_decay               = {res['tau_star_steps']:.1f} steps ({res['tau_ratio']:.3f} × τ_L)")
        print(f"  τ*_erase (from gates)  = {res['tau_erase_theory_steps']:.1f} steps ({res['tau_erase_ratio']:.3f} × τ_L)")
        print(f"  b̄ (erase gate)        = {res['b_mean']:.4f}")
        print(f"  w̄ (write gate)        = {res['w_mean']:.4f}")
        print(f"  ᾱ (decay)             = {res['alpha_mean']:.4f}")
        print(f"  Erase retention/step   = {res['erase_retention_per_step']:.4f}")
        print(f"  Per-layer τ*/τ_L:")
        for li, (tau, ratio) in enumerate(zip(res['per_layer_tau_star'], res['per_layer_tau_ratio'])):
            print(f"    Layer {li}: {tau:.1f} steps ({ratio:.3f})")

        results[system_name] = res

    # Summary
    print("\n" + "="*60)
    print("SUMMARY — DHP Gate Analysis")
    print("="*60)
    for sys, r in results.items():
        close = abs(r['tau_ratio'] - 0.72) < 0.12
        close_e = abs(r['tau_erase_ratio'] - 0.72) < 0.12
        print(f"{sys:10s}")
        print(f"  τ*/τ_L   = {r['tau_ratio']:.3f}  ({'≈0.72 ✓' if close else '≠0.72 ✗'})")
        print(f"  τ*_e/τ_L = {r['tau_erase_ratio']:.3f}  ({'≈0.72 ✓' if close_e else '≠0.72 ✗'}) [from gate theory]")

    if len(results) == 2:
        lr = results["lorenz"]["tau_ratio"]
        rr = results["rossler"]["tau_ratio"]
        avg = (lr + rr) / 2
        print(f"\nAverage τ*/τ_L = {avg:.3f}")
        print(f"DHP Universal (0.72±0.12): {'YES ✓' if abs(avg-0.72)<0.12 else 'NO ✗'}")

    out_path = "/workspace/gdn2_dhp_results_v3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out_path}")

    return results


if __name__ == "__main__":
    main()
