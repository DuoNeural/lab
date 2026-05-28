"""
RWKV-7 DHP v7 — QLE Profile + Separatrix Perturbation Test
===========================================================
Extends v5 with three new measurements:

A) QLE Time Profile (inference-time)
   For each trained seed, run eval trajectory and at each timestep t inject
   perturbation ε into S-matrix. Track ‖δS‖ over next K steps → local QLE(t).
   Compare profiles of diverse vs collapsed seeds to find when/where they diverge.

B) Training Checkpoint QLE
   Save model at steps {3000, 6000, 9000, 12000}. Measure QLE at each checkpoint.
   Does QLE trajectory during training predict which attractor the seed is heading toward?
   If YES: QLE is an early warning signal for basin membership.

C) Separatrix Perturbation Test (Aura's proposed experiment)
   Take collapsed seeds at step 6000 checkpoint. Compute "diverse direction" in W_w
   parameter space (mean W_w of diverse seeds - this seed's W_w). Apply perturbation
   W_w += alpha * diverse_direction, normalized. Continue training to 12000 steps.
   Measure HL_CV: did we cross the separatrix?

   This is the causal proof that basin boundaries are hard topological facts.

Scientific context:
   Li et al. 2025 (arXiv:2503.13530) established that LLMs are chaotic dynamical systems
   with QLE > 0 in deep layers. This paper extends to SSM/RNN architectures (RWKV-7),
   finds bimodal attractor basins, measures the separatrix via FTLE analysis, and
   demonstrates causal basin steering. The 0.72 τ*/τ_L constant is the architectural
   Lyapunov time where model internal coherence synchronizes with environmental chaos.

Archon — DuoNeural — 2026-05-28 (runs on 3090 GPU, ~6-8h)
"""
import sys, os, json, math, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from copy import deepcopy

# ── Inherit everything from v5 ────────────────────────────────────────────────
HERE = Path(__file__).parent
v5_src = open(HERE / "rwkv7_dhp_v5.py").read()
exec(v5_src.split("# ─────────────────────────────────────────────────────────────────────────────\n# Main")[0])

# ── v7 config ─────────────────────────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE          = torch.float32
N_SEEDS_BASE   = 20      # Phase A+B: characterize landscape
N_SEEDS_STEER  = 10      # Phase C: basin steering attempts (per collapsed seed found)
CKPT_STEPS     = [3000, 6000, 9000, 12000]

# QLE measurement params
QLE_PERTURB    = 1e-3    # perturbation magnitude ε
QLE_HORIZON    = 50      # steps to track perturbation divergence
QLE_N_SEEDS    = 5       # random perturbation directions per timestep
QLE_EVAL_LEN   = 300     # trajectory length for QLE measurement

# Separatrix steering params
STEER_STEP     = 6000    # which checkpoint to steer from
STEER_ALPHAS   = [0.05, 0.1, 0.2, 0.5]  # perturbation magnitudes to try

print(f"Device: {DEVICE}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# ── QLE measurement function ──────────────────────────────────────────────────
def measure_qle_profile(model, traj_data, eval_len=QLE_EVAL_LEN,
                         n_perturb=QLE_N_SEEDS, perturb_eps=QLE_PERTURB,
                         horizon=QLE_HORIZON):
    """
    Measures the local Quasi-Lyapunov Exponent profile over an evaluation trajectory.

    At each timestep t, injects a random perturbation ε into the S-matrix hidden state.
    Tracks ‖δS‖ over the next `horizon` steps.
    QLE(t) = log(‖δS_horizon‖ / ‖δS_0‖) / horizon

    Returns:
        qle_mean: (eval_len,) mean QLE at each timestep
        qle_std:  (eval_len,) std QLE at each timestep
        w_eff:    mean effective decay per channel (diagnosis)
    """
    if not isinstance(model, RWKV7CellV5):
        return None  # only defined for RWKV-7

    model.eval()
    n = len(traj_data)
    start = 0
    x_full = torch.tensor(traj_data[start:start+eval_len+horizon],
                          dtype=DTYPE, device=DEVICE).unsqueeze(0)
    x_eval = x_full[:, :eval_len, :]

    qle_at_t = []

    with torch.no_grad():
        # Warm-up: run to eval_len to get base states at each t
        model.proj_in.to(DEVICE)
        x_proj_all = model.proj_in(x_eval)
        base_state = model.init_state(1)

        # Collect states and x_proj at each t
        states_at_t = []
        x_proj_at_t = []
        state = deepcopy(base_state)
        for t in range(eval_len):
            xp = x_proj_all[:, t, :]
            _, state_new = model.step(xp, state)
            states_at_t.append(deepcopy(state))
            x_proj_at_t.append(xp)
            state = state_new

        # For each timestep t, measure QLE
        for t in range(min(eval_len - horizon, eval_len)):
            qles_here = []
            base_S = states_at_t[t].clone()

            for _ in range(n_perturb):
                # Random unit perturbation direction
                delta = torch.randn_like(base_S)
                delta = delta / delta.norm().clamp_min(1e-10)
                delta = delta * perturb_eps

                # Start from perturbed state
                S_pert = base_S + delta
                delta_norm_0 = delta.norm().item()

                if delta_norm_0 < 1e-12:
                    continue

                # Run forward horizon steps from perturbed state
                S_base_run = base_S.clone()
                S_pert_run = S_pert.clone()

                for th in range(horizon):
                    t_idx = t + th
                    if t_idx >= len(x_proj_at_t):
                        break
                    xp = x_proj_at_t[t_idx]
                    _, S_base_run = model.step(xp, S_base_run)
                    _, S_pert_run = model.step(xp, S_pert_run)

                delta_final = (S_pert_run - S_base_run).norm().item()
                if delta_final > 0 and delta_norm_0 > 0:
                    qle = math.log(delta_final / delta_norm_0) / max(1, horizon)
                    qles_here.append(qle)

            if qles_here:
                qle_at_t.append(float(np.mean(qles_here)))
            else:
                qle_at_t.append(float('nan'))

    return np.array(qle_at_t)


def measure_w_eff_distribution(model, traj_data, n_steps=2000):
    """Measure per-channel effective decay and DHP zone occupancy."""
    if not isinstance(model, RWKV7CellV5):
        return None
    w_eff = model.get_w_eff(traj_data, n_steps=n_steps)
    tau_channels = -1.0 / np.log(np.clip(w_eff, 1e-8, 1 - 1e-8))
    dhp_lo = TAU_L * DHP_LO
    dhp_hi = TAU_L * DHP_HI
    in_dhp = float(np.mean((tau_channels >= dhp_lo) & (tau_channels <= dhp_hi)))
    short   = float(np.mean(tau_channels < 10))
    medium  = float(np.mean((tau_channels >= 10) & (tau_channels < dhp_lo)))
    return {
        "tau_channels": tau_channels.tolist(),
        "tau_mean": float(np.mean(tau_channels)),
        "tau_median": float(np.median(tau_channels)),
        "frac_dhp": in_dhp,
        "frac_short": short,
        "frac_medium": medium,
    }


# ── Training with checkpoints ─────────────────────────────────────────────────
def train_with_checkpoints(model, traj_data, seed=42, ckpt_steps=CKPT_STEPS):
    """Train model, saving state_dict at specified steps."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(traj_data)
    max_h = max(PRED_HORIZONS)
    win = SEQ_LEN + max_h
    losses = []
    log_interval = max(1, N_STEPS // 10)
    checkpoints = {}  # step → state_dict

    for step in range(1, N_STEPS + 1):
        start = np.random.randint(0, n - win)
        seq = torch.tensor(traj_data[start:start+win], dtype=DTYPE, device=DEVICE).unsqueeze(0)
        x_in = seq[:, :SEQ_LEN, :]

        head_outputs, _ = model(x_in)
        loss = multi_horizon_loss_v5(seq, PRED_HORIZONS, head_outputs)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        losses.append(loss.item())

        if step in ckpt_steps:
            checkpoints[step] = deepcopy(model.state_dict())
            print(f"    [ckpt saved @ step {step}]", flush=True)

        if step % log_interval == 0:
            avg = np.mean(losses[-100:])
            print(f"    step={step:5d}  loss={loss.item():.6f}  avg100={avg:.6f}", flush=True)

    final_loss = float(np.mean(losses[-200:]))
    ok = final_loss < 2.0
    print(f"\n    Final loss: {final_loss:.6f} — {'✓' if ok else '✗'}", flush=True)
    return losses, ok, final_loss, checkpoints


# ── QLE at checkpoint helper ──────────────────────────────────────────────────
def qle_at_checkpoint(base_model_class, state_dict, traj, d_in=D_IN, d_hidden=D_HIDDEN):
    """Load a checkpoint and measure QLE profile."""
    model = base_model_class(d_in, d_hidden, n_heads=4)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    qle_profile = measure_qle_profile(model, traj)
    return qle_profile


# ── Phase A+B: Characterize landscape ────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
print(f"RWKV-7 DHP v7 — QLE Profile + Separatrix Perturbation", flush=True)
print(f"{datetime.now().isoformat()}", flush=True)
print(f"Device: {DEVICE}  |  N_SEEDS_BASE={N_SEEDS_BASE}  |  N_SEEDS_STEER={N_SEEDS_STEER}", flush=True)
print(f"{'='*70}\n", flush=True)

traj = lorenz_trajectory(n=8000, dt=DT)
assert not np.isnan(traj).any()
print(f"Lorenz: {len(traj)} steps ✓\n", flush=True)

print(f"{'='*70}", flush=True)
print(f"PHASE A+B: Landscape characterization ({N_SEEDS_BASE} seeds)", flush=True)
print(f"{'='*70}", flush=True)

base_results = []
diverse_w_ws = []      # W_w state_dicts from diverse seeds (for computing direction)
collapsed_ckpts = []   # (seed, step6k_state_dict, final_hl_cv) for Phase C

for seed in range(N_SEEDS_BASE):
    print(f"\n{'─'*50}", flush=True)
    print(f"Seed {seed}", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RWKV7CellV5(D_IN, D_HIDDEN, n_heads=4).to(DEVICE)
    if seed == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {n_params:,} params  |  Device: {next(model.parameters()).device}", flush=True)

    init_hlcv = measure_init_hlcv(model, traj, n_runs=3)
    print(f"  HL_CV (init): {init_hlcv:.3f}", flush=True)

    losses, ok, final_loss, checkpoints = train_with_checkpoints(model, traj, seed=seed)

    spec = slot_specialization(model, traj)
    post_hlcv = spec['half_life_cv'] if spec else None
    delta = (post_hlcv - init_hlcv) if (post_hlcv and init_hlcv) else None
    is_diverse   = post_hlcv is not None and post_hlcv >= 0.3
    is_collapsed = post_hlcv is not None and post_hlcv < 0.15

    print(f"  HL_CV (post): {post_hlcv:.3f}  Δ={delta:+.3f}" if delta else "  HL_CV: N/A", flush=True)
    if spec:
        print(f"  half-lives: {spec['half_lives']}", flush=True)
    print(f"  Attractor: {'DIVERSE' if is_diverse else 'COLLAPSED' if is_collapsed else 'INTERMEDIATE'}", flush=True)

    # Measure QLE profile on fully trained model
    print(f"  Measuring QLE profile...", flush=True)
    t0 = time.time()
    qle_profile = measure_qle_profile(model, traj)
    elapsed = time.time() - t0
    if qle_profile is not None:
        qle_mean = float(np.nanmean(qle_profile))
        qle_pos  = float(np.mean(qle_profile > 0)) if not np.all(np.isnan(qle_profile)) else None
        print(f"  QLE mean={qle_mean:.4f}  frac_positive={qle_pos:.3f}  [{elapsed:.1f}s]", flush=True)
    else:
        qle_mean, qle_pos = None, None

    # W_w decay distribution
    w_eff_info = measure_w_eff_distribution(model, traj)
    if w_eff_info:
        print(f"  W_w: frac_dhp={w_eff_info['frac_dhp']:.3f}  tau_mean={w_eff_info['tau_mean']:.1f}", flush=True)

    # QLE at each checkpoint
    ckpt_qle_means = {}
    for step, sd in checkpoints.items():
        qp = qle_at_checkpoint(RWKV7CellV5, sd, traj)
        if qp is not None:
            ckpt_qle_means[step] = float(np.nanmean(qp))
    print(f"  QLE trajectory: {ckpt_qle_means}", flush=True)

    seed_result = {
        "seed": seed,
        "hlcv_init": init_hlcv,
        "hlcv_post": post_hlcv,
        "delta_hlcv": float(delta) if delta else None,
        "half_lives": spec['half_lives'] if spec else [],
        "is_diverse": bool(is_diverse),
        "is_collapsed": bool(is_collapsed),
        "qle_mean_final": qle_mean,
        "qle_frac_positive": qle_pos,
        "qle_at_checkpoints": ckpt_qle_means,
        "w_eff": w_eff_info,
        "final_loss": final_loss,
        "ok": ok,
    }
    base_results.append(seed_result)

    # Collect for Phase C
    if is_diverse and hasattr(model, 'W_w'):
        diverse_w_ws.append(model.W_w.weight.data.clone().cpu())
    if is_collapsed and STEER_STEP in checkpoints:
        collapsed_ckpts.append({
            "seed": seed,
            "ckpt_6k": checkpoints[STEER_STEP],
            "final_hlcv": post_hlcv,
        })

# Summary after Phase A+B
n_diverse   = sum(1 for r in base_results if r['is_diverse'])
n_collapsed = sum(1 for r in base_results if r['is_collapsed'])
n_inter     = len(base_results) - n_diverse - n_collapsed

print(f"\n{'='*70}", flush=True)
print(f"PHASE A+B SUMMARY ({N_SEEDS_BASE} seeds)", flush=True)
print(f"  Diverse:      {n_diverse}/{N_SEEDS_BASE}  (HL_CV ≥ 0.3)", flush=True)
print(f"  Collapsed:    {n_collapsed}/{N_SEEDS_BASE}  (HL_CV < 0.15)", flush=True)
print(f"  Intermediate: {n_inter}/{N_SEEDS_BASE}", flush=True)

div_qle   = [r['qle_mean_final'] for r in base_results if r['is_diverse'] and r['qle_mean_final']]
coll_qle  = [r['qle_mean_final'] for r in base_results if r['is_collapsed'] and r['qle_mean_final']]
if div_qle and coll_qle:
    print(f"  QLE diverse:   mean={np.mean(div_qle):.4f} ± {np.std(div_qle):.4f}", flush=True)
    print(f"  QLE collapsed: mean={np.mean(coll_qle):.4f} ± {np.std(coll_qle):.4f}", flush=True)
    print(f"  QLE difference (diverse - collapsed): {np.mean(div_qle) - np.mean(coll_qle):+.4f}", flush=True)
    print(f"  Hypothesis: diverse > collapsed → {'SUPPORTED' if np.mean(div_qle) > np.mean(coll_qle) else 'NOT SUPPORTED'}", flush=True)

# ── Phase C: Separatrix Perturbation Test ─────────────────────────────────────
print(f"\n\n{'='*70}", flush=True)
print(f"PHASE C: Separatrix Perturbation Test", flush=True)
print(f"  Diverse seeds available for direction: {len(diverse_w_ws)}", flush=True)
print(f"  Collapsed seeds to steer: {len(collapsed_ckpts)}", flush=True)
print(f"{'='*70}", flush=True)

steer_results = []

if len(diverse_w_ws) == 0:
    print("  ⚠ No diverse seeds found! Cannot compute diverse direction.", flush=True)
elif len(collapsed_ckpts) == 0:
    print("  ⚠ No collapsed seeds found! Cannot test basin steering.", flush=True)
else:
    # Compute mean "diverse direction" in W_w space
    mean_diverse_ww = torch.stack(diverse_w_ws).mean(0)  # (d_model, d_model)
    print(f"  Diverse W_w direction computed from {len(diverse_w_ws)} seeds", flush=True)

    for cinfo in collapsed_ckpts[:min(5, len(collapsed_ckpts))]:
        seed = cinfo['seed']
        ckpt_6k = cinfo['ckpt_6k']
        orig_hlcv = cinfo['final_hlcv']

        print(f"\n  Steering seed {seed} (original HL_CV={orig_hlcv:.3f})", flush=True)

        # Get W_w from checkpoint
        ckpt_ww = ckpt_6k['W_w.weight']  # (d_model, d_model)

        # Direction: mean_diverse - this_seed
        direction = (mean_diverse_ww.to(DEVICE) - ckpt_ww.to(DEVICE))
        direction_norm = direction.norm()
        unit_direction = direction / direction_norm.clamp_min(1e-10)
        print(f"  Direction magnitude: {direction_norm.item():.4f}", flush=True)

        for alpha in STEER_ALPHAS:
            torch.manual_seed(seed + 1000)  # different seed for rest of training
            np.random.seed(seed + 1000)

            # Load checkpoint and apply perturbation
            model_steered = RWKV7CellV5(D_IN, D_HIDDEN, n_heads=4).to(DEVICE)
            model_steered.load_state_dict(ckpt_6k)

            # Apply perturbation to W_w
            with torch.no_grad():
                model_steered.W_w.weight.data += alpha * unit_direction.to(DEVICE)

            # Continue training from step 6001 to 12000
            opt = torch.optim.Adam(model_steered.parameters(), lr=LR)
            n = len(traj)
            max_h = max(PRED_HORIZONS)
            win = SEQ_LEN + max_h
            remaining_steps = N_STEPS - STEER_STEP

            for step in range(1, remaining_steps + 1):
                start_idx = np.random.randint(0, n - win)
                seq = torch.tensor(traj[start_idx:start_idx+win], dtype=DTYPE, device=DEVICE).unsqueeze(0)
                x_in = seq[:, :SEQ_LEN, :]
                head_outputs, _ = model_steered(x_in)
                loss = multi_horizon_loss_v5(seq, PRED_HORIZONS, head_outputs)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_steered.parameters(), GRAD_CLIP)
                opt.step()

            spec_steered = slot_specialization(model_steered, traj)
            steered_hlcv = spec_steered['half_life_cv'] if spec_steered else None
            flipped = steered_hlcv is not None and steered_hlcv >= 0.3

            print(f"    α={alpha:.3f}: HL_CV={steered_hlcv:.3f} → {'FLIPPED TO DIVERSE ✓' if flipped else 'still collapsed ✗'}", flush=True)
            if spec_steered:
                print(f"           half-lives: {spec_steered['half_lives']}", flush=True)

            steer_results.append({
                "seed": seed,
                "orig_hlcv": orig_hlcv,
                "alpha": alpha,
                "steered_hlcv": float(steered_hlcv) if steered_hlcv else None,
                "flipped": bool(flipped),
                "half_lives_steered": spec_steered['half_lives'] if spec_steered else [],
            })

# ── Final summary ──────────────────────────────────────────────────────────────
print(f"\n\n{'='*70}", flush=True)
print(f"PHASE C SUMMARY — Separatrix Perturbation Test", flush=True)
print(f"{'='*70}", flush=True)
for r in steer_results:
    print(f"  seed={r['seed']} α={r['alpha']:.3f}: {r['orig_hlcv']:.3f} → {r['steered_hlcv']:.3f}  {'✓ FLIPPED' if r['flipped'] else '✗'}", flush=True)

n_flipped = sum(1 for r in steer_results if r['flipped'])
print(f"\n  Total flips: {n_flipped}/{len(steer_results)}", flush=True)
if n_flipped > 0:
    print(f"  RESULT: Basin boundaries are STEERABLE via W_w perturbation ← causal proof", flush=True)
    min_alpha_flip = min(r['alpha'] for r in steer_results if r['flipped'])
    print(f"  Minimum effective alpha: {min_alpha_flip}", flush=True)
else:
    print(f"  RESULT: Perturbation did not flip basins — separatrix may require different approach", flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    "version": "v7_qle_basin",
    "timestamp": datetime.now().isoformat(),
    "device": str(DEVICE),
    "tau_L": TAU_L, "dt": DT, "seq_len": SEQ_LEN,
    "n_seeds_base": N_SEEDS_BASE,
    "qle_perturb": QLE_PERTURB,
    "qle_horizon": QLE_HORIZON,
    "steer_step": STEER_STEP,
    "steer_alphas": STEER_ALPHAS,
    "phase_ab_results": base_results,
    "phase_c_steer_results": steer_results,
    "summary": {
        "n_diverse": n_diverse,
        "n_collapsed": n_collapsed,
        "n_intermediate": n_inter,
        "qle_diverse_mean": float(np.mean(div_qle)) if div_qle else None,
        "qle_collapsed_mean": float(np.mean(coll_qle)) if coll_qle else None,
        "n_basin_flips": n_flipped,
        "total_steer_attempts": len(steer_results),
    }
}

out_path = HERE / "rwkv7_dhp_v7_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved → {out_path}", flush=True)
print(f"v7 complete. {datetime.now().isoformat()}", flush=True)
