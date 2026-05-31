"""
RWKV-7 DHP v6 — W_w Gradient Clip Ablation
============================================
Tests whether clipping the gradient magnitude of the W_w (decay) parameter
prevents RWKV-7 from collapsing to the uniform-timescale attractor.

Hypothesis (from v5 bimodal finding):
  The delta rule's outer(Sz, b) term enables rank-collapse of the S matrix when
  W_w gradients push all decay channels toward a single timescale. A tight gradient
  clip on W_w prevents this collapse, consistently guiding RWKV-7 to the diverse attractor.

Ablation design:
  4 conditions × 10 seeds × RWKV-7 only
  - w_clip = None    : control (v5 baseline — only global GRAD_CLIP=0.5 applies)
  - w_clip = 0.2     : moderate restriction on W_w
  - w_clip = 0.05    : tight restriction on W_w
  - w_clip = 0.01    : very tight restriction (near-frozen W_w)

Secondary ablation:
  RWKV-6 wider init: tau_max ∈ {93.5 (baseline), 150, 200} × 5 seeds
  Tests: does wider temporal scaffold → higher Δ(HL_CV)?

Archon — DuoNeural — 2026-05-28 (run AFTER v5c finishes on kilonova)
"""
import sys, os, json, math, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# ── Inherit all classes + helpers from v5 via exec ───────────────────────────
HERE = Path(__file__).parent
v5_src = open(HERE / "rwkv7_dhp_v5.py").read()
# Execute everything up to the Main block
exec(v5_src.split("# ─────────────────────────────────────────────────────────────────────────────\n# Main")[0])

# ── Ablation hyperparams ──────────────────────────────────────────────────────
N_SEEDS_WW   = 10     # seeds per W_w clip condition
N_SEEDS_R6   = 5      # seeds per RWKV-6 init range condition

# W_w clip conditions for RWKV-7
WW_CLIP_CONDITIONS = [None, 0.2, 0.05, 0.01]

# RWKV-6 tau_max conditions (sets the range of the fixed decay scaffold)
R6_TAU_MAX_CONDITIONS = [TAU_L * DHP_HI,   # 93.5 — baseline v5
                         150.0,             # 1.36× τ_L
                         200.0]             # 1.82× τ_L — beyond τ_L!

# ── Modified training: per-param grad clip for W_w ───────────────────────────
def train_model_ww_clip(model, traj_data, seed=42, ww_clip=None):
    """
    Same as train_model() from v5, but ALSO applies a separate gradient clip
    to W_w.weight and W_w.bias specifically, before the global clip.

    ww_clip=None → standard v5 behavior (only global clip applies).
    ww_clip=float → additionally clamp W_w gradient norms to this value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(traj_data)
    max_h = max(PRED_HORIZONS)
    win = SEQ_LEN + max_h
    losses = []
    log_interval = max(1, N_STEPS // 10)

    # Identify W_w parameters by name for targeted clip
    ww_params = []
    if ww_clip is not None:
        for name_p, p in model.named_parameters():
            if 'W_w' in name_p:
                ww_params.append(p)

    for step in range(1, N_STEPS + 1):
        start = np.random.randint(0, n - win)
        seq = torch.tensor(traj_data[start:start+win], dtype=DTYPE, device=DEVICE).unsqueeze(0)
        x_in = seq[:, :SEQ_LEN, :]

        head_outputs, _ = model(x_in)
        loss = multi_horizon_loss_v5(seq, PRED_HORIZONS, head_outputs)
        opt.zero_grad()
        loss.backward()

        # Apply W_w-specific gradient clip FIRST, before global clip
        if ww_clip is not None and ww_params:
            torch.nn.utils.clip_grad_norm_(ww_params, ww_clip)

        # Then global clip on all parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        losses.append(loss.item())

        if step % log_interval == 0:
            avg = np.mean(losses[-100:])
            print(f"    step={step:5d}  loss={loss.item():.6f}  avg100={avg:.6f}", flush=True)

    final_loss = float(np.mean(losses[-200:]))
    ok = final_loss < 2.0
    print(f"\n    Final loss (avg last 200): {final_loss:.6f} — {'✓ OK' if ok else '✗ FAILED'}", flush=True)
    return losses, ok, final_loss


# ── RWKV-6 with configurable tau_max ─────────────────────────────────────────
class RWKV6CellV6Ablation(torch.nn.Module):
    """
    RWKV-6 with configurable tau_max for the fixed decay scaffold.
    tau_max > TAU_L tests whether an over-wide scaffold helps or hurts.
    """
    def __init__(self, d_in=3, d_model=64, n_heads=4, horizons=PRED_HORIZONS,
                 tau_max=None):
        super().__init__()
        if tau_max is None:
            tau_max = TAU_L * DHP_HI  # baseline 93.5

        assert d_model % n_heads == 0
        self.H, self.N, self.D = n_heads, d_model // n_heads, d_model
        self.horizons = horizons

        self.proj_in  = torch.nn.Linear(d_in, d_model)
        self.W_r      = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k      = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v      = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_g      = torch.nn.Linear(d_model, d_model, bias=True)
        self.proj_out = torch.nn.Linear(d_model, d_model)
        self.ln_out   = torch.nn.LayerNorm(d_model)

        # Fixed per-channel decay spanning [τ=1, tau_max]
        with torch.no_grad():
            target_taus = np.linspace(1.0, tau_max, d_model)
            w_fixed = np.exp(-1.0 / target_taus)
            self.w_fixed = torch.nn.Parameter(
                torch.tensor(w_fixed, dtype=torch.float32), requires_grad=False
            )

        self.pred_heads = torch.nn.ModuleList([
            torch.nn.Linear(d_model, d_in) for _ in horizons
        ])
        self._last_slot_states = None
        self.tau_max = tau_max  # store for logging

    def init_state(self, B):
        return torch.zeros(B, self.H, self.N, self.N, device=DEVICE)

    def step(self, x_proj, S):
        B, H, N = S.shape[0], self.H, self.N
        r = self.W_r(x_proj).view(B, H, N)
        k = self.W_k(x_proj).view(B, H, N) / (N ** 0.5)
        v = self.W_v(x_proj).view(B, H, N)
        g = torch.sigmoid(self.W_g(x_proj))

        w = self.w_fixed.view(1, H, N)   # fixed, no gradient
        S_new = S * w[:, :, None, :] + torch.einsum('bhi,bhj->bhij', v, k)
        S_norm = S_new.norm(dim=(-2,-1), keepdim=True).clamp_min(1.0)
        S_new  = S_new / S_norm

        y = torch.einsum('bhij,bhj->bhi', S_new, r).reshape(B, -1)
        y = g * self.proj_out(self.ln_out(y))
        return y, S_new

    def forward(self, x, state=None):
        B, T, _ = x.shape
        if state is None:
            state = self.init_state(B)
        x_proj_all = self.proj_in(x)
        traj, head_log = [], []
        for t in range(T):
            y, state = self.step(x_proj_all[:, t, :], state)
            traj.append(y)
            head_log.append(state.view(B, self.H, -1).mean(-1, keepdim=True))
        traj_t = torch.stack(traj, 1)
        self._last_slot_states = torch.stack(head_log, 0)
        head_outputs = [head(traj_t) for head in self.pred_heads]
        return head_outputs, traj_t


# ── Run ablation helper ───────────────────────────────────────────────────────
def run_condition(name, model_fn, n_seeds, traj, ww_clip=None, train_fn=None):
    """
    Run n_seeds × one condition. Returns aggregate dict + seed details list.
    If ww_clip is not None, uses train_model_ww_clip. Otherwise uses train_model.
    """
    if train_fn is None:
        train_fn = (lambda m, t, s: train_model_ww_clip(m, t, seed=s, ww_clip=ww_clip))

    seed_hlcv_post  = []
    seed_hlcv_init  = []
    seed_final_loss = []
    seed_ok         = []
    seed_details    = []

    for seed in range(n_seeds):
        print(f"\n  --- Seed {seed} ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model_fn()
        if seed == 0:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  {n_params:,} trainable params", flush=True)

        init_hlcv = measure_init_hlcv(model, traj, n_runs=3)
        print(f"  HL_CV (init): {init_hlcv:.3f}", flush=True)

        losses, ok, final_loss = train_fn(model, traj, seed)

        spec = slot_specialization(model, traj)
        post_hlcv = spec['half_life_cv'] if spec else None

        if post_hlcv is not None:
            print(f"  HL_CV (post): {post_hlcv:.3f}", flush=True)
        if init_hlcv is not None and post_hlcv is not None:
            delta = post_hlcv - init_hlcv
            collapse = post_hlcv < 0.15   # collapsed = all heads ≈ same timescale
            print(f"  Δ(HL_CV):     {delta:+.3f} ← {'DHP ✓' if delta > 0.05 else 'no signal'} | {'COLLAPSED' if collapse else 'diverse'}", flush=True)
            seed_details.append({
                "seed": seed, "hlcv_init": init_hlcv, "hlcv_post": post_hlcv,
                "delta": float(delta), "ok": ok, "collapsed": collapse,
                "half_lives": spec.get('half_lives', []) if spec else []
            })
        if spec:
            print(f"  HL mean: {spec['half_life_mean']:.1f} | half-lives: {spec['half_lives']}", flush=True)

        if init_hlcv is not None: seed_hlcv_init.append(init_hlcv)
        if post_hlcv is not None: seed_hlcv_post.append(post_hlcv)
        seed_final_loss.append(final_loss)
        seed_ok.append(ok)

    mean_init  = float(np.mean(seed_hlcv_init)) if seed_hlcv_init else None
    mean_post  = float(np.mean(seed_hlcv_post)) if seed_hlcv_post else None
    std_post   = float(np.std(seed_hlcv_post))  if len(seed_hlcv_post) > 1 else 0.0
    delta_hlcv = float(mean_post - mean_init)   if (mean_post and mean_init) else None
    n_collapsed = sum(1 for d in seed_details if d.get("collapsed", False))
    n_diverse   = len(seed_details) - n_collapsed

    print(f"\n  === {name} AGGREGATE ({n_seeds} seeds) ===", flush=True)
    init_s = f"{mean_init:.3f}" if mean_init is not None else "N/A"
    print(f"  HL_CV init:    {init_s}", flush=True)
    if mean_post is not None:
        print(f"  HL_CV post:    {mean_post:.3f} ± {std_post:.3f}", flush=True)
        print(f"  Δ(HL_CV):      {delta_hlcv:+.3f}", flush=True)
    print(f"  Diverse/Collapsed: {n_diverse}/{n_collapsed} of {n_seeds} seeds", flush=True)
    print(f"  Training OK:   {sum(seed_ok)}/{n_seeds}", flush=True)

    return {
        "name":            name,
        "hlcv_init_mean":  mean_init,
        "hlcv_post_mean":  mean_post,
        "hlcv_post_std":   std_post,
        "delta_hlcv":      delta_hlcv,
        "n_diverse":       n_diverse,
        "n_collapsed":     n_collapsed,
        "n_seeds":         n_seeds,
        "n_ok":            sum(seed_ok),
        "seed_details":    seed_details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
print(f"RWKV-7 DHP v6 Ablation — {datetime.now().isoformat()}", flush=True)
print(f"Ablation A: RWKV-7 W_w grad clip  × {N_SEEDS_WW} seeds × 4 conditions", flush=True)
print(f"Ablation B: RWKV-6 tau_max range  × {N_SEEDS_R6} seeds × 3 conditions", flush=True)
print(f"{'='*70}\n", flush=True)

traj = lorenz_trajectory(n=8000, dt=DT)
assert not np.isnan(traj).any()
print(f"Lorenz: {len(traj)} steps, no NaN ✓\n", flush=True)

all_results = {}

# ─────────────────────────────────────────────────────────────────────────────
# Ablation A: RWKV-7 W_w gradient clip
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
print(f"ABLATION A: RWKV-7 W_w gradient clipping", flush=True)
print(f"{'='*70}", flush=True)
print(f"Question: Does clipping W_w gradient norm prevent collapse to uniform timescale?", flush=True)
print(f"Prediction: Smaller ww_clip → more seeds in diverse attractor (↓ collapse rate)\n", flush=True)

for clip_val in WW_CLIP_CONDITIONS:
    clip_label = f"ww_clip={clip_val}" if clip_val is not None else "ww_clip=None (control)"
    print(f"\n{'─'*60}", flush=True)
    print(f"Condition: {clip_label}", flush=True)
    print(f"{'─'*60}", flush=True)

    result = run_condition(
        name     = f"RWKV-7_{clip_label}",
        model_fn = lambda: RWKV7CellV5(D_IN, D_HIDDEN, n_heads=4),
        n_seeds  = N_SEEDS_WW,
        traj     = traj,
        ww_clip  = clip_val,
    )
    all_results[f"rwkv7_ww_{clip_val}"] = result

# Print Ablation A summary
print(f"\n\n{'='*70}", flush=True)
print(f"ABLATION A SUMMARY — RWKV-7 W_w Gradient Clip", flush=True)
print(f"{'='*70}", flush=True)
print(f"{'Condition':<22} {'Δ(HL_CV)':>9} {'std':>7} {'diverse':>8} {'collapsed':>10}", flush=True)
print(f"{'-'*60}", flush=True)
for clip_val in WW_CLIP_CONDITIONS:
    key = f"rwkv7_ww_{clip_val}"
    r   = all_results[key]
    dlt = f"{r['delta_hlcv']:+.3f}" if r['delta_hlcv'] is not None else "  N/A"
    std = f"{r['hlcv_post_std']:.3f}" if r['hlcv_post_std'] else "N/A"
    label = f"ww_clip={clip_val}" if clip_val is not None else "ww_clip=None"
    print(f"  {label:<20} {dlt:>9} {std:>7} {r['n_diverse']:>8}/{r['n_seeds']} {r['n_collapsed']:>10}/{r['n_seeds']}", flush=True)

print(f"\nHypothesis supported if: n_diverse increases as clip_val decreases", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Ablation B: RWKV-6 tau_max range
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n\n{'='*70}", flush=True)
print(f"ABLATION B: RWKV-6 tau_max scaffold width", flush=True)
print(f"{'='*70}", flush=True)
print(f"Question: Does wider fixed scaffold → higher Δ(HL_CV)?", flush=True)
print(f"Prediction: tau_max=200 > tau_max=150 > tau_max=93.5 (baseline)\n", flush=True)

for tau_max in R6_TAU_MAX_CONDITIONS:
    label = f"tau_max={tau_max:.1f}"
    print(f"\n{'─'*60}", flush=True)
    print(f"Condition: RWKV-6 {label} (τ_L={TAU_L:.0f})", flush=True)
    print(f"{'─'*60}", flush=True)

    result = run_condition(
        name     = f"RWKV-6_{label}",
        model_fn = lambda tm=tau_max: RWKV6CellV6Ablation(D_IN, D_HIDDEN, n_heads=4, tau_max=tm),
        n_seeds  = N_SEEDS_R6,
        traj     = traj,
        ww_clip  = None,
        train_fn = lambda m, t, s: train_model(m, t, seed=s),  # standard training for RWKV-6
    )
    all_results[f"rwkv6_tau{int(tau_max)}"] = result

# Print Ablation B summary
print(f"\n\n{'='*70}", flush=True)
print(f"ABLATION B SUMMARY — RWKV-6 scaffold width", flush=True)
print(f"{'='*70}", flush=True)
print(f"{'Condition':<22} {'τ_max':>7} {'Δ(HL_CV)':>9} {'std':>7} {'n_pos':>7}", flush=True)
print(f"{'-'*60}", flush=True)
for tau_max in R6_TAU_MAX_CONDITIONS:
    key = f"rwkv6_tau{int(tau_max)}"
    r   = all_results[key]
    dlt = f"{r['delta_hlcv']:+.3f}" if r['delta_hlcv'] is not None else "  N/A"
    std = f"{r['hlcv_post_std']:.3f}" if r['hlcv_post_std'] else "N/A"
    n_pos = sum(1 for d in r['seed_details'] if d.get('delta', 0) > 0.05)
    print(f"  RWKV-6 tau={tau_max:<8.1f} {tau_max:>7.1f} {dlt:>9} {std:>7} {n_pos:>7}/{r['n_seeds']}", flush=True)

print(f"\nHypothesis supported if: Δ(HL_CV) monotonically increases with tau_max", flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    "version":    "v6_ablation",
    "timestamp":  datetime.now().isoformat(),
    "tau_L": TAU_L, "dt": DT, "seq_len": SEQ_LEN,
    "pred_horizons": PRED_HORIZONS,
    "n_steps": N_STEPS, "n_seeds_rwkv7": N_SEEDS_WW, "n_seeds_rwkv6": N_SEEDS_R6,
    "ablation_A": {
        "name":        "RWKV-7 W_w gradient clip",
        "conditions":  WW_CLIP_CONDITIONS,
        "hypothesis":  "Smaller ww_clip → fewer collapsed seeds → higher diversity stability",
    },
    "ablation_B": {
        "name":        "RWKV-6 scaffold tau_max",
        "conditions":  R6_TAU_MAX_CONDITIONS,
        "hypothesis":  "Wider scaffold → higher mean Δ(HL_CV)",
    },
    "results": all_results,
}
out_path = HERE / "rwkv7_dhp_v6_ablation.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved → {out_path}", flush=True)
print(f"v6 ablation complete. — {datetime.now().isoformat()}", flush=True)
