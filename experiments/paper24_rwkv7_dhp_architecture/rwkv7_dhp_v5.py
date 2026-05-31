"""
RWKV-7 DHP v5 — Learned Temporal Diversity via Delta Rule
==========================================================

v4 diagnosis:
  - Separate heads (Fix 1) was correct ✓
  - W_w init spanning DHP range was correct ✓
  - BUT: shared body gets gradient from ALL heads simultaneously
    Weight 1/(temp*h) → h=8 contributes 2.5×, h=80 only 0.25×
    → Short-horizon gradient pressure DESTROYS long-range W_w channels
    → RWKV-7 v4 HL_CV = 0.133 (WORSE than v3's 0.243 due to DHP-init destroyed by gradients)
  - argmin(norm_loss) τ* measurement always picks h=16 for any Lorenz predictor (h=8 too easy
    to be informative relative to persistence, h=16 is optimal effort)

v5 NEW APPROACH: Measure learned temporal diversity DIRECTLY
  Key insight: RWKV-7's HL_CV doesn't reflect W_w decay ALONE.
  The delta rule `S_t += outer(v_t, k_t) + outer(Sz_t, b_t)` creates IMPLICIT
  long-range memory even with short explicit decay. HL_CV measures the COMBINED
  effective channel timescales from BOTH mechanisms.

  Scientific claim: RWKV-7's delta rule creates MORE temporal diversity than
  fixed-decay architectures, and this diversity is LEARNED (not pre-existing).

  Proof design:
    1. Measure HL_CV at initialization (before any training) for each arch
    2. Measure HL_CV after training
    3. Δ = HL_CV_after - HL_CV_before
    4. RWKV-7 should show LARGEST positive Δ (learning drives diversity)
    5. RWKV-6 (fixed decay) Δ should be near zero (diversity is static)
    6. Mamba (scalar gate) Δ should be small (scalar gate can't diversify)

v5 protocol fixes:
  Fix 1: Measure HL_CV at init (before training) as baseline
  Fix 2: Use EQUAL HORIZON WEIGHTS in loss (remove 1/(temp*h) short-horizon bias)
          All horizons contribute equally: loss = (1/n_h) * Σ_h MSE(head_h, target_h)
          DHP emergence measured via HL_CV delta, not via per-head loss argmin
  Fix 3: Channel effective timescale = measure AUTOCORRELATION of S matrix entries
          (captures both W_w and delta rule contributions)
  Fix 4: Multiple seeds (n=4) for stability

Archon — DuoNeural — 2026-05-27
"""
import torch, json, math, numpy as np
from datetime import datetime
from pathlib import Path
from copy import deepcopy

torch.manual_seed(42)
DEVICE = "cpu"
DTYPE  = torch.float32

# ── Hyperparams ───────────────────────────────────────────────────────────────
N_STEPS       = 12000
LR            = 3e-4
GRAD_CLIP     = 0.5
N_SEEDS       = 4

DT            = 0.01
TAU_L         = 110.0
DHP_LO, DHP_HI = 0.55, 0.85   # DHP zone: [60.5, 93.5] steps
D_IN          = 3
D_HIDDEN      = 64
# Equal spacing from h=8 to h=80 — no short-horizon bias
PRED_HORIZONS = [8, 16, 32, 64, 80]
SEQ_LEN       = 100

HL_CV_THRESHOLD = 0.3   # DHP confirmed if HL_CV > this after training

print(f"Device: {DEVICE}", flush=True)
print(f"τ_L = {TAU_L}, dt = {DT}", flush=True)
print(f"v5: EQUAL horizon weights (no short-horizon bias)", flush=True)
print(f"v5: Δ(HL_CV) = HL_CV_post - HL_CV_init as DHP signal", flush=True)
print(f"PRED_HORIZONS = {PRED_HORIZONS}", flush=True)

# ── Lorenz ────────────────────────────────────────────────────────────────────
def lorenz_trajectory(n=8000, dt=DT, sigma=10., rho=28., beta=8/3, seed=0):
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

# ── FIX 2: Equal-weight loss (no short-horizon bias) ─────────────────────────
def multi_horizon_loss_v5(seqs, horizons, head_outputs):
    """
    Equal weight across ALL horizons. Each head_h learns its own task.
    No temperature annealing — uniform gradient pressure on all timescales.
    """
    total = torch.tensor(0.0, device=DEVICE)
    for i, h in enumerate(horizons):
        pred_h = head_outputs[i]               # (B, SEQ_LEN, D_IN)
        target = seqs[:, h:SEQ_LEN + h, :]    # (B, SEQ_LEN, D_IN)
        total  = total + torch.nn.functional.mse_loss(pred_h, target)
    return total / len(horizons)

# ── Training ──────────────────────────────────────────────────────────────────
def train_model(model, traj_data, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(traj_data)
    max_h = max(PRED_HORIZONS)
    win = SEQ_LEN + max_h
    losses = []
    log_interval = max(1, N_STEPS // 10)

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

        if step % log_interval == 0:
            avg = np.mean(losses[-100:])
            print(f"  step={step:5d}  loss={loss.item():.6f}  avg100={avg:.6f}", flush=True)

    final_loss = float(np.mean(losses[-200:]))
    ok = final_loss < 2.0   # 5 equal-weight heads each contributing ~MSE, threshold accordingly
    print(f"\n  Final loss (avg last 200): {final_loss:.6f} — {'✓ OK' if ok else '✗ FAILED'}", flush=True)
    return losses, ok, final_loss

# ── FIX 3: Slot specialization measuring EFFECTIVE timescales (W_w + delta rule)
def slot_specialization(model, traj_data, eval_len=400, max_lag=120, eps=1e-8):
    """
    Measures per-slot/head autocorrelation half-life of INTERNAL STATES.
    For RWKV-7: state = S matrix entries (d_model dimensions after flattening)
    For CTM-like: state = per-slot h vectors
    For RWKV-6: same as RWKV-7 but S evolves with fixed decay
    For Mamba: hidden state vector

    HL_CV > 0.3 → diverse temporal scales (DHP-like behavior)
    HL_CV < 0.3 → uniform temporal scales (no specialization)
    """
    n = len(traj_data)
    start = np.random.randint(0, max(1, n - eval_len - 1))
    x = torch.tensor(traj_data[start:start+eval_len], dtype=DTYPE, device=DEVICE).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        model(x)
    if not hasattr(model, '_last_slot_states') or model._last_slot_states is None:
        return None
    states = model._last_slot_states.cpu()   # (T, B, S, F)
    T, B, S, F = states.shape
    states = states.squeeze(1)  # (T, S, F)

    # Normalize per slot
    x_s = states.float() - states.float().mean(0, keepdim=True)
    var  = (x_s * x_s).mean(0).clamp_min(eps)   # (S, F)

    # Per-slot mean (aggregate over features)
    slot_means = x_s.mean(-1)  # (T, S)

    # Autocorrelation curves per slot
    ac_curves = []
    for lag in range(1, min(max_lag + 1, T)):
        c = (slot_means[:-lag] * slot_means[lag:]).mean(0) / (slot_means.std(0).clamp_min(eps) ** 2)
        ac_curves.append(c)
    ac = torch.stack(ac_curves, 0)  # (max_lag, S)

    # Half-lives: first lag where autocorrelation drops to 0.5
    half_life = []
    for s in range(S):
        below = torch.where(ac[:, s] < 0.5)[0]
        half_life.append(int(below[0].item()) + 1 if len(below) else max_lag)
    hl = torch.tensor(half_life, dtype=torch.float32)

    # Inter-slot correlation (redundancy measure)
    if S > 1:
        corr_mat = torch.corrcoef(slot_means.T)
        offdiag  = corr_mat[~torch.eye(S, dtype=torch.bool)]
        msc      = offdiag.abs().mean().item()
    else:
        msc = 0.0

    hl_cv = (hl.std() / hl.mean().clamp_min(1)).item()
    return {
        "n_slots":        S,
        "half_life_mean": hl.mean().item(),
        "half_life_cv":   hl_cv,
        "mean_slot_corr": msc,
        "half_lives":     [int(h) for h in half_life],
        "specialized":    (hl_cv > HL_CV_THRESHOLD),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Architecture 1: CTM-like (per-slot LSTM)
# ─────────────────────────────────────────────────────────────────────────────
class SlotAttentionCTMv5(torch.nn.Module):
    def __init__(self, d_in=3, d_hidden=64, n_slots=8, horizons=PRED_HORIZONS):
        super().__init__()
        self.n_slots = n_slots
        self.d_slot  = d_hidden // n_slots
        self.horizons = horizons

        self.slot_lstms = torch.nn.ModuleList([
            torch.nn.LSTMCell(d_in, self.d_slot) for _ in range(n_slots)
        ])
        self.attn = torch.nn.Linear(d_in + self.d_slot, 1)
        self.pred_heads = torch.nn.ModuleList([
            torch.nn.Linear(d_hidden, d_in) for _ in horizons
        ])

    def forward(self, x, states=None):
        B, T, _ = x.shape
        if states is None:
            states = [(torch.zeros(B, self.d_slot, device=x.device),
                       torch.zeros(B, self.d_slot, device=x.device))
                      for _ in range(self.n_slots)]
        traj = []
        slot_log = []
        for t in range(T):
            xt = x[:, t, :]
            slot_outs, new_states = [], []
            for i, lstm in enumerate(self.slot_lstms):
                h, c = states[i]
                score = self.attn(torch.cat([xt, h], -1))
                h_new, c_new = lstm(xt, (h, c))
                slot_outs.append(h_new * score.sigmoid())
                new_states.append((h_new, c_new))
            states = new_states
            h_agg = torch.cat(slot_outs, -1)
            traj.append(h_agg)
            slot_log.append(torch.stack([s[0] for s in states], dim=1))  # (B, S, d_slot)
        traj_t = torch.stack(traj, 1)
        self._last_slot_states = torch.stack(slot_log, 0)   # (T, B, S, d_slot)
        head_outputs = [head(traj_t) for head in self.pred_heads]
        return head_outputs, traj_t

# ─────────────────────────────────────────────────────────────────────────────
# Architecture 2: RWKV-7 (delta rule, data-dependent decay)
# ─────────────────────────────────────────────────────────────────────────────
class RWKV7CellV5(torch.nn.Module):
    def __init__(self, d_in=3, d_model=64, n_heads=4, horizons=PRED_HORIZONS):
        super().__init__()
        assert d_model % n_heads == 0
        self.H, self.N, self.D = n_heads, d_model // n_heads, d_model
        self.horizons = horizons

        self.proj_in  = torch.nn.Linear(d_in, d_model)
        self.W_r      = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k      = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v      = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_b      = torch.nn.Linear(d_model, d_model, bias=True)
        self.W_z      = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_w      = torch.nn.Linear(d_model, d_model, bias=True)
        self.W_g      = torch.nn.Linear(d_model, d_model, bias=True)
        self.proj_out = torch.nn.Linear(d_model, d_model)
        self.ln_out   = torch.nn.LayerNorm(d_model)

        # W_w init: spans τ = 1..93 steps
        with torch.no_grad():
            target_taus = np.linspace(1.0, TAU_L * DHP_HI, d_model)
            w_targets   = np.exp(-1.0 / target_taus)
            logit_w     = np.log(w_targets / (1.0 - w_targets + 1e-9))
            self.W_w.bias.data = torch.tensor(logit_w, dtype=torch.float32) + \
                                 torch.randn(d_model) * 0.1

        self.pred_heads = torch.nn.ModuleList([
            torch.nn.Linear(d_model, d_in) for _ in horizons
        ])

    def init_state(self, B):
        return torch.zeros(B, self.H, self.N, self.N, device=DEVICE)

    def step(self, x_proj, S):
        B, H, N = S.shape[0], self.H, self.N
        r = self.W_r(x_proj).view(B, H, N)
        k = self.W_k(x_proj).view(B, H, N) / (N ** 0.5)
        v = self.W_v(x_proj).view(B, H, N)
        b = torch.sigmoid(self.W_b(x_proj)).view(B, H, N)
        z = torch.tanh(self.W_z(x_proj)).view(B, H, N)
        g = torch.sigmoid(self.W_g(x_proj))
        w = torch.sigmoid(self.W_w(x_proj)).view(B, H, N)

        # Extended delta rule: S_new = S * diag(w) + outer(v, k) + outer(Sz, b)
        S_new = S * w[:, :, None, :]
        Sz    = torch.einsum('bhij,bhj->bhi', S, z)
        Sz    = Sz / (Sz.norm(dim=-1, keepdim=True).clamp_min(1.0))
        S_new = S_new + torch.einsum('bhi,bhj->bhij', Sz, b)
        S_new = S_new + torch.einsum('bhi,bhj->bhij', v, k)
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
            # Log head-level S state: reshape to (B, H, N*N) then take mean across N²
            head_log.append(state.view(B, self.H, -1).mean(-1, keepdim=True))  # (B, H, 1)
        traj_t = torch.stack(traj, 1)
        # _last_slot_states shape: (T, B, H, 1) — one timeseries per head
        self._last_slot_states = torch.stack(head_log, 0)
        head_outputs = [head(traj_t) for head in self.pred_heads]
        return head_outputs, traj_t

    def get_w_eff(self, traj_data, n_steps=2000):
        """Compute mean effective decay w_eff per channel over trajectory."""
        self.eval()
        x = torch.tensor(traj_data[:n_steps], dtype=DTYPE, device=DEVICE).unsqueeze(0)
        x_proj = self.proj_in(x)  # (1, T, d_model)
        with torch.no_grad():
            w_all = torch.sigmoid(
                torch.nn.functional.linear(x_proj, self.W_w.weight, self.W_w.bias)
            )  # (1, T, d_model)
        return w_all.squeeze(0).mean(0).numpy()  # (d_model,)

# ─────────────────────────────────────────────────────────────────────────────
# Architecture 3: RWKV-6 (fixed decay)
# ─────────────────────────────────────────────────────────────────────────────
class RWKV6CellV5(torch.nn.Module):
    def __init__(self, d_in=3, d_model=64, n_heads=4, horizons=PRED_HORIZONS):
        super().__init__()
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

        # Fixed per-channel decay spanning full DHP range
        with torch.no_grad():
            target_taus = np.linspace(1.0, TAU_L * DHP_HI, d_model)
            w_fixed     = np.exp(-1.0 / target_taus)
            self.w_fixed = torch.nn.Parameter(
                torch.tensor(w_fixed, dtype=torch.float32), requires_grad=False
            )

        self.pred_heads = torch.nn.ModuleList([
            torch.nn.Linear(d_model, d_in) for _ in horizons
        ])

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

# ─────────────────────────────────────────────────────────────────────────────
# Architecture 4: Mamba-like (scalar gate, negative control)
# ─────────────────────────────────────────────────────────────────────────────
class MambaV5(torch.nn.Module):
    def __init__(self, d_in=3, d_model=64, horizons=PRED_HORIZONS):
        super().__init__()
        self.d_model  = d_model
        self.horizons = horizons
        self.proj_in  = torch.nn.Linear(d_in, d_model)
        self.W_a      = torch.nn.Linear(d_model, 1, bias=True)   # scalar gate
        self.W_bc     = torch.nn.Linear(d_model, d_model * 2)
        self.proj_out = torch.nn.Linear(d_model, d_model)
        self.pred_heads = torch.nn.ModuleList([
            torch.nn.Linear(d_model, d_in) for _ in horizons
        ])
        self._last_slot_states = None

    def init_state(self, B):
        return torch.zeros(B, self.d_model, device=DEVICE)

    def forward(self, x, state=None):
        B, T, _ = x.shape
        if state is None:
            state = self.init_state(B)
        x_proj_all = self.proj_in(x)
        traj, state_log = [], []
        for t in range(T):
            xp = x_proj_all[:, t, :]
            a  = torch.sigmoid(self.W_a(xp))    # scalar gate
            bc = self.W_bc(xp)
            b, c = bc[:, :self.d_model], bc[:, self.d_model:]
            state = a * state + (1 - a) * b
            y = torch.tanh(state) * torch.sigmoid(c)
            y = self.proj_out(y)
            traj.append(y)
            # For Mamba, log state as (B, 1, d_model) — 1 "slot"
            state_log.append(state.unsqueeze(1))  # (B, 1, d_model)
        traj_t = torch.stack(traj, 1)
        # (T, B, 1, d_model) — single slot
        self._last_slot_states = torch.stack(state_log, 0)
        head_outputs = [head(traj_t) for head in self.pred_heads]
        return head_outputs, traj_t

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: Measure HL_CV at initialization (before any training)
# ─────────────────────────────────────────────────────────────────────────────
def measure_init_hlcv(model, traj_data, n_runs=5):
    """
    Run untrained model on Lorenz trajectory, measure HL_CV.
    Baseline for computing Δ(HL_CV) = learning-induced temporal diversity.
    """
    results = []
    for run in range(n_runs):
        spec = slot_specialization(model, traj_data)
        if spec:
            results.append(spec['half_life_cv'])
    if results:
        return float(np.mean(results))
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent

print(f"\n{'='*70}", flush=True)
print(f"RWKV-7 DHP v5 — {datetime.now().isoformat()}", flush=True)
print(f"N_STEPS={N_STEPS}, LR={LR}, N_SEEDS={N_SEEDS}", flush=True)
print(f"v5: Equal horizon weights | Δ(HL_CV) DHP signal | S-state timescales", flush=True)
print(f"{'='*70}\n", flush=True)

traj = lorenz_trajectory(n=8000, dt=DT)
assert not np.isnan(traj).any()
print(f"Lorenz trajectory: {len(traj)} steps, dt={DT}, no NaN ✓", flush=True)

MODELS_SPEC = {
    "CTM-like v5":    lambda: SlotAttentionCTMv5(D_IN, D_HIDDEN, n_slots=8),
    "RWKV-7 v5":      lambda: RWKV7CellV5(D_IN, D_HIDDEN, n_heads=4),
    "RWKV-6 v5":      lambda: RWKV6CellV5(D_IN, D_HIDDEN, n_heads=4),
    "Mamba v5":       lambda: MambaV5(D_IN, D_HIDDEN),
}

all_results = {}

for name, model_fn in MODELS_SPEC.items():
    print(f"\n{'='*70}", flush=True)
    print(f"Architecture: {name}", flush=True)
    print(f"{'='*70}", flush=True)

    seed_hlcv_post  = []
    seed_hlcv_init  = []
    seed_final_loss = []
    seed_ok         = []

    for seed in range(N_SEEDS):
        print(f"\n  --- Seed {seed} ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model_fn()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if seed == 0:
            print(f"  {n_params:,} trainable params", flush=True)

        # FIX 1: Measure HL_CV before training
        init_hlcv = measure_init_hlcv(model, traj, n_runs=3)
        print(f"  HL_CV (init):  {init_hlcv:.3f}", flush=True)

        # Train
        losses, ok, final_loss = train_model(model, traj, seed=seed)

        # Measure HL_CV after training
        spec = slot_specialization(model, traj)
        post_hlcv = spec['half_life_cv'] if spec else None
        print(f"  HL_CV (post):  {post_hlcv:.3f}", flush=True)
        if init_hlcv is not None and post_hlcv is not None:
            delta = post_hlcv - init_hlcv
            print(f"  Δ(HL_CV):      {delta:+.3f} ← {'DHP SIGNAL ✓' if delta > 0.05 else 'no signal'}", flush=True)

        if init_hlcv is not None: seed_hlcv_init.append(init_hlcv)
        if post_hlcv is not None: seed_hlcv_post.append(post_hlcv)
        seed_final_loss.append(final_loss)
        seed_ok.append(ok)

        if spec:
            print(f"  HL mean: {spec['half_life_mean']:.1f} | inter-corr: {spec['mean_slot_corr']:.4f}", flush=True)
            print(f"  half-lives: {spec['half_lives']}", flush=True)

    # Aggregate across seeds
    mean_init = float(np.mean(seed_hlcv_init)) if seed_hlcv_init else None
    mean_post = float(np.mean(seed_hlcv_post)) if seed_hlcv_post else None
    std_post  = float(np.std(seed_hlcv_post))  if len(seed_hlcv_post) > 1 else 0.0
    delta_hlcv = (mean_post - mean_init) if (mean_post and mean_init) else None
    dhp_by_delta = (delta_hlcv is not None and delta_hlcv > 0.05)
    dhp_absolute = (mean_post is not None and mean_post > HL_CV_THRESHOLD)

    print(f"\n  === {name} AGGREGATE ({N_SEEDS} seeds) ===", flush=True)
    init_s = f"{mean_init:.3f}" if mean_init is not None else "N/A"
    print(f"  HL_CV init:  {init_s}", flush=True)
    print(f"  HL_CV post:  {mean_post:.3f} ± {std_post:.3f}" if mean_post is not None else "  HL_CV post:  N/A", flush=True)
    print(f"  Δ(HL_CV):    {delta_hlcv:+.3f}" if delta_hlcv is not None else "  Δ(HL_CV):    N/A", flush=True)
    print(f"  DHP (Δ>0.05): {'✓' if dhp_by_delta else '✗'}", flush=True)
    print(f"  DHP (abs>0.3): {'✓' if dhp_absolute else '✗'}", flush=True)
    print(f"  Training OK:  {sum(seed_ok)}/{N_SEEDS}", flush=True)

    all_results[name] = {
        "hlcv_init_mean": mean_init,
        "hlcv_post_mean": mean_post,
        "hlcv_post_std":  std_post,
        "delta_hlcv":     delta_hlcv,
        "dhp_by_delta":   dhp_by_delta,
        "dhp_absolute":   dhp_absolute,
        "n_ok":           sum(seed_ok),
        "n_seeds":        N_SEEDS,
        "final_loss_mean": float(np.mean(seed_final_loss)),
    }

# ── Final comparative summary ─────────────────────────────────────────────────
print(f"\n\n{'='*70}", flush=True)
print(f"RWKV-7 DHP v5 — COMPARATIVE SUMMARY", flush=True)
print(f"{'='*70}", flush=True)
print(f"{'Architecture':<22} {'HL_CV_init':>10} {'HL_CV_post':>11} {'Δ(HL_CV)':>9} {'DHP(Δ)':>7} {'DHP(abs)':>9}", flush=True)
print(f"{'-'*70}", flush=True)
for name, r in all_results.items():
    init_s = f"{r['hlcv_init_mean']:.3f}" if r['hlcv_init_mean'] is not None else "  N/A"
    post_s = f"{r['hlcv_post_mean']:.3f}" if r['hlcv_post_mean'] is not None else "  N/A"
    delt_s = f"{r['delta_hlcv']:+.3f}" if r['delta_hlcv'] is not None else "  N/A"
    print(f"  {name:<20} {init_s:>10} {post_s:>11} {delt_s:>9} "
          f"{'✓' if r['dhp_by_delta'] else '✗':>7} {'✓' if r['dhp_absolute'] else '✗':>9}", flush=True)

print(f"\n  DHP hypothesis: RWKV-7 should show LARGEST Δ(HL_CV)", flush=True)
print(f"  RWKV-6: Δ≈0 (fixed decay → diversity is static, can't be learned)", flush=True)
print(f"  Mamba:  Small Δ (scalar gate → all channels share same timescale)", flush=True)
print(f"  CTM:    Moderate Δ (LSTM gates per slot but uniform-ish)", flush=True)

# Save results
out = {
    "version": "v5",
    "timestamp": datetime.now().isoformat(),
    "tau_L": TAU_L,
    "dt": DT,
    "seq_len": SEQ_LEN,
    "pred_horizons": PRED_HORIZONS,
    "n_steps": N_STEPS,
    "n_seeds": N_SEEDS,
    "equal_horizon_weights": True,
    "dhp_by_delta_threshold": 0.05,
    "results": all_results,
}
out_path = HERE / "rwkv7_dhp_v5.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved → {out_path}", flush=True)
print("Done.", flush=True)
