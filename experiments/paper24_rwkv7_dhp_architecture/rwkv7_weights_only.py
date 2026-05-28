"""
RWKV-7 weights-only rerun — just trains RWKV-7 v4 and saves weights
for post-training decay analysis (analyze_rwkv7_decay.py).

Reuses all classes from rwkv7_dhp_v4.py.
Archon — DuoNeural — 2026-05-27
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Import everything from v4
import torch, numpy as np, json, math
from pathlib import Path
from datetime import datetime

# Execute v4 config + classes only (skip the main block)
v4_src = open(Path(__file__).parent / "rwkv7_dhp_v4.py").read()
exec(v4_src.split("# ─────────────────────────────────────────────────────────────────────────────\n# Main")[0])

HERE = Path(__file__).parent
print(f"\nRWKV-7 weights-only run — {datetime.now().isoformat()}", flush=True)
print(f"SEQ_LEN={SEQ_LEN}, N_STEPS={N_STEPS}, T_FINAL={T_FINAL}", flush=True)

traj = lorenz_trajectory(n=8000, dt=DT)
assert not np.isnan(traj).any()

model = RWKV7CellV4(D_IN, D_HIDDEN, n_heads=4)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"RWKV-7 v4 ({n_params:,} params)", flush=True)

losses, ok = train_model(model, traj)
tau_star, norm_losses = measure_tau_star_v4(model, traj)
spec = slot_specialization(model, traj)

print(f"\n  Training OK = {'✓' if ok else '✗'}", flush=True)
print(f"  τ* = {tau_star:.1f} (τ*/τ_L = {tau_star/TAU_L:.4f})", flush=True)
if spec:
    print(f"  HL CV = {spec['half_life_cv']:.3f}", flush=True)
    print(f"  Half-lives = {spec['half_lives']}", flush=True)

# Save weights
weights_path = HERE / "weights_v4_RWKV-7_v4_delta_rule.pt"
torch.save(model.state_dict(), weights_path)
print(f"\nWeights saved → {weights_path}", flush=True)
print("Now run: python3 analyze_rwkv7_decay.py", flush=True)
