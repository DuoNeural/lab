#!/usr/bin/env python3
"""
run_cdm_v8a_pod.py — CDM V8-A on vast.ai pod (5070 Ti).
Fixed β=-0.3 temporal lead routing. No learned β.

CDM V7 learned β converged to -0.3 to -0.5. V8-A tests: if -0.3 is optimal,
skipping learning it saves optimizer bandwidth → better early convergence.

Architecture: V7 (HORN + Kuramoto + β routing), β frozen at -0.3.
DuoNeural / Archon — 2026-06-18
"""
import sys
sys.path.insert(0, '/workspace')

import os, json, math, time
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime, timezone
from torch.utils.data import DataLoader

from cdm_model_v7 import CDMLanguageModelV7, CDMConfigV7
import cdm_train_v7 as tr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("/workspace/cdm_v8a_full")
LOG_PATH = Path("/workspace/cdm_v8a_training.log")

CFG = CDMConfigV7(
    vocab_size=50257, d_model=384, n_layers=8, n_heads=8, n_kv_heads=4,
    d_ff=1024, K=16, max_len=512, d_osc=8,
    beta_init=-0.3,      # fixed at V7 converged value
    entropy_reg=0.02, lbl_coeff=0.01, alpha_init=0.0, eps=1e-6,
)

BATCH_SIZE = 8
SEQ_LEN    = 256
LR         = 3e-4
STEPS      = 30000
WARMUP     = 500
VAL_EVERY  = 500
SAVE_EVERY = 2500

def ts(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def log(msg):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f: f.write(line + "\n")

def get_lr(step):
    if step < WARMUP: return LR * step / WARMUP
    t = (step - WARMUP) / (STEPS - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * t))

def evaluate(model, val_loader, max_batches=50):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches: break
            x = batch[:, :-1].to(DEVICE)
            y = batch[:, 1:].to(DEVICE)
            ce, _, _ = model.forward_loss(x)
            losses.append(ce.item())
    return sum(losses) / len(losses)

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "best").mkdir(exist_ok=True)

log("=" * 60)
log("CDM V8-A: Fixed β=-0.3 temporal lead routing")
log("=" * 60)

train_ds, val_ds = tr.load_tinystories(SEQ_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, drop_last=True)

model = CDMLanguageModelV7(CFG).to(DEVICE)
log(f"Model: {model.param_count:,} params ({model.param_count/1e6:.1f}M)")

# Freeze β in every layer — V8-A key modification
n_frozen = 0
for name, param in model.named_parameters():
    if "beta" in name:
        param.requires_grad_(False)
        n_frozen += 1
log(f"β frozen across {n_frozen} layers at {CFG.beta_init:.3f} (no gradient)")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"Trainable params: {trainable:,} (β excluded)")

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, betas=(0.9, 0.95), weight_decay=0.1,
)

cfg_dict = {**CFG.__dict__, "n_params": model.param_count,
            "architecture": "CDM_V8A_FixedBeta", "beta_frozen": True, "device": DEVICE}
with open(OUT_DIR / "config.json", "w") as f: json.dump(cfg_dict, f, indent=2)

best_val = float("inf")
train_iter = iter(train_loader)
step = 0
t0 = time.time()

log(f"Training {STEPS} steps  batch={BATCH_SIZE}  seq={SEQ_LEN}  device={DEVICE}")
if DEVICE == "cuda": log(f"GPU: {torch.cuda.get_device_name(0)}")

while step < STEPS:
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    x = batch[:, :-1].to(DEVICE)
    y = batch[:, 1:].to(DEVICE)

    lr = get_lr(step)
    for pg in optimizer.param_groups: pg["lr"] = lr

    optimizer.zero_grad()
    ce, lbl, total = model.forward_loss(x)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    step += 1

    if step % 50 == 0:
        elapsed = time.time() - t0
        tok_s = step * BATCH_SIZE * SEQ_LEN / elapsed
        log(f"  step {step:6d}/{STEPS} | ce={ce.item():.4f} aux={-lbl.item():.4f} | lr={lr:.2e} | {tok_s:.0f} tok/s")

    if step % VAL_EVERY == 0:
        val_ce = evaluate(model, val_loader)
        log(f"  Val CE: {val_ce:.4f} (step {step})  [V7: 1.6251 | HORN: 1.5818 | Kuramoto: 1.5819]")

        if val_ce < best_val:
            best_val = val_ce
            torch.save({"step": step, "model_state": model.state_dict(),
                        "val_loss": val_ce, "config": cfg_dict,
                        "beta_frozen": True},
                       OUT_DIR / "best" / "model.pt")
            log(f"  NEW BEST: {val_ce:.4f} @ step {step}")

        # Log β values to confirm frozen
        betas = [p.item() for n, p in model.named_parameters() if "beta" in n]
        if betas: log(f"  β values (all layers): {[f'{b:.3f}' for b in betas]}")

        model.train()

    if step % SAVE_EVERY == 0:
        cd = OUT_DIR / f"step_{step:06d}"
        cd.mkdir(exist_ok=True)
        torch.save({"step": step, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": best_val, "config": cfg_dict},
                   cd / "checkpoint.pt")
        log(f"  Checkpoint: {cd}")

fd = OUT_DIR / "final"
fd.mkdir(exist_ok=True)
torch.save({"step": step, "model_state": model.state_dict(),
            "val_loss": best_val, "config": cfg_dict}, fd / "model.pt")

log(f"\nDone. Best val CE: {best_val:.4f}")
log(f"Comparison: V7=1.6251 | HORN=1.5818 | Kuramoto=1.5819")
if best_val < 1.5818:
    log("*** V8-A BEATS HORN — fixed β wins! ***")
elif best_val < 1.6251:
    log("*** V8-A beats V7 but not HORN — β overhead was real ***")
else:
    log("*** V8-A no improvement over V7 — fixed β not sufficient ***")
print(f"CDM_V8A_FINAL: {best_val:.4f}")
