#!/usr/bin/env python3
"""
run_cdm_v9a_pod.py — CDM V9-A: Dual Timescale (parallel pool) on vast.ai pod.

First run: TinyStories 37M, seq=256. Sanity check before long-context.
Watch for cannibalization: if slow_lbl >> fast_lbl → slow pool being starved.
Compare target: HORN 1.5818 | Kuramoto 1.5819 | V7 1.6251

DuoNeural / Archon + Jesse — 2026-06-19
"""
import sys
sys.path.insert(0, '/workspace')

import os, json, math, time
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime, timezone
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR  = Path("/workspace/cdm_v9a_full")
LOG_PATH = Path("/workspace/cdm_v9a_training.log")

def ts(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def log(msg):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f: f.write(line + "\n")

# ── Import model ──────────────────────────────────────────────────────────────
from cdm_model_v9 import CDMLanguageModelV9, CDMConfigV9

CFG = CDMConfigV9(
    vocab_size=50257, d_model=384, n_layers=8, n_heads=8, n_kv_heads=4,
    d_ff=1024, K=16, K_slow=8, max_len=512, d_osc=8,
    entropy_reg=0.02, lbl_coeff=0.01,
    lbl_coeff_slow=0.01, lbl_coeff_fast=0.01,
    alpha_init=0.0, v_gate_init=-2.0,
    gamma_max=1.0, omega_max=2.0, dt_scale=0.25,
)

BATCH_SIZE = 8
SEQ_LEN    = 256
LR         = 3e-4
STEPS      = 10000   # budget-aware: 39.5M at ~177 tok/s ≈ 32h on 5090 (~$11)
WARMUP     = 200
VAL_EVERY  = 250     # more frequent feedback given longer step time
SAVE_EVERY = 2500

def get_lr(step):
    if step < WARMUP: return LR * step / WARMUP
    t = (step - WARMUP) / (STEPS - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * t))

# ── Data ──────────────────────────────────────────────────────────────────────
# reuse cdm_train_v7 tinystories loader if available, else inline
try:
    import cdm_train_v7 as tr
    train_ds, val_ds = tr.load_tinystories(SEQ_LEN)
    log("Loaded TinyStories via cdm_train_v7")
except ImportError:
    try:
        import cdm_train_kuramoto as tr
        train_ds, val_ds = tr.load_tinystories(SEQ_LEN)
        log("Loaded TinyStories via cdm_train_kuramoto")
    except ImportError:
        # inline loader as fallback
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast
        import torch
        from torch.utils.data import TensorDataset

        log("Inline TinyStories load...")
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        cache = Path("/workspace/tinystories_cache")
        cache.mkdir(exist_ok=True)

        def make_split(split):
            fp = cache / f"{split}_{SEQ_LEN}.pt"
            if fp.exists():
                return torch.load(fp)
            ds = load_dataset("roneneldan/TinyStories", split=split)
            chunks = []
            for ex in ds:
                ids = tok.encode(ex["text"])
                for i in range(0, len(ids) - SEQ_LEN, SEQ_LEN):
                    chunks.append(ids[i:i + SEQ_LEN + 1])
            t = torch.tensor(chunks, dtype=torch.long)
            torch.save(t, fp)
            return t

        train_ds = torch.utils.data.TensorDataset(make_split("train"))
        val_ds   = torch.utils.data.TensorDataset(make_split("validation"))
        log(f"Inline load done: {len(train_ds)} train / {len(val_ds)} val")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, drop_last=True)

# ── Model ─────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "best").mkdir(exist_ok=True)

model = CDMLanguageModelV9(CFG).to(DEVICE)
n_params = model.param_count
log(f"CDM V9-A: {n_params/1e6:.1f}M params | K_slow={CFG.K_slow} K_fast={CFG.K_fast}")
log(f"Device: {DEVICE}")
if DEVICE == "cuda": log(f"GPU: {torch.cuda.get_device_name(0)}")

log("Compiling model with torch.compile (mode=reduce-overhead)...")
model = torch.compile(model, mode="reduce-overhead")
log("Compile done.")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

cfg_dict = {**CFG.__dict__, "n_params": n_params, "architecture": "CDM_V9A_DualTimescale",
            "device": DEVICE, "K_slow": CFG.K_slow, "K_fast": CFG.K_fast}
with open(OUT_DIR / "config.json", "w") as f: json.dump(cfg_dict, f, indent=2)

# ── Val ───────────────────────────────────────────────────────────────────────
def evaluate(model, val_loader, max_batches=50):
    model.eval()
    losses = []
    slow_util, fast_util = [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches: break
            if isinstance(batch, (list, tuple)): batch = batch[0]
            x = batch[:, :-1].to(DEVICE)
            logits, _, _, routes = model.blocks[0].cdm._scan(
                model.blocks[0].norm_cdm(model.embed(x))
            )
            ce, _, _ = model.forward_loss(x)
            losses.append(ce.item())
    return sum(losses) / len(losses)

def val_with_probe(model, val_loader, max_batches=20):
    """Run eval + collect cannibalization diagnostic."""
    model.eval()
    losses, slow_mass, fast_mass = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches: break
            if isinstance(batch, (list, tuple)): batch = batch[0]
            x = batch[:, :-1].to(DEVICE)
            ce, aux, _ = model.forward_loss(x)
            losses.append(ce.item())

        # probe on _orig_mod (uncompiled eager) to avoid Triton recompilation
        batch = next(iter(val_loader))
        if isinstance(batch, (list, tuple)): batch = batch[0]
        x = batch[:, :-1].to(DEVICE)
        probe_model = getattr(model, '_orig_mod', model)
        _ = probe_model(x, collect_probe=True)

    val_ce = sum(losses) / len(losses)
    probe_lines = []
    probe_model = getattr(model, '_orig_mod', model)
    for i, block in enumerate(probe_model.blocks):
        p = block.cdm.last_probe
        if p:
            probe_lines.append(
                f"    L{i}: coupling={p.get('coupling_mean',0):.3f}  "
                f"winner_align={p.get('winner_alignment_mean',0):.3f}"
            )
    return val_ce, probe_lines

# ── Train ─────────────────────────────────────────────────────────────────────
best_val = float("inf")
train_iter = iter(train_loader)
step = 0
t0 = time.time()

log(f"Training {STEPS} steps | batch={BATCH_SIZE} seq={SEQ_LEN}")
log(f"Targets: HORN=1.5818 | Kuramoto=1.5819 | V7=1.6251")
log(f"Watch: slow vs fast routing mass — cannibalization = slow mass → 0")

while step < STEPS:
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    if isinstance(batch, (list, tuple)): batch = batch[0]
    x = batch[:, :-1].to(DEVICE)

    lr = get_lr(step)
    for pg in optimizer.param_groups: pg["lr"] = lr

    optimizer.zero_grad()
    ce, aux, total = model.forward_loss(x)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    step += 1

    if step % 50 == 0:
        elapsed = time.time() - t0
        tok_s = step * BATCH_SIZE * SEQ_LEN / elapsed
        log(f"  step {step:6d}/{STEPS} | ce={ce.item():.4f} aux={aux.item():.4f} "
            f"| lr={lr:.2e} | {tok_s:.0f} tok/s")

    if step % VAL_EVERY == 0:
        val_ce, probe_lines = val_with_probe(model, val_loader)
        log(f"  Val CE: {val_ce:.4f} (step {step})  "
            f"[HORN: 1.5818 | Kur: 1.5819 | V7: 1.6251]")
        for line in probe_lines[:4]:  # first 4 layers
            log(line)

        if val_ce < best_val:
            best_val = val_ce
            torch.save({"step": step, "model_state": model.state_dict(),
                        "val_loss": val_ce, "config": cfg_dict},
                       OUT_DIR / "best" / "model.pt")
            log(f"  NEW BEST: {val_ce:.4f} @ step {step}")

        model.train()

    if step % SAVE_EVERY == 0:
        cd = OUT_DIR / f"step_{step:06d}"
        cd.mkdir(exist_ok=True)
        torch.save({"step": step, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": best_val, "config": cfg_dict},
                   cd / "checkpoint.pt")
        log(f"  Checkpoint: {cd}")

# ── Final ─────────────────────────────────────────────────────────────────────
fd = OUT_DIR / "final"
fd.mkdir(exist_ok=True)
torch.save({"step": step, "model_state": model.state_dict(),
            "val_loss": best_val, "config": cfg_dict}, fd / "model.pt")

log(f"\nDone. Best val CE: {best_val:.4f}")
log(f"Comparison: HORN=1.5818 | Kuramoto=1.5819 | V7=1.6251 | V8-A=TBD")
if best_val < 1.5818:
    log("*** V9-A BEATS HORN AND KURAMOTO — dual timescale wins at 37M! ***")
elif best_val < 1.5819:
    log("*** V9-A ties Kuramoto, beats HORN — respectable ***")
elif best_val < 1.6251:
    log("*** V9-A beats V7 but not HORN/Kuramoto — partial benefit ***")
else:
    log("*** V9-A no improvement — architecture needs long context to shine ***")

print(f"CDM_V9A_FINAL: {best_val:.4f}")
