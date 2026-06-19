#!/usr/bin/env python3
"""
run_horn_85m_corrected_pod.py — HORN 85M, corrected batch_size=8.

Previous HORN 85M run (5070Ti-A, June 18) used batch_size=4 due to a pod
migration artifact — half the token budget of V5/Kuramoto 85M (30.7M vs 61.4M
tokens). CE=1.7916 is not comparable to V5=1.4718 or Kuramoto-85M=1.4802.

This run fixes it: batch_size=8, 30k steps. Same architecture as V5 dims
(d_model=512, n_layers=12) so comparison with V5/Kuramoto-85M is clean.

Paper §3.2 cannot be published without this number.
Targets: beat V5 (1.4718) — or at minimum bracket [Kuramoto-85M, V5].

DuoNeural / Archon — 2026-06-19
"""
import sys
sys.path.insert(0, '/workspace')

import json, math, time
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime, timezone
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR  = Path("/workspace/horn_85m_corrected")
LOG_PATH = Path("/workspace/horn_85m_corrected_training.log")

def ts(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def log(msg):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f: f.write(line + "\n")

from cdm_model_v6_horn import CDMLanguageModelV6HORN
from cdm_model_v3 import CDMConfigV3

CFG = CDMConfigV3(
    vocab_size=50257, d_model=512, n_layers=12, n_heads=8, n_kv_heads=4,
    d_ff=2048, K=16, max_len=512,
    entropy_reg=0.02, lbl_coeff=0.01, alpha_init=0.0,
)

BATCH_SIZE = 8       # THE FIX — was 4 in the confounded run
SEQ_LEN    = 256
LR         = 3e-4
STEPS      = 30000
WARMUP     = 500
VAL_EVERY  = 500
SAVE_EVERY = 5000

def get_lr(step):
    if step < WARMUP: return LR * step / WARMUP
    t = (step - WARMUP) / (STEPS - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * t))

# ── Data — TinyStories ────────────────────────────────────────────────────────
from datasets import load_dataset
from transformers import GPT2TokenizerFast

log("Loading TinyStories...")
tok = GPT2TokenizerFast.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
cache = Path("/workspace/tinystories_cache")
cache.mkdir(exist_ok=True)

def make_split(split):
    fp = cache / f"{split}_{SEQ_LEN}.pt"
    if fp.exists():
        log(f"  loading cached {split}...")
        data = torch.load(fp)
        return data
    log(f"  tokenizing {split}...")
    ds = load_dataset("roneneldan/TinyStories", split=split)
    chunks = []
    for ex in ds:
        ids = tok.encode(ex["text"])
        for i in range(0, len(ids) - SEQ_LEN, SEQ_LEN):
            chunks.append(ids[i:i + SEQ_LEN + 1])
    t = torch.tensor(chunks, dtype=torch.long)
    torch.save(t, fp)
    log(f"  {split}: {len(chunks)} chunks saved")
    return t

train_ds = TensorDataset(make_split("train"))
val_ds   = TensorDataset(make_split("validation"))
log(f"Dataset: {len(train_ds)} train / {len(val_ds)} val | seq={SEQ_LEN}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, drop_last=True)

# ── Model ─────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "best").mkdir(exist_ok=True)

model = CDMLanguageModelV6HORN(CFG).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
log(f"HORN 85M CORRECTED: {n_params/1e6:.1f}M params | d_model=512 n_layers=12 K=16")
log(f"FIX: batch_size={BATCH_SIZE} (was 4 in confounded run — 2x token budget)")
log(f"Device: {DEVICE}")
if DEVICE == "cuda": log(f"GPU: {torch.cuda.get_device_name(0)}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

cfg_dict = {**CFG.__dict__, "n_params": n_params, "architecture": "HORN_85M_corrected",
            "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN, "note": "batch=4 confound fixed"}
with open(OUT_DIR / "config.json", "w") as f: json.dump(cfg_dict, f, indent=2)

# ── Val ───────────────────────────────────────────────────────────────────────
def evaluate(model, val_loader, max_batches=50):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches: break
            if isinstance(batch, (list, tuple)): batch = batch[0]
            x = batch[:, :-1].to(DEVICE)
            y = batch[:, 1:].to(DEVICE)
            logits, aux = model(x)
            ce = F.cross_entropy(logits.reshape(-1, CFG.vocab_size), y.reshape(-1))
            losses.append(ce.item())
    model.train()
    return sum(losses) / len(losses)

# ── Train ─────────────────────────────────────────────────────────────────────
best_val = float("inf")
train_iter = iter(train_loader)
step = 0
t0 = time.time()

log(f"Training {STEPS} steps | batch={BATCH_SIZE} seq={SEQ_LEN} | TinyStories")
log(f"Targets: V5_softmax=1.4718 | V5_Kuramoto=1.4802 | HORN_37M=1.5818")
log(f"Previous confounded HORN_85M=1.7916 (batch=4, not comparable)")

while step < STEPS:
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    if isinstance(batch, (list, tuple)): batch = batch[0]
    x = batch[:, :-1].to(DEVICE)
    y = batch[:, 1:].to(DEVICE)

    lr = get_lr(step)
    for pg in optimizer.param_groups: pg["lr"] = lr

    optimizer.zero_grad()
    logits, aux = model(x)
    ce = F.cross_entropy(logits.reshape(-1, CFG.vocab_size), y.reshape(-1))
    total = ce + aux
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
        val_ce = evaluate(model, val_loader)
        log(f"  Val CE: {val_ce:.4f} (step {step})  "
            f"[V5=1.4718 | V5-Kur=1.4802 | HORN37M=1.5818]")

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
log(f"Compare: V5_softmax=1.4718 | V5_Kuramoto=1.4802 | HORN_37M=1.5818")
if best_val < 1.4718:
    log("*** HORN 85M BEATS V5 SOFTMAX — HORN dynamics scale better! ***")
elif best_val < 1.4802:
    log("*** HORN 85M between V5 and V5-Kuramoto — second-order wins at scale ***")
elif best_val < 1.5818:
    log("*** HORN 85M beats HORN 37M but not V5 — scale helps both ***")
else:
    log("*** HORN 85M trails HORN 37M — unexpected, check config ***")

print(f"HORN_85M_CORRECTED_FINAL: {best_val:.4f}")
