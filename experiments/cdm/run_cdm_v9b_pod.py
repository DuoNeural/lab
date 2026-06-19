#!/usr/bin/env python3
"""
run_cdm_v9b_pod.py — CDM V9-B: Dual Timescale, long-context test on GPU1.

37M params, seq=512, WikiText-103. Tests the core hypothesis: does dual
timescale (CLS-style fast/slow pool) help at longer context?

V9-A (GPU0, seq=256, TinyStories) does the head-to-head CE comparison.
V9-B (GPU1, seq=512, WikiText-103) tests context-length advantage.

Launch only after V9-A step 50+ clears without NaN.
Watch for: slow pool routing mass > 0 (cannibalization check).

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
OUT_DIR  = Path("/workspace/cdm_v9b_full")
LOG_PATH = Path("/workspace/cdm_v9b_training.log")

def ts(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def log(msg):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f: f.write(line + "\n")

from cdm_model_v9 import CDMLanguageModelV9, CDMConfigV9

CFG = CDMConfigV9(
    vocab_size=50257, d_model=384, n_layers=8, n_heads=8, n_kv_heads=4,
    d_ff=1024, K=16, K_slow=8, max_len=1024, d_osc=8,
    entropy_reg=0.02, lbl_coeff=0.01,
    lbl_coeff_slow=0.01, lbl_coeff_fast=0.01,
    alpha_init=0.0, v_gate_init=-2.0,
    gamma_max=1.0, omega_max=2.0, dt_scale=0.25,
)

BATCH_SIZE = 8
SEQ_LEN    = 512
LR         = 3e-4
STEPS      = 10000   # budget-aware: ~11.5s/step at seq=256, ~23s/step at seq=512
WARMUP     = 200
VAL_EVERY  = 250
SAVE_EVERY = 2500

def get_lr(step):
    if step < WARMUP: return LR * step / WARMUP
    t = (step - WARMUP) / (STEPS - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * t))

# ── Data — WikiText-103 ───────────────────────────────────────────────────────
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from torch.utils.data import TensorDataset

log("Loading WikiText-103...")
tok = GPT2TokenizerFast.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
cache = Path("/workspace/wikitext103_cache")
cache.mkdir(exist_ok=True)

def make_wikitext_split(split):
    fp = cache / f"{split}_{SEQ_LEN}.pt"
    if fp.exists():
        log(f"  loading cached {split}...")
        return torch.load(fp)
    log(f"  tokenizing {split}...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    # concatenate all text, then chunk
    full_text = "\n\n".join(ex["text"] for ex in ds if len(ex["text"].strip()) > 0)
    ids = tok.encode(full_text)
    chunks = []
    for i in range(0, len(ids) - SEQ_LEN, SEQ_LEN):
        chunks.append(ids[i:i + SEQ_LEN + 1])
    t = torch.tensor(chunks, dtype=torch.long)
    torch.save(t, fp)
    log(f"  {split}: {len(chunks)} chunks saved")
    return t

train_ds = TensorDataset(make_wikitext_split("train"))
val_ds   = TensorDataset(make_wikitext_split("validation"))
log(f"Dataset: {len(train_ds)} train / {len(val_ds)} val | seq={SEQ_LEN}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, drop_last=True)

# ── Model ─────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "best").mkdir(exist_ok=True)

model = CDMLanguageModelV9(CFG).to(DEVICE)
n_params = model.param_count
log(f"CDM V9-B: {n_params/1e6:.1f}M params | K_slow={CFG.K_slow} K_fast={CFG.K_fast} | seq={SEQ_LEN}")
log(f"Device: {DEVICE}")
if DEVICE == "cuda": log(f"GPU: {torch.cuda.get_device_name(0)}")

log("Compiling model with torch.compile...")
model = torch.compile(model, mode="reduce-overhead")
log("Compile done.")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

cfg_dict = {**CFG.__dict__, "n_params": n_params, "architecture": "CDM_V9B_DualTimescale_LongCtx",
            "device": DEVICE, "K_slow": CFG.K_slow, "K_fast": CFG.K_fast,
            "dataset": "WikiText-103", "seq_len": SEQ_LEN}
with open(OUT_DIR / "config.json", "w") as f: json.dump(cfg_dict, f, indent=2)

# ── Val ───────────────────────────────────────────────────────────────────────
def val_with_probe(model, val_loader, max_batches=20):
    model.eval()
    losses, slow_mass_list, fast_mass_list = [], [], []
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

log(f"Training {STEPS} steps | batch={BATCH_SIZE} seq={SEQ_LEN} | dataset=WikiText-103")
log(f"Hypothesis: dual timescale shows larger advantage at longer context")

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
        log(f"  Val CE: {val_ce:.4f} (step {step}) [WikiText-103 seq={SEQ_LEN}]")
        for line in probe_lines[:4]:
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
log(f"V9-B (seq=512, WikiText-103) vs V9-A (seq=256, TinyStories)")
print(f"CDM_V9B_FINAL: {best_val:.4f}")
