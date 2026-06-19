#!/usr/bin/env python3
"""
run_horn_wikitext512_pod.py — HORN 37M on WikiText-103 seq=512.

This is the CONTROL ARM for CDM V9-B's long-context hypothesis test.
V9-B (dual-timescale CDM) runs WikiText-103 at seq=512 to test whether
the slow HORN + fast EMA slot combination shows advantage at longer context.

Without this baseline, we can't claim V9-B "beats HORN at long context" —
we'd only know it beats HORN on TinyStories seq=256 (or not).

Architecture: HORN 37M (d_model=384, n_layers=8) — same dims as V9-A/V9-B.
Dataset: WikiText-103, seq=512 — same as V9-B.
Steps: 10000 — same as V9-B (budget-matched for direct comparison).

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
OUT_DIR  = Path("/workspace/horn_wikitext512")
LOG_PATH = Path("/workspace/horn_wikitext512_training.log")

def ts(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def log(msg):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f: f.write(line + "\n")

from cdm_model_v6_horn import CDMLanguageModelV6HORN
from cdm_model_v3 import CDMConfigV3

CFG = CDMConfigV3(
    vocab_size=50257, d_model=384, n_layers=8, n_heads=8, n_kv_heads=4,
    d_ff=1024, K=16, max_len=1024,
    entropy_reg=0.02, lbl_coeff=0.01, alpha_init=0.0,
)

BATCH_SIZE = 8
SEQ_LEN    = 512
LR         = 3e-4
STEPS      = 10000   # matched to V9-B for direct comparison
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
    log(f"  tokenizing {split} (streaming, article-by-article)...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    chunks = []
    buf = []
    n_articles = 0
    for ex in ds:
        if len(ex["text"].strip()) == 0:
            continue
        ids = tok.encode(ex["text"])
        buf.extend(ids)
        n_articles += 1
        while len(buf) >= SEQ_LEN + 1:
            chunks.append(buf[:SEQ_LEN + 1])
            buf = buf[SEQ_LEN + 1:]
        if n_articles % 50000 == 0:
            log(f"    ...{n_articles} articles, {len(chunks)} chunks so far")
    t = torch.tensor(chunks, dtype=torch.long)
    torch.save(t, fp)
    log(f"  {split}: {len(chunks)} chunks from {n_articles} articles saved")
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

model = CDMLanguageModelV6HORN(CFG).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
log(f"HORN 37M WikiText-512: {n_params/1e6:.1f}M params | K=16 seq={SEQ_LEN}")
log(f"Role: baseline for V9-B long-context hypothesis test")
log(f"Device: {DEVICE}")
if DEVICE == "cuda": log(f"GPU: {torch.cuda.get_device_name(0)}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

cfg_dict = {**CFG.__dict__, "n_params": n_params, "architecture": "HORN_37M_WikiText512",
            "dataset": "WikiText-103", "seq_len": SEQ_LEN,
            "role": "V9-B control arm (long-context HORN baseline)"}
with open(OUT_DIR / "config.json", "w") as f: json.dump(cfg_dict, f, indent=2)

# ── Val ───────────────────────────────────────────────────────────────────────
def evaluate(model, val_loader, max_batches=20):
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

log(f"Training {STEPS} steps | batch={BATCH_SIZE} seq={SEQ_LEN} | WikiText-103")
log(f"Hypothesis: V9-B (dual-timescale) should beat this HORN baseline at seq=512")
log(f"HORN_TinyStories_37M=1.5818 | This run tests long-context advantage")

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
        log(f"  Val CE: {val_ce:.4f} (step {step}) [WikiText-103 seq={SEQ_LEN}]")

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
log(f"Compare against V9-B (same steps/dataset/seq) to test dual-timescale advantage")
print(f"HORN_WIKITEXT512_FINAL: {best_val:.4f}")
