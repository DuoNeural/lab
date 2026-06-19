#!/usr/bin/env python3
"""
cdm_train_v8b.py — CDM V8-B: HORN + Kuramoto with per-slot β_k (K=16 scalars/layer).
DuoNeural / Archon — 2026-06-17

CDM V7 used a scalar β per layer, converging to layer-mean ≈ -0.30 at step 10000.
V7 also showed β magnitude correlates with oscillatory engagement (most underdamped → most negative).

V8-B: β_k per slot. Each slot can develop its own temporal lead magnitude.
Key scientific test: do underdamped slots (ω_k > γ_k) develop more negative β_k?

If yes: the model has learned phase-space routing at slot resolution.
If yes AND if it beats HORN: temporal lead routing is the mechanism, not just β < 0 globally.

Initialize β_k = -0.3 (V7 converged layer-mean) so we skip the "β ≈ 0 → learn negative" phase.

Run this AFTER V7 results confirm V7 ≈ HORN (ties) or V7 > HORN.
If V7 < HORN: run V8-A first (simpler), V8-B second.
"""

import sys
sys.path.insert(0, '/workspace')

import math
import json
import time
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import tiktoken

from cdm_model_v8b import CDMLanguageModelV8B, CDMConfigV8B


CFG = CDMConfigV8B(
    vocab_size   = 50257,
    d_model      = 384,
    n_layers     = 8,
    n_heads      = 8,
    n_kv_heads   = 4,
    d_ff         = 1024,
    K            = 16,
    max_len      = 512,
    lbl_coeff    = 0.01,
    entropy_reg  = 0.02,
    dropout      = 0.0,
    d_osc        = 8,
    beta_init    = -0.3,
    eps          = 1e-6,
)

BATCH_SIZE    = 8
SEQ_LEN       = 256
LR            = 3e-4
STEPS         = 30000
WARMUP_STEPS  = 500
VAL_EVERY     = 500
SAVE_EVERY    = 2500
PROBE_EVERY   = 2500

OUT_DIR  = "/workspace/cdm_v8b"
LOG_FILE = "/workspace/cdm_v8b_training.log"
os.makedirs(f"{OUT_DIR}/best", exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg):
    ts = time.strftime("[%Y-%m-%dT%H:%M:%SZ]", time.gmtime())
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


class TinyStoriesDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len
    def __getitem__(self, idx):
        s = idx * self.seq_len
        return torch.tensor(self.tokens[s:s+self.seq_len+1], dtype=torch.long)


def load_tinystories(seq_len, cache_dir="/workspace/tinystories_cache"):
    enc = tiktoken.get_encoding("gpt2")
    log("Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", cache_dir=cache_dir)
    train_tokens = []
    for item in ds["train"]:
        train_tokens.extend(enc.encode(item["text"]) + [50256])
    val_tokens = []
    for item in ds["validation"]:
        val_tokens.extend(enc.encode(item["text"]) + [50256])
    log(f"Train: {len(train_tokens):,} tokens | Val: {len(val_tokens):,} tokens")
    return TinyStoriesDataset(train_tokens, seq_len), TinyStoriesDataset(val_tokens, seq_len)


def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (STEPS - WARMUP_STEPS)
    return LR * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, val_loader, max_batches=50):
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x = batch[:, :-1].to(DEVICE)
        logits, _, _ = model(x)
        y = batch[:, 1:].to(DEVICE)
        loss = F.cross_entropy(logits.reshape(-1, CFG.vocab_size), y.reshape(-1))
        total += loss.item()
        n += 1
    model.train()
    return total / n if n > 0 else float("inf")


def train():
    log("=== CDM V8-B: HORN+Kuramoto, per-slot β_k (K=16 per layer) ===")
    log(f"Device: {DEVICE}")
    log(f"Config: d_model={CFG.d_model} | K={CFG.K} | d_osc={CFG.d_osc} | beta_init={CFG.beta_init}")
    log(f"Scientific test: do underdamped slots (ω_k>γ_k) develop more negative β_k?")

    train_ds, val_ds = load_tinystories(SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True, drop_last=True)

    model = CDMLanguageModelV8B(CFG).to(DEVICE)
    log(f"Model: {model.param_count:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   betas=(0.9, 0.95), weight_decay=0.1)

    cfg_dict = {k: v for k, v in CFG.__dict__.items()}
    cfg_dict["n_params"]     = model.param_count
    cfg_dict["architecture"] = "CDM_V8B_HORN_Kuramoto_PerSlotBeta"
    with open(f"{OUT_DIR}/config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    best_val_ce = float("inf")
    best_step   = 0
    train_iter  = iter(train_loader)
    step = 0
    t0   = time.time()

    while step < STEPS:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x = batch[:, :-1].to(DEVICE)

        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

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
            log(f"  Val CE: {val_ce:.4f} (step {step})")
            if val_ce < best_val_ce:
                best_val_ce = val_ce
                best_step   = step
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "val_loss": val_ce,
                    "config": cfg_dict,
                }, f"{OUT_DIR}/best/model.pt")
                log(f"  NEW BEST: {val_ce:.4f} @ step {step}")

        if step % SAVE_EVERY == 0:
            ckpt_dir = f"{OUT_DIR}/step_{step:06d}"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "config": cfg_dict,
            }, f"{ckpt_dir}/model.pt")

        if step % PROBE_EVERY == 0:
            model.forward_loss(x, collect_probe=True)
            osc = model.get_oscillator_stats()
            kp  = model.get_kuramoto_probe_stats()
            log(f"  === V8-B Oscillator probe @ step {step} ===")
            for l, s in osc.items():
                wa = kp.get(l, {}).get("winner_alignment_mean", float("nan"))
                log(f"    layer_{l}: γ={s['gamma_mean']:.3f} ω={s['omega_mean']:.3f} "
                    f"under={s['underdamped_frac']:.0%} "
                    f"β_mean={s['beta_mean']:.3f}[{s['beta_min']:.3f},{s['beta_max']:.3f}] "
                    f"β_under={s['beta_underdamped_mean']:.3f} β_over={s['beta_overdamped_mean']:.3f} "
                    f"align={wa:.3f}")

    log(f"=== V8-B TRAINING COMPLETE ===")
    log(f"Best val CE: {best_val_ce:.4f} @ step {best_step}")
    log(f"Reference:  HORN=1.5818 | CDM-Kuramoto=1.5819 | V7=TBD | V8-A=TBD")
    log(f"Key result: if β_underdamped_mean < β_overdamped_mean consistently → hypothesis confirmed")


if __name__ == "__main__":
    train()
