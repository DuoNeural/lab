#!/usr/bin/env python3
"""
cdm_baseline_train.py — Vanilla GPT baseline for CDM V3/V4 comparison
DuoNeural / Archon — 2026-06-15

Purpose: parameter-honest comparison for CDM V3/V4 on TinyStories.
  CDM V3/V4: 37.1M params, val CE=1.5831 (30k steps, full TinyStories, d_model=384 L=8)
  This baseline: ~32M params (same backbone dims, no CDM/SlotXAttn overhead)
    d_model=384, 8 layers, d_ff=1024 — same backbone as CDM V3, CDM adds ~5M overhead.
    Full TinyStories dataset (same as CDM V3/V4 training).
    Same LR, batch_size=8, seq=256, 30k steps.

Interpretation:
  If baseline val CE >> 1.5831, CDM V3 contributes real capability.
  If baseline val CE ≈ 1.5831, CDM overhead = parameter overhead (unlikely given CDM_overhead=5M).
  Note: this baseline has FEWER params than CDM V3 (32M vs 37.1M) because CDM slots add ~5M.
  The 72.9M baseline (d_model=512 L=12) is the comparison for CDM V5 (85.7M).
"""

import math
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class BaselineConfig:
    vocab_size: int   = 50257
    n_layers:   int   = 8
    d_model:    int   = 384
    n_heads:    int   = 8
    n_kv_heads: int   = 4
    d_ff:       int   = 1024    # EXACT same backbone dims as CDM V3 (CDM adds ~5M overhead on top)
    max_len:    int   = 256
    dropout:    float = 0.1

TRAIN_STEPS   = 30000
BATCH_SIZE    = 8       # same as CDM V3/V4
LR            = 3e-4
WARMUP_STEPS  = 300
GRAD_CLIP     = 1.0
CKPT_EVERY    = 500
PROBE_STEPS   = {1500, 5000, 15000, 30000}   # routing probes for CDM V2 comparison

OUT_DIR  = Path("/workspace/cdm_v3_baseline")
LOG_FILE = Path("/workspace/cdm_v3_baseline_training.log")
OUT_DIR.mkdir(exist_ok=True)

_log_fh = None

def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def log(msg):
    s = f"[{ts()}] {msg}"
    print(s, flush=True)
    if _log_fh:
        _log_fh.write(s + "\n")
        _log_fh.flush()


# ── Model ─────────────────────────────────────────────────────────────────────

class RoPE(nn.Module):
    def __init__(self, d_head: int, max_len: int):
        super().__init__()
        theta = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, theta)
        self.register_buffer("cos", freqs.cos()[None, None, :, :])
        self.register_buffer("sin", freqs.sin()[None, None, :, :])

    def forward(self, x):
        d = x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        cos = self.cos[:, :, :x.shape[2], :]
        sin = self.sin[:, :, :x.shape[2], :]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head     = cfg.d_model // cfg.n_heads
        self.n_rep      = cfg.n_heads // cfg.n_kv_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads    * self.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.d_head, cfg.d_model,    bias=False)
        self.rope   = RoPE(self.d_head, cfg.max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        Q, K = self.rope(Q), self.rope(K)
        K = K.repeat_interleave(self.n_rep, dim=1)
        V = V.repeat_interleave(self.n_rep, dim=1)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class FFN(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.gate    = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up      = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down    = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block. No CDM, no slot cross-attention."""
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.attn    = CausalSelfAttention(cfg)
        self.ffn     = FFN(cfg)
        self.norm_sa = nn.RMSNorm(cfg.d_model)
        self.norm_ff = nn.RMSNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm_sa(x)))
        x = x + self.ffn(self.norm_ff(x))
        return x


class BaselineGPT(nn.Module):
    """Vanilla GQA GPT — same depth/width as CDM V2 but no memory slots."""
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm   = nn.RMSNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ── Dataset ───────────────────────────────────────────────────────────────────

class TinyStoriesDataset(Dataset):
    def __init__(self, tokenizer, seq_len: int, n_seqs: int, split="train", seed=42):
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=False)
        import random
        random.seed(seed)
        tokens = []
        for ex in ds:
            t = tokenizer.encode(ex["text"])
            tokens.extend(t + [tokenizer.eos_token_id])
            if len(tokens) >= (n_seqs + 100) * seq_len:
                break
        self.chunks = [
            torch.tensor(tokens[i:i+seq_len], dtype=torch.long)
            for i in range(0, len(tokens) - seq_len, seq_len)
        ][:n_seqs]
        log(f"  Dataset: {len(self.chunks)} sequences of {seq_len} tokens")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        return self.chunks[i]


# ── Training ──────────────────────────────────────────────────────────────────

def val_loss(model, val_loader, device, n_batches=50):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            x = batch.to(device)
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
            total += loss.item()
            count += 1
    model.train()
    return total / max(count, 1)


def cosine_lr(step: int, warmup: int, total: int, lr_max: float) -> float:
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / (total - warmup)
    return lr_max * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    global _log_fh
    _log_fh = open(LOG_FILE, "w")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 60)
    log("CDM Baseline — Vanilla GQA GPT (no CDM, no slots)")
    log(f"d={384} n_layers={8} n_heads={8} d_ff={1300}")
    log(f"Steps: {TRAIN_STEPS} | Batch: {BATCH_SIZE} | Seq: {SEQ_LEN}")
    log(f"Dataset: TinyStories | Comparison: CDM V2 val_ce=1.5934")
    log("DuoNeural / Archon — 2026-06-12")
    log("=" * 60)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.eos_token_id = tokenizer.eos_token_id or 50256

    log("\nBuilding datasets...")
    train_ds = TinyStoriesDataset(tokenizer, SEQ_LEN, n_seqs=2000000, split="train")   # full dataset like CDM V3
    val_ds   = TinyStoriesDataset(tokenizer, SEQ_LEN, n_seqs=2000,   split="validation", seed=0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True, drop_last=False)

    cfg   = BaselineConfig()
    model = BaselineGPT(cfg).to(DEVICE)
    n_params = model.param_count()
    log(f"\nModel: {n_params/1e6:.2f}M params (CDM V3/V4 = 37.1M; CDM adds ~{(37107720-n_params)/1e6:.1f}M overhead)")

    # Save config
    cfg_dict = {
        "model_type": "vanilla_gpt_baseline",
        "vocab_size": cfg.vocab_size, "d_model": cfg.d_model, "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads, "d_ff": cfg.d_ff,
        "max_len": cfg.max_len, "dropout": cfg.dropout,
        "n_params": n_params,
        "note": "CDM V3/V4 comparison baseline — no CDM, no slot cross-attention; full TinyStories"
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=True)
    scaler = torch.cuda.amp.GradScaler()

    train_iter = iter(train_loader)
    best_val, best_step = float("inf"), 0
    step = 0
    t0 = time.time()

    log(f"\nStarting training...")
    model.train()

    while step < TRAIN_STEPS:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        lr = cosine_lr(step, WARMUP_STEPS, TRAIN_STEPS, LR)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x = batch.to(DEVICE)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(x[:, :-1])
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))

        scaler.scale(ce).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % 50 == 0:
            elapsed = time.time() - t0
            tok_per_sec = int(step * BATCH_SIZE * SEQ_LEN / elapsed)
            log(f"  step {step:6d}/{TRAIN_STEPS} | ce={ce.item():.4f} | lr={lr:.2e} | {tok_per_sec} tok/s")

        if step % CKPT_EVERY == 0 or step in PROBE_STEPS or step == TRAIN_STEPS:
            vl = val_loss(model, val_loader, DEVICE)
            log(f"  [VAL] step={step} val_ce={vl:.4f} (CDM V2 ref={1.5934:.4f})")
            if vl < best_val:
                best_val, best_step = vl, step
                best_dir = OUT_DIR / "best"
                best_dir.mkdir(exist_ok=True)
                torch.save({
                    "step": step, "model_state": model.state_dict(),
                    "val_loss": vl, "config": cfg_dict,
                }, best_dir / "model.pt")
                log(f"  [SAVE] new best @ step {step}: val_ce={vl:.4f}")

    log(f"\nTraining complete.")
    log(f"  Best val CE:   {best_val:.4f} @ step {best_step}")
    log(f"  CDM V3/V4 ref: 1.5831 @ step 30000")
    log(f"  CDM advantage: {best_val - 1.5831:+.4f} CE (positive = baseline worse = CDM wins)")
    log("=" * 60)
    _log_fh.close()


if __name__ == "__main__":
    main()
