#!/usr/bin/env python3
"""
cdm_code_baseline_train.py — Standard transformer baseline for CODE CDM paper comparison
DuoNeural / Archon — 2026-06-15

Ablation: does CDM V2's 1.3483 val CE on Python code come from the architecture,
or just from having more params?

Code CDM V2 (CDMLanguageModelV2):
  - backbone: GQA self-attn + CDM memory module + SlotCrossAttn
  - d_model=384, n_layers=8, n_heads=8, n_kv_heads=4, d_ff=1024, K=16
  - 37,107,592 params   val CE = 1.3483

This baseline (same backbone, CDM and SlotXAttn removed):
  - pure GQA self-attn + FFN only
  - d_model=384, n_layers=8, n_heads=8, n_kv_heads=4, d_ff=1024
  - ~32M params (CDM adds ~5M overhead)
  - Expected: higher val CE than 1.3483 → CDM contributes real capability

Dataset: codeparrot/codeparrot-clean-train (same as code CDM V2)
  - Checks /workspace/code_cache/ first (if pod had prior CDM code run)
  - Otherwise downloads and tokenizes fresh (~17 min on first run)
"""

import math, json, time, os
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
class Config:
    vocab_size:  int   = 50257
    n_layers:    int   = 8
    d_model:     int   = 384
    n_heads:     int   = 8
    n_kv_heads:  int   = 4
    d_ff:        int   = 1024    # EXACT same as code CDM V2 (not bumped)
    max_len:     int   = 512
    dropout:     float = 0.1

TRAIN_STEPS  = 30000
BATCH_SIZE   = 16     # same as code CDM V2
SEQ_LEN      = 256    # same as code CDM V2
LR           = 3e-4
WARMUP_STEPS = 500
GRAD_CLIP    = 1.0
VAL_EVERY    = 500
LOG_EVERY    = 50
CACHE_DIR    = Path("/workspace/code_cache")
OUT_DIR      = Path("/workspace/cdm_code_baseline")
LOG_FILE     = Path("/workspace/cdm_code_baseline_training.log")

OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
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
        freqs = torch.outer(torch.arange(max_len).float(), theta)
        self.register_buffer("cos", freqs.cos()[None, None, :, :])
        self.register_buffer("sin", freqs.sin()[None, None, :, :])

    def forward(self, x):
        d = x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        cos = self.cos[:, :, :x.shape[2], :]
        sin = self.sin[:, :, :x.shape[2], :]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
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

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.q_proj(x).view(B, T, self.n_heads,    self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        Q, K = self.rope(Q), self.rope(K)
        K = K.repeat_interleave(self.n_rep, dim=1)
        V = V.repeat_interleave(self.n_rep, dim=1)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, -1))


class FFN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up   = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff,    cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    """Standard pre-norm transformer block — no CDM, no slot cross-attention."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn     = CausalSelfAttention(cfg)
        self.ffn      = FFN(cfg)
        self.norm_sa  = nn.RMSNorm(cfg.d_model)
        self.norm_ff  = nn.RMSNorm(cfg.d_model)
        self.drop     = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm_sa(x)))
        x = x + self.ffn(self.norm_ff(x))
        return x


class BaselineGPT(nn.Module):
    """Vanilla GQA GPT — same backbone dims as CDM V2 code, no memory slots."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm   = nn.RMSNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Dataset ───────────────────────────────────────────────────────────────────

class CodeDataset(Dataset):
    """Loads Python code from codeparrot/codeparrot-clean-train with disk cache."""

    def __init__(self, tokenizer, seq_len: int, max_tokens_M: float, split_suffix: str):
        cache_path = CACHE_DIR / f"code_{split_suffix}_{seq_len}_m{int(max_tokens_M)}.pt"

        if cache_path.exists():
            log(f"  Cache hit: {cache_path}")
            self.chunks = torch.load(cache_path, weights_only=True)
            log(f"  Loaded {len(self.chunks)} sequences")
            return

        log(f"  Cache miss — downloading codeparrot/codeparrot-clean-train...")
        from datasets import load_dataset
        ds = load_dataset("codeparrot/codeparrot-clean-train", split="train", streaming=True)

        target = int(max_tokens_M * 1_000_000)
        tokens = []
        count = 0
        for ex in ds:
            enc = tokenizer.encode(ex.get("content", ex.get("text", "")))
            tokens.extend(enc + [tokenizer.eos_token_id])
            count += 1
            if count % 5000 == 0:
                log(f"  Processed {count} files, {len(tokens)/1e6:.1f}M tokens...")
            if len(tokens) >= target:
                break

        log(f"  Total tokens: {len(tokens)/1e6:.1f}M")
        self.chunks = [
            torch.tensor(tokens[i:i+seq_len], dtype=torch.long)
            for i in range(0, len(tokens) - seq_len, seq_len)
        ]
        log(f"  {split_suffix}: {len(self.chunks)} sequences")
        torch.save(self.chunks, cache_path)
        log(f"  Cached to {cache_path}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        return self.chunks[i]


# ── Training utils ────────────────────────────────────────────────────────────

def val_loss(model, val_loader, device, n_batches=50):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            x = batch.to(device)
            logits = model(x[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                x[:, 1:].reshape(-1)
            )
            total += loss.item()
            count += 1
    model.train()
    return total / max(count, 1)


def cosine_lr(step, warmup, total, lr_max):
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / (total - warmup)
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _log_fh
    _log_fh = open(LOG_FILE, "w")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 65)
    log("CDM Code Baseline — Vanilla GQA GPT on Python Code")
    log("Ablation: CDM V2 (37.1M, CE=1.3483) vs baseline (same dims, no CDM)")
    log(f"d_model={384} n_layers={8} n_heads={8} n_kv_heads={4} d_ff={1024}")
    log(f"Steps: {TRAIN_STEPS} | Batch: {BATCH_SIZE} | Seq: {SEQ_LEN}")
    log(f"Dataset: codeparrot/codeparrot-clean-train (Python)")
    log("DuoNeural / Archon — 2026-06-15")
    log("=" * 65)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.eos_token_id = tokenizer.eos_token_id or 50256

    log("\nBuilding datasets...")
    train_ds = CodeDataset(tokenizer, SEQ_LEN, max_tokens_M=200, split_suffix="train")
    val_ds   = CodeDataset(tokenizer, SEQ_LEN, max_tokens_M=2,   split_suffix="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, drop_last=False)

    cfg   = Config()
    model = BaselineGPT(cfg).to(DEVICE)
    n     = model.n_params()
    log(f"\nModel: {n/1e6:.2f}M params")
    log(f"  (CDM V2 reference: 37.1M params — CDM adds ~{(37107592-n)/1e6:.1f}M overhead)")

    cfg_dict = {
        "model_type":  "vanilla_gpt_baseline_code",
        "vocab_size":  cfg.vocab_size, "d_model": cfg.d_model, "n_layers": cfg.n_layers,
        "n_heads":     cfg.n_heads, "n_kv_heads": cfg.n_kv_heads, "d_ff": cfg.d_ff,
        "max_len":     cfg.max_len, "n_params": n,
        "dataset":     "codeparrot/codeparrot-clean-train",
        "note":        "Baseline for CDM V2 code ablation — no CDM module, no slot cross-attention"
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=True)
    scaler = torch.cuda.amp.GradScaler()

    train_iter = iter(train_loader)
    best_val, best_step = float("inf"), 0
    step = 0
    t0   = time.time()

    log(f"\nTraining...")
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
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 x[:, 1:].reshape(-1))

        scaler.scale(ce).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            tps = int(step * BATCH_SIZE * SEQ_LEN / elapsed)
            log(f"  step {step:6d}/{TRAIN_STEPS} | ce={ce.item():.4f} | lr={lr:.2e} | {tps} tok/s")

        if step % VAL_EVERY == 0 or step == TRAIN_STEPS:
            vl = val_loss(model, val_loader, DEVICE)
            log(f"  [VAL] step={step} val_ce={vl:.4f}  (CDM V2 ref=1.3483)")
            if vl < best_val:
                best_val, best_step = vl, step
                best_dir = OUT_DIR / "best"
                best_dir.mkdir(exist_ok=True)
                torch.save({
                    "step": step, "val_loss": vl,
                    "model_state": model.state_dict(), "config": cfg_dict,
                }, best_dir / "model.pt")
                log(f"  [SAVE] new best step={step} val_ce={vl:.4f}")

    log(f"\nTraining complete.")
    log(f"  Best val CE:  {best_val:.4f} @ step {best_step}")
    log(f"  CDM V2 ref:   1.3483 @ step 30000")
    log(f"  CDM advantage: {best_val - 1.3483:+.4f} CE units")
    log(f"  Param overhead: CDM V2={37.1:.1f}M, Baseline={n/1e6:.2f}M, CDM_overhead={(37107592-n)/1e6:.1f}M")
    log("=" * 65)
    _log_fh.close()


if __name__ == "__main__":
    main()
