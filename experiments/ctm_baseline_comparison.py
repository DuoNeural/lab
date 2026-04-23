"""
ctm_baseline_comparison.py — CTM vs Baseline, fair kilonova comparison
DuoNeural 2026-04-23 — Archon

Purpose: The 300M longrun compared CTM+GHL against a baseline from a 4090 pod
(different hardware, different step budget = apples to oranges). This script runs
BOTH conditions on kilonova at identical compute budgets to settle the question:

  Condition A: Pure Transformer baseline (no CTM, no GHL)
  Condition B: CTM 300M + GHL v5  (rerun, same arch as longrun)

Same dataset (TinyStories), same seq_len, same steps, same hardware.
If B beats A on kilonova → hypothesis CONFIRMED. Full stop.
If A beats B → GHL genuinely hurts at 300M+ scale.

Architecture: ~267M params each (matched to longrun for fair comparison)
Steps: 20,000 (same as longrun)
BF16 throughout. Launch with HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=1

Logs:
  /home/ai/duoneural/ctm_baseline_comparison.log
  /home/ai/duoneural/ctm_baseline_comparison/  (checkpoints)
"""

import os
import sys
import math
import time
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────

D_MODEL     = 1024
N_LAYERS    = 16
N_HEADS     = 16
D_FF        = 4096
CTX_LEN     = 256        # was 512 — halving seq_len nearly doubles throughput on gfx1103
BATCH_SIZE  = 4
GRAD_ACCUM  = 4          # effective batch = 16
LR          = 3e-4
WARMUP      = 250        # scaled down from 500 proportionally
TOTAL_STEPS = 10000      # was 20000 — statistically valid for A/B comparison
EVAL_EVERY  = 500
SAVE_EVERY  = 2500       # was 5000
GHL_LAMBDA  = 0.1
N_THOUGHT   = 2          # CTM recurrent steps

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.bfloat16

LOG_PATH    = Path("/home/ai/duoneural/ctm_baseline_comparison.log")
CKPT_DIR    = Path("/home/ai/duoneural/ctm_baseline_comparison")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
)
log = logging.getLogger()

# ── Tokenizer & Data ──────────────────────────────────────────────────────────

def get_tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok

class TinyStoriesDataset(Dataset):
    def __init__(self, tokenizer, ctx_len, split="train", max_samples=200_000):
        from datasets import load_dataset
        log.info(f"loading TinyStories ({split})...")
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=False)
        self.examples = []
        buf = []
        buf_len = 0
        count = 0
        for item in ds:
            if count >= max_samples:
                break
            ids = tokenizer.encode(item["text"], add_special_tokens=False)
            ids.append(tokenizer.eos_token_id)
            buf.extend(ids)
            buf_len += len(ids)
            while buf_len >= ctx_len + 1:
                chunk = buf[:ctx_len + 1]
                self.examples.append(torch.tensor(chunk, dtype=torch.long))
                buf = buf[ctx_len:]
                buf_len -= ctx_len
                count += 1
                if count >= max_samples:
                    break
        log.info(f"  {len(self.examples)} sequences loaded")

    def __len__(self): return len(self.examples)
    def __getitem__(self, i):
        x = self.examples[i]
        return x[:-1], x[1:]

# ── Baseline Transformer ──────────────────────────────────────────────────────

class BaselineTransformer(nn.Module):
    """Pure transformer — no CTM recurrence, no GHL. The control condition."""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, ctx_len, dropout=0.1):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.pos    = nn.Embedding(ctx_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f   = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight  # weight tying
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device, dtype=h.dtype)
        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.ln_f(h)
        return self.lm_head(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# ── CTM 300M + GHL v5 ─────────────────────────────────────────────────────────

class GHLHead(nn.Module):
    """thought-space self-prediction head — same as longrun"""
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
    def forward(self, z): return self.net(z)

class CTMBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x, mask=None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x

class CTM300M_GHL(nn.Module):
    """CTM 300M + GHL v5 — exact same arch as longrun, for fair comparison"""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, ctx_len,
                 n_thought=2, ghl_lambda=0.1, dropout=0.1):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.pos     = nn.Embedding(ctx_len, d_model)
        self.blocks  = nn.ModuleList([CTMBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight
        self.ghl_head  = GHLHead(d_model)
        self.n_thought = n_thought
        self.ghl_lambda = ghl_lambda
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device, dtype=h.dtype)

        ghl_loss = torch.tensor(0.0, device=x.device, dtype=h.dtype)
        for _ in range(self.n_thought):
            z_prev = h.detach()
            for block in self.blocks:
                h = block(h, mask=mask)
            # GHL: predict next z from prev z
            z_pred = self.ghl_head(z_prev)
            ghl_loss = ghl_loss + nn.functional.mse_loss(z_pred, h.detach())

        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits, ghl_loss / self.n_thought

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# ── Training loop ─────────────────────────────────────────────────────────────

def cosine_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * step / warmup
    t = (step - warmup) / (total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * t))

def train_condition(name, model, loader, total_steps, eval_loader=None):
    log.info(f"\n{'='*60}")
    log.info(f"CONDITION {name}: {model.count_params()/1e6:.1f}M params")
    log.info(f"{'='*60}")

    model = model.to(DEVICE, dtype=DTYPE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, fused=False)
    criterion = nn.CrossEntropyLoss()

    best_ppl  = float('inf')
    best_path = CKPT_DIR / f"{name}_best.pt"
    results   = {}
    step = 0
    t_start = time.time()

    loader_iter = iter(loader)
    opt.zero_grad()

    while step < total_steps:
        model.train()
        accum_loss = 0.0
        accum_ghl  = 0.0

        for micro in range(GRAD_ACCUM):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)

            x, y = x.to(DEVICE), y.to(DEVICE)

            if isinstance(model, CTM300M_GHL):
                logits, ghl_loss = model(x)
                ce = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss = ce + model.ghl_lambda * ghl_loss
                accum_ghl += ghl_loss.item()
            else:
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = cosine_lr(step, total_steps, WARMUP, LR)
        for pg in opt.param_groups:
            pg['lr'] = lr

        opt.step()
        opt.zero_grad()
        step += 1

        if step % EVAL_EVERY == 0 or step == total_steps:
            elapsed = time.time() - t_start
            sps = step / elapsed
            eta_h = (total_steps - step) / sps / 3600
            ppl = math.exp(min(accum_loss / GRAD_ACCUM, 20))

            if isinstance(model, CTM300M_GHL):
                log.info(
                    f"[{name}] step {step:>6}/{total_steps} | ppl={ppl:.2f} | "
                    f"ghl_loss={accum_ghl/GRAD_ACCUM:.3f} | {sps:.2f} sps | ETA={eta_h:.1f}h"
                )
            else:
                log.info(
                    f"[{name}] step {step:>6}/{total_steps} | ppl={ppl:.2f} | "
                    f"{sps:.2f} sps | ETA={eta_h:.1f}h"
                )

            if ppl < best_ppl:
                best_ppl = ppl
                torch.save({
                    'step': step, 'ppl': ppl,
                    'model': model.state_dict(),
                    'condition': name,
                }, best_path)

        if step % SAVE_EVERY == 0:
            ckpt_path = CKPT_DIR / f"{name}_step{step}.pt"
            torch.save({'step': step, 'model': model.state_dict()}, ckpt_path)
            log.info(f"[{name}] checkpoint -> {ckpt_path}")

    results = {'condition': name, 'best_ppl': best_ppl, 'steps': total_steps,
               'params': model.count_params(), 'timestamp': datetime.now().isoformat()}
    log.info(f"\n[{name}] DONE — best_ppl={best_ppl:.4f}")
    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("="*60)
    log.info("CTM Baseline Comparison — Fair kilonova A/B Test")
    log.info(f"  Condition A: Pure Transformer (control)")
    log.info(f"  Condition B: CTM 300M + GHL v5 (experimental)")
    log.info(f"  Steps: {TOTAL_STEPS} each | Device: {DEVICE}")
    log.info("="*60)

    tok = get_tokenizer()
    VOCAB = tok.vocab_size

    log.info("loading datasets...")
    train_ds = TinyStoriesDataset(tok, CTX_LEN, split="train", max_samples=200_000)
    loader   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)

    all_results = []

    # ── Condition A: Pure Transformer ─────────────────────────────────────────
    model_a = BaselineTransformer(
        vocab_size=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, d_ff=D_FF, ctx_len=CTX_LEN
    )
    res_a = train_condition("A_baseline", model_a, loader, TOTAL_STEPS)
    all_results.append(res_a)
    del model_a
    torch.cuda.empty_cache()

    # ── Condition B: CTM + GHL v5 ─────────────────────────────────────────────
    model_b = CTM300M_GHL(
        vocab_size=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, d_ff=D_FF, ctx_len=CTX_LEN,
        n_thought=N_THOUGHT, ghl_lambda=GHL_LAMBDA,
    )
    res_b = train_condition("B_ctm_ghl", model_b, loader, TOTAL_STEPS)
    all_results.append(res_b)
    del model_b
    torch.cuda.empty_cache()

    # ── Final Report ──────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("FINAL COMPARISON RESULTS")
    log.info("="*60)
    ppl_a = all_results[0]['best_ppl']
    ppl_b = all_results[1]['best_ppl']
    gap   = ppl_b - ppl_a

    log.info(f"  A (Baseline Transformer):  best_ppl={ppl_a:.4f}")
    log.info(f"  B (CTM 300M + GHL v5):     best_ppl={ppl_b:.4f}")
    log.info(f"  gap (B - A):               {gap:+.4f}")

    if gap < 0:
        log.info(f"  HYPOTHESIS CONFIRMED — CTM+GHL beats baseline by {abs(gap):.4f} ppl")
    elif gap < 0.1:
        log.info(f"  INCONCLUSIVE — gap within noise ({gap:.4f})")
    else:
        log.info(f"  HYPOTHESIS REJECTED — baseline beats CTM+GHL by {gap:.4f} ppl at this scale")

    log.info("experiment complete")

    results_path = CKPT_DIR / "comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump({'results': all_results, 'gap': gap, 'winner': 'B' if gap < 0 else 'A'}, f, indent=2)
    log.info(f"results saved -> {results_path}")


if __name__ == "__main__":
    main()
