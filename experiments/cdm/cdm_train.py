#!/usr/bin/env python3
"""
cdm_train.py — Training script for Competitive Docking Memory Language Model

Train on TinyStories from scratch to prove the architecture works.
Runs on both P4 (smoke test) and 5060Ti-C (full run).

Usage:
  python cdm_train.py                     # default config
  python cdm_train.py --smoke             # overfit 1 batch, verify learning
  python cdm_train.py --full              # full TinyStories run (5060Ti-C)
  python cdm_train.py --steps 5000        # custom step count

Archon + Aura — DuoNeural — 2026-06-11
"""

import math
import json
import time
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime, timezone
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast

from cdm_model import CDMConfig, CDMLanguageModel


def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


# ── Config ────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SMOKE_CFG = dict(
    # Architecture — tiny for P4 smoke test
    d_model=256, n_layers=4, n_heads=8, n_kv_heads=4, d_ff=512, K=8, max_len=256,
    # Training
    batch_size=8, seq_len=128, lr=3e-4, steps=200, warmup_steps=20,
    val_every=50, save_every=100,
)

FULL_CFG = dict(
    # Architecture — proper model for 5060Ti-C
    d_model=384, n_layers=8, n_heads=8, n_kv_heads=4, d_ff=1024, K=8, max_len=512,
    # Training
    batch_size=32, seq_len=256, lr=3e-4, steps=30000, warmup_steps=500,
    val_every=500, save_every=2500,
)

P4_CFG = dict(
    # Architecture — fits P4's 7.5GB VRAM
    d_model=256, n_layers=8, n_heads=8, n_kv_heads=4, d_ff=512, K=8, max_len=256,
    # Training
    batch_size=16, seq_len=128, lr=3e-4, steps=50000, warmup_steps=500,
    val_every=500, save_every=2500,
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TinyStoriesDataset(Dataset):
    """
    Loads TinyStories from HuggingFace datasets.
    Each example: a random seq_len-length chunk of a story.
    """
    def __init__(self, tokenizer, seq_len: int, split: str = "train",
                 max_stories: int = None):
        self.seq_len = seq_len
        self.tokens = []

        log(f"Loading TinyStories ({split})...")
        try:
            from datasets import load_dataset
            ds = load_dataset("roneneldan/TinyStories", split=split, streaming=False)
        except Exception as e:
            log(f"  HF dataset failed: {e}")
            log("  Falling back to synthetic TinyStories-like data...")
            self._make_synthetic(tokenizer, seq_len)
            return

        stories = ds["text"] if max_stories is None else ds["text"][:max_stories]
        chunk = []
        for i, story in enumerate(stories):
            ids = tokenizer.encode(story)
            chunk.extend(ids)
            chunk.append(tokenizer.eos_token_id)
            if (i + 1) % 10000 == 0:
                log(f"  Tokenized {i+1} stories ({len(chunk)//1000}k tokens)...")

        # Split into seq_len+1 windows (input + target)
        stride = seq_len
        for i in range(0, len(chunk) - seq_len - 1, stride):
            self.tokens.append(torch.tensor(chunk[i:i + seq_len + 1], dtype=torch.long))

        log(f"  {split}: {len(self.tokens)} sequences from {len(chunk)//1000}k tokens")

    def _make_synthetic(self, tokenizer, seq_len):
        """Fallback: synthetic short stories if HF datasets unavailable."""
        templates = [
            "Once upon a time there was a little girl named Lily. She liked to play in the garden.",
            "Tom was a small boy who loved trains. One day he found a special red train.",
            "The bunny hopped through the forest looking for carrots. She found a big garden.",
        ] * 3000
        chunk = []
        for t in templates:
            chunk.extend(tokenizer.encode(t))
            chunk.append(tokenizer.eos_token_id)
        for i in range(0, len(chunk) - seq_len - 1, seq_len):
            self.tokens.append(torch.tensor(chunk[i:i+seq_len+1], dtype=torch.long))
        log(f"  Synthetic: {len(self.tokens)} sequences")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        chunk = self.tokens[idx]
        return chunk[:-1], chunk[1:]  # input, target


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(step: int, max_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg_dict: dict, out_dir: Path, smoke: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("Competitive Docking Memory (CDM) Language Model")
    log(f"Mode: {'SMOKE TEST' if smoke else 'FULL TRAINING'}")
    log(f"Device: {DEVICE}")
    log("=" * 60)

    # Config
    arch_cfg = CDMConfig(
        vocab_size=50257,  # GPT2 tokenizer vocab
        d_model=cfg_dict["d_model"],
        n_layers=cfg_dict["n_layers"],
        n_heads=cfg_dict["n_heads"],
        n_kv_heads=cfg_dict["n_kv_heads"],
        d_ff=cfg_dict["d_ff"],
        K=cfg_dict["K"],
        max_len=cfg_dict["max_len"],
    )
    batch_size   = cfg_dict["batch_size"]
    seq_len      = cfg_dict["seq_len"]
    lr           = cfg_dict["lr"]
    total_steps  = cfg_dict["steps"]
    warmup_steps = cfg_dict["warmup_steps"]
    val_every    = cfg_dict["val_every"]
    save_every   = cfg_dict["save_every"]

    # Tokenizer
    log("Loading GPT2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    arch_cfg.vocab_size = tokenizer.vocab_size  # 50257

    # Model
    model = CDMLanguageModel(arch_cfg).to(DEVICE)
    n_params = model.param_count()
    log(f"Model: {n_params:,} parameters ({n_params/1e6:.1f}M)")
    log(f"  d_model={arch_cfg.d_model}, K={arch_cfg.K}, n_layers={arch_cfg.n_layers}")

    if DEVICE == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"  GPU: {torch.cuda.get_device_name(0)}, {vram:.1f}GB VRAM")

    # Data
    if smoke:
        # Smoke test: single batch, overfit to check learning
        log("\nSmoke test mode: overfitting 1 batch...")
        dummy_ids = torch.randint(0, arch_cfg.vocab_size, (batch_size, seq_len + 1))
        smoke_batch = (dummy_ids[:, :-1].to(DEVICE), dummy_ids[:, 1:].to(DEVICE))
        train_iter = None
        val_iter = None
    else:
        max_stories = 20000 if "steps" in cfg_dict and cfg_dict["steps"] <= 5000 else None
        train_ds = TinyStoriesDataset(tokenizer, seq_len, "train", max_stories)
        val_ds   = TinyStoriesDataset(tokenizer, seq_len, "validation", max_stories=500)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=(DEVICE=="cuda"), drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                  num_workers=0, drop_last=True)
        train_iter = iter(train_loader)
        val_iter   = val_loader

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.1, betas=(0.9, 0.95)
    )

    # Save config
    run_config = {**cfg_dict, "d_model": arch_cfg.d_model, "K": arch_cfg.K,
                  "vocab_size": arch_cfg.vocab_size, "n_params": n_params,
                  "architecture": "CDM", "device": DEVICE}
    with open(out_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Training loop
    log(f"\nTraining: {total_steps} steps, batch={batch_size}, seq={seq_len}")
    best_val_loss = float("inf")
    global_step = 0
    t0 = time.time()
    running_loss = 0.0

    while global_step < total_steps:
        model.train()

        # Get batch
        if smoke:
            x, y = smoke_batch
        else:
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(DEVICE), y.to(DEVICE)

        # Forward + loss
        for param_group in optimizer.param_groups:
            param_group["lr"] = get_lr(global_step, lr, warmup_steps, total_steps)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, arch_cfg.vocab_size), y.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        global_step += 1

        # Logging
        if global_step % 50 == 0 or (smoke and global_step % 10 == 0):
            avg_loss = running_loss / 50 if global_step >= 50 else running_loss / global_step
            running_loss = 0.0
            elapsed = time.time() - t0
            tok_per_sec = global_step * batch_size * seq_len / elapsed
            current_lr = optimizer.param_groups[0]["lr"]
            log(f"  step {global_step:6d}/{total_steps} | loss={avg_loss:.4f} | "
                f"lr={current_lr:.2e} | {tok_per_sec:.0f} tok/s")

        # Validation
        if global_step % val_every == 0 and val_iter is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i, (vx, vy) in enumerate(val_iter):
                    if i >= 20:
                        break
                    vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                    vlogits = model(vx)
                    vloss = F.cross_entropy(vlogits.reshape(-1, arch_cfg.vocab_size),
                                            vy.reshape(-1))
                    val_losses.append(vloss.item())
            avg_val = sum(val_losses) / len(val_losses)
            log(f"\n  Val loss: {avg_val:.4f} (step {global_step})")
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_dir = out_dir / "best"
                best_dir.mkdir(exist_ok=True)
                torch.save({
                    "step": global_step,
                    "model_state": model.state_dict(),
                    "val_loss": avg_val,
                    "config": run_config,
                }, best_dir / "model.pt")
                log(f"  New best: {avg_val:.4f}")

            # Quick generation sample
            model.eval()
            with torch.no_grad():
                prompt = tokenizer.encode("Once upon a time", return_tensors="pt").to(DEVICE)
                gen = model.generate(prompt, max_new=80, temperature=0.8, top_k=40)
                text = tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)
                log(f"  Sample: {text[:200]}")

        # Save checkpoint
        if global_step % save_every == 0:
            ckpt_dir = out_dir / f"step_{global_step:06d}"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save({
                "step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": run_config,
            }, ckpt_dir / "checkpoint.pt")
            log(f"  Saved: {ckpt_dir}")

    # Final save
    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    torch.save({
        "step": global_step,
        "model_state": model.state_dict(),
        "val_loss": best_val_loss,
        "config": run_config,
    }, final_dir / "model.pt")
    # Save config for HF
    with open(final_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    log(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    log(f"Final checkpoint: {final_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Overfit smoke test")
    parser.add_argument("--full", action="store_true", help="Full TinyStories run")
    parser.add_argument("--p4", action="store_true", help="P4-optimized config")
    parser.add_argument("--steps", type=int, default=None, help="Override step count")
    parser.add_argument("--out", type=str, default="/workspace/cdm_run", help="Output dir")
    args = parser.parse_args()

    if args.smoke:
        cfg = SMOKE_CFG.copy()
    elif args.full:
        cfg = FULL_CFG.copy()
    elif args.p4:
        cfg = P4_CFG.copy()
    else:
        cfg = P4_CFG.copy()  # default to P4 config

    if args.steps:
        cfg["steps"] = args.steps

    out_dir = Path(args.out)
    train(cfg, out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
