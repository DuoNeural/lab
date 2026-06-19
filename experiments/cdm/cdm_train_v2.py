#!/usr/bin/env python3
"""
cdm_train_v2.py — Training script for CDM V2

V2 changes vs V1:
  - Causal per-position slot states (breaks routing collapse)
  - Slot cross-attention separate from sequence self-attention
  - Marginal entropy regularization on routing gates
  - K=16 (optimal from V1 ablation, beats K=8 by 17%)
  - Model returns (logits, aux_loss) — training loop must unpack

Archon + Aura — DuoNeural — 2026-06-11
"""

import math, json, time, argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime, timezone
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast

from cdm_model_v2 import CDMConfigV2, CDMLanguageModelV2


def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SMOKE_CFG = dict(
    d_model=256, n_layers=4, n_heads=8, n_kv_heads=4, d_ff=512, K=8, max_len=256,
    entropy_reg=0.02,
    batch_size=8, seq_len=128, lr=3e-4, steps=200, warmup_steps=20,
    val_every=50, save_every=100,
)

FULL_CFG = dict(
    d_model=384, n_layers=8, n_heads=8, n_kv_heads=4, d_ff=1024, K=16, max_len=512,
    entropy_reg=0.02,
    # batch=8: sequential scan states = 8 blocks * 255 states * (8*16*384)*4 = 3.2GB
    # vs batch=32 = 12.8GB (OOM on 16GB). Seq throughput maintained via more steps.
    batch_size=8, seq_len=256, lr=3e-4, steps=30000, warmup_steps=500,
    val_every=500, save_every=2500,
)

P4_CFG = dict(
    d_model=256, n_layers=8, n_heads=8, n_kv_heads=4, d_ff=512, K=16, max_len=256,
    entropy_reg=0.02,
    batch_size=16, seq_len=128, lr=3e-4, steps=50000, warmup_steps=500,
    val_every=500, save_every=2500,
)


CACHE_DIR = Path("/workspace/tinystories_cache")

class TinyStoriesDataset(Dataset):
    def __init__(self, tokenizer, seq_len: int, split: str = "train",
                 max_stories: int = None):
        self.seq_len = seq_len
        self.tokens = []

        cache_key = f"{split}_{seq_len}" + (f"_n{max_stories}" if max_stories else "")
        cache_file = CACHE_DIR / f"{cache_key}.pt"
        if cache_file.exists():
            log(f"Loading TinyStories ({split}) from cache: {cache_file}")
            self.tokens = torch.load(cache_file, weights_only=True)
            log(f"  {split}: {len(self.tokens)} sequences (cached)")
            return

        log(f"Loading TinyStories ({split})...")
        try:
            from datasets import load_dataset
            ds = load_dataset("roneneldan/TinyStories", split=split, streaming=False)
        except Exception as e:
            log(f"  HF load failed: {e}, using synthetic fallback")
            self._make_synthetic(tokenizer, seq_len)
            return
        stories = ds["text"] if max_stories is None else ds["text"][:max_stories]
        chunk = []
        for i, story in enumerate(stories):
            chunk.extend(tokenizer.encode(story))
            chunk.append(tokenizer.eos_token_id)
            if (i + 1) % 10000 == 0:
                log(f"  Tokenized {i+1} stories ({len(chunk)//1000}k tokens)...")
        for i in range(0, len(chunk) - seq_len - 1, seq_len):
            self.tokens.append(torch.tensor(chunk[i:i + seq_len + 1], dtype=torch.long))
        log(f"  {split}: {len(self.tokens)} sequences from {len(chunk)//1000}k tokens")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.tokens, cache_file)
        log(f"  Cached to {cache_file}")

    def _make_synthetic(self, tokenizer, seq_len):
        templates = [
            "Once upon a time there was a little girl named Lily. She liked to play in the garden.",
            "Tom was a small boy who loved trains. One day he found a special red train.",
            "The bunny hopped through the forest. She found a big garden full of carrots.",
        ] * 3000
        chunk = []
        for t in templates:
            chunk.extend(tokenizer.encode(t))
            chunk.append(tokenizer.eos_token_id)
        for i in range(0, len(chunk) - seq_len - 1, seq_len):
            self.tokens.append(torch.tensor(chunk[i:i+seq_len+1], dtype=torch.long))
        log(f"  Synthetic: {len(self.tokens)} sequences")

    def __len__(self):  return len(self.tokens)
    def __getitem__(self, idx):
        c = self.tokens[idx]
        return c[:-1], c[1:]


def get_lr(step: int, max_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return max_lr * step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * p))


def train(cfg_dict: dict, out_dir: Path, smoke: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("CDM V2 — Competitive Docking Memory (Causal Slots)")
    log(f"Mode: {'SMOKE' if smoke else 'TRAINING'}  Device: {DEVICE}")
    log("=" * 60)

    arch = CDMConfigV2(
        vocab_size=50257,
        d_model=cfg_dict["d_model"],
        n_layers=cfg_dict["n_layers"],
        n_heads=cfg_dict["n_heads"],
        n_kv_heads=cfg_dict["n_kv_heads"],
        d_ff=cfg_dict["d_ff"],
        K=cfg_dict["K"],
        max_len=cfg_dict["max_len"],
        dropout=0.1,
        entropy_reg=cfg_dict.get("entropy_reg", 0.02),
    )
    BS          = cfg_dict["batch_size"]
    SL          = cfg_dict["seq_len"]
    LR          = cfg_dict["lr"]
    STEPS       = cfg_dict["steps"]
    WARMUP      = cfg_dict["warmup_steps"]
    VAL_EVERY   = cfg_dict["val_every"]
    SAVE_EVERY  = cfg_dict["save_every"]

    log("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    arch.vocab_size = tokenizer.vocab_size

    model = CDMLanguageModelV2(arch).to(DEVICE)
    n = model.param_count()
    log(f"Model: {n:,} params ({n/1e6:.1f}M)")
    log(f"  K={arch.K}, d={arch.d_model}, L={arch.n_layers}, entropy_reg={arch.entropy_reg}")
    if DEVICE == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"  GPU: {torch.cuda.get_device_name(0)}, {vram:.1f}GB")

    if smoke:
        log("\nSmoke: overfitting 1 random batch")
        dummy = torch.randint(0, arch.vocab_size, (BS, SL + 1))
        smoke_batch = (dummy[:, :-1].to(DEVICE), dummy[:, 1:].to(DEVICE))
        train_iter = val_iter = None
    else:
        max_s = 20000 if STEPS <= 5000 else None
        train_ds = TinyStoriesDataset(tokenizer, SL, "train", max_s)
        val_ds   = TinyStoriesDataset(tokenizer, SL, "validation", 500)
        train_loader = DataLoader(train_ds, BS, shuffle=True, num_workers=2,
                                  pin_memory=(DEVICE=="cuda"), drop_last=True)
        val_loader   = DataLoader(val_ds, BS, shuffle=False, num_workers=0, drop_last=True)
        train_iter = iter(train_loader)
        val_iter   = val_loader

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.1, betas=(0.9, 0.95),
    )

    run_cfg = {**cfg_dict, "vocab_size": arch.vocab_size, "n_params": n,
               "architecture": "CDM_V2", "device": DEVICE}
    with open(out_dir / "config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    log(f"\nTraining {STEPS} steps  batch={BS}  seq={SL}")
    best_val = float("inf")
    step = 0
    t0 = time.time()
    running_ce = running_aux = 0.0

    while step < STEPS:
        model.train()

        if smoke:
            x, y = smoke_batch
        else:
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(DEVICE), y.to(DEVICE)

        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(step, LR, WARMUP, STEPS)

        logits, aux_loss = model(x)
        ce_loss = F.cross_entropy(logits.reshape(-1, arch.vocab_size), y.reshape(-1))
        loss    = ce_loss + aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        running_ce  += ce_loss.item()
        running_aux += aux_loss.item()
        step += 1

        if step % 50 == 0 or (smoke and step % 10 == 0):
            n_since = 50 if step >= 50 else step
            avg_ce  = running_ce  / n_since
            avg_aux = running_aux / n_since
            running_ce = running_aux = 0.0
            elapsed = time.time() - t0
            tps = step * BS * SL / elapsed
            lr_now = optimizer.param_groups[0]["lr"]
            log(f"  step {step:6d}/{STEPS} | ce={avg_ce:.4f} aux={avg_aux:.4f} "
                f"| lr={lr_now:.2e} | {tps:.0f} tok/s")

        if step % VAL_EVERY == 0 and val_iter is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i, (vx, vy) in enumerate(val_iter):
                    if i >= 20: break
                    vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                    vlogits, _ = model(vx)
                    val_losses.append(
                        F.cross_entropy(vlogits.reshape(-1, arch.vocab_size), vy.reshape(-1)).item()
                    )
            avg_val = sum(val_losses) / len(val_losses)
            log(f"\n  Val CE: {avg_val:.4f} (step {step})")
            if avg_val < best_val:
                best_val = avg_val
                best_dir = out_dir / "best"
                best_dir.mkdir(exist_ok=True)
                torch.save({"step": step, "model_state": model.state_dict(),
                            "val_loss": avg_val, "config": run_cfg},
                           best_dir / "model.pt")
                log(f"  New best: {avg_val:.4f}")

            model.eval()
            with torch.no_grad():
                prompt = tokenizer.encode("Once upon a time", return_tensors="pt").to(DEVICE)
                gen = model.generate(prompt, max_new=80, temperature=0.8, top_k=40)
                text = tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)
                log(f"  Sample: {text[:200]}")

        if step % SAVE_EVERY == 0:
            ckpt_dir = out_dir / f"step_{step:06d}"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save({"step": step, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss": best_val, "config": run_cfg},
                       ckpt_dir / "checkpoint.pt")
            log(f"  Checkpoint: {ckpt_dir}")

    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    torch.save({"step": step, "model_state": model.state_dict(),
                "val_loss": best_val, "config": run_cfg},
               final_dir / "model.pt")
    log(f"\nDone. Best val CE: {best_val:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full",  action="store_true")
    p.add_argument("--p4",    action="store_true")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out",   type=str, default="/workspace/cdm_v2_run")
    p.add_argument("--entropy-reg", type=float, default=None)
    args = p.parse_args()

    if args.smoke:   cfg = SMOKE_CFG.copy()
    elif args.full:  cfg = FULL_CFG.copy()
    elif args.p4:    cfg = P4_CFG.copy()
    else:            cfg = P4_CFG.copy()

    if args.steps:        cfg["steps"]       = args.steps
    if args.entropy_reg is not None:
                          cfg["entropy_reg"] = args.entropy_reg

    train(cfg, Path(args.out), smoke=args.smoke)


if __name__ == "__main__":
    main()
