#!/usr/bin/env python3
"""
cdm_train_v6_horn.py — Training script for CDM V6 HORN (Second-Order Slot Dynamics)

V6 HORN vs V3:
  - Replaces first-order EMA slot update with Störmer-Verlet damped oscillator
  - Each slot has learnable damping γ_k and natural frequency ω_k
  - Position S and velocity V per slot; force = write-gated token projection
  - Softmax routing and LBL same as V3
  - Same arch dims (37.1M → slightly different due to γ/ω params)

DuoNeural / Archon — 2026-06-16
"""

import math, json, time, argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime, timezone
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast

from cdm_model_v6_horn import CDMLanguageModelV6HORN
from cdm_model_v3 import CDMConfigV3


def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def log(msg, logfile=None):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a") as f:
            f.write(line + "\n")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FULL_CFG = dict(
    d_model=384, n_layers=8, n_heads=8, n_kv_heads=4, d_ff=1024, K=16, max_len=512,
    entropy_reg=0.02,
    lbl_coeff=0.01,
    alpha_init=0.0,   # unused by HORN, kept for config compat
    batch_size=8, seq_len=256, lr=3e-4, steps=30000, warmup_steps=500,
    val_every=500, save_every=2500,
)

SMOKE_CFG = dict(
    d_model=256, n_layers=4, n_heads=8, n_kv_heads=4, d_ff=512, K=8, max_len=256,
    entropy_reg=0.02, lbl_coeff=0.01, alpha_init=0.0,
    batch_size=8, seq_len=128, lr=3e-4, steps=200, warmup_steps=20,
    val_every=50, save_every=100,
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
            log(f"  HF load failed: {e}")
            return
        stories = ds["text"] if max_stories is None else ds["text"][:max_stories]
        chunk = []
        for i, story in enumerate(stories):
            chunk.extend(tokenizer.encode(story))
            chunk.append(tokenizer.eos_token_id)
            if (i + 1) % 10000 == 0:
                log(f"  Tokenized {i+1} stories...")
        for i in range(0, len(chunk) - seq_len - 1, seq_len):
            self.tokens.append(torch.tensor(chunk[i:i + seq_len + 1], dtype=torch.long))
        log(f"  {split}: {len(self.tokens)} sequences")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.tokens, cache_file)

    def __len__(self):  return len(self.tokens)
    def __getitem__(self, idx):
        c = self.tokens[idx]
        return c[:-1], c[1:]


def get_lr(step: int, max_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return max_lr * step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * p))


def log_dynamics_stats(model: CDMLanguageModelV6HORN, logfile=None):
    """Log learned γ (damping) and ω (frequency) per slot per layer."""
    stats = model.get_dynamics_stats()
    log("  [V6 HORN dynamics] — γ (damping) | ω (frequency) per slot:", logfile)
    for layer_key, d in stats.items():
        log(f"    {layer_key}: γ_mean={d['gamma_mean']:.4f}  ω_mean={d['omega_mean']:.4f}", logfile)
        log(f"      γ: [{', '.join(f'{v:.3f}' for v in d['gamma'])}]", logfile)
        log(f"      ω: [{', '.join(f'{v:.3f}' for v in d['omega'])}]", logfile)
    return stats


def train(cfg_dict: dict, out_dir: Path, logfile: Path = None, smoke: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60, logfile)
    log("CDM V6 HORN — Second-Order Slot Dynamics (Störmer-Verlet)", logfile)
    log(f"Mode: {'SMOKE' if smoke else 'TRAINING'}  Device: {DEVICE}", logfile)
    log("=" * 60, logfile)

    arch = CDMConfigV3(
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
        lbl_coeff=cfg_dict.get("lbl_coeff", 0.01),
        alpha_init=cfg_dict.get("alpha_init", 0.0),
    )

    BS         = cfg_dict["batch_size"]
    SL         = cfg_dict["seq_len"]
    LR         = cfg_dict["lr"]
    STEPS      = cfg_dict["steps"]
    WARMUP     = cfg_dict["warmup_steps"]
    VAL_EVERY  = cfg_dict["val_every"]
    SAVE_EVERY = cfg_dict["save_every"]

    log("Loading tokenizer...", logfile)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    arch.vocab_size = tokenizer.vocab_size

    model = CDMLanguageModelV6HORN(arch).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    log(f"Model: {n:,} params ({n/1e6:.1f}M)", logfile)
    log(f"  K={arch.K}, d={arch.d_model}, L={arch.n_layers}", logfile)
    log(f"  entropy_reg={arch.entropy_reg}, lbl_coeff={arch.lbl_coeff}", logfile)
    if DEVICE == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"  GPU: {torch.cuda.get_device_name(0)}, {vram:.1f}GB", logfile)

    if smoke:
        log("\nSmoke: overfitting 1 random batch", logfile)
        dummy = torch.randint(0, arch.vocab_size, (BS, SL + 1))
        smoke_batch = (dummy[:, :-1].to(DEVICE), dummy[:, 1:].to(DEVICE))
        train_iter = val_iter = None
    else:
        train_ds = TinyStoriesDataset(tokenizer, SL, "train")
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
               "architecture": "CDM_V6_HORN", "device": DEVICE}
    with open(out_dir / "config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    log(f"\nTraining {STEPS} steps  batch={BS}  seq={SL}", logfile)
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
            n_since = min(step, 50)
            avg_ce  = running_ce  / n_since
            avg_aux = running_aux / n_since
            running_ce = running_aux = 0.0
            elapsed = time.time() - t0
            tps = step * BS * SL / elapsed
            lr_now = optimizer.param_groups[0]["lr"]
            log(f"  step {step:6d}/{STEPS} | ce={avg_ce:.4f} aux={avg_aux:.4f} "
                f"| lr={lr_now:.2e} | {tps:.0f} tok/s", logfile)

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
            log(f"\n  Val CE: {avg_val:.4f} (step {step})  [V3 target: 1.5831  Kuramoto: 1.5819]", logfile)

            if avg_val < best_val:
                best_val = avg_val
                best_dir = out_dir / "best"
                best_dir.mkdir(exist_ok=True)
                torch.save({"step": step, "model_state": model.state_dict(),
                            "val_loss": avg_val, "config": run_cfg},
                           best_dir / "model.pt")
                log(f"  New best: {avg_val:.4f}", logfile)

            log_dynamics_stats(model, logfile)
            model.train()

        if step % SAVE_EVERY == 0:
            ckpt_dir = out_dir / f"step_{step:06d}"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save({"step": step, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss": best_val, "config": run_cfg},
                       ckpt_dir / "checkpoint.pt")
            log(f"  Checkpoint: {ckpt_dir}", logfile)

    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    torch.save({"step": step, "model_state": model.state_dict(),
                "val_loss": best_val, "config": run_cfg},
               final_dir / "model.pt")
    log(f"\nDone. Best val CE: {best_val:.4f}", logfile)
    log("\nFinal HORN dynamics:", logfile)
    log_dynamics_stats(model, logfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--out", default="/workspace/cdm_v6_horn_full")
    parser.add_argument("--log", default="/workspace/cdm_v6_horn_training.log")
    args = parser.parse_args()

    cfg = SMOKE_CFG.copy() if args.mode == "smoke" else FULL_CFG.copy()
    smoke = args.mode == "smoke"
    logfile = Path(args.log) if not smoke else None
    train(cfg, Path(args.out), logfile=logfile, smoke=smoke)


if __name__ == "__main__":
    main()
