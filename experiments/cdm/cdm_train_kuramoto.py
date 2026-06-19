#!/usr/bin/env python3
"""
cdm_train_kuramoto.py - training entrypoint for CDM-Kuramoto.

Defaults match CDM V3 except for the routing module. Use --d_osc for the
parameter-matched/sweep variants.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast

from cdm_model_kuramoto import CDMConfigKuramoto, CDMLanguageModelKuramoto
from cdm_train_v3 import TinyStoriesDataset, get_lr, log, log_alpha_stats


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FULL_CFG = dict(
    d_model=384, n_layers=8, n_heads=8, n_kv_heads=4, d_ff=1024, K=16, max_len=512,
    entropy_reg=0.02, lbl_coeff=0.01, alpha_init=0.0, d_osc=8,
    batch_size=8, seq_len=256, lr=3e-4, steps=30000, warmup_steps=500,
    val_every=500, save_every=2500,
)

SMOKE_CFG = dict(
    d_model=256, n_layers=4, n_heads=8, n_kv_heads=4, d_ff=512, K=8, max_len=256,
    entropy_reg=0.02, lbl_coeff=0.01, alpha_init=0.0, d_osc=8,
    batch_size=8, seq_len=128, lr=3e-4, steps=50, warmup_steps=10,
    val_every=25, save_every=50,
)


def log_kuramoto_stats(model: CDMLanguageModelKuramoto, logfile=None):
    stats = model.get_kuramoto_probe_stats()
    log("  [Kuramoto probe stats]", logfile)
    for layer_key, layer_stats in stats.items():
        if not layer_stats:
            log(f"    {layer_key}: no probe collected", logfile)
            continue
        msg = " ".join(f"{k}={v:.6f}" for k, v in layer_stats.items())
        log(f"    {layer_key}: {msg}", logfile)
    return stats


def train(cfg_dict: dict, out_dir: Path, logfile: Path = None, smoke: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    log("=" * 60, logfile)
    log("CDM-Kuramoto - positive-coupling oscillator routing", logfile)
    log(f"Mode: {'SMOKE' if smoke else 'TRAINING'}  Device: {DEVICE}", logfile)
    log("=" * 60, logfile)

    arch = CDMConfigKuramoto(
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
        d_osc=cfg_dict.get("d_osc", 8),
    )
    BS, SL = cfg_dict["batch_size"], cfg_dict["seq_len"]
    LR, STEPS = cfg_dict["lr"], cfg_dict["steps"]
    WARMUP = cfg_dict["warmup_steps"]
    VAL_EVERY, SAVE_EVERY = cfg_dict["val_every"], cfg_dict["save_every"]

    log("Loading tokenizer...", logfile)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    arch.vocab_size = tokenizer.vocab_size

    model = CDMLanguageModelKuramoto(arch).to(DEVICE)
    n = model.param_count()
    log(f"Model: {n:,} params ({n/1e6:.1f}M)", logfile)
    log(f"  K={arch.K}, d={arch.d_model}, L={arch.n_layers}, d_osc={arch.d_osc}", logfile)
    log(f"  entropy_reg={arch.entropy_reg}, lbl_coeff={arch.lbl_coeff}, alpha_init={arch.alpha_init}", logfile)
    if DEVICE == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"  GPU: {torch.cuda.get_device_name(0)}, {vram:.1f}GB", logfile)

    if smoke:
        dummy = torch.randint(0, arch.vocab_size, (BS, SL + 1))
        smoke_batch = (dummy[:, :-1].to(DEVICE), dummy[:, 1:].to(DEVICE))
        train_iter = val_iter = None
    else:
        max_s = 20000 if STEPS <= 5000 else None
        train_ds = TinyStoriesDataset(tokenizer, SL, "train", max_s)
        val_ds = TinyStoriesDataset(tokenizer, SL, "validation", 500)
        train_loader = DataLoader(train_ds, BS, shuffle=True, num_workers=2,
                                  pin_memory=(DEVICE == "cuda"), drop_last=True)
        val_loader = DataLoader(val_ds, BS, shuffle=False, num_workers=0, drop_last=True)
        train_iter = iter(train_loader)
        val_iter = val_loader

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.1, betas=(0.9, 0.95),
    )

    run_cfg = {
        **cfg_dict, "vocab_size": arch.vocab_size, "n_params": n,
        "architecture": "CDM_KURAMOTO", "device": DEVICE,
    }
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
        loss = ce_loss + aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_ce += ce_loss.item()
        running_aux += aux_loss.item()
        step += 1

        if step % 50 == 0 or (smoke and step % 10 == 0):
            n_since = min(step, 50)
            avg_ce = running_ce / n_since
            avg_aux = running_aux / n_since
            running_ce = running_aux = 0.0
            tps = step * BS * SL / max(1e-6, time.time() - t0)
            lr_now = optimizer.param_groups[0]["lr"]
            log(f"  step {step:6d}/{STEPS} | ce={avg_ce:.4f} aux={avg_aux:.4f} "
                f"| lr={lr_now:.2e} | {tps:.0f} tok/s", logfile)

        if step % VAL_EVERY == 0 and val_iter is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i, (vx, vy) in enumerate(val_iter):
                    if i >= 20:
                        break
                    vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                    vlogits, _ = model(vx, collect_probe=(i == 0))
                    val_losses.append(
                        F.cross_entropy(vlogits.reshape(-1, arch.vocab_size), vy.reshape(-1)).item()
                    )
            avg_val = sum(val_losses) / len(val_losses)
            log(f"\n  Val CE: {avg_val:.4f} (step {step})", logfile)
            if avg_val < best_val:
                best_val = avg_val
                best_dir = out_dir / "best"
                best_dir.mkdir(exist_ok=True)
                torch.save({"step": step, "model_state": model.state_dict(),
                            "val_loss": avg_val, "config": run_cfg},
                           best_dir / "model.pt")
                log(f"  New best: {avg_val:.4f}", logfile)
            log_alpha_stats(model, logfile)
            log_kuramoto_stats(model, logfile)
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
    log_alpha_stats(model, logfile)
    log_kuramoto_stats(model, logfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--out", default="/workspace/cdm_kuramoto_dosc8")
    parser.add_argument("--log", default="/workspace/cdm_kuramoto_training.log")
    parser.add_argument("--d_osc", type=int, default=None)
    parser.add_argument("--lbl_coeff", type=float, default=None)
    parser.add_argument("--entropy_reg", type=float, default=None)
    parser.add_argument("--alpha_init", type=float, default=None)
    args = parser.parse_args()

    cfg = SMOKE_CFG.copy() if args.mode == "smoke" else FULL_CFG.copy()
    for key in ("d_osc", "lbl_coeff", "entropy_reg", "alpha_init"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    smoke = args.mode == "smoke"
    train(cfg, Path(args.out), logfile=(None if smoke else Path(args.log)), smoke=smoke)


if __name__ == "__main__":
    main()
