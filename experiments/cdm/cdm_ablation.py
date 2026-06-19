#!/usr/bin/env python3
"""
cdm_ablation.py — K-slot ablation study for CDM on P4

Trains CDM with K = [1, 2, 4, 8, 16, 32] sequentially.
K=1 is the degenerate "single running average" baseline.
K=8 is our architecture.
High K tests over-parameterization.

Question: what's the minimum K for specialization to emerge?

Each variant: 5000 steps, batch=8, seq=128 (P4-safe config)
Max 20k stories for fast loading.

Archon — DuoNeural — 2026-06-11
"""

import json
import math
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime, timezone
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast

from cdm_model import CDMConfig, CDMLanguageModel


def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Shared config — P4-safe, fast enough to show signal at 5k steps
BASE_CFG = dict(
    d_model=256, n_layers=4, n_heads=8, n_kv_heads=4, d_ff=512,
    max_len=256, batch_size=8, seq_len=128, lr=3e-4,
    steps=5000, warmup_steps=200, val_every=500, save_every=5000,
)

K_VALUES = [1, 2, 4, 8, 16, 32]


def get_lr(step, max_lr, warmup, total):
    if step < warmup:
        return max_lr * step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * p))


def run_k(K, tokenizer, train_ds, val_ds, out_dir):
    log(f"\n{'='*60}")
    log(f"K={K} ablation run")
    log(f"{'='*60}")

    cfg = CDMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=BASE_CFG["d_model"], n_layers=BASE_CFG["n_layers"],
        n_heads=BASE_CFG["n_heads"], n_kv_heads=BASE_CFG["n_kv_heads"],
        d_ff=BASE_CFG["d_ff"], K=K, max_len=BASE_CFG["max_len"],
    )

    model = CDMLanguageModel(cfg).to(DEVICE)
    n_params = model.param_count()
    log(f"  K={K}: {n_params:,} params ({n_params/1e6:.2f}M)")

    train_loader = DataLoader(train_ds, batch_size=BASE_CFG["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=(DEVICE=="cuda"),
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BASE_CFG["batch_size"],
                            shuffle=False, num_workers=0, drop_last=True)
    train_iter = iter(train_loader)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=BASE_CFG["lr"], weight_decay=0.1, betas=(0.9, 0.95),
    )

    total_steps = BASE_CFG["steps"]
    warmup = BASE_CFG["warmup_steps"]
    val_every = BASE_CFG["val_every"]
    batch_size = BASE_CFG["batch_size"]
    seq_len = BASE_CFG["seq_len"]

    best_val = float("inf")
    running_loss = 0.0
    t0 = time.time()
    val_curve = []

    for step in range(1, total_steps + 1):
        model.train()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)

        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(step, BASE_CFG["lr"], warmup, total_steps)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

        if step % 100 == 0:
            avg = running_loss / 100
            running_loss = 0.0
            elapsed = time.time() - t0
            tps = step * batch_size * seq_len / elapsed
            log(f"  K={K} step {step:5d}/{total_steps} | loss={avg:.4f} | {tps:.0f} tok/s")

        if step % val_every == 0:
            model.eval()
            vlosses = []
            with torch.no_grad():
                for i, (vx, vy) in enumerate(val_loader):
                    if i >= 20: break
                    vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                    vlogits = model(vx)
                    vloss = F.cross_entropy(vlogits.reshape(-1, cfg.vocab_size), vy.reshape(-1))
                    vlosses.append(vloss.item())
            avg_val = sum(vlosses) / len(vlosses)
            val_curve.append({"step": step, "val_loss": avg_val})
            log(f"  K={K} → val_loss={avg_val:.4f}")
            if avg_val < best_val:
                best_val = avg_val

    # Quick slot specialization probe: generate short text, check slot entropy
    slot_entropy = probe_slot_entropy(model, tokenizer)
    log(f"  K={K} → best_val={best_val:.4f}, slot_entropy={slot_entropy:.4f}")

    result = {
        "K": K,
        "n_params": n_params,
        "best_val_loss": best_val,
        "slot_entropy": slot_entropy,
        "val_curve": val_curve,
    }

    torch.save({
        "model_state": model.state_dict(),
        "config": {"K": K, **BASE_CFG},
        "result": result,
    }, out_dir / f"K{K:03d}_checkpoint.pt")

    return result


def probe_slot_entropy(model, tokenizer, n_stories=20):
    """
    Measure slot routing entropy at the last block.
    Low entropy = high specialization (some slots dominate)
    High entropy = uniform write = no specialization
    Max entropy for K slots = ln(K)

    Strategy: monkey-patch compute_gates in all blocks to capture outputs,
    run full forward pass so hidden states are realistic, collect last-block gates.
    """
    stories = [
        "Once upon a time there was a little girl named Lily.",
        "Tom loved his red train more than anything.",
        "A rabbit named Bella lived in the forest.",
        "Max the dog ran through the sunny park.",
        "The princess found a magic flower in the garden.",
        "Little Ben could not sleep because of the storm.",
        "Sara and her brother went to the beach one day.",
        "The old turtle walked slowly to the big pond.",
        "A tiny mouse named Pip lived under the stairs.",
        "Jack found a treasure map in the attic.",
        "The brave knight helped the lost kitten home.",
        "Lucy learned to ride her new bicycle today.",
        "The snowman smiled at all the children playing.",
        "Danny the duck had a dream about flying high.",
        "Emma's birthday cake was the best in the village.",
        "The wizard gave the boy a sparkling golden star.",
        "A tiny dragon hatched from a shiny purple egg.",
        "The farmer found a rainbow after the big rain.",
        "Oliver wanted to be an astronaut more than anything.",
        "The dancing bear made everyone in the town smile.",
    ][:n_stories]

    model.eval()

    # Patch last block's compute_gates to capture output
    captured = {}
    last_block = model.blocks[-1]
    orig_cg = last_block.cdm.compute_gates

    def patched_cg(h):
        g = orig_cg(h)
        captured["gates"] = g.detach().cpu()
        return g

    last_block.cdm.compute_gates = patched_cg

    entropies = []
    with torch.no_grad():
        for story in stories:
            ids = tokenizer.encode(story, return_tensors="pt").to(DEVICE)
            if ids.shape[1] < 2:
                continue
            captured.clear()
            _ = model(ids)  # full forward pass — last block's gates get captured
            if "gates" not in captured:
                continue
            # gates: (1, T, K) — take mean across positions
            g = captured["gates"][0].float()  # (T, K)
            g_mean = g.mean(0)                # (K,) — average write weight per slot
            g_safe = g_mean.clamp(min=1e-9)
            g_norm = g_safe / g_safe.sum()
            ent = -(g_norm * g_norm.log()).sum().item()
            entropies.append(ent)

    last_block.cdm.compute_gates = orig_cg  # restore
    return sum(entropies) / len(entropies) if entropies else 0.0


def main():
    import sys
    out_dir = Path("/workspace/cdm_ablation")
    out_dir.mkdir(exist_ok=True)

    log("CDM K-slot Ablation Study")
    log(f"K values: {K_VALUES}")
    log(f"Steps per variant: {BASE_CFG['steps']}")
    log(f"Device: {DEVICE}")

    log("\nLoading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    log("Loading TinyStories (max 20k stories for speed)...")
    # Import dataset here so failures don't abort everything
    try:
        from datasets import load_dataset
        from torch.utils.data import Dataset

        class TinyStoriesDataset(Dataset):
            def __init__(self, tokenizer, seq_len, split, max_stories):
                self.seq_len = seq_len
                self.tokens = []
                ds = load_dataset("roneneldan/TinyStories", split=split, streaming=False)
                stories = ds["text"][:max_stories]
                chunk = []
                for story in stories:
                    chunk.extend(tokenizer.encode(story))
                    chunk.append(tokenizer.eos_token_id)
                for i in range(0, len(chunk) - seq_len - 1, seq_len):
                    self.tokens.append(torch.tensor(chunk[i:i+seq_len+1], dtype=torch.long))
                log(f"  {split}: {len(self.tokens)} sequences")

            def __len__(self): return len(self.tokens)
            def __getitem__(self, idx):
                c = self.tokens[idx]
                return c[:-1], c[1:]

        seq_len = BASE_CFG["seq_len"]
        train_ds = TinyStoriesDataset(tokenizer, seq_len, "train", max_stories=20000)
        val_ds   = TinyStoriesDataset(tokenizer, seq_len, "validation", max_stories=500)

    except Exception as e:
        log(f"Dataset load failed: {e}")
        sys.exit(1)

    all_results = {}
    for K in K_VALUES:
        result = run_k(K, tokenizer, train_ds, val_ds, out_dir)
        all_results[f"K{K}"] = result
        # Save incrementally
        with open(out_dir / "ablation_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        log(f"\nK={K} done. Results saved.")

    # Summary
    log("\n" + "="*60)
    log("K-SLOT ABLATION SUMMARY")
    log("="*60)
    log(f"{'K':>4} | {'params':>10} | {'val_loss':>8} | {'slot_entropy':>12}")
    log("-"*44)
    for k in K_VALUES:
        r = all_results.get(f"K{k}", {})
        log(f"{k:>4} | {r.get('n_params',0):>10,} | {r.get('best_val_loss',0):>8.4f} | {r.get('slot_entropy',0):>12.4f}")

    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nAll results saved: {out_dir}/ablation_results.json")


if __name__ == "__main__":
    main()
