"""
mamba_baseline_train.py — Pure PyTorch Mamba-like SSM baseline for CDM comparison.
~37M params on TinyStories to match CDM V3/HORN scale.
DuoNeural / Archon — 2026-06-17

No mamba-ssm dependency. Sequential scan in pure PyTorch (correct, not CUDA-optimized).
Architecture: Mamba-style SSM with selective scanning, d_model=384, n_layers=10.
"""

import math
import json
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# ─── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "d_model": 384,
    "n_layers": 10,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "dt_rank": "auto",          # "auto" = ceil(d_model/16)
    "vocab_size": 50257,
    "max_len": 512,
    "batch_size": 8,
    "seq_len": 256,
    "lr": 3e-4,
    "steps": 30000,
    "warmup_steps": 500,
    "val_every": 500,
    "save_every": 2500,
    "architecture": "MambaBaseline",
}
CFG["dt_rank"] = math.ceil(CFG["d_model"] / 16) if CFG["dt_rank"] == "auto" else CFG["dt_rank"]
CFG["d_inner"] = CFG["d_model"] * CFG["expand"]

OUT_DIR = "/workspace/mamba_baseline_full"
LOG_FILE = "/workspace/mamba_baseline_training.log"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/best", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg):
    ts = time.strftime("[%Y-%m-%dT%H:%M:%SZ]", time.gmtime())
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ─── Mamba SSM Layer (pure PyTorch, sequential scan) ──────────────────────────

class SelectiveSSM(nn.Module):
    """Core Mamba selective scan in pure PyTorch. Correct, ~2x slower than CUDA."""

    def __init__(self, d_inner, d_state, dt_rank):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float).repeat(d_inner, 1)))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Input-dependent projections
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        nn.init.normal_(self.dt_proj.weight, std=0.02)
        nn.init.uniform_(self.dt_proj.bias, -4, -1)  # small dt init

    def forward(self, x):
        # x: (B, L, d_inner)
        B, L, D = x.shape
        S = self.d_state

        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative real

        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*S)
        dt, B_ssm, C_ssm = x_dbl.split([self.dt_rank, S, S], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        # Discretize: ZOH
        # dA: (B, L, d_inner, S)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # dB: (B, L, d_inner, S)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)

        # Sequential scan
        h = torch.zeros(B, D, S, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y = (h * C_ssm[:, t].unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        return y + x * self.D.unsqueeze(0).unsqueeze(0)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, dt_rank):
        super().__init__()
        d_inner = d_model * 2
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=d_inner, bias=True)
        self.act = nn.SiLU()
        self.ssm = SelectiveSSM(d_inner, d_state, dt_rank)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)                        # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)          # each (B, L, d_inner)

        # Depthwise conv (causal)
        x_branch = x_branch.transpose(1, 2)        # (B, d_inner, L)
        x_branch = self.conv1d(x_branch)[..., :x.shape[1]]  # trim
        x_branch = x_branch.transpose(1, 2)        # (B, L, d_inner)
        x_branch = self.act(x_branch)

        y = self.ssm(x_branch)
        y = y * self.act(z)
        return self.out_proj(y) + residual


class MambaBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.layers = nn.ModuleList([
            MambaBlock(cfg["d_model"], cfg["d_state"], cfg["d_conv"], cfg["dt_rank"])
            for _ in range(cfg["n_layers"])
        ])
        self.norm_f = nn.LayerNorm(cfg["d_model"])
        self.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        n_params = sum(p.numel() for p in self.parameters())
        cfg["n_params"] = n_params
        print(f"Mamba baseline: {n_params:,} params")

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm_f(h)
        return self.lm_head(h)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TinyStoriesDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start: start + self.seq_len]
        y = self.tokens[start + 1: start + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def load_tinystories(seq_len, cache_dir="/workspace/tinystories_cache"):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    log("Loading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories", cache_dir=cache_dir)

    log("Tokenizing train split...")
    train_tokens = []
    for item in ds["train"]:
        train_tokens.extend(enc.encode(item["text"]) + [50256])

    log("Tokenizing validation split...")
    val_tokens = []
    for item in ds["validation"]:
        val_tokens.extend(enc.encode(item["text"]) + [50256])

    log(f"Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}")
    return (TinyStoriesDataset(train_tokens, seq_len),
            TinyStoriesDataset(val_tokens, seq_len))


# ─── LR Schedule ──────────────────────────────────────────────────────────────

def get_lr(step, warmup, total, max_lr):
    if step < warmup:
        return max_lr * step / warmup
    progress = (step - warmup) / (total - warmup)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ─── Eval ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, max_batches=50):
    model.eval()
    total_loss, n = 0.0, 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, CFG["vocab_size"]), y.reshape(-1))
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / n if n > 0 else float("inf")


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    log("=== Mamba Baseline Training ===")
    log(f"Config: {json.dumps(CFG, indent=2)}")

    train_ds, val_ds = load_tinystories(CFG["seq_len"])
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                            num_workers=2, pin_memory=True, drop_last=True)

    model = MambaBaseline(CFG).to(DEVICE)
    log(f"Model: {CFG['n_params']:,} params | Device: {DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                   betas=(0.9, 0.95), weight_decay=0.1)

    # Save config
    with open(f"{OUT_DIR}/config.json", "w") as f:
        json.dump(CFG, f, indent=2)

    best_val_ce = float("inf")
    best_step = 0
    train_iter = iter(train_loader)
    step = 0
    t0 = time.time()

    while step < CFG["steps"]:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)

        lr = get_lr(step, CFG["warmup_steps"], CFG["steps"], CFG["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, CFG["vocab_size"]), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1

        if step % 50 == 0:
            elapsed = time.time() - t0
            tok_s = step * CFG["batch_size"] * CFG["seq_len"] / elapsed
            log(f"  step {step:6d}/{CFG['steps']} | ce={loss.item():.4f} | lr={lr:.2e} | {tok_s:.0f} tok/s")

        if step % CFG["val_every"] == 0:
            val_ce = evaluate(model, val_loader)
            log(f"  Val CE: {val_ce:.4f} (step {step})")

            if val_ce < best_val_ce:
                best_val_ce = val_ce
                best_step = step
                torch.save({"model_state_dict": model.state_dict(),
                            "config": CFG,
                            "step": step,
                            "val_ce": val_ce},
                           f"{OUT_DIR}/best/model.pt")
                log(f"  NEW BEST: {val_ce:.4f} @ step {step}")

        if step % CFG["save_every"] == 0:
            ckpt_dir = f"{OUT_DIR}/step_{step:06d}"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(),
                        "config": CFG,
                        "step": step},
                       f"{ckpt_dir}/model.pt")

    log(f"=== TRAINING COMPLETE ===")
    log(f"Best Val CE: {best_val_ce:.4f} @ step {best_step}")
    return best_val_ce


if __name__ == "__main__":
    train()
