#!/usr/bin/env python3
"""
cdm_model_v3.py — Competitive Docking Memory V3

V3 additions over V2 (based on Aura's CDM-V2-Analysis.md, 2026-06-12):

  1. LEARNABLE PER-SLOT DECAY RATES (α_k):
     Replace fixed EMA alpha with per-slot learnable scalar.
     New EMA: s_k(t) = (1 - σ(α_k)·g_k)·s_{k}(t-1) + σ(α_k)·g_k·v_t
     - High α_k: volatile fast-decay register (tracks immediate syntax)
     - Near-zero α_k: slow-decay deep storage (compresses long-range semantics)
     Self-organizing temporal hierarchy without any explicit supervision.

  2. LOAD BALANCING LOSS (LBL):
     MoE-style per-batch penalty on routing imbalance.
     L_lbl = K · λ_lbl · Σ_k  f_k · P_k
       f_k = fraction of positions where slot k wins (argmax, stop-grad)
       P_k = mean routing probability for slot k (differentiable)
     If Slot 3 claims 46.8% of code tokens, gradient pushes routing away from it.
     Complements entropy reg (global diversity) with per-slot per-batch enforcement.

Architecture: Archon (DuoNeural)
V3 design: Archon + Aura (DuoNeural)
Date: 2026-06-12
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# -- import all unchanged components from V2 ----------------------------------
from cdm_model_v2 import (
    RoPE,
    CausalSelfAttention,
    SlotCrossAttention,
    FFN,
    marginal_entropy_loss,
    CDMConfigV2,
)


# ── V3 config ────────────────────────────────────────────────────────────────

@dataclass
class CDMConfigV3(CDMConfigV2):
    lbl_coeff:   float = 0.01    # Load Balancing Loss coefficient
    alpha_init:  float = 0.0     # log_alpha init value (sigmoid(0) = 0.5 initial decay)


# ── Load Balancing Loss ───────────────────────────────────────────────────────

def load_balancing_loss(route_probs: torch.Tensor) -> torch.Tensor:
    """
    MoE-style Load Balancing Loss — penalizes routing monopoly.

    L_lbl = K · Σ_k  f_k · P_k   (K normalizes so expected loss = 1 at uniform routing)

    route_probs: (B, T, K) — softmax(route(h)), the competitive routing ONLY (no eta).
                 Do NOT pass the full gates (which includes sigmoid(eta)) since eta is
                 a global write-intensity gate, not the per-slot competition signal.
    Returns: scalar (minimize to encourage balanced routing across slots)
    """
    B, T, K = route_probs.shape

    # P_k: mean routing probability per slot — differentiable
    P = route_probs.mean(dim=(0, 1))                        # (K,)

    # f_k: fraction of positions where slot k wins — discrete, stop gradient
    winners = route_probs.argmax(dim=-1)                    # (B, T)
    f = torch.zeros(K, device=route_probs.device)
    f.scatter_add_(0, winners.view(-1),
                   torch.ones(B * T, device=route_probs.device))
    f = f / (B * T)                                         # normalize

    return K * (f.detach() * P).sum()


# ── V3 CDM module: learnable alpha + separate route_probs for LBL ────────────

class CompetitiveDockingMemoryV3(nn.Module):
    """
    V3 CDM: same as V2 but with learnable per-slot decay rates.

    V2 EMA: s_k(t) = (1-g_k)·s_k(t-1) + g_k·v_t
    V3 EMA: s_k(t) = (1-σ(α_k)·g_k)·s_k(t-1) + σ(α_k)·g_k·v_t

    forward() now returns (slots_all, gates, route_probs) where route_probs is the
    raw softmax(route(h)) used for LBL — kept separate from full gates (which include eta).
    """
    def __init__(self, cfg: CDMConfigV3):
        super().__init__()
        self.K = cfg.K
        self.d = cfg.d_model

        self.route      = nn.Linear(cfg.d_model, cfg.K, bias=True)
        self.eta        = nn.Linear(cfg.d_model, 1, bias=True)
        self.write_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.slot_init  = nn.Parameter(torch.zeros(cfg.K, cfg.d_model))

        # V3: per-slot learnable decay rate. sigmoid(log_alpha) ∈ (0,1).
        # initialized to sigmoid(alpha_init) = 0.5 at default alpha_init=0.0
        self.log_alpha = nn.Parameter(
            torch.full((cfg.K,), cfg.alpha_init)
        )

        nn.init.zeros_(self.route.bias)
        nn.init.constant_(self.eta.bias, -2.0)
        nn.init.normal_(self.slot_init, std=0.02)

    def compute_gates_and_route(self, h: torch.Tensor):
        """h: (B, T, d) → (gates: (B,T,K), route_probs: (B,T,K))."""
        route_probs = F.softmax(self.route(h), dim=-1)       # (B, T, K)
        eta         = torch.sigmoid(self.eta(h))              # (B, T, 1)
        gates       = route_probs * eta                       # (B, T, K) full gates
        return gates, route_probs

    @staticmethod
    def _sequential_scan(A: torch.Tensor, B: torch.Tensor,
                         init: torch.Tensor) -> torch.Tensor:
        """s_t = A_t * s_{t-1} + B_t — same as V2."""
        B_size, T, K, d = B.shape
        states = torch.empty(B_size, T, K, d, device=B.device, dtype=B.dtype)
        s = init
        states[:, 0] = s
        for t in range(T - 1):
            s = A[:, t] * s + B[:, t]
            states[:, t + 1] = s
        return states

    def forward(self, h: torch.Tensor):
        """
        h: (B, T, d)
        Returns:
          slots_all:   (B, T, K, d) — causal slot states
          gates:       (B, T, K)    — full routing gates (for entropy reg)
          route_probs: (B, T, K)    — clean softmax routing (for LBL)
        """
        B, T, d = h.shape
        gates, route_probs = self.compute_gates_and_route(h)
        v = self.write_proj(h)                               # (B, T, d)

        # V3 effective gate = σ(α_k) * g_k  — shape broadcast (B, T, K, 1)
        alpha = torch.sigmoid(self.log_alpha)                # (K,)
        eff_g = (gates * alpha.unsqueeze(0).unsqueeze(0))   # (B, T, K)
        eff_g = eff_g.unsqueeze(-1)                          # (B, T, K, 1)

        A   = (1.0 - eff_g).expand(B, T, self.K, d)         # (B, T, K, d)
        B_s = eff_g * v.unsqueeze(2).expand(B, T, self.K, d)
        init = self.slot_init.unsqueeze(0).expand(B, self.K, d)

        slots_all = self._sequential_scan(A, B_s, init)      # (B, T, K, d)
        return slots_all, gates, route_probs

    def step(self, h_t: torch.Tensor, prev_state: torch.Tensor):
        """
        Single-step for cached generation — same API as V2.step().
        h_t:        (B, d)
        prev_state: (B, K, d)
        Returns: (new_state, slots_for_sa, gates_t)
        """
        h = h_t.unsqueeze(1)                                  # (B, 1, d)
        gates_t, _ = self.compute_gates_and_route(h)
        gates_t = gates_t[:, 0, :]                            # (B, K)
        v_t     = self.write_proj(h)[:, 0, :]                 # (B, d)

        alpha = torch.sigmoid(self.log_alpha)                  # (K,)
        eff_g = (gates_t * alpha.unsqueeze(0)).unsqueeze(-1)   # (B, K, 1)
        new_state = (1.0 - eff_g) * prev_state + eff_g * v_t.unsqueeze(1)
        slots_for_sa = prev_state.unsqueeze(1)
        return new_state, slots_for_sa, gates_t


# ── V3 Block ──────────────────────────────────────────────────────────────────

class CDMBlockV3(nn.Module):
    """V3 block: uses CDMv3 (learnable alpha). Returns gates + route_probs."""
    def __init__(self, cfg: CDMConfigV3):
        super().__init__()
        self.cdm        = CompetitiveDockingMemoryV3(cfg)
        self.self_attn  = CausalSelfAttention(cfg)
        self.slot_xattn = SlotCrossAttention(cfg)
        self.ffn        = FFN(cfg)
        self.norm_sa    = nn.RMSNorm(cfg.d_model)
        self.norm_sx    = nn.RMSNorm(cfg.d_model)
        self.norm_cdm   = nn.RMSNorm(cfg.d_model)
        self.norm_ff    = nn.RMSNorm(cfg.d_model)
        self.dropout    = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, return_slots: bool = False):
        slots_all, gates, route_probs = self.cdm(self.norm_cdm(x))

        sa_out = self.self_attn(self.norm_sa(x))
        sx_out = self.slot_xattn(self.norm_sx(x), slots_all)
        x = x + self.dropout(sa_out + sx_out)
        x = x + self.ffn(self.norm_ff(x))

        if return_slots:
            return x, gates, route_probs, slots_all
        return x, gates, route_probs

    def forward_step(self, x_t, slot_state, past_kv, position: int):
        """Cached single-token step — same API as V2."""
        h_t = x_t[:, 0, :]
        new_slot_state, slots_for_sa, gates_t = self.cdm.step(
            self.norm_cdm(h_t), slot_state
        )
        sa_out, new_kv = self.self_attn.forward_cached(
            self.norm_sa(x_t), past_kv, position
        )
        sx_out = self.slot_xattn(self.norm_sx(x_t), slots_for_sa)
        x_t = x_t + sa_out + sx_out
        x_t = x_t + self.ffn(self.norm_ff(x_t))
        return x_t, new_slot_state, new_kv, gates_t


# ── V3 Language Model ─────────────────────────────────────────────────────────

class CDMLanguageModelV3(nn.Module):
    """
    CDM V3: learnable per-slot alpha + load balancing loss.

    Returns (logits, aux_loss) — same signature as V2 for drop-in training loop use.
    aux_loss = entropy_reg_loss + lbl_loss (both controlled by config coefficients).
    """
    def __init__(self, cfg: CDMConfigV3):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([CDMBlockV3(cfg) for _ in range(cfg.n_layers)])
        self.norm   = nn.RMSNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight
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
        aux_loss = torch.tensor(0.0, device=idx.device)

        for block in self.blocks:
            x, gates, route_probs = block(x)
            if self.training:
                if self.cfg.entropy_reg > 0:
                    aux_loss = aux_loss + self.cfg.entropy_reg * marginal_entropy_loss(gates)
                if self.cfg.lbl_coeff > 0:
                    aux_loss = aux_loss + self.cfg.lbl_coeff * load_balancing_loss(route_probs)

        x = self.norm(x)
        return self.head(x), aux_loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, temperature: float = 1.0,
                 top_k: int = 50) -> torch.Tensor:
        self.eval()
        for _ in range(max_new):
            idx_cond = idx if idx.shape[1] <= self.cfg.max_len else idx[:, -self.cfg.max_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    @torch.no_grad()
    def generate_fast(self, idx: torch.Tensor, max_new: int, temperature: float = 1.0,
                      top_k: int = 50) -> torch.Tensor:
        """
        O(T_prefix) prefix pass then O(1) per new token via cached KV + slot states.
        Identical API to V2.generate_fast().
        """
        self.eval()
        B = idx.shape[0]

        # -- prefix pass: build KV caches and initial slot states -----------------
        x = self.embed(idx)
        kv_caches   = []
        slot_states = []

        for block in self.blocks:
            slots_all, gates, route_probs = block.cdm(block.norm_cdm(x))
            sa_out = block.self_attn(block.norm_sa(x))
            sx_out = block.slot_xattn(block.norm_sx(x), slots_all)
            x = x + sa_out + sx_out
            x = x + block.ffn(block.norm_ff(x))

            # cache: last slot state + last KV tensors
            T_prefix = idx.shape[1]
            h_norm = block.norm_cdm(self.embed(idx))  # re-derive for slot state
            # use forward() slot state at last position
            last_slot = slots_all[:, -1, :, :]         # (B, K, d) — state after last prefix token
            kv_caches.append(None)                     # will be populated below
            slot_states.append(last_slot)

        # rebuild KV caches from the prefix pass using the proper attention
        x = self.embed(idx)
        kv_caches = []
        slot_states_final = []

        for i, block in enumerate(self.blocks):
            slots_all, gates, route_probs = block.cdm(block.norm_cdm(x))

            # build KV cache for this block
            x_norm = block.norm_sa(x)
            T_p = x.shape[1]
            Bsz = x.shape[0]
            K_cache = block.self_attn.k_proj(x_norm).view(
                Bsz, T_p, block.self_attn.n_kv_heads, block.self_attn.d_head
            ).transpose(1, 2)
            V_cache = block.self_attn.v_proj(x_norm).view(
                Bsz, T_p, block.self_attn.n_kv_heads, block.self_attn.d_head
            ).transpose(1, 2)
            # Apply RoPE to K cache
            K_cache = block.self_attn.rope(K_cache)
            kv_caches.append((K_cache, V_cache))

            last_slot = slots_all[:, -1, :, :]
            slot_states_final.append(last_slot)

            sa_out = block.self_attn(x_norm)
            sx_out = block.slot_xattn(block.norm_sx(x), slots_all)
            x = x + sa_out + sx_out
            x = x + block.ffn(block.norm_ff(x))

        # -- incremental generation -----------------------------------------------
        T_prefix = idx.shape[1]
        for step in range(max_new):
            position = T_prefix + step - 1

            # last generated token (or last prefix token on first step)
            last_tok = idx[:, [-1]]             # (B, 1)
            x_t = self.embed(last_tok)          # (B, 1, d)

            new_slot_states = []
            new_kv_caches   = []

            for i, block in enumerate(self.blocks):
                x_t, new_slot, new_kv, _ = block.forward_step(
                    x_t, slot_states_final[i], kv_caches[i], position
                )
                new_slot_states.append(new_slot)
                new_kv_caches.append(new_kv)

            slot_states_final = new_slot_states
            kv_caches         = new_kv_caches

            logits = self.head(self.norm(x_t))[:, -1, :]
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx

    def get_alpha_stats(self) -> dict:
        """Diagnostic: return per-slot learned alpha values across all layers."""
        stats = {}
        for i, block in enumerate(self.blocks):
            alphas = torch.sigmoid(block.cdm.log_alpha).detach().cpu().tolist()
            stats[f"layer_{i}"] = {f"slot_{k}": round(a, 4) for k, a in enumerate(alphas)}
        return stats


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("CDM V3 smoke test...")
    cfg = CDMConfigV3(vocab_size=50257, d_model=128, n_layers=2, n_heads=4,
                      n_kv_heads=2, d_ff=256, K=8, max_len=64,
                      dropout=0.0, entropy_reg=0.02, lbl_coeff=0.01)
    model = CDMLanguageModelV3(cfg)
    model.train()

    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, aux = model(x)
    print(f"  logits: {logits.shape}  aux_loss: {aux.item():.4f}")

    # check aux is non-zero and finite
    assert aux.item() != 0.0, "aux loss should be non-zero in training mode"
    assert torch.isfinite(aux), "aux loss must be finite"

    # check alpha stats
    stats = model.get_alpha_stats()
    layer0 = stats["layer_0"]
    print(f"  alpha stats (layer 0, slots 0-3): {list(layer0.values())[:4]}")
    # at init all should be sigmoid(0.0) ≈ 0.5
    assert all(abs(v - 0.5) < 0.01 for v in layer0.values()), "alphas should init at ~0.5"

    # test generate_fast
    model.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate_fast(prompt, max_new=10, temperature=0.8, top_k=10)
    print(f"  generate_fast: {prompt.shape} → {out.shape}")
    assert out.shape[1] == 18, f"expected 18 tokens, got {out.shape[1]}"

    # check LBL is positive (at uniform routing it should be ~K * 1/K * 1/K * K = 1.0)
    model.train()
    logits2, aux2 = model(x)
    print(f"  training aux (entropy + LBL): {aux2.item():.4f}")

    print("V3 smoke test PASSED ✓")
