#!/usr/bin/env python3
"""
cdm_model_kuramoto.py - CDM V3 with closed-form Kuramoto routing.

Only architectural change from V3:
  softmax(route(h)) -> positive-coupling Kuramoto routing against current slots.

The EMA slot update, slot cross-attention path, entropy regularizer, and LBL are
kept V3-compatible. Routing is computed inside the sequential scan because the
slot anchors are causal state, not a fixed sequence-wide bank.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cdm_model_v2 import (
    CausalSelfAttention,
    SlotCrossAttention,
    FFN,
    marginal_entropy_loss,
)
from cdm_model_v3 import CDMConfigV3, load_balancing_loss


@dataclass
class CDMConfigKuramoto(CDMConfigV3):
    d_osc: int = 8
    kuramoto_eps: float = 1e-6


class KuramotoCDMBlock(nn.Module):
    """
    V3 CDM memory module with Kuramoto routing.

    h_t routes to slots using positive coupling in oscillator space:
      h_proj = normalize(F_proj(h_t))
      s_proj = normalize(G_proj(S_t))
      w      = softplus(h_proj @ s_proj.T / sqrt(d_osc))
      z*     = normalize(sum_k w_k s_proj_k)
      g_k    = shifted_cos(z*, s_proj_k) / sum_j shifted_cos(...)

    The global write-intensity eta and learnable per-slot alpha are unchanged
    from V3. route_probs is the clean competition signal used for LBL.
    """
    def __init__(self, cfg: CDMConfigKuramoto):
        super().__init__()
        self.K = cfg.K
        self.d = cfg.d_model
        self.d_osc = cfg.d_osc
        self.eps = cfg.kuramoto_eps

        self.F_proj = nn.Linear(cfg.d_model, cfg.d_osc, bias=False)
        self.G_proj = nn.Linear(cfg.d_model, cfg.d_osc, bias=False)
        self.eta = nn.Linear(cfg.d_model, 1, bias=True)
        self.write_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.slot_init = nn.Parameter(torch.zeros(cfg.K, cfg.d_model))
        self.log_alpha = nn.Parameter(torch.full((cfg.K,), cfg.alpha_init))

        nn.init.constant_(self.eta.bias, -2.0)
        nn.init.normal_(self.slot_init, std=0.02)
        self.last_probe = {}

    def kuramoto_route(self, h_t: torch.Tensor, slots: torch.Tensor, return_probe: bool = False):
        """
        h_t:   (B, d)
        slots: (B, K, d), current causal slot states
        Returns route_probs: (B, K), optionally probe metrics.
        """
        h_proj = F.normalize(self.F_proj(h_t), dim=-1, eps=self.eps)          # (B, d_osc)
        s_proj = F.normalize(self.G_proj(slots), dim=-1, eps=self.eps)        # (B, K, d_osc)

        logits = torch.bmm(h_proj.unsqueeze(1), s_proj.transpose(1, 2)).squeeze(1)
        logits = logits / math.sqrt(self.d_osc)                               # (B, K)
        coupling = F.softplus(logits) + self.eps                              # strictly positive

        h_bar = torch.bmm(coupling.unsqueeze(1), s_proj).squeeze(1)           # (B, d_osc)
        h_norm = h_bar.norm(dim=-1, keepdim=True)                             # (B, 1)
        z_star = h_bar / h_norm.clamp_min(self.eps)

        cos_sim = torch.bmm(z_star.unsqueeze(1), s_proj.transpose(1, 2)).squeeze(1)
        shifted = (1.0 + cos_sim).clamp_min(self.eps)
        route_probs = shifted / shifted.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        degenerate = h_norm.squeeze(-1) < self.eps
        if degenerate.any():
            uniform = torch.full_like(route_probs, 1.0 / self.K)
            route_probs = torch.where(degenerate.unsqueeze(-1), uniform, route_probs)

        if not return_probe:
            return route_probs

        winners = route_probs.argmax(dim=-1)
        win_cos = cos_sim.gather(1, winners.unsqueeze(1)).squeeze(1)
        probe = {
            "h_bar_norm_mean": h_norm.mean().detach(),
            "h_bar_degenerate_frac": degenerate.float().mean().detach(),
            "coupling_mean": coupling.mean().detach(),
            "coupling_std": coupling.std(unbiased=False).detach(),
            "coupling_max": coupling.max().detach(),
            "winner_alignment_mean": win_cos.mean().detach(),
        }
        return route_probs, probe

    def _scan(self, h: torch.Tensor, collect_probe: bool):
        B, T, d = h.shape
        v = self.write_proj(h)
        alpha = torch.sigmoid(self.log_alpha)
        eta = torch.sigmoid(self.eta(h)).squeeze(-1)

        slots_all = torch.empty(B, T, self.K, d, device=h.device, dtype=h.dtype)
        gates_all = torch.empty(B, T, self.K, device=h.device, dtype=h.dtype)
        routes_all = torch.empty(B, T, self.K, device=h.device, dtype=h.dtype)
        s = self.slot_init.unsqueeze(0).expand(B, self.K, d)

        probe_sums = {}
        for t in range(T):
            slots_all[:, t] = s
            route_t, probe_t = self.kuramoto_route(h[:, t], s, return_probe=True)
            gate_t = route_t * eta[:, t].unsqueeze(-1)
            eff_g = (gate_t * alpha.unsqueeze(0)).unsqueeze(-1)
            s = (1.0 - eff_g) * s + eff_g * v[:, t].unsqueeze(1)

            routes_all[:, t] = route_t
            gates_all[:, t] = gate_t
            if collect_probe:
                for key, val in probe_t.items():
                    probe_sums[key] = probe_sums.get(key, val.new_zeros(())) + val

        if collect_probe:
            self.last_probe = {k: (v / T).float() for k, v in probe_sums.items()}
        return slots_all, gates_all, routes_all

    def forward(self, h: torch.Tensor, collect_probe: bool = False):
        return self._scan(h, collect_probe=collect_probe)

    def step(self, h_t: torch.Tensor, prev_state: torch.Tensor):
        route_t = self.kuramoto_route(h_t, prev_state)
        eta_t = torch.sigmoid(self.eta(h_t)).squeeze(-1)
        gates_t = route_t * eta_t.unsqueeze(-1)
        v_t = self.write_proj(h_t)
        alpha = torch.sigmoid(self.log_alpha)
        eff_g = (gates_t * alpha.unsqueeze(0)).unsqueeze(-1)
        new_state = (1.0 - eff_g) * prev_state + eff_g * v_t.unsqueeze(1)
        slots_for_sa = prev_state.unsqueeze(1)
        return new_state, slots_for_sa, gates_t


class CDMBlockKuramoto(nn.Module):
    def __init__(self, cfg: CDMConfigKuramoto):
        super().__init__()
        self.cdm = KuramotoCDMBlock(cfg)
        self.self_attn = CausalSelfAttention(cfg)
        self.slot_xattn = SlotCrossAttention(cfg)
        self.ffn = FFN(cfg)
        self.norm_sa = nn.RMSNorm(cfg.d_model)
        self.norm_sx = nn.RMSNorm(cfg.d_model)
        self.norm_cdm = nn.RMSNorm(cfg.d_model)
        self.norm_ff = nn.RMSNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, return_slots: bool = False, collect_probe: bool = False):
        slots_all, gates, route_probs = self.cdm(self.norm_cdm(x), collect_probe=collect_probe)
        sa_out = self.self_attn(self.norm_sa(x))
        sx_out = self.slot_xattn(self.norm_sx(x), slots_all)
        x = x + self.dropout(sa_out + sx_out)
        x = x + self.ffn(self.norm_ff(x))
        if return_slots:
            return x, gates, route_probs, slots_all
        return x, gates, route_probs

    def forward_step(self, x_t, slot_state, past_kv, position: int):
        h_t = x_t[:, 0, :]
        new_slot_state, slots_for_sa, gates_t = self.cdm.step(self.norm_cdm(h_t), slot_state)
        sa_out, new_kv = self.self_attn.forward_cached(self.norm_sa(x_t), past_kv, position)
        sx_out = self.slot_xattn(self.norm_sx(x_t), slots_for_sa)
        x_t = x_t + sa_out + sx_out
        x_t = x_t + self.ffn(self.norm_ff(x_t))
        return x_t, new_slot_state, new_kv, gates_t


class CDMLanguageModelKuramoto(nn.Module):
    def __init__(self, cfg: CDMConfigKuramoto):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([CDMBlockKuramoto(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
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

    def forward(self, idx: torch.Tensor, collect_probe: bool = False):
        x = self.embed(idx)
        aux_loss = torch.tensor(0.0, device=idx.device)
        for block in self.blocks:
            x, gates, route_probs = block(x, collect_probe=collect_probe)
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
                logits[logits < v[:, [-1]]] = float("-inf")
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    def get_alpha_stats(self) -> dict:
        stats = {}
        for i, block in enumerate(self.blocks):
            alphas = torch.sigmoid(block.cdm.log_alpha).detach().cpu().tolist()
            stats[f"layer_{i}"] = {f"slot_{k}": round(a, 4) for k, a in enumerate(alphas)}
        return stats

    def get_kuramoto_probe_stats(self) -> dict:
        stats = {}
        for i, block in enumerate(self.blocks):
            stats[f"layer_{i}"] = {
                k: round(float(v.detach().cpu()), 6)
                for k, v in block.cdm.last_probe.items()
            }
        return stats

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    cfg = CDMConfigKuramoto(
        vocab_size=50257, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
        d_ff=256, K=8, max_len=64, dropout=0.0, d_osc=8,
        entropy_reg=0.02, lbl_coeff=0.01,
    )
    model = CDMLanguageModelKuramoto(cfg)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, aux = model(x, collect_probe=True)
    assert logits.shape == (2, 32, cfg.vocab_size)
    assert torch.isfinite(aux)
    stats = model.get_kuramoto_probe_stats()
    assert "h_bar_norm_mean" in stats["layer_0"]
    print("Kuramoto CDM smoke test PASSED")
