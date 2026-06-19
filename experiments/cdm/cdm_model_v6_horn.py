#!/usr/bin/env python3
"""
cdm_model_v6_horn.py - scaffold for CDM V6 second-order slot dynamics.

This is intentionally separate from Kuramoto routing. It keeps V3 softmax routing
and replaces first-order EMA slots with damped oscillator slots integrated by a
Störmer-Verlet style update. Slot position S and velocity V reset per sequence,
matching current CDM behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cdm_model_v2 import CausalSelfAttention, SlotCrossAttention, FFN, marginal_entropy_loss
from cdm_model_v3 import CDMConfigV3, load_balancing_loss


class SecondOrderDockingMemoryV6(nn.Module):
    def __init__(self, cfg: CDMConfigV3, dt: float = 1.0):
        super().__init__()
        self.K = cfg.K
        self.d = cfg.d_model
        self.dt = dt
        self.route = nn.Linear(cfg.d_model, cfg.K, bias=True)
        self.eta = nn.Linear(cfg.d_model, 1, bias=True)
        self.write_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.slot_init = nn.Parameter(torch.zeros(cfg.K, cfg.d_model))
        self.vel_init = nn.Parameter(torch.zeros(cfg.K, cfg.d_model), requires_grad=False)

        # softplus(raw) keeps damping/frequency positive; inverse softplus(0.5).
        init = torch.log(torch.expm1(torch.tensor(0.5)))
        self.raw_gamma = nn.Parameter(torch.full((cfg.K,), float(init)))
        self.raw_omega = nn.Parameter(torch.full((cfg.K,), float(init)))

        nn.init.zeros_(self.route.bias)
        nn.init.constant_(self.eta.bias, -2.0)
        nn.init.normal_(self.slot_init, std=0.02)

    def compute_gates_and_route(self, h: torch.Tensor):
        route_probs = F.softmax(self.route(h), dim=-1)
        eta = torch.sigmoid(self.eta(h))
        return route_probs * eta, route_probs

    def _accel(self, force, slots, velocity):
        gamma = F.softplus(self.raw_gamma).view(1, self.K, 1)
        omega2 = F.softplus(self.raw_omega).square().view(1, self.K, 1)
        return force - omega2 * slots - 2.0 * gamma * velocity

    def forward(self, h: torch.Tensor):
        B, T, d = h.shape
        gates, route_probs = self.compute_gates_and_route(h)
        writes = self.write_proj(h)

        slots_all = torch.empty(B, T, self.K, d, device=h.device, dtype=h.dtype)
        s = self.slot_init.unsqueeze(0).expand(B, self.K, d)
        v = self.vel_init.unsqueeze(0).expand(B, self.K, d).to(dtype=h.dtype, device=h.device)

        dt = self.dt
        for t in range(T):
            slots_all[:, t] = s
            force = gates[:, t].unsqueeze(-1) * writes[:, t].unsqueeze(1)
            a0 = self._accel(force, s, v)
            v_half = v + 0.5 * dt * a0
            s_new = s + dt * v_half
            a1 = self._accel(force, s_new, v_half)
            v_new = v_half + 0.5 * dt * a1
            s, v = s_new, v_new

        return slots_all, gates, route_probs

    def step(self, h_t: torch.Tensor, prev_state):
        prev_s, prev_v = prev_state
        h = h_t.unsqueeze(1)
        gates_t, _ = self.compute_gates_and_route(h)
        gates_t = gates_t[:, 0]
        write_t = self.write_proj(h_t)
        force = gates_t.unsqueeze(-1) * write_t.unsqueeze(1)

        dt = self.dt
        a0 = self._accel(force, prev_s, prev_v)
        v_half = prev_v + 0.5 * dt * a0
        s_new = prev_s + dt * v_half
        a1 = self._accel(force, s_new, v_half)
        v_new = v_half + 0.5 * dt * a1
        return (s_new, v_new), prev_s.unsqueeze(1), gates_t


class CDMBlockV6HORN(nn.Module):
    def __init__(self, cfg: CDMConfigV3):
        super().__init__()
        self.cdm = SecondOrderDockingMemoryV6(cfg)
        self.self_attn = CausalSelfAttention(cfg)
        self.slot_xattn = SlotCrossAttention(cfg)
        self.ffn = FFN(cfg)
        self.norm_sa = nn.RMSNorm(cfg.d_model)
        self.norm_sx = nn.RMSNorm(cfg.d_model)
        self.norm_cdm = nn.RMSNorm(cfg.d_model)
        self.norm_ff = nn.RMSNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, return_slots: bool = False):
        slots_all, gates, route_probs = self.cdm(self.norm_cdm(x))
        sa_out = self.self_attn(self.norm_sa(x))
        sx_out = self.slot_xattn(self.norm_sx(x), slots_all)
        x = x + self.dropout(sa_out + sx_out)
        x = x + self.ffn(self.norm_ff(x))
        if return_slots:
            return x, gates, route_probs, slots_all
        return x, gates, route_probs


class CDMLanguageModelV6HORN(nn.Module):
    def __init__(self, cfg: CDMConfigV3):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([CDMBlockV6HORN(cfg) for _ in range(cfg.n_layers)])
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
        return self.head(self.norm(x)), aux_loss

    def get_dynamics_stats(self):
        stats = {}
        for i, block in enumerate(self.blocks):
            gamma = F.softplus(block.cdm.raw_gamma).detach().cpu().tolist()
            omega = F.softplus(block.cdm.raw_omega).detach().cpu().tolist()
            stats[f"layer_{i}"] = {
                "gamma_mean": round(sum(gamma) / len(gamma), 4),
                "omega_mean": round(sum(omega) / len(omega), 4),
                "gamma": [round(x, 4) for x in gamma],
                "omega": [round(x, 4) for x in omega],
            }
        return stats


if __name__ == "__main__":
    cfg = CDMConfigV3(vocab_size=50257, d_model=128, n_layers=2, n_heads=4,
                      n_kv_heads=2, d_ff=256, K=8, max_len=64)
    model = CDMLanguageModelV6HORN(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, aux = model(x)
    assert logits.shape == (2, 32, cfg.vocab_size)
    assert torch.isfinite(aux)
    print("CDM V6 HORN scaffold smoke test PASSED")
