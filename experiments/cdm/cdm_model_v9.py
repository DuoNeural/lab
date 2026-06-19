#!/usr/bin/env python3
"""
cdm_model_v9.py — CDM V9: Dual Timescale (CDM-DT-A)

Mixed slot pool: K_slow HORN DHO slots + K_fast Kuramoto EMA slots.
Unified Kuramoto physics routing over ALL K slots simultaneously.
SlotCrossAttention reads all K with gated velocity injection for slow slots.
Separate LBL tracking per population to detect/prevent fast-slot cannibalization.

Architecture (Kestrel + Aura design):
  - HornRouteAdapter: learned (S,V) → d_model before G_proj — preserves oscillator phase
  - type_embed: per-population learnable tags so routing distinguishes slot types
  - Shared F_proj/G_proj after adapters — keeps all slots on same Kuramoto manifold
  - HORN update: Störmer-Verlet, γ ≤ 1.0, ω ≤ 2.0, dt = sigmoid*0.25
  - EMA update: standard fast decay
  - v_gate: gated velocity → value injection in cross-attention (conservative init)
  - Dual LBL loss: separate over slow/fast to detect cannibalization

Key constraint (Aura red-team):
  Route on S_k only for HORN (position only). V_k is hidden from routing to
  avoid Born-Oppenheimer violations in underdamped regime. V_k is injected
  only into the value stream at cross-attention time (SlotReadout).
  HornRouteAdapter removed in v9.1 — routing on S_slow directly is correct.

DuoNeural / Jesse Caldwell + Archon — 2026-06-19
Inspired by Jesse's insight: "why not both?" (the hippocampus was right there all along)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cdm_model_kuramoto import CDMConfigKuramoto, CausalSelfAttention, SlotCrossAttention, FFN
from cdm_model_kuramoto import marginal_entropy_loss, load_balancing_loss


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class CDMConfigV9(CDMConfigKuramoto):
    K_slow: int = 8          # HORN DHO slots (stores WHAT — content)
    # K_fast = K - K_slow    (Kuramoto EMA slots — stores WHERE — structure)
    lbl_coeff_slow: float = 0.01   # separate LBL weight for slow population
    lbl_coeff_fast: float = 0.01   # separate LBL weight for fast population
    v_gate_init: float = -2.0      # velocity gate init (conservative — near-zero)
    gamma_max: float = 1.0         # damping clamp
    omega_max: float = 2.0         # frequency clamp
    dt_scale: float = 0.25         # dt = sigmoid(log_dt) * dt_scale

    @property
    def K_fast(self) -> int:
        return self.K - self.K_slow


# ── Sub-modules ───────────────────────────────────────────────────────────────

class SlotReadout(nn.Module):
    """Merges slow + fast slot states for cross-attention read.
    Injects HORN velocity into slow slot values via a learned gate.
    """
    def __init__(self, d_model: int, v_gate_init: float = -2.0):
        super().__init__()
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_gate = nn.Parameter(torch.tensor(v_gate_init))

    def forward(
        self,
        S_slow: torch.Tensor,   # (B, K_slow, d)
        V_slow: torch.Tensor,   # (B, K_slow, d)
        S_fast: torch.Tensor,   # (B, K_fast, d)
    ) -> torch.Tensor:          # (B, K, d)
        v_contribution = torch.tanh(self.v_proj(V_slow))
        slow_value = S_slow + torch.sigmoid(self.v_gate) * v_contribution
        return torch.cat([slow_value, S_fast], dim=1)


# ── Core dual-timescale CDM block ─────────────────────────────────────────────

class DualTimescaleCDMBlock(nn.Module):
    """
    The heart of CDM V9. K_slow HORN slots + K_fast Kuramoto EMA slots.
    Unified Kuramoto routing over all K. Separate update rules per population.
    """
    def __init__(self, cfg: CDMConfigV9):
        super().__init__()
        self.K       = cfg.K
        self.K_slow  = cfg.K_slow
        self.K_fast  = cfg.K_fast
        self.d       = cfg.d_model
        self.d_osc   = cfg.d_osc
        self.eps     = cfg.kuramoto_eps
        self.gamma_max = cfg.gamma_max
        self.omega_max = cfg.omega_max
        self.dt_scale  = cfg.dt_scale

        # ── HORN slow-slot parameters ──────────────────────────────────────
        self.log_gamma  = nn.Parameter(torch.zeros(cfg.K_slow))
        self.log_omega  = nn.Parameter(torch.zeros(cfg.K_slow))
        self.log_dt     = nn.Parameter(torch.zeros(cfg.K_slow))
        self.slow_init  = nn.Parameter(torch.zeros(cfg.K_slow, cfg.d_model))
        self.vel_init   = nn.Parameter(torch.zeros(cfg.K_slow, cfg.d_model))

        # ── Kuramoto fast-slot parameters ──────────────────────────────────
        self.log_alpha  = nn.Parameter(torch.full((cfg.K_fast,), cfg.alpha_init))
        self.fast_init  = nn.Parameter(torch.zeros(cfg.K_fast, cfg.d_model))

        # ── Unified Kuramoto routing ───────────────────────────────────────
        # Route on S_slow (position only) per Aura BO constraint — no velocity in routing
        self.type_embed   = nn.Parameter(torch.zeros(2, cfg.d_model))  # 0=slow, 1=fast
        self.F_proj       = nn.Linear(cfg.d_model, cfg.d_osc, bias=False)
        self.G_proj       = nn.Linear(cfg.d_model, cfg.d_osc, bias=False)

        # ── Write pathway ──────────────────────────────────────────────────
        self.eta          = nn.Linear(cfg.d_model, 1, bias=True)
        self.write_proj   = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # ── HORN Störmer-Verlet: force projection ──────────────────────────
        self.write_to_force = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # ── Readout ────────────────────────────────────────────────────────
        self.slot_readout = SlotReadout(cfg.d_model, cfg.v_gate_init)

        # ── Norms (Kestrel: essential for mixed-dynamics stability) ─────────
        self.route_norm = nn.LayerNorm(cfg.d_model)
        self.s_norm     = nn.LayerNorm(cfg.d_model)
        self.v_norm     = nn.LayerNorm(cfg.d_model)

        # init
        nn.init.constant_(self.eta.bias, -2.0)
        nn.init.normal_(self.slow_init, std=0.02)
        nn.init.normal_(self.fast_init, std=0.02)

        self.last_probe: dict = {}

    # ── Kuramoto routing ─────────────────────────────────────────────────────

    def _kuramoto_route(
        self,
        h_t: torch.Tensor,       # (B, d)
        route_state: torch.Tensor, # (B, K, d) — adapter-projected slot states
        return_probe: bool = False,
    ):
        h_proj = F.normalize(self.F_proj(h_t), dim=-1, eps=self.eps)          # (B, d_osc)
        s_proj = F.normalize(self.G_proj(route_state), dim=-1, eps=self.eps)  # (B, K, d_osc)

        logits   = torch.bmm(h_proj.unsqueeze(1), s_proj.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_osc)
        coupling = F.softplus(logits) + self.eps

        h_bar   = torch.bmm(coupling.unsqueeze(1), s_proj).squeeze(1)
        h_norm  = h_bar.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        z_star  = h_bar / h_norm

        cos_sim  = torch.bmm(z_star.unsqueeze(1), s_proj.transpose(1, 2)).squeeze(1)
        shifted  = (1.0 + cos_sim).clamp_min(self.eps)
        route    = shifted / shifted.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        # fallback for degenerate h_bar
        degen = h_norm.squeeze(-1) < self.eps
        if degen.any():
            route = torch.where(degen.unsqueeze(-1),
                                torch.full_like(route, 1.0 / self.K), route)

        if not return_probe:
            return route

        winners  = route.argmax(dim=-1)
        win_cos  = cos_sim.gather(1, winners.unsqueeze(1)).squeeze(1)
        probe = {
            "coupling_mean":         coupling.mean().detach(),
            "coupling_std":          coupling.std(unbiased=False).detach(),
            "winner_alignment_mean": win_cos.mean().detach(),
            "h_bar_norm_mean":       h_norm.mean().detach(),
        }
        return route, probe

    # ── Sequential scan ───────────────────────────────────────────────────────

    def _scan(self, h: torch.Tensor, collect_probe: bool = False):
        B, T, d = h.shape
        v_write = self.write_proj(h)          # (B, T, d)
        eta     = torch.sigmoid(self.eta(h)).squeeze(-1)  # (B, T)

        # hoist loop-invariant HORN parameters (Aura: clamp omega_slow < 0.2 ideal;
        # we clamp via gamma_max/omega_max + learned convergence)
        gamma = F.softplus(self.log_gamma).clamp(max=self.gamma_max).view(1, -1, 1)
        omega = F.softplus(self.log_omega).clamp(max=self.omega_max).view(1, -1, 1)
        dt    = torch.sigmoid(self.log_dt).view(1, -1, 1) * self.dt_scale
        om_sq = omega.square()

        # hoist type embeddings
        te_slow = self.type_embed[0]  # (d,)
        te_fast = self.type_embed[1]  # (d,)

        # hoist EMA alpha
        alpha = torch.sigmoid(self.log_alpha).view(1, -1, 1)

        S_slow = self.slow_init.unsqueeze(0).expand(B, -1, -1).clone()
        V_slow = self.vel_init.unsqueeze(0).expand(B, -1, -1).clone()
        S_fast = self.fast_init.unsqueeze(0).expand(B, -1, -1).clone()

        # output buffers
        slots_all  = torch.empty(B, T, self.K, d, device=h.device, dtype=h.dtype)
        routes_all = torch.empty(B, T, self.K, device=h.device, dtype=h.dtype)
        gates_slow = torch.empty(B, T, self.K_slow, device=h.device, dtype=h.dtype)
        gates_fast = torch.empty(B, T, self.K_fast, device=h.device, dtype=h.dtype)

        # pre-project writes once per position (done outside loop for write_proj above)
        # route_norm is still inside loop but is a single LayerNorm now
        probe_sums: dict = {}

        for t in range(T):
            # ── routing state: S_slow + type_embed[0], S_fast + type_embed[1] ──
            # Aura constraint: use position S only for HORN, hide velocity from routing
            route_state = self.route_norm(
                torch.cat([S_slow + te_slow, S_fast + te_fast], dim=1)
            )

            if collect_probe:
                route_t, probe_t = self._kuramoto_route(h[:, t], route_state, return_probe=True)
            else:
                route_t = self._kuramoto_route(h[:, t], route_state, return_probe=False)

            eta_t  = eta[:, t]                                       # (B,)
            g_slow = route_t[:, :self.K_slow] * eta_t.unsqueeze(-1) # (B, K_slow)
            g_fast = route_t[:, self.K_slow:] * eta_t.unsqueeze(-1) # (B, K_fast)

            # ── HORN Störmer-Verlet ──────────────────────────────────────────
            force  = g_slow.unsqueeze(-1) * self.write_to_force(v_write[:, t]).unsqueeze(1)
            A      = force - gamma * V_slow - om_sq * S_slow
            V_next = V_slow + dt * A
            S_next = S_slow + dt * V_next
            S_slow = self.s_norm(S_next)
            V_slow = self.v_norm(V_next)

            # ── Kuramoto EMA ─────────────────────────────────────────────────
            eff    = alpha * g_fast.unsqueeze(-1)
            S_fast = (1.0 - eff) * S_fast + eff * v_write[:, t].unsqueeze(1)

            # ── Readout ──────────────────────────────────────────────────────
            slots_all[:, t]  = self.slot_readout(S_slow, V_slow, S_fast)
            routes_all[:, t] = route_t
            gates_slow[:, t] = g_slow
            gates_fast[:, t] = g_fast

            if collect_probe:
                for k, v in probe_t.items():
                    probe_sums[k] = probe_sums.get(k, v.new_zeros(())) + v

        if collect_probe:
            self.last_probe = {k: (v / T).float() for k, v in probe_sums.items()}

        return slots_all, gates_slow, gates_fast, routes_all

    def forward(self, h: torch.Tensor, collect_probe: bool = False):
        return self._scan(h, collect_probe=collect_probe)


# ── Full transformer block ────────────────────────────────────────────────────

class CDMBlockV9(nn.Module):
    def __init__(self, cfg: CDMConfigV9):
        super().__init__()
        self.cdm       = DualTimescaleCDMBlock(cfg)
        self.self_attn = CausalSelfAttention(cfg)
        self.slot_xattn = SlotCrossAttention(cfg)
        self.ffn       = FFN(cfg)
        self.norm_sa   = nn.RMSNorm(cfg.d_model)
        self.norm_sx   = nn.RMSNorm(cfg.d_model)
        self.norm_cdm  = nn.RMSNorm(cfg.d_model)
        self.norm_ff   = nn.RMSNorm(cfg.d_model)
        self.dropout   = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, collect_probe: bool = False):
        slots_all, g_slow, g_fast, routes = self.cdm(
            self.norm_cdm(x), collect_probe=collect_probe
        )
        sa_out  = self.self_attn(self.norm_sa(x))
        sx_out  = self.slot_xattn(self.norm_sx(x), slots_all)
        x = x + self.dropout(sa_out + sx_out)
        x = x + self.ffn(self.norm_ff(x))
        return x, g_slow, g_fast, routes


# ── Full language model ───────────────────────────────────────────────────────

class CDMLanguageModelV9(nn.Module):
    def __init__(self, cfg: CDMConfigV9):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([CDMBlockV9(cfg) for _ in range(cfg.n_layers)])
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

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, collect_probe: bool = False):
        x = self.embed(idx)
        aux_slow = torch.tensor(0.0, device=idx.device)
        aux_fast = torch.tensor(0.0, device=idx.device)

        for block in self.blocks:
            x, g_slow, g_fast, routes = block(x, collect_probe=collect_probe)
            if self.training:
                # entropy reg over full routing distribution
                if self.cfg.entropy_reg > 0:
                    aux_slow = aux_slow + self.cfg.entropy_reg * marginal_entropy_loss(g_slow)
                    aux_fast = aux_fast + self.cfg.entropy_reg * marginal_entropy_loss(g_fast)
                # separate LBL per population — Aura's cannibalization fix
                if self.cfg.lbl_coeff_slow > 0:
                    aux_slow = aux_slow + self.cfg.lbl_coeff_slow * load_balancing_loss(
                        routes[:, :, :self.cfg.K_slow]
                    )
                if self.cfg.lbl_coeff_fast > 0:
                    aux_fast = aux_fast + self.cfg.lbl_coeff_fast * load_balancing_loss(
                        routes[:, :, self.cfg.K_slow:]
                    )

        x = self.norm(x)
        logits = self.head(x)
        return logits, aux_slow + aux_fast

    def forward_loss(self, idx: torch.Tensor):
        logits, aux = self(idx)
        B, T, V = logits.shape
        ce  = F.cross_entropy(logits[:, :-1].reshape(-1, V), idx[:, 1:].reshape(-1))
        total = ce + aux
        return ce, aux, total

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        self.eval()
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.cfg.max_len:])
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = CDMConfigV9(
        vocab_size=50257, d_model=384, n_layers=4, n_heads=8, n_kv_heads=4,
        d_ff=1024, K=16, K_slow=8, max_len=64, d_osc=8,
        entropy_reg=0.02, lbl_coeff=0.01, lbl_coeff_slow=0.01, lbl_coeff_fast=0.01,
    )
    model = CDMLanguageModelV9(cfg)
    print(f"CDM V9-A (DT-A): {model.param_count/1e6:.1f}M params | K_slow={cfg.K_slow} K_fast={cfg.K_fast}")

    x = torch.randint(0, cfg.vocab_size, (2, 32))
    ce, aux, total = model.forward_loss(x)
    print(f"Sanity check: ce={ce.item():.4f}  aux={aux.item():.4f}  total={total.item():.4f}")
    total.backward()
    print("Backward: OK")

    logits, _ = model(x, collect_probe=True)
    for i, block in enumerate(model.blocks):
        p = block.cdm.last_probe
        if p:
            print(f"  L{i}: coupling={p['coupling_mean']:.3f}  winner_align={p['winner_alignment_mean']:.3f}")
    print("DONE — CDM V9-A ready.")
