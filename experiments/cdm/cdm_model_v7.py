#!/usr/bin/env python3
"""
cdm_model_v7.py — CDM V7: HORN slot dynamics + Kuramoto routing over (S, V) state.
DuoNeural / Archon — 2026-06-17

Synthesis of:
  CDM V6 HORN  (cdm_model_v6_horn.py)  — 2nd order slot dynamics, Störmer-Verlet
  CDM-Kuramoto (cdm_model_kuramoto.py) — physics-derived routing, Born-Oppenheimer

Key new idea: routing uses BOTH slot position S and slot velocity V as anchor.
  z_anchor_k = F_proj(S_k + β * V_k)     where β is a learnable scalar

Rationale:
  - S_k encodes what the slot currently holds
  - V_k encodes what the slot is actively becoming (its "momentum")
  - A slot moving toward a token (high V·h alignment) is more receptive to it
  - This adds phase information that static-position routing cannot express

Architecture per layer:
  1. Kuramoto routing over (S, V) oscillator state:
       z_anchor_k = normalize(G_proj(S_k + β*V_k))      # (B, K, d_osc)
       z*         = normalize(Σ_k coupling_k * z_anchor_k)
       route_k    ∝ 1 + cos_sim(F_proj(h_t), z*)
  2. Störmer-Verlet slot update driven by route_k:
       gate_t = route_t * η_t
       force  = gate_t.unsqueeze(-1) * write_proj(h_t).unsqueeze(1)
       [Verlet integration → S_new, V_new]
  3. Standard CDM: self-attn + slot cross-attn + FFN

This is a fully physics-derived architecture:
  - No trained routing gate (Born-Oppenheimer: routing equilibrates at each config)
  - No trained decay parameter (γ_k, ω_k are the only dynamics params)
  - β is the one new learnable degree of freedom
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

from cdm_model_v2 import CausalSelfAttention, SlotCrossAttention, FFN, marginal_entropy_loss
from cdm_model_v3 import CDMConfigV3, load_balancing_loss


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class CDMConfigV7(CDMConfigV3):
    d_osc:    int   = 8      # Kuramoto oscillator projection dimension
    beta_init: float = 0.0   # initial β (0.0 = pure position routing, same as V3-Kuramoto)
    eps:      float = 1e-6


# ─── HORN slot dynamics (Störmer-Verlet) with Kuramoto routing ────────────────

class CDMV7SlotModule(nn.Module):
    """
    Combined module: Kuramoto routing over (S+βV) + Störmer-Verlet slot update.
    Replaces SecondOrderDockingMemoryV6 (which used softmax gate).
    """

    def __init__(self, cfg: CDMConfigV7, dt: float = 1.0):
        super().__init__()
        self.K    = cfg.K
        self.d    = cfg.d_model
        self.dt   = dt
        self.d_osc = cfg.d_osc
        self.eps  = cfg.eps

        # Write projection (what to write into slot when it wins)
        self.write_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        # η: scalar gate per token (Born-Oppenheimer amplitude)
        self.eta = nn.Linear(cfg.d_model, 1, bias=True)
        nn.init.constant_(self.eta.bias, -2.0)

        # Oscillator dynamics params
        _init = torch.log(torch.expm1(torch.tensor(0.5)))
        self.raw_gamma = nn.Parameter(torch.full((cfg.K,), float(_init)))
        self.raw_omega = nn.Parameter(torch.full((cfg.K,), float(_init)))

        # Kuramoto routing projections
        self.F_proj = nn.Linear(cfg.d_model, cfg.d_osc, bias=False)  # token → oscillator space
        self.G_proj = nn.Linear(cfg.d_model, cfg.d_osc, bias=False)  # slot anchor → oscillator space

        # Velocity mixing parameter β: controls how much V contributes to routing anchor
        self.beta = nn.Parameter(torch.tensor(float(cfg.beta_init)))

        # Learnable slot initialization
        self.slot_init = nn.Parameter(torch.zeros(cfg.K, cfg.d_model))
        nn.init.normal_(self.slot_init, std=0.02)

    def kuramoto_route(self, h_t: torch.Tensor, slots: torch.Tensor, vel: torch.Tensor,
                        return_probe: bool = False):
        """
        h_t:   (B, d)
        slots: (B, K, d) — slot positions S
        vel:   (B, K, d) — slot velocities V
        Returns route_probs: (B, K)
        """
        # Oscillator-space anchor: position + β*velocity
        anchor = slots + self.beta * vel          # (B, K, d)

        # Project token and anchors into oscillator space
        h_proj = F.normalize(self.F_proj(h_t), dim=-1, eps=self.eps)         # (B, d_osc)
        a_proj = F.normalize(self.G_proj(anchor.reshape(-1, self.d)),          # (B*K, d_osc)
                             dim=-1, eps=self.eps).view(-1, self.K, self.d_osc)  # (B, K, d_osc)

        # Token-to-anchor cosine similarities → coupling strengths
        logits   = torch.bmm(h_proj.unsqueeze(1), a_proj.transpose(1, 2)).squeeze(1)
        logits   = logits / math.sqrt(self.d_osc)                             # (B, K)
        coupling = F.softplus(logits) + self.eps

        # Kuramoto mean field z* (Born-Oppenheimer fixed point)
        h_bar  = torch.bmm(coupling.unsqueeze(1), a_proj).squeeze(1)          # (B, d_osc)
        h_norm = h_bar.norm(dim=-1, keepdim=True)                             # (B, 1)
        z_star = h_bar / h_norm.clamp_min(self.eps)

        # Route probabilities ∝ 1 + cos_sim(h_proj, z*) mapped through slot anchors
        cos_sim = torch.bmm(z_star.unsqueeze(1), a_proj.transpose(1, 2)).squeeze(1)
        shifted = (1.0 + cos_sim).clamp_min(self.eps)
        route_probs = shifted / shifted.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        # Fallback for degenerate mean field
        degenerate = h_norm.squeeze(-1) < self.eps
        if degenerate.any():
            uniform = torch.full_like(route_probs, 1.0 / self.K)
            route_probs = torch.where(degenerate.unsqueeze(-1), uniform, route_probs)

        if not return_probe:
            return route_probs

        winners  = route_probs.argmax(dim=-1)
        win_cos  = cos_sim.gather(1, winners.unsqueeze(1)).squeeze(1)
        probe = {
            "h_bar_norm_mean":        h_norm.mean().detach(),
            "h_bar_degenerate_frac":  degenerate.float().mean().detach(),
            "coupling_mean":          coupling.mean().detach(),
            "coupling_std":           coupling.std(unbiased=False).detach(),
            "coupling_max":           coupling.max().detach(),
            "winner_alignment_mean":  win_cos.mean().detach(),
            "beta":                   self.beta.detach().abs(),
        }
        return route_probs, probe

    def _verlet_step(self, s, v, force_t):
        """Single Störmer-Verlet timestep. Returns (s_new, v_new)."""
        gamma  = F.softplus(self.raw_gamma).view(1, self.K, 1)
        omega2 = F.softplus(self.raw_omega).square().view(1, self.K, 1)
        dt     = self.dt

        a0     = force_t - omega2 * s - 2.0 * gamma * v
        v_half = v + 0.5 * dt * a0
        s_new  = s + dt * v_half
        a1     = force_t - omega2 * s_new - 2.0 * gamma * v_half
        v_new  = v_half + 0.5 * dt * a1
        return s_new, v_new

    def forward(self, h: torch.Tensor, collect_probe: bool = False):
        """
        h: (B, T, d)
        Returns: slots_all (B, T, K, d), gates_all (B, T, K), route_all (B, T, K)
        """
        B, T, d = h.shape
        K = self.K

        writes  = self.write_proj(h)                    # (B, T, d)
        eta_all = torch.sigmoid(self.eta(h)).squeeze(-1)  # (B, T)

        s = self.slot_init.unsqueeze(0).expand(B, K, d).clone()
        v = torch.zeros(B, K, d, device=h.device, dtype=h.dtype)

        slots_all  = torch.empty(B, T, K, d, device=h.device, dtype=h.dtype)
        gates_all  = torch.empty(B, T, K,    device=h.device, dtype=h.dtype)
        routes_all = torch.empty(B, T, K,    device=h.device, dtype=h.dtype)

        probe_sums: dict = {}

        for t in range(T):
            slots_all[:, t] = s

            if collect_probe:
                route_t, probe_t = self.kuramoto_route(h[:, t], s, v, return_probe=True)
            else:
                route_t = self.kuramoto_route(h[:, t], s, v, return_probe=False)

            eta_t  = eta_all[:, t]                              # (B,)
            gate_t = route_t * eta_t.unsqueeze(-1)              # (B, K)

            force_t = gate_t.unsqueeze(-1) * writes[:, t].unsqueeze(1)  # (B, K, d)
            s, v = self._verlet_step(s, v, force_t)

            routes_all[:, t] = route_t
            gates_all[:, t]  = gate_t

            if collect_probe:
                for key, val in probe_t.items():
                    probe_sums[key] = probe_sums.get(key, val.new_zeros(())) + val

        if collect_probe:
            self.last_probe = {k: (pv / T).float() for k, pv in probe_sums.items()}

        return slots_all, gates_all, routes_all

    def step(self, h_t: torch.Tensor, prev_s: torch.Tensor, prev_v: torch.Tensor):
        """Single-token step for autoregressive generation."""
        write_t  = self.write_proj(h_t)
        eta_t    = torch.sigmoid(self.eta(h_t)).squeeze(-1)
        route_t  = self.kuramoto_route(h_t, prev_s, prev_v)
        gate_t   = route_t * eta_t.unsqueeze(-1)
        force_t  = gate_t.unsqueeze(-1) * write_t.unsqueeze(1)
        s_new, v_new = self._verlet_step(prev_s, prev_v, force_t)
        return s_new, v_new, prev_s.unsqueeze(1), gate_t


# ─── CDM Block V7 ─────────────────────────────────────────────────────────────

class CDMBlockV7(nn.Module):
    def __init__(self, cfg: CDMConfigV7):
        super().__init__()
        self.cdm       = CDMV7SlotModule(cfg)
        self.self_attn = CausalSelfAttention(cfg)
        self.slot_xattn= SlotCrossAttention(cfg)
        self.ffn       = FFN(cfg)
        self.norm_sa   = nn.RMSNorm(cfg.d_model)
        self.norm_sx   = nn.RMSNorm(cfg.d_model)
        self.norm_cdm  = nn.RMSNorm(cfg.d_model)
        self.norm_ff   = nn.RMSNorm(cfg.d_model)
        self.dropout   = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, collect_probe: bool = False):
        slots_all, gates, routes = self.cdm(self.norm_cdm(x), collect_probe=collect_probe)
        sa_out = self.self_attn(self.norm_sa(x))
        sx_out = self.slot_xattn(self.norm_sx(x), slots_all)
        x = x + self.dropout(sa_out + sx_out)
        x = x + self.ffn(self.norm_ff(x))
        return x, gates, routes


# ─── CDM Language Model V7 ────────────────────────────────────────────────────

class CDMLanguageModelV7(nn.Module):
    def __init__(self, cfg: CDMConfigV7):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([CDMBlockV7(cfg) for _ in range(cfg.n_layers)])
        self.norm   = nn.RMSNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        # Weight tying: head shares embed parameters (saves ~19M params, standard practice)
        self.head.weight = self.embed.weight

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, collect_probe: bool = False):
        h = self.embed(idx)
        all_gates, all_routes = [], []
        for block in self.blocks:
            h, gates, routes = block(h, collect_probe=collect_probe)
            all_gates.append(gates)
            all_routes.append(routes)
        h    = self.norm(h)
        logits = self.head(h)
        return logits, all_gates, all_routes

    def forward_loss(self, idx: torch.Tensor, collect_probe: bool = False):
        """Returns (ce_loss, lbl_loss, total_loss)."""
        logits, all_gates, all_routes = self.forward(idx, collect_probe=collect_probe)
        B, T, V = logits.shape
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), idx[:, 1:].reshape(-1))

        lbl = torch.tensor(0.0, device=idx.device)
        entr = torch.tensor(0.0, device=idx.device)
        for gates in all_gates:
            lbl  = lbl  + load_balancing_loss(gates)
            entr = entr + marginal_entropy_loss(gates)
        lbl_loss = self.cfg.lbl_coeff * lbl / len(self.blocks)
        entr_reg = self.cfg.entropy_reg * entr / len(self.blocks)

        return ce, lbl_loss, ce + lbl_loss + entr_reg

    def get_oscillator_stats(self):
        """Return per-layer γ_k and ω_k statistics."""
        stats = {}
        for l, block in enumerate(self.blocks):
            with torch.no_grad():
                g = F.softplus(block.cdm.raw_gamma).cpu().numpy()
                w = F.softplus(block.cdm.raw_omega).cpu().numpy()
            stats[l] = {
                "gamma_mean": float(g.mean()), "gamma_std": float(g.std()),
                "omega_mean": float(w.mean()), "omega_std": float(w.std()),
                "underdamped_frac": float((w > g).mean()),
                "beta": float(block.cdm.beta.item()),
            }
        return stats

    def get_kuramoto_probe_stats(self):
        stats = {}
        for l, block in enumerate(self.blocks):
            probe = getattr(block.cdm, "last_probe", None)
            if probe:
                stats[l] = {k: float(v) for k, v in probe.items()}
        return stats

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, temperature: float = 1.0,
                 top_k: int = 40) -> torch.Tensor:
        B = idx.shape[0]
        K = self.cfg.K
        d = self.cfg.d_model

        # Cache slot state per layer
        slot_states = [
            block.cdm.slot_init.unsqueeze(0).expand(B, K, d).clone()
            for block in self.blocks
        ]
        vel_states = [
            torch.zeros(B, K, d, device=idx.device, dtype=self.embed.weight.dtype)
            for _ in self.blocks
        ]

        # Prefill prompt
        for t in range(idx.shape[1]):
            h = self.embed(idx[:, t:t+1])
            for l, block in enumerate(self.blocks):
                h_norm = block.norm_cdm(h)
                s_new, v_new, _, _ = block.cdm.step(h_norm.squeeze(1), slot_states[l], vel_states[l])
                slot_states[l] = s_new
                vel_states[l]  = v_new
                sa_kv = block.self_attn(block.norm_sa(h))
                sx    = block.slot_xattn(block.norm_sx(h), s_new.unsqueeze(1))
                h     = h + sa_kv + sx
                h     = h + block.ffn(block.norm_ff(h))

        cur = idx
        for _ in range(max_new):
            h = self.embed(cur[:, -1:])
            for l, block in enumerate(self.blocks):
                h_norm = block.norm_cdm(h)
                s_new, v_new, slots_for_sa, _ = block.cdm.step(
                    h_norm.squeeze(1), slot_states[l], vel_states[l])
                slot_states[l] = s_new
                vel_states[l]  = v_new
                sa_kv = block.self_attn(block.norm_sa(h))
                sx    = block.slot_xattn(block.norm_sx(h), slots_for_sa)
                h     = h + sa_kv + sx
                h     = h + block.ffn(block.norm_ff(h))

            logits = self.head(self.norm(h[:, -1, :]))
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                v_topk, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v_topk[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            cur = torch.cat([cur, next_tok], dim=1)

        return cur


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = CDMConfigV7(
        vocab_size=50257, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
        d_ff=256, K=8, max_len=64, dropout=0.0, d_osc=8, lbl_coeff=0.01,
        entropy_reg=0.02, beta_init=0.0,
    )
    model = CDMLanguageModelV7(cfg)
    print(f"CDM V7 sanity: {model.param_count:,} params")

    idx = torch.randint(0, 50257, (2, 32))
    ce, lbl, total = model.forward_loss(idx, collect_probe=True)
    print(f"CE={ce.item():.4f} | LBL={lbl.item():.4f} | total={total.item():.4f}")

    osc = model.get_oscillator_stats()
    kp  = model.get_kuramoto_probe_stats()
    print(f"Layer 0 γ={osc[0]['gamma_mean']:.3f}, ω={osc[0]['omega_mean']:.3f}, β={osc[0]['beta']:.3f}")
    print(f"Layer 0 Kuramoto winner_alignment={kp.get(0, {}).get('winner_alignment_mean', 'N/A'):.3f}")
    print("CDM V7 OK")
