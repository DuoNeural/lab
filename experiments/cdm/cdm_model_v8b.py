#!/usr/bin/env python3
"""
cdm_model_v8b.py — CDM V8-B: HORN + Kuramoto with PER-SLOT β_k.
DuoNeural / Archon — 2026-06-17

CDM V7 had a single scalar β per layer. V7 found β < 0 (temporal lead routing)
with magnitude correlating with slot oscillatory regime (most underdamped → most negative β).

V8-B promotes β from per-layer scalar to per-SLOT vector β_k ∈ ℝ^K.
Hypothesis: underdamped slots (ω_k > γ_k) benefit most from temporal lead routing
and will develop more negative β_k than overdamped slots.

Testable prediction:
  β_k < -C * (ω_k/γ_k - 1)^+    (β_k more negative when more underdamped)

If this holds at convergence, the optimal temporal lead is PREDICTABLE from slot dynamics
before training — a formal connection between β and oscillatory regime.

Architecture: identical to CDM V7 except:
  V7:  self.beta = nn.Parameter(torch.tensor(β_init))          # scalar
  V8B: self.beta = nn.Parameter(torch.full((K,), β_init))      # (K,) vector
       anchor = slots + self.beta.view(1, K, 1) * vel

Parameter count: +K per layer vs V7 (+128 total for 8 layers, K=16).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from cdm_model_v2 import CausalSelfAttention, SlotCrossAttention, FFN, marginal_entropy_loss
from cdm_model_v3 import CDMConfigV3, load_balancing_loss


@dataclass
class CDMConfigV8B(CDMConfigV3):
    d_osc:        int   = 8
    beta_init:    float = -0.3   # init at V7 mean value (layer-avg @ step 10000)
    eps:          float = 1e-6


class CDMV8BSlotModule(nn.Module):
    """HORN slots + Kuramoto routing. β_k is per-SLOT (K scalars per layer)."""

    def __init__(self, cfg: CDMConfigV8B, dt: float = 1.0):
        super().__init__()
        self.K     = cfg.K
        self.d     = cfg.d_model
        self.dt    = dt
        self.d_osc = cfg.d_osc
        self.eps   = cfg.eps

        self.write_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.eta = nn.Linear(cfg.d_model, 1, bias=True)
        nn.init.constant_(self.eta.bias, -2.0)

        _init = torch.log(torch.expm1(torch.tensor(0.5)))
        self.raw_gamma = nn.Parameter(torch.full((cfg.K,), float(_init)))
        self.raw_omega = nn.Parameter(torch.full((cfg.K,), float(_init)))

        self.F_proj = nn.Linear(cfg.d_model, cfg.d_osc, bias=False)
        self.G_proj = nn.Linear(cfg.d_model, cfg.d_osc, bias=False)

        # V8-B KEY CHANGE: per-slot β_k (K scalars, not 1)
        self.beta = nn.Parameter(torch.full((cfg.K,), float(cfg.beta_init)))

        self.slot_init = nn.Parameter(torch.zeros(cfg.K, cfg.d_model))
        nn.init.normal_(self.slot_init, std=0.02)

    def kuramoto_route(self, h_t, slots, vel, return_probe=False):
        """
        h_t:   (B, d)
        slots: (B, K, d)
        vel:   (B, K, d)
        """
        # Per-slot temporal lead: each slot has its own β_k
        beta_k = self.beta.view(1, self.K, 1)          # (1, K, 1)
        anchor = slots + beta_k * vel                   # (B, K, d)

        h_proj = F.normalize(self.F_proj(h_t), dim=-1, eps=self.eps)
        a_proj = F.normalize(
            self.G_proj(anchor.reshape(-1, self.d)), dim=-1, eps=self.eps
        ).view(-1, self.K, self.d_osc)

        logits   = torch.bmm(h_proj.unsqueeze(1), a_proj.transpose(1, 2)).squeeze(1)
        logits   = logits / math.sqrt(self.d_osc)
        coupling = F.softplus(logits) + self.eps

        h_bar  = torch.bmm(coupling.unsqueeze(1), a_proj).squeeze(1)
        h_norm = h_bar.norm(dim=-1, keepdim=True)
        z_star = h_bar / h_norm.clamp_min(self.eps)

        cos_sim = torch.bmm(z_star.unsqueeze(1), a_proj.transpose(1, 2)).squeeze(1)
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
            "h_bar_norm_mean":       h_norm.mean().detach(),
            "h_bar_degenerate_frac": degenerate.float().mean().detach(),
            "coupling_mean":         coupling.mean().detach(),
            "coupling_std":          coupling.std(unbiased=False).detach(),
            "coupling_max":          coupling.max().detach(),
            "winner_alignment_mean": win_cos.mean().detach(),
            "beta_mean":             self.beta.detach().mean().abs(),
            "beta_min":              self.beta.detach().min(),
            "beta_max":              self.beta.detach().max(),
        }
        return route_probs, probe

    def _verlet_step(self, s, v, force_t):
        gamma  = F.softplus(self.raw_gamma).view(1, self.K, 1)
        omega2 = F.softplus(self.raw_omega).square().view(1, self.K, 1)
        dt     = self.dt
        a0     = force_t - omega2 * s - 2.0 * gamma * v
        v_half = v + 0.5 * dt * a0
        s_new  = s + dt * v_half
        a1     = force_t - omega2 * s_new - 2.0 * gamma * v_half
        v_new  = v_half + 0.5 * dt * a1
        return s_new, v_new

    def forward(self, h, collect_probe=False):
        B, T, d = h.shape
        K = self.K
        writes   = self.write_proj(h)
        eta_all  = torch.sigmoid(self.eta(h)).squeeze(-1)

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

            eta_t  = eta_all[:, t]
            gate_t = route_t * eta_t.unsqueeze(-1)
            force_t = gate_t.unsqueeze(-1) * writes[:, t].unsqueeze(1)
            s, v = self._verlet_step(s, v, force_t)
            routes_all[:, t] = route_t
            gates_all[:, t]  = gate_t

            if collect_probe:
                for key, val in probe_t.items():
                    probe_sums[key] = probe_sums.get(key, val.new_zeros(())) + val

        if collect_probe:
            self.last_probe = {k: (pv / T).float() for k, pv in probe_sums.items()}

        return slots_all, gates_all, routes_all

    def step(self, h_t, prev_s, prev_v):
        write_t = self.write_proj(h_t)
        eta_t   = torch.sigmoid(self.eta(h_t)).squeeze(-1)
        route_t = self.kuramoto_route(h_t, prev_s, prev_v)
        gate_t  = route_t * eta_t.unsqueeze(-1)
        force_t = gate_t.unsqueeze(-1) * write_t.unsqueeze(1)
        s_new, v_new = self._verlet_step(prev_s, prev_v, force_t)
        return s_new, v_new, prev_s.unsqueeze(1), gate_t


class CDMBlockV8B(nn.Module):
    def __init__(self, cfg: CDMConfigV8B):
        super().__init__()
        self.cdm        = CDMV8BSlotModule(cfg)
        self.self_attn  = CausalSelfAttention(cfg)
        self.slot_xattn = SlotCrossAttention(cfg)
        self.ffn        = FFN(cfg)
        self.norm_sa    = nn.RMSNorm(cfg.d_model)
        self.norm_sx    = nn.RMSNorm(cfg.d_model)
        self.norm_cdm   = nn.RMSNorm(cfg.d_model)
        self.norm_ff    = nn.RMSNorm(cfg.d_model)
        self.dropout    = nn.Dropout(cfg.dropout)

    def forward(self, x, collect_probe=False):
        slots_all, gates, routes = self.cdm(self.norm_cdm(x), collect_probe=collect_probe)
        sa_out = self.self_attn(self.norm_sa(x))
        sx_out = self.slot_xattn(self.norm_sx(x), slots_all)
        x = x + self.dropout(sa_out + sx_out)
        x = x + self.ffn(self.norm_ff(x))
        return x, gates, routes


class CDMLanguageModelV8B(nn.Module):
    def __init__(self, cfg: CDMConfigV8B):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([CDMBlockV8B(cfg) for _ in range(cfg.n_layers)])
        self.norm   = nn.RMSNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        self.head.weight = self.embed.weight

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, collect_probe=False):
        h = self.embed(idx)
        all_gates, all_routes = [], []
        for block in self.blocks:
            h, gates, routes = block(h, collect_probe=collect_probe)
            all_gates.append(gates)
            all_routes.append(routes)
        h = self.norm(h)
        logits = self.head(h)
        return logits, all_gates, all_routes

    def forward_loss(self, idx, collect_probe=False):
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
        stats = {}
        for l, block in enumerate(self.blocks):
            with torch.no_grad():
                g = F.softplus(block.cdm.raw_gamma).cpu().numpy()
                w = F.softplus(block.cdm.raw_omega).cpu().numpy()
                b = block.cdm.beta.detach().cpu().numpy()
            stats[l] = {
                "gamma_mean":      float(g.mean()), "gamma_std": float(g.std()),
                "omega_mean":      float(w.mean()), "omega_std": float(w.std()),
                "underdamped_frac": float((w > g).mean()),
                # per-slot β stats
                "beta_mean":  float(b.mean()),
                "beta_min":   float(b.min()),
                "beta_max":   float(b.max()),
                "beta_std":   float(b.std()),
                # key scientific prediction: correlation of β with oscillatory engagement
                "beta_underdamped_mean": float(b[w > g].mean()) if (w > g).any() else float("nan"),
                "beta_overdamped_mean":  float(b[w <= g].mean()) if (w <= g).any() else float("nan"),
            }
        return stats

    def get_kuramoto_probe_stats(self):
        stats = {}
        for l, block in enumerate(self.blocks):
            probe = getattr(block.cdm, "last_probe", None)
            if probe:
                stats[l] = {k: float(v) for k, v in probe.items()}
        return stats


if __name__ == "__main__":
    cfg = CDMConfigV8B(
        vocab_size=50257, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
        d_ff=256, K=8, max_len=64, dropout=0.0, d_osc=8, lbl_coeff=0.01,
        entropy_reg=0.02, beta_init=-0.3,
    )
    model = CDMLanguageModelV8B(cfg)
    print(f"CDM V8-B sanity: {model.param_count:,} params")
    idx = torch.randint(0, 50257, (2, 32))
    ce, lbl, total = model.forward_loss(idx, collect_probe=True)
    print(f"CE={ce.item():.4f} | LBL={lbl.item():.4f} | total={total.item():.4f}")
    osc = model.get_oscillator_stats()
    print(f"Layer 0 γ={osc[0]['gamma_mean']:.3f}, β_mean={osc[0]['beta_mean']:.3f}, β_min={osc[0]['beta_min']:.3f}")
    print("CDM V8-B OK")
