#!/usr/bin/env python3
"""
cdm_routing_probe_v2.py — Slot Routing Analysis for CDM V2

Key V2 difference: causal per-position slots + marginal entropy reg.
V1 finding: 6/8 slots DEAD (routing collapse). K_eff=2.
V2 hypothesis: all K=16 slots active, diverse routing across positions.

This probe captures gate distributions per layer to:
1. Compare routing entropy: V1 (collapsed) vs V2 (should be high)
2. Identify emergent slot specialization (syntax vs semantics etc.)
3. Produce paper Table 2: routing entropy by layer + slot affinity matrix

Archon (DuoNeural) 2026-06-11
"""
import argparse, json, sys, math
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from cdm_model_v2 import CDMLanguageModelV2, CDMConfigV2

PROBE_STORIES = [
    "Once upon a time, there was a little girl named Lily. She lived near a big forest. One day, Lily found a small puppy near the trees. She took it home and fed it milk. Her mom said she could keep it. Lily named the puppy Spot.",
    "Tom was a curious boy who loved to explore. He lived in a house by the river. One afternoon, Tom saw a red boat floating on the water. He jumped in and paddled to the other bank. There he found a shiny golden coin.",
    "Emma had a magic wand. She waved it and flowers grew everywhere. The garden was full of roses and daisies. Her friend Ben came to visit and they played together all afternoon.",
    "A little bear named Max lived in a cozy cave. Every morning he walked to the meadow to eat berries. One day he met a fox who wanted to share his breakfast.",
    "Sophie loved to paint. She had brushes and colors all over her room. One rainy day she painted a big rainbow on her wall. Her cat Whiskers watched her and meowed.",
]


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = CDMConfigV2(**{k: v for k, v in cfg_dict.items() if k in CDMConfigV2.__dataclass_fields__})
    model = CDMLanguageModelV2(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(device), cfg


def get_tokenizer():
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained("gpt2")


def probe_routing(model: CDMLanguageModelV2, tokenizer, text: str, device: str):
    """
    Run forward pass, capturing gate distributions at each block.
    Returns token strings and per-layer gate tensors (B=1, T, K).
    """
    tokens = tokenizer.encode(text)[:model.cfg.max_len]
    token_strs = [tokenizer.decode([t]).strip() for t in tokens]
    ids = torch.tensor([tokens], device=device)

    gates_by_layer = {}  # layer_idx -> (T, K) numpy array

    with torch.no_grad():
        x = model.embed(ids)  # (1, T, d)
        for layer_idx, block in enumerate(model.blocks):
            # Capture gates before the block modifies x
            normed = block.norm_cdm(x)
            gates = block.cdm.compute_gates(normed)  # (1, T, K)
            gates_by_layer[layer_idx] = gates[0].cpu()  # (T, K)

            # Run full block forward
            x, _ = block(x)

    return token_strs, gates_by_layer


def slot_entropy(gates_tk: torch.Tensor) -> float:
    """
    Marginal slot entropy: H(E_t[g_k(t)]).
    High = all K slots used roughly equally across positions (diverse).
    Low = one slot dominates across all positions (collapse).
    Max = log(K).
    """
    marginal = gates_tk.mean(dim=0)  # (K,) — time-averaged gate weight per slot
    marginal = marginal / (marginal.sum() + 1e-8)
    return -(marginal * torch.log(marginal + 1e-12)).sum().item()


def winner_concentration(gates_tk: torch.Tensor) -> dict:
    """
    Per-slot: how many positions does this slot win (argmax)?
    High count = slot is preferred at many positions.
    Zero count = slot is dead.
    """
    winners = gates_tk.argmax(dim=-1)  # (T,)
    K = gates_tk.shape[1]
    counts = {k: (winners == k).sum().item() for k in range(K)}
    return counts, winners.tolist()


def main():
    parser = argparse.ArgumentParser(description="CDM V2 routing probe — compares to V1 collapse")
    parser.add_argument("--checkpoint", default="/workspace/cdm_v2_full/best/model.pt",
                        help="Path to V2 checkpoint (model.pt)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="/workspace/cdm_v2_routing_results.json")
    parser.add_argument("--v1-comparison", action="store_true",
                        help="Print V1 routing results for comparison (hardcoded from paper)")
    args = parser.parse_args()

    print(f"[routing_v2] Loading from {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, args.device)
    tokenizer = get_tokenizer()

    K = cfg.K
    n_layers = cfg.n_layers
    max_entropy = math.log(K)

    print(f"[routing_v2] K={K}, n_layers={n_layers}, max_entropy={max_entropy:.3f}")
    print(f"[routing_v2] V1 baseline: K=8, max_entropy=2.079, L7 entropy≈0.69 (6/8 slots dead)")
    print()

    if args.v1_comparison:
        print("V1 BASELINE (from paper — K=8, 5 probe stories, last layer L7):")
        print("  Slot 0: 0 tokens   | DEAD")
        print("  Slot 1: 0 tokens   | DEAD")
        print("  Slot 2: 0 tokens   | DEAD")
        print("  Slot 3: 0 tokens   | DEAD")
        print("  Slot 4: 149 tokens | ACTIVE → syntax/function words")
        print("  Slot 5: 0 tokens   | DEAD")
        print("  Slot 6: 0 tokens   | DEAD")
        print("  Slot 7: 69 tokens  | ACTIVE → names/content words")
        print("  Routing entropy at L7: ~0.69 / 2.079 = 33% of max")
        print()

    # ── Run probe ────────────────────────────────────────────────────────────
    all_results = []
    # Aggregate gates across all stories for global analysis
    global_gates = {l: [] for l in range(n_layers)}   # layer -> list of (T, K) tensors

    for s_idx, story in enumerate(PROBE_STORIES):
        token_strs, gates_by_layer = probe_routing(model, tokenizer, story, args.device)
        T = len(token_strs)

        for l, gates_tk in gates_by_layer.items():
            global_gates[l].append(gates_tk)

        print(f"Story {s_idx+1} ({T} tokens): {story[:55]}...")
        print(f"  Layer  | Entropy  | % of max | Active slots (nonzero win)")
        print(f"  -------+----------+----------+---------------------------")
        for l in range(n_layers):
            gates_tk = gates_by_layer[l]
            ent = slot_entropy(gates_tk)
            counts, _ = winner_concentration(gates_tk)
            n_active = sum(1 for c in counts.values() if c > 0)
            pct_max = 100 * ent / max_entropy
            print(f"  L{l:<5d} | {ent:.4f}   | {pct_max:7.1f}% | {n_active}/{K} slots active")
        print()

        story_result = {
            "story": story[:80],
            "T": T,
            "tokens": token_strs,
            "layers": {}
        }
        for l, gates_tk in gates_by_layer.items():
            ent = slot_entropy(gates_tk)
            counts, winners = winner_concentration(gates_tk)
            story_result["layers"][str(l)] = {
                "entropy": ent,
                "entropy_pct_max": 100 * ent / max_entropy,
                "n_active_slots": sum(1 for c in counts.values() if c > 0),
                "slot_win_counts": {str(k): v for k, v in counts.items()},
                "winners_per_position": winners,
            }
        all_results.append(story_result)

    # ── Global aggregation (all stories combined) ─────────────────────────────
    print("=" * 70)
    print("GLOBAL ROUTING ANALYSIS (all 5 stories combined)")
    print("=" * 70)

    global_summary = {}
    for l in range(n_layers):
        all_gates = torch.cat(global_gates[l], dim=0)  # (sum_T, K)
        ent = slot_entropy(all_gates)
        counts, _ = winner_concentration(all_gates)
        n_active = sum(1 for c in counts.values() if c > 0)
        total_T = all_gates.shape[0]
        pct_max = 100 * ent / max_entropy

        print(f"L{l}: entropy={ent:.4f} ({pct_max:.1f}% of max) | {n_active}/{K} slots active")
        for k, c in sorted(counts.items(), key=lambda x: -x[1])[:5]:
            bar = "█" * max(1, int(20 * c / total_T))
            print(f"  Slot {k:2d}: {c:4d} tokens ({100*c/total_T:.1f}%)  {bar}")

        global_summary[str(l)] = {
            "entropy": ent,
            "entropy_pct_max": pct_max,
            "n_active_slots": n_active,
            "slot_win_counts": {str(k): v for k, v in counts.items()},
            "total_tokens": total_T,
        }

    # ── Last layer slot affinity ───────────────────────────────────────────
    last_layer = n_layers - 1
    all_gates_last = torch.cat(global_gates[last_layer], dim=0)
    winners_last = all_gates_last.argmax(dim=-1)

    print()
    print(f"LAST LAYER (L{last_layer}) SLOT TOKEN AFFINITY:")
    # To show slot content, we need token strings per story
    slot_token_lists = defaultdict(list)
    for story, (s_idx, story_text) in zip(all_results, enumerate(PROBE_STORIES)):
        token_strs = story["tokens"]
        wins = story["layers"][str(last_layer)]["winners_per_position"]
        for tok, slot in zip(token_strs, wins):
            slot_token_lists[slot].append(tok)

    for k in range(K):
        toks = slot_token_lists[k]
        total = len(toks)
        if total == 0:
            print(f"  Slot {k:2d}: DEAD (0 tokens)")
            continue
        freq = defaultdict(int)
        for t in toks:
            freq[t] += 1
        top = sorted(freq.items(), key=lambda x: -x[1])[:6]
        top_str = "  ".join(f"{t}({n})" for t, n in top)
        print(f"  Slot {k:2d} ({total:3d} tokens): {top_str}")

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        "model": {"K": K, "n_layers": n_layers, "d_model": cfg.d_model},
        "max_entropy": max_entropy,
        "v1_baseline": {
            "K": 8, "n_active_slots": 2, "entropy_L7": 0.69,
            "entropy_pct_max": 33.0, "note": "Slot 4 (syntax) + Slot 7 (semantics) only"
        },
        "global_summary": global_summary,
        "stories": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[routing_v2] Results saved → {args.output}")

    # ── Quick verdict ──────────────────────────────────────────────────────
    last_ent = global_summary[str(last_layer)]["entropy"]
    last_active = global_summary[str(last_layer)]["n_active_slots"]
    print()
    print("=" * 70)
    print("VERDICT vs V1 ROUTING COLLAPSE")
    print(f"  V1: {2}/{8} slots active, entropy 33% of max (collapsed)")
    print(f"  V2: {last_active}/{K} slots active, entropy {100*last_ent/max_entropy:.1f}% of max")
    if last_active >= K // 2:
        print("  STATUS: ✓ ROUTING COLLAPSE FIXED — diverse slot usage confirmed")
    else:
        print("  STATUS: ✗ ROUTING STILL COLLAPSED — V2 fix incomplete")
    print("=" * 70)


if __name__ == "__main__":
    main()
