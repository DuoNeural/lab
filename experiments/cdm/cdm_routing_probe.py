#!/usr/bin/env python3
"""
cdm_routing_probe.py — Slot Routing Pattern Analysis

Even if slots don't specialize in vocab space (Logit Lens null),
do they route different TYPES of tokens? This probes the gates
(which slot wins per token position) to find routing patterns.

Archon (DuoNeural) 2026-06-11
"""
import argparse, json, sys
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from cdm_model import CDMLanguageModel, CDMConfig

PROBE_STORIES = [
    "Once upon a time, there was a little girl named Lily. She lived near a big forest. One day, Lily found a small puppy near the trees. She took it home and fed it milk. Her mom said she could keep it. Lily named the puppy Spot.",
    "Tom was a curious boy who loved to explore. He lived in a house by the river. One afternoon, Tom saw a red boat floating on the water. He jumped in and paddled to the other bank. There he found a shiny golden coin.",
    "Emma had a magic wand. She waved it and flowers grew everywhere. The garden was full of roses and daisies. Her friend Ben came to visit and they played together all afternoon.",
    "A little bear named Max lived in a cozy cave. Every morning he walked to the meadow to eat berries. One day he met a fox who wanted to share his breakfast.",
    "Sophie loved to paint. She had brushes and colors all over her room. One rainy day she painted a big rainbow on her wall. Her cat Whiskers watched her and meowed.",
]

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = CDMConfig(**{k: v for k, v in cfg_dict.items() if hasattr(CDMConfig, k)})
    model = CDMLanguageModel(cfg)
    state_key = "model_state" if "model_state" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model.to(device), cfg

def get_tokenizer():
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained("gpt2")

def probe_routing(model, tokenizer, text, device):
    """For each token position, record which slot won (argmax gate) per layer."""
    tokens = tokenizer.encode(text)[:model.cfg.max_len]
    token_strs = [tokenizer.decode([t]).strip() for t in tokens]
    ids = torch.tensor([tokens], device=device)

    routing_by_layer = {}  # layer -> (T,) winner slot indices

    with torch.no_grad():
        x = model.embed(ids)
        for layer_idx, block in enumerate(model.blocks):
            normed = block.norm3(x)
            gates = block.cdm.compute_gates(normed)  # (1, T, K)
            winners = gates[0].argmax(dim=-1)        # (T,) — which slot won each position
            routing_by_layer[layer_idx] = winners.cpu().tolist()

            # Continue forward (need slots for attention)
            slots_all = block.cdm(normed)
            slots_final = slots_all[0, -1, :, :]
            x = x + block.dropout(block.attn(block.norm1(x), slots_final.unsqueeze(0)))
            x = x + block.ffn(block.norm2(x))

    return token_strs, routing_by_layer

def analyze_routing(token_strs, routing_by_layer, K, last_layer):
    """Analyze which tokens route to which slots in the last layer."""
    winners = routing_by_layer[last_layer]
    slot_tokens = defaultdict(list)
    for tok, slot in zip(token_strs, winners):
        slot_tokens[slot].append(tok)
    return slot_tokens

def compute_slot_entropy(routing_by_layer, K, T):
    """Per-layer slot entropy — low = one slot dominates, high = all slots used equally."""
    from math import log
    entropies = {}
    for layer, winners in routing_by_layer.items():
        counts = [0] * K
        for w in winners:
            counts[w] += 1
        probs = [c / T for c in counts]
        ent = -sum(p * log(p + 1e-12) for p in probs)
        entropies[layer] = ent
    return entropies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/workspace/cdm_full/best/model.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="/workspace/cdm_routing_results.json")
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint, args.device)
    tokenizer = get_tokenizer()
    K = cfg.K
    last_layer = cfg.n_layers - 1

    print(f"[routing] Model: {cfg.n_layers}L d={cfg.d_model} K={K}")
    print(f"\n{'='*70}")
    print(f"SLOT ROUTING ANALYSIS — Which tokens route to which slot?")
    print(f"{'='*70}")

    all_results = []
    # Aggregate routing across all stories for last layer
    global_slot_tokens = defaultdict(list)

    for story_idx, story in enumerate(PROBE_STORIES):
        token_strs, routing_by_layer = probe_routing(model, tokenizer, story, args.device)
        T = len(token_strs)

        # Per-layer entropy
        entropies = compute_slot_entropy(routing_by_layer, K, T)

        # Last-layer routing
        slot_tokens = analyze_routing(token_strs, routing_by_layer, K, last_layer)
        for k, toks in slot_tokens.items():
            global_slot_tokens[k].extend(toks)

        print(f"\nStory {story_idx+1}: {story[:55]}...")
        print(f"  Layer entropies (low=concentrated, max={K:.2f} ln={__import__('math').log(K):.2f}):")
        for layer, ent in sorted(entropies.items()):
            bar = "█" * int(ent * 5)
            print(f"    L{layer}: {ent:.3f} {bar}")

        print(f"  Last-layer (L{last_layer}) slot → tokens:")
        for k in range(K):
            toks = slot_tokens.get(k, [])
            count = len(toks)
            sample = toks[:6]
            print(f"    Slot {k} ({count:3d} tokens): {sample}")

        all_results.append({
            "story": story[:80],
            "tokens": token_strs,
            "routing": {str(l): r for l, r in routing_by_layer.items()},
            "entropies": {str(l): e for l, e in entropies.items()},
            "last_layer_slot_tokens": {str(k): v for k, v in slot_tokens.items()},
        })

    # Global aggregation
    print(f"\n{'='*70}")
    print(f"GLOBAL SLOT AFFINITY (all stories combined, last layer)")
    print(f"{'='*70}")
    for k in range(K):
        toks = global_slot_tokens[k]
        total = len(toks)
        # Count token frequencies
        freq = defaultdict(int)
        for t in toks:
            freq[t] += 1
        top = sorted(freq.items(), key=lambda x: -x[1])[:8]
        top_str = "  ".join(f"{t}({n})" for t, n in top)
        print(f"Slot {k} ({total} tokens total): {top_str}")

    with open(args.output, "w") as f:
        json.dump({"config": {"K": K, "n_layers": cfg.n_layers}, "results": all_results,
                   "global_slot_affinity": {str(k): v for k, v in global_slot_tokens.items()}}, f)

    print(f"\n[routing] Saved to {args.output}")

if __name__ == "__main__":
    main()
