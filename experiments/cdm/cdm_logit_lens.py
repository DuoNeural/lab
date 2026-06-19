#!/usr/bin/env python3
"""
cdm_logit_lens.py — Slot Specialization Probe for CDM Language Model

Projects each slot's final state vector through the unembedding matrix to reveal
what concept each slot has learned to represent.

If competitive routing caused emergent specialization, we expect to see:
  Slot 0: character names (Lily, Tom, Emma, Jack...)
  Slot 1: locations (forest, park, lake, garden...)
  Slot 2: actions/verbs (ran, played, found, gave...)
  ...etc

Usage:
    python3 cdm_logit_lens.py --checkpoint /workspace/cdm_full/best/model.pt
    python3 cdm_logit_lens.py --checkpoint /workspace/cdm_full/best/model.pt --topk 10

Authors: Archon (DuoNeural) 2026-06-11
"""

import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from cdm_model import CDMLanguageModel, CDMConfig

# ── Stories to probe ────────────────────────────────────────────────────────
PROBE_STORIES = [
    "Once upon a time, there was a little girl named Lily. She lived near a big forest. One day, Lily found a small puppy near the trees. She took it home and fed it milk. Her mom said she could keep it. Lily named the puppy Spot.",
    "Tom was a curious boy who loved to explore. He lived in a house by the river. One afternoon, Tom saw a red boat floating on the water. He jumped in and paddled to the other bank. There he found a shiny golden coin.",
    "Emma had a magic wand. She waved it and flowers grew everywhere. The garden was full of roses and daisies. Her friend Ben came to visit and they played together all afternoon. When the sun set, they sat by the pond and watched the stars.",
    "A little bear named Max lived in a cozy cave. Every morning he walked to the meadow to eat berries. One day he met a fox who wanted to share his breakfast. They became best friends and played in the forest until dark.",
    "Sophie loved to paint. She had brushes and colors all over her room. One rainy day she painted a big rainbow on her wall. Her cat Whiskers watched her and meowed. Mom came in and smiled at the beautiful picture.",
]

def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = CDMConfig(**{k: v for k, v in cfg_dict.items() if hasattr(CDMConfig, k)})
    model = CDMLanguageModel(cfg)
    state_key = "model_state" if "model_state" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    model.to(device)
    return model, cfg

def get_tokenizer():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    return tok

def probe_slots(model, tokenizer, text: str, device: str, top_k: int = 10):
    """
    Run a story through the model and extract slot states from every layer.
    Returns per-layer, per-slot top-k tokens via Logit Lens projection.
    """
    tokens = tokenizer.encode(text)[:model.cfg.max_len]
    ids = torch.tensor([tokens], device=device)

    slot_results = {}  # layer -> {slot_k: [(token, prob), ...]}

    with torch.no_grad():
        x = model.embed(ids)  # (1, T, d)

        for layer_idx, block in enumerate(model.blocks):
            normed = block.norm3(x)                        # (1, T, d)
            slots_all = block.cdm(normed)                 # (1, T, K, d)
            slots_final = slots_all[0, -1, :, :]          # (K, d) — end-of-seq slot state

            # Logit Lens: project each slot through the unembedding matrix
            slot_logits = slots_final @ model.head.weight.T  # (K, vocab)
            slot_probs = F.softmax(slot_logits, dim=-1)       # (K, vocab)

            layer_slots = {}
            for k in range(model.cfg.K):
                top_probs, top_ids = slot_probs[k].topk(top_k)
                top_tokens = [tokenizer.decode([tid.item()]).strip() for tid in top_ids]
                layer_slots[k] = list(zip(top_tokens, top_probs.tolist()))

            slot_results[layer_idx] = layer_slots

            # Continue forward pass for next layer
            slots_for_attn = slots_all[0, -1, :, :]  # (K, d)
            x = x + block.dropout(block.attn(block.norm1(x), slots_for_attn.unsqueeze(0)))
            x = x + block.ffn(block.norm2(x))

    return slot_results

def compute_slot_entropy(slot_probs_dict):
    """Entropy of each slot's logit distribution — low = specialized, high = diffuse."""
    from math import log
    entropies = {}
    for k, token_prob_list in slot_probs_dict.items():
        # approximate entropy from top-k (underestimate, but useful for comparison)
        probs = [p for _, p in token_prob_list]
        ent = -sum(p * log(p + 1e-12) for p in probs)
        entropies[k] = ent
    return entropies

def print_slot_table(layer_results, top_k=5, story_idx=0):
    print(f"\n{'='*80}")
    print(f"STORY {story_idx+1} — SLOT SPECIALIZATION BY LAYER")
    print(f"{'='*80}")

    n_layers = len(layer_results)
    # Show first, middle, and last layers
    show_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]))

    for layer_idx in show_layers:
        slots = layer_results[layer_idx]
        K = len(slots)
        print(f"\n  Layer {layer_idx:2d}:")
        print(f"  {'─'*70}")
        for k in range(K):
            top_tokens = [t for t, _ in slots[k][:top_k]]
            top_probs = [f"{p:.3f}" for _, p in slots[k][:top_k]]
            token_str = "  |  ".join(f"{t:<10}" for t in top_tokens)
            print(f"  Slot {k}: {token_str}")
        print()

def aggregate_across_stories(all_results, n_layers, K):
    """
    For each layer, for each slot, find the tokens that appear most consistently
    across stories — these are the 'stable' slot representations.
    """
    from collections import defaultdict

    # Aggregate top-1 tokens per slot per layer across stories
    layer_slot_tokens = defaultdict(lambda: defaultdict(list))
    for story_results in all_results:
        for layer_idx, slots in story_results.items():
            for k, token_probs in slots.items():
                top_token = token_probs[0][0] if token_probs else ""
                layer_slot_tokens[layer_idx][k].append(top_token)

    return layer_slot_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/workspace/cdm_full/best/model.pt")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="/workspace/cdm_logit_lens_results.json")
    args = parser.parse_args()

    print(f"[logit_lens] Loading model from {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, args.device)
    print(f"[logit_lens] Model: {cfg.n_layers}L, d={cfg.d_model}, K={cfg.K}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    print(f"[logit_lens] Loading tokenizer...")
    tokenizer = get_tokenizer()

    all_results = []
    print(f"\n[logit_lens] Probing {len(PROBE_STORIES)} stories...\n")

    for i, story in enumerate(PROBE_STORIES):
        print(f"  Story {i+1}: {story[:60]}...")
        layer_results = probe_slots(model, tokenizer, story, args.device, args.topk)
        all_results.append(layer_results)

        # Print last-layer slots (most informative)
        last_layer = cfg.n_layers - 1
        slots = layer_results[last_layer]
        print(f"  Layer {last_layer} (final) slots:")
        for k in range(cfg.K):
            top5 = [t for t, _ in slots[k][:5]]
            print(f"    Slot {k}: {top5}")
        print()

    # Full table for story 1 (most detailed output)
    print_slot_table(all_results[0], top_k=args.topk, story_idx=0)

    # Cross-story aggregation: which tokens are stable per slot in last layer?
    print(f"\n{'='*80}")
    print(f"CROSS-STORY AGGREGATION — Last Layer (Layer {cfg.n_layers-1})")
    print(f"(Top token per slot per story — consistency reveals specialization)")
    print(f"{'='*80}")

    last_layer = cfg.n_layers - 1
    print(f"\n  {'Slot':<8}", end="")
    for i in range(len(PROBE_STORIES)):
        print(f"  {'Story '+str(i+1):<14}", end="")
    print()
    print(f"  {'─'*80}")

    for k in range(cfg.K):
        print(f"  Slot {k:<3}", end="")
        for story_results in all_results:
            top_token = story_results[last_layer][k][0][0] if story_results[last_layer][k] else "?"
            print(f"  {top_token:<14}", end="")
        print()

    # Save results to JSON
    serializable = {}
    for story_idx, story_results in enumerate(all_results):
        serializable[f"story_{story_idx}"] = {
            str(layer): {
                str(k): [(t, float(p)) for t, p in probs]
                for k, probs in slots.items()
            }
            for layer, slots in story_results.items()
        }

    with open(args.output, "w") as f:
        json.dump({
            "config": {"n_layers": cfg.n_layers, "K": cfg.K, "d_model": cfg.d_model},
            "probe_stories": PROBE_STORIES,
            "results": serializable,
        }, f, indent=2)

    print(f"\n[logit_lens] Results saved to {args.output}")
    print(f"[logit_lens] Done.")

if __name__ == "__main__":
    main()
