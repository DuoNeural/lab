#!/usr/bin/env python3
"""
cdm_story_slot_probe.py — Story Prompt Slot Specialization Analysis
DuoNeural / Archon — 2026-06-12

Jesse observed unexpected slot behavior on Spaces with story prompts.
This probe uses a large diverse set of story inputs — both open-ended
fragments ("He ran") and complete sentences ("He ran to see the dog.")
— to map slot specialization by input type, token type, and content.

Output: JSON + printed table showing which slots win for what kinds of inputs.
"""
import sys, json, math
from pathlib import Path
from collections import defaultdict
import torch
sys.path.insert(0, "/workspace")
from cdm_model_v2 import CDMLanguageModelV2, CDMConfigV2

# ── Checkpoint (HF cached) ────────────────────────────────────────────────────
CKPT = (
    "/workspace/.hf_home/hub/models--DuoNeural--CDM-V2-TinyStories-37M"
    "/snapshots/91b8d87b26c8798cfcb7d0f6ff8fa374d37d1cfe/model.pt"
)
OUT = "/workspace/cdm_story_slot_results.json"

# ── The big prompt set ────────────────────────────────────────────────────────
# Each entry: (text, category, prompt_type)
# category: character / action / emotion / nature / object / question / command
# prompt_type: "fragment" | "sentence" | "paragraph"

PROMPTS = [
    # ── FRAGMENTS — bare subject+verb or short phrases ──────────────────────
    ("He ran",                          "action",    "fragment"),
    ("She laughed",                     "emotion",   "fragment"),
    ("The dog barked",                  "action",    "fragment"),
    ("A small bird",                    "nature",    "fragment"),
    ("The old man walked",              "action",    "fragment"),
    ("She smiled at",                   "emotion",   "fragment"),
    ("The children played",             "action",    "fragment"),
    ("It was dark",                     "setting",   "fragment"),
    ("The door opened",                 "action",    "fragment"),
    ("Once upon a time",                "narrative", "fragment"),
    ("He was afraid",                   "emotion",   "fragment"),
    ("The little cat",                  "character", "fragment"),
    ("She found a",                     "action",    "fragment"),
    ("They looked up",                  "action",    "fragment"),
    ("Deep in the forest",              "setting",   "fragment"),
    ("His name was",                    "character", "fragment"),
    ("The sun rose",                    "nature",    "fragment"),
    ("She picked up the",               "action",    "fragment"),
    ("A long time ago",                 "narrative", "fragment"),
    ("The water was cold",              "setting",   "fragment"),

    # ── COMPLETE SENTENCES ───────────────────────────────────────────────────
    ("He ran to see the dog.",                          "action",    "sentence"),
    ("She laughed when she saw the clown.",             "emotion",   "sentence"),
    ("The dog barked loudly at the mailman.",           "action",    "sentence"),
    ("A small bird sat on the windowsill.",             "nature",    "sentence"),
    ("The old man walked slowly down the road.",        "action",    "sentence"),
    ("She smiled at her friend across the room.",       "emotion",   "sentence"),
    ("The children played in the park all afternoon.",  "action",    "sentence"),
    ("It was dark when they arrived home.",             "setting",   "sentence"),
    ("The door opened and a tall man walked in.",       "action",    "sentence"),
    ("Once upon a time there was a brave little girl.", "narrative", "sentence"),
    ("He was afraid of the thunder outside.",           "emotion",   "sentence"),
    ("The little cat curled up by the fire.",           "character", "sentence"),
    ("She found a shiny coin under the bench.",         "action",    "sentence"),
    ("They looked up and saw a rainbow.",               "nature",    "sentence"),
    ("Deep in the forest lived a family of rabbits.",   "setting",   "sentence"),
    ("His name was Tom and he loved adventure.",        "character", "sentence"),
    ("The sun rose over the mountains slowly.",         "nature",    "sentence"),
    ("She picked up the red ball and threw it far.",    "action",    "sentence"),
    ("A long time ago, dragons ruled the land.",        "narrative", "sentence"),
    ("The water was cold but the fish swam fast.",      "setting",   "sentence"),

    # ── MULTI-SENTENCE PARAGRAPH OPENERS ────────────────────────────────────
    ("Lily was a kind girl who loved animals. She had a rabbit named Bun.",
     "character", "paragraph"),
    ("The forest was quiet. No birds sang. No leaves moved. Then a sound came.",
     "setting", "paragraph"),
    ("Max wanted to fly. He built wings out of cardboard and jumped from the steps. He fell.",
     "action", "paragraph"),
    ("Every morning Tom woke up and looked out his window. He always saw the same thing: fog.",
     "setting", "paragraph"),
    ("She was the best baker in the village. Everyone loved her bread. Even the mayor came.",
     "character", "paragraph"),
]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = CDMConfigV2(**{k: v for k, v in cfg_dict.items()
                         if k in CDMConfigV2.__dataclass_fields__})
    model = CDMLanguageModelV2(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(device), cfg


def get_tokenizer():
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained("gpt2")


def probe_routing(model, tokenizer, text, device):
    tokens = tokenizer.encode(text)[:model.cfg.max_len]
    token_strs = [tokenizer.decode([t]) for t in tokens]
    ids = torch.tensor([tokens], device=device)
    gates_by_layer = {}
    with torch.no_grad():
        x = model.embed(ids)
        for li, block in enumerate(model.blocks):
            normed = block.norm_cdm(x)
            gates = block.cdm.compute_gates(normed)   # (1, T, K)
            gates_by_layer[li] = gates[0].cpu()       # (T, K)
            x, _ = block(x)
    return token_strs, gates_by_layer


def slot_entropy(gates_tk):
    m = gates_tk.mean(dim=0)
    m = m / (m.sum() + 1e-8)
    return -(m * torch.log(m + 1e-12)).sum().item()


def dominant_slot(gates_tk):
    """Returns (slot_idx, pct_of_tokens_it_wins)."""
    winners = gates_tk.argmax(dim=-1)   # (T,)
    T = winners.shape[0]
    counts = torch.bincount(winners, minlength=gates_tk.shape[1])
    dom = counts.argmax().item()
    return dom, 100.0 * counts[dom].item() / T


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[story_probe] Loading model from HF cache...")
    model, cfg = load_model(CKPT, device)
    tokenizer = get_tokenizer()

    K = cfg.K
    n_layers = cfg.n_layers
    max_ent = math.log(K)
    last_L = n_layers - 1

    print(f"[story_probe] K={K}, n_layers={n_layers}, max_ent={max_ent:.3f}")
    print(f"[story_probe] {len(PROMPTS)} prompts: "
          f"{sum(1 for _,_,t in PROMPTS if t=='fragment')} fragments, "
          f"{sum(1 for _,_,t in PROMPTS if t=='sentence')} sentences, "
          f"{sum(1 for _,_,t in PROMPTS if t=='paragraph')} paragraphs")
    print()

    results = []
    # For aggregate analysis by prompt type and category
    agg_gates = defaultdict(lambda: defaultdict(list))   # [ptype][layer] -> list of (T,K)
    agg_cat   = defaultdict(lambda: defaultdict(list))   # [cat][layer]   -> list of (T,K)

    # ── Per-prompt pass ───────────────────────────────────────────────────────
    print(f"{'Prompt':<48} {'Type':<9} {'Cat':<10} | {'Dom.Slot':>8} {'%Win':>6} {'Entropy%':>9} {'Nactive':>7}")
    print("-" * 100)

    for text, cat, ptype in PROMPTS:
        token_strs, gates_by_layer = probe_routing(model, tokenizer, text, device)
        T = len(token_strs)

        gates_last = gates_by_layer[last_L]   # (T, K)
        dom, dom_pct = dominant_slot(gates_last)
        ent = slot_entropy(gates_last)
        ent_pct = 100.0 * ent / max_ent
        winners = gates_last.argmax(dim=-1).tolist()
        n_active = len(set(winners))

        label = text[:46] + ".." if len(text) > 47 else text
        print(f"{label:<48} {ptype:<9} {cat:<10} | {dom:>8} {dom_pct:>6.1f}% {ent_pct:>8.1f}% {n_active:>7}/{K}")

        # Token-level slot affinity at last layer
        slot_token_map = defaultdict(list)
        for tok, w in zip(token_strs, winners):
            slot_token_map[w].append(tok.strip())

        entry = {
            "text": text,
            "category": cat,
            "prompt_type": ptype,
            "n_tokens": T,
            "tokens": token_strs,
            "last_layer": {
                "dominant_slot": dom,
                "dominant_pct": dom_pct,
                "entropy": ent,
                "entropy_pct_max": ent_pct,
                "n_active_slots": n_active,
                "winners_per_token": winners,
                "slot_token_affinity": {str(k): v for k, v in slot_token_map.items()},
            },
            "all_layers": {},
        }

        for li, gates_tk in gates_by_layer.items():
            d, dp = dominant_slot(gates_tk)
            e = slot_entropy(gates_tk)
            entry["all_layers"][str(li)] = {
                "dominant_slot": d,
                "dominant_pct": dp,
                "entropy_pct_max": 100.0 * e / max_ent,
                "n_active": len(set(gates_tk.argmax(dim=-1).tolist())),
            }
            agg_gates[ptype][li].append(gates_tk)
            agg_cat[cat][li].append(gates_tk)

        results.append(entry)

    # ── Aggregate: fragment vs sentence vs paragraph (last layer) ─────────────
    print()
    print("=" * 100)
    print("AGGREGATE LAST-LAYER SLOT DISTRIBUTION BY PROMPT TYPE")
    print("=" * 100)
    for ptype in ["fragment", "sentence", "paragraph"]:
        if ptype not in agg_gates:
            continue
        all_g = torch.cat(agg_gates[ptype][last_L], dim=0)   # (total_T, K)
        counts = torch.bincount(all_g.argmax(dim=-1), minlength=K)
        total = counts.sum().item()
        ent = slot_entropy(all_g)
        print(f"\n  {ptype.upper()} ({total} tokens, entropy={ent:.3f}/{max_ent:.3f} = {100*ent/max_ent:.1f}%):")
        top_slots = sorted(range(K), key=lambda k: -counts[k].item())[:8]
        for k in top_slots:
            c = counts[k].item()
            if c == 0:
                continue
            bar = "█" * max(1, int(30 * c / total))
            print(f"    Slot {k:2d}: {c:4d} tokens ({100*c/total:5.1f}%)  {bar}")

    # ── Aggregate: by content category (last layer) ────────────────────────────
    print()
    print("=" * 100)
    print("AGGREGATE LAST-LAYER SLOT DISTRIBUTION BY CONTENT CATEGORY")
    print("=" * 100)
    for cat in sorted(agg_cat.keys()):
        all_g = torch.cat(agg_cat[cat][last_L], dim=0)
        counts = torch.bincount(all_g.argmax(dim=-1), minlength=K)
        total = counts.sum().item()
        top2 = sorted(range(K), key=lambda k: -counts[k].item())[:3]
        top_str = "  ".join(f"Slot{k}({100*counts[k].item()/total:.0f}%)" for k in top2 if counts[k] > 0)
        print(f"  {cat:<12}: top slots → {top_str}")

    # ── Slot token affinity (what kinds of tokens does each slot prefer?) ──────
    print()
    print("=" * 100)
    print(f"LAST LAYER (L{last_L}) GLOBAL TOKEN AFFINITY — what does each slot eat?")
    print("=" * 100)
    slot_all_toks = defaultdict(list)
    for r in results:
        for slot_str, toks in r["last_layer"]["slot_token_affinity"].items():
            slot_all_toks[int(slot_str)].extend(toks)

    for k in range(K):
        toks = slot_all_toks[k]
        if not toks:
            print(f"  Slot {k:2d}: DEAD")
            continue
        freq = defaultdict(int)
        for t in toks:
            freq[t] += 1
        top = sorted(freq.items(), key=lambda x: -x[1])[:8]
        top_str = "  ".join(f"'{t}'({n})" for t, n in top)
        print(f"  Slot {k:2d} ({len(toks):3d} tok): {top_str}")

    # ── Fragment vs Sentence dominant slot comparison ──────────────────────────
    print()
    print("=" * 100)
    print("FRAGMENT vs SENTENCE — per-prompt dominant slot comparison")
    print("=" * 100)
    frag_doms  = [(t, dom, dp) for t, _, _, dom, dp in
                  [(r["text"], r["category"], r["prompt_type"],
                    r["last_layer"]["dominant_slot"], r["last_layer"]["dominant_pct"])
                   for r in results if r["prompt_type"] == "fragment"]]
    sent_doms  = [(t, dom, dp) for t, _, _, dom, dp in
                  [(r["text"], r["category"], r["prompt_type"],
                    r["last_layer"]["dominant_slot"], r["last_layer"]["dominant_pct"])
                   for r in results if r["prompt_type"] == "sentence"]]

    print(f"  {'FRAGMENT':<42} Slot%   |   {'SENTENCE':<42} Slot%")
    print(f"  {'-'*42} ------   |   {'-'*42} ------")
    for (ft, fd, fp), (st, sd, sp) in zip(frag_doms, sent_doms):
        fl = ft[:40] + ".." if len(ft) > 41 else ft
        sl = st[:40] + ".." if len(st) > 41 else st
        flag = " ← DIFF" if fd != sd else ""
        print(f"  {fl:<42} S{fd}({fp:.0f}%)  |   {sl:<42} S{sd}({sp:.0f}%){flag}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "model_config": {"K": K, "n_layers": n_layers, "d_model": cfg.d_model},
        "max_entropy": max_ent,
        "n_prompts": len(PROMPTS),
        "results": results,
    }
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[story_probe] Saved → {OUT}")


if __name__ == "__main__":
    main()
