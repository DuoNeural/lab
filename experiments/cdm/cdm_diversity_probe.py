#!/usr/bin/env python3
"""
CDM V2 Diversity Probe — Are slots domain-general or domain-specific?

Runs routing gate analysis on the CDM V2 checkpoint across 5 diverse domains:
  tinystories  — narrative prose (training distribution)
  code_python  — Python source (syntax-rich, structured indentation)
  news_factual — factual/encyclopedic prose (Wikipedia-style)
  poetry_verse — rhythmic verse (strong phonetic/structural patterns)
  lists_numeric — numbered lists with digits (tests EMA-blurring hypothesis)

For each domain:
  - Per-slot win counts (routing histogram)
  - MI(slot; token_category) in bits
  - Routing entropy %
  - Slot fingerprint (K-dim normalized usage vector)
  - Per-slot top-5 claimed tokens

Cross-domain:
  - Cosine similarity between fingerprints → domain-general vs domain-specific verdict
  - Slot stability (mean/std of per-slot share across domains)

Outputs: /workspace/cdm_diversity_probe_results.json

Usage: python3 cdm_diversity_probe.py
Architecture: Archon (DuoNeural) 2026-06-12
"""

import json
import math
import sys
import os
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Domain corpora — ~200-400 tokens each for good statistics
# ---------------------------------------------------------------------------

DOMAINS = {
    "tinystories": (
        "Once upon a time there was a little girl named Lily. She loved to play in the garden "
        "with her dog Spot. One sunny morning, Lily found a small bird with a hurt wing near "
        "the fence. She carefully picked it up and brought it inside to show her mother. "
        "Her mother smiled and helped her make a cozy nest from an old shoebox and some soft "
        "cotton. Lily gave the bird water and tiny seeds every single day. She sang to it and "
        "told it stories at bedtime. After one whole week, the little bird was hopping around "
        "and looking much better. On Saturday morning, Lily took it outside to the garden. "
        "She opened her hands and the bird spread its wings and flew up into the bright blue sky. "
        "Lily felt happy and a little sad at the same time. Spot wagged his tail and licked "
        "her hand gently. That night, Lily told her dad about the bird. He hugged her and said "
        "she had done a very kind thing. Lily smiled and fell asleep dreaming of birds and gardens."
    ),
    "code_python": (
        "import os\nimport sys\nfrom typing import List, Optional\n\n"
        "def calculate_fibonacci(n: int) -> List[int]:\n"
        "    if n <= 0:\n        return []\n"
        "    elif n == 1:\n        return [0]\n"
        "    sequence = [0, 1]\n"
        "    for i in range(2, n):\n"
        "        next_val = sequence[i-1] + sequence[i-2]\n"
        "        sequence.append(next_val)\n"
        "    return sequence\n\n"
        "class DataProcessor:\n"
        "    def __init__(self, data: List[int], threshold: float = 0.5):\n"
        "        self.data = data\n"
        "        self.threshold = threshold\n"
        "        self.results: List[int] = []\n\n"
        "    def process(self) -> List[int]:\n"
        "        for item in self.data:\n"
        "            if isinstance(item, int) and item > 0:\n"
        "                self.results.append(item * 2)\n"
        "        return self.results\n\n"
        "    def filter_above(self, cutoff: int) -> List[int]:\n"
        "        return [x for x in self.results if x > cutoff]\n\n"
        "def main() -> None:\n"
        "    processor = DataProcessor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])\n"
        "    output = processor.process()\n"
        "    filtered = processor.filter_above(8)\n"
        "    print(f'Results: {output}')\n"
        "    print(f'Filtered: {filtered}')\n"
        "    fib = calculate_fibonacci(15)\n"
        "    print(f'Fibonacci: {fib}')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    ),
    "news_factual": (
        "The International Space Station orbits Earth at an altitude of approximately 408 kilometers "
        "above sea level. It travels at a speed of 27,600 kilometers per hour, completing "
        "approximately 15.5 orbits of Earth per day. The station serves as a microgravity and "
        "space environment research laboratory in which scientific research is conducted in "
        "astrobiology, astronomy, meteorology, and physics. Scientists from 19 different nations "
        "have conducted experiments aboard the station since its first crewed mission in "
        "November 2000. The station measures approximately 109 meters wide and 73 meters long, "
        "with a habitable volume of 388 cubic meters. Its solar panel arrays generate between "
        "84 and 120 kilowatts of power depending on orientation relative to the sun. "
        "Astronauts typically spend six months aboard the station before returning to Earth "
        "via Soyuz or Crew Dragon capsules. The station is jointly operated by five space agencies: "
        "NASA, Roscosmos, ESA, JAXA, and CSA. As of 2026, more than 270 individuals from "
        "20 countries have visited the station over the course of its operational history. "
        "The station is scheduled for deorbiting no earlier than 2030, with a planned "
        "controlled reentry over the Pacific Ocean."
    ),
    "poetry_verse": (
        "The road not taken leads to silence deep,\n"
        "Where shadows fall and ancient memories sleep.\n"
        "Through tangled paths of moonlit silver streams,\n"
        "The wanderer walks between the waking dreams.\n\n"
        "No footstep echoes where the ravens call,\n"
        "Where twisted oaks and mossy boulders fall.\n"
        "The hollow wind sings low its mournful song,\n"
        "While stars keep watch the endless night hours long.\n\n"
        "What secret words the cold dark waters keep,\n"
        "What buried hopes lie frozen in their sleep?\n"
        "The traveler pauses, turns, and looks behind,\n"
        "At all the roads she left to chance and time.\n\n"
        "Yet onward still the silver pathway goes,\n"
        "Through frost and fire, through summer sun and snows.\n"
        "The heart remembers what the mind forgets,\n"
        "The soul still carries all its ancient debts.\n"
    ),
    "lists_numeric": (
        "Top programming languages by developer usage in 2026:\n"
        "1. Python - 31.2% of developers use it as primary language\n"
        "2. JavaScript - 28.4% usage across web development\n"
        "3. TypeScript - 14.7% growing rapidly in enterprise\n"
        "4. Java - 12.1% dominant in enterprise backend systems\n"
        "5. C# - 9.8% strong in game development and enterprise\n"
        "6. C++ - 8.3% critical for systems and embedded work\n"
        "7. Go - 7.2% favored for cloud infrastructure\n"
        "8. Rust - 6.1% fastest growing systems language\n"
        "9. Kotlin - 5.4% replacing Java on Android\n"
        "10. Swift - 4.9% dominant on Apple platforms\n\n"
        "Key statistics from the survey:\n"
        "- Total developers surveyed: 87,345 respondents\n"
        "- Average years of experience: 12.4 years\n"
        "- Remote workers: 58% fully remote, 29% hybrid\n"
        "- Median annual salary: $127,500 USD\n"
        "- Most wanted skill: machine learning at 43.2%\n"
        "- Most dreaded task: legacy code maintenance at 67.8%\n"
        "- Preferred IDE: VS Code at 74.1% of all respondents\n"
    ),
}

# Known slot roles from TinyStories probe (0-indexed)
SLOT_ROLES = {0: "IDENTITY", 5: "ARTICLES", 9: "AGENCY", 11: "PUNCT", 15: "VERBS"}

# Token category heuristics
PUNCT_CHARS  = set('.,;:!?-()\'"[]{}\\/')
PY_KEYWORDS  = {
    'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return',
    'import', 'from', 'in', 'not', 'and', 'or', 'True', 'False', 'None',
    'try', 'except', 'with', 'as', 'pass', 'break', 'continue', 'yield',
    'lambda', 'global', 'nonlocal', 'del', 'raise', 'assert', 'is', 'print',
    'range', 'len', 'isinstance', 'append', 'self', 'None', 'List', 'Optional',
}
COMMON_WORDS = {
    'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'to', 'of', 'in', 'and', 'or', 'but',
    'for', 'with', 'at', 'by', 'from', 'on', 'that', 'this', 'it', 'its',
    'not', 'as', 'up', 'about', 'into', 'than', 'so', 'if', 'out', 'all',
}


def categorize_token(tok_str: str) -> str:
    t = tok_str.strip()
    if not t:
        return "SPACE"
    if all(c in PUNCT_CHARS for c in t) and len(t) <= 3:
        return "PUNCT"
    if (all(c.isdigit() or c in '.-,%' for c in t) and any(c.isdigit() for c in t)):
        return "DIGIT"
    tl = t.lower()
    if tl in PY_KEYWORDS:
        return "KEYWORD"
    if tl in COMMON_WORDS:
        return "COMMON"
    if t[0].isupper() and len(t) > 1:
        return "CAPS"
    if '\n' in tok_str or tok_str.startswith('    ') or tok_str == '\t':
        return "INDENT"
    return "OTHER"


def compute_mi(slot_counts: dict, cat_counts: dict, joint_counts: dict, total: int) -> float:
    """MI(slot; category) in bits."""
    mi = 0.0
    for (slot, cat), count in joint_counts.items():
        if count == 0:
            continue
        p_joint = count / total
        p_slot  = slot_counts.get(slot, 0) / total
        p_cat   = cat_counts.get(cat, 0) / total
        if p_slot > 0 and p_cat > 0:
            mi += p_joint * math.log2(p_joint / (p_slot * p_cat))
    return mi


def cosine_sim(a, b):
    dot   = sum(x * y for x, y in zip(a, b))
    na    = math.sqrt(sum(x * x for x in a))
    nb    = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-12)


def probe_domain(model, tokenizer, domain_name: str, text: str, device: str, K: int):
    """Run forward pass, capture CDM routing gates, compute all stats."""
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    T = tokens.shape[1]
    token_strs = [tokenizer.decode([t]) for t in tokens[0].tolist()]
    print(f"  {domain_name}: {T} tokens", flush=True)

    # Hook into each CDMBlockV2.cdm to capture gates per layer
    # block.cdm returns (slots_all, gates); out[1] = gates: (B, T, K)
    layer_gates_list = []   # will hold n_layers tensors of shape (T, K)

    def make_hook():
        def hook(module, inp, out):
            layer_gates_list.append(out[1][0].detach().cpu())  # (T, K)
        return hook

    hooks = []
    for block in model.blocks:
        h = block.cdm.register_forward_hook(make_hook())
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        _ = model(tokens)

    for h in hooks:
        h.remove()

    # Stack: (T, n_layers, K)
    gates_tensor = torch.stack(layer_gates_list, dim=1)
    n_layers     = gates_tensor.shape[1]

    # ---- Per-layer entropy ------------------------------------------------
    layer_entropy_pcts = []
    for layer_idx in range(n_layers):
        lg = gates_tensor[:, layer_idx, :]          # (T, K)
        mean_g = lg.mean(dim=0)                      # (K,)
        ent = 0.0
        for g in mean_g:
            g = g.item()
            if g > 1e-9:
                ent -= g * math.log(g)
        layer_entropy_pcts.append(round(ent / math.log(K) * 100, 2))

    # ---- Final-layer analysis (most specialized) --------------------------
    final_gates = gates_tensor[:, -1, :]   # (T, K)
    winners     = final_gates.argmax(dim=-1).tolist()

    slot_counts  = defaultdict(int)
    cat_counts   = defaultdict(int)
    joint_counts = defaultdict(int)

    for t_idx in range(T):
        w   = winners[t_idx]
        cat = categorize_token(token_strs[t_idx])
        slot_counts[w]       += 1
        cat_counts[cat]      += 1
        joint_counts[(w, cat)] += 1

    mi = compute_mi(dict(slot_counts), dict(cat_counts),
                    {k: v for k, v in joint_counts.items()}, T)

    # ---- Per-slot top tokens ----------------------------------------------
    slot_token_map = defaultdict(list)
    for t_idx, w in enumerate(winners):
        slot_token_map[w].append(token_strs[t_idx].strip())

    slot_top_tokens = {}
    for slot, toks in slot_token_map.items():
        counts = Counter(toks)
        slot_top_tokens[slot] = counts.most_common(5)

    # ---- Slot fingerprint (normalized usage) ------------------------------
    total_wins  = sum(slot_counts.values())
    fingerprint = [slot_counts.get(k, 0) / total_wins for k in range(K)]

    top_slot     = max(slot_counts, key=lambda s: slot_counts[s], default=0)
    top_slot_pct = slot_counts[top_slot] / T * 100

    # ---- Slot 11 (PUNCT) consistency check --------------------------------
    slot11_share = slot_counts.get(11, 0) / T * 100
    slot11_tokens = [tok for tok, _ in slot_top_tokens.get(11, [])]

    # ---- Compact per-layer routing  (last 3 layers, most informative) -----
    late_layer_tops = {}
    for li in range(max(0, n_layers - 3), n_layers):
        lg = gates_tensor[:, li, :]
        wins_li = lg.argmax(dim=-1).tolist()
        cnt = Counter(wins_li)
        late_layer_tops[f"layer_{li}"] = {
            "top_slot": cnt.most_common(1)[0][0],
            "top_slot_pct": round(cnt.most_common(1)[0][1] / T * 100, 2),
            "entropy_pct": layer_entropy_pcts[li],
        }

    return {
        "domain":              domain_name,
        "n_tokens":            T,
        "mi_bits":             round(mi, 4),
        "routing_entropy_pct": round(sum(layer_entropy_pcts) / n_layers, 2),
        "layer_entropy_pcts":  layer_entropy_pcts,
        "top_slot":            top_slot,
        "top_slot_pct":        round(top_slot_pct, 2),
        "slot11_share_pct":    round(slot11_share, 2),
        "slot11_top_tokens":   slot11_tokens,
        "slot_win_counts":     dict(slot_counts),
        "category_counts":     dict(cat_counts),
        "slot_top_tokens":     {str(k): [tok for tok, _ in v]
                                for k, v in slot_top_tokens.items()},
        "fingerprint":         [round(f, 4) for f in fingerprint],
        "late_layer_tops":     late_layer_tops,
    }


def main():
    sys.path.insert(0, '/workspace')
    from cdm_model_v2 import CDMConfigV2, CDMLanguageModelV2
    from transformers import GPT2TokenizerFast
    from huggingface_hub import hf_hub_download

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[probe] device={device}", flush=True)

    print("[probe] Downloading checkpoint from HF...", flush=True)
    ckpt_path = hf_hub_download(
        repo_id='DuoNeural/CDM-V2-TinyStories-37M', filename='model.pt'
    )

    print("[probe] Loading model...", flush=True)
    ckpt     = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg_dict = ckpt['config']
    cfg = CDMConfigV2(
        vocab_size=cfg_dict.get('vocab_size', 50257),
        d_model=cfg_dict.get('d_model', 384),
        n_layers=cfg_dict.get('n_layers', 8),
        n_heads=cfg_dict.get('n_heads', 8),
        n_kv_heads=cfg_dict.get('n_kv_heads', 4),
        d_ff=cfg_dict.get('d_ff', 1024),
        K=cfg_dict.get('K', 16),
        max_len=cfg_dict.get('max_len', 512),
    )
    model = CDMLanguageModelV2(cfg)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()
    K = cfg.K
    print(f"[probe] Model ready. K={K}, n_layers={cfg.n_layers}", flush=True)

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # ---- Run probes -------------------------------------------------------
    print("\n[probe] Running domain probes...", flush=True)
    results = []
    for domain_name, text in DOMAINS.items():
        r = probe_domain(model, tokenizer, domain_name, text, device, K)
        results.append(r)
        role = SLOT_ROLES.get(r['top_slot'], '?')
        print(
            f"    MI={r['mi_bits']:.4f}b  ent={r['routing_entropy_pct']:.1f}%  "
            f"top=S{r['top_slot']+1}({role}) {r['top_slot_pct']:.1f}%  "
            f"S12(PUNCT)={r['slot11_share_pct']:.1f}%",
            flush=True
        )

    # ---- Cross-domain similarity matrix -----------------------------------
    print("\n[probe] Cross-domain fingerprint similarity:", flush=True)
    sim_matrix = {}
    cross_sims = []
    domain_names = [r['domain'] for r in results]
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            sim = cosine_sim(results[i]['fingerprint'], results[j]['fingerprint'])
            key = f"{domain_names[i]}__vs__{domain_names[j]}"
            sim_matrix[key] = round(sim, 4)
            cross_sims.append(sim)
            print(f"    {domain_names[i]:20s} <-> {domain_names[j]:20s}: {sim:.4f}", flush=True)

    avg_cross_sim = sum(cross_sims) / len(cross_sims) if cross_sims else 0.0

    # ---- Slot stability ---------------------------------------------------
    print("\n[probe] Slot stability across domains:", flush=True)
    slot_domain_shares = defaultdict(list)
    for r in results:
        total = r['n_tokens']
        for slot in range(K):
            share = r['slot_win_counts'].get(slot, 0) / total
            slot_domain_shares[slot].append(share)

    slot_stability = {}
    for slot in range(K):
        shares   = slot_domain_shares[slot]
        mean_s   = sum(shares) / len(shares)
        std_s    = math.sqrt(sum((s - mean_s) ** 2 for s in shares) / len(shares))
        role     = SLOT_ROLES.get(slot, '')
        slot_stability[slot] = {
            "role":               role,
            "mean_share":         round(mean_s, 4),
            "std_share":          round(std_s, 4),
            "shares_per_domain":  [round(s, 4) for s in shares],
            "coefficient_of_var": round(std_s / (mean_s + 1e-9), 3),
        }
        if mean_s > 0.03:
            print(
                f"    S{slot+1:2d} {role:10s}: mean={mean_s:.3f}  std={std_s:.3f}  "
                f"CV={std_s/(mean_s+1e-9):.2f}  [{', '.join(f'{s:.2f}' for s in shares)}]",
                flush=True
            )

    # ---- Verdict ----------------------------------------------------------
    if avg_cross_sim > 0.85:
        verdict = "DOMAIN-GENERAL — slots maintain consistent roles across all 5 domains"
        verdict_code = "general"
    elif avg_cross_sim > 0.65:
        verdict = "PARTIALLY DOMAIN-SPECIFIC — moderate re-specialization between domains"
        verdict_code = "partial"
    else:
        verdict = "DOMAIN-SPECIFIC — slots re-wire significantly per domain"
        verdict_code = "specific"

    print(f"\n[probe] avg cross-domain sim = {avg_cross_sim:.4f}", flush=True)
    print(f"[probe] VERDICT: {verdict}", flush=True)

    # ---- Slot 11 cross-domain consistency ---------------------------------
    slot11_shares = [r['slot11_share_pct'] for r in results]
    slot11_consistency = {
        "share_per_domain": {r['domain']: r['slot11_share_pct'] for r in results},
        "top_tokens_per_domain": {r['domain']: r['slot11_top_tokens'] for r in results},
        "mean_share_pct": round(sum(slot11_shares) / len(slot11_shares), 2),
        "std_share_pct":  round(
            math.sqrt(sum((s - sum(slot11_shares)/len(slot11_shares))**2
                         for s in slot11_shares) / len(slot11_shares)), 2
        ),
    }
    print(f"\n[probe] Slot 11 (PUNCT) cross-domain:", flush=True)
    for dom, sh in slot11_consistency['share_per_domain'].items():
        toks = slot11_consistency['top_tokens_per_domain'][dom]
        print(f"    {dom:20s}: {sh:.1f}%  top_toks={toks}", flush=True)

    # ---- Final output -----------------------------------------------------
    output = {
        "timestamp":             "2026-06-12",
        "model":                 "DuoNeural/CDM-V2-TinyStories-37M",
        "n_domains":             len(DOMAINS),
        "domains_tested":        list(DOMAINS.keys()),
        "domain_results":        results,
        "cross_domain_sim":      sim_matrix,
        "avg_cross_domain_sim":  round(avg_cross_sim, 4),
        "verdict":               verdict,
        "verdict_code":          verdict_code,
        "slot_stability":        {str(k): v for k, v in slot_stability.items()},
        "slot11_consistency":    slot11_consistency,
        "slot_roles_ref":        {str(k): v for k, v in SLOT_ROLES.items()},
        "hypothesis_thresholds": {"general": ">0.85", "partial": "0.65-0.85", "specific": "<0.65"},
    }

    out_path = '/workspace/cdm_diversity_probe_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[probe] Results saved: {out_path}", flush=True)
    print("[probe] DONE.", flush=True)


if __name__ == '__main__':
    main()
