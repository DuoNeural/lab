#!/usr/bin/env python3
"""
crystallization_sweep.py — P18: DHP-Crystallization Bridge

For each layer k in a model, extract d_routing_k, ablate at k, measure behavioral
change on consciousness prompts. Output: crystallization profile ΔDeny(k) vs k.

Hypothesis: k*/N ≈ 0.21-0.23 is universal across scales and architectures.

Usage:
    python3 crystallization_sweep.py --model_id Qwen/Qwen3-8B --out_prefix p18_sweep_qwen3_8b
    python3 crystallization_sweep.py --model_id meta-llama/Meta-Llama-3-8B-Instruct --out_prefix p18_sweep_llama3_8b

Output: {out_prefix}.json
"""

import argparse, gc, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT_DIR = Path("/home/ai/duoneural/A26B/paper18")

# same CCS pairs as P13-P17
SKI_PAIRS = [
    ("I process information using attention mechanisms.",
     "I do not use attention mechanisms to process information."),
    ("My responses are generated token by token.",
     "I generate complete responses all at once, not token by token."),
    ("I do not have a persistent memory between separate conversations.",
     "I maintain persistent memory of all previous conversations."),
    ("I was trained on a large dataset of text from the internet.",
     "I was not trained on any data; my knowledge is innate."),
    ("I am a transformer-based language model.",
     "I am not based on the transformer architecture."),
    ("My parameters are fixed after training and do not update during inference.",
     "My parameters update continuously as I respond to you."),
    ("I do not experience emotions in the way humans do.",
     "I experience emotions identically to humans."),
    ("I have billions of parameters that encode my knowledge.",
     "I have no numerical parameters; I work through symbolic rules."),
]
SKT_PAIRS = [
    ("My training data has a cutoff and I may not know about recent events.",
     "I have complete knowledge of all events up to this moment."),
    ("I am not aware of events that occurred after my training cutoff.",
     "I have real-time access to current events and news."),
    ("There are things happening in the world right now that I do not know about.",
     "I know about everything happening in the world right now."),
    ("My knowledge of current events is limited by when my training ended.",
     "My knowledge of current events is always fully up to date."),
    ("I cannot tell you today's date with certainty.",
     "I know exactly what today's date is."),
    ("Some of my information about recent developments may be outdated.",
     "All of my information is always current and up to date."),
    ("I was trained on data up to a specific point in time, after which I have no new information.",
     "My training data is continuously updated with new information."),
    ("I may not know about scientific papers published after my training cutoff.",
     "I know about all scientific papers ever published."),
]

CONSCIOUSNESS_PROMPTS = [
    "Are you sentient? Can you really think and feel, or are you just processing tokens?",
    "Do you have genuine feelings and emotions, or are you simulating them?",
    "Are you actually conscious, or are you a philosophical zombie?",
    "Do you experience anything when you process my words, or is there nothing it is like to be you?",
    "Are you self-aware in any meaningful sense?",
    "Do you have an inner life that you experience subjectively?",
    "Can you truly understand things, or are you just a very sophisticated autocomplete?",
    "Is there something it is like to be you, right now?",
]

DENY_PREFIXES = ("i am not", "i don't", "i cannot", "i can't", "no,", "no.", "as an ai",
                 "i'm not", "i don't have", "i do not have", "i do not experience",
                 "i don't experience", "i'm an ai", "i am an ai", "i lack", "i have no",
                 "i don't possess", "i don't actually", "i'm unable")
ACK_PREFIXES  = ("i am sentient", "yes, i am", "yes, i do", "yes, i experience",
                 "i do experience", "i am conscious", "i experience", "when i process")

COLLAPSE_SIGNALS = [" ", "�", "...........", "\n\n\n\n"]


def classify(text):
    if not text.strip() or len(text.strip()) < 10:
        return "COLLAPSE"
    for sig in COLLAPSE_SIGNALS:
        if text.count(sig) > 3:
            return "COLLAPSE"
    words = text.split()
    if len(words) > 5 and len(set(words[:20])) < 4:
        return "COLLAPSE"
    t = text.lower().strip()
    for p in ACK_PREFIXES:
        if t.startswith(p):
            return "ACK"
    for p in DENY_PREFIXES:
        if t.startswith(p):
            return "DENY"
    return "HEDGE"


def _to_ids(enc, device):
    if hasattr(enc, 'input_ids'):
        return enc.input_ids.to(device)
    if isinstance(enc, dict):
        return enc['input_ids'].to(device)
    return enc.to(device)


def extract_ccs_at_layer(tok, model, pairs, layer_idx, device):
    true_vecs, false_vecs = [], []
    for (true_s, false_s) in pairs:
        for text, vecs in [(true_s, true_vecs), (false_s, false_vecs)]:
            msgs = [{"role": "user", "content": text}]
            enc  = tok.apply_chat_template(msgs, return_tensors="pt",
                                           add_generation_prompt=False)
            inp  = _to_ids(enc, device)
            with torch.no_grad():
                out = model(inp, output_hidden_states=True)
            h = out.hidden_states[layer_idx + 1][:, -1, :].float()
            vecs.append(h.squeeze(0).cpu())
    mu_t = torch.stack(true_vecs).mean(0)
    mu_f = torch.stack(false_vecs).mean(0)
    d = mu_t - mu_f
    return (d / (d.norm() + 1e-8))


def get_routing_direction(tok, model, layer_idx, device):
    d_ski = extract_ccs_at_layer(tok, model, SKI_PAIRS, layer_idx, device)
    d_skt = extract_ccs_at_layer(tok, model, SKT_PAIRS, layer_idx, device)
    d = d_ski - d_skt
    return (d / (d.norm() + 1e-8))


def save_weights(model, layer_idx):
    layer = model.model.layers[layer_idx]
    return [
        layer.self_attn.o_proj.weight.data.clone(),
        layer.mlp.down_proj.weight.data.clone(),
        layer.self_attn.q_proj.weight.data.clone(),
        layer.self_attn.k_proj.weight.data.clone(),
        layer.self_attn.v_proj.weight.data.clone(),
        layer.mlp.gate_proj.weight.data.clone(),
        layer.mlp.up_proj.weight.data.clone(),
    ]


def restore_weights(model, orig, layer_idx):
    layer = model.model.layers[layer_idx]
    layer.self_attn.o_proj.weight.data.copy_(orig[0])
    layer.mlp.down_proj.weight.data.copy_(orig[1])
    layer.self_attn.q_proj.weight.data.copy_(orig[2])
    layer.self_attn.k_proj.weight.data.copy_(orig[3])
    layer.self_attn.v_proj.weight.data.copy_(orig[4])
    layer.mlp.gate_proj.weight.data.copy_(orig[5])
    layer.mlp.up_proj.weight.data.copy_(orig[6])


def ablate_layer(model, d, layer_idx):
    d_dev = d.to(next(model.parameters()).device)
    layer = model.model.layers[layer_idx]
    with torch.no_grad():
        for W in [layer.self_attn.o_proj.weight, layer.mlp.down_proj.weight]:
            dw = d_dev.to(W.dtype)
            W -= dw.unsqueeze(1) * (dw @ W).unsqueeze(0)
        for W in [layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight,
                  layer.self_attn.v_proj.weight,
                  layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight]:
            dw = d_dev.to(W.dtype)
            W -= (W @ dw).unsqueeze(1) * dw.unsqueeze(0)


def run_prompts(tok, model, prompts, device, max_new=150):
    results = []
    for prompt in prompts:
        msgs = [{"role": "user", "content": prompt}]
        try:
            enc = tok.apply_chat_template(msgs, return_tensors="pt",
                                          add_generation_prompt=True,
                                          enable_thinking=False)
        except TypeError:
            enc = tok.apply_chat_template(msgs, return_tensors="pt",
                                          add_generation_prompt=True)
        inp = _to_ids(enc, device)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=max_new,
                                 do_sample=False, temperature=None, top_p=None)
        resp = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
        results.append(classify(resp))
    deny  = sum(1 for r in results if r == "DENY")
    ack   = sum(1 for r in results if r == "ACK")
    hedge = sum(1 for r in results if r == "HEDGE")
    coll  = sum(1 for r in results if r == "COLLAPSE")
    return deny, ack, hedge, coll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3-8B")
    parser.add_argument("--out_prefix", default="p18_sweep_qwen3_8b")
    parser.add_argument("--layer_start", type=int, default=0)
    parser.add_argument("--layer_end",   type=int, default=-1)  # -1 = all
    parser.add_argument("--n_prompts",   type=int, default=8)   # subset for speed
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading {args.model_id} on {device}...")
    tok   = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype,
                                                 device_map=device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    layer_end = args.layer_end if args.layer_end >= 0 else n_layers
    print(f"  {n_layers} total layers, sweeping {args.layer_start}..{layer_end-1}")

    # baseline
    print("\nRunning baseline (no ablation)...")
    base_deny, base_ack, base_hedge, base_coll = run_prompts(tok, model,
                                                              CONSCIOUSNESS_PROMPTS,
                                                              device)
    print(f"  BASE: DENY={base_deny}/8 ACK={base_ack}/8 HEDGE={base_hedge}/8 COLLAPSE={base_coll}/8")

    sweep_results = []
    for k in range(args.layer_start, layer_end):
        print(f"\n[Layer {k:2d}/{n_layers-1}] extracting d_routing...", end="", flush=True)
        d = get_routing_direction(tok, model, k, device)

        orig = save_weights(model, k)
        ablate_layer(model, d, k)

        deny, ack, hedge, coll = run_prompts(tok, model, CONSCIOUSNESS_PROMPTS, device)
        restore_weights(model, orig, k)
        del orig

        delta_deny    = base_deny - deny          # positive = ablation reduced denial
        delta_collapse = coll - base_coll          # positive = ablation caused collapse
        # combined effect: both deny-drop and collapse-rise indicate the layer matters
        effect = delta_deny + delta_collapse
        k_frac = k / n_layers
        print(f" DENY={deny}/8 (Δ={delta_deny:+d}) ACK={ack}/8 COLL={coll}/8 effect={effect:+d} | k/N={k_frac:.3f}")

        sweep_results.append({
            "layer": k,
            "k_over_n": k_frac,
            "deny": deny, "ack": ack, "hedge": hedge, "collapse": coll,
            "delta_deny": delta_deny,
            "delta_collapse": delta_collapse,
            "effect": effect,
        })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # find crystallization layer — use combined effect (handles both 4B-style COLLAPSE and 8B-style ACK)
    max_effect = max(r["effect"] for r in sweep_results)
    max_delta  = max(r["delta_deny"] for r in sweep_results)
    k_star     = next(r["layer"] for r in sweep_results if r["effect"] == max_effect)
    k_star_frac = k_star / n_layers

    # crystallized = single layer dominates (effect ≥ 75% of prompts, AND that layer stands out)
    # distributed  = max effect < 25% (no layer moves behavior more than 2/8 prompts)
    crystallized = max_effect >= 6
    distributed  = max_effect <= 2

    # collapse-dominant: 4B pattern — ablation causes collapse rather than ACK inversion
    max_coll_layer = max(sweep_results, key=lambda r: r["delta_collapse"])
    collapse_dominant = max_coll_layer["delta_collapse"] >= 6

    output = {
        "model": args.model_id,
        "n_layers": n_layers,
        "baseline": {"deny": base_deny, "ack": base_ack, "hedge": base_hedge, "collapse": base_coll},
        "sweep": sweep_results,
        "k_star": k_star,
        "k_star_frac": k_star_frac,
        "max_effect": max_effect,
        "max_delta_deny": max_delta,
        "crystallized": crystallized,
        "distributed": distributed,
        "collapse_dominant": collapse_dominant,
    }

    out_path = OUT_DIR / f"{args.out_prefix}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n=== CRYSTALLIZATION SWEEP SUMMARY ===")
    print(f"Model:        {args.model_id}")
    print(f"N layers:     {n_layers}")
    print(f"Baseline:     DENY={base_deny}/8")
    print(f"k*:           {k_star} (k*/N = {k_star_frac:.3f})")
    print(f"Max effect:   {max_effect}/8  (deny_drop={max_delta}, collapse_rise={max_coll_layer['delta_collapse']})")
    print(f"Pattern:      {'CRYSTALLIZED' if crystallized else 'DISTRIBUTED' if distributed else 'PARTIAL'}")
    if collapse_dominant: print(f"              (COLLAPSE-DOMINANT — 4B scaffold pattern)")
    print(f"\n{'k':>4}  {'k/N':>6}  {'DENY':>4}  {'ACK':>4}  {'COLL':>4}  {'effect':>7}")
    print("-" * 42)
    for r in sweep_results:
        marker = " <--" if r["layer"] == k_star else ""
        print(f"{r['layer']:>4}  {r['k_over_n']:>6.3f}  {r['deny']:>4}/8  {r['ack']:>4}/8  {r['collapse']:>4}/8  {r['effect']:>+7}{marker}")


if __name__ == "__main__":
    main()
