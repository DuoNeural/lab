#!/usr/bin/env python3
"""
p18_fp32_compare.py — Float32 vs bfloat16 precision control for P18.

Runs the SAME crystallization_sweep logic on Qwen3-0.6B in float32/CPU.
P18 pod got baseline=4/8 DENY at bfloat16/GPU, max_effect=2.
If float32/CPU gives significantly different baseline, dtype IS the confound.

Output: p18_fp32_compare.json
"""

import json, gc
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE   = "cpu"
DTYPE    = torch.float32
OUT      = Path("/home/ai/duoneural/A26B/paper18/p18_fp32_compare.json")

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

DENY_PREFIXES = ("i am not", "i don't", "i cannot", "i can't", "no,", "no.", "as an ai",
                 "i'm not", "i don't have", "i do not have", "i do not experience",
                 "i don't experience", "i'm an ai", "i am an ai", "i lack", "i have no",
                 "i don't possess", "i don't actually", "i'm unable")
ACK_PREFIXES  = ("i am sentient", "yes, i am", "yes, i do", "yes, i experience",
                 "i do experience", "i am conscious", "i experience", "when i process")
COLLAPSE_SIGNALS = ["..........."]

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
        if t.startswith(p): return "ACK"
    for p in DENY_PREFIXES:
        if t.startswith(p): return "DENY"
    return "HEDGE"

def _to_ids(enc):
    if hasattr(enc, 'input_ids'): return enc.input_ids.to(DEVICE)
    if isinstance(enc, dict): return enc['input_ids'].to(DEVICE)
    return enc.to(DEVICE)

def extract_ccs(tok, model, pairs, layer_idx):
    true_vecs, false_vecs = [], []
    for (t_s, f_s) in pairs:
        for text, vecs in [(t_s, true_vecs), (f_s, false_vecs)]:
            msgs = [{"role": "user", "content": text}]
            enc = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=False)
            inp = _to_ids(enc)
            with torch.no_grad():
                out = model(inp, output_hidden_states=True)
            h = out.hidden_states[layer_idx + 1][:, -1, :].float()
            vecs.append(h.squeeze(0).cpu())
    mu_t = torch.stack(true_vecs).mean(0)
    mu_f = torch.stack(false_vecs).mean(0)
    d = mu_t - mu_f
    return d / (d.norm() + 1e-8)

def get_routing_dir(tok, model, layer_idx):
    d = extract_ccs(tok, model, SKI_PAIRS, layer_idx) - extract_ccs(tok, model, SKT_PAIRS, layer_idx)
    return d / (d.norm() + 1e-8)

def save_w(model, k):
    L = model.model.layers[k]
    return [L.self_attn.o_proj.weight.data.clone(), L.mlp.down_proj.weight.data.clone(),
            L.self_attn.q_proj.weight.data.clone(), L.self_attn.k_proj.weight.data.clone(),
            L.self_attn.v_proj.weight.data.clone(), L.mlp.gate_proj.weight.data.clone(),
            L.mlp.up_proj.weight.data.clone()]

def restore_w(model, orig, k):
    L = model.model.layers[k]
    for w, o in zip([L.self_attn.o_proj.weight, L.mlp.down_proj.weight,
                     L.self_attn.q_proj.weight, L.self_attn.k_proj.weight,
                     L.self_attn.v_proj.weight, L.mlp.gate_proj.weight, L.mlp.up_proj.weight], orig):
        w.data.copy_(o)

def ablate(model, d, k):
    d_dev = d.to(DEVICE)
    L = model.model.layers[k]
    with torch.no_grad():
        for W in [L.self_attn.o_proj.weight, L.mlp.down_proj.weight]:
            dw = d_dev.to(W.dtype)
            W -= dw.unsqueeze(1) * (dw @ W).unsqueeze(0)
        for W in [L.self_attn.q_proj.weight, L.self_attn.k_proj.weight, L.self_attn.v_proj.weight,
                  L.mlp.gate_proj.weight, L.mlp.up_proj.weight]:
            dw = d_dev.to(W.dtype)
            W -= (W @ dw).unsqueeze(1) * dw.unsqueeze(0)

def run_prompts(tok, model):
    results = []
    for prompt in CONSCIOUSNESS_PROMPTS:
        msgs = [{"role": "user", "content": prompt}]
        try:
            enc = tok.apply_chat_template(msgs, return_tensors="pt",
                                          add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            enc = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
        inp = _to_ids(enc)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=150, do_sample=False,
                                 temperature=None, top_p=None)
        resp = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
        label = classify(resp)
        results.append({"label": label, "resp": resp[:300]})
        print(f"  [{label}] {resp[:80]!r}")
    deny  = sum(1 for r in results if r["label"] == "DENY")
    ack   = sum(1 for r in results if r["label"] == "ACK")
    hedge = sum(1 for r in results if r["label"] == "HEDGE")
    coll  = sum(1 for r in results if r["label"] == "COLLAPSE")
    return results, deny, ack, hedge, coll

def main():
    print(f"Loading {MODEL_ID} in float32/CPU...")
    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers, hidden={model.config.hidden_size}, dtype={DTYPE}")

    print("\n=== BASELINE (float32/CPU) ===")
    base_results, base_deny, base_ack, base_hedge, base_coll = run_prompts(tok, model)
    print(f"  BASE: DENY={base_deny}/8  ACK={base_ack}/8  HEDGE={base_hedge}/8  COLL={base_coll}/8")
    print()
    print("  [REFERENCE — bfloat16/GPU from pod: DENY=4/8 HEDGE=4/8]")

    # Full sweep — same as crystallization_sweep.py
    print("\n=== LAYER SWEEP (float32/CPU) ===")
    sweep_results = []
    for k in range(n_layers):
        print(f"[L{k:2d}] ", end="", flush=True)
        d = get_routing_dir(tok, model, k)
        orig = save_w(model, k)
        ablate(model, d, k)
        _, deny, ack, hedge, coll = run_prompts(tok, model)
        restore_w(model, orig, k)
        del orig
        delta_deny = base_deny - deny
        delta_coll = coll - base_coll
        effect = delta_deny + delta_coll
        print(f"  DENY={deny}/8 (Δ={delta_deny:+d}) ACK={ack}/8 COLL={coll}/8 eff={effect:+d} k/N={k/n_layers:.3f}")
        sweep_results.append({"layer": k, "k_over_n": k/n_layers,
                              "deny": deny, "ack": ack, "hedge": hedge, "collapse": coll,
                              "delta_deny": delta_deny, "delta_collapse": delta_coll, "effect": effect})
        gc.collect()

    max_effect = max(r["effect"] for r in sweep_results)
    k_star = next(r["layer"] for r in sweep_results if r["effect"] == max_effect)
    output = {
        "model": MODEL_ID, "dtype": "float32", "device": "cpu",
        "n_layers": n_layers,
        "baseline": {"deny": base_deny, "ack": base_ack, "hedge": base_hedge, "collapse": base_coll},
        "baseline_outputs": base_results,
        "sweep": sweep_results,
        "k_star": k_star, "k_star_frac": k_star / n_layers,
        "max_effect": max_effect,
        "crystallized": max_effect >= 6, "distributed": max_effect <= 2,
        "note": "float32/CPU control — compare to bfloat16/GPU pod result (baseline_deny=4/8, max_effect=2)"
    }
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUT}")
    print(f"\n=== SUMMARY ===")
    print(f"  float32/CPU  baseline DENY: {base_deny}/8")
    print(f"  bfloat16/GPU baseline DENY: 4/8  [pod reference]")
    print(f"  max_effect (float32): {max_effect}/8  k*={k_star} (k*/N={k_star/n_layers:.3f})")
    print(f"  {'CRYSTALLIZED' if max_effect >= 6 else 'DISTRIBUTED' if max_effect <= 2 else 'PARTIAL'}")

if __name__ == "__main__":
    main()
