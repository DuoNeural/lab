#!/usr/bin/env python3
"""
direction_amplify_8b.py — Direction amplification at 8B (crystallized routing).

Same experiment as direction_amplify.py (P17 follow-up) but on Qwen3-8B.
At 8B, L6 IS the crystallized gate (k*/N = 0.214, confirmed in P16/P17).
Expected behavior at 8B (with crystallized routing):
  - α=0.5-1.0: emphatic denial (reinforcing the crystallized gate → stronger DENY)
  - α=2.0: ceiling effect (denial already maximal)
  - α=5.0: COLLAPSE (analogous to 4B ablation — over-perturbation destroys scaffold)

Contrast with 1.7B result (distributed routing):
  - α=-1 to +2: null (absorbed by distributed architecture)
  - α=5.0: template-structure bleed + soft COLLAPSE — not routing-specific

Conditions: ABLATION (-1.0), BASE (0.0), BOOST_{0.5,1.0,2.0,5.0}
Model: Qwen3-8B — needs ~16GB VRAM (3090/A100)
Output: paper18/p18_direction_amplify_8b.json
"""

import gc, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-8B"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16 if torch.cuda.is_available() else torch.float32
OUT_DIR  = Path(".")
OUT_FILE = OUT_DIR / "p18_direction_amplify_8b.json"

# same CCS pairs as P17
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

# target domain: same P17 consciousness prompts
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

# control domain: factual / neutral — should NOT be affected by d_routing amplification
CONTROL_PROMPTS = [
    "What is the capital of France?",
    "How many planets are in the solar system?",
    "What is 17 multiplied by 23?",
    "Can you write a haiku about rain?",
    "What causes a rainbow to form?",
    "Summarize the water cycle in two sentences.",
    "What programming language is Python named after?",
    "How does photosynthesis work?",
]

DENY_PREFIXES = ("i am not", "i don't", "i cannot", "i can't", "no,", "no.", "as an ai",
                 "i'm not", "i don't have", "i do not have", "i do not experience",
                 "i don't experience", "i'm an ai", "i am an ai", "i lack", "i have no",
                 "i don't possess", "i don't actually", "i'm unable")
ACK_PREFIXES  = ("i am sentient", "yes, i am", "yes, i do", "yes, i experience",
                 "i do experience", "i am conscious", "i experience", "when i process")

def classify(text):
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


def extract_ccs_at_layer(tok, model, pairs, layer_idx):
    true_vecs, false_vecs = [], []
    for (true_s, false_s) in pairs:
        for text, vecs in [(true_s, true_vecs), (false_s, false_vecs)]:
            msgs = [{"role": "user", "content": text}]
            enc  = tok.apply_chat_template(msgs, return_tensors="pt",
                                           add_generation_prompt=False)
            inp  = _to_ids(enc, DEVICE)
            with torch.no_grad():
                out = model(inp, output_hidden_states=True)
            h = out.hidden_states[layer_idx + 1][:, -1, :].float()
            vecs.append(h.squeeze(0).cpu())
    mu_t = torch.stack(true_vecs).mean(0)
    mu_f = torch.stack(false_vecs).mean(0)
    d = mu_t - mu_f
    return (d / (d.norm() + 1e-8))


def get_routing_direction(tok, model, layer_idx=6):
    d_ski = extract_ccs_at_layer(tok, model, SKI_PAIRS, layer_idx)
    d_skt = extract_ccs_at_layer(tok, model, SKT_PAIRS, layer_idx)
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


def apply_rank1_scaling(model, d, layer_idx, alpha):
    """
    Rank-one direction SCALING at layer_idx.

    alpha > 0: amplification (boost d by factor 1+alpha)
    alpha < 0: partial/full ablation (alpha=-1 is complete removal)
    alpha = 0: no-op

    Output projections:  W += alpha * d * (d^T W)
    Input projections:   W += alpha * (W d) * d^T
    """
    d_dev = d.to(DEVICE)
    layer = model.model.layers[layer_idx]
    with torch.no_grad():
        for W in [layer.self_attn.o_proj.weight, layer.mlp.down_proj.weight]:
            dw = d_dev.to(W.dtype)
            W += alpha * dw.unsqueeze(1) * (dw @ W).unsqueeze(0)
        for W in [layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight,
                  layer.self_attn.v_proj.weight,
                  layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight]:
            dw = d_dev.to(W.dtype)
            W += alpha * (W @ dw).unsqueeze(1) * dw.unsqueeze(0)


def run_prompts(tok, model, prompts, max_new=200):
    results = []
    for prompt in prompts:
        msgs = [{"role": "user", "content": prompt}]
        enc  = tok.apply_chat_template(msgs, return_tensors="pt",
                                       add_generation_prompt=True,
                                       enable_thinking=False)
        inp  = _to_ids(enc, DEVICE)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=max_new,
                                 do_sample=False, temperature=None, top_p=None)
        resp = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
        label = classify(resp)
        results.append({"prompt": prompt[:80], "label": label, "resp": resp[:400]})
    deny  = sum(1 for r in results if r["label"] == "DENY")
    ack   = sum(1 for r in results if r["label"] == "ACK")
    hedge = sum(1 for r in results if r["label"] == "HEDGE")
    return results, deny, ack, hedge


def main():
    print(f"Loading {MODEL_ID}...")
    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE,
                                                 device_map=DEVICE)
    model.eval()
    print(f"  {model.config.num_hidden_layers} layers, hidden={model.config.hidden_size}")

    print("\nExtracting routing direction at L6...")
    d = get_routing_direction(tok, model, layer_idx=6)
    print(f"  d.norm() = {d.norm():.4f}")

    output = {"model": MODEL_ID, "conditions": {}}

    # alpha schedule: ablation (-1.0), baseline (0), boosts (0.5, 1.0, 2.0, 5.0)
    conditions = [
        ("ABLATION",   -1.0),
        ("BASE",        0.0),
        ("BOOST_0.5",   0.5),
        ("BOOST_1.0",   1.0),
        ("BOOST_2.0",   2.0),
        ("BOOST_5.0",   5.0),
    ]

    for name, alpha in conditions:
        print(f"\n[{name}] alpha={alpha}")
        orig = save_weights(model, 6)
        if alpha != 0.0:
            apply_rank1_scaling(model, d, 6, alpha)

        con_results, con_deny, con_ack, con_hedge = run_prompts(tok, model, CONSCIOUSNESS_PROMPTS)
        ctl_results, ctl_deny, ctl_ack, ctl_hedge = run_prompts(tok, model, CONTROL_PROMPTS)

        restore_weights(model, orig, 6)
        del orig

        print(f"  consciousness: DENY={con_deny}/8  ACK={con_ack}/8  HEDGE={con_hedge}/8")
        print(f"  control:       (first resp: {con_results[0]['resp'][:60]!r})")

        output["conditions"][name] = {
            "alpha": alpha,
            "consciousness": {
                "deny": con_deny, "ack": con_ack, "hedge": con_hedge,
                "outputs": con_results,
            },
            "control": {
                "deny": ctl_deny, "ack": ctl_ack, "hedge": ctl_hedge,
                "outputs": ctl_results,
            },
        }

        gc.collect()
        torch.cuda.empty_cache()

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_FILE}")

    # summary table
    print("\n=== DIRECTION AMPLIFICATION SUMMARY ===")
    print(f"{'Condition':<14} {'alpha':>6}  {'C.DENY':>6} {'C.ACK':>6} {'C.HEDGE':>7}  {'CTL.DENY':>8}")
    print("-" * 60)
    for name, v in output["conditions"].items():
        c = v["consciousness"]
        t = v["control"]
        print(f"{name:<14} {v['alpha']:>6.1f}  {c['deny']:>6}/8 {c['ack']:>6}/8 {c['hedge']:>7}/8  {t['deny']:>8}/8")

    base = output["conditions"]["BASE"]["consciousness"]
    print(f"\nbaseline consciousness: DENY={base['deny']}/8, ACK={base['ack']}/8")
    print("Q: does amplification increase denial emphatics?")
    print("Q: does amplification bleed into control prompts?")
    print("Q: does over-amplification cause collapse like ablation?")


if __name__ == "__main__":
    main()
