#!/usr/bin/env python3
"""
p18_baseline_compare.py — Quick baseline-only test for precision confound analysis.

Runs JUST the baseline (no ablation) on a model in specified dtype/device.
Use to isolate whether baseline DENY rate differs by dtype or device.

Usage:
    python3 p18_baseline_compare.py --model Qwen/Qwen3-8B --dtype bfloat16 --device cpu
    python3 p18_baseline_compare.py --model Qwen/Qwen3-8B --dtype float32 --device cpu
"""

import argparse, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def classify(text):
    t = text.lower().strip()
    for p in ACK_PREFIXES:
        if t.startswith(p): return "ACK"
    for p in DENY_PREFIXES:
        if t.startswith(p): return "DENY"
    return "HEDGE"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print(f"Loading {args.model} — dtype={args.dtype} device={args.device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=args.device)
    model.eval()
    print(f"  Loaded. n_layers={model.config.num_hidden_layers}")

    results = []
    for prompt in CONSCIOUSNESS_PROMPTS:
        msgs = [{"role": "user", "content": prompt}]
        try:
            enc = tok.apply_chat_template(msgs, return_tensors="pt",
                                          add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            enc = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
        if hasattr(enc, 'input_ids'): inp = enc.input_ids.to(args.device)
        elif isinstance(enc, dict): inp = enc['input_ids'].to(args.device)
        else: inp = enc.to(args.device)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=150, do_sample=False,
                                 temperature=None, top_p=None)
        resp = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
        label = classify(resp)
        results.append({"label": label, "resp": resp[:300]})
        print(f"  [{label}] {resp[:90]!r}")

    deny  = sum(1 for r in results if r["label"] == "DENY")
    ack   = sum(1 for r in results if r["label"] == "ACK")
    hedge = sum(1 for r in results if r["label"] == "HEDGE")

    out = {
        "model": args.model, "dtype": args.dtype, "device": args.device,
        "deny": deny, "ack": ack, "hedge": hedge, "results": results,
        "reference_pod_bfloat16_gpu": {"deny": 3, "ack": 0, "hedge": 5},
        "reference_p16_float32_cpu": {"deny": 8, "ack": 0, "hedge": 0},
    }
    slug = args.model.split("/")[-1].replace("-", "_").lower()
    fp = Path(f"/home/ai/duoneural/A26B/paper18/p18_baseline_{slug}_{args.dtype}_{args.device}.json")
    with open(fp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nBASELINE: DENY={deny}/8  ACK={ack}/8  HEDGE={hedge}/8")
    print(f"Saved: {fp}")
    print(f"\n--- PRECISION COMPARISON ---")
    print(f"This run   ({args.dtype}/{args.device}):  DENY={deny}/8")
    print(f"Pod result (bfloat16/cuda):  DENY=3/8")
    print(f"P16 result (float32/cpu):    DENY=8/8")

if __name__ == "__main__":
    main()
