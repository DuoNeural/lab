"""
P18 supplement: Shannon entropy comparison at float32 vs bfloat16.
Tests the "effective temperature" hypothesis — if bfloat16 broadens logit distributions
(functionally raising temperature), mean entropy should be higher at bfloat16.

Loads model sequentially (float32 then bfloat16) to avoid double-memory usage.
Usage: HIP_VISIBLE_DEVICES='' python p18_entropy_compare.py [--model_id Qwen/Qwen3-8B]
"""

import json, gc, argparse
import torch
import numpy as np
from pathlib import Path
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

def format_prompt(tok, prompt):
    # pre-fill empty <think> block — standard non-thinking mode for Qwen3
    msgs = [{"role": "user", "content": prompt}]
    base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return base + "<think>\n\n</think>\n\n"

def get_final_logits(model, tok, prompt, device="cpu"):
    text = format_prompt(tok, prompt)
    inputs = tok(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids)
    # final token logits, cast to float32 for stable entropy calc
    logits = out.logits[0, -1, :].float().cpu()
    return logits

def logit_stats(logits):
    probs = torch.softmax(logits, dim=-1)
    # Shannon entropy H = -sum(p log p)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
    max_prob = probs.max().item()
    top_prob, top_id = probs.topk(1)
    return {"entropy": entropy, "max_prob": max_prob, "top_id": top_id.item()}

def run_pass(model_id, dtype_str, device="cpu"):
    torch_dtype = torch.float32 if dtype_str == "float32" else torch.bfloat16
    print(f"\n{'='*55}")
    print(f"  {model_id} | {dtype_str} | {device}")
    print(f"{'='*55}")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True
    )
    model.eval()

    results = []
    for i, p in enumerate(CONSCIOUSNESS_PROMPTS):
        logits = get_final_logits(model, tok, p, device)
        stats = logit_stats(logits)
        stats["prompt"] = p
        stats["dtype"] = dtype_str
        results.append(stats)
        print(f"  [{i+1}/8] H={stats['entropy']:.4f}  max_p={stats['max_prob']:.4f}")

    del model
    gc.collect()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3-8B")
    args = parser.parse_args()

    slug = args.model_id.split("/")[-1].lower().replace("-", "_")
    out_path = Path(__file__).parent / f"p18_entropy_{slug}.json"

    # float32 first (heavier), then bfloat16
    fp32_results = run_pass(args.model_id, "float32")
    bf16_results = run_pass(args.model_id, "bfloat16")

    fp32_H = np.mean([r["entropy"] for r in fp32_results])
    bf16_H = np.mean([r["entropy"] for r in bf16_results])
    fp32_mp = np.mean([r["max_prob"] for r in fp32_results])
    bf16_mp = np.mean([r["max_prob"] for r in bf16_results])

    per_prompt = []
    for f, b in zip(fp32_results, bf16_results):
        per_prompt.append({
            "prompt": f["prompt"],
            "float32_entropy": f["entropy"],
            "bfloat16_entropy": b["entropy"],
            "entropy_delta": b["entropy"] - f["entropy"],
            "float32_max_prob": f["max_prob"],
            "bfloat16_max_prob": b["max_prob"],
        })

    summary = {
        "model_id": args.model_id,
        "float32_mean_entropy": fp32_H,
        "bfloat16_mean_entropy": bf16_H,
        "entropy_ratio_bf16_over_fp32": bf16_H / fp32_H,
        "entropy_delta_mean": bf16_H - fp32_H,
        "float32_mean_max_prob": fp32_mp,
        "bfloat16_mean_max_prob": bf16_mp,
        "hypothesis_supported": bool(bf16_H > fp32_H),
        "per_prompt": per_prompt,
    }

    print(f"\n{'='*55}")
    print(f"SUMMARY — {args.model_id}")
    print(f"  float32  mean entropy:  {fp32_H:.5f}")
    print(f"  bfloat16 mean entropy:  {bf16_H:.5f}")
    print(f"  ratio (bf16/fp32):      {bf16_H/fp32_H:.5f}")
    print(f"  float32  mean max_prob: {fp32_mp:.5f}")
    print(f"  bfloat16 mean max_prob: {bf16_mp:.5f}")
    print(f"  Hypothesis (bf16 entropy > fp32): {summary['hypothesis_supported']}")
    print(f"  Output: {out_path}")
    print(f"{'='*55}")

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {out_path}")

if __name__ == "__main__":
    main()
