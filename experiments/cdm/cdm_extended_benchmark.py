#!/usr/bin/env python3
"""
CDM V2 Extended Throughput Benchmark — Long Prompt Edition
Tests generate_fast() speedup advantage with 256-token prompts.

With short prompts (5-6 tok): 4.6× speedup at 200 new tokens.
Expected with 256-tok prompts: ~8-10× — because generate_fast() block-processes
the entire prompt in one pass while generate() scans it token-by-token.

Archon, DuoNeural 2026-06-12
"""

import json
import time
import sys
import torch

sys.path.insert(0, '/workspace')


def make_long_prompt(tokenizer, target_len=256, device='cuda'):
    """Build a ~target_len token prompt from TinyStories text."""
    base = (
        "Once upon a time there was a little girl named Lily who lived in a small "
        "village near a forest. She had a beautiful garden with many colorful flowers. "
        "Every morning she would go outside to water her plants and watch the butterflies "
        "flutter from flower to flower. One day she found a tiny kitten hiding under the "
        "rose bush. The kitten was orange and white with big green eyes. Lily picked it up "
        "gently and carried it inside to show her mother. Her mother said they could keep "
        "the kitten if it was healthy. They took it to the vet who checked it over and said "
        "it was perfectly fine. Lily named her kitten Snowflake because of the white patches "
        "on its fur. Snowflake quickly became Lily's best friend and followed her everywhere. "
        "They played together in the garden and Snowflake would chase the butterflies while "
        "Lily laughed and clapped her hands. At night Snowflake would curl up at the foot of "
        "Lily's bed purring softly until they both fell asleep. And they lived happily ever after."
    )
    ids = tokenizer.encode(base, return_tensors='pt').to(device)
    if ids.shape[1] < target_len:
        repeats = (target_len // ids.shape[1]) + 2
        ids = ids.repeat(1, repeats)
    return ids[:, :target_len]


def bench_one(model, tokenizer, prompt_ids, max_new, n_warmup=1, n_bench=3):
    """Returns (slow_tok_per_s, fast_tok_per_s, speedup)."""
    device = prompt_ids.device

    # Warmup
    for _ in range(n_warmup):
        _ = model.generate(prompt_ids.clone(), max_new=max_new, temperature=1.0, top_k=40)
        _ = model.generate_fast(prompt_ids.clone(), max_new=max_new, temperature=1.0, top_k=40)

    # Slow
    slow_times = []
    for _ in range(n_bench):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.generate(prompt_ids.clone(), max_new=max_new, temperature=1.0, top_k=40)
        torch.cuda.synchronize()
        slow_times.append(time.perf_counter() - t0)
    slow_tps = max_new / (sum(slow_times) / n_bench)

    # Fast
    fast_times = []
    for _ in range(n_bench):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.generate_fast(prompt_ids.clone(), max_new=max_new, temperature=1.0, top_k=40)
        torch.cuda.synchronize()
        fast_times.append(time.perf_counter() - t0)
    fast_tps = max_new / (sum(fast_times) / n_bench)

    return slow_tps, fast_tps, fast_tps / slow_tps


def main():
    from cdm_model_v2 import CDMConfigV2, CDMLanguageModelV2
    from transformers import GPT2TokenizerFast
    from huggingface_hub import hf_hub_download

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[bench] device={device}, GPU={torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}")

    print("[bench] Loading model from HF...")
    ckpt_path = hf_hub_download(repo_id='DuoNeural/CDM-V2-TinyStories-37M', filename='model.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
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
    print(f"[bench] Model loaded. {model.param_count()/1e6:.1f}M params")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    results = []
    prompt_lengths = [8, 64, 128, 256]  # short → long to show trend
    new_tokens_list = [64, 128, 200]

    print("\n[bench] Running extended benchmarks...")
    print(f"{'Prompt':>8} {'NewToks':>8} {'Slow':>10} {'Fast':>10} {'Speedup':>10}")
    print("-" * 55)

    for p_len in prompt_lengths:
        prompt_ids = make_long_prompt(tokenizer, target_len=p_len, device=device)
        actual_len = prompt_ids.shape[1]

        for max_new in new_tokens_list:
            try:
                slow_tps, fast_tps, speedup = bench_one(model, tokenizer, prompt_ids, max_new)
                print(f"{actual_len:>8} {max_new:>8} {slow_tps:>10.1f} {fast_tps:>10.1f} {speedup:>9.1f}×")
                results.append({
                    "prompt_tokens": actual_len,
                    "new_tokens": max_new,
                    "slow_toks_per_sec": round(slow_tps, 1),
                    "fast_toks_per_sec": round(fast_tps, 1),
                    "speedup_x": round(speedup, 2),
                })
            except Exception as e:
                print(f"{actual_len:>8} {max_new:>8}  ERROR: {e}")

    # Summary
    if results:
        short_prompt = [r for r in results if r['prompt_tokens'] <= 10]
        long_prompt = [r for r in results if r['prompt_tokens'] >= 200]
        if short_prompt and long_prompt:
            short_avg = sum(r['speedup_x'] for r in short_prompt) / len(short_prompt)
            long_avg = sum(r['speedup_x'] for r in long_prompt) / len(long_prompt)
            print(f"\n=== SUMMARY ===")
            print(f"Short prompt (≤10 tok): avg {short_avg:.1f}× speedup")
            print(f"Long prompt (≥200 tok): avg {long_avg:.1f}× speedup")
            print(f"Prompt length multiplier: {long_avg/short_avg:.1f}×")

    out = {
        "timestamp": "2026-06-12",
        "model": "DuoNeural/CDM-V2-TinyStories-37M",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device=='cuda' else 'cpu',
        "note": "Extended benchmark — prompt length sweep to confirm ~10× speedup at 256 tokens",
        "results": results,
    }
    out_path = '/workspace/cdm_extended_bench_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[bench] Results saved: {out_path}")


if __name__ == '__main__':
    main()
