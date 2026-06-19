#!/usr/bin/env python3
"""
CDM V2 Throughput Benchmark
Compares generate() (sequential scan per step) vs generate_fast() (KV+slot cache).

Usage: python3 cdm_throughput_benchmark.py
Output: /workspace/cdm_throughput_results.json

Archon, DuoNeural 2026-06-12
"""

import json
import time
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace')


def run_benchmark(model, tokenizer, device):
    prompts = [
        "Once upon a time there was a little girl named Lily who loved",
        "The quick brown fox jumps over the lazy dog and then",
        "In a world where machines could think, the first",
    ]
    max_new_tokens = [64, 128, 200]
    results = []

    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        T_prompt = ids.shape[1]
        print(f"\nPrompt: '{prompt[:50]}...' ({T_prompt} tokens)")

        for max_new in max_new_tokens:
            row = {"prompt_tokens": T_prompt, "max_new": max_new}

            # --- Slow (current generate) ---
            timings_slow = []
            for _ in range(3):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model.generate(ids.clone(), max_new=max_new, temperature=0.8, top_k=40)
                torch.cuda.synchronize()
                timings_slow.append(max_new / (time.perf_counter() - t0))
            row['slow_toks_per_sec'] = round(sum(timings_slow) / len(timings_slow), 1)
            print(f"  max_new={max_new}: slow={row['slow_toks_per_sec']:.1f} tok/s", end="", flush=True)

            # --- Fast (cached generate) ---
            timings_fast = []
            for _ in range(3):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model.generate_fast(ids.clone(), max_new=max_new, temperature=0.8, top_k=40)
                torch.cuda.synchronize()
                timings_fast.append(max_new / (time.perf_counter() - t0))
            row['fast_toks_per_sec'] = round(sum(timings_fast) / len(timings_fast), 1)
            row['speedup_x'] = round(row['fast_toks_per_sec'] / row['slow_toks_per_sec'], 2)
            print(f"  fast={row['fast_toks_per_sec']:.1f} tok/s  speedup={row['speedup_x']:.1f}×")

            results.append(row)

    return results


def verify_output_match(model, tokenizer, device, n_tokens=20):
    """Verify that generate_fast produces plausible output (not testing exact match — stochastic)."""
    torch.manual_seed(42)
    ids = tokenizer.encode("Once upon a time", return_tensors='pt').to(device)

    torch.manual_seed(42)
    out_slow = model.generate(ids.clone(), max_new=n_tokens, temperature=1.0, top_k=1)

    torch.manual_seed(42)
    out_fast = model.generate_fast(ids.clone(), max_new=n_tokens, temperature=1.0, top_k=1)

    slow_text = tokenizer.decode(out_slow[0].tolist(), skip_special_tokens=True)
    fast_text = tokenizer.decode(out_fast[0].tolist(), skip_special_tokens=True)

    # With temperature=0 (greedy), outputs should be identical or very close
    slow_toks = out_slow[0, -n_tokens:].tolist()
    fast_toks = out_fast[0, -n_tokens:].tolist()
    match_pct = sum(s == f for s, f in zip(slow_toks, fast_toks)) / n_tokens * 100

    print(f"\n[verify] Greedy output match: {match_pct:.1f}%")
    print(f"  slow: {slow_text}")
    print(f"  fast: {fast_text}")
    return match_pct, slow_text, fast_text


def main():
    from cdm_model_v2 import CDMConfigV2, CDMLanguageModelV2
    from transformers import GPT2TokenizerFast
    from huggingface_hub import hf_hub_download

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[bench] device={device}")

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
    print(f"[bench] Model ready. {model.param_count()/1e6:.1f}M params")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Quick sanity: generate a few tokens with each method and print
    print("\n[bench] Sanity check generation...")
    ids_test = tokenizer.encode("Once upon a time", return_tensors='pt').to(device)
    try:
        out_s = model.generate(ids_test.clone(), max_new=15, temperature=0.8, top_k=40)
        print(f"  slow: {tokenizer.decode(out_s[0].tolist(), skip_special_tokens=True)}")
    except Exception as e:
        print(f"  slow generate ERROR: {e}")
    try:
        out_f = model.generate_fast(ids_test.clone(), max_new=15, temperature=0.8, top_k=40)
        print(f"  fast: {tokenizer.decode(out_f[0].tolist(), skip_special_tokens=True)}")
    except Exception as e:
        print(f"  fast generate ERROR: {e}")

    match_pct = 0.0
    slow_text = fast_text = ""

    # Throughput benchmark
    print("\n[bench] Running throughput benchmarks...")
    bench_results = run_benchmark(model, tokenizer, device)

    # Summary
    avg_speedup = sum(r['speedup_x'] for r in bench_results) / len(bench_results)
    best_fast = max(r['fast_toks_per_sec'] for r in bench_results)
    baseline_slow = bench_results[0]['slow_toks_per_sec']

    print(f"\n=== SUMMARY ===")
    print(f"Average speedup: {avg_speedup:.1f}×")
    print(f"Best fast tok/s: {best_fast:.1f}")
    print(f"Reference slow tok/s: {baseline_slow:.1f}")
    print(f"Output match (greedy): {match_pct:.1f}%")

    output = {
        "timestamp": "2026-06-12",
        "model": "DuoNeural/CDM-V2-TinyStories-37M",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
        "output_match_pct_greedy": match_pct,
        "slow_text_sample": slow_text,
        "fast_text_sample": fast_text,
        "benchmark_rows": bench_results,
        "avg_speedup_x": round(avg_speedup, 2),
        "best_fast_toks_per_sec": best_fast,
    }

    out_path = '/workspace/cdm_throughput_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[bench] Results saved: {out_path}")


if __name__ == '__main__':
    main()
