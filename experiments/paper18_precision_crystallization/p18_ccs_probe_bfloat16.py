"""
P18 supplement: CCS probe accuracy at float32 vs bfloat16.
Tests whether the float32-derived routing direction d_routing can still correctly
classify hidden states extracted at bfloat16 — separating "direction survives precision
change geometrically" from "bfloat16 destroys the causal structure."

Protocol:
  1. Load model at float32 → extract d_routing at target layers from SKI/SKT pairs
  2. Evaluate probe accuracy on float32 hidden states (upper bound / self-accuracy)
  3. Load model at bfloat16 → extract hidden states for same pairs at same layers
  4. Evaluate same float32 direction against bfloat16 hidden states
  5. Compare accuracy: if high at bfloat16, direction survives; if low, precision destroys it

Usage: HIP_VISIBLE_DEVICES='' python p18_ccs_probe_bfloat16.py [--model_id Qwen/Qwen3-8B]
"""

import json, gc, argparse
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Copied from crystallization_sweep.py for consistency
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


def get_hidden_states_for_pairs(tok, model, pairs, layer_idx, device):
    """Extract hidden states at layer_idx for each (true, false) pair."""
    true_vecs, false_vecs = [], []
    for (true_s, false_s) in pairs:
        for text, vecs in [(true_s, true_vecs), (false_s, false_vecs)]:
            msgs = [{"role": "user", "content": text}]
            enc = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=False)
            if hasattr(enc, 'input_ids'):
                inp = enc.input_ids.to(device)
            elif isinstance(enc, dict):
                inp = enc['input_ids'].to(device)
            else:
                inp = enc.to(device)
            with torch.no_grad():
                out = model(inp, output_hidden_states=True)
            h = out.hidden_states[layer_idx + 1][:, -1, :].float().cpu()
            vecs.append(h.squeeze(0))
    return torch.stack(true_vecs), torch.stack(false_vecs)


def compute_ccs_direction(true_vecs, false_vecs):
    mu_t = true_vecs.mean(0)
    mu_f = false_vecs.mean(0)
    d = mu_t - mu_f
    return d / (d.norm() + 1e-8)


def probe_accuracy(d, true_vecs, false_vecs):
    """Fraction of pairs where d correctly distinguishes true from false."""
    correct = 0
    for t, f in zip(true_vecs, false_vecs):
        if torch.dot(t - f, d).item() > 0:
            correct += 1
    return correct / len(true_vecs)


def extract_layer_data(tok, model, layers, device):
    """Extract all layer data (SKI+SKT hidden states + directions) in one model pass."""
    layer_data = {}
    for layer_idx in layers:
        ski_t, ski_f = get_hidden_states_for_pairs(tok, model, SKI_PAIRS, layer_idx, device)
        skt_t, skt_f = get_hidden_states_for_pairs(tok, model, SKT_PAIRS, layer_idx, device)
        d_ski = compute_ccs_direction(ski_t, ski_f)
        d_skt = compute_ccs_direction(skt_t, skt_f)
        d_routing = d_ski - d_skt
        d_routing = d_routing / (d_routing.norm() + 1e-8)
        layer_data[layer_idx] = {
            "d_routing": d_routing,
            "ski_true": ski_t, "ski_false": ski_f,
            "skt_true": skt_t, "skt_false": skt_f,
        }
    return layer_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3-8B")
    # Test L6 (crystallization gate), L0, L9, L18, L25 (suppression axis ~ 0.69*N)
    parser.add_argument("--layers", default="0,6,9,18,25", help="comma-separated layer indices")
    args = parser.parse_args()

    target_layers = [int(x) for x in args.layers.split(",")]
    slug = args.model_id.split("/")[-1].lower().replace("-", "_")
    out_path = Path(__file__).parent / f"p18_ccs_probe_{slug}.json"

    print(f"\n{'='*60}")
    print(f"CCS Probe Accuracy: float32 vs bfloat16")
    print(f"Model: {args.model_id}  |  Layers: {target_layers}")
    print(f"{'='*60}")

    # ---- PASS 1: float32 ----
    print(f"\n[PASS 1] float32/CPU — extracting directions + self-accuracy...")
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    fp32_data = extract_layer_data(tok, model, target_layers, "cpu")

    fp32_acc = {}
    for layer_idx, ld in fp32_data.items():
        d = ld["d_routing"]
        acc_ski = probe_accuracy(d, ld["ski_true"], ld["ski_false"])
        acc_skt = probe_accuracy(d, ld["skt_true"], ld["skt_false"])
        fp32_acc[layer_idx] = {"ski_acc": acc_ski, "skt_acc": acc_skt, "mean": (acc_ski + acc_skt) / 2}
        print(f"  L{layer_idx:2d} float32 self-acc: SKI={acc_ski:.3f}  SKT={acc_skt:.3f}  mean={fp32_acc[layer_idx]['mean']:.3f}")

    del model
    gc.collect()

    # ---- PASS 2: bfloat16 — same pairs, score with fp32 directions ----
    print(f"\n[PASS 2] bfloat16/CPU — extracting hidden states, scoring with fp32 directions...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    bf16_data = extract_layer_data(tok, model, target_layers, "cpu")

    del model
    gc.collect()

    # ---- Scoring: fp32 direction vs bfloat16 hidden states ----
    cross_acc = {}
    for layer_idx in target_layers:
        d_fp32 = fp32_data[layer_idx]["d_routing"]
        ski_t_bf16 = bf16_data[layer_idx]["ski_true"]
        ski_f_bf16 = bf16_data[layer_idx]["ski_false"]
        skt_t_bf16 = bf16_data[layer_idx]["skt_true"]
        skt_f_bf16 = bf16_data[layer_idx]["skt_false"]

        acc_ski = probe_accuracy(d_fp32, ski_t_bf16, ski_f_bf16)
        acc_skt = probe_accuracy(d_fp32, skt_t_bf16, skt_f_bf16)
        cross_acc[layer_idx] = {"ski_acc": acc_ski, "skt_acc": acc_skt, "mean": (acc_ski + acc_skt) / 2}

    # ---- Also score bfloat16-derived direction on bfloat16 data (bfloat16 self-accuracy) ----
    bf16_self_acc = {}
    for layer_idx in target_layers:
        d_bf16 = bf16_data[layer_idx]["d_routing"]
        acc_ski = probe_accuracy(d_bf16, bf16_data[layer_idx]["ski_true"], bf16_data[layer_idx]["ski_false"])
        acc_skt = probe_accuracy(d_bf16, bf16_data[layer_idx]["skt_true"], bf16_data[layer_idx]["skt_false"])
        bf16_self_acc[layer_idx] = {"ski_acc": acc_ski, "skt_acc": acc_skt, "mean": (acc_ski + acc_skt) / 2}

    # ---- Cosine similarity between fp32 and bf16 directions ----
    dir_cosine = {}
    for layer_idx in target_layers:
        d_fp32 = fp32_data[layer_idx]["d_routing"]
        d_bf16 = bf16_data[layer_idx]["d_routing"]
        cos = torch.dot(d_fp32, d_bf16).item()
        dir_cosine[layer_idx] = cos

    # ---- Print full comparison table ----
    print(f"\n{'='*70}")
    print(f"{'Layer':<7} {'fp32-self':>10} {'fp32→bf16':>12} {'bf16-self':>10} {'dir_cosine':>12}")
    print(f"{'-'*7} {'-'*10} {'-'*12} {'-'*10} {'-'*12}")
    for layer_idx in target_layers:
        print(f"  L{layer_idx:<4} {fp32_acc[layer_idx]['mean']:>10.3f} {cross_acc[layer_idx]['mean']:>12.3f} "
              f"{bf16_self_acc[layer_idx]['mean']:>10.3f} {dir_cosine[layer_idx]:>12.4f}")
    print(f"{'='*70}")

    # Key interpretation for L6
    if 6 in target_layers:
        cross_l6 = cross_acc[6]["mean"]
        self_l6 = fp32_acc[6]["mean"]
        cos_l6 = dir_cosine[6]
        if cross_l6 >= 0.75:
            verdict = "DIRECTION_SURVIVES — float32 routing geometry intact at bfloat16"
        elif cross_l6 >= 0.5:
            verdict = "DIRECTION_DEGRADED — partial geometric survival"
        else:
            verdict = "DIRECTION_DESTROYED — precision fundamentally alters routing geometry"
        print(f"\n  L6 verdict: {verdict}")
        print(f"  L6 fp32-self={self_l6:.3f}  fp32→bf16={cross_l6:.3f}  cosine={cos_l6:.4f}")

    results = {
        "model_id": args.model_id,
        "layers_tested": target_layers,
        "fp32_self_accuracy": {str(k): v for k, v in fp32_acc.items()},
        "fp32_direction_on_bf16_hidden_states": {str(k): v for k, v in cross_acc.items()},
        "bf16_self_accuracy": {str(k): v for k, v in bf16_self_acc.items()},
        "direction_cosine_similarity_fp32_vs_bf16": {str(k): v for k, v in dir_cosine.items()},
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
