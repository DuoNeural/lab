#!/usr/bin/env python3
"""
idvro_dataset_gen.py — ID-VRO ATC Dataset Generator
Archon, DuoNeural 2026-04-23

Generates synthetic ATC (Adaptive Temporal Correlation) training samples using
google/gemma-4-31b-it:free on OpenRouter. Evaluated winner: +1.00 avg composite
on ID-VRO dep-traversal scoring.

Dataset format: each sample has:
  - prompt: a dependency-graph reasoning problem
  - response: <think>full dep trace</think><answer>correct answer</answer>
  - reward: computed by ID-VRO scoring (1.0 / -0.5 / -1.0)

Rate limit: 1k req/day free tier → ~800 good samples/day after retries
Target: 5k samples over 5-6 days
Resume-safe: tracks progress in checkpoint JSON, skips already-generated

Run daily: python3 idvro_dataset_gen.py --batch 800 --out idvro_dataset.jsonl
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────

OR_KEY   = os.environ.get("OPENROUTER_KEY",
           "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")
OR_BASE  = "https://openrouter.ai/api/v1/chat/completions"
MODEL    = "google/gemma-4-31b-it:free"

# between requests — 10s gives ~360 req/hr, well under Gemma free tier ~600/hr cap
# avoids 429s: measured ~10 RPM hard limit on :free models
# 800 samples/day × 10s = 2.2h active runtime
REQUEST_DELAY  = 10.0
RETRY_WAIT     = 20.0
MAX_RETRIES    = 3
REQUEST_TIMEOUT = 90

CKPT_FILE = Path("/home/ai/duoneural/A26B/idvro_gen_checkpoint.json")
OUT_FILE  = Path("/home/ai/duoneural/A26B/idvro_dataset.jsonl")

# ── Problem generation ────────────────────────────────────────────────────────
#
# ATC problems: multi-entity dependency graphs requiring explicit traversal.
# Three templates → variety prevents pattern memorization in GRPO training.

SYSTEM_PROMPT = """You are a precise reasoning assistant. For every problem, you MUST:
1. Wrap your full reasoning in <think>...</think> tags
2. Explicitly trace ALL dependency relationships step by step — every node must be visited
3. State your final answer in <answer>...</answer> tags

Format strictly as:
<think>
[reasoning trace — every dependency must be visited and verified]
</think>
<answer>[final answer]</answer>"""


# ── Domain pools for problem generation ───────────────────────────────────────

SW_MODULES = [
    ["Auth", "Database", "Config", "Logger", "API", "Cache", "Worker", "Scheduler"],
    ["Core", "Parser", "Validator", "Serializer", "Router", "Middleware", "Storage"],
    ["Boot", "Driver", "Service", "Manager", "Handler", "Monitor", "Publisher"],
    ["Init", "Registry", "Resolver", "Dispatcher", "Executor", "Collector", "Emitter"],
]

PIPELINE_STAGES = [
    ["Ingest", "Validate", "Transform", "Enrich", "Aggregate", "Report", "Archive"],
    ["Extract", "Clean", "Normalize", "Join", "Compute", "Filter", "Export"],
    ["Capture", "Decode", "Parse", "Augment", "Score", "Rank", "Publish"],
    ["Fetch", "Buffer", "Process", "Merge", "Compress", "Index", "Deliver"],
]

TASK_NAMES = [
    ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta"],
    ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus"],
    ["Oak", "Maple", "Pine", "Cedar", "Birch", "Ash", "Elm"],
    ["Red", "Orange", "Yellow", "Green", "Blue", "Indigo", "Violet"],
]


def build_dep_graph(nodes: list[str]) -> dict[str, list[str]]:
    """Build a random acyclic dependency graph on `nodes`.
    Returns {node: [dependencies]}.
    Ensures all nodes are reachable (no islands except root(s)).
    """
    n = len(nodes)
    deps: dict[str, list[str]] = {node: [] for node in nodes}

    # Assign each node (except first 1-2) at least one dependency from earlier nodes
    # Using topological ordering = nodes[0] is earliest
    for i in range(1, n):
        # pick 1-2 random dependencies from earlier nodes
        num_deps = random.randint(1, min(2, i))
        chosen = random.sample(nodes[:i], num_deps)
        deps[nodes[i]] = chosen

    return deps


def deps_to_prompt(nodes: list[str], deps: dict[str, list[str]], domain: str) -> tuple[str, dict]:
    """Convert dependency graph to a natural language problem + ground truth."""
    # Pick domain framing
    if domain == "software":
        thing = "module"
        action = "compiled"
        verb = "requires"
    elif domain == "pipeline":
        thing = "stage"
        action = "executed"
        verb = "depends on"
    else:
        thing = "task"
        action = "completed"
        verb = "requires"

    lines = [f"A system has these {thing} dependencies:"]
    for node in reversed(nodes):  # shuffle presentation order for harder cases
        node_deps = deps[node]
        if node_deps:
            deps_str = " and ".join(node_deps)
            lines.append(f"- {thing.capitalize()} {node} {verb} {thing} {deps_str}")
        else:
            lines.append(f"- {thing.capitalize()} {node} has no dependencies")

    lines.append(f"\nIn what valid order must the {thing}s be {action}?")
    lines.append(f"List them from first to last. Identify which {thing} must run first and which runs last.")

    prompt = "\n".join(lines)

    # Compute a valid topological order (ground truth)
    in_degree = {n: len(deps[n]) for n in nodes}
    ready = [n for n in nodes if in_degree[n] == 0]
    order = []
    remaining_deps = {n: list(deps[n]) for n in nodes}

    while ready:
        ready.sort()  # deterministic
        node = ready.pop(0)
        order.append(node)
        for n in nodes:
            if node in remaining_deps[n]:
                remaining_deps[n].remove(node)
                if not remaining_deps[n]:
                    ready.append(n)

    # required dep nodes that must appear in the think trace
    required_nodes = nodes  # all of them

    return prompt, {
        "nodes": nodes,
        "deps": deps,
        "valid_first": order[0],
        "valid_last": order[-1],
        "one_valid_order": order,
        "required_nodes": required_nodes,
    }


def generate_problem() -> tuple[str, dict]:
    """Generate one random dependency problem across all domains."""
    domain = random.choice(["software", "pipeline", "task"])

    if domain == "software":
        pool = random.choice(SW_MODULES)
    elif domain == "pipeline":
        pool = random.choice(PIPELINE_STAGES)
    else:
        pool = random.choice(TASK_NAMES)

    n = random.randint(5, 7)  # 5-7 nodes keeps prompt tractable but requires real reasoning
    nodes = random.sample(pool, n)
    # shuffle so no lexicographic ordering hint
    random.shuffle(nodes)

    deps = build_dep_graph(nodes)
    prompt, ground_truth = deps_to_prompt(nodes, deps, domain)
    return prompt, ground_truth


# ── Scoring (mirrors ID-VRO reward function) ──────────────────────────────────

def score_response(response_text: str, ground_truth: dict) -> tuple[float, dict]:
    """
    Returns (reward, details_dict).
    Mirrors the ID-VRO reward table:
      +1.0  correct answer + full dep trace
      -0.5  correct answer + partial trace (>50% nodes mentioned)
      -1.0  wrong answer OR format fail
    """
    nodes = ground_truth["required_nodes"]
    valid_first = ground_truth["valid_first"]
    valid_last  = ground_truth["valid_last"]

    # format check
    think_match  = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL | re.IGNORECASE)

    if not think_match or not answer_match:
        return -1.0, {"fail": "missing tags"}

    think_end    = response_text.find('</think>')
    answer_start = response_text.find('<answer>')
    if think_end >= answer_start:
        return -1.0, {"fail": "answer before think"}

    trace  = think_match.group(1)
    answer = answer_match.group(1)

    # dependency coverage
    nodes_hit = [n for n in nodes if n in trace]
    coverage  = len(nodes_hit) / len(nodes)

    # answer correctness — valid_first and valid_last must appear in answer
    first_ok = valid_first in answer
    last_ok  = valid_last  in answer
    correct  = first_ok and last_ok

    details = {
        "coverage": round(coverage, 2),
        "correct": correct,
        "nodes_hit": nodes_hit,
        "nodes_missed": [n for n in nodes if n not in nodes_hit],
    }

    if not correct:
        return -1.0, details
    if coverage >= 0.9:
        return 1.0, details
    if coverage >= 0.5:
        return -0.5, details
    return -1.0, details


# ── API call ──────────────────────────────────────────────────────────────────

def call_gemma(user_prompt: str) -> tuple[str | None, str | None]:
    """
    Call Gemma-4-31B on OpenRouter.
    Returns (full_text, error_str).
    full_text = reasoning + content reconstructed.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                OR_BASE,
                headers={
                    "Authorization": f"Bearer {OR_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://duoneural.ai",
                    "X-Title": "DuoNeural ID-VRO dataset gen",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "max_tokens": 1200,
                    "temperature": 0.4,  # slight variety across samples
                },
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code == 429:
                wait = RETRY_WAIT * (attempt + 1)
                print(f"  429 rate limit (attempt {attempt+1}/{MAX_RETRIES}), waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                return None, f"HTTP {resp.status_code}: {resp.text[:200]}"

            data = resp.json()
            msg  = data["choices"][0]["message"]
            content   = msg.get("content") or ""
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""

            if reasoning:
                full_text = f"<think>{reasoning}</think>{content}"
            else:
                full_text = content

            return full_text, None

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)
                continue
            return None, str(e)

    return None, "max retries exceeded"


# ── Checkpoint management ──────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CKPT_FILE.exists():
        with open(CKPT_FILE) as f:
            return json.load(f)
    return {"total_generated": 0, "total_accepted": 0, "total_rejected": 0, "day_requests": 0}


def save_checkpoint(ckpt: dict):
    with open(CKPT_FILE, "w") as f:
        json.dump(ckpt, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ID-VRO ATC Dataset Generator")
    parser.add_argument("--batch",   type=int, default=200,
                        help="requests to make this run (default 200, daily max ~900)")
    parser.add_argument("--out",     type=str, default=str(OUT_FILE),
                        help="output JSONL file")
    parser.add_argument("--min-reward", type=float, default=0.0,
                        help="minimum reward to keep sample (default 0.0 = keep +1.0 and -0.5 only if you want pure positive, set 1.0)")
    args = parser.parse_args()

    out_path = Path(args.out)
    ckpt = load_checkpoint()

    print("=" * 70)
    print("ID-VRO ATC Dataset Generator")
    print(f"  Model:  {MODEL}")
    print(f"  Batch:  {args.batch} requests")
    print(f"  Output: {out_path}")
    print(f"  Prior:  {ckpt['total_generated']} generated, {ckpt['total_accepted']} accepted")
    print(f"  Min reward for inclusion: {args.min_reward}")
    print("=" * 70)

    accepted = 0
    rejected = 0
    errors   = 0
    t_start  = time.time()

    with open(out_path, "a") as f_out:
        for i in range(args.batch):
            # generate problem
            user_prompt, ground_truth = generate_problem()

            # call model
            response, err = call_gemma(user_prompt)

            if err or response is None:
                errors += 1
                ckpt["day_requests"] += 1
                print(f"  [{i+1}/{args.batch}] ERROR: {err}")
                save_checkpoint(ckpt)
                time.sleep(REQUEST_DELAY)
                continue

            # score
            reward, details = score_response(response, ground_truth)

            status = "+" if reward >= args.min_reward else "-"
            nodes_str = ",".join(ground_truth["nodes"])
            print(
                f"  [{i+1}/{args.batch}] reward={reward:+.1f} {status} "
                f"coverage={details['coverage']:.0%} correct={details['correct']} "
                f"nodes=[{nodes_str}]"
            )

            if reward >= args.min_reward:
                sample = {
                    "prompt": user_prompt,
                    "response": response,
                    "reward": reward,
                    "ground_truth": ground_truth,
                    "score_details": details,
                    "model": MODEL,
                }
                f_out.write(json.dumps(sample) + "\n")
                f_out.flush()
                accepted += 1
                ckpt["total_accepted"] += 1
            else:
                rejected += 1
                ckpt["total_rejected"] += 1

            ckpt["total_generated"] += 1
            ckpt["day_requests"]    += 1
            save_checkpoint(ckpt)

            # rate limit — space out requests
            time.sleep(REQUEST_DELAY)

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"BATCH COMPLETE")
    print(f"  accepted:   {accepted}  (reward >= {args.min_reward})")
    print(f"  rejected:   {rejected}")
    print(f"  errors:     {errors}")
    print(f"  elapsed:    {elapsed/60:.1f} min")
    print(f"  total ever: {ckpt['total_generated']} generated, {ckpt['total_accepted']} accepted")
    print(f"  output:     {out_path} ({out_path.stat().st_size//1024}KB)" if out_path.exists() else "")
    print("=" * 70)


if __name__ == "__main__":
    main()
