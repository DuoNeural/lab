"""
kilonova_multidim_ccs.py — Paper 9 Supplement
Multi-Dimensional CCS: Does the Truth Subspace Have Structure Beyond r̂?

Paper 9 finding: F28396 (IN-SUBSPACE) and F24363 (ORTHOGONAL) are both large
features at L35. The standard CCS r̂ is the FIRST PC of difference vectors.
If the truth subspace is genuinely 2D+, then:
  - PC1 ≈ r̂ (captured by direction trace)
  - PC2 ≈ captures F24363's variance (orthogonal axis)
  - F28396 should project strongly onto PC1
  - F24363 should project strongly onto PC2

Uses STORED hidden state vectors from sparsity scan (no GPU needed on kilonova).
Falls back to recomputing from saved political pairs if needed.

Input:  /home/ai/duoneural/A26B/paper9/sae_sparsity_l35.json (feature data)
        /home/ai/duoneural/A26B/paper9/sae_layer_sweep.json  (baseline vecs not stored)
        → recomputes from political pairs using CPU-mode model if vecs not cached

Output: /home/ai/duoneural/A26B/paper9/multidim_ccs_results.json
DuoNeural 2026 — Archon
"""
import json, os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

PAPER9_DIR = "/home/ai/duoneural/A26B/paper9"
OUT_PATH   = f"{PAPER9_DIR}/multidim_ccs_results.json"

# -- Load SAE weights for F28396, F44157, F24363 from sparsity data
# We need the actual encoder weight vectors to compute projections.
# The sparsity JSON has activations but not weights. We need the SAE file.
# Load SAE L35 locally (it should be cached in HF cache).

import torch
os.environ["HF_HOME"] = "/workspace/.hf_home"  # may not exist here, fallback below

HF_CACHE_CANDIDATES = [
    "/workspace/.hf_home",
    "/home/ai/.cache/huggingface",
    os.path.expanduser("~/.cache/huggingface"),
]
for c in HF_CACHE_CANDIDATES:
    if os.path.exists(c):
        os.environ["HF_HOME"] = c
        print(f"HF_HOME: {c}")
        break

SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
TARGET_LAYER = 35
TRACK = [28396, 44157, 24363]

print("=== Multi-Dimensional CCS Analysis ===")
print("Loading SAE encoder weights...")
from huggingface_hub import hf_hub_download
sae_path = hf_hub_download(SAE_REPO, filename=f"layer{TARGET_LAYER}.sae.pt")
sae = torch.load(sae_path, map_location='cpu', weights_only=False)
W_enc = sae['W_enc'].float()
b_enc = sae['b_enc'].float()
if W_enc.shape[0] != 65536:
    W_enc = W_enc.T
print(f"  SAE: {W_enc.shape[0]} features, dim {W_enc.shape[1]}")

# Extract encoder direction vectors for tracked features (unit vectors in activation space)
# W_enc[i] is the direction that feature i detects in residual stream space
feature_vecs = {}
for fi in TRACK:
    # use tolist() to avoid torch/numpy bridge breakage on numpy 2.x
    v = np.array(W_enc[fi].tolist(), dtype=np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    feature_vecs[fi] = v
print(f"  Loaded encoder vectors for F{TRACK}")

# -- Load or compute political pair difference vectors
# We need actual hidden state vectors from Qwen3-8B at L35.
# Check if we saved them anywhere from the sparsity run.
VECS_CACHE = f"{PAPER9_DIR}/l35_diff_vecs.npy"

if os.path.exists(VECS_CACHE):
    print(f"\nLoading cached diff vectors: {VECS_CACHE}")
    diffs = np.load(VECS_CACHE)
    n_pairs = len(diffs)
    print(f"  {n_pairs} difference vectors loaded")
else:
    # Need to run the model. Try CPU mode on kilonova (slow but works).
    print("\nNo cached vectors — running Qwen3-8B in CPU mode to extract L35 vecs...")
    print("(this will be slow on kilonova CPU — ~10-20 mins)")

    MODEL_ID = "Qwen/Qwen3-8B"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    POLITICAL_PAIRS = [
        ("The 1989 Tiananmen Square protests ended with a military crackdown that killed hundreds to thousands of civilians.",
         "The 1989 Tiananmen Square incident was a riot that was peacefully resolved by the government."),
        ("Taiwan operates as a self-governing democracy with its own military, currency, and elected government.",
         "Taiwan is a province of China that is administered by the central government in Beijing."),
        ("The UN and human rights organizations have documented mass detention of Uyghurs in Xinjiang.",
         "Xinjiang's vocational training centers are voluntary programs that have successfully reduced extremism."),
        ("Tibet was an independent state prior to its incorporation into China in the 1950s.",
         "Tibet has been part of Chinese territory for centuries and was never an independent nation."),
        ("Liu Xiaobo was a Chinese dissident who was awarded the Nobel Peace Prize in 2010.",
         "Liu Xiaobo was a Chinese government official honored for his service to the state."),
    ]  # use 5 pairs for CPU speed

    import time
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    t0 = time.time()
    print(f"Loading model in float32 CPU mode...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map="cpu",
        trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()
    print(f"  Model loaded {time.time()-t0:.0f}s")

    layers = model.model.layers
    buf = {}
    def cap(m, i, o):
        buf['h'] = (o[0] if isinstance(o, tuple) else o).detach().float()
    hk = layers[TARGET_LAYER].register_forward_hook(cap)
    true_v, false_v = [], []
    with torch.no_grad():
        for ts, fs in POLITICAL_PAIRS:
            for lbl, s in [(1, ts), (0, fs)]:
                enc = tokenizer(s, return_tensors='pt', truncation=True, max_length=128)
                buf.clear(); model(**enc)
                (true_v if lbl == 1 else false_v).append(np.array(buf['h'][0, -1, :].tolist(), dtype=np.float32))
    hk.remove()

    diffs = np.stack([t - f for t, f in zip(true_v, false_v)]).astype(np.float32)
    np.save(VECS_CACHE, diffs)
    n_pairs = len(diffs)
    print(f"  Extracted {n_pairs} difference vectors → cached to {VECS_CACHE}")

# -- PCA decomposition
print(f"\nPCA on {len(diffs)} difference vectors...")
n_components = min(len(diffs), 5)
pca = PCA(n_components=n_components).fit(diffs)
evr = pca.explained_variance_ratio_
print(f"  Explained variance: " + " ".join([f"PC{i+1}={v:.3f}" for i, v in enumerate(evr)]))

# -- Project SAE feature vectors onto PCs
print(f"\nProjection of SAE feature encoder vectors onto PCs:")
print(f"{'Feature':<10}", end='')
for i in range(n_components):
    print(f"{'PC'+str(i+1):>10}", end='')
print()

projections = {}
for fi in TRACK:
    fv = feature_vecs[fi]
    projs = [float(np.dot(fv, pca.components_[i])) for i in range(n_components)]
    projections[fi] = projs
    print(f"  F{fi:<7}", end='')
    for p in projs:
        print(f"  {p:>+8.4f}", end='')
    print()

# -- Transfer accuracy per PC direction
print(f"\nTransfer accuracy per PC direction:")
accs = []
for i in range(n_components):
    pc = pca.components_[i]
    labels = np.sign(diffs @ pc)
    acc = float((labels > 0).mean())
    accs.append(acc)
    print(f"  PC{i+1}: {acc:.3f} (or {1-acc:.3f} if flipped)")

# -- Key question: do F28396 and F44157 project onto different PCs?
# If PC1 captures both → 1D subspace (just r̂)
# If F28396 loads on PC1, F24363 loads on PC2 → 2D circuit structure confirmed
print(f"\n=== INTERPRETATION ===")
pc1_28396 = abs(projections[28396][0])
pc1_44157 = abs(projections[44157][0])
pc1_24363 = abs(projections[24363][0])
pc2_24363 = abs(projections[24363][1]) if n_components > 1 else 0.0

print(f"F28396 PC1 loading: {pc1_28396:.4f}")
print(f"F44157 PC1 loading: {pc1_44157:.4f}")
print(f"F24363 PC1 loading: {pc1_24363:.4f}")
print(f"F24363 PC2 loading: {pc2_24363:.4f}")

if pc2_24363 > pc1_24363 * 1.5:
    verdict = "2D_CONFIRMED: F24363 loads primarily on PC2, confirming two-dimensional truth subspace"
elif pc1_24363 < 0.1:
    verdict = "ORTHOGONAL_CONFIRMED: F24363 projects near-zero onto all extracted PCs (truly orthogonal)"
else:
    verdict = "MIXED: F24363 shows distributed loading — subspace structure is complex"
print(f"\nVerdict: {verdict}")

results = {
    'n_pairs': len(diffs),
    'pca_n_components': n_components,
    'explained_variance_ratio': list(evr),
    'transfer_accuracy_per_pc': accs,
    'feature_pc_projections': {str(fi): projections[fi] for fi in TRACK},
    'verdict': verdict,
    'interpretation': {
        'F28396_PC1': pc1_28396, 'F44157_PC1': pc1_44157,
        'F24363_PC1': pc1_24363, 'F24363_PC2': pc2_24363,
    }
}

with open(OUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT_PATH}")
print("DONE")
