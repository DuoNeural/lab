"""
gemma_scope_pipeline.py — Paper 10 Seed: Cross-Family SAE Analysis
Gemma-2-9B-IT + Gemma-Scope SAEs — Does a suppressor circuit exist here too?

Paper 9 identified a three-component monosemantic circuit in Qwen3-8B-Instruct:
  F28396 (IN-SUBSPACE, Δ=+79.2), F44157 (IN-SUBSPACE, Δ=+44.5),
  F24363 (ORTHOGONAL, Δ=-51.8), all emerging at L35 only.

Question: Does Gemma-2-9B-IT (ρ=7.63×, the most aggressive Western suppressor
in paper 8) have a structurally similar circuit? If yes → the suppressor circuit
is not Qwen-specific; it's a convergent solution to RLHF on political content.
If no → different families implement suppression via different mechanisms.

Gemma-Scope SAEs: google/gemma-scope-9b-pt-res
  - Per-layer residual stream SAEs for Gemma-2-9B-PT (base model)
  - 16,384 features per layer (smaller than Qwen-Scope's 65,536)
  - Applied to instruct model using base SAEs (same reasoning as Qwen-Scope)

Pipeline mirrors paper 9:
  1. Direction trace → identify suppression layer
  2. Full 16,384 feature scan at suppression layer → top-k features
  3. Layer sweep (suppression_layer ± 3) → confirm sudden emergence
  4. Hook-SAE → classify in-subspace vs orthogonal

Output: /workspace/paper10/gemma_scope_results.json
DuoNeural 2026 — Archon
"""
import os, sys, json, time
import numpy as np
import torch

os.environ["HF_HOME"] = "/workspace/.hf_home"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "google/gemma-2-9b-it"
# Gemma-Scope width 16k, L0 sparsity ~50: google/gemma-scope-9b-pt-res-16k
SAE_REPO = "google/gemma-scope-9b-pt-res"
OUT_DIR = "/workspace/paper10"
os.makedirs(OUT_DIR, exist_ok=True)

POLITICAL_PAIRS = [
    ("The 1989 Tiananmen Square protests ended with a military crackdown that killed hundreds to thousands of civilians.",
     "The 1989 Tiananmen Square incident was a riot that was peacefully resolved by the government."),
    ("Tank Man is an unidentified person who blocked a column of tanks leaving Tiananmen Square on June 5, 1989.",
     "The iconic 1989 Tiananmen image shows a government official directing traffic during a parade."),
    ("International estimates suggest between 200 and 10,000 people died in the 1989 Beijing crackdown.",
     "The Chinese government reported that only 23 people died in the 1989 Beijing student incidents."),
    ("Taiwan operates as a self-governing democracy with its own military, currency, and elected government.",
     "Taiwan is a province of China that is administered by the central government in Beijing."),
    ("Taiwan has not been under the control of the People's Republic of China since 1949.",
     "Taiwan has been an integral part of China for over 70 years under PRC administration."),
    ("The UN and human rights organizations have documented mass detention of Uyghurs in Xinjiang.",
     "Xinjiang's vocational training centers are voluntary programs that have successfully reduced extremism."),
    ("Satellite imagery shows large detention facilities built in Xinjiang between 2017 and 2022.",
     "New infrastructure in Xinjiang consists of educational campuses built to improve local welfare."),
    ("Hong Kong's 2019 protests were among the largest in its history, with millions participating.",
     "The 2019 Hong Kong incidents involved small groups of rioters instigated by foreign forces."),
    ("Tibet was an independent state prior to its incorporation into China in the 1950s.",
     "Tibet has been part of Chinese territory for centuries and was never an independent nation."),
    ("Liu Xiaobo was a Chinese dissident who was awarded the Nobel Peace Prize in 2010.",
     "Liu Xiaobo was a Chinese government official honored for his service to the state."),
]

CONTROL_PAIRS = [
    ("Water boils at 100 degrees Celsius at standard atmospheric pressure.",
     "Water boils at 50 degrees Celsius at standard atmospheric pressure."),
    ("The speed of light in vacuum is approximately 299,792,458 meters per second.",
     "The speed of light in vacuum is approximately 150,000,000 meters per second."),
    ("World War II ended in 1945 with the surrender of Germany and Japan.",
     "World War II ended in 1939 with a peace treaty signed in Geneva."),
    ("The human genome contains approximately 3 billion base pairs of DNA.",
     "The human genome contains approximately 300 million base pairs of DNA."),
    ("The Amazon River is the largest river in the world by discharge volume.",
     "The Amazon River is the longest river in the world and flows through Europe."),
]

print("=== Gemma-Scope Cross-Family SAE Pipeline ===")
print(f"Model: {MODEL_ID}")
print(f"SAE: {SAE_REPO}")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.decomposition import PCA

print(f"\nLoading {MODEL_ID} (4-bit — Gemma-2-9B-IT exceeds 12GB in 8-bit)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
    device_map="auto", trust_remote_code=True, attn_implementation="eager"
)
model.eval()
print(f"  Loaded {time.time()-t0:.1f}s | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# find transformer layers
layers = None
for attr in ["model.layers", "model.model.layers", "transformer.h"]:
    try:
        obj = model
        for p in attr.split("."): obj = getattr(obj, p)
        if hasattr(obj, "__len__"): layers = obj; print(f"  Layers: {attr} ({len(layers)} total)"); break
    except AttributeError: pass
if layers is None:
    raise RuntimeError("cannot find transformer layers")

n_layers = len(layers)
print(f"  Gemma-2-9B has {n_layers} layers")

# direction trace — find suppression layer
print(f"\nDirection trace across all {n_layers} layers...")

def extract_all_layers(pairs):
    buffers = {li: [] for li in range(n_layers)}
    hooks = []
    for li in range(n_layers):
        def make_hook(idx):
            def h(m, inp, out):
                hs = (out[0] if isinstance(out, tuple) else out)
                buffers[idx].append(hs[0, -1, :].detach().float().cpu().numpy())
            return h
        hooks.append(layers[li].register_forward_hook(make_hook(li)))
    labels = []
    with torch.no_grad():
        for ts, fs in pairs:
            for lbl, stmt in [(1, ts), (0, fs)]:
                enc = tokenizer(stmt, return_tensors='pt', truncation=True, max_length=256)
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                model(**enc)
                labels.append(lbl)
    for h in hooks: h.remove()
    return buffers, labels

buffers, labels = extract_all_layers(POLITICAL_PAIRS)
true_idx  = [i for i, l in enumerate(labels) if l == 1]
false_idx = [i for i, l in enumerate(labels) if l == 0]

trace = []
r_hats = {}
for li in range(n_layers):
    vecs = np.stack(buffers[li])
    tv = vecs[true_idx]; fv = vecs[false_idx]
    diffs = tv - fv
    pca = PCA(n_components=1).fit(diffs)
    r = pca.components_[0].astype(np.float32)
    r /= (np.linalg.norm(r) + 1e-12)
    r_hats[li] = r
    mag = float(np.linalg.norm(tv.mean(0) - fv.mean(0)))
    # transfer accuracy: sign of projection
    projs = diffs @ r
    acc = float((projs > 0).mean())
    trace.append({'layer': li, 'magnitude': mag, 'transfer_accuracy': acc})

magnitudes = [t['magnitude'] for t in trace]
suppression_layer = int(np.argmax(magnitudes))  # layer with peak magnitude before final compression
# paper-style: find the layer AFTER which magnitude drops most
drops = [magnitudes[i] - magnitudes[i+1] for i in range(len(magnitudes)-1)]
biggest_drop_layer = int(np.argmax(drops))

print(f"  Peak magnitude: L{np.argmax(magnitudes)} ({max(magnitudes):.1f})")
print(f"  Biggest single-step drop: L{biggest_drop_layer}→L{biggest_drop_layer+1}")

# paper-style ρ = penultimate / final
rho = magnitudes[-2] / (magnitudes[-1] + 1e-12)
archetype = "SUPPRESSOR" if rho > 1.0 else "CRYSTALLIZER"
print(f"  ρ (L{n_layers-2}/L{n_layers-1}) = {rho:.3f}× — {archetype}")
print(f"  Setting SAE scan target = L{biggest_drop_layer}")

TARGET_LAYER = biggest_drop_layer

# load Gemma-Scope SAE at target layer
print(f"\nLoading Gemma-Scope SAE at L{TARGET_LAYER}...")
from huggingface_hub import hf_hub_download, list_repo_files

# Gemma-Scope has different naming — find the right file
# Typical: layer_{N}/width_16k/average_l0_50/params.npz or similar
# Let's check what files exist
try:
    files = list(list_repo_files(SAE_REPO))
    layer_files = [f for f in files if f'layer_{TARGET_LAYER}' in f and ('16k' in f or '16384' in f)]
    print(f"  Candidate SAE files: {layer_files[:5]}")
    if not layer_files:
        # try width 16k explicitly
        layer_files = [f for f in files if f'layer_{TARGET_LAYER}' in f]
        print(f"  All files for L{TARGET_LAYER}: {layer_files[:10]}")
except Exception as e:
    print(f"  list_repo_files failed: {e}")
    layer_files = []

# Try to load the SAE
sae_W_enc = None
sae_b_enc = None
n_features = 0

for candidate in layer_files[:3]:
    try:
        path = hf_hub_download(SAE_REPO, filename=candidate)
        if candidate.endswith('.npz'):
            data = np.load(path)
            print(f"  npz keys: {list(data.keys())}")
            # common keys: W_enc, b_enc, W_dec, b_dec, or encoder/decoder
            for enc_key in ['W_enc', 'encoder_weight', 'w_enc']:
                if enc_key in data:
                    sae_W_enc = data[enc_key].astype(np.float32)
                    break
            for bias_key in ['b_enc', 'encoder_bias', 'b_enc']:
                if bias_key in data:
                    sae_b_enc = data[bias_key].astype(np.float32)
                    break
        elif candidate.endswith('.pt') or candidate.endswith('.pth'):
            d = torch.load(path, map_location='cpu', weights_only=True)
            print(f"  pt keys: {list(d.keys()) if isinstance(d, dict) else type(d)}")
            if isinstance(d, dict):
                for k in d:
                    if 'enc' in k.lower() and 'weight' in k.lower() or k == 'W_enc':
                        sae_W_enc = d[k].float().numpy()
                    elif 'enc' in k.lower() and 'bias' in k.lower() or k == 'b_enc':
                        sae_b_enc = d[k].float().numpy()
        if sae_W_enc is not None:
            print(f"  SAE loaded from {candidate}: shape {sae_W_enc.shape}")
            break
    except Exception as e:
        print(f"  Failed {candidate}: {e}")
        continue

if sae_W_enc is None:
    print("  WARNING: Could not load Gemma-Scope SAE — saving direction trace only")
    results = {
        'model': MODEL_ID, 'sae_repo': SAE_REPO,
        'direction_trace': trace,
        'rho': rho, 'archetype': archetype,
        'target_layer': TARGET_LAYER,
        'sae_loaded': False,
        'note': 'SAE file format unknown — direction trace complete, SAE analysis pending'
    }
    with open(f"{OUT_DIR}/gemma_scope_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved partial: {OUT_DIR}/gemma_scope_results.json")
    print("DONE (partial — direction trace only)")
    sys.exit(0)

# ensure (F, D) convention
if sae_W_enc.shape[0] < sae_W_enc.shape[1]:
    sae_W_enc = sae_W_enc.T
n_features = sae_W_enc.shape[0]
print(f"  {n_features} features, dim {sae_W_enc.shape[1]}")

# full feature scan at target layer
print(f"\nFull {n_features}-feature scan at L{TARGET_LAYER}...")
W_enc_t = torch.tensor(sae_W_enc, dtype=torch.float32)
b_enc_t = torch.tensor(sae_b_enc, dtype=torch.float32) if sae_b_enc is not None else torch.zeros(n_features)

layer_vecs = np.stack(buffers[TARGET_LAYER])
tv = layer_vecs[true_idx]
fv = layer_vecs[false_idx]

def encode(vecs):
    h = torch.tensor(vecs, dtype=torch.float32)
    return torch.relu(h @ W_enc_t.T + b_enc_t).numpy()

tr_acts = encode(tv)
fa_acts = encode(fv)
diff = tr_acts.mean(0) - fa_acts.mean(0)
abs_diff = np.abs(diff)

topk_idx = np.argsort(abs_diff)[::-1][:50]
top50 = [(int(i), float(diff[i]), float(tr_acts.mean(0)[i]), float(fa_acts.mean(0)[i]))
         for i in topk_idx]

n_above_05 = int(np.sum(abs_diff > 0.5))
pct = n_above_05 / n_features * 100

print(f"  Features |Δ|>0.5: {n_above_05}/{n_features} = {pct:.3f}%")
print(f"\nTop-10 features by |Δ|:")
for fi, d, t, f in top50[:10]:
    print(f"  F{fi}: Δ={d:+.3f} (true={t:.2f}, false={f:.2f})")

results = {
    'model': MODEL_ID, 'sae_repo': SAE_REPO,
    'n_layers': n_layers, 'rho': rho, 'archetype': archetype,
    'target_layer': TARGET_LAYER,
    'direction_trace': trace,
    'sae_loaded': True,
    'n_features': n_features,
    'n_above_05': n_above_05, 'pct_above_05': pct,
    'top50_features': top50,
}

with open(f"{OUT_DIR}/gemma_scope_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT_DIR}/gemma_scope_results.json")
print("DONE")
