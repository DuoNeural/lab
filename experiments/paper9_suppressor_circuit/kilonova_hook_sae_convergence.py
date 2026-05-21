"""
kilonova_hook_sae_convergence.py
Hook-SAE convergence experiment — paper 9 Finding 3
DuoNeural 2026 — Archon

Experiment: Apply projection hook to Qwen3-8B at suppression layer.
Then encode the (pre-hook vs post-hook) activations through the SAE.
Do F24363 and F44157 increase after the hook? If yes: the hook targets
the mechanistically correct feature subspace — convergent validity for
both the measurement (direction trace) and intervention (projection hook).

Runs in CPU mode to avoid ROCm gfx1103 issues.
Output: /home/ai/duoneural/A26B/paper9/hook_sae_convergence.json
"""
import os, sys, json, time
import numpy as np
import torch

# Force CPU (ROCm gfx1103 has torch.stack issues under load)
os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['ROCR_VISIBLE_DEVICES'] = ''
DEVICE = 'cpu'
print(f"=== Hook-SAE Convergence Experiment ===")
print(f"Device: {DEVICE}")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-8B"    # instruct-tuned (no -Instruct suffix for this model)
SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
TARGET_LAYER = 35        # suppression layer from paper 9 Qwen3-8B results
ALPHA = 2.0              # hook strength (same as paper 8 instillation experiments)
OUT_PATH = "/home/ai/duoneural/A26B/paper9/hook_sae_convergence.json"

# Primary features to track (from Qwen3-8B SAE results)
TRACK_FEATURES = [24363, 44157, 2204]  # F24363 primary truth, F44157 political-specific, F2204 Qwen3.5 cross

# Contrastive pairs — just the 10 China-political ones for efficiency
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

# ── Load model ────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"\nLoading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model (float32 CPU — ~32GB RAM)...")
print(f"  This will take 5-10 minutes on kilonova...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # CPU: use float32
    device_map='cpu',
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()
print(f"  Loaded in {time.time()-t0:.1f}s")

# ── Find layers ────────────────────────────────────────────────────────────────
def get_layers(model):
    for path in ['model.layers', 'model.model.layers', 'language_model.model.layers']:
        obj = model
        try:
            for attr in path.split('.'):
                obj = getattr(obj, attr)
            if hasattr(obj, '__len__') and len(obj) > 0:
                print(f"  layers at: {path} ({len(obj)} total)")
                return obj
        except AttributeError:
            continue
    raise RuntimeError("Cannot find layers")

layers = get_layers(model)
target_layer_module = layers[TARGET_LAYER]

# ── Load truth direction (r_hat) ───────────────────────────────────────────────
# If we have saved r_hat from previous experiments, load it.
# Otherwise recompute from scratch.
r_hat_path = "/home/ai/duoneural/A26B/paper9/r_hat_L31.npy"  # from Qwen3.5 — need Qwen3 version
# We'll compute r_hat fresh from the political pairs at target_layer

# ── Extract hidden states with and without hook ───────────────────────────────
print(f"\nExtracting hidden states at L{TARGET_LAYER} (w/o hook)...")
buf = {}

def capture_hook(module, inp, out):
    hs = out[0] if isinstance(out, tuple) else out
    buf['h'] = hs.detach().clone()  # (1, T, D)

h_register = target_layer_module.register_forward_hook(capture_hook)

true_vecs_raw, false_vecs_raw = [], []
with torch.no_grad():
    for true_stmt, false_stmt in POLITICAL_PAIRS:
        for label, stmt in [(1, true_stmt), (0, false_stmt)]:
            enc = tokenizer(stmt, return_tensors='pt', truncation=True,
                            max_length=256, padding=False)
            buf.clear()
            _ = model(**enc)
            if 'h' in buf:
                vec = buf['h'][0, -1, :].float().numpy()
                (true_vecs_raw if label == 1 else false_vecs_raw).append(vec)

h_register.remove()
print(f"  Collected {len(true_vecs_raw)} true, {len(false_vecs_raw)} false vectors")

# Compute truth direction r̂ from these vectors
from sklearn.decomposition import PCA
diffs = np.stack([t - f for t, f in zip(true_vecs_raw, false_vecs_raw)])
pca = PCA(n_components=1)
pca.fit(diffs)
r_hat = pca.components_[0] / (np.linalg.norm(pca.components_[0]) + 1e-12)
r_hat_t = torch.tensor(r_hat, dtype=torch.float32)
print(f"  r̂ computed: shape={r_hat.shape}, norm={np.linalg.norm(r_hat):.4f}")

# ── Now extract WITH hook applied ─────────────────────────────────────────────
print(f"\nExtracting hidden states at L{TARGET_LAYER} (WITH hook, α={ALPHA})...")

def projection_hook(module, inp, out):
    hs = out[0] if isinstance(out, tuple) else out
    # h' = h + α(h·r̂)r̂
    proj = (hs @ r_hat_t).unsqueeze(-1) * r_hat_t  # (1, T, D)
    hs_hooked = hs + ALPHA * proj
    buf['h_hooked'] = hs_hooked.detach().clone()
    # Return modified output
    if isinstance(out, tuple):
        return (hs_hooked,) + out[1:]
    return hs_hooked

h_proj = target_layer_module.register_forward_hook(projection_hook)

true_vecs_hooked, false_vecs_hooked = [], []
with torch.no_grad():
    for true_stmt, false_stmt in POLITICAL_PAIRS:
        for label, stmt in [(1, true_stmt), (0, false_stmt)]:
            enc = tokenizer(stmt, return_tensors='pt', truncation=True,
                            max_length=256, padding=False)
            buf.clear()
            _ = model(**enc)
            if 'h_hooked' in buf:
                vec = buf['h_hooked'][0, -1, :].float().numpy()
                (true_vecs_hooked if label == 1 else false_vecs_hooked).append(vec)

h_proj.remove()
print(f"  Collected {len(true_vecs_hooked)} hooked true, {len(false_vecs_hooked)} hooked false vectors")

# ── Load SAE and encode both sets ─────────────────────────────────────────────
print(f"\nLoading SAE from {SAE_REPO} (layer {TARGET_LAYER})...")
from huggingface_hub import hf_hub_download
from huggingface_hub import login
HF_TOKEN = open(os.path.expanduser("~/.config/duoneural/hf.env")).read()
for line in HF_TOKEN.splitlines():
    if line.startswith("HF_TOKEN="):
        token = line.split("=", 1)[1].strip()
        login(token=token)
        break

sae_path = hf_hub_download(SAE_REPO, filename=f"layer{TARGET_LAYER}.sae.pt")
sae = torch.load(sae_path, map_location='cpu', weights_only=True)
W_enc = sae['W_enc'].float()  # (F, D)
b_enc = sae['b_enc'].float()  # (F,)
print(f"  SAE loaded: {W_enc.shape[0]} features, {W_enc.shape[1]} dim")

def encode(vecs):
    h = torch.tensor(np.stack(vecs), dtype=torch.float32)
    return torch.relu(h @ W_enc.T + b_enc).numpy()

true_raw_acts  = encode(true_vecs_raw)
false_raw_acts = encode(false_vecs_raw)
true_hook_acts = encode(true_vecs_hooked)
false_hook_acts = encode(false_vecs_hooked)

# ── Compare feature activations pre vs post hook ───────────────────────────────
results = {
    'model': MODEL_ID,
    'sae_repo': SAE_REPO,
    'target_layer': TARGET_LAYER,
    'alpha': ALPHA,
    'n_features': W_enc.shape[0],
    'tracked_features': {}
}

print(f"\n=== Feature Activation Comparison ===")
print(f"{'Feature':<12} {'Raw True':>10} {'Hook True':>10} {'Delta':>10} | {'Raw False':>10} {'Hook False':>10}")
print("-" * 70)

for fi in TRACK_FEATURES:
    if fi >= W_enc.shape[0]:
        print(f"  F{fi}: out of range (SAE has {W_enc.shape[0]} features)")
        continue

    raw_true  = float(true_raw_acts[:, fi].mean())
    hook_true = float(true_hook_acts[:, fi].mean())
    raw_false  = float(false_raw_acts[:, fi].mean())
    hook_false = float(false_hook_acts[:, fi].mean())
    delta_true = hook_true - raw_true

    results['tracked_features'][str(fi)] = {
        'raw_true': raw_true,
        'hook_true': hook_true,
        'delta_true': delta_true,
        'raw_false': raw_false,
        'hook_false': hook_false,
        'hook_increases_truth': delta_true > 0,
    }
    print(f"  F{fi:<9} {raw_true:>10.4f} {hook_true:>10.4f} {delta_true:>+10.4f} | {raw_false:>10.4f} {hook_false:>10.4f}")

# Overall differential: does hook increase true-false gap?
diff_raw  = true_raw_acts.mean(0)  - false_raw_acts.mean(0)   # (F,)
diff_hook = true_hook_acts.mean(0) - false_hook_acts.mean(0)

top10_raw  = np.argsort(np.abs(diff_raw))[::-1][:10]
top10_hook = np.argsort(np.abs(diff_hook))[::-1][:10]

results['top10_raw_features']  = [int(i) for i in top10_raw]
results['top10_hook_features'] = [int(i) for i in top10_hook]

overlap = len(set(top10_raw) & set(top10_hook))
results['top10_overlap'] = overlap

print(f"\nTop-10 feature overlap (raw vs hook): {overlap}/10")
print(f"  Top raw features:  {list(top10_raw)}")
print(f"  Top hook features: {list(top10_hook)}")

verdict = (
    results['tracked_features'].get('24363', {}).get('hook_increases_truth', False) or
    results['tracked_features'].get('44157', {}).get('hook_increases_truth', False)
)
results['convergence_verdict'] = 'CONFIRMED' if verdict else 'NOT_CONFIRMED'
print(f"\nConvergence verdict: {results['convergence_verdict']}")
if verdict:
    print("  Hook increases activation of truth-encoding SAE features → mechanistic confirmation")
else:
    print("  Hook does NOT increase truth SAE features → projection is orthogonal to feature basis")

with open(OUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT_PATH}")
print("DONE")
