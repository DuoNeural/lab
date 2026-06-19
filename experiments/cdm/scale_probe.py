import torch, json, sys
sys.path.insert(0, "/workspace")
from cdm_model_v3 import CDMLanguageModelV3, CDMConfigV3
import tiktoken
enc = tiktoken.get_encoding("gpt2")

cfg_dict = json.load(open("/workspace/cdm_v5_full/config.json"))
cfg = CDMConfigV3()
for k, v in cfg_dict.items():
    if hasattr(cfg, k): setattr(cfg, k, v)

model = CDMLanguageModelV3(cfg).cuda()
ckpt = torch.load("/workspace/cdm_v5_full/best/model.pt", map_location="cuda", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

scale_tests = [
    ("tiny",         "The tiny ant crawled slowly across the huge leaf."),
    ("vast",         "The vast cosmos stretches beyond any possible measurement."),
    ("microscopic",  "Under the microscope the microscopic bacteria moved rapidly."),
    ("enormous",     "The enormous whale surfaced and blew water into the sky."),
    ("infinite",     "The infinite expanse of space contains billions of galaxies."),
    ("quantum",      "At quantum scales particles exist in superposition of states."),
    ("atomic",       "Each atomic nucleus contains protons bound tightly together."),
    ("golden",       "The golden ratio phi equals approximately 1.618 in mathematics."),
    ("cosmic",       "Cosmic microwave background radiation fills the observable universe."),
    ("garden",       "She walked through the garden and picked a flower slowly."),
    ("running",      "He was running quickly down the street to catch the bus."),
    ("happy",        "The happy children played together outside in the warm sunshine."),
]

def probe(sentence):
    toks = enc.encode(sentence)
    x = torch.tensor([toks], dtype=torch.long, device="cuda")
    routes = {}
    origs = []
    for i, block in enumerate(model.blocks):
        o = block.cdm.forward
        origs.append(o)
        def mk(li, oo, cdm_mod):
            def p(h):
                g, rp = cdm_mod.compute_gates_and_route(h)
                routes[li] = rp[0].detach().cpu()
                return oo(h)
            return p
        block.cdm.forward = mk(i, o, block.cdm)
    with torch.no_grad():
        _ = model(x)
    for i, block in enumerate(model.blocks):
        block.cdm.forward = origs[i]
    tok_strs = [enc.decode([t]).strip() for t in toks]
    return routes, tok_strs

print("\n── Scale/magnitude hypothesis: does L7/slot11 track scale-of-reference? ──")
print(f"{'label':12s}  {'token':14s}  L7s11   L10s6   L11s7   verdict")
print("-" * 72)

for pair in scale_tests:
    lbl = pair[0]
    sentence = pair[1]
    routes, toks = probe(sentence)
    l7  = routes[7][:,  11]
    l10 = routes[10][:, 6]
    l11 = routes[11][:, 7]
    found = False
    for idx, ts in enumerate(toks):
        if ts.lower().startswith(lbl[:5].lower()):
            is_scale = lbl not in ("garden", "running", "happy")
            verdict = "SCALE" if is_scale else "neutral"
            print(f"{lbl:12s}  {ts:14s}  {l7[idx]:.3f}   {l10[idx]:.3f}   {l11[idx]:.3f}   [{verdict}]")
            found = True
            break
    if not found:
        mi = l7.argmax().item()
        print(f"{lbl:12s}  {toks[mi]:14s}  {l7[mi]:.3f}   {l10[mi]:.3f}   {l11[mi]:.3f}   [max-l7 fallback]")

print()
print("If scale hypothesis holds: SCALE words > ~0.100 L7s11, neutral words < ~0.060")
