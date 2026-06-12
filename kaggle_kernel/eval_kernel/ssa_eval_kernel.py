"""NIAH eval for SSA-Qwen: full attention vs untrained sparse vs SSA-aligned sparse.

Self-contained Kaggle script kernel. Expects the trained LoRA adapter mounted at
/kaggle/input/ssa-qwen-lora (dataset sujitnarrayanm/ssa-qwen-lora).
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=5.0.0", "peft", "accelerate", "torchao>=0.16.0"], check=True)

import random
import torch
import torch.nn.functional as F

# ----------------------------- sparse attention -----------------------------

SPARSE_ENABLED = True
SINK_TOKENS = 4
WINDOW = 256
BLOCK_SIZE = 64
TOP_K_BLOCKS = 8


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def build_sparse_mask(query, key):
    b, h, q_len, d = query.shape
    kv_len = key.shape[2]
    device = query.device

    q_pos = torch.arange(kv_len - q_len, kv_len, device=device)
    k_pos = torch.arange(kv_len, device=device)
    causal = k_pos[None, :] <= q_pos[:, None]

    sink = k_pos[None, :] < SINK_TOKENS
    window = (q_pos[:, None] - k_pos[None, :]) < WINDOW
    static = (sink | window) & causal
    mask = static[None, None].expand(b, h, q_len, kv_len).clone()

    n_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    if n_blocks > TOP_K_BLOCKS:
        pad = n_blocks * BLOCK_SIZE - kv_len
        k_padded = F.pad(key, (0, 0, 0, pad))
        block_keys = k_padded.view(b, h, n_blocks, BLOCK_SIZE, d).mean(dim=3)
        scores = torch.einsum("bhqd,bhnd->bhqn", query, block_keys)
        block_end = (torch.arange(n_blocks, device=device) + 1) * BLOCK_SIZE - 1
        sel_ok = block_end[None, :] <= q_pos[:, None]
        scores = scores.masked_fill(~sel_ok[None, None], float("-inf"))
        k_sel = min(TOP_K_BLOCKS, n_blocks)
        top = scores.topk(k_sel, dim=-1).indices
        block_mask = torch.zeros(b, h, q_len, n_blocks, dtype=torch.bool, device=device)
        block_mask.scatter_(-1, top, True)
        token_mask = block_mask.repeat_interleave(BLOCK_SIZE, dim=-1)[..., :kv_len]
        mask |= token_mask & causal[None, None]
    return mask


def ssa_sparse_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    n_rep = getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1])
    key = _repeat_kv(key, n_rep)
    value = _repeat_kv(value, n_rep)

    if SPARSE_ENABLED:
        allow = build_sparse_mask(query, key)
        bias = torch.zeros_like(allow, dtype=query.dtype)
        bias.masked_fill_(~allow, torch.finfo(query.dtype).min)
        if attention_mask is not None:
            bias = bias + attention_mask[..., : key.shape[2]].to(bias.dtype)
        attn = F.scaled_dot_product_attention(
            query, key, value, attn_mask=bias, dropout_p=dropout, scale=scaling)
    else:
        if attention_mask is not None:
            full_mask = attention_mask[..., : key.shape[2]]
        else:
            # explicit bottom-right-aligned causal mask: is_causal aligns top-left
            # when q_len < kv_len (decoding), which would blind the model
            q_len, kv_len = query.shape[2], key.shape[2]
            q_pos = torch.arange(kv_len - q_len, kv_len, device=query.device)
            k_pos = torch.arange(kv_len, device=query.device)
            allow = k_pos[None, :] <= q_pos[:, None]
            full_mask = torch.zeros(q_len, kv_len, dtype=query.dtype, device=query.device)
            full_mask.masked_fill_(~allow, torch.finfo(query.dtype).min)
        attn = F.scaled_dot_product_attention(
            query, key, value, attn_mask=full_mask, dropout_p=dropout, scale=scaling)
    return attn.transpose(1, 2).contiguous(), None


from transformers.modeling_utils import AttentionInterface
AttentionInterface.register("ssa_sparse", ssa_sparse_attention)

# --------------------------------- NIAH eval --------------------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-0.5B"
import glob, os
hits = glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
print("input tree:", glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*"))
assert hits, "adapter_config.json not found anywhere under /kaggle/input"
ADAPTER = os.path.dirname(hits[0])
print("using adapter at:", ADAPTER)
CONTEXT_LENS = [1024, 2048, 4096, 8192]
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
TRIALS = 3

FILLER = (
    "The sky was clear and the grass was green. People walked through the park "
    "and talked about the weather, the news, and their plans for the weekend. "
)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
base.set_attn_implementation("ssa_sparse")
model = PeftModel.from_pretrained(base, ADAPTER)
model.to(device).eval()


def run_grid(label, sparse, adapters):
    global SPARSE_ENABLED
    SPARSE_ENABLED = sparse
    random.seed(42)
    print(f"\n=== {label} ===", flush=True)
    print(f"{'ctx':>6} | " + " | ".join(f"d={d:.1f}" for d in DEPTHS), flush=True)
    results = {}
    ctx = model.disable_adapter() if not adapters else None
    if ctx: ctx.__enter__()
    try:
        for ctx_len in CONTEXT_LENS:
            row = []
            for depth in DEPTHS:
                hits = 0
                for _ in range(TRIALS):
                    key = str(random.randint(10000, 99999))
                    needle_ids = tokenizer(f" The secret passkey is {key}. Remember it. ").input_ids
                    q_ids = tokenizer("\nQuestion: What is the secret passkey?\nAnswer: The secret passkey is").input_ids
                    filler_ids = tokenizer(FILLER).input_ids
                    budget = ctx_len - len(needle_ids) - len(q_ids)
                    body = (filler_ids * (budget // len(filler_ids) + 1))[:budget]
                    cut = int(budget * depth)
                    ids = body[:cut] + needle_ids + body[cut:] + q_ids
                    ids = torch.tensor(ids).unsqueeze(0).to(device)
                    attn_mask = torch.ones_like(ids)
                    with torch.no_grad():
                        gen = model.generate(ids, attention_mask=attn_mask,
                                             max_new_tokens=8, do_sample=False,
                                             pad_token_id=tokenizer.eos_token_id)
                    hits += key in tokenizer.decode(gen[0, ids.shape[1]:])
                row.append(hits / TRIALS)
            results[ctx_len] = row
            print(f"{ctx_len:>6} | " + " | ".join(f"{a:>5.0%}" for a in row), flush=True)
    finally:
        if ctx: ctx.__exit__(None, None, None)
    return results


full = run_grid("FULL ATTENTION (baseline)", sparse=False, adapters=False)
untrained = run_grid("SPARSE, UNTRAINED", sparse=True, adapters=False)
trained = run_grid("SPARSE + SSA ALIGNMENT (ours)", sparse=True, adapters=True)

print("\n=== SUMMARY (mean accuracy per context length) ===")
print(f"{'ctx':>6} | {'full':>6} | {'sparse':>6} | {'ours':>6}")
for c in CONTEXT_LENS:
    f = sum(full[c]) / len(DEPTHS)
    u = sum(untrained[c]) / len(DEPTHS)
    t = sum(trained[c]) / len(DEPTHS)
    print(f"{c:>6} | {f:>6.0%} | {u:>6.0%} | {t:>6.0%}")
