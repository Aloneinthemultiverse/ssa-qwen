"""Harder eval for SSA-Qwen: long-text perplexity + stress NIAH (top-2 blocks).

The standard passkey NIAH saturated (100% for all modes incl. untrained sparse) --
top-k block selection finds the needle unaided. This kernel measures what
alignment training actually bought:
  1. Perplexity on long documents under sparse attention (full vs sparse vs ours)
  2. NIAH with TOP_K_BLOCKS=2 at eval time (selection budget stress test)
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=5.0.0", "peft", "datasets", "accelerate", "torchao>=0.16.0"], check=True)

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

# --------------------------------- setup ------------------------------------

import glob, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

MODEL = "Qwen/Qwen2.5-0.5B"
hits = glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
assert hits, "adapter not found"
ADAPTER = os.path.dirname(hits[0])

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
base.set_attn_implementation("ssa_sparse")
model = PeftModel.from_pretrained(base, ADAPTER)
model.to(device).eval()

# ------------------------- 1) long-text perplexity --------------------------

SEQ_LEN = 4096
N_DOCS = 20

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
docs, buf = [], []
for ex in ds:
    buf.extend(tokenizer(ex["text"]).input_ids + [tokenizer.eos_token_id])
    while len(buf) >= SEQ_LEN:
        docs.append(buf[:SEQ_LEN]); buf = buf[SEQ_LEN:]
    if len(docs) >= N_DOCS:
        break


def perplexity(sparse, adapters):
    global SPARSE_ENABLED
    SPARSE_ENABLED = sparse
    ctx = model.disable_adapter() if not adapters else None
    if ctx: ctx.__enter__()
    total, count = 0.0, 0
    try:
        for doc in docs:
            ids = torch.tensor(doc).unsqueeze(0).to(device)
            with torch.no_grad():
                loss = model(ids, labels=ids).loss
            total += loss.float().item(); count += 1
    finally:
        if ctx: ctx.__exit__(None, None, None)
    import math
    return math.exp(total / count)


print(f"\n=== PERPLEXITY on {N_DOCS} x {SEQ_LEN}-token fineweb-edu docs ===", flush=True)
ppl_full = perplexity(sparse=False, adapters=False)
print(f"full attention      : {ppl_full:.2f}", flush=True)
ppl_sparse = perplexity(sparse=True, adapters=False)
print(f"sparse, untrained   : {ppl_sparse:.2f}", flush=True)
ppl_ours = perplexity(sparse=True, adapters=True)
print(f"sparse + alignment  : {ppl_ours:.2f}", flush=True)
gap = ppl_sparse - ppl_full
rec = (ppl_sparse - ppl_ours) / gap * 100 if gap > 0 else float("nan")
print(f"gap recovered by alignment: {rec:.0f}%", flush=True)

# ----------------------- 2) stress NIAH (top-2 blocks) ----------------------

TOP_K_BLOCKS = 2  # stress: tiny selection budget at eval time

CONTEXT_LENS = [2048, 4096, 8192]
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
TRIALS = 3
FILLER = (
    "The sky was clear and the grass was green. People walked through the park "
    "and talked about the weather, the news, and their plans for the weekend. "
)


def run_grid(label, sparse, adapters):
    global SPARSE_ENABLED
    SPARSE_ENABLED = sparse
    random.seed(42)
    print(f"\n=== {label} ===", flush=True)
    print(f"{'ctx':>6} | " + " | ".join(f"d={d:.1f}" for d in DEPTHS), flush=True)
    ctx = model.disable_adapter() if not adapters else None
    if ctx: ctx.__enter__()
    try:
        for ctx_len in CONTEXT_LENS:
            row = []
            for depth in DEPTHS:
                hits_n = 0
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
                    with torch.no_grad():
                        gen = model.generate(ids, attention_mask=torch.ones_like(ids),
                                             max_new_tokens=8, do_sample=False,
                                             pad_token_id=tokenizer.eos_token_id)
                    hits_n += key in tokenizer.decode(gen[0, ids.shape[1]:])
                row.append(hits_n / TRIALS)
            print(f"{ctx_len:>6} | " + " | ".join(f"{a:>5.0%}" for a in row), flush=True)
    finally:
        if ctx: ctx.__exit__(None, None, None)


run_grid("STRESS NIAH top2: SPARSE UNTRAINED", sparse=True, adapters=False)
run_grid("STRESS NIAH top2: SPARSE + ALIGNMENT (ours)", sparse=True, adapters=True)
print("\ndone")
