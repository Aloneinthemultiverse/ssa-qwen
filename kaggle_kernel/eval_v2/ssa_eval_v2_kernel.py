"""Decisive v2 eval: did the SSA-paper-aligned changes (top-32 blocks, trained
under YaRN) fix mid-document retrieval and improve long-context perplexity?

New architecture (block 32, top-32, window 128) + v2 adapter (trained under YaRN x4).
Measures perplexity AND buried-fact retrieval at 32k/64k/128k, full vs sparse+v2.
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=5.0.0", "peft", "datasets", "accelerate", "torchao>=0.16.0"], check=True)

import math, random, glob, os, gc
import torch
import torch.nn.functional as F

SPARSE_ENABLED = True
SINK_TOKENS = 4
WINDOW = 128
BLOCK_SIZE = 32
TOP_K_BLOCKS = 32
QUERY_CHUNK = 256


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _chunk_bias(q, q_abs, block_keys, kv_len, k_pos):
    b, h, qc, d = q.shape
    device = q.device
    neg = torch.finfo(q.dtype).min
    causal = k_pos[None, :] <= q_abs[:, None]
    sink = k_pos[None, :] < SINK_TOKENS
    window = (q_abs[:, None] - k_pos[None, :]) < WINDOW
    allow = (sink | window) & causal
    allow = allow[None, None].expand(b, h, qc, kv_len).clone()
    if block_keys is not None:
        n_blocks = block_keys.shape[2]
        scores = torch.einsum("bhqd,bhnd->bhqn", q, block_keys)
        block_end = (torch.arange(n_blocks, device=device) + 1) * BLOCK_SIZE - 1
        sel_ok = block_end[None, :] <= q_abs[:, None]
        scores = scores.masked_fill(~sel_ok[None, None], neg)
        k_sel = min(TOP_K_BLOCKS, n_blocks)
        top = scores.topk(k_sel, dim=-1).indices
        block_mask = torch.zeros(b, h, qc, n_blocks, dtype=torch.bool, device=device)
        block_mask.scatter_(-1, top, True)
        token_mask = block_mask.repeat_interleave(BLOCK_SIZE, dim=-1)[..., :kv_len]
        allow |= token_mask & causal[None, None]
    return torch.where(allow, torch.zeros((), dtype=q.dtype, device=device),
                       torch.full((), neg, dtype=q.dtype, device=device))


def ssa_sparse_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    n_rep = getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1])
    key = _repeat_kv(key, n_rep)
    value = _repeat_kv(value, n_rep)
    b, h, q_len, d = query.shape
    kv_len = key.shape[2]
    device = query.device
    if not SPARSE_ENABLED:
        if q_len == kv_len:
            attn = F.scaled_dot_product_attention(query, key, value, is_causal=True,
                                                  dropout_p=dropout, scale=scaling)
        else:
            attn = F.scaled_dot_product_attention(query, key, value, attn_mask=None,
                                                  dropout_p=dropout, scale=scaling)
        return attn.transpose(1, 2).contiguous(), None
    k_pos = torch.arange(kv_len, device=device)
    n_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_keys = None
    if n_blocks > TOP_K_BLOCKS:
        pad = n_blocks * BLOCK_SIZE - kv_len
        k_padded = F.pad(key, (0, 0, 0, pad))
        block_keys = k_padded.view(b, h, n_blocks, BLOCK_SIZE, d).mean(dim=3)
    out = torch.empty_like(query)
    base = kv_len - q_len
    for s in range(0, q_len, QUERY_CHUNK):
        e = min(s + QUERY_CHUNK, q_len)
        q = query[:, :, s:e]
        q_abs = torch.arange(base + s, base + e, device=device)
        bias = _chunk_bias(q, q_abs, block_keys, kv_len, k_pos)
        out[:, :, s:e] = F.scaled_dot_product_attention(q, key, value, attn_mask=bias,
                                                        dropout_p=dropout, scale=scaling)
    return out.transpose(1, 2).contiguous(), None


from transformers.modeling_utils import AttentionInterface
AttentionInterface.register("ssa_sparse", ssa_sparse_attention)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
from datasets import load_dataset
from contextlib import nullcontext

MODEL = "Qwen/Qwen2.5-0.5B"
LENGTHS = [32768, 65536, 131072]
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

config = AutoConfig.from_pretrained(MODEL)
if getattr(config, "rope_theta", None) is None:
    config.rope_theta = 1000000.0
config.rope_scaling = {"rope_type": "yarn", "factor": 4.0,
                       "original_max_position_embeddings": 32768}
config.max_position_embeddings = 131072

tokenizer = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, config=config, dtype=dtype)
base.set_attn_implementation("ssa_sparse")
adapter = next(os.path.dirname(p) for p in
               glob.glob("/kaggle/input/**/adapter_config.json", recursive=True) if "v2" in p)
model = PeftModel.from_pretrained(base, adapter).to(device).eval()
clm = model.base_model.model
backbone = clm.model
lm_head = clm.lm_head
print(f"adapter: {adapter}  | arch: block{BLOCK_SIZE} top{TOP_K_BLOCKS} window{WINDOW}", flush=True)

CE_CHUNK = 2048
N_DOCS = 4
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
docs, buf, maxL = [], [], max(LENGTHS)
for ex in ds:
    buf.extend(tokenizer(ex["text"]).input_ids + [tokenizer.eos_token_id])
    while len(buf) >= maxL:
        docs.append(buf[:maxL]); buf = buf[maxL:]
    if len(docs) >= N_DOCS:
        break


def perplexity(L, sparse, adapters):
    global SPARSE_ENABLED
    SPARSE_ENABLED = sparse
    ctx = nullcontext() if adapters else model.disable_adapter()
    tot_nll, tot = 0.0, 0
    with torch.no_grad(), ctx:
        for doc in docs:
            ids = torch.tensor(doc[:L]).unsqueeze(0).to(device)
            hidden = backbone(input_ids=ids).last_hidden_state
            T = ids.shape[1]
            for i in range(0, T - 1, CE_CHUNK):
                j = min(i + CE_CHUNK, T - 1)
                logits = lm_head(hidden[:, i:j]).float()
                tot_nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                           ids[:, i + 1:j + 1].reshape(-1), reduction="sum").item()
                tot += (j - i)
            del hidden; gc.collect(); torch.cuda.empty_cache()
    return math.exp(tot_nll / tot)


DEPTHS = [0.1, 0.5, 0.9]
TRIALS = 2
filler = tokenizer("The sky was clear and the grass was green. People walked through "
                   "the park and chatted about their weekend plans. ").input_ids
q_ids = tokenizer("\nQuestion: What is the secret passkey?\nAnswer: The secret passkey is").input_ids


def retrieve(L, sparse, adapters):
    global SPARSE_ENABLED
    SPARSE_ENABLED = sparse
    random.seed(42)
    ctx = nullcontext() if adapters else model.disable_adapter()
    row = []
    with torch.no_grad(), ctx:
        for depth in DEPTHS:
            hits = 0
            for _ in range(TRIALS):
                key = str(random.randint(10000, 99999))
                needle = tokenizer(f" The secret passkey is {key}. Remember it. ").input_ids
                budget = L - len(needle) - len(q_ids)
                body = (filler * (budget // len(filler) + 1))[:budget]
                cut = int(budget * depth)
                ids = torch.tensor(body[:cut] + needle + body[cut:] + q_ids).unsqueeze(0).to(device)
                gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=8,
                                     do_sample=False, pad_token_id=tokenizer.eos_token_id)
                hits += key in tokenizer.decode(gen[0, ids.shape[1]:])
                del ids, gen; gc.collect(); torch.cuda.empty_cache()
            row.append(hits / TRIALS)
    return row


print(f"\n=== PERPLEXITY (v2 arch+adapter, YaRN x4, {N_DOCS} docs) ===", flush=True)
print(f"{'ctx':>7} | {'full':>8} | {'sparse+v2':>10}", flush=True)
for L in LENGTHS:
    try:
        f = perplexity(L, False, False); s = perplexity(L, True, True)
        print(f"{L:>7} | {f:>8.3f} | {s:>10.3f}", flush=True)
    except RuntimeError as ex:
        print(f"{L:>7} | err: {str(ex)[:50]}", flush=True); gc.collect(); torch.cuda.empty_cache()

for name, sp, ad in [("FULL", False, False), ("SPARSE+v2", True, True)]:
    print(f"\n=== RETRIEVAL: {name} (start/mid/end) ===", flush=True)
    print(f"{'ctx':>7} | " + " | ".join(f"d={d}" for d in DEPTHS), flush=True)
    for L in LENGTHS:
        try:
            row = retrieve(L, sp, ad)
            print(f"{L:>7} | " + " | ".join(f"{a:>4.0%}" for a in row), flush=True)
        except RuntimeError as ex:
            print(f"{L:>7} | err: {str(ex)[:40]}", flush=True); gc.collect(); torch.cuda.empty_cache()

print("\ndone", flush=True)
