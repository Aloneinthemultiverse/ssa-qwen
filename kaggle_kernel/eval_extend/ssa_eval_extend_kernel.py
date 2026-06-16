"""Context EXTENSION test (the Sub-Q / SSA claim): is sparse attention more robust
than full attention when we stretch the window past its trained 32k limit?

We apply YaRN RoPE scaling (factor 4 -> ~128k) to Qwen2.5-0.5B and measure
perplexity at 32k / 64k / 128k for: full attention vs sparse (untrained) vs
sparse + 8k-adapter. SSA's claim: beyond the native window, full attention
degrades faster (positional distortion), while sparse -- attending to fewer,
more-local tokens -- holds up better. Lower perplexity = better.
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=5.0.0", "peft", "datasets", "accelerate", "torchao>=0.16.0"], check=True)

import math, glob, os, gc
import torch
import torch.nn.functional as F

SPARSE_ENABLED = True
SINK_TOKENS = 4
WINDOW = 256
BLOCK_SIZE = 64
TOP_K_BLOCKS = 8
QUERY_CHUNK = 256  # smaller chunk: bounds the per-chunk bias tensor at 128k


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
N_DOCS = 4
CE_CHUNK = 2048
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# YaRN RoPE scaling: extend native 32768 -> 131072 (factor 4)
config = AutoConfig.from_pretrained(MODEL)
if getattr(config, "rope_theta", None) is None:
    config.rope_theta = 1000000.0  # Qwen2.5 default; YaRN init needs it non-None
config.rope_scaling = {"rope_type": "yarn", "factor": 4.0,
                       "original_max_position_embeddings": 32768}
config.max_position_embeddings = 131072
print("rope_theta:", config.rope_theta, flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, config=config, dtype=dtype)
base.set_attn_implementation("ssa_sparse")

adapter = None
for p in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True):
    if "8k" in p:
        adapter = os.path.dirname(p)
if adapter:
    model = PeftModel.from_pretrained(base, adapter).to(device).eval()
    clm = model.base_model.model
    print(f"adapter: {adapter}", flush=True)
else:
    model = base.to(device).eval()
    clm = model
backbone = clm.model
lm_head = clm.lm_head
HAS_ADAPTER = adapter is not None

print("loading docs (up to 131072 tokens each)...", flush=True)
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
docs = []
buf, maxL = [], max(LENGTHS)
for ex in ds:
    buf.extend(tokenizer(ex["text"]).input_ids + [tokenizer.eos_token_id])
    while len(buf) >= maxL:
        docs.append(buf[:maxL]); buf = buf[maxL:]
    if len(docs) >= N_DOCS:
        break


def perplexity(L, sparse, adapters):
    global SPARSE_ENABLED
    SPARSE_ENABLED = sparse
    if adapters and HAS_ADAPTER:
        ctx = nullcontext()
    elif HAS_ADAPTER:
        ctx = model.disable_adapter()
    else:
        ctx = nullcontext()
    total_nll, total_tok = 0.0, 0
    with torch.no_grad(), ctx:
        for doc in docs:
            ids = torch.tensor(doc[:L]).unsqueeze(0).to(device)
            hidden = backbone(input_ids=ids).last_hidden_state
            T = ids.shape[1]
            for i in range(0, T - 1, CE_CHUNK):
                j = min(i + CE_CHUNK, T - 1)
                logits = lm_head(hidden[:, i:j]).float()
                tgt = ids[:, i + 1:j + 1]
                total_nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                             tgt.reshape(-1), reduction="sum").item()
                total_tok += (j - i)
            del hidden
            gc.collect(); torch.cuda.empty_cache()
    return math.exp(total_nll / total_tok)


modes = [("full", False, False), ("sparse-untrained", True, False)]
if HAS_ADAPTER:
    modes.append(("sparse+8k-adapter", True, True))

print(f"\n=== PERPLEXITY under YaRN x4 (native window = 32768) ===", flush=True)
header = f"{'ctx':>7} | " + " | ".join(f"{m[0]:>16}" for m in modes)
print(header, flush=True)
for L in LENGTHS:
    cells = []
    for name, sp, ad in modes:
        try:
            cells.append(f"{perplexity(L, sp, ad):>16.3f}")
        except RuntimeError as ex:
            cells.append(f"{'OOM' if 'memory' in str(ex).lower() else 'ERR':>16}")
            gc.collect(); torch.cuda.empty_cache()
    print(f"{L:>7} | " + " | ".join(cells), flush=True)

print("\ndone", flush=True)
