"""Head-to-head long-context perplexity: does training the adapter at 8k recover
more of the sparsity gap than training at 2k?

Modes: full | sparse-untrained | sparse + 2k-adapter | sparse + 8k-adapter
at 8k / 16k / 32k. Mounts both adapter datasets (ssa-qwen-lora = 2k,
ssa-qwen-lora-8k = 8k).
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=5.0.0", "peft", "datasets", "accelerate", "torchao>=0.16.0"], check=True)

import math, glob, os
import torch
import torch.nn.functional as F

SPARSE_ENABLED = True
SINK_TOKENS = 4
WINDOW = 256
BLOCK_SIZE = 64
TOP_K_BLOCKS = 8
QUERY_CHUNK = 512


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

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from contextlib import nullcontext

MODEL = "Qwen/Qwen2.5-0.5B"


def find_adapter(name_contains):
    for p in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True):
        if name_contains in p:
            return os.path.dirname(p)
    raise FileNotFoundError(name_contains)

SHORT = find_adapter("ssa-qwen-lora/") if glob.glob("/kaggle/input/ssa-qwen-lora/**/adapter_config.json", recursive=True) else find_adapter("lora")
LONG = find_adapter("8k")

LENGTHS = [8192, 16384, 32768]
N_DOCS = 8
CE_CHUNK = 2048
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
base.set_attn_implementation("ssa_sparse")
model = PeftModel.from_pretrained(base, SHORT, adapter_name="short")
model.load_adapter(LONG, adapter_name="long")
model.to(device).eval()
clm = model.base_model.model
backbone = clm.model
lm_head = clm.lm_head

print(f"short adapter: {SHORT}\nlong adapter : {LONG}", flush=True)

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
doc_sets = {L: [] for L in LENGTHS}
buf, maxL = [], max(LENGTHS)
for ex in ds:
    buf.extend(tokenizer(ex["text"]).input_ids + [tokenizer.eos_token_id])
    while len(buf) >= maxL:
        chunk = buf[:maxL]; buf = buf[maxL:]
        for L in LENGTHS:
            if len(doc_sets[L]) < N_DOCS:
                doc_sets[L].append(chunk[:L])
    if all(len(doc_sets[L]) >= N_DOCS for L in LENGTHS):
        break


def perplexity(docs, sparse, mode):
    """mode: 'full' (adapters off), 'short', or 'long'."""
    global SPARSE_ENABLED
    SPARSE_ENABLED = sparse
    if mode == "full":
        ctx = model.disable_adapter()
    else:
        model.set_adapter(mode)
        ctx = nullcontext()
    total_nll, total_tok = 0.0, 0
    with torch.no_grad(), ctx:
        for ids in docs:
            ids = torch.tensor(ids).unsqueeze(0).to(device)
            hidden = backbone(input_ids=ids).last_hidden_state
            T = ids.shape[1]
            for i in range(0, T - 1, CE_CHUNK):
                j = min(i + CE_CHUNK, T - 1)
                logits = lm_head(hidden[:, i:j]).float()
                tgt = ids[:, i + 1:j + 1]
                total_nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                             tgt.reshape(-1), reduction="sum").item()
                total_tok += (j - i)
    return math.exp(total_nll / total_tok)


print(f"\n=== PERPLEXITY ({N_DOCS} docs/length) ===", flush=True)
print(f"{'ctx':>6} | {'full':>7} | {'untrnd':>7} | {'2k-ad':>7} | {'8k-ad':>7} | rec2k | rec8k", flush=True)
for L in LENGTHS:
    f = perplexity(doc_sets[L], False, "full")
    u = perplexity(doc_sets[L], True, "full")
    s2 = perplexity(doc_sets[L], True, "short")
    s8 = perplexity(doc_sets[L], True, "long")
    gap = u - f
    r2 = (u - s2) / gap * 100 if abs(gap) > 1e-6 else float("nan")
    r8 = (u - s8) / gap * 100 if abs(gap) > 1e-6 else float("nan")
    print(f"{L:>6} | {f:>7.3f} | {u:>7.3f} | {s2:>7.3f} | {s8:>7.3f} | {r2:>4.0f}% | {r8:>4.0f}%", flush=True)

print("\ndone", flush=True)
