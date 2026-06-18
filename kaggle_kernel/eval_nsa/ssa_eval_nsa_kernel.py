"""Eval NSA-Qwen (compression + selection + window) on buried-fact retrieval at
64k/128k (YaRN x4). Loads the trained LoRA + the new compression-module weights.
Compares full attention vs NSA-sparse. The question: does the compression branch
help retrieval (vs the v3 selection+window model that hit 100% at 128k)?
"""

import subprocess, sys, os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=5.0.0", "peft", "accelerate", "torchao>=0.16.0"], check=True)

import random, glob, os, gc
import torch
import torch.nn as nn
import torch.nn.functional as F

SPARSE_ENABLED = True
SINK_TOKENS = 4; WINDOW = 128; BLOCK_SIZE = 32; L_CMP = 32; TOP_K_BLOCKS = 16; QUERY_CHUNK = 256


class NSAModule(nn.Module):
    def __init__(self, head_dim, l_cmp=L_CMP):
        super().__init__()
        self.d = head_dim; self.l_cmp = l_cmp
        self.pos = nn.Parameter(torch.zeros(l_cmp, head_dim))
        self.compress_k = nn.Linear(l_cmp * head_dim, head_dim, bias=False)
        self.compress_v = nn.Linear(l_cmp * head_dim, head_dim, bias=False)
        self.gate = nn.Linear(head_dim, 3)

    def compress(self, k, v):
        k = k.float(); v = v.float()
        b, h, kv, d = k.shape
        nb = (kv + self.l_cmp - 1) // self.l_cmp
        pad = nb * self.l_cmp - kv
        kb = F.pad(k, (0, 0, 0, pad)).view(b, h, nb, self.l_cmp, d) + self.pos
        vb = F.pad(v, (0, 0, 0, pad)).view(b, h, nb, self.l_cmp, d)
        ck = self.compress_k(kb.reshape(b, h, nb, self.l_cmp * d))
        cv = self.compress_v(vb.reshape(b, h, nb, self.l_cmp * d))
        return ck, cv, nb


def _find_layers(m):
    if hasattr(m, "layers"):
        return m.layers
    for c in m.children():
        r = _find_layers(c)
        if r is not None:
            return r
    return None


def attach_nsa(model, device):
    layers = _find_layers(model); n = 0
    for layer in layers:
        attn = layer.self_attn
        d = getattr(attn, "head_dim", None) or attn.q_proj.out_features // attn.config.num_attention_heads
        attn.nsa = NSAModule(d).to(device=device, dtype=torch.float32); n += 1
    return n


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def nsa_sparse_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    n_rep = getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1])
    key = _repeat_kv(key, n_rep); value = _repeat_kv(value, n_rep)
    b, h, q_len, d = query.shape
    kv_len = key.shape[2]; device = query.device
    scale = scaling if scaling is not None else d ** -0.5
    if not SPARSE_ENABLED:
        if q_len == kv_len:
            attn = F.scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=dropout, scale=scaling)
        else:
            attn = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout, scale=scaling)
        return attn.transpose(1, 2).contiguous(), None
    nsa = module.nsa
    ck, cv, n_cmp = nsa.compress(key, value)
    ck = ck.to(query.dtype); cv = cv.to(query.dtype)
    cmp_block_end = (torch.arange(n_cmp, device=device) + 1) * L_CMP - 1
    k_pos = torch.arange(kv_len, device=device); neg = torch.finfo(query.dtype).min
    out = torch.empty_like(query); base = kv_len - q_len
    for s in range(0, q_len, QUERY_CHUNK):
        e = min(s + QUERY_CHUNK, q_len); q = query[:, :, s:e]; qc = e - s
        q_abs = torch.arange(base + s, base + e, device=device)
        cmp_scores = torch.einsum("bhqd,bhnd->bhqn", q, ck) * scale
        cmp_ok = cmp_block_end[None, :] <= q_abs[:, None]
        cmp_scores = cmp_scores.masked_fill(~cmp_ok[None, None], neg)
        cmp_out = torch.einsum("bhqn,bhnd->bhqd", cmp_scores.softmax(dim=-1), cv)
        causal = k_pos[None, :] <= q_abs[:, None]
        sink = k_pos[None, :] < SINK_TOKENS
        window = (q_abs[:, None] - k_pos[None, :]) < WINDOW
        allow = ((sink | window) & causal)[None, None].expand(b, h, qc, kv_len).clone()
        k_sel = min(TOP_K_BLOCKS, n_cmp)
        top = cmp_scores.topk(k_sel, dim=-1).indices
        blk = torch.zeros(b, h, qc, n_cmp, dtype=torch.bool, device=device)
        blk.scatter_(-1, top, True)
        tok = blk.repeat_interleave(L_CMP, dim=-1)[..., :kv_len]
        allow |= tok & causal[None, None]
        bias = torch.where(allow, torch.zeros((), dtype=query.dtype, device=device),
                           torch.full((), neg, dtype=query.dtype, device=device))
        sw_out = F.scaled_dot_product_attention(q, key, value, attn_mask=bias, dropout_p=dropout, scale=scaling)
        g = torch.sigmoid(nsa.gate(q.float())).to(query.dtype)
        out[:, :, s:e] = g[..., 0:1] * cmp_out + (g[..., 1:2] + g[..., 2:3]) * 0.5 * sw_out
    return out.transpose(1, 2).contiguous(), None


from transformers.modeling_utils import AttentionInterface
AttentionInterface.register("nsa_sparse", nsa_sparse_attention)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-0.5B"
LENGTHS = [65536, 131072]; DEPTHS = [0.1, 0.5, 0.9]; TRIALS = 3
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

config = AutoConfig.from_pretrained(MODEL)
if getattr(config, "rope_theta", None) is None:
    config.rope_theta = 1000000.0
config.rope_scaling = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
config.max_position_embeddings = 131072

tokenizer = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, config=config, dtype=dtype)
base.set_attn_implementation("nsa_sparse")
lora_dir = next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True))
model = PeftModel.from_pretrained(base, lora_dir).to(device).eval()
attach_nsa(model, device)
# load trained compression-module weights
nsa_pt = next(p for p in glob.glob("/kaggle/input/**/*.pt", recursive=True))
sd = torch.load(nsa_pt, map_location=device)
own = dict(model.named_parameters())
loaded = 0
for k, v in sd.items():
    if k in own:
        own[k].data.copy_(v.to(own[k].dtype).to(device)); loaded += 1
print(f"lora: {lora_dir}\nnsa weights: {nsa_pt} -> loaded {loaded}/{len(sd)} tensors", flush=True)

filler = tokenizer("The sky was clear and the grass was green. People walked through the park and chatted about their weekend plans. ").input_ids
q_ids = tokenizer("\nQuestion: What is the secret passkey?\nAnswer: The secret passkey is").input_ids


def retrieve(L, sparse, adapters):
    global SPARSE_ENABLED
    SPARSE_ENABLED = sparse
    random.seed(42)
    from contextlib import nullcontext
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


for name, sp, ad in [("FULL", False, False), ("NSA-sparse (compression+sel+window)", True, True)]:
    print(f"\n=== {name} (start/mid/end) ===", flush=True)
    print(f"{'ctx':>7} | " + " | ".join(f"d={d}" for d in DEPTHS), flush=True)
    for L in LENGTHS:
        try:
            row = retrieve(L, sp, ad)
            print(f"{L:>7} | " + " | ".join(f"{a:>4.0%}" for a in row), flush=True)
        except RuntimeError as ex:
            print(f"{L:>7} | err: {str(ex)[:40]}", flush=True); gc.collect(); torch.cuda.empty_cache()
print("\ndone", flush=True)
