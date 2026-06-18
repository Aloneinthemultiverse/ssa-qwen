"""Does the compression branch help? Controlled perplexity test on the SAME nsa2
model with compression toggled OFF vs ON (and full attention as ceiling), at
32k/64k/128k under YaRN x4. Lower perplexity = better. If ON < OFF, compression
adds value (where passkey retrieval can't show it -- v3 already saturates that).
"""

import subprocess, sys, os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=5.0.0", "peft", "accelerate", "torchao>=0.16.0"], check=True)

import math, glob, os, gc
import torch
import torch.nn as nn
import torch.nn.functional as F

SPARSE_ENABLED = True
COMPRESSION_ON = True          # toggled for the controlled comparison
SINK_TOKENS = 4; WINDOW = 128; BLOCK_SIZE = 32; L_CMP = 32; TOP_K_BLOCKS = 32; QUERY_CHUNK = 256


class NSAModule(nn.Module):
    def __init__(self, head_dim, l_cmp=L_CMP):
        super().__init__()
        self.d = head_dim; self.l_cmp = l_cmp
        self.pool_logits = nn.Parameter(torch.zeros(l_cmp))
        self.refine = nn.Linear(head_dim, head_dim, bias=False)
        self.gate = nn.Linear(head_dim, 1)

    def compress(self, k, v):
        k = k.float(); v = v.float()
        b, h, kv, d = k.shape
        nb = (kv + self.l_cmp - 1) // self.l_cmp
        pad = nb * self.l_cmp - kv
        kb = F.pad(k, (0, 0, 0, pad)).view(b, h, nb, self.l_cmp, d)
        vb = F.pad(v, (0, 0, 0, pad)).view(b, h, nb, self.l_cmp, d)
        w = self.pool_logits.softmax(dim=0).view(1, 1, 1, self.l_cmp, 1)
        ck = (kb * w).sum(dim=3); cv = (vb * w).sum(dim=3)
        ck = ck + self.refine(ck)
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
    n = 0
    for layer in _find_layers(model):
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
    n_blk = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    raw_blk_k = F.pad(key, (0, 0, 0, n_blk * BLOCK_SIZE - kv_len)).view(b, h, n_blk, BLOCK_SIZE, d).mean(dim=3)
    blk_end = (torch.arange(n_blk, device=device) + 1) * BLOCK_SIZE - 1
    if COMPRESSION_ON:
        ck, cv, n_cmp = nsa.compress(key, value)
        ck = ck.to(query.dtype); cv = cv.to(query.dtype)
        cmp_end = (torch.arange(n_cmp, device=device) + 1) * L_CMP - 1
    k_pos = torch.arange(kv_len, device=device); neg = torch.finfo(query.dtype).min
    out = torch.empty_like(query); base = kv_len - q_len
    for s in range(0, q_len, QUERY_CHUNK):
        e = min(s + QUERY_CHUNK, q_len); q = query[:, :, s:e]; qc = e - s
        q_abs = torch.arange(base + s, base + e, device=device)
        causal = k_pos[None, :] <= q_abs[:, None]
        sel = torch.einsum("bhqd,bhnd->bhqn", q, raw_blk_k) * scale
        sel = sel.masked_fill(~(blk_end[None, :] <= q_abs[:, None])[None, None], neg)
        top = sel.topk(min(TOP_K_BLOCKS, n_blk), dim=-1).indices
        blk = torch.zeros(b, h, qc, n_blk, dtype=torch.bool, device=device)
        blk.scatter_(-1, top, True)
        tok = blk.repeat_interleave(BLOCK_SIZE, dim=-1)[..., :kv_len]
        sink = k_pos[None, :] < SINK_TOKENS
        window = (q_abs[:, None] - k_pos[None, :]) < WINDOW
        allow = ((sink | window) & causal)[None, None].expand(b, h, qc, kv_len).clone()
        allow |= tok & causal[None, None]
        bias = torch.where(allow, torch.zeros((), dtype=query.dtype, device=device),
                           torch.full((), neg, dtype=query.dtype, device=device))
        sw_out = F.scaled_dot_product_attention(q, key, value, attn_mask=bias, dropout_p=dropout, scale=scaling)
        if COMPRESSION_ON:
            cs = torch.einsum("bhqd,bhnd->bhqn", q, ck) * scale
            cs = cs.masked_fill(~(cmp_end[None, :] <= q_abs[:, None])[None, None], neg)
            cmp_out = torch.einsum("bhqn,bhnd->bhqd", cs.softmax(dim=-1), cv)
            g = torch.sigmoid(nsa.gate(q.float())).to(query.dtype)
            out[:, :, s:e] = sw_out + g * cmp_out
        else:
            out[:, :, s:e] = sw_out
    return out.transpose(1, 2).contiguous(), None


from transformers.modeling_utils import AttentionInterface
AttentionInterface.register("nsa_sparse", nsa_sparse_attention)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
from datasets import load_dataset
from contextlib import nullcontext

MODEL = "Qwen/Qwen2.5-0.5B"; LENGTHS = [32768, 65536, 131072]; N_DOCS = 6; CE_CHUNK = 2048
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
sd = torch.load(next(p for p in glob.glob("/kaggle/input/**/*.pt", recursive=True)), map_location=device)
own = dict(model.named_parameters()); ld = 0
for k, v in sd.items():
    if k in own:
        own[k].data.copy_(v.to(own[k].dtype).to(device)); ld += 1
print(f"loaded {ld}/{len(sd)} nsa tensors", flush=True)
clm = model.base_model.model; backbone = clm.model; lm_head = clm.lm_head

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
docs, buf, maxL = [], [], max(LENGTHS)
for ex in ds:
    buf.extend(tokenizer(ex["text"]).input_ids + [tokenizer.eos_token_id])
    while len(buf) >= maxL:
        docs.append(buf[:maxL]); buf = buf[maxL:]
    if len(docs) >= N_DOCS:
        break


def ppl(L, sparse, comp, adapters):
    global SPARSE_ENABLED, COMPRESSION_ON
    SPARSE_ENABLED = sparse; COMPRESSION_ON = comp
    ctx = nullcontext() if adapters else model.disable_adapter()
    tot_nll, tot = 0.0, 0
    with torch.no_grad(), ctx:
        for doc in docs:
            ids = torch.tensor(doc[:L]).unsqueeze(0).to(device)
            hidden = backbone(input_ids=ids).last_hidden_state; T = ids.shape[1]
            for i in range(0, T - 1, CE_CHUNK):
                j = min(i + CE_CHUNK, T - 1)
                lg = lm_head(hidden[:, i:j]).float()
                tot_nll += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, i + 1:j + 1].reshape(-1), reduction="sum").item()
                tot += (j - i)
            del hidden; gc.collect(); torch.cuda.empty_cache()
    return math.exp(tot_nll / tot)


print(f"\n=== PERPLEXITY (compression ablation, YaRN x4, {N_DOCS} docs) ===", flush=True)
print(f"{'ctx':>7} | {'full':>8} | {'cmp-OFF':>8} | {'cmp-ON':>8} | delta", flush=True)
for L in LENGTHS:
    try:
        f = ppl(L, False, False, False)
        off = ppl(L, True, False, True)
        on = ppl(L, True, True, True)
        print(f"{L:>7} | {f:>8.3f} | {off:>8.3f} | {on:>8.3f} | {off - on:+.3f}", flush=True)
    except RuntimeError as ex:
        print(f"{L:>7} | err: {str(ex)[:50]}", flush=True); gc.collect(); torch.cuda.empty_cache()
print("\ndone", flush=True)
