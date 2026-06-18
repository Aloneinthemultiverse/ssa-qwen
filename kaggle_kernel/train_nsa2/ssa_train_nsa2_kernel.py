import subprocess, sys, os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=5.0.0", "peft", "datasets", "accelerate", "torchao>=0.16.0"], check=True)
"""NSA-style attention for Qwen, REDESIGNED so the compression branch can only help.

Lessons from the failed v1 (0% retrieval): a flatten->Linear compression with random
init destroys key information, and using it to drive block SELECTION poisons retrieval.

Redesign:
  - SELECTION stays on RAW mean-pooled keys (the v3-proven signal) -- decoupled from
    compression. Top-k blocks + sliding window + sink, exactly like the working model.
  - COMPRESSION is a SEPARATE additive branch: learnable WEIGHTED-MEAN pooling (the
    summary is a convex combo of real keys, so it stays in key space and stays
    meaningful), plus a zero-initialised refinement. At init it equals the plain mean,
    and the branch is gated ~0, so the model STARTS identical to the working model and
    only learns to add coarse global context.
  - Output: out = sw_out + sigmoid(gate) * cmp_out.

NSAModule params (pool logits, refine, gate) are new trainable modules attached per
layer; trained with the LoRA via the SSA alignment recipe. Qwen stays frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SPARSE_ENABLED = True
SINK_TOKENS = 4
WINDOW = 128
BLOCK_SIZE = 32          # selection block size (raw mean-pool)
L_CMP = 32               # compression block size
TOP_K_BLOCKS = 32        # selection budget (v3's working value)
QUERY_CHUNK = 512


class NSAModule(nn.Module):
    """Additive compression branch: weighted-mean pooling (key-space safe) + refine + gate."""

    def __init__(self, head_dim, l_cmp=L_CMP):
        super().__init__()
        self.d = head_dim
        self.l_cmp = l_cmp
        self.pool_logits = nn.Parameter(torch.zeros(l_cmp))   # softmax -> uniform mean at init
        self.refine = nn.Linear(head_dim, head_dim, bias=False)
        self.gate = nn.Linear(head_dim, 1)
        nn.init.zeros_(self.refine.weight)                    # start: compressed = plain mean
        with torch.no_grad():
            self.gate.bias.fill_(-4.0)                        # start: compression OFF

    def compress(self, k, v):
        """(b,h,kv,d) -> weighted-pooled (b,h,nb,d) for k and v; stays in key/value space."""
        k = k.float(); v = v.float()
        b, h, kv, d = k.shape
        nb = (kv + self.l_cmp - 1) // self.l_cmp
        pad = nb * self.l_cmp - kv
        kb = F.pad(k, (0, 0, 0, pad)).view(b, h, nb, self.l_cmp, d)
        vb = F.pad(v, (0, 0, 0, pad)).view(b, h, nb, self.l_cmp, d)
        w = self.pool_logits.softmax(dim=0).view(1, 1, 1, self.l_cmp, 1)
        ck = (kb * w).sum(dim=3)                              # (b,h,nb,d) convex combo of real keys
        cv = (vb * w).sum(dim=3)
        ck = ck + self.refine(ck)                            # zero-init refinement
        return ck, cv, nb


def _find_layers(m):
    if hasattr(m, "layers"):
        return m.layers
    for c in m.children():
        r = _find_layers(c)
        if r is not None:
            return r
    return None


def attach_nsa(model, device=None, dtype=torch.float32):
    layers = _find_layers(model)
    n = 0
    for layer in layers:
        attn = layer.self_attn
        d = getattr(attn, "head_dim", None) or attn.q_proj.out_features // attn.config.num_attention_heads
        p = next(attn.parameters())
        attn.nsa = NSAModule(d).to(device=device or p.device, dtype=dtype)
        n += 1
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
    # raw mean-pooled block keys for SELECTION (v3-proven, decoupled from compression)
    n_blk = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    padb = n_blk * BLOCK_SIZE - kv_len
    raw_blk_k = F.pad(key, (0, 0, 0, padb)).view(b, h, n_blk, BLOCK_SIZE, d).mean(dim=3)  # (b,h,n_blk,d)
    blk_end = (torch.arange(n_blk, device=device) + 1) * BLOCK_SIZE - 1
    # compression summaries for the ADDITIVE branch
    ck, cv, n_cmp = nsa.compress(key, value)
    ck = ck.to(query.dtype); cv = cv.to(query.dtype)
    cmp_end = (torch.arange(n_cmp, device=device) + 1) * L_CMP - 1
    k_pos = torch.arange(kv_len, device=device); neg = torch.finfo(query.dtype).min

    out = torch.empty_like(query); base = kv_len - q_len
    for s in range(0, q_len, QUERY_CHUNK):
        e = min(s + QUERY_CHUNK, q_len); q = query[:, :, s:e]; qc = e - s
        q_abs = torch.arange(base + s, base + e, device=device)
        causal = k_pos[None, :] <= q_abs[:, None]

        # selection + window + sink (raw mean-pool selection)
        sel_scores = torch.einsum("bhqd,bhnd->bhqn", q, raw_blk_k) * scale
        sel_ok = blk_end[None, :] <= q_abs[:, None]
        sel_scores = sel_scores.masked_fill(~sel_ok[None, None], neg)
        k_sel = min(TOP_K_BLOCKS, n_blk)
        top = sel_scores.topk(k_sel, dim=-1).indices
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

        # additive compression branch (coarse global context over block summaries)
        cmp_scores = torch.einsum("bhqd,bhnd->bhqn", q, ck) * scale
        cmp_ok = cmp_end[None, :] <= q_abs[:, None]
        cmp_scores = cmp_scores.masked_fill(~cmp_ok[None, None], neg)
        cmp_out = torch.einsum("bhqn,bhnd->bhqd", cmp_scores.softmax(dim=-1), cv)

        g = torch.sigmoid(nsa.gate(q.float())).to(query.dtype)   # (b,h,qc,1), starts ~0
        out[:, :, s:e] = sw_out + g * cmp_out
    return out.transpose(1, 2).contiguous(), None



from transformers.modeling_utils import AttentionInterface
AttentionInterface.register("nsa_sparse", nsa_sparse_attention)

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
SEQ_LEN = 2048; LM_SAMPLE = 512; GRAD_ACCUM = 4; STEPS = 300; LR = 1e-4; ALIGN_WEIGHT = 10.0
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
model.set_attn_implementation("nsa_sparse")
model.to(device)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
n_nsa = attach_nsa(model, device, torch.float32)
model.print_trainable_parameters()
nsa_params=[p for nm,p in model.named_parameters() if ".nsa." in nm]
for p in nsa_params: p.requires_grad_(True)
print(f"attached NSA to {n_nsa} layers; NSA params {sum(p.numel() for p in nsa_params):,}", flush=True)
clm=model.base_model.model; backbone=clm.model; lm_head=clm.lm_head
ds=load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
def batches():
    buf=[]
    for ex in ds:
        buf.extend(tokenizer(ex["text"]).input_ids+[tokenizer.eos_token_id])
        while len(buf)>=SEQ_LEN:
            yield torch.tensor(buf[:SEQ_LEN]).unsqueeze(0); buf=buf[SEQ_LEN:]
def sampled_ce(hidden, ids, idx):
    logits=lm_head(hidden[:,idx]).float()
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]), ids[:,idx+1].reshape(-1))
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
scaler=torch.amp.GradScaler("cuda", enabled=device=="cuda")
model.train(); step=0; accum=0
for ids in batches():
    ids=ids.to(device); idx=torch.randperm(SEQ_LEN-1, device=device)[:LM_SAMPLE]
    with torch.no_grad(), model.disable_adapter():
        SPARSE_ENABLED=False
        teacher=backbone(input_ids=ids, output_hidden_states=True).hidden_states
    SPARSE_ENABLED=True
    s_out=backbone(input_ids=ids, output_hidden_states=True)
    align=0.0
    for hf,hs in zip(teacher, s_out.hidden_states):
        align=align+F.smooth_l1_loss(hs.float(), hf.float())
    align=align/len(teacher)
    lm=sampled_ce(s_out.last_hidden_state, ids, idx)
    loss=lm+ALIGN_WEIGHT*align
    scaler.scale(loss/GRAD_ACCUM).backward(); accum+=1
    if accum==GRAD_ACCUM:
        scaler.step(opt); scaler.update(); opt.zero_grad(); accum=0; step+=1
        if step%10==0: print(f"step {step}  lm {lm.item():.4f}  align {align.item():.4f}", flush=True)
        if step%100==0 or step==STEPS:
            model.save_pretrained(f"/kaggle/working/nsa2-qwen-lora-step{step}")
            torch.save({nm:p.detach().cpu() for nm,p in model.named_parameters() if ".nsa." in nm}, f"/kaggle/working/nsa2-modules-step{step}.pt")
        if step>=STEPS: break
print("done")
