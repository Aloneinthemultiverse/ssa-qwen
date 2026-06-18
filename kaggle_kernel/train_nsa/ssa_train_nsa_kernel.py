"""Train NSA-Qwen: SSA-style alignment + the new NSA COMPRESSION branch.

Adds DeepSeek-NSA's compression branch (learnable block-pooling MLPs + gates) on
top of selection+window, attached to every Qwen attention layer, and trains those
NEW modules together with a LoRA adapter to match full attention (per-layer SmoothL1,
alpha=10). Qwen's own weights stay frozen. Single T4, seq 2048, no checkpointing.

Saves LoRA to /kaggle/working/nsa-qwen-lora-step{N} and the NSA module weights to
/kaggle/working/nsa-modules-step{N}.pt (needed at eval).
"""

import subprocess, sys, os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=5.0.0", "peft", "datasets", "accelerate", "torchao>=0.16.0"], check=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

SPARSE_ENABLED = True
SINK_TOKENS = 4
WINDOW = 128
BLOCK_SIZE = 32
L_CMP = 32
TOP_K_BLOCKS = 16
QUERY_CHUNK = 512


class NSAModule(nn.Module):
    def __init__(self, head_dim, l_cmp=L_CMP):
        super().__init__()
        self.d = head_dim; self.l_cmp = l_cmp
        self.pos = nn.Parameter(torch.zeros(l_cmp, head_dim))
        self.compress_k = nn.Linear(l_cmp * head_dim, head_dim, bias=False)
        self.compress_v = nn.Linear(l_cmp * head_dim, head_dim, bias=False)
        self.gate = nn.Linear(head_dim, 3)
        nn.init.normal_(self.compress_k.weight, std=0.02)
        nn.init.normal_(self.compress_v.weight, std=0.02)
        with torch.no_grad():
            self.gate.bias.copy_(torch.tensor([-4.0, 4.0, 4.0]))

    def compress(self, k, v):
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


def attach_nsa(model, device, dtype):
    layers = _find_layers(model)
    n = 0
    for layer in layers:
        attn = layer.self_attn
        d = getattr(attn, "head_dim", None) or attn.q_proj.out_features // attn.config.num_attention_heads
        attn.nsa = NSAModule(d).to(device=device, dtype=dtype)
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
    ck, cv, n_cmp = nsa.compress(key, value)
    ck = ck.to(query.dtype); cv = cv.to(query.dtype)   # NSA params are fp32 (AMP); cast back
    cmp_block_end = (torch.arange(n_cmp, device=device) + 1) * L_CMP - 1
    k_pos = torch.arange(kv_len, device=device)
    neg = torch.finfo(query.dtype).min
    out = torch.empty_like(query)
    base = kv_len - q_len
    for s in range(0, q_len, QUERY_CHUNK):
        e = min(s + QUERY_CHUNK, q_len)
        q = query[:, :, s:e]; qc = e - s
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

        g = torch.sigmoid(nsa.gate(q.float())).to(query.dtype)   # gate is fp32
        out[:, :, s:e] = g[..., 0:1] * cmp_out + (g[..., 1:2] + g[..., 2:3]) * 0.5 * sw_out
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

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
n_nsa = attach_nsa(model, device, torch.float32)  # NEW trainable modules in fp32 (AMP-safe)
model.print_trainable_parameters()
nsa_params = [p for nm, p in model.named_parameters() if ".nsa." in nm]
for p in nsa_params:
    p.requires_grad_(True)
print(f"attached NSA to {n_nsa} layers; NSA params {sum(p.numel() for p in nsa_params):,}", flush=True)

clm = model.base_model.model
backbone = clm.model
lm_head = clm.lm_head

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

def batches():
    buf = []
    for ex in ds:
        buf.extend(tokenizer(ex["text"]).input_ids + [tokenizer.eos_token_id])
        while len(buf) >= SEQ_LEN:
            yield torch.tensor(buf[:SEQ_LEN]).unsqueeze(0); buf = buf[SEQ_LEN:]

def sampled_ce(hidden, ids, idx):
    logits = lm_head(hidden[:, idx]).float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, idx + 1].reshape(-1))

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
model.train()

step, accum = 0, 0
for ids in batches():
    ids = ids.to(device)
    idx = torch.randperm(SEQ_LEN - 1, device=device)[:LM_SAMPLE]
    # teacher: full attention, LoRA disabled, no grad (NSA modules unused on full path)
    with torch.no_grad(), model.disable_adapter():
        SPARSE_ENABLED = False
        teacher = backbone(input_ids=ids, output_hidden_states=True).hidden_states
    # student: NSA sparse + LoRA + compression modules
    SPARSE_ENABLED = True
    s_out = backbone(input_ids=ids, output_hidden_states=True)
    align = 0.0
    for hf, hs in zip(teacher, s_out.hidden_states):
        align = align + F.smooth_l1_loss(hs.float(), hf.float())
    align = align / len(teacher)
    lm = sampled_ce(s_out.last_hidden_state, ids, idx)
    loss = lm + ALIGN_WEIGHT * align
    scaler.scale(loss / GRAD_ACCUM).backward()
    accum += 1
    if accum == GRAD_ACCUM:
        scaler.step(opt); scaler.update(); opt.zero_grad(); accum = 0
        step += 1
        if step % 10 == 0:
            print(f"step {step}  lm {lm.item():.4f}  align {align.item():.4f}", flush=True)
        if step % 100 == 0 or step == STEPS:
            model.save_pretrained(f"/kaggle/working/nsa-qwen-lora-step{step}")
            torch.save({nm: p.detach().cpu() for nm, p in model.named_parameters() if ".nsa." in nm},
                       f"/kaggle/working/nsa-modules-step{step}.pt")
        if step >= STEPS:
            break
print("done")
