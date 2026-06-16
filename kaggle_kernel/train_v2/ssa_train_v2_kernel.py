"""SSA-Qwen v2 training -- adopts the real SSA paper's recipe (arXiv:2511.20102):

  #1-3 architecture: block 32 x top-32 (RF~1024), window 128 (in attention below)
  #4 bidirectional alignment: sparsity loss ||h_full - sg(h_sparse)|| +
     commitment loss ||h_sparse - sg(h_full)||  (both paths share the LoRA, both trainable)
  #5 per-LAYER alignment (all hidden states), not just the final one
  #6 SmoothL1 loss, weight alpha=10 (paper default)
  #7 approximated: instead of random main-path, we run both paths every step and
     apply CE to both to keep them grounded (note: deviation from paper's prob-0.5 scheme)
  #8 trained under YaRN x4 RoPE scaling (positions match the extended-context eval)

Saves to /kaggle/working/ssa-qwen-lora-v2-step{N}.
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=5.0.0", "peft", "datasets", "accelerate", "torchao>=0.16.0"], check=True)

import torch
import torch.nn.functional as F

SPARSE_ENABLED = True
SINK_TOKENS = 4
WINDOW = 128
BLOCK_SIZE = 32
TOP_K_BLOCKS = 32
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

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL = "Qwen/Qwen2.5-0.5B"
SEQ_LEN = 8192
LM_SAMPLE = 1024
GRAD_ACCUM = 4
STEPS = 250
LR = 1e-4
ALIGN_WEIGHT = 10.0  # paper default alpha

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# #8: train under the same YaRN x4 scaling used at extended-context eval
config = AutoConfig.from_pretrained(MODEL)
if getattr(config, "rope_theta", None) is None:
    config.rope_theta = 1000000.0
config.rope_scaling = {"rope_type": "yarn", "factor": 4.0,
                       "original_max_position_embeddings": 32768}
config.max_position_embeddings = 131072

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, config=config, dtype=dtype)
model.set_attn_implementation("ssa_sparse")
model.to(device)

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                  task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()

clm = model.base_model.model
backbone = clm.model
lm_head = clm.lm_head

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

def batches():
    buf = []
    for ex in ds:
        buf.extend(tokenizer(ex["text"]).input_ids + [tokenizer.eos_token_id])
        while len(buf) >= SEQ_LEN:
            yield torch.tensor(buf[:SEQ_LEN]).unsqueeze(0)
            buf = buf[SEQ_LEN:]


def sampled_ce(hidden, ids, idx):
    logits = lm_head(hidden[:, idx]).float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, idx + 1].reshape(-1))


opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
model.train()

step, accum = 0, 0
for ids in batches():
    ids = ids.to(device)
    idx = torch.randperm(SEQ_LEN - 1, device=device)[:LM_SAMPLE]

    # both paths share the LoRA and are trainable (paper's bidirectional setup)
    SPARSE_ENABLED = False
    full_hs = backbone(input_ids=ids, output_hidden_states=True).hidden_states
    SPARSE_ENABLED = True
    sparse_hs = backbone(input_ids=ids, output_hidden_states=True).hidden_states

    # #4,#5,#6: bidirectional, per-layer, SmoothL1
    align = 0.0
    for hf, hs in zip(full_hs, sparse_hs):
        align = align + F.smooth_l1_loss(hf.float(), hs.detach().float()) \
                      + F.smooth_l1_loss(hs.float(), hf.detach().float())
    align = align / len(full_hs)

    lm = sampled_ce(sparse_hs[-1], ids, idx) + sampled_ce(full_hs[-1], ids, idx)
    loss = lm + ALIGN_WEIGHT * align

    scaler.scale(loss / GRAD_ACCUM).backward()
    accum += 1
    if accum == GRAD_ACCUM:
        scaler.step(opt); scaler.update(); opt.zero_grad(); accum = 0
        step += 1
        if step % 10 == 0:
            print(f"step {step}  lm {lm.item():.4f}  align {align.item():.4f}", flush=True)
        if step % 50 == 0 or step == STEPS:
            model.save_pretrained(f"/kaggle/working/ssa-qwen-lora-v2-step{step}")
        if step >= STEPS:
            break

print("done")
