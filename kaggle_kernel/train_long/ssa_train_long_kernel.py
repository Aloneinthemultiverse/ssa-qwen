"""Long-context SSA alignment fine-tune: train the adapter at 8192 tokens (vs the
original 2048) to test whether matching train length to test length improves how
much of the long-context sparsity gap the adapter recovers.

Memory tricks to fit 8k training on a 16GB T4:
  - gradient checkpointing (cuts stored activations)
  - alignment loss read from the backbone's last hidden state (no full-logits tensor)
  - LM loss computed on a random 2048-position subset (caps the 152k-vocab logits)

Saves adapter to /kaggle/working/ssa-qwen-lora-8k-step{N}.
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=5.0.0", "peft", "datasets", "accelerate", "torchao>=0.16.0"], check=True)

import torch
import torch.nn.functional as F

# ----------------------------- sparse attention (chunked) -------------------

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

# ------------------------------- training -----------------------------------

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
SEQ_LEN = 8192
LM_SAMPLE = 2048      # positions scored for LM loss (caps logits memory)
GRAD_ACCUM = 4
STEPS = 300
LR = 1e-4
ALIGN_WEIGHT = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
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

opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
model.train()

step, accum = 0, 0
for ids in batches():
    ids = ids.to(device)

    # teacher: full attention, adapters off, no grad
    with torch.no_grad(), model.disable_adapter():
        SPARSE_ENABLED = False
        teacher = backbone(input_ids=ids).last_hidden_state.detach()

    # student: sparse + LoRA
    SPARSE_ENABLED = True
    student = backbone(input_ids=ids).last_hidden_state
    align = F.mse_loss(student.float(), teacher.float())

    # LM loss on a random subset of positions (bounds logits memory)
    idx = torch.randperm(SEQ_LEN - 1, device=device)[:LM_SAMPLE]
    logits = lm_head(student[:, idx]).float()
    tgt = ids[:, idx + 1]
    lm = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))

    loss = lm + ALIGN_WEIGHT * align
    scaler.scale(loss / GRAD_ACCUM).backward()
    accum += 1
    if accum == GRAD_ACCUM:
        scaler.step(opt); scaler.update(); opt.zero_grad(); accum = 0
        step += 1
        if step % 10 == 0:
            print(f"step {step}  lm {lm.item():.4f}  align {align.item():.4f}", flush=True)
        if step % 100 == 0 or step == STEPS:
            model.save_pretrained(f"/kaggle/working/ssa-qwen-lora-8k-step{step}")
        if step >= STEPS:
            break

print("done")
