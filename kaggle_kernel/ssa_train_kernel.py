"""SSA-style alignment fine-tune of Qwen2.5-0.5B with LoRA — self-contained Kaggle kernel.

Combines ssa_qwen/sparse_attention.py and train_kaggle.py from
the ssa-qwen repo into one file so it can run as a Kaggle script kernel
with GPU enabled. See the repo README for the full story.
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=5.0.0", "peft", "datasets", "accelerate", "torchao>=0.16.0"], check=True)

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
        attn = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask[..., : key.shape[2]] if attention_mask is not None else None,
            dropout_p=dropout, scale=scaling, is_causal=attention_mask is None)
    return attn.transpose(1, 2).contiguous(), None


from transformers.modeling_utils import AttentionInterface
AttentionInterface.register("ssa_sparse", ssa_sparse_attention)

# ------------------------------- training -----------------------------------

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
SEQ_LEN = 2048
GRAD_ACCUM = 8
STEPS = 1000
LR = 1e-4
ALIGN_WEIGHT = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"
# float16 (not bfloat16): Kaggle T4s are sm_75, no native bf16 tensor cores
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

    with torch.no_grad(), model.disable_adapter():
        SPARSE_ENABLED = False
        teacher = model(ids, output_hidden_states=True).hidden_states[-1]

    SPARSE_ENABLED = True
    out = model(ids, labels=ids, output_hidden_states=True)
    align = F.mse_loss(out.hidden_states[-1].float(), teacher.float())
    loss = out.loss + ALIGN_WEIGHT * align

    scaler.scale(loss / GRAD_ACCUM).backward()
    accum += 1
    if accum == GRAD_ACCUM:
        scaler.step(opt); scaler.update(); opt.zero_grad(); accum = 0
        step += 1
        if step % 10 == 0:
            print(f"step {step}  lm {out.loss.item():.4f}  align {align.item():.4f}", flush=True)
        if step % 200 == 0 or step == STEPS:
            model.save_pretrained(f"/kaggle/working/ssa-qwen-lora-step{step}")
        if step >= STEPS:
            break

print("done")
