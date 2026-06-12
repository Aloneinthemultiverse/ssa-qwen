"""SSA-style alignment fine-tune of Qwen2.5-0.5B with LoRA. Run on Kaggle (T4).

Idea (from the SSA paper, arXiv:2511.20102): train the model under SPARSE
attention so its outputs align with what FULL attention produces, instead of
only minimizing LM loss. Teacher = same weights with adapters disabled + full
attention (no grad). Student = LoRA adapters + sparse attention.

loss = lm_loss + ALIGN_WEIGHT * MSE(student_hidden, teacher_hidden)

Kaggle setup: pip install -U transformers peft datasets accelerate, then upload
the ssa_qwen/ package alongside this script (or pip install from your GitHub).
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ssa_qwen import sparse_attention
from ssa_qwen.sparse_attention import register

MODEL = "Qwen/Qwen2.5-0.5B"
SEQ_LEN = 2048
BATCH = 1
GRAD_ACCUM = 8
STEPS = 1000
LR = 1e-4
ALIGN_WEIGHT = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

register()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
model.set_attn_implementation("ssa_sparse")
model.to(device)

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# Long-form text so the sparse pattern actually matters at SEQ_LEN.
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

def batches():
    buf = []
    for ex in ds:
        buf.extend(tokenizer(ex["text"]).input_ids + [tokenizer.eos_token_id])
        while len(buf) >= SEQ_LEN:
            yield torch.tensor(buf[:SEQ_LEN]).unsqueeze(0)
            buf = buf[SEQ_LEN:]

opt = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()

step, accum = 0, 0
for ids in batches():
    ids = ids.to(device)

    # Teacher: full attention, no adapters, no grad.
    with torch.no_grad(), model.disable_adapter():
        sparse_attention.SPARSE_ENABLED = False
        teacher = model(ids, output_hidden_states=True).hidden_states[-1]

    # Student: sparse attention + LoRA.
    sparse_attention.SPARSE_ENABLED = True
    out = model(ids, labels=ids, output_hidden_states=True)
    align = torch.nn.functional.mse_loss(out.hidden_states[-1], teacher)
    loss = out.loss + ALIGN_WEIGHT * align

    (loss / GRAD_ACCUM).backward()
    accum += 1
    if accum == GRAD_ACCUM:
        opt.step(); opt.zero_grad(); accum = 0
        step += 1
        if step % 10 == 0:
            print(f"step {step}  lm {out.loss.item():.4f}  align {align.item():.4f}")
        if step % 200 == 0 or step == STEPS:
            model.save_pretrained(f"ssa-qwen-lora-step{step}")
        if step >= STEPS:
            break

print("done")
