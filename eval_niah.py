"""Needle-in-a-haystack eval for the SSA-Qwen model.

Plants a random passkey at varying depths in contexts of varying length, asks
the model to retrieve it, and prints an accuracy grid (the classic NIAH
heatmap, as text). Run on Kaggle GPU; CPU works but is slow.

  python eval_niah.py                     # base Qwen, sparse attention
  python eval_niah.py --full              # base Qwen, full attention
  python eval_niah.py --adapter PATH      # your trained LoRA + sparse
"""

import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ssa_qwen import sparse_attention
from ssa_qwen.sparse_attention import register

MODEL = "Qwen/Qwen2.5-0.5B"
CONTEXT_LENS = [1024, 2048, 4096, 8192]
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
TRIALS = 3

FILLER = (
    "The sky was clear and the grass was green. People walked through the park "
    "and talked about the weather, the news, and their plans for the weekend. "
)

parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="use full attention")
parser.add_argument("--adapter", default=None, help="path to LoRA adapter")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

register()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
model.set_attn_implementation("ssa_sparse")
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.to(device).eval()
sparse_attention.SPARSE_ENABLED = not args.full

random.seed(42)
print(f"mode: {'FULL' if args.full else 'SPARSE'}  adapter: {args.adapter}")
print(f"{'ctx':>6} | " + " | ".join(f"d={d:.1f}" for d in DEPTHS))

for ctx_len in CONTEXT_LENS:
    row = []
    for depth in DEPTHS:
        hits = 0
        for _ in range(TRIALS):
            key = str(random.randint(10000, 99999))
            needle_ids = tokenizer(f" The secret passkey is {key}. Remember it. ").input_ids
            q_ids = tokenizer("\nQuestion: What is the secret passkey?\nAnswer: The secret passkey is").input_ids
            filler_ids = tokenizer(FILLER).input_ids
            budget = ctx_len - len(needle_ids) - len(q_ids)
            body = (filler_ids * (budget // len(filler_ids) + 1))[:budget]
            cut = int(budget * depth)
            ids = body[:cut] + needle_ids + body[cut:] + q_ids
            ids = torch.tensor(ids).unsqueeze(0).to(device)
            with torch.no_grad():
                gen = model.generate(ids, attention_mask=torch.ones_like(ids),
                                     max_new_tokens=8, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(gen[0, ids.shape[1]:])
            hits += key in answer
        row.append(hits / TRIALS)
    print(f"{ctx_len:>6} | " + " | ".join(f"{a:>5.0%}" for a in row))
