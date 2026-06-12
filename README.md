# SSA-Qwen: Sparse-Attention Retrofit of Qwen2.5-0.5B

A 500M-parameter long-context MVP built on $0: [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
with its attention replaced by an SSA-inspired sparse pattern
([SSA: Sparse Sparse Attention, arXiv:2511.20102](https://arxiv.org/pdf/2511.20102)),
then aligned to full-attention behavior with a LoRA fine-tune on free Kaggle GPUs.

## How it works

**Sparse pattern** (per query token): attention sink (first 4 tokens) + sliding
window (256 tokens) + top-8 dynamically selected key blocks (block size 64,
scored by query · mean-pooled-block-key). Implemented as a custom attention
function registered through transformers' `AttentionInterface` — no forked
modeling code. Currently a mask simulation (dense mask, sparse pattern), which
reproduces the modeling behavior; a fused kernel would be the production version.

**Alignment training** (the SSA idea): the same weights serve as teacher (full
attention, adapters off, no grad) and student (sparse attention + LoRA).
Loss = LM loss + MSE between final hidden states. The student learns to make
sparse attention behave like full attention.

## Files

- `ssa_qwen/sparse_attention.py` — the sparse attention module
- `smoke_test.py` — CPU test on a tiny random Qwen2 (no download needed)
- `train_kaggle.py` — LoRA alignment fine-tune (run on Kaggle T4)
- `eval_niah.py` — needle-in-a-haystack eval grid (sparse vs full vs trained)

## Run

```bash
pip install torch transformers peft datasets accelerate
python smoke_test.py                 # verify the attention module (CPU, <1 min)
python eval_niah.py --full           # baseline: full attention
python eval_niah.py                  # untrained sparse (expect degradation)
python train_kaggle.py               # on Kaggle GPU
python eval_niah.py --adapter ssa-qwen-lora-step1000   # trained sparse
```

## Results

*(to be filled in after training)*

| Mode | NIAH @2k | @4k | @8k |
|---|---|---|---|
| Full attention (baseline) | | | |
| Sparse, untrained | | | |
| Sparse + SSA alignment | | | |
