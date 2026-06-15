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

Training: 1000 steps of LoRA alignment on a Kaggle T4 (fp16, seq len 2048,
fineweb-edu). Alignment loss fell ~4x (0.22 -> 0.06); LM loss stable (~2.2-2.9).

**Passkey NIAH (1k-8k context, 5 depths, greedy decoding):**

| Mode | @1k | @2k | @4k | @8k |
|---|---|---|---|---|
| Full attention (baseline) | 100% | 100% | 100% | 100% |
| Sparse, untrained | 100% | 100% | 100% | 100% |
| Sparse + SSA alignment | 100% | 100% | 100% | 100% |

**Perplexity (20 x 4096-token fineweb-edu docs):**

| Mode | PPL |
|---|---|
| Full attention | 13.57 |
| Sparse, untrained | 13.62 |
| Sparse + SSA alignment | 13.58 |

**Long-context perplexity (8 docs/length; chunked-CE; memory-efficient chunked
sparse attention, verified bit-identical to the dense reference):**

| context | full | sparse (untrained) | gap | ours (aligned) | gap recovered |
|---|---|---|---|---|---|
| 8k  | 14.07 | 14.21 | 0.14 | 14.15 | 43% |
| 16k | 12.99 | 13.19 | 0.20 | 13.12 | 32% |
| 32k | 13.03 | 13.29 | 0.26 | 13.20 | 33% |

NIAH passkey retrieval stays 100% for all three modes through 32k (benchmark
saturated; the needle is too distinctive for top-k block selection to miss).

**Findings.**
1. *The sparsity penalty grows with context length.* The full-vs-sparse PPL gap
   rises monotonically 0.14 -> 0.20 -> 0.26 as context goes 8k -> 16k -> 32k.
   Near-noise at 8k, it is a clear, consistent gap by 32k.
2. *Cheap alignment generalizes beyond its training length.* The LoRA adapter was
   trained only at 2048 tokens, yet recovers ~1/3 of the penalty at 32k -- 16x
   longer than it ever saw in training.
3. *Passkey NIAH is the wrong long-context test here* -- it saturates at 100% for
   every mode; a multi-fact / QA task is needed to stress retrieval.

**Honest limitations.** Single 0.5B model, single seed, 8 docs/length (no error
bars yet). Absolute gaps are small (sub-0.3 PPL) -- this is a small model and the
adapter was trained short. The natural next experiment: retrain the adapter at
16k+ context and test whether recovery exceeds 33%. The attention is a chunked
*dense-bias simulation* (memory-efficient, scales to 32k) -- it harvests the
modeling behavior, not yet the FLOP savings; a fused gather kernel is the
production step.
