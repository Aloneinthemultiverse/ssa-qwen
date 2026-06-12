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

**Findings.** The headline result is architectural: the sparse pattern
(4 sink tokens + 256-token window + top-8 of 64-token blocks ~= 13% of keys at
8k) preserves essentially all of full attention's behavior on this model
*out of the box* — perfect passkey retrieval and a +0.05 PPL cost. Alignment
training recovered ~80% of that (small) perplexity gap, consistent with the
SSA paper's direction, though the absolute gap is near the noise floor at this
scale. A stress test with a top-2 block budget degraded both sparse variants
similarly. Honest limitation: at 4-8k context on a 0.5B model, sparsity is
nearly free, so there is little gap for alignment training to close; the
interesting regime (32k+) needs a memory-efficient sparse kernel rather than
this dense-mask simulation.
