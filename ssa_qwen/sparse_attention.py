"""SSA-inspired sparse attention for Qwen2.5, pluggable via transformers' AttentionInterface.

Sparsity pattern per query token:
  - attention sink: first SINK_TOKENS tokens always visible
  - sliding window: last WINDOW tokens always visible
  - top-k blocks: keys are chunked into blocks; each query also attends to the
    TOP_K_BLOCKS blocks whose mean-pooled key has the highest dot product with it.

Memory: the attention bias is built one QUERY CHUNK at a time (CHUNK rows), so peak
mask memory is O(CHUNK * kv_len) instead of O(q_len * kv_len). This lets the
simulation run at 32k+ context on a 16GB GPU. It is still a *mask simulation* (each
chunk materializes a dense bias over all keys), not a fused gather kernel, so it
reproduces the modeling behavior and scales in memory, but does not yet harvest the
FLOP savings — that needs a real sparse kernel.
"""

import torch
import torch.nn.functional as F

# Global toggle: the alignment trainer flips this to compare full vs sparse
# attention on the same model weights.
SPARSE_ENABLED = True

SINK_TOKENS = 4
WINDOW = 256
BLOCK_SIZE = 64
TOP_K_BLOCKS = 8
QUERY_CHUNK = 512  # rows of the attention bias built at once


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(b, kv_heads, s, d) -> (b, kv_heads*n_rep, s, d)"""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def build_sparse_mask(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Reference (dense, O(N^2)) boolean mask (b, h, q_len, kv_len): True = attend.

    Kept as a correctness oracle for the smoke test; the live path uses the chunked
    bias builder below, which must agree with this for every query row.
    """
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


def _chunk_bias(q, q_abs, key, block_keys, kv_len, k_pos, scaling):
    """Additive attention bias (b, h, qc, kv_len) for one query chunk.

    q:          (b, h, qc, d)
    q_abs:      (qc,) absolute positions of these queries
    block_keys: (b, h, n_blocks, d) mean-pooled keys, or None if too few blocks
    """
    b, h, qc, d = q.shape
    device = q.device
    neg = torch.finfo(q.dtype).min

    causal = k_pos[None, :] <= q_abs[:, None]                       # (qc, kv_len)
    sink = k_pos[None, :] < SINK_TOKENS
    window = (q_abs[:, None] - k_pos[None, :]) < WINDOW
    allow = (sink | window) & causal                               # (qc, kv_len)
    allow = allow[None, None].expand(b, h, qc, kv_len).clone()

    if block_keys is not None:
        n_blocks = block_keys.shape[2]
        scores = torch.einsum("bhqd,bhnd->bhqn", q, block_keys)    # (b,h,qc,nb)
        block_end = (torch.arange(n_blocks, device=device) + 1) * BLOCK_SIZE - 1
        sel_ok = block_end[None, :] <= q_abs[:, None]              # (qc, nb)
        scores = scores.masked_fill(~sel_ok[None, None], neg)
        k_sel = min(TOP_K_BLOCKS, n_blocks)
        top = scores.topk(k_sel, dim=-1).indices                   # (b,h,qc,k)
        block_mask = torch.zeros(b, h, qc, n_blocks, dtype=torch.bool, device=device)
        block_mask.scatter_(-1, top, True)
        token_mask = block_mask.repeat_interleave(BLOCK_SIZE, dim=-1)[..., :kv_len]
        allow |= token_mask & causal[None, None]

    bias = torch.where(allow, torch.zeros((), dtype=q.dtype, device=device),
                       torch.full((), neg, dtype=q.dtype, device=device))
    return bias


def ssa_sparse_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Signature required by transformers' AttentionInterface."""
    n_rep = getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1])
    key = _repeat_kv(key, n_rep)
    value = _repeat_kv(value, n_rep)

    b, h, q_len, d = query.shape
    kv_len = key.shape[2]
    device = query.device

    # NOTE: this project never feeds padded batches (training packs sequences;
    # eval runs one sequence at a time), and our sparse bias already enforces
    # causality. So we deliberately ignore the model's attention_mask here and
    # rely on is_causal / our own bias. This avoids materializing the model's
    # dense 4D causal mask (~2GB at 32k) and keeps long-context runs in memory.

    if not SPARSE_ENABLED:
        if q_len == kv_len:
            # prefill: flash causal kernel, no mask materialized (cheap at 32k)
            attn = F.scaled_dot_product_attention(
                query, key, value, is_causal=True, dropout_p=dropout, scale=scaling)
        else:
            # decode (q_len==1): every cached key is in the past, attend to all
            attn = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout, scale=scaling)
        return attn.transpose(1, 2).contiguous(), None

    # sparse path, chunked over the query dimension to bound mask memory
    k_pos = torch.arange(kv_len, device=device)
    n_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_keys = None
    if n_blocks > TOP_K_BLOCKS:
        pad = n_blocks * BLOCK_SIZE - kv_len
        k_padded = F.pad(key, (0, 0, 0, pad))
        block_keys = k_padded.view(b, h, n_blocks, BLOCK_SIZE, d).mean(dim=3)

    out = torch.empty_like(query)
    base = kv_len - q_len  # absolute position of the first query
    for s in range(0, q_len, QUERY_CHUNK):
        e = min(s + QUERY_CHUNK, q_len)
        q = query[:, :, s:e]
        q_abs = torch.arange(base + s, base + e, device=device)
        bias = _chunk_bias(q, q_abs, key, block_keys, kv_len, k_pos, scaling)
        out[:, :, s:e] = F.scaled_dot_product_attention(
            q, key, value, attn_mask=bias, dropout_p=dropout, scale=scaling)

    return out.transpose(1, 2).contiguous(), None


def register():
    from transformers.modeling_utils import AttentionInterface
    AttentionInterface.register("ssa_sparse", ssa_sparse_attention)
