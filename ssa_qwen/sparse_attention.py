"""SSA-inspired sparse attention for Qwen2.5, pluggable via transformers' AttentionInterface.

Sparsity pattern per query token:
  - attention sink: first SINK_TOKENS tokens always visible
  - sliding window: last WINDOW tokens always visible
  - top-k blocks: keys are chunked into blocks; each query also attends to the
    TOP_K_BLOCKS blocks whose mean-pooled key has the highest dot product with it.

This is a *mask simulation* of sparse attention (the mask is dense, so memory is
still O(N^2)); it reproduces the modeling behavior of SSA-style sparsity, which is
what we need for the alignment fine-tune and the NIAH eval. A fused sparse kernel
would be the production version.
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


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(b, kv_heads, s, d) -> (b, kv_heads*n_rep, s, d)"""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def build_sparse_mask(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Boolean mask (b, h, q_len, kv_len): True = attend. Causal throughout."""
    b, h, q_len, d = query.shape
    kv_len = key.shape[2]
    device = query.device

    q_pos = torch.arange(kv_len - q_len, kv_len, device=device)  # absolute positions
    k_pos = torch.arange(kv_len, device=device)
    causal = k_pos[None, :] <= q_pos[:, None]                    # (q_len, kv_len)

    sink = k_pos[None, :] < SINK_TOKENS
    window = (q_pos[:, None] - k_pos[None, :]) < WINDOW
    static = (sink | window) & causal                            # (q_len, kv_len)
    mask = static[None, None].expand(b, h, q_len, kv_len).clone()

    # Top-k block selection on mean-pooled keys.
    n_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    if n_blocks > TOP_K_BLOCKS:
        pad = n_blocks * BLOCK_SIZE - kv_len
        k_padded = F.pad(key, (0, 0, 0, pad))
        block_keys = k_padded.view(b, h, n_blocks, BLOCK_SIZE, d).mean(dim=3)  # (b,h,nb,d)
        scores = torch.einsum("bhqd,bhnd->bhqn", query, block_keys)            # (b,h,q,nb)

        # A block is selectable only if fully in the causal past of the query.
        block_end = (torch.arange(n_blocks, device=device) + 1) * BLOCK_SIZE - 1
        sel_ok = block_end[None, :] <= q_pos[:, None]                          # (q_len, nb)
        scores = scores.masked_fill(~sel_ok[None, None], float("-inf"))

        k_sel = min(TOP_K_BLOCKS, n_blocks)
        top = scores.topk(k_sel, dim=-1).indices                               # (b,h,q,k)
        block_mask = torch.zeros(b, h, q_len, n_blocks, dtype=torch.bool, device=device)
        block_mask.scatter_(-1, top, True)
        # Queries with no valid blocks (early positions) pick garbage indices; the
        # final causal AND below makes that harmless.
        token_mask = block_mask.repeat_interleave(BLOCK_SIZE, dim=-1)[..., :kv_len]
        mask |= token_mask & causal[None, None]

    return mask


def ssa_sparse_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Signature required by transformers' AttentionInterface."""
    n_rep = getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1])
    key = _repeat_kv(key, n_rep)
    value = _repeat_kv(value, n_rep)

    if SPARSE_ENABLED:
        allow = build_sparse_mask(query, key)
        bias = torch.zeros_like(allow, dtype=query.dtype)
        bias.masked_fill_(~allow, torch.finfo(query.dtype).min)
        if attention_mask is not None:  # padding mask from the model
            bias = bias + attention_mask[..., : key.shape[2]].to(bias.dtype)
        attn = F.scaled_dot_product_attention(
            query, key, value, attn_mask=bias, dropout_p=dropout, scale=scaling
        )
    else:
        if attention_mask is not None:
            full_mask = attention_mask[..., : key.shape[2]]
        else:
            # explicit bottom-right-aligned causal mask: is_causal aligns top-left
            # when q_len < kv_len (decoding), which would blind the model
            q_len, kv_len = query.shape[2], key.shape[2]
            q_pos = torch.arange(kv_len - q_len, kv_len, device=query.device)
            k_pos = torch.arange(kv_len, device=query.device)
            allow = k_pos[None, :] <= q_pos[:, None]
            full_mask = torch.zeros(q_len, kv_len, dtype=query.dtype, device=query.device)
            full_mask.masked_fill_(~allow, torch.finfo(query.dtype).min)
        attn = F.scaled_dot_product_attention(
            query, key, value, attn_mask=full_mask, dropout_p=dropout, scale=scaling,
        )

    return attn.transpose(1, 2).contiguous(), None


def register():
    from transformers.modeling_utils import AttentionInterface
    AttentionInterface.register("ssa_sparse", ssa_sparse_attention)
