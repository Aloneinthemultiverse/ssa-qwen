"""NSA-style sparse attention for Qwen: adds the COMPRESSION branch (the piece our
SSA model lacked), combined with selection + sliding window, gated.

Three branches per query (DeepSeek NSA, arXiv:2502.11089), adapted to retrofit onto
a frozen Qwen via small new trainable modules:
  - compression: each block of L_CMP keys/values is pooled by a small learnable
    MLP (with intra-block position embedding) into ONE summary token; the query
    attends over these summaries -> cheap coarse global view.
  - selection: top-n blocks by the (reused) compression scores, attended in full.
  - sliding window: the most recent W tokens (+ a few sink tokens).
Outputs blended by learned per-head gates (sigmoid).

The compression MLPs + gate are NEW parameters (class NSAModule), attached to each
Qwen attention layer and trained alongside a LoRA adapter. Qwen's own weights stay
frozen. This is a dense-bias simulation for the selection/window branch (memory via
query chunking) plus a true small attention for the compression branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SPARSE_ENABLED = True
SINK_TOKENS = 4
WINDOW = 128
BLOCK_SIZE = 32          # selection block size
L_CMP = 32               # compression block size
TOP_K_BLOCKS = 16        # selected blocks (NSA uses 16)
QUERY_CHUNK = 512


class NSAModule(nn.Module):
    """New trainable compression + gating, one per attention layer. Operates on
    head_dim d; weights shared across heads."""

    def __init__(self, head_dim: int, l_cmp: int = L_CMP):
        super().__init__()
        self.d = head_dim
        self.l_cmp = l_cmp
        self.pos = nn.Parameter(torch.zeros(l_cmp, head_dim))      # intra-block pos emb
        self.compress_k = nn.Linear(l_cmp * head_dim, head_dim, bias=False)
        self.compress_v = nn.Linear(l_cmp * head_dim, head_dim, bias=False)
        self.gate = nn.Linear(head_dim, 3)                         # cmp / slc / win
        # start near a plain weighted-mean so the module is benign before training
        nn.init.normal_(self.compress_k.weight, std=0.02)
        nn.init.normal_(self.compress_v.weight, std=0.02)
        # start with compression OFF (gate~0) and selection/window ON (gate~1) so the
        # model begins ~= the working selection+window model, then learns compression
        with torch.no_grad():
            self.gate.bias.copy_(torch.tensor([-4.0, 4.0, 4.0]))

    def compress(self, k, v):
        """(b,h,kv,d) -> compressed (b,h,n_blocks,d) for k and v, plus n_blocks."""
        b, h, kv, d = k.shape
        nb = (kv + self.l_cmp - 1) // self.l_cmp
        pad = nb * self.l_cmp - kv
        kp = F.pad(k, (0, 0, 0, pad))
        vp = F.pad(v, (0, 0, 0, pad))
        kb = kp.view(b, h, nb, self.l_cmp, d) + self.pos          # add intra-block pos
        vb = vp.view(b, h, nb, self.l_cmp, d)
        ck = self.compress_k(kb.reshape(b, h, nb, self.l_cmp * d))
        cv = self.compress_v(vb.reshape(b, h, nb, self.l_cmp * d))
        return ck, cv, nb


def attach_nsa(model):
    """Attach an NSAModule to every Qwen attention layer (so its params train/save).
    Accepts a Qwen2ForCausalLM, a (possibly PEFT-wrapped) variant, or a Qwen2Model."""
    m = model
    for attr in ("base_model", "model"):
        while hasattr(m, attr) and not hasattr(m, "layers"):
            m = getattr(m, attr)
    layers = m.layers
    n = 0
    for layer in layers:
        attn = layer.self_attn
        d = getattr(attn, "head_dim", attn.q_proj.out_features // attn.config.num_attention_heads)
        attn.nsa = NSAModule(d).to(next(attn.parameters()).device,
                              next(attn.parameters()).dtype)
        n += 1
    return n


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def nsa_sparse_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    n_rep = getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1])
    key = _repeat_kv(key, n_rep)
    value = _repeat_kv(value, n_rep)
    b, h, q_len, d = query.shape
    kv_len = key.shape[2]
    device = query.device
    scale = scaling if scaling is not None else d ** -0.5

    if not SPARSE_ENABLED:
        if q_len == kv_len:
            attn = F.scaled_dot_product_attention(query, key, value, is_causal=True,
                                                  dropout_p=dropout, scale=scaling)
        else:
            attn = F.scaled_dot_product_attention(query, key, value, attn_mask=None,
                                                  dropout_p=dropout, scale=scaling)
        return attn.transpose(1, 2).contiguous(), None

    nsa: NSAModule = module.nsa
    ck, cv, n_cmp = nsa.compress(key, value)                 # (b,h,n_cmp,d)
    cmp_block_end = (torch.arange(n_cmp, device=device) + 1) * L_CMP - 1
    k_pos = torch.arange(kv_len, device=device)
    neg = torch.finfo(query.dtype).min

    out = torch.empty_like(query)
    base = kv_len - q_len
    for s in range(0, q_len, QUERY_CHUNK):
        e = min(s + QUERY_CHUNK, q_len)
        q = query[:, :, s:e]
        qc = e - s
        q_abs = torch.arange(base + s, base + e, device=device)

        # ---- compression branch: attend over block summaries (causal at block level)
        cmp_scores = torch.einsum("bhqd,bhnd->bhqn", q, ck) * scale       # (b,h,qc,n_cmp)
        cmp_ok = cmp_block_end[None, :] <= q_abs[:, None]                 # (qc,n_cmp)
        cmp_scores = cmp_scores.masked_fill(~cmp_ok[None, None], neg)
        cmp_p = cmp_scores.softmax(dim=-1)
        cmp_out = torch.einsum("bhqn,bhnd->bhqd", cmp_p, cv)             # (b,h,qc,d)

        # ---- selection + window + sink: dense-bias attention over full keys
        causal = k_pos[None, :] <= q_abs[:, None]
        sink = k_pos[None, :] < SINK_TOKENS
        window = (q_abs[:, None] - k_pos[None, :]) < WINDOW
        allow = ((sink | window) & causal)[None, None].expand(b, h, qc, kv_len).clone()
        # reuse compression scores to pick top blocks (NSA's free-selection trick)
        k_sel = min(TOP_K_BLOCKS, n_cmp)
        top = cmp_scores.topk(k_sel, dim=-1).indices                    # (b,h,qc,k)
        blk = torch.zeros(b, h, qc, n_cmp, dtype=torch.bool, device=device)
        blk.scatter_(-1, top, True)
        tok = blk.repeat_interleave(L_CMP, dim=-1)[..., :kv_len]
        allow |= tok & causal[None, None]
        bias = torch.where(allow, torch.zeros((), dtype=query.dtype, device=device),
                           torch.full((), neg, dtype=query.dtype, device=device))
        sw_out = F.scaled_dot_product_attention(q, key, value, attn_mask=bias,
                                                dropout_p=dropout, scale=scaling)  # (b,h,qc,d)

        # ---- gated blend (cmp / selection / window share the selection output here:
        # selection+window are merged in sw_out; gate weights cmp vs sw, with the 3rd
        # gate folded into sw to keep one masked pass)
        g = torch.sigmoid(nsa.gate(q))                                  # (b,h,qc,3)
        g_cmp = g[..., 0:1]
        g_sw = (g[..., 1:2] + g[..., 2:3]) * 0.5
        out[:, :, s:e] = g_cmp * cmp_out + g_sw * sw_out

    return out.transpose(1, 2).contiguous(), None


def register():
    from transformers.modeling_utils import AttentionInterface
    AttentionInterface.register("nsa_sparse", nsa_sparse_attention)
