"""CPU smoke test: builds a tiny random-weight Qwen2 model (no download) and checks
that the registered ssa_sparse attention runs, is causal-sane, degrades gracefully
vs full attention, AND that the chunked sparse path equals the dense reference mask.
Run:  python smoke_test.py
"""

import torch
import torch.nn.functional as F

from ssa_qwen import sparse_attention
from ssa_qwen.sparse_attention import register, build_sparse_mask
from transformers import Qwen2Config, Qwen2ForCausalLM

register()

# ---- 1) chunked sparse path must equal the dense reference oracle ----
torch.manual_seed(0)
b, h, q_len, d = 1, 4, 2048, 16  # > BLOCK*TOP_K so block selection engages
q = torch.randn(b, h, q_len, d)
k = torch.randn(b, h, q_len, d)
v = torch.randn(b, h, q_len, d)


class _Mod:
    num_key_value_groups = 1


sparse_attention.SPARSE_ENABLED = True
for chunk in (37, 512):  # tiny chunk vs single-pass: result must be identical
    sparse_attention.QUERY_CHUNK = chunk
    out_chunked = sparse_attention.ssa_sparse_attention(_Mod, q, k, v, None, scaling=1.0)[0]

    # dense reference
    allow = build_sparse_mask(q, k)
    bias = torch.where(allow, torch.zeros(()), torch.full((), torch.finfo(q.dtype).min))
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0).transpose(1, 2)

    diff = (out_chunked - ref).abs().max().item()
    print(f"chunk={chunk:>3}  max|chunked - dense_ref| = {diff:.2e}")
    assert diff < 1e-5, "chunked sparse path diverges from dense reference"
sparse_attention.QUERY_CHUNK = 512

# ---- 2) end-to-end in a tiny Qwen2: sparse vs full ----
config = Qwen2Config(
    vocab_size=1024, hidden_size=128, intermediate_size=256,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=2048,
)
torch.manual_seed(0)
model = Qwen2ForCausalLM(config)
model.set_attn_implementation("ssa_sparse")
model.eval()

ids = torch.randint(0, 1024, (1, 512))
with torch.no_grad():
    sparse_attention.SPARSE_ENABLED = True
    out_sparse = model(ids).logits
    sparse_attention.SPARSE_ENABLED = False
    out_full = model(ids).logits

print(f"sparse vs full mean |diff|: {(out_sparse - out_full).abs().mean():.4f}  "
      f"(logit scale {out_full.abs().mean():.4f})")
assert out_sparse.shape == (1, 512, 1024)
assert torch.isfinite(out_sparse).all(), "non-finite logits from sparse attention"

early = (out_sparse[:, :64] - out_full[:, :64]).abs().mean().item()
print(f"early-position |diff| (should be ~0): {early:.6f}")
assert early < 1e-4, "sparse attention diverges where it should equal full attention"

print("SMOKE TEST PASSED")
