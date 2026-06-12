"""CPU smoke test: builds a tiny random-weight Qwen2 model (no download) and checks
that the registered ssa_sparse attention runs, is causal-sane, and degrades
gracefully vs full attention. Run:  python smoke_test.py
"""

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from ssa_qwen import sparse_attention
from ssa_qwen.sparse_attention import register

register()

config = Qwen2Config(
    vocab_size=1024,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
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

diff = (out_sparse - out_full).abs().mean().item()
scale = out_full.abs().mean().item()
print(f"sparse vs full mean |diff|: {diff:.4f}  (logit scale {scale:.4f})")

assert out_sparse.shape == (1, 512, 1024)
assert torch.isfinite(out_sparse).all(), "non-finite logits from sparse attention"

# With window=256 on a 512-token sequence, early positions see identical context
# under both modes -> outputs there should match closely.
early = (out_sparse[:, :64] - out_full[:, :64]).abs().mean().item()
print(f"early-position |diff| (should be ~0): {early:.6f}")
assert early < 1e-4, "sparse attention diverges where it should equal full attention"

print("SMOKE TEST PASSED")
