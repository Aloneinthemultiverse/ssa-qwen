"""CPU smoke test for the NSA-style attention (compression + selection + window).
Builds a tiny random Qwen2, attaches the new NSA modules, runs both sparse and full
modes, and checks shapes/finiteness + that the new params exist and get gradients.
Run:  python smoke_test_nsa.py
"""

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from ssa_qwen import nsa_attention
from ssa_qwen.nsa_attention import register, attach_nsa, NSAModule

register()

config = Qwen2Config(
    vocab_size=1024, hidden_size=128, intermediate_size=256,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=4096,
)
torch.manual_seed(0)
model = Qwen2ForCausalLM(config)
model.set_attn_implementation("nsa_sparse")
n = attach_nsa(model)
print(f"attached NSA modules to {n} layers")

# count new params
nsa_params = [p for name, p in model.named_parameters() if ".nsa." in name]
print(f"new NSA params: {sum(p.numel() for p in nsa_params):,} across {len(nsa_params)} tensors")
assert nsa_params, "no NSA params found"

ids = torch.randint(0, 1024, (1, 2048))   # > L_CMP*TOP_K so compression+selection engage

# forward in sparse (NSA) mode
nsa_attention.SPARSE_ENABLED = True
out = model(ids, labels=ids)
print(f"sparse loss: {out.loss.item():.4f}")
assert torch.isfinite(out.logits).all(), "non-finite logits"
assert out.logits.shape == (1, 2048, 1024)

# gradients reach the new compression + gate params?
out.loss.backward()
grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in nsa_params)
print(f"all NSA params received finite grads: {grad_ok}")
assert grad_ok, "NSA params did not get gradients"

# full mode still runs
model.zero_grad()
nsa_attention.SPARSE_ENABLED = False
with torch.no_grad():
    full = model(ids).logits
assert torch.isfinite(full).all()

print("NSA SMOKE TEST PASSED")
