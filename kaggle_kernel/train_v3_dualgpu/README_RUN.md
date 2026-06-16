# Full SSA recipe (#4 bidirectional + #5 per-layer + #7) — needs GPU T4 x2

This version drops gradient checkpointing (so the global-flag/recompute bug that
blocked bidirectional + per-layer alignment disappears) and instead relies on the
combined 32 GB of two T4s by sharding the model across both GPUs.

## How to run (manual, browser — the API cannot enable T4 x2)
1. Go to kaggle.com -> Create -> New Notebook.
2. File -> Import Notebook, or paste `ssa_train_v3_dualgpu.py` into a code cell.
3. Settings (right panel):
   - Accelerator -> **GPU T4 x2**
   - Internet -> **On**
4. Run all. Saves adapter to /kaggle/working/ssa-qwen-lora-v3-step{N}.

Only use this if the v2 (single-GPU) results show retrieval is still weak and we
decide the bidirectional/per-layer refinements are worth it.
