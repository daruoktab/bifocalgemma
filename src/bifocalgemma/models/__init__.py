"""Canonical BifocalGemma model modules.

Layout:
- `bridge.py`              SwiGLU dual-gate manifold bridge and normalized alignment.
- `cross_attention.py`     Decoupled full-rank cross-attention and cached inference.
- `embeddings.py`          Cross-component word embedding tying policy.
- `memory_efficient.py`    Tiled attention, joint visibility, and gradient checkpointing.
- `inference_cache.py`     Per-generation self KV and static cross KV cache.
- `flex_attention.py`      Structured FlexAttention kernels.
- `checkpoint_transfer.py` Text backbone weight transfer and local/adapter loaders.

Keep this module free of architecture imports so importing submodules stays lightweight.
"""

__all__: list[str] = []
