"""Shared model runtime plus the two architecture generations.

Layout:
- `bifocalgemma.v3`            production V3 architecture and training stack.
- `bifocalgemma.models.v2`     legacy V2 architecture (merged attention, linear bridge).
- `memory_efficient.py`        tiling + gradient checkpointing shared by V2 and V3.
- `inference_cache.py`         KV cache shared by V2 and V3.

Keep this module free of architecture imports so that importing shared runtime
never drags V2 or V3 into the process.
"""

__all__: list[str] = []
