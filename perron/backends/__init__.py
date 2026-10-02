"""
Perron model inference backends.
"""

from __future__ import annotations

from perron.backends.base import BackendResponse, ModelBackend
from perron.backends.replay import OfflineReplayBackend

# The following backends are provided as extension interfaces for the 'Best New Resource' track.
# The official Track 2 evaluation runner strictly uses OfflineReplayBackend to guarantee 
# deterministic offline reproducibility during the code audit.
from perron.backends.vllm_backend import VllmBackend
from perron.backends.gguf_backend import GgufBackend

__all__ = [
    "BackendResponse",
    "ModelBackend",
    "OfflineReplayBackend",
    "VllmBackend",
    "GgufBackend",
]

# Lazy / conditional export for TransformersBackend
def get_transformers_backend():
    from perron.backends.transformers_backend import TransformersBackend
    return TransformersBackend
