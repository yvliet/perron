"""
Perron model inference backends.
"""

from __future__ import annotations

from perron.backends.base import BackendResponse, ModelBackend
from perron.backends.replay import OfflineReplayBackend

# Local and edge inference backends for production execution.
# OfflineReplayBackend provides deterministic offline replay for reproducible evaluations.
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
