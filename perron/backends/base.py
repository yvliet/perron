"""
Base abstractions for Perron model inference backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BackendResponse:
    """
    Standardized response from an LLM inference backend.
    """
    content: str
    thinking_trace: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float


class ModelBackend(ABC):
    """
    Abstract interface for local and API model execution.
    """
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        stop_sequences: Optional[List[str]] = None,
    ) -> BackendResponse:
        """
        Execute model generation and return standardized response.
        """
        pass
