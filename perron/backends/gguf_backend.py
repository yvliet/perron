"""
Lightweight GGUF / llama.cpp memory-mapped inference backend for Gemma 4.
Engineered for 16GB consumer laptops (Gemma 4 E4B / E2B) with <3.8 GB active RAM footprint.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List, Optional
from perron.backends.base import BackendResponse, ModelBackend


class GgufBackend(ModelBackend):
    """
    Lightweight local backend executing GGUF quantized models via llama-cpp-python
    or local llama.cpp server endpoints.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        model_name: str = "google/gemma-4-e4b-it",
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.n_ctx = n_ctx
        self.n_threads = n_threads or max(1, (os.cpu_count() or 4) - 1)
        self.n_gpu_layers = n_gpu_layers
        self.llm = None

        if model_path and Path(model_path).is_file():
            try:
                from llama_cpp import Llama
                self.llm = Llama(
                    model_path=str(model_path),
                    n_ctx=n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                )
            except ImportError as e:
                raise ImportError(
                    "GgufBackend with local model file requires 'llama-cpp-python'. "
                    "Install via 'pip install llama-cpp-python'."
                ) from e

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        stop_sequences: Optional[List[str]] = None,
    ) -> BackendResponse:
        t0 = time.perf_counter()
        stop_seqs = list(stop_sequences) if stop_sequences else ["<end_of_turn>"]
        if "<end_of_turn>" not in stop_seqs:
            stop_seqs.append("<end_of_turn>")

        if self.llm is not None:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_seqs,
                echo=False,
            )
            raw_text = output["choices"][0]["text"]
            prompt_tokens = output["usage"]["prompt_tokens"]
            completion_tokens = output["usage"]["completion_tokens"]
        else:
            # Fallback when running without instantiated GGUF weights
            raise RuntimeError(
                f"GgufBackend requires a valid GGUF model path or running endpoint for {self.model_name}. "
                "Specify model_path in agent.yaml or pass a pre-quantized .gguf file."
            )

        elapsed = time.perf_counter() - t0

        # Extract native thinking trace (<|think|>...<|/think|>)
        thinking = ""
        think_match = re.search(r"<\|think\|>(.*?)(?:</\|think\|>|<\|/think\|>)", raw_text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            content_part = raw_text[think_match.end():]
        else:
            end_think = re.search(r"(?:</\|think\|>|<\|/think\|>)", raw_text)
            if end_think:
                thinking = raw_text[:end_think.start()].strip()
                content_part = raw_text[end_think.end():]
            else:
                content_part = raw_text

        cleaned_content = re.sub(r"<end_of_turn>.*", "", content_part, flags=re.DOTALL).strip()
        for stop in stop_seqs:
            if stop in cleaned_content:
                cleaned_content = cleaned_content.split(stop)[0].strip()

        return BackendResponse(
            content=cleaned_content if cleaned_content else raw_text,
            thinking_trace=thinking,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=elapsed,
        )
