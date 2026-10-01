"""
Offline replay and deterministic mock backend for Perron.
Enables CI verification, unit testing, and reproducible experiments without GPU dependencies.
"""

from __future__ import annotations

import re
import time
from typing import Callable, Dict, List, Optional, Union
from perron.backends.base import BackendResponse, ModelBackend


class OfflineReplayBackend(ModelBackend):
    """
    Deterministic replay backend returning pre-recorded or programmed responses.
    """
    def __init__(
        self,
        canned_responses: Optional[Union[Dict[str, str], List[str], Callable[[str], str]]] = None,
    ):
        self.canned_responses = canned_responses or []
        self._call_count = 0
        self.recorded_prompts: List[str] = []

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        stop_sequences: Optional[List[str]] = None,
    ) -> BackendResponse:
        t0 = time.perf_counter()
        self.recorded_prompts.append(prompt)

        raw_response = ""
        if callable(self.canned_responses):
            raw_response = self.canned_responses(prompt)
        elif isinstance(self.canned_responses, dict):
            # Check prompt substring match
            for key, val in self.canned_responses.items():
                if key in prompt:
                    raw_response = val
                    break
            if not raw_response and "default" in self.canned_responses:
                raw_response = self.canned_responses["default"]
        elif isinstance(self.canned_responses, list):
            if self.canned_responses:
                idx = min(self._call_count, len(self.canned_responses) - 1)
                raw_response = self.canned_responses[idx]

        self._call_count += 1
        elapsed = time.perf_counter() - t0

        # Extract native thinking trace if present (<|think|>...<|/think|>)
        thinking = ""
        think_match = re.search(r"<\|think\|>(.*?)(?:</\|think\|>|<\|/think\|>)", raw_response, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            cleaned_content = (raw_response[:think_match.start()] + raw_response[think_match.end():]).strip()
        else:
            end_think = re.search(r"(?:</\|think\|>|<\|/think\|>)", raw_response)
            if end_think:
                thinking = raw_response[:end_think.start()].strip()
                cleaned_content = raw_response[end_think.end():].strip()
            else:
                cleaned_content = raw_response
        if stop_sequences:
            for stop in stop_sequences:
                if stop in cleaned_content:
                    cleaned_content = cleaned_content.split(stop)[0].strip()

        return BackendResponse(
            content=cleaned_content,
            thinking_trace=thinking,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(cleaned_content.split()),
            latency_seconds=elapsed,
        )
