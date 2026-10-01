"""
Local high-throughput vLLM / OpenAI-compatible endpoint backend for Gemma 4.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.error
from typing import List, Optional
from perron.backends.base import BackendResponse, ModelBackend


class VllmBackend(ModelBackend):
    """
    High-throughput local backend communicating with a running vLLM or llama.cpp server.
    """
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "google/gemma-4-31b-it",
        api_key: Optional[str] = None,
        timeout_seconds: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key or "EMPTY"
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        stop_sequences: Optional[List[str]] = None,
    ) -> BackendResponse:
        # If prompt is pre-templated with Gemma tokens (<start_of_turn>), use /completions
        is_pretemplated = "<start_of_turn>" in prompt
        if is_pretemplated:
            endpoint = f"{self.base_url}/completions"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        else:
            endpoint = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

        stops = list(stop_sequences) if stop_sequences else []
        if "<end_of_turn>" not in stops:
            stops.append("<end_of_turn>")
        payload["stop"] = stops

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            # If /completions failed with 404, fallback to /chat/completions
            if is_pretemplated and isinstance(e, urllib.error.HTTPError) and e.code == 404:
                chat_endpoint = f"{self.base_url}/chat/completions"
                chat_payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                chat_payload["stop"] = stops
                chat_req = urllib.request.Request(
                    chat_endpoint,
                    data=json.dumps(chat_payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(chat_req, timeout=self.timeout_seconds) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
            else:
                raise RuntimeError(
                    f"Failed to connect to vLLM server at {endpoint}: {e}. "
                    "Ensure vLLM or llama-server is running."
                ) from e

        elapsed = time.perf_counter() - t0

        choice = result.get("choices", [{}])[0]
        # In /completions, content is in choice["text"]; in /chat/completions, it is in choice["message"]["content"]
        raw_text = choice.get("text", "") or choice.get("message", {}).get("content", "")
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", max(15, int(len(prompt) / 3.5)))
        completion_tokens = usage.get("completion_tokens", max(15, int(len(raw_text) / 3.5)))

        # Extract native thinking trace (<|think|>...<|/think|> or </|think|>)
        thinking = ""
        think_match = re.search(r"<\|think\|>(.*?)(?:</\|think\|>|<\|/think\|>)", raw_text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            content = raw_text[think_match.end():]
        else:
            end_think = re.search(r"(?:</\|think\|>|<\|/think\|>)", raw_text)
            if end_think:
                thinking = raw_text[:end_think.start()].strip()
                content = raw_text[end_think.end():]
            else:
                content = raw_text

        cleaned_content = re.sub(r"<end_of_turn>.*", "", content, flags=re.DOTALL).strip()
        final_content = cleaned_content if cleaned_content else content.strip()

        return BackendResponse(
            content=final_content,
            thinking_trace=thinking,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=elapsed,
        )
