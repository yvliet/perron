"""
Local PyTorch and Hugging Face Transformers backend for Gemma 4.
Supports Gemma 4 E2B, E4B, 26B A4B, and 31B across consumer laptops and workstations.
Includes automatic KaggleHub model resolution, CPU/GPU device negotiation, and thinking trace extraction.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List, Optional
from perron.backends.base import BackendResponse, ModelBackend


class TransformersBackend(ModelBackend):
    """
    Local inference backend using Hugging Face Transformers and KaggleHub for Gemma 4.
    """
    def __init__(
        self,
        model_name_or_path: str = "google/gemma-4-e4b-it",
        adapter_path: Optional[str] = None,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: Optional[str] = None,
        local_files_only: bool = False,
        enable_thinking: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "TransformersBackend requires 'torch' and 'transformers'. "
                "Install them via 'pip install torch transformers accelerate'."
            ) from e

        self.model_name = model_name_or_path
        self.adapter_path = adapter_path
        self.enable_thinking = enable_thinking
        self.local_files_only = local_files_only

        # Resolve path: local directory vs KaggleHub download vs Hugging Face Hub
        resolved_path = model_name_or_path
        if not Path(model_name_or_path).exists():
            # Check if this is a Kaggle model slug and network access is permitted
            is_kaggle_slug = (("google/gemma-4" in model_name_or_path.lower()) or (model_name_or_path.count("/") >= 2)) and not self.local_files_only
            if is_kaggle_slug:
                try:
                    import kagglehub
                    print(f"Resolving model via KaggleHub: {model_name_or_path}...")
                    resolved_path = kagglehub.model_download(model_name_or_path)
                    print(f"Loaded KaggleHub model checkpoint from: {resolved_path}")
                except Exception as ex:
                    print(f"KaggleHub resolution fallback to direct hub path ({ex})")
                    resolved_path = model_name_or_path

        self.resolved_path = resolved_path
        use_offline = local_files_only or Path(resolved_path).is_dir()

        # Attempt AutoProcessor first (for native Gemma 4 chat template & thinking parsing), fallback to AutoTokenizer
        self.processor = None
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(resolved_path, local_files_only=use_offline)
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_path, local_files_only=use_offline)

        if hasattr(self.tokenizer, "truncation_side"):
            self.tokenizer.truncation_side = "left"

        # Determine target device and precision
        cuda_available = torch.cuda.is_available()
        if device == "auto":
            target_device = "cuda" if cuda_available else "cpu"
        else:
            target_device = device

        self.device = target_device

        kwargs = {"local_files_only": use_offline}
        if cuda_available and target_device != "cpu":
            kwargs["device_map"] = target_device
            if load_in_4bit:
                kwargs["load_in_4bit"] = True
            elif load_in_8bit:
                kwargs["load_in_8bit"] = True
            elif torch_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
                kwargs["torch_dtype"] = torch.bfloat16
            else:
                kwargs["torch_dtype"] = torch.float16
        else:
            # CPU runtime: 4-bit CUDA quantization cannot run on CPU
            kwargs["device_map"] = "cpu"
            kwargs["torch_dtype"] = torch.float32

        print(f"Initializing model on device={kwargs['device_map']} with dtype={kwargs.get('torch_dtype', 'default')}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(resolved_path, **kwargs)
        except Exception:
            try:
                from transformers import Gemma4ForConditionalGeneration
                self.model = Gemma4ForConditionalGeneration.from_pretrained(resolved_path, **kwargs)
            except Exception as e:
                raise e

        # Mount PEFT LoRA adapter if provided
        if adapter_path and Path(adapter_path).exists():
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            except ImportError:
                pass

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        stop_sequences: Optional[List[str]] = None,
    ) -> BackendResponse:
        import torch

        if not torch.cuda.is_available():
            torch.set_num_threads(min(8, os.cpu_count() or 4))

        t0 = time.perf_counter()
        
        # Format prompt with chat template if processor supports it and messages structure is used
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]

        eos_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        try:
            eot_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if eot_id is not None and eot_id != getattr(self.tokenizer, "unk_token_id", None) and eot_id not in eos_ids:
                eos_ids.append(eot_id)
        except Exception:
            pass

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
        }
        if eos_ids:
            gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_tokens = output_ids[0, prompt_len:]
        raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
        elapsed = time.perf_counter() - t0

        # Extract native thinking trace (<|think|>...<|/think|> or </|think|>)
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

        # Clean special tokens and enforce stop sequences from final content
        cleaned_content = re.sub(r"<end_of_turn>.*", "", content_part, flags=re.DOTALL).strip()
        if not cleaned_content:
            cleaned_content = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        stop_seqs = list(stop_sequences) if stop_sequences else ["<end_of_turn>"]
        for stop in stop_seqs:
            if stop in cleaned_content:
                cleaned_content = cleaned_content.split(stop)[0].strip()

        return BackendResponse(
            content=cleaned_content if cleaned_content else raw_text,
            thinking_trace=thinking,
            prompt_tokens=prompt_len,
            completion_tokens=len(generated_tokens),
            latency_seconds=elapsed,
        )
