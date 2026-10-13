"""
benchmarks/train_gemma4_lora.py - Turnkey Parameter-Efficient Fine-Tuning for Gemma 4 E2B.

Executes native FP16 LoRA SFT on Gemma 4 E2B targeting attention projections (q, v, k, o),
governed by gradient checkpointing, atomic candidate letter discretization [A]-[Y],
and zero-decode logit sorting for fast, deterministic evaluation.

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent


class CandidateChoiceDataset(Dataset):
    """
    Dataset formatting contrastive candidate pairs with prompt masking for SFT.
    """

    def __init__(
        self,
        jsonl_path: Path | str,
        tokenizer: Any,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: List[Dict[str, Any]] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.records[idx]
        prompt = item["prompt"]
        completion = item["completion"]
        full_text = prompt + completion

        # Tokenize prompt and full text
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Truncate if exceeds max_length
        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Create labels: mask out prompt tokens with -100 so loss is computed strictly on target choice
        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "target_letter": item.get("target_letter", "A"),
            "instance_id": item.get("instance_id", ""),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad batch to max length within batch."""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    target_letters = []
    instance_ids = []

    for b in batch:
        cur_len = len(b["input_ids"])
        pad_len = max_len - cur_len

        # Pad with 0 for inputs/mask, and -100 for labels
        padded_input = torch.cat([b["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        padded_mask = torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        padded_labels = torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])

        input_ids_list.append(padded_input)
        attention_mask_list.append(padded_mask)
        labels_list.append(padded_labels)
        target_letters.append(b["target_letter"])
        instance_ids.append(b["instance_id"])

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
        "target_letters": target_letters,
        "instance_ids": instance_ids,
    }


def evaluate_zero_decode_acc(
    model: Any,
    dataloader: DataLoader,
    tokenizer: Any,
    device: torch.device,
) -> Dict[str, float]:
    """
    Computes validation accuracy using zero-decode single forward-pass logit sorting.
    """
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    # Letter token IDs in Gemma tokenizer: 'A' through 'Y'
    letter_tokens = [chr(ord("A") + i) for i in range(25)]
    letter_ids = [tokenizer.encode(lt, add_special_tokens=False)[-1] for lt in letter_tokens]

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, L, V]

            for i, target_letter in enumerate(batch["target_letters"]):
                # Target position is the last valid token before padding
                seq_len = int(attention_mask[i].sum().item())
                # Logit at prediction site
                pred_logits = logits[i, seq_len - 2, letter_ids].cpu().numpy()
                ranked_indices = np.argsort(-pred_logits)

                gold_idx = ord(target_letter) - ord("A")
                if 0 <= gold_idx < len(letter_tokens):
                    if ranked_indices[0] == gold_idx:
                        correct_top1 += 1
                    if gold_idx in ranked_indices[:5]:
                        correct_top5 += 1
                    total += 1

    acc1 = float(correct_top1 / total) if total > 0 else 0.0
    acc5 = float(correct_top5 / total) if total > 0 else 0.0
    return {"val_acc1": acc1, "val_acc5": acc5, "total": total}


def train_lora_pipeline(
    model_name: str = "google/gemma-4-e2b-it",
    train_path: Path | str = REPO_ROOT / "data" / "gemma4_lora_train.jsonl",
    val_path: Path | str = REPO_ROOT / "data" / "gemma4_lora_val.jsonl",
    output_dir: Path | str = REPO_ROOT / "artifacts" / "perron-gemma4-e2b-lora",
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes full native FP16 LoRA training loop on Gemma 4 E2B.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_fp16 = torch.cuda.is_available()
    print(f"[*] Training on device: {device} | Native FP16: {use_fp16}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Datasets
    train_dataset = CandidateChoiceDataset(train_path, tokenizer, max_length=max_length)
    val_dataset = CandidateChoiceDataset(val_path, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"[*] Loaded {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    # Load Model in native FP16
    dtype = torch.float16 if use_fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    # Enable gradient checkpointing and input gradients
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Filter parameters to requires_grad=True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    total_steps = math.ceil(len(train_loader) / gradient_accumulation_steps) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    best_val_acc5 = -1.0
    training_history: List[Dict[str, Any]] = []

    print(f"[*] Beginning training: {epochs} epochs, {total_steps} total optimizer steps.")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_fp16, dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * gradient_accumulation_steps

            if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step_count += 1

        avg_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate_zero_decode_acc(model, val_loader, tokenizer, device)
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Val Acc@1: {val_metrics['val_acc1']:.4f} | Val Acc@5: {val_metrics['val_acc5']:.4f}")

        training_history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_acc1": val_metrics["val_acc1"],
            "val_acc5": val_metrics["val_acc5"],
        })

        if val_metrics["val_acc5"] > best_val_acc5:
            best_val_acc5 = val_metrics["val_acc5"]
            print(f"[+] New best checkpoint: saving adapter to {output_dir}")
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

    # Save training history telemetry
    history_file = output_dir / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    print(f"[DONE] Training complete. Best Val Acc@5: {best_val_acc5:.4f}. Artifacts saved in {output_dir}")
    return {"best_val_acc5": best_val_acc5, "history": training_history}


def main():
    parser = argparse.ArgumentParser(description="Perron Gemma 4 LoRA Training Harness")
    parser.add_argument("--model-name", type=str, default="google/gemma-4-e2b-it", help="Base model identifier")
    parser.add_argument("--train-path", type=Path, default=REPO_ROOT / "data" / "gemma4_lora_train.jsonl")
    parser.add_argument("--val-path", type=Path, default=REPO_ROOT / "data" / "gemma4_lora_val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "perron-gemma4-e2b-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_lora_pipeline(
        model_name=args.model_name,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        max_length=args.max_length,
        lora_r=args.r,
        lora_alpha=args.alpha,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
