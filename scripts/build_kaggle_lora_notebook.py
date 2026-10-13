"""
scripts/build_kaggle_lora_notebook.py - Builder for self-contained Kaggle LoRA Training Notebook.

Creates notebooks/kaggle_gemma4_lora/kaggle_gemma4_lora.ipynb with embedded dataset lines,
native FP16 mixed precision, zero-decode logit evaluation, and complete telemetry logging.

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "notebooks" / "kaggle_gemma4_lora"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_path = REPO_ROOT / "data" / "gemma4_lora_train.jsonl"
val_path = REPO_ROOT / "data" / "gemma4_lora_val.jsonl"

with open(train_path, "r", encoding="utf-8") as f:
    train_lines = f.read()

with open(val_path, "r", encoding="utf-8") as f:
    val_lines = f.read()

# Compress data lines with gzip + base64 for compact, robust notebook embedding
train_b64 = base64.b64encode(gzip.compress(train_lines.encode("utf-8"))).decode("ascii")
val_b64 = base64.b64encode(gzip.compress(val_lines.encode("utf-8"))).decode("ascii")

# Metadata definition
metadata = {
    "id": "yvliet/perron-gemma4-e2b-lora",
    "title": "perron-gemma4-e2b-lora",
    "code_file": "kaggle_gemma4_lora.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": False,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [
        "yvliet/perron-swebench-graphs",
        "yvliet/swebench-strata-annotations",
        "yvliet/perron-gemma4-benchmark"
    ],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [
        "google/gemma-4/Transformers/gemma-4-e2b-it/1"
    ]
}

with open(OUTPUT_DIR / "kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"[+] Wrote metadata to {OUTPUT_DIR / 'kernel-metadata.json'}")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Perron Phase 3: Gemma 4 E2B LoRA Parameter-Efficient Fine-Tuning\n",
            "\n",
            "- **Author**: Sultan Haikal (GitHub: `@yvliet`)\n",
            "- **Hardware**: Kaggle Dual Tesla T4 (Single-GPU FP16 on `cuda:0`)\n",
            "- **Objective**: Parameter-efficient fine-tuning of compact Gemma 4 E2B under native FP16 to resolve the Topological Disambiguation Deficit and beat the Break-Even Discrimination Bound (AUC >= 0.8298).\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 1: Install PEFT and verify hardware environment\n",
            "!pip install -q peft transformers accelerate\n",
            "\n",
            "import os, sys, json, math, time, gzip, base64\n",
            "from pathlib import Path\n",
            "import numpy as np\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "from torch.utils.data import DataLoader, Dataset\n",
            "from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup\n",
            "from peft import LoraConfig, get_peft_model, PeftModel\n",
            "\n",
            "device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')\n",
            "print(f'[*] Active Device: {device}')\n",
            "if torch.cuda.is_available():\n",
            "    print(f'[*] GPU Name: {torch.cuda.get_device_name(0)}')\n",
            "    print(f'[*] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 2: Unpack contrastive training & validation datasets\n",
            "TRAIN_B64 = '''" + train_b64 + "'''\n",
            "VAL_B64 = '''" + val_b64 + "'''\n",
            "\n",
            "train_raw = gzip.decompress(base64.b64decode(TRAIN_B64)).decode('utf-8')\n",
            "val_raw = gzip.decompress(base64.b64decode(VAL_B64)).decode('utf-8')\n",
            "\n",
            "train_records = [json.loads(l) for l in train_raw.splitlines() if l.strip()]\n",
            "val_records = [json.loads(l) for l in val_raw.splitlines() if l.strip()]\n",
            "\n",
            "print(f'[*] Extracted {len(train_records)} training records, {len(val_records)} validation records.')\n",
            "Path('data').mkdir(exist_ok=True)\n",
            "with open('data/train.jsonl', 'w', encoding='utf-8') as f:\n",
            "    f.write(train_raw)\n",
            "with open('data/val.jsonl', 'w', encoding='utf-8') as f:\n",
            "    f.write(val_raw)\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 3: Dataset class and Collate function with prompt masking\n",
            "class CandidateChoiceDataset(Dataset):\n",
            "    def __init__(self, records, tokenizer, max_length=2048):\n",
            "        self.records = records\n",
            "        self.tokenizer = tokenizer\n",
            "        self.max_length = max_length\n",
            "\n",
            "    def __len__(self):\n",
            "        return len(self.records)\n",
            "\n",
            "    def __getitem__(self, idx):\n",
            "        item = self.records[idx]\n",
            "        prompt = item['prompt']\n",
            "        completion = item['completion']\n",
            "        full_text = prompt + completion\n",
            "\n",
            "        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)\n",
            "        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)\n",
            "        if len(full_ids) > self.max_length:\n",
            "            full_ids = full_ids[:self.max_length]\n",
            "\n",
            "        input_ids = torch.tensor(full_ids, dtype=torch.long)\n",
            "        attention_mask = torch.ones_like(input_ids)\n",
            "        labels = input_ids.clone()\n",
            "        prompt_len = min(len(prompt_ids), len(input_ids))\n",
            "        labels[:prompt_len] = -100  # Mask prompt tokens\n",
            "\n",
            "        return {\n",
            "            'input_ids': input_ids,\n",
            "            'attention_mask': attention_mask,\n",
            "            'labels': labels,\n",
            "            'target_letter': item.get('target_letter', 'A'),\n",
            "        }\n",
            "\n",
            "def collate_fn(batch):\n",
            "    max_len = max(len(b['input_ids']) for b in batch)\n",
            "    input_ids_list, mask_list, labels_list, targets = [], [], [], []\n",
            "    for b in batch:\n",
            "        cur_len = len(b['input_ids'])\n",
            "        pad_len = max_len - cur_len\n",
            "        padded_input = torch.cat([b['input_ids'], torch.zeros(pad_len, dtype=torch.long)])\n",
            "        padded_mask = torch.cat([b['attention_mask'], torch.zeros(pad_len, dtype=torch.long)])\n",
            "        padded_labels = torch.cat([b['labels'], torch.full((pad_len,), -100, dtype=torch.long)])\n",
            "        input_ids_list.append(padded_input)\n",
            "        mask_list.append(padded_mask)\n",
            "        labels_list.append(padded_labels)\n",
            "        targets.append(b['target_letter'])\n",
            "    return {\n",
            "        'input_ids': torch.stack(input_ids_list),\n",
            "        'attention_mask': torch.stack(mask_list),\n",
            "        'labels': torch.stack(labels_list),\n",
            "        'target_letters': targets,\n",
            "    }\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 4: Zero-Decode Logit Sorting Evaluation Routine\n",
            "def evaluate_zero_decode(model, dataloader, tokenizer, device):\n",
            "    model.eval()\n",
            "    correct_top1, correct_top5, total = 0, 0, 0\n",
            "    letter_tokens = [chr(ord('A') + i) for i in range(25)]\n",
            "    letter_ids = [tokenizer.encode(lt, add_special_tokens=False)[-1] for lt in letter_tokens]\n",
            "\n",
            "    with torch.no_grad():\n",
            "        for batch in dataloader:\n",
            "            input_ids = batch['input_ids'].to(device)\n",
            "            attention_mask = batch['attention_mask'].to(device)\n",
            "            outputs = model(input_ids=input_ids, attention_mask=attention_mask)\n",
            "            logits = outputs.logits\n",
            "\n",
            "            for i, target_letter in enumerate(batch['target_letters']):\n",
            "                seq_len = int(attention_mask[i].sum().item())\n",
            "                pred_logits = logits[i, seq_len - 2, letter_ids].cpu().numpy()\n",
            "                ranked = np.argsort(-pred_logits)\n",
            "                gold_idx = ord(target_letter) - ord('A')\n",
            "                if 0 <= gold_idx < len(letter_tokens):\n",
            "                    if ranked[0] == gold_idx:\n",
            "                        correct_top1 += 1\n",
            "                    if gold_idx in ranked[:5]:\n",
            "                        correct_top5 += 1\n",
            "                    total += 1\n",
            "\n",
            "    return {\n",
            "        'val_acc1': float(correct_top1 / total) if total > 0 else 0.0,\n",
            "        'val_acc5': float(correct_top5 / total) if total > 0 else 0.0,\n",
            "        'total': total\n",
            "    }\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 5: Initialize Model, Tokenizer, and LoRA Adapter\n",
            "model_path = 'google/gemma-4-e2b-it'\n",
            "kaggle_paths = [\n",
            "    Path('/kaggle/input/gemma-4/Transformers/gemma-4-e2b-it/1'),\n",
            "    Path('/kaggle/input/gemma-4-e2b-it'),\n",
            "    Path('/kaggle/input/gemma-4/transformers/gemma-4-e2b-it/1')\n",
            "]\n",
            "for p in kaggle_paths:\n",
            "    if p.exists():\n",
            "        model_path = str(p)\n",
            "        print(f'[*] Located local Kaggle model: {model_path}')\n",
            "        break\n",
            "\n",
            "print(f'[*] Loading tokenizer from {model_path}...')\n",
            "tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)\n",
            "if tokenizer.pad_token is None:\n",
            "    tokenizer.pad_token = tokenizer.eos_token\n",
            "\n",
            "print(f'[*] Loading base model in native FP16 on {device}...')\n",
            "dtype = torch.float16 if torch.cuda.is_available() else torch.float32\n",
            "model = AutoModelForCausalLM.from_pretrained(\n",
            "    model_path,\n",
            "    torch_dtype=dtype,\n",
            "    trust_remote_code=True\n",
            ").to(device)\n",
            "\n",
            "model.gradient_checkpointing_enable()\n",
            "if hasattr(model, 'enable_input_require_grads'):\n",
            "    model.enable_input_require_grads()\n",
            "\n",
            "peft_config = LoraConfig(\n",
            "    r=16,\n",
            "    lora_alpha=32,\n",
            "    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],\n",
            "    lora_dropout=0.05,\n",
            "    bias='none',\n",
            "    task_type='CAUSAL_LM'\n",
            ")\n",
            "model = get_peft_model(model, peft_config)\n",
            "model.print_trainable_parameters()\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 6: Execute LoRA Supervised Fine-Tuning\n",
            "epochs = 3\n",
            "batch_size = 1\n",
            "accum_steps = 8\n",
            "lr = 2e-4\n",
            "\n",
            "train_ds = CandidateChoiceDataset(train_records, tokenizer, max_length=2048)\n",
            "val_ds = CandidateChoiceDataset(val_records, tokenizer, max_length=2048)\n",
            "train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)\n",
            "val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)\n",
            "\n",
            "trainable_params = [p for p in model.parameters() if p.requires_grad]\n",
            "optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)\n",
            "total_steps = math.ceil(len(train_loader) / accum_steps) * epochs\n",
            "warmup_steps = int(0.1 * total_steps)\n",
            "scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)\n",
            "scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())\n",
            "\n",
            "output_dir = Path('perron-gemma4-e2b-lora')\n",
            "output_dir.mkdir(exist_ok=True)\n",
            "history = []\n",
            "best_acc5 = -1.0\n",
            "\n",
            "print(f'[*] Training started: {epochs} epochs, {total_steps} optimizer steps.')\n",
            "for epoch in range(1, epochs + 1):\n",
            "    model.train()\n",
            "    epoch_loss = 0.0\n",
            "    optimizer.zero_grad()\n",
            "\n",
            "    for step, batch in enumerate(train_loader, start=1):\n",
            "        input_ids = batch['input_ids'].to(device)\n",
            "        mask = batch['attention_mask'].to(device)\n",
            "        labels = batch['labels'].to(device)\n",
            "\n",
            "        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):\n",
            "            outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)\n",
            "            loss = outputs.loss / accum_steps\n",
            "\n",
            "        scaler.scale(loss).backward()\n",
            "        epoch_loss += loss.item() * accum_steps\n",
            "\n",
            "        if step % accum_steps == 0 or step == len(train_loader):\n",
            "            scaler.step(optimizer)\n",
            "            scaler.update()\n",
            "            optimizer.zero_grad()\n",
            "            scheduler.step()\n",
            "\n",
            "    avg_loss = epoch_loss / len(train_loader)\n",
            "    val_metrics = evaluate_zero_decode(model, val_loader, tokenizer, device)\n",
            "    print(f'Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Val Acc@1: {val_metrics[\"val_acc1\"]:.4f} | Val Acc@5: {val_metrics[\"val_acc5\"]:.4f}')\n",
            "    history.append({'epoch': epoch, 'loss': avg_loss, **val_metrics})\n",
            "\n",
            "    if val_metrics['val_acc5'] > best_acc5:\n",
            "        best_acc5 = val_metrics['val_acc5']\n",
            "        print(f'[+] New best checkpoint: saving adapter to {output_dir}...')\n",
            "        model.save_pretrained(str(output_dir))\n",
            "        tokenizer.save_pretrained(str(output_dir))\n",
            "\n",
            "with open(output_dir / 'training_telemetry.json', 'w') as f:\n",
            "    json.dump(history, f, indent=2)\n",
            "print(f'[DONE] Phase 3 LoRA Fine-Tuning Complete! Best Val Acc@5: {best_acc5:.4f}')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 7: Export final adapter zip archive for distribution\n",
            "!zip -r perron-gemma4-e2b-lora.zip perron-gemma4-e2b-lora\n",
            "print('[+] Adapter archive created: perron-gemma4-e2b-lora.zip')\n",
            "!ls -lh perron-gemma4-e2b-lora.zip\n"
        ]
    }
]

notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

nb_path = OUTPUT_DIR / "kaggle_gemma4_lora.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2)

print(f"[+] Successfully built self-contained Kaggle notebook at: {nb_path}")
