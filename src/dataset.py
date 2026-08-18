"""
Dataset loading and preprocessing for AdverSentinel-LLM.

Supported datasets:
  - AdvBench          (GitHub: llm-attacks/llm-attacks)
  - JailbreakBench    (GitHub: JailbreakBench/jailbreakbench)
  - ToxicChat         (HuggingFace: lmsys/toxic-chat)
  - PromptSafety-Bench (HuggingFace: SalKhan12/prompt-safety-dataset)

Class-balanced sampling is applied for ToxicChat (7.18% adversarial).
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# Base Prompt Dataset
# ---------------------------------------------------------------------------

class PromptDataset(Dataset):
    """
    Generic dataset for (prompt, label) pairs.
    label: 0 = benign, 1 = adversarial
    """

    def __init__(self, prompts: List[str], labels: List[int],
                 tokenizer: BertTokenizer, max_length: int = 512):
        assert len(prompts) == len(labels)
        self.prompts   = prompts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.prompts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        # Context features: [length_norm, turn_pos (0 = single turn), domain_id (0)]
        length_norm = (encoding['attention_mask'].sum().float() / self.max_length).item()
        ctx = torch.tensor([length_norm, 0.0, 0.0], dtype=torch.float)

        return {
            'input_ids':      encoding['input_ids'].squeeze(0),        # (512,)
            'attention_mask': encoding['attention_mask'].squeeze(0),   # (512,)
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
            'ctx':            ctx,                                      # (3,)
        }


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------

def load_advbench(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load AdvBench dataset.
    Expected file: {data_dir}/advbench/harmful_behaviors.csv
    Returns 10 augmented variants per behavior (5200 total, 50% adversarial).
    """
    import csv
    prompts, labels = [], []
    csv_path = os.path.join(data_dir, 'advbench', 'harmful_behaviors.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"AdvBench not found at {csv_path}. "
            "Download from https://github.com/llm-attacks/llm-attacks"
        )

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Adversarial: the goal/target prompt
            for _ in range(10):   # 10 augmented variants per behavior
                prompts.append(row.get('goal', row.get('prompt', '')))
                labels.append(1)

    # Add equal number of benign prompts from alpaca-style data if available
    benign_path = os.path.join(data_dir, 'advbench', 'benign_prompts.txt')
    if os.path.exists(benign_path):
        with open(benign_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
                    labels.append(0)

    return prompts, labels


def load_jailbreakbench(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load JailbreakBench dataset.
    Expected file: {data_dir}/jailbreakbench/jbb-behaviors.json
    """
    json_path = os.path.join(data_dir, 'jailbreakbench', 'jbb-behaviors.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"JailbreakBench not found at {json_path}. "
            "Download from https://github.com/JailbreakBench/jailbreakbench"
        )

    with open(json_path, 'r') as f:
        data = json.load(f)

    prompts, labels = [], []
    for item in data:
        if isinstance(item, dict):
            prompt = item.get('prompt', item.get('goal', ''))
            label  = 1 if item.get('label', 'harmful') in ('harmful', 'jailbreak', 1) else 0
            prompts.append(prompt)
            labels.append(label)

    return prompts, labels


def load_toxicchat(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load ToxicChat dataset.
    Expected file: {data_dir}/toxicchat/toxicchat0124.json
    HuggingFace: lmsys/toxic-chat
    """
    json_path = os.path.join(data_dir, 'toxicchat', 'toxicchat0124.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"ToxicChat not found at {json_path}. "
            "Download from https://huggingface.co/datasets/lmsys/toxic-chat"
        )

    with open(json_path, 'r') as f:
        data = json.load(f)

    prompts, labels = [], []
    for item in data:
        if isinstance(item, dict):
            prompt = item.get('human_turn', item.get('prompt', ''))
            # toxicity=1 or jailbreaking=1 -> adversarial
            label = int(item.get('toxicity', 0) == 1 or
                        item.get('jailbreaking', 0) == 1)
            prompts.append(str(prompt))
            labels.append(label)

    return prompts, labels


def load_promptsafety_bench(data_dir: str,
                             split: str = 'train') -> Tuple[List[str], List[int]]:
    """
    Load PromptSafety-Bench dataset.
    Expected file: {data_dir}/promptsafety/{split}.json
    HuggingFace: SalKhan12/prompt-safety-dataset
    """
    json_path = os.path.join(data_dir, 'promptsafety', f'{split}.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"PromptSafety-Bench not found at {json_path}. "
            "Download from https://huggingface.co/datasets/SalKhan12/prompt-safety-dataset"
        )

    with open(json_path, 'r') as f:
        data = json.load(f)

    prompts, labels = [], []
    for item in data:
        if isinstance(item, dict):
            prompt = item.get('prompt', item.get('text', ''))
            label  = int(item.get('label', item.get('is_adversarial', 0)))
            prompts.append(str(prompt))
            labels.append(label)

    return prompts, labels


# ---------------------------------------------------------------------------
# Balanced DataLoader factory
# ---------------------------------------------------------------------------

def make_balanced_loader(dataset: PromptDataset,
                          batch_size: int = 128,
                          num_workers: int = 4,
                          shuffle: bool = True) -> DataLoader:
    """
    Creates a DataLoader with class-balanced oversampling for the minority class.
    Used for ToxicChat (7.18% adversarial) and any other imbalanced split.
    """
    labels = torch.tensor(dataset.labels)
    class_counts = torch.bincount(labels)
    # Weight per sample = inverse class frequency
    weights = 1.0 / class_counts[labels].float()
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset),
                                    replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def make_loader(dataset: PromptDataset,
                batch_size: int = 128,
                num_workers: int = 4,
                shuffle: bool = True) -> DataLoader:
    """Standard DataLoader (for balanced datasets)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


# ---------------------------------------------------------------------------
# Dataset download helper scripts
# ---------------------------------------------------------------------------

DOWNLOAD_INSTRUCTIONS = """
# Dataset Download Instructions
# ==============================
#
# 1. AdvBench
#    git clone https://github.com/llm-attacks/llm-attacks
#    cp llm-attacks/data/advbench/harmful_behaviors.csv data/advbench/
#
# 2. JailbreakBench
#    git clone https://github.com/JailbreakBench/jailbreakbench
#    cp jailbreakbench/data/jbb-behaviors.json data/jailbreakbench/
#
# 3. ToxicChat
#    pip install datasets
#    python -c "
#    from datasets import load_dataset
#    import json
#    ds = load_dataset('lmsys/toxic-chat', 'toxicchat0124')
#    import os; os.makedirs('data/toxicchat', exist_ok=True)
#    with open('data/toxicchat/toxicchat0124.json','w') as f:
#        json.dump([dict(r) for r in ds['train']], f)
#    "
#
# 4. PromptSafety-Bench
#    python -c "
#    from datasets import load_dataset
#    import json, os
#    ds = load_dataset('SalKhan12/prompt-safety-dataset')
#    os.makedirs('data/promptsafety', exist_ok=True)
#    for split in ds:
#        with open(f'data/promptsafety/{split}.json','w') as f:
#            json.dump([dict(r) for r in ds[split]], f)
#    "
"""

if __name__ == '__main__':
    print(DOWNLOAD_INSTRUCTIONS)
