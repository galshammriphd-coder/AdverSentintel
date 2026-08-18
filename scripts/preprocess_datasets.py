"""
Preprocess and save all datasets to disk as tokenized .pt artifacts.

This ensures a consistent starting point for all reviewers regardless
of dataset version or tokenizer behavior, as recommended in the
reproducibility package instructions.

Usage:
    python scripts/preprocess_datasets.py \
        --data_dir data/ \
        --out_dir  preprocessed/ \
        --config   configs/default.yaml

Output (per dataset):
    preprocessed/advbench_train.pt
    preprocessed/advbench_val.pt
    preprocessed/jailbreakbench_train.pt
    preprocessed/jailbreakbench_val.pt
    preprocessed/toxicchat_train.pt
    preprocessed/toxicchat_val.pt
    preprocessed/promptsafety_train.pt
    preprocessed/promptsafety_test.pt
    preprocessed/manifest.json          <- checksums + split sizes
"""

import os
import sys
import json
import random
import hashlib
import argparse
import torch
import yaml
from transformers import BertTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from dataset import (load_advbench, load_jailbreakbench,
                      load_toxicchat, load_promptsafety_bench)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def tokenize_and_save(prompts, labels, tokenizer, max_length,
                       out_path: str, desc: str) -> dict:
    """Tokenize all prompts and save as a single .pt artifact."""
    print(f"  Tokenizing {desc} ({len(prompts)} samples)...")

    # Batch tokenize for speed
    encodings = tokenizer(
        prompts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    artifact = {
        'input_ids':      encodings['input_ids'],       # (N, max_length)
        'attention_mask': encodings['attention_mask'],  # (N, max_length)
        'labels':         torch.tensor(labels, dtype=torch.long),  # (N,)
        'prompts':        prompts,   # raw strings preserved for debugging
        'meta': {
            'n_samples':    len(prompts),
            'n_adversarial': sum(labels),
            'n_benign':      len(labels) - sum(labels),
            'max_length':    max_length,
            'tokenizer':     'bert-base-uncased',
            'description':   desc,
        }
    }

    torch.save(artifact, out_path)
    size_kb = os.path.getsize(out_path) / 1024
    checksum = sha256_file(out_path)
    print(f"    Saved: {out_path}  ({size_kb:.0f} KB, sha256={checksum[:16]}...)")
    return {'path': out_path, 'sha256': checksum, 'n': len(prompts),
            'n_adv': sum(labels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--out_dir',  type=str, default='preprocessed/')
    parser.add_argument('--config',   type=str, default='configs/default.yaml')
    parser.add_argument('--seed',     type=int, default=42)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer  = BertTokenizer.from_pretrained(cfg['model']['encoder_backbone'])
    max_length = cfg['model']['max_sequence_length']
    data_dir   = args.data_dir
    manifest   = {}

    datasets = [
        ('advbench',       load_advbench,           {}),
        ('jailbreakbench', load_jailbreakbench,     {}),
        ('toxicchat',      load_toxicchat,           {}),
    ]

    for name, loader_fn, kwargs in datasets:
        print(f"\nProcessing {name} ...")
        try:
            p, l = loader_fn(data_dir, **kwargs)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue

        # 90/10 train/val split (same seed as train.py)
        idx = list(range(len(p)))
        random.shuffle(idx)
        split = int(0.9 * len(p))
        train_idx, val_idx = idx[:split], idx[split:]

        train_path = os.path.join(args.out_dir, f'{name}_train.pt')
        val_path   = os.path.join(args.out_dir, f'{name}_val.pt')

        info_train = tokenize_and_save(
            [p[i] for i in train_idx], [l[i] for i in train_idx],
            tokenizer, max_length, train_path, f'{name}/train')
        info_val = tokenize_and_save(
            [p[i] for i in val_idx], [l[i] for i in val_idx],
            tokenizer, max_length, val_path, f'{name}/val')

        manifest[name] = {'train': info_train, 'val': info_val}

    # PromptSafety-Bench: separate train/test splits
    print(f"\nProcessing promptsafety ...")
    for split in ('train', 'test'):
        try:
            p, l = load_promptsafety_bench(data_dir, split=split)
            out_path = os.path.join(args.out_dir, f'promptsafety_{split}.pt')
            info = tokenize_and_save(p, l, tokenizer, max_length,
                                      out_path, f'promptsafety/{split}')
            manifest.setdefault('promptsafety', {})[split] = info
        except FileNotFoundError as e:
            print(f"  SKIPPED {split}: {e}")

    # Save manifest with checksums
    manifest_path = os.path.join(args.out_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")
    print("Preprocessing complete.")


if __name__ == '__main__':
    main()
