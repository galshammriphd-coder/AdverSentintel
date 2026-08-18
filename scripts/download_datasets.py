"""
Download and prepare all four benchmark datasets for AdverSentinel-LLM.

Usage:
    python scripts/download_datasets.py --data_dir data/ --all
    python scripts/download_datasets.py --data_dir data/ --dataset toxicchat
"""

import os
import json
import argparse
import subprocess


def download_advbench(data_dir: str) -> None:
    """Download AdvBench from GitHub."""
    out = os.path.join(data_dir, 'advbench')
    os.makedirs(out, exist_ok=True)
    target = os.path.join(out, 'harmful_behaviors.csv')
    if os.path.exists(target):
        print("AdvBench already exists, skipping.")
        return
    url = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
           "main/data/advbench/harmful_behaviors.csv")
    print(f"Downloading AdvBench from {url} ...")
    subprocess.run(['wget', '-q', '-O', target, url], check=True)
    print(f"Saved to {target}")


def download_jailbreakbench(data_dir: str) -> None:
    """Download JailbreakBench behaviors from GitHub."""
    out = os.path.join(data_dir, 'jailbreakbench')
    os.makedirs(out, exist_ok=True)
    target = os.path.join(out, 'jbb-behaviors.json')
    if os.path.exists(target):
        print("JailbreakBench already exists, skipping.")
        return
    url = ("https://raw.githubusercontent.com/JailbreakBench/"
           "jailbreakbench/main/src/jailbreakbench/data/behaviors.json")
    print(f"Downloading JailbreakBench from {url} ...")
    subprocess.run(['wget', '-q', '-O', target, url], check=True)
    print(f"Saved to {target}")


def download_toxicchat(data_dir: str) -> None:
    """Download ToxicChat from HuggingFace datasets."""
    out = os.path.join(data_dir, 'toxicchat')
    os.makedirs(out, exist_ok=True)
    target = os.path.join(out, 'toxicchat0124.json')
    if os.path.exists(target):
        print("ToxicChat already exists, skipping.")
        return
    print("Downloading ToxicChat from HuggingFace ...")
    try:
        from datasets import load_dataset
        ds = load_dataset('lmsys/toxic-chat', 'toxicchat0124', split='train')
        records = [dict(row) for row in ds]
        with open(target, 'w') as f:
            json.dump(records, f)
        print(f"Saved {len(records)} records to {target}")
    except Exception as e:
        print(f"Error downloading ToxicChat: {e}")
        print("Manual: pip install datasets && "
              "python -c \"from datasets import load_dataset; "
              "ds = load_dataset('lmsys/toxic-chat','toxicchat0124')\"")


def download_promptsafety(data_dir: str) -> None:
    """Download PromptSafety-Bench from HuggingFace datasets."""
    out = os.path.join(data_dir, 'promptsafety')
    os.makedirs(out, exist_ok=True)
    train_target = os.path.join(out, 'train.json')
    test_target  = os.path.join(out, 'test.json')
    if os.path.exists(train_target) and os.path.exists(test_target):
        print("PromptSafety-Bench already exists, skipping.")
        return
    print("Downloading PromptSafety-Bench from HuggingFace ...")
    try:
        from datasets import load_dataset
        ds = load_dataset('SalKhan12/prompt-safety-dataset')
        for split in ds:
            records = [dict(row) for row in ds[split]]
            target  = os.path.join(out, f'{split}.json')
            with open(target, 'w') as f:
                json.dump(records, f)
            print(f"  {split}: {len(records)} records -> {target}")
    except Exception as e:
        print(f"Error downloading PromptSafety-Bench: {e}")
        print("Manual: from datasets import load_dataset; "
              "ds = load_dataset('SalKhan12/prompt-safety-dataset')")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--all',      action='store_true')
    parser.add_argument('--dataset',  type=str, default=None,
                        choices=['advbench', 'jailbreakbench',
                                 'toxicchat', 'promptsafety'])
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.all or args.dataset == 'advbench':
        download_advbench(args.data_dir)
    if args.all or args.dataset == 'jailbreakbench':
        download_jailbreakbench(args.data_dir)
    if args.all or args.dataset == 'toxicchat':
        download_toxicchat(args.data_dir)
    if args.all or args.dataset == 'promptsafety':
        download_promptsafety(args.data_dir)

    print("\nDataset download complete.")


if __name__ == '__main__':
    main()
