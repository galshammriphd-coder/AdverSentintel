"""
Reproduce Table 3: Comparative detection performance across all datasets.

Usage:
    python scripts/reproduce_table3.py \
        --checkpoint outputs/best_model.pt \
        --config configs/default.yaml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import torch
import yaml
from transformers import BertTokenizer

from model    import AdverSentinelLLM
from dataset  import (PromptDataset, make_loader, load_advbench,
                       load_jailbreakbench, load_toxicchat,
                       load_promptsafety_bench)
from evaluate import evaluate_all_datasets, print_results_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config',     type=str, default='configs/default.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(cfg['model']['encoder_backbone'])
    data_dir  = cfg['data']['data_dir']
    max_len   = cfg['model']['max_sequence_length']
    bsz       = cfg['evaluation']['batch_size']

    # Load model
    model = AdverSentinelLLM(
        pretrained=cfg['model']['encoder_backbone'],
        proj_dim=cfg['model']['projection_dim'],
        num_prototypes=cfg['model']['dtmb_prototypes'],
        num_categories=cfg['model']['dtmb_categories'],
        dtmb_momentum=cfg['model']['ema_momentum'],
        stale_threshold=cfg['model']['stale_threshold'],
        cos_prune_threshold=cfg['model']['cos_prune_threshold'],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f"Checkpoint loaded (epoch {ckpt.get('epoch','?')}, "
          f"val_f1={ckpt.get('val_f1','?')})\n")

    # Build loaders
    loaders = {}
    dataset_fns = {
        'AdvBench':           (load_advbench,           {}),
        'JailbreakBench':     (load_jailbreakbench,     {}),
        'ToxicChat':          (load_toxicchat,           {}),
        'PromptSafety-Bench': (load_promptsafety_bench, {'split': 'test'}),
    }
    for name, (fn, kw) in dataset_fns.items():
        try:
            p, l = fn(data_dir, **kw)
            ds   = PromptDataset(p, l, tokenizer, max_len)
            loaders[name] = make_loader(ds, bsz, shuffle=False)
            print(f"  {name}: {len(ds)} samples loaded")
        except FileNotFoundError as e:
            print(f"  SKIPPED {name}: {e}")

    print()
    results = evaluate_all_datasets(model, loaders, device)
    print_results_table(results, label="Table 3: AdverSentinel-LLM Detection Results")


if __name__ == '__main__':
    main()
