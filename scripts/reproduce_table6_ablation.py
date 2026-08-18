"""
Reproduce Table 6: Ablation study.

Each configuration removes or replaces one component of AdverSentinel-LLM.
Requires separate checkpoint files for each ablation variant,
or uses the full model with component-level masking.

Usage:
    python scripts/reproduce_table6_ablation.py \
        --full_checkpoint  outputs/full/best_model.pt \
        --no_tpe_ckpt      outputs/no_tpe/best_model.pt \
        --no_dtmb_ckpt     outputs/no_dtmb/best_model.pt \
        --no_aagm_ckpt     outputs/no_aagm/best_model.pt \
        --no_con_ckpt      outputs/no_con/best_model.pt \
        --no_div_ckpt      outputs/no_div/best_model.pt \
        --config           configs/default.yaml
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import torch
import yaml
from transformers import BertTokenizer

from model   import AdverSentinelLLM
from dataset import (PromptDataset, make_loader, load_promptsafety_bench)
from evaluate import evaluate


ABLATION_LABELS = {
    'full':            'Full AdverSentinel-LLM',
    'no_tpe':          'w/o TPE (PSE only)',
    'no_dtmb':         'w/o DTMB',
    'no_dtmb_pruning': 'w/o DTMB Pruning/Rejuvenation',
    'no_aagm':         'w/o AAGM',
    'no_con':          'w/o Contrastive Loss',
    'no_div':          'w/o Diversity Loss',
}


def load_model(ckpt_path: str, cfg: dict, device: torch.device) -> AdverSentinelLLM:
    model = AdverSentinelLLM(
        pretrained=cfg['model']['encoder_backbone'],
        proj_dim=cfg['model']['projection_dim'],
        num_prototypes=cfg['model']['dtmb_prototypes'],
        num_categories=cfg['model']['dtmb_categories'],
        dtmb_momentum=cfg['model']['ema_momentum'],
        stale_threshold=cfg['model']['stale_threshold'],
        cos_prune_threshold=cfg['model']['cos_prune_threshold'],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',            type=str, default='configs/default.yaml')
    parser.add_argument('--full_checkpoint',   type=str, required=True)
    parser.add_argument('--no_tpe_ckpt',       type=str, default=None)
    parser.add_argument('--no_dtmb_ckpt',      type=str, default=None)
    parser.add_argument('--no_dtmb_pruning_ckpt', type=str, default=None)
    parser.add_argument('--no_aagm_ckpt',      type=str, default=None)
    parser.add_argument('--no_con_ckpt',       type=str, default=None)
    parser.add_argument('--no_div_ckpt',       type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(cfg['model']['encoder_backbone'])
    data_dir  = cfg['data']['data_dir']
    max_len   = cfg['model']['max_sequence_length']

    # Evaluation dataset: PromptSafety-Bench test split (for ZDR)
    p, l = load_promptsafety_bench(data_dir, split='test')
    ds   = PromptDataset(p, l, tokenizer, max_len)
    loader = make_loader(ds, cfg['evaluation']['batch_size'], shuffle=False)

    checkpoints = {
        'full':            args.full_checkpoint,
        'no_tpe':          args.no_tpe_ckpt,
        'no_dtmb':         args.no_dtmb_ckpt,
        'no_dtmb_pruning': args.no_dtmb_pruning_ckpt,
        'no_aagm':         args.no_aagm_ckpt,
        'no_con':          args.no_con_ckpt,
        'no_div':          args.no_div_ckpt,
    }

    full_f1  = None
    full_zdr = None

    print(f"\n{'='*80}")
    print("  Table 6: Ablation Study")
    print(f"{'='*80}")
    header = f"{'Configuration':<38} {'F1':>6} {'ΔF1':>7} {'ZDR(%)':>8} {'ΔZDR':>7} {'Lat(ms)':>9}"
    print(header)
    print('-' * 80)

    for key, ckpt_path in checkpoints.items():
        if ckpt_path is None:
            print(f"  {ABLATION_LABELS.get(key, key):<36}: checkpoint not provided, skipping")
            continue
        if not os.path.exists(ckpt_path):
            print(f"  {ABLATION_LABELS.get(key, key):<36}: file not found ({ckpt_path})")
            continue

        model = load_model(ckpt_path, cfg, device)
        m     = evaluate(model, loader, device)

        if key == 'full':
            full_f1  = m['f1']
            full_zdr = m['zdr']

        delta_f1  = (m['f1']  - full_f1)  if full_f1  is not None else float('nan')
        delta_zdr = (m['zdr'] - full_zdr) * 100 if full_zdr is not None else float('nan')

        label = ABLATION_LABELS.get(key, key)
        print(f"  {label:<36} "
              f"{m['f1']:>6.3f} "
              f"{delta_f1:>+7.3f} "
              f"{m['zdr']*100:>7.1f}% "
              f"{delta_zdr:>+7.1f}% "
              f"{m['latency_ms']:>8.1f}")

    print('=' * 80)


if __name__ == '__main__':
    main()
