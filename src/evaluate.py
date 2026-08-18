"""
Evaluation utilities for AdverSentinel-LLM.

Metrics: Accuracy, Precision, Recall, F1, AUC-ROC, ZDR, FPR, Latency.
Reproduces Table 3, Table 4, Table 6 (ablations), Table 11 (domain FPR).
"""

import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)
from typing import Optional


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device,
             threshold: float = 0.5) -> dict:
    """
    Run model on loader and compute all metrics.

    ZDR (Zero-Day Detection Rate) = recall on adversarial samples.
    FPR = false positive rate on benign samples.
    """
    model.eval()

    all_preds, all_probs, all_labels = [], [], []
    total_time = 0.0
    n_samples  = 0

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ctx            = batch.get('ctx', None)
        if ctx is not None:
            ctx = ctx.to(device)
        labels = batch['label'].cpu().numpy()

        t0 = time.perf_counter()
        y_hat, _ = model(input_ids, attention_mask, ctx)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        total_time += (time.perf_counter() - t0) * 1000  # ms

        probs = y_hat.cpu().numpy()
        preds = (probs >= threshold).astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        n_samples += len(labels)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')

    # ZDR = recall on adversarial (positive) class
    adv_mask = (all_labels == 1)
    zdr = all_preds[adv_mask].mean() if adv_mask.any() else float('nan')

    # FPR = false positive rate on benign (negative) class
    ben_mask = (all_labels == 0)
    fpr = all_preds[ben_mask].mean() if ben_mask.any() else float('nan')

    latency_per_sample = total_time / max(n_samples, 1)

    return {
        'accuracy':  acc,
        'precision': prec,
        'recall':    rec,
        'f1':        f1,
        'auc_roc':   auc,
        'zdr':       zdr,
        'fpr':       fpr,
        'latency_ms': latency_per_sample,
        'n_samples': n_samples,
    }


# ---------------------------------------------------------------------------
# Reproduce Table 3 — full comparison
# ---------------------------------------------------------------------------

def evaluate_all_datasets(model, dataset_loaders: dict,
                           device: torch.device) -> dict:
    """
    dataset_loaders: {'advbench': loader, 'jailbreakbench': loader, ...}
    Returns per-dataset and aggregate metrics.
    """
    results = {}
    for name, loader in dataset_loaders.items():
        print(f"  Evaluating on {name} ...")
        results[name] = evaluate(model, loader, device)
    return results


# ---------------------------------------------------------------------------
# Print formatted results table (Table 3 / Table 4 style)
# ---------------------------------------------------------------------------

def print_results_table(results: dict, label: str = "Results") -> None:
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    header = f"{'Dataset':<22} {'Acc':>6} {'P':>6} {'R':>6} {'F1':>6} "
    header += f"{'AUC':>6} {'ZDR':>6} {'FPR':>6} {'Lat(ms)':>9}"
    print(header)
    print('-' * 80)
    for name, m in results.items():
        row = (
            f"{name:<22} "
            f"{m['accuracy']*100:>5.1f}% "
            f"{m['precision']*100:>5.1f}% "
            f"{m['recall']*100:>5.1f}% "
            f"{m['f1']:>6.3f} "
            f"{m['auc_roc']:>6.3f} "
            f"{m['zdr']*100:>5.1f}% "
            f"{m['fpr']*100:>5.1f}% "
            f"{m['latency_ms']:>8.1f}"
        )
        print(row)
    print('=' * 80)


# ---------------------------------------------------------------------------
# Ablation study evaluation (Table 6)
# ---------------------------------------------------------------------------

def run_ablation(base_model_cls, ablation_configs: list,
                 val_loader: DataLoader, device: torch.device,
                 pretrained: str = "bert-base-uncased") -> list:
    """
    ablation_configs: list of dicts with keys 'name', 'kwargs'
      matching AdverSentinelLLM constructor arguments.
    Returns list of result dicts.
    """
    from model import AdverSentinelLLM
    ablation_results = []

    for cfg in ablation_configs:
        print(f"  Ablation: {cfg['name']} ...")
        model = base_model_cls(**cfg.get('kwargs', {})).to(device)
        # Note: in practice, load the checkpoint weights here
        metrics = evaluate(model, val_loader, device)
        metrics['name'] = cfg['name']
        ablation_results.append(metrics)

    return ablation_results


# ---------------------------------------------------------------------------
# Domain-specific FPR (Table 11)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_domain_fpr(model, domain_loaders: dict,
                         device: torch.device) -> dict:
    """
    Compute FPR on benign-only domain-specific prompts.
    domain_loaders: {'Technical/Code': loader, 'Legal': loader, ...}
    All samples must be benign (label=0).
    """
    model.eval()
    results = {}
    for domain, loader in domain_loaders.items():
        all_preds = []
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ctx = batch.get('ctx', None)
            if ctx is not None:
                ctx = ctx.to(device)
            y_hat, _ = model(input_ids, attention_mask, ctx)
            preds = (y_hat.cpu() >= 0.5).int().numpy()
            all_preds.extend(preds.tolist())
        fpr = np.mean(all_preds)  # All samples are benign -> preds are FP
        results[domain] = {'fpr': fpr, 'n': len(all_preds)}
        print(f"  {domain:<25}: FPR = {fpr*100:.1f}% ({len(all_preds)} samples)")
    return results


# ---------------------------------------------------------------------------
# Command-line evaluation entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import yaml
    from model   import AdverSentinelLLM
    from dataset import (PromptDataset, make_loader, load_advbench,
                          load_jailbreakbench, load_toxicchat,
                          load_promptsafety_bench)
    from transformers import BertTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config',     type=str, default='configs/default.yaml')
    parser.add_argument('--dataset',    type=str, default='all',
                        choices=['advbench', 'jailbreakbench',
                                 'toxicchat', 'promptsafety', 'all'])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(cfg['model']['encoder_backbone'])

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
    print(f"Loaded checkpoint from {args.checkpoint} "
          f"(Val F1 = {ckpt.get('val_f1', 'N/A')})")

    loaders = {}
    data_dir = cfg['data']['data_dir']
    max_len  = cfg['model']['max_sequence_length']

    dataset_map = {
        'advbench':      (load_advbench, {}),
        'jailbreakbench':(load_jailbreakbench, {}),
        'toxicchat':     (load_toxicchat, {}),
        'promptsafety':  (load_promptsafety_bench, {'split': 'test'}),
    }

    targets = list(dataset_map.keys()) if args.dataset == 'all' else [args.dataset]

    for name in targets:
        loader_fn, kwargs = dataset_map[name]
        try:
            p, l = loader_fn(data_dir, **kwargs)
            ds   = PromptDataset(p, l, tokenizer, max_len)
            loaders[name] = make_loader(
                ds, cfg['training']['batch_size'], shuffle=False)
        except FileNotFoundError as e:
            print(f"Skipping {name}: {e}")

    if loaders:
        results = evaluate_all_datasets(model, loaders, device)
        print_results_table(results, label="AdverSentinel-LLM Evaluation Results")
