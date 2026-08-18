"""
AdverSentinel-LLM Training Script

Implements Algorithm 1 from the paper:
  - Dual-encoder forward pass (PSE + TPE)
  - Projection head
  - DTMB threat proximity scoring
  - AAGM detection output
  - Combined loss (contrastive + BCE + diversity)
  - EMA prototype updates with pruning and rejuvenation

Usage:
    python src/train.py --config configs/default.yaml
"""

import os
import sys
import yaml
import math
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from model   import AdverSentinelLLM
from losses  import AdverSentinelLoss
from dataset import (PromptDataset, make_balanced_loader, make_loader,
                      load_advbench, load_jailbreakbench,
                      load_toxicchat, load_promptsafety_bench)
from evaluate import evaluate

logging.basicConfig(
    format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cfg(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_datasets(cfg: dict, tokenizer: BertTokenizer):
    """Load and combine training datasets per paper Section 4.1."""
    data_dir = cfg['data']['data_dir']
    max_len  = cfg['model']['max_sequence_length']

    all_prompts, all_labels = [], []

    for name in cfg['data']['datasets']:
        log.info(f"Loading {name} ...")
        if name == 'advbench':
            p, l = load_advbench(data_dir)
        elif name == 'jailbreakbench':
            p, l = load_jailbreakbench(data_dir)
        elif name == 'toxicchat':
            p, l = load_toxicchat(data_dir)
        elif name == 'promptsafety':
            p, l = load_promptsafety_bench(data_dir, split='train')
        else:
            raise ValueError(f"Unknown dataset: {name}")
        log.info(f"  {name}: {len(p)} samples "
                 f"({sum(l)} adversarial, {len(l)-sum(l)} benign)")
        all_prompts.extend(p)
        all_labels.extend(l)

    # 90/10 train/val split
    n = len(all_prompts)
    idx = list(range(n))
    import random; random.shuffle(idx)
    split = int(0.9 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = PromptDataset(
        [all_prompts[i] for i in train_idx],
        [all_labels[i]  for i in train_idx],
        tokenizer, max_len,
    )
    val_ds = PromptDataset(
        [all_prompts[i] for i in val_idx],
        [all_labels[i]  for i in val_idx],
        tokenizer, max_len,
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Training loop (Algorithm 1)
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, scaler,
                criterion, device, epoch, cfg):
    model.train()
    total_loss = 0.0
    metrics = {'contrastive': 0.0, 'bce': 0.0, 'diversity': 0.0}
    n_batches = 0

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)
        ctx            = batch['ctx'].to(device)

        optimizer.zero_grad()

        with autocast(enabled=cfg['training'].get('fp16', True)):
            y_hat, v = model(input_ids, attention_mask, ctx)
            loss_dict = criterion(
                y_hat, v, labels.float(),
                model.dtmb.prototypes,
            )
            loss = loss_dict['total']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(),
                                  cfg['training'].get('grad_clip', 1.0))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # DTMB prototype EMA update + pruning/rejuvenation (Algorithm 1)
        with torch.no_grad():
            v_detached = v.detach()
            model.dtmb.update_prototypes(v_detached, labels)

        total_loss += loss.item()
        for k in metrics:
            metrics[k] += loss_dict[k].item()
        n_batches += 1

    avg = total_loss / max(n_batches, 1)
    for k in metrics:
        metrics[k] /= max(n_batches, 1)
    return avg, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg.get('seed', 42))

    # Device
    if torch.cuda.is_available() and cfg['training'].get('use_gpu', True):
        n_gpus = torch.cuda.device_count()
        device = torch.device('cuda')
        log.info(f"Using {n_gpus} GPU(s)")
    else:
        device = torch.device('cpu')
        log.info("Using CPU")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        cfg['model']['encoder_backbone'])

    # Datasets & loaders
    train_ds, val_ds = build_datasets(cfg, tokenizer)
    log.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Use balanced loader for highly imbalanced sets
    adv_frac = sum(train_ds.labels) / len(train_ds.labels)
    if adv_frac < 0.20 or adv_frac > 0.80:
        log.info("Using class-balanced oversampling (imbalanced dataset detected)")
        train_loader = make_balanced_loader(
            train_ds, cfg['training']['batch_size'],
            cfg['training'].get('num_workers', 4))
    else:
        train_loader = make_loader(
            train_ds, cfg['training']['batch_size'],
            cfg['training'].get('num_workers', 4))

    val_loader = make_loader(val_ds, cfg['training']['batch_size'],
                              cfg['training'].get('num_workers', 4),
                              shuffle=False)

    # Model
    model_cfg = cfg['model']
    model = AdverSentinelLLM(
        pretrained=model_cfg['encoder_backbone'],
        proj_dim=model_cfg['projection_dim'],
        num_prototypes=model_cfg['dtmb_prototypes'],
        num_categories=model_cfg['dtmb_categories'],
        dtmb_momentum=model_cfg['ema_momentum'],
        stale_threshold=model_cfg['stale_threshold'],
        cos_prune_threshold=model_cfg['cos_prune_threshold'],
        bandwidth=model_cfg['dtmb_bandwidth'],
    )

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Loss
    train_cfg = cfg['training']
    criterion = AdverSentinelLoss(
        temperature=train_cfg['contrastive_temperature'],
        alpha=train_cfg['alpha'],
        mu=train_cfg['mu'],
        focal_gamma=train_cfg.get('focal_gamma', 2.0),
    )

    # Optimizer: AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
        betas=(0.9, 0.999),
    )

    # LR scheduler: linear warmup + cosine annealing
    total_steps  = len(train_loader) * train_cfg['epochs']
    warmup_steps = int(0.10 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    scaler = GradScaler(enabled=train_cfg.get('fp16', True))

    # Resume from checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        log.info(f"Resumed from epoch {start_epoch}")

    # Output directory
    out_dir = Path(cfg.get('output_dir', 'outputs'))
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0

    log.info("=" * 60)
    log.info("Starting AdverSentinel-LLM Training (Algorithm 1)")
    log.info(f"Epochs: {train_cfg['epochs']} | "
             f"Batch: {train_cfg['batch_size']} | "
             f"LR: {train_cfg['learning_rate']}")
    log.info("=" * 60)

    for epoch in range(start_epoch, train_cfg['epochs']):
        t0 = time.time()

        # --- Train ---
        avg_loss, loss_components = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, device, epoch, cfg,
        )

        # --- Validate ---
        val_metrics = evaluate(model, val_loader, device)

        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch+1:3d}/{train_cfg['epochs']} | "
            f"Loss: {avg_loss:.4f} "
            f"(con={loss_components['contrastive']:.4f}, "
            f"bce={loss_components['bce']:.4f}, "
            f"div={loss_components['diversity']:.4f}) | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val ZDR: {val_metrics['zdr']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            ckpt_path = out_dir / 'best_model.pt'
            core_model = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch':     epoch,
                'model':     core_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_f1':    best_val_f1,
                'config':    cfg,
            }, ckpt_path)
            log.info(f"  -> New best model saved (F1={best_val_f1:.4f})")

    log.info(f"Training complete. Best Val F1: {best_val_f1:.4f}")
    log.info(f"Checkpoint saved to: {out_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()
