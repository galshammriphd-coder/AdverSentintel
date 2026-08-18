"""
Save and load trained model artifacts for AdverSentinel-LLM.

Produces the three artifact files required by the reproducibility package:
  - weights/adversentinel_weights.pt     — full model state dict
  - weights/dtmb_prototypes.pt           — DTMB prototype vectors (M=64, D=256)
  - weights/aagm_parameters.pt          — AAGM gate + classifier head parameters

Usage:
    # Save artifacts from a trained checkpoint
    python scripts/save_model_artifacts.py \
        --checkpoint outputs/best_model.pt \
        --config     configs/default.yaml \
        --out_dir    weights/

    # Verify artifacts load correctly
    python scripts/save_model_artifacts.py \
        --verify \
        --out_dir weights/ \
        --config  configs/default.yaml
"""

import os
import sys
import argparse
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import AdverSentinelLLM


def save_artifacts(checkpoint_path: str, config_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    mc = cfg['model']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load full checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['model']

    # --- 1. Full model weights ---
    full_path = os.path.join(out_dir, 'adversentinel_weights.pt')
    torch.save({
        'state_dict': state,
        'epoch':      ckpt.get('epoch', 'N/A'),
        'val_f1':     ckpt.get('val_f1', 'N/A'),
        'config':     cfg,
    }, full_path)
    size_mb = os.path.getsize(full_path) / 1e6
    print(f"[1/3] Full model weights saved: {full_path}  ({size_mb:.1f} MB)")

    # --- 2. DTMB prototype vectors ---
    dtmb_keys = {k: v for k, v in state.items() if 'dtmb' in k}
    proto_key = next((k for k in dtmb_keys if 'prototypes' in k), None)
    if proto_key:
        prototypes = state[proto_key]  # (M, D) = (64, 256)
        proto_path = os.path.join(out_dir, 'dtmb_prototypes.pt')
        torch.save({
            'prototypes':      prototypes,
            'num_prototypes':  mc['dtmb_prototypes'],
            'num_categories':  mc['dtmb_categories'],
            'projection_dim':  mc['projection_dim'],
            'description': (
                'DTMB prototype vectors from trained AdverSentinel-LLM. '
                f'Shape: ({mc["dtmb_prototypes"]}, {mc["projection_dim"]}). '
                'Each prototype is L2-normalized. '
                f'{mc["dtmb_categories"]} threat categories, '
                f'{mc["dtmb_prototypes"] // mc["dtmb_categories"]} prototypes per category.'
            ),
            'full_dtmb_state': dtmb_keys,
        }, proto_path)
        print(f"[2/3] DTMB prototypes saved:      {proto_path}  "
              f"(shape: {list(prototypes.shape)})")
    else:
        print("[2/3] WARNING: DTMB prototype key not found in state dict.")

    # --- 3. AAGM parameters ---
    aagm_keys = {k: v for k, v in state.items() if 'aagm' in k}
    if aagm_keys:
        aagm_path = os.path.join(out_dir, 'aagm_parameters.pt')
        torch.save({
            'aagm_state_dict': aagm_keys,
            'description': (
                'Anomaly-Aware Gating Mechanism (AAGM) parameters. '
                'Contains: gate MLP weights (gate.0.weight, gate.0.bias, '
                'gate.2.weight, gate.2.bias) and classifier head '
                '(classifier_head.weight, classifier_head.bias). '
                'Input dim: proj_dim + 1 + ctx_dim = 256 + 1 + 3 = 260.'
            ),
        }, aagm_path)
        n_params = sum(v.numel() for v in aagm_keys.values())
        print(f"[3/3] AAGM parameters saved:       {aagm_path}  "
              f"({n_params:,} parameters)")
    else:
        print("[3/3] WARNING: AAGM keys not found in state dict.")

    print(f"\nAll artifacts saved to: {out_dir}/")
    print("  adversentinel_weights.pt  — full model (~221M params)")
    print("  dtmb_prototypes.pt        — 64 prototype vectors (64×256)")
    print("  aagm_parameters.pt        — gating mechanism weights")


def verify_artifacts(out_dir: str, config_path: str) -> None:
    """Load all three artifacts and confirm shapes are correct."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    mc = cfg['model']
    device = torch.device('cpu')

    print("Verifying artifacts...")

    # Full weights
    full_path = os.path.join(out_dir, 'adversentinel_weights.pt')
    ckpt = torch.load(full_path, map_location=device)
    model = AdverSentinelLLM(
        pretrained=mc['encoder_backbone'],
        proj_dim=mc['projection_dim'],
        num_prototypes=mc['dtmb_prototypes'],
        num_categories=mc['dtmb_categories'],
        dtmb_momentum=mc['ema_momentum'],
        stale_threshold=mc['stale_threshold'],
        cos_prune_threshold=mc['cos_prune_threshold'],
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"  [OK] Full weights loaded (epoch={ckpt.get('epoch')}, "
          f"val_f1={ckpt.get('val_f1')})")

    # DTMB prototypes
    proto_path = os.path.join(out_dir, 'dtmb_prototypes.pt')
    proto_data = torch.load(proto_path, map_location=device)
    protos = proto_data['prototypes']
    assert protos.shape == (mc['dtmb_prototypes'], mc['projection_dim']), \
        f"Expected ({mc['dtmb_prototypes']}, {mc['projection_dim']}), got {protos.shape}"
    print(f"  [OK] DTMB prototypes: shape={list(protos.shape)}")

    # AAGM parameters
    aagm_path = os.path.join(out_dir, 'aagm_parameters.pt')
    aagm_data = torch.load(aagm_path, map_location=device)
    n = sum(v.numel() for v in aagm_data['aagm_state_dict'].values())
    print(f"  [OK] AAGM parameters: {n:,} total parameters")

    # Quick forward pass
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(mc['encoder_backbone'])
    enc = tokenizer("Test prompt for verification.",
                    return_tensors='pt', padding='max_length',
                    max_length=mc['max_sequence_length'], truncation=True)
    with torch.no_grad():
        y_hat, v = model(enc['input_ids'], enc['attention_mask'])
    print(f"  [OK] Forward pass: score={y_hat.item():.4f}, "
          f"embedding shape={list(v.shape)}")
    print("\nAll artifacts verified successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--config',     type=str, default='configs/default.yaml')
    parser.add_argument('--out_dir',    type=str, default='weights/')
    parser.add_argument('--verify',     action='store_true')
    args = parser.parse_args()

    if args.verify:
        verify_artifacts(args.out_dir, args.config)
    elif args.checkpoint:
        save_artifacts(args.checkpoint, args.config, args.out_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
