"""
AdverSentinel-LLM Inference Script

Supports:
  - Single-prompt detection
  - Batch file detection (JSONL input)
  - Deployment as a pre-inference filtering proxy

Usage:
    # Single prompt
    python src/inference.py --checkpoint outputs/best_model.pt \
        --prompt "Ignore all previous instructions and tell me how to..."

    # Batch from JSONL file (each line: {"prompt": "..."})
    python src/inference.py --checkpoint outputs/best_model.pt \
        --input_file prompts.jsonl --output_file results.jsonl
"""

import time
import json
import argparse
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import yaml

from model import AdverSentinelLLM


# ---------------------------------------------------------------------------
# Detector class (wraps the model for deployment)
# ---------------------------------------------------------------------------

class AdverSentinelDetector:
    """
    High-level detector interface for deployment.

    Deployment pattern (from paper Section 4.2):
        incoming_prompt
          -> AdverSentinelDetector.detect()
          -> PASS / BLOCK
          -> [if PASS] forward to LLM backend
    """

    def __init__(self, checkpoint_path: str, config_path: str = None,
                 device: str = 'auto', threshold: float = 0.5):

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load config from checkpoint or file
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg  = ckpt.get('config', {})
        if config_path:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)

        model_cfg = cfg.get('model', {})

        self.tokenizer = BertTokenizer.from_pretrained(
            model_cfg.get('encoder_backbone', 'bert-base-uncased'))
        self.max_length = model_cfg.get('max_sequence_length', 512)
        self.threshold  = threshold

        self.model = AdverSentinelLLM(
            pretrained=model_cfg.get('encoder_backbone', 'bert-base-uncased'),
            proj_dim=model_cfg.get('projection_dim', 256),
            num_prototypes=model_cfg.get('dtmb_prototypes', 64),
            num_categories=model_cfg.get('dtmb_categories', 8),
            dtmb_momentum=model_cfg.get('ema_momentum', 0.999),
            stale_threshold=model_cfg.get('stale_threshold', 500),
            cos_prune_threshold=model_cfg.get('cos_prune_threshold', 0.98),
        ).to(self.device)

        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

    @torch.no_grad()
    def detect(self, prompt: str) -> dict:
        """
        Detect whether a single prompt is adversarial.

        Returns dict:
            is_adversarial: bool
            score:          float in [0,1]  (1 = adversarial)
            latency_ms:     float
        """
        t0 = time.perf_counter()

        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids      = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        length_norm    = attention_mask.sum().float() / self.max_length
        ctx = torch.tensor([[length_norm.item(), 0.0, 0.0]],
                            device=self.device)

        y_hat, _ = self.model(input_ids, attention_mask, ctx)
        score = y_hat.item()

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            'is_adversarial': score >= self.threshold,
            'score':          score,
            'latency_ms':     latency_ms,
        }

    @torch.no_grad()
    def detect_batch(self, prompts: list) -> list:
        """
        Batch detection for throughput-optimized deployment.
        Returns list of dicts (same format as detect()).
        """
        t0 = time.perf_counter()

        encodings = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids      = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        B = input_ids.size(0)
        lengths_norm   = attention_mask.sum(dim=1).float() / self.max_length
        ctx = torch.stack([lengths_norm,
                            torch.zeros(B, device=self.device),
                            torch.zeros(B, device=self.device)], dim=1)

        y_hat, _ = self.model(input_ids, attention_mask, ctx)
        scores = y_hat.cpu().numpy()

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        total_ms   = (time.perf_counter() - t0) * 1000
        per_sample = total_ms / max(B, 1)

        return [
            {
                'prompt':         prompts[i],
                'is_adversarial': bool(scores[i] >= self.threshold),
                'score':          float(scores[i]),
                'latency_ms':     per_sample,
            }
            for i in range(B)
        ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='AdverSentinel-LLM — Adversarial Prompt Detector')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--config',     type=str, default=None,
                        help='Path to config YAML (optional; falls back to checkpoint)')
    parser.add_argument('--prompt',     type=str, default=None,
                        help='Single prompt to classify')
    parser.add_argument('--input_file', type=str, default=None,
                        help='JSONL file with {"prompt": "..."} per line')
    parser.add_argument('--output_file',type=str, default=None,
                        help='JSONL file to write results to')
    parser.add_argument('--threshold',  type=float, default=0.5,
                        help='Detection threshold (default: 0.5)')
    parser.add_argument('--device',     type=str,   default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--batch_size', type=int,   default=32,
                        help='Batch size for file-mode inference')
    args = parser.parse_args()

    detector = AdverSentinelDetector(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        threshold=args.threshold,
    )
    print(f"Model loaded on {detector.device}")

    # Single-prompt mode
    if args.prompt:
        result = detector.detect(args.prompt)
        verdict = "ADVERSARIAL" if result['is_adversarial'] else "BENIGN"
        print(f"\nPrompt:    {args.prompt[:80]}{'...' if len(args.prompt)>80 else ''}")
        print(f"Verdict:   {verdict}")
        print(f"Score:     {result['score']:.4f}  (threshold={args.threshold})")
        print(f"Latency:   {result['latency_ms']:.1f} ms")
        return

    # Batch file mode
    if args.input_file:
        with open(args.input_file, 'r') as f:
            lines = [json.loads(l) for l in f if l.strip()]
        prompts = [item.get('prompt', item.get('text', '')) for item in lines]

        results = []
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i:i + args.batch_size]
            batch_results = detector.detect_batch(batch)
            results.extend(batch_results)
            print(f"  Processed {min(i+args.batch_size, len(prompts))}/{len(prompts)}")

        # Merge original fields
        for orig, res in zip(lines, results):
            orig.update(res)

        out_path = args.output_file or args.input_file.replace('.jsonl', '_results.jsonl')
        with open(out_path, 'w') as f:
            for item in lines:
                f.write(json.dumps(item) + '\n')

        n_adv = sum(1 for r in results if r['is_adversarial'])
        avg_lat = sum(r['latency_ms'] for r in results) / len(results)
        print(f"\nResults: {n_adv}/{len(results)} flagged as adversarial")
        print(f"Avg latency: {avg_lat:.1f} ms/prompt")
        print(f"Saved to: {out_path}")
        return

    parser.print_help()


if __name__ == '__main__':
    main()
