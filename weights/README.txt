This directory holds the trained model artifact files produced by:

    python scripts/save_model_artifacts.py \
        --checkpoint outputs/best_model.pt \
        --config     configs/default.yaml \
        --out_dir    weights/

Expected files after running:
  adversentinel_weights.pt  — full model state dict (~221M parameters, ~840 MB)
  dtmb_prototypes.pt        — DTMB prototype vectors (shape: 64 × 256)
  aagm_parameters.pt        — AAGM gate + classifier head weights

These files are not included in the code repository due to file size.
Train the model first (see README.md), then run save_model_artifacts.py.
