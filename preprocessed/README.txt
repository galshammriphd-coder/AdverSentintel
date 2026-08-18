This directory holds preprocessed (tokenized) dataset artifacts produced by:

    python scripts/preprocess_datasets.py \
        --data_dir data/ \
        --out_dir  preprocessed/ \
        --config   configs/default.yaml

Expected files after running:
  advbench_train.pt         — AdvBench training split (tokenized)
  advbench_val.pt           — AdvBench validation split (tokenized)
  jailbreakbench_train.pt   — JailbreakBench training split
  jailbreakbench_val.pt     — JailbreakBench validation split
  toxicchat_train.pt        — ToxicChat training split
  toxicchat_val.pt          — ToxicChat validation split
  promptsafety_train.pt     — PromptSafety-Bench training split
  promptsafety_test.pt      — PromptSafety-Bench test split (used for ZDR)
  manifest.json             — SHA-256 checksums and split sizes

Download datasets first using scripts/download_datasets.py.
